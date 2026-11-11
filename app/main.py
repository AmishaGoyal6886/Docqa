import asyncio
import hashlib
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status

from app.config import get_settings
from app.models import DocumentInfo, HealthResponse, QueryRequest, QueryResponse, Source
from app.services.cache import TTLCache
from app.services.ollama import OllamaClient
from app.services.pdf import chunk_pages, extract_pages
from app.services.query_log import QueryLogger
from app.services.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.vector_store = VectorStore(settings.embedding_model)
    app.state.answer_cache = TTLCache[dict](settings.cache_ttl_seconds, settings.cache_max_entries)
    app.state.ollama = OllamaClient(
        settings.ollama_base_url, settings.ollama_model, settings.ollama_timeout_seconds
    )
    app.state.query_logger = QueryLogger(settings.mysql_url)
    try:
        await app.state.query_logger.open()
    except Exception:
        logger.exception("MySQL is unavailable; continuing without query logging")
    yield
    await app.state.ollama.close()
    await app.state.query_logger.close()


app = FastAPI(
    title=settings.app_name,
    description="Upload PDFs and ask grounded questions using local embeddings, FAISS, and Ollama.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health(request: Request) -> HealthResponse:
    query_logger: QueryLogger = request.app.state.query_logger
    store: VectorStore = request.app.state.vector_store
    return HealthResponse(
        status="ok", indexed_documents=store.document_count, mysql_logging=query_logger.pool is not None
    )


@app.post("/documents", response_model=DocumentInfo, status_code=status.HTTP_201_CREATED, tags=["documents"])
async def upload_document(request: Request, file: UploadFile = File(...)) -> DocumentInfo:
    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf") or file.content_type not in {
        "application/pdf",
        "application/octet-stream",
        None,
    }:
        raise HTTPException(status_code=415, detail="Upload a PDF file")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    content = await file.read(max_bytes + 1)
    await file.close()
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"PDF exceeds the {settings.max_upload_mb} MB limit")
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF")

    try:
        pages = await asyncio.to_thread(extract_pages, content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {exc}") from exc
    if not pages:
        raise HTTPException(status_code=422, detail="No extractable text found; scanned PDFs need OCR")

    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    try:
        document = await request.app.state.vector_store.add_document(filename, pages, chunks)
    except Exception as exc:
        logger.exception("Failed to index %s", filename)
        raise HTTPException(status_code=503, detail=f"Could not create embeddings: {exc}") from exc
    await request.app.state.answer_cache.clear()
    return DocumentInfo(
        document_id=document.document_id,
        filename=document.filename,
        pages=document.pages,
        chunks=len(document.chunks),
    )


@app.post("/query", response_model=QueryResponse, tags=["questions"])
async def query_documents(payload: QueryRequest, request: Request) -> QueryResponse:
    started = time.perf_counter()
    store: VectorStore = request.app.state.vector_store
    cache: TTLCache[dict] = request.app.state.answer_cache
    query_logger: QueryLogger = request.app.state.query_logger
    if payload.document_id and not await store.has_document(payload.document_id):
        raise HTTPException(status_code=404, detail="Document not found in this server session")
    if store.document_count == 0:
        raise HTTPException(status_code=400, detail="Upload a PDF before asking a question")

    normalized_question = " ".join(payload.question.lower().split())
    cache_material = f"{payload.document_id or '*'}|{payload.top_k or settings.top_k}|{normalized_question}"
    cache_key = hashlib.sha256(cache_material.encode()).hexdigest()
    cached = await cache.get(cache_key)
    if cached is not None:
        latency_ms = (time.perf_counter() - started) * 1000
        result = QueryResponse(**cached, latency_ms=latency_ms, cached=True)
        await query_logger.log(payload.question, payload.document_id, latency_ms, True)
        return result

    try:
        matches = await store.search(payload.question, payload.document_id, payload.top_k or settings.top_k)
        if not matches:
            raise HTTPException(status_code=404, detail="No matching document text was found")
        context = "\n\n".join(
            f"[Document: {chunk.filename}; page {chunk.page}]\n{chunk.text}" for chunk, _ in matches
        )
        answer = await request.app.state.ollama.answer(payload.question, context)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Question answering failed")
        raise HTTPException(
            status_code=503,
            detail="Question answering failed. Check that Ollama is running and the configured model is available.",
        ) from exc

    sources = [
        Source(
            document_id=chunk.document_id,
            filename=chunk.filename,
            page=chunk.page,
            chunk_id=chunk.chunk_id,
            score=score,
            text=chunk.text,
        )
        for chunk, score in matches
    ]
    latency_ms = (time.perf_counter() - started) * 1000
    cache_value = {"answer": answer, "sources": [source.model_dump() for source in sources]}
    await cache.set(cache_key, cache_value)
    await query_logger.log(payload.question, payload.document_id, latency_ms, False)
    return QueryResponse(**cache_value, latency_ms=latency_ms, cached=False)
