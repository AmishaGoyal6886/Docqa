import asyncio
import logging
import uuid
from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Chunk:
    document_id: str
    filename: str
    page: int
    chunk_id: int
    text: str


@dataclass
class IndexedDocument:
    document_id: str
    filename: str
    pages: int
    index: faiss.IndexFlatIP
    chunks: list[Chunk]


class VectorStore:
    """In-memory FAISS indexes backed by normalized sentence embeddings."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._documents: dict[str, IndexedDocument] = {}
        self._lock = asyncio.Lock()

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors)
        return vectors

    async def add_document(
        self, filename: str, pages: list[tuple[int, str]], chunks: list[tuple[int, str]]
    ) -> IndexedDocument:
        document_id = str(uuid.uuid4())
        model = await asyncio.to_thread(self._get_model)
        texts = [text for _, text in chunks]
        vectors = await asyncio.to_thread(model.encode, texts, convert_to_numpy=True, show_progress_bar=False)
        vectors = await asyncio.to_thread(self._normalize, vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        records = [
            Chunk(document_id, filename, page, position, text)
            for position, (page, text) in enumerate(chunks)
        ]
        document = IndexedDocument(document_id, filename, len(pages), index, records)
        async with self._lock:
            self._documents[document_id] = document
        return document

    async def search(self, query: str, document_id: str | None, top_k: int) -> list[tuple[Chunk, float]]:
        model = await asyncio.to_thread(self._get_model)
        vector = await asyncio.to_thread(
            model.encode, [query], convert_to_numpy=True, show_progress_bar=False
        )
        vector = await asyncio.to_thread(self._normalize, vector)
        async with self._lock:
            documents = (
                [self._documents[document_id]]
                if document_id in self._documents
                else ([] if document_id else list(self._documents.values()))
            )
            results: list[tuple[Chunk, float]] = []
            for document in documents:
                scores, indices = document.index.search(vector, min(top_k, len(document.chunks)))
                results.extend(
                    (document.chunks[int(idx)], float(score))
                    for score, idx in zip(scores[0], indices[0], strict=True)
                    if idx >= 0
                )
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]

    async def has_document(self, document_id: str) -> bool:
        async with self._lock:
            return document_id in self._documents
