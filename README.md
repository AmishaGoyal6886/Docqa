# DocQA: RAG Question-Answering API

An async FastAPI service for asking questions about uploaded PDFs. It extracts and chunks PDF text, embeds chunks with Sentence-Transformers, retrieves passages with FAISS, and asks a locally served Ollama model to answer with page citations. Responses are cached in memory; MySQL query and latency logging is optional.

## Requirements

- Python 3.11 or newer
- Ollama installed and running locally
- At least one Ollama chat model pulled (default: `llama3.2`)
- MySQL 8.0+ only if you want persistent query logs

The first run downloads the configured sentence embedding model. Embedding and vector indexes live in memory, so uploaded documents must be re-uploaded after a server restart.

## Run locally

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
Copy-Item .env.example .env
ollama pull llama3.2
uvicorn app.main:app --reload
```

Open the interactive API docs at <http://localhost:8000/docs>. Check service status at <http://localhost:8000/health>.

## API

### Upload a PDF

```powershell
curl.exe -F "file=@./report.pdf" http://localhost:8000/documents
```

The response includes a `document_id` for selecting that PDF later. Uploads are limited to 30 MB by default (`MAX_UPLOAD_MB`). Only text PDFs are supported; scanned PDFs require OCR before upload.

### Ask a question

```powershell
curl.exe -X POST http://localhost:8000/query `
  -H "Content-Type: application/json" `
  -d '{"question":"What are the main findings?","document_id":"YOUR_DOCUMENT_ID"}'
```

Omit `document_id` to search across all PDFs currently indexed in this process. `top_k` can be set per request (1–20); otherwise `TOP_K` is used. The response includes the answer, source passages with page numbers and similarity scores, elapsed milliseconds, and whether the answer came from cache.

## Configuration

Copy `.env.example` to `.env` and adjust the settings. Set `OLLAMA_BASE_URL` if Ollama is not at `http://localhost:11434`. Set `EMBEDDING_MODEL` to another compatible Sentence-Transformers model if desired.

To enable MySQL logging, create a database and user, then set for example:

```dotenv
MYSQL_URL=mysql://docqa:your-password@localhost:3306/docqa
```

At startup the API creates a `query_logs` table if needed. If MySQL is unavailable, the API still serves requests and reports logging as disabled in `/health`.

## Project layout

```text
app/
  main.py                 FastAPI routes and application lifecycle
  config.py               Environment-based settings
  models.py               Request and response schemas
  services/
    cache.py              In-memory TTL/LRU cache
    ollama.py             Async Ollama client
    pdf.py                PDF text extraction and chunking
    query_log.py          Optional MySQL query logging
    vector_store.py       Sentence-Transformers embeddings and FAISS retrieval
```
