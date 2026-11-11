from datetime import datetime

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    document_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    document_id: str
    filename: str
    page: int
    chunk_id: int
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: float
    cached: bool = False


class HealthResponse(BaseModel):
    status: str
    indexed_documents: int
    mysql_logging: bool


class QueryLog(BaseModel):
    question: str
    document_id: str | None
    latency_ms: float
    cached: bool
    created_at: datetime
