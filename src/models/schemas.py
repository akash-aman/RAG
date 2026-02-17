"""
Pydantic request / response schemas for the RAG API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Auth ─────────────────────────────────────────────────────────────────

class AuthRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    username: str
    role: str
    org_id: str


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    org_id: str = Field(default="default")


# ── Ingestion ────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    task_id: str
    status: str = "accepted"
    message: str = ""


class IngestStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None


class DocumentInfo(BaseModel):
    doc_id: str
    source: str = ""
    user_id: str = ""
    org_id: str = ""
    chunk_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: int = 0


class DeleteResponse(BaseModel):
    deleted: int = 0
    message: str = ""


# ── Query ────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10_000)
    filters: Dict[str, Any] = Field(default_factory=dict)
    stream: bool = Field(default=False)
    enable_hyde: Optional[bool] = None
    enable_reranking: Optional[bool] = None
    enable_self_rag: Optional[bool] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=50)


class Source(BaseModel):
    text: str
    source: str = ""
    score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ── Internal Models ──────────────────────────────────────────────────────

class DocumentChunk(BaseModel):
    id: str = ""
    text: str
    index: int = 0
    token_count: int = 0
    source: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedDocument(BaseModel):
    id: int = 0
    doc_id: str = ""
    text: str
    source: str = ""
    score: float = 0.0
    distance: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ── Health ───────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = ""
    milvus_connected: bool = False
