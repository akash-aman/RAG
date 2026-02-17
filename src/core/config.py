"""
Application configuration using Pydantic BaseSettings.
All config is loaded from environment variables / .env file.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Central configuration for the RAG application."""

    # ── App ──────────────────────────────────────────────────────────────
    app_name: str = "RAG-LLM"
    app_version: str = "2.0.0"
    debug: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    # ── API ──────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8081)
    cors_origins: str = Field(default="*")

    # ── JWT / Auth ───────────────────────────────────────────────────────
    jwt_secret_key: str = Field(default="super-secret-change-me-in-prod")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expire_minutes: int = Field(default=1440)  # 24 hours

    # ── Milvus ───────────────────────────────────────────────────────────
    milvus_host: str = Field(default="localhost")
    milvus_port: str = Field(default="19530")
    milvus_collection: str = Field(default="rag_documents")

    # ── Redis ────────────────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0")

    # ── LLM ──────────────────────────────────────────────────────────────
    llm_provider: str = Field(default="lm_studio")  # lm_studio | openai | ollama
    llm_base_url: str = Field(default="http://127.0.0.1:1234")
    llm_api_key: str = Field(default="lm-studio")
    llm_model: str = Field(default="local-model")
    llm_temperature: float = Field(default=0.7)
    llm_max_tokens: int = Field(default=1024)

    # ── Embeddings ───────────────────────────────────────────────────────
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    embedding_dim: int = Field(default=384)

    # ── Reranker ─────────────────────────────────────────────────────────
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker_top_k: int = Field(default=5)

    # ── Ingestion ────────────────────────────────────────────────────────
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=128)
    chunk_strategy: str = Field(default="sentence_aware")

    # ── Rate Limiting ────────────────────────────────────────────────────
    rate_limit_default: str = Field(default="60/minute")
    rate_limit_query: str = Field(default="30/minute")
    rate_limit_ingest: str = Field(default="10/minute")

    # ── Retrieval ────────────────────────────────────────────────────────
    retrieval_top_k: int = Field(default=20)
    retrieval_final_k: int = Field(default=5)
    enable_hyde: bool = Field(default=True)
    enable_reranking: bool = Field(default=True)
    enable_self_rag: bool = Field(default=True)

    # ── Upload ───────────────────────────────────────────────────────────
    upload_dir: str = Field(default="/tmp/rag_uploads")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


# Singleton
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return cached settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
