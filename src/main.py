"""
FastAPI application entry point.
Includes CORS, rate limiting, routers, and lifecycle events.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.core.config import get_settings
from src.models.schemas import HealthResponse
from src.utils.logger.logger import get_logger

logger = get_logger(name="App")

# ── OpenAPI Tags ─────────────────────────────────────────────────────

TAGS_METADATA = [
    {
        "name": "Health",
        "description": "Health check and status endpoints.",
    },
    {
        "name": "Authentication",
        "description": "JWT login and user info. Use the token from `/login` as a Bearer token.",
    },
    {
        "name": "Ingestion",
        "description": "Upload documents (PDF, TXT, MD) for indexing into the RAG knowledge base.",
    },
    {
        "name": "Query",
        "description": "Ask questions against the knowledge base. Supports streaming (SSE) and feature toggles.",
    },
]


# ── Rate Limiter ─────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)


# ── Lifespan ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    settings = get_settings()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # Connect to Milvus on startup
    try:
        from src.vector_db.client import get_vector_client
        client = get_vector_client()
        logger.info("Milvus connection established")
    except Exception as e:
        logger.warning(f"Milvus not available at startup: {e}")

    yield

    # Shutdown
    try:
        from src.vector_db.client import get_vector_client
        get_vector_client().disconnect()
    except Exception:
        pass
    logger.info("Application shutdown complete")


# ── App Factory ──────────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
## Production-Grade RAG System

A full-featured Retrieval-Augmented Generation API with:

- 🔍 **Hybrid Search** — Dense vectors + BM25 keyword matching (RRF fusion)
- 🎯 **Cross-Encoder Reranking** — ms-marco-MiniLM-L-6-v2
- 💡 **HyDE** — Hypothetical Document Embeddings for query enhancement
- 🧩 **Sub-Query Decomposition** — Breaks complex questions into simple retrieval tasks
- 📝 **Prompt Optimization** — Lost-in-the-middle reordering + compression
- 🔄 **Self-RAG** — Self-reflection loop with automatic re-retrieval
- 🔐 **JWT Authentication** — Role-based access control
- ⚡ **SSE Streaming** — Real-time token streaming

### Quick Start
1. Login via `/api/v1/auth/login` to get a JWT token
2. Click **Authorize** 🔒 above and enter: `Bearer <your_token>`
3. Upload a document via `/api/v1/ingest`
4. Query the knowledge base via `/api/v1/query`
        """,
        openapi_tags=TAGS_METADATA,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
        contact={
            "name": "RAG-LLM",
            "url": "https://github.com/akash-aman/RAG-LLM",
        },
        license_info={
            "name": "MIT",
        },
    )

    # ── Middleware ────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins.split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── Routers ──────────────────────────────────────────────────
    from src.api.v1.auth import router as auth_router
    from src.api.v1.ingest import router as ingest_router
    from src.api.v1.query import router as query_router
    from src.api.v1.query_stream import router as query_stream_router

    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(ingest_router, prefix="/api/v1")
    app.include_router(query_router, prefix="/api/v1")
    app.include_router(query_stream_router, prefix="/api/v1")

    # ── Health Check ─────────────────────────────────────────────
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        milvus_ok = False
        try:
            from src.vector_db.client import get_vector_client
            milvus_ok = get_vector_client().is_connected()
        except Exception:
            pass

        return HealthResponse(
            status="ok",
            version=settings.app_version,
            milvus_connected=milvus_ok,
        )

    @app.get("/", tags=["Health"])
    async def root():
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
        }

    return app


# The application instance used by uvicorn
app = create_app()
