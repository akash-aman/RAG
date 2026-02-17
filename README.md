# 🚀 RAG-LLM: Production-Grade Retrieval-Augmented Generation

A full-featured, production-ready **Retrieval-Augmented Generation** system built with FastAPI, Milvus, and LM Studio. Implements advanced RAG techniques including **Hybrid Search**, **HyDE**, **Cross-Encoder Reranking**, **Sub-Query Decomposition**, **Prompt Optimization**, and **Self-RAG**.

---

## 📑 Table of Contents

- [Architecture Overview](#-architecture-overview)
- [RAG Pipeline — Step by Step](#-rag-pipeline--step-by-step)
- [Folder Structure](#-folder-structure)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [Configuration](#-configuration)
- [Understanding the Code — Jupyter Notebook](#-understanding-the-code--jupyter-notebook)

---

## 🏗 Architecture Overview

```
┌─────────────┐     ┌────────────────────────────────────────┐     ┌──────────┐
│   Client    │────▶│           FastAPI Application          │────▶│  Milvus  │
│ (Swagger UI)│     │                                        │     │ Vector DB│
└─────────────┘     │  ┌───────────────────────────────────┐ │     └──────────┘
                    │  │        RAG Controller             │ │
                    │  │                                   │ │     ┌──────────┐
                    │  │  Query ──▶ HyDE ──▶ Hybrid Search │ │────▶│  Redis   │
                    │  │    │                    │         │ │     │ (Celery) │
                    │  │    ▼                    ▼         │ │     └──────────┘
                    │  │  Rerank ──▶ Optimize ──▶ LLM Gen  │ │
                    │  │                          │        │ │     ┌──────────┐
                    │  │                     Self-RAG Loop │ │────▶│ LM Studio│
                    │  └───────────────────────────────────┘ │     │ (LLM)    │
                    └────────────────────────────────────────┘     └──────────┘
```

### Layer Architecture

| Layer | Directory | Responsibility |
|-------|-----------|----------------|
| **API** | `src/api/` | HTTP endpoints, request/response handling, JWT auth |
| **Controller** | `src/controllers/` | Business logic orchestration, RAG pipeline |
| **Services** | `src/services/` | Domain logic — ingestion, retrieval, generation |
| **Infrastructure** | `src/vector_db/` | Database clients, schema management |
| **Core** | `src/core/` | Config, security, shared utilities |
| **Models** | `src/models/` | Pydantic schemas, data models |

---

## 🔄 RAG Pipeline — Step by Step

When a user sends a query, the following happens inside the `RAGController`:

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│  Step 1: QUERY ENHANCEMENT (HyDE)           │
│                                             │
│  • LLM generates a "hypothetical answer"    │
│  • That answer is embedded into a vector    │
│  • This vector is used for retrieval        │
│  • Why? The hypothetical doc is closer in   │
│    embedding space to real answers than the │
│    original question                        │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Step 2: HYBRID SEARCH                      │
│                                             │
│  Two parallel search strategies:            │
│                                             │
│  Dense Search (Milvus):                     │
│    • COSINE similarity on HNSW index        │
│    • Captures semantic meaning              │
│                                             │
│  Sparse Search (BM25):                      │
│    • Keyword-based term matching            │
│    • Captures exact keyword relevance       │
│                                             │
│  Fusion:                                    │
│    • Reciprocal Rank Fusion (RRF)           │
│    • Merges both ranked lists               │
│    • RRF(d) = Σ 1/(k + rank(d))             │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Step 3: RERANKING                          │
│                                             │
│  • Cross-Encoder: ms-marco-MiniLM-L-6-v2    │
│  • Scores each (query, passage) pair        │
│  • Much more accurate than bi-encoder       │
│  • Re-orders by true relevance score        │
│  • Selects top-K most relevant chunks       │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Step 4: PROMPT OPTIMIZATION                │
│                                             │
│  Lost-in-the-Middle Reordering:             │
│    • LLMs pay more attention to start/end   │
│    • Best chunks → positions 1 and N        │
│    • Weaker chunks → middle positions       │
│                                             │
│  Prompt Compression:                        │
│    • Removes filler phrases                 │
│    • Normalizes whitespace                  │
│    • Trims to token budget                  │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Step 5: LLM GENERATION                     │
│                                             │
│  • Context + query → LLM (LM Studio)        │
│  • System prompt enforces grounding         │
│  • "Only answer from the provided context"  │
│  • Supports streaming (SSE)                 │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Step 6: SELF-RAG (Reflection)              │
│                                             │
│  • LLM evaluates its own answer             │
│  • Checks: Is it grounded? Complete?        │
│  • If confidence < threshold:               │
│    → Generates a refined query              │
│    → Re-retrieves with new query            │
│    → Merges new context + regenerates       │
│  • Maximum 1 retry to avoid loops           │
└─────────────────────────────────────────────┘
    │
    ▼
  Final Answer + Sources + Metadata
```

### Ingestion Pipeline

When a document is uploaded:

```
Upload File (PDF/TXT/MD)
    │
    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│  Load File   │───▶│   Chunking   │───▶│  Embedding   │───▶│  Milvus  │
│  (PyPDF2)    │    │  (512 tok,   │    │  (MiniLM-L6  │    │  Insert  │
│              │    │  128 overlap)│    │   384-dim)   │    │          │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────┘
                         │                    │
                    Text cleaning        L2-normalized
                    (whitespace,         dense vectors
                     special chars)      (cosine ready)
```

---

## 📂 Folder Structure

```
RAG-LLM/
│
├── src/                          # Main application source
│   ├── main.py                   # FastAPI app factory, CORS, rate limiting, lifespan
│   ├── rag_utils.py              # Legacy RAG utilities (preserved for reference)
│   │
│   ├── core/                     # Cross-cutting concerns
│   │   ├── config.py             # Pydantic Settings — all env vars centralized
│   │   └── security.py           # JWT auth, password hashing, RBAC, input sanitization
│   │
│   ├── models/                   # Data models
│   │   ├── schemas.py            # Pydantic request/response schemas for all endpoints
│   │   └── database.py           # In-memory user store (replace with DB in production)
│   │
│   ├── api/                      # API layer
│   │   ├── dependencies.py       # FastAPI dependency injection (user store, vector client)
│   │   └── v1/                   # Versioned API routes
│   │       ├── auth.py           # POST /register, POST /login, GET /me, GET /users
│   │       ├── ingest.py         # POST /ingest, GET /ingest, DELETE /ingest/{doc_id}
│   │       └── query.py          # POST /query (JSON + SSE streaming)
│   │
│   ├── controllers/              # Business logic orchestration
│   │   └── rag_controller.py     # Full RAG pipeline: HyDE → Search → Rerank → LLM → Self-RAG
│   │
│   ├── services/                 # Domain services (core RAG logic lives here)
│   │   ├── ingestion/            # Document processing pipeline
│   │   │   ├── chunker.py        # Text loading (PDF/TXT) + RecursiveCharacterTextSplitter
│   │   │   ├── embedder.py       # Sentence-transformer embeddings (all-MiniLM-L6-v2)
│   │   │   └── indexer.py        # Orchestrates: load → chunk → embed → store in Milvus
│   │   │
│   │   ├── retrieval/            # Search & retrieval strategies
│   │   │   ├── hybrid_search.py  # Dense (Milvus) + Sparse (BM25) with RRF fusion
│   │   │   ├── reranker.py       # Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
│   │   │   ├── hyde.py           # HyDE: hypothetical document embedding for better recall
│   │   │   └── sub_query.py      # Decomposes complex queries into 2-4 sub-queries
│   │   │
│   │   └── generator/            # LLM interaction & response generation
│   │       ├── llm_client.py     # Multi-provider LLM client (LM Studio/OpenAI/Ollama)
│   │       ├── prompt_optimizer.py # Lost-in-the-middle reordering + compression
│   │       └── self_rag.py       # Self-reflection loop with confidence evaluation
│   │
│   ├── vector_db/                # Vector database layer
│   │   ├── client.py             # Milvus client — CRUD, hybrid search, schema migration
│   │   └── schema.py             # Collection schema constants and index configs
│   │
│   └── utils/                    # Shared utilities
│       ├── logger/logger.py      # Centralized logging config
│       └── embedding/generic.py  # Advanced TextTransformer (multiple chunking strategies)
│
├── workers/                      # Async task processing
│   ├── celery_app.py             # Celery configuration (Redis broker)
│   └── ingestion_worker.py       # Background document ingestion task
│
├── tests/                        # Test suite
│   ├── test_api.py               # Integration tests (12 tests covering all endpoints)
│   └── sample.txt                # Sample document for testing
│
├── notebooks/                    # Jupyter notebooks
│   ├── rag_explained.ipynb       # ⬅️  Interactive walkthrough of the RAG pipeline
│   ├── embedding.ipynb           # Embedding experiments
│   └── vectordb.ipynb            # Vector database experiments
│
├── frontend/                     # Frontend app (SolidJS)
├── data/                         # Data directories
├── docker-compose.yml            # Full stack: Milvus + Redis + API + Worker
├── Dockerfile                    # Multi-stage Docker build
├── requirements.txt              # Python dependencies
└── .env                          # Environment variables
```

---

## ⚙️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **API Framework** | FastAPI | Async HTTP server with auto-generated OpenAPI docs |
| **Vector Database** | Milvus | Dense vector storage with HNSW/COSINE indexing |
| **Embedding Model** | all-MiniLM-L6-v2 | 384-dim sentence embeddings |
| **Reranker Model** | ms-marco-MiniLM-L-6-v2 | Cross-encoder for passage reranking |
| **LLM Provider** | LM Studio / OpenAI / Ollama | Text generation via OpenAI-compatible API |
| **Task Queue** | Celery + Redis | Async background document processing |
| **Auth** | JWT (python-jose) | Token-based authentication with RBAC |
| **Rate Limiting** | SlowAPI | Per-endpoint rate limits |
| **Sparse Search** | rank_bm25 | BM25 keyword matching for hybrid search |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (for Milvus & Redis)
- LM Studio running locally (or any OpenAI-compatible LLM server)

### 1. Start Infrastructure

```bash
# Start Milvus (vector DB) and Redis (task queue)
docker-compose up -d standalone redis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Edit `.env` with your settings:

```env
# LLM (point to your LM Studio or OpenAI endpoint)
LLM_BASE_URL=http://127.0.0.1:1234/v1
LLM_API_KEY=lm-studio
LLM_MODEL=local-model

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530

# JWT Secret (change in production!)
JWT_SECRET_KEY=super-secret-change-me-in-prod
```

### 4. Run the API

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8081
```

### 5. Open Swagger UI

Visit **http://localhost:8081/docs** for interactive API documentation.

**Quick workflow:**
1. `POST /api/v1/auth/register` — create a user
2. `POST /api/v1/auth/login` — get a JWT token
3. Click **Authorize** 🔒 → enter `Bearer <your_token>`
4. `POST /api/v1/ingest` — upload a document
5. `GET /api/v1/ingest` — see your ingested documents
6. `POST /api/v1/query` — ask questions about your documents

---

## 📡 API Reference

### Authentication

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/api/v1/auth/register` | None | Create a new user account |
| `POST` | `/api/v1/auth/login` | None | Login → returns JWT token |
| `GET` | `/api/v1/auth/me` | Bearer | Current user info |
| `GET` | `/api/v1/auth/users` | Admin | List all users |

### Document Ingestion

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/api/v1/ingest` | Bearer | Upload document (PDF/TXT/MD) |
| `GET` | `/api/v1/ingest` | Bearer | List your ingested documents |
| `DELETE` | `/api/v1/ingest/{doc_id}` | Bearer | Delete a specific document |
| `DELETE` | `/api/v1/ingest` | Bearer | Delete all your documents |

### RAG Query

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/api/v1/query` | Bearer | Ask a question (supports SSE streaming) |

**Query Request Body:**
```json
{
  "query": "What is retrieval-augmented generation?",
  "stream": false,
  "enable_hyde": true,
  "enable_reranking": true,
  "enable_self_rag": true,
  "top_k": 5,
  "filters": {}
}
```

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | App info |
| `GET` | `/health` | Health check + Milvus status |

---

## ⚙ Configuration

All settings are in `.env` and loaded via `src/core/config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | `http://127.0.0.1:1234/v1` | LLM endpoint |
| `LLM_MODEL` | `local-model` | Model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `EMBEDDING_DIM` | `384` | Embedding dimension |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `128` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `20` | Candidates from search |
| `RETRIEVAL_FINAL_K` | `5` | Final chunks after reranking |
| `ENABLE_HYDE` | `true` | Enable HyDE query enhancement |
| `ENABLE_RERANKING` | `true` | Enable cross-encoder reranking |
| `ENABLE_SELF_RAG` | `true` | Enable self-reflection loop |
| `RATE_LIMIT_QUERY` | `30/minute` | Query endpoint rate limit |
| `JWT_SECRET_KEY` | `...` | JWT signing key |

---

## 📓 Understanding the Code — Jupyter Notebook

For a deep-dive into how each component works, open the interactive notebook:

```bash
jupyter notebook notebooks/rag_explained.ipynb
```

The notebook walks through:
1. **Document Chunking** — how text is split into overlapping segments
2. **Embedding** — how chunks become 384-dim vectors
3. **Vector Search** — how Milvus finds similar documents
4. **BM25 Sparse Search** — keyword-based retrieval
5. **Hybrid Fusion (RRF)** — combining dense + sparse results
6. **Cross-Encoder Reranking** — improving result quality
7. **HyDE** — hypothetical document embeddings
8. **Prompt Optimization** — lost-in-the-middle reordering
9. **Self-RAG** — self-reflection and re-retrieval

---

## 🧪 Running Tests

```bash
# Run all 12 integration tests
python -m pytest tests/test_api.py -v
```

Tests cover: health checks, authentication, document ingestion, and RAG queries.

---

## 🐳 Docker Deployment

```bash
# Full stack (Milvus + Redis + API + Celery Worker)
docker-compose up -d

# API only (if infra already running)
docker-compose up -d rag-api
```

---

## 📄 License

MIT
