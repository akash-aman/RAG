"""
Collection schema definitions and index configurations for Milvus.
"""

from pymilvus import DataType

# ── Schema Constants ─────────────────────────────────────────────────

COLLECTION_FIELDS = {
    "id": {"dtype": DataType.INT64, "is_primary": True, "auto_id": True},
    "text": {"dtype": DataType.VARCHAR, "max_length": 65535},
    "dense_embedding": {"dtype": DataType.FLOAT_VECTOR, "dim": 384},
    "source": {"dtype": DataType.VARCHAR, "max_length": 1024},
    "user_id": {"dtype": DataType.VARCHAR, "max_length": 256},
    "org_id": {"dtype": DataType.VARCHAR, "max_length": 256},
    "doc_id": {"dtype": DataType.VARCHAR, "max_length": 256},
    "chunk_index": {"dtype": DataType.INT64},
    "metadata": {"dtype": DataType.JSON},
    "created_at": {"dtype": DataType.INT64},
}

# ── Index Configurations ─────────────────────────────────────────────

HNSW_INDEX = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 128},
}

IVF_FLAT_INDEX = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 256},
}

# ── Search Parameters ────────────────────────────────────────────────

HNSW_SEARCH_PARAMS = {
    "metric_type": "COSINE",
    "params": {"ef": 128},
}

IVF_SEARCH_PARAMS = {
    "metric_type": "COSINE",
    "params": {"nprobe": 16},
}
