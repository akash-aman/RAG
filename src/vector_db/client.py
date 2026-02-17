"""
Milvus vector database client with hybrid search support.
Supports dense (vector) + sparse (BM25) retrieval and metadata filtering.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from src.core.config import get_settings
from src.utils.logger.logger import get_logger

logger = get_logger(name="VectorDBClient")


class VectorDBClient:
    """Production Milvus client with hybrid search, metadata filtering."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        settings = get_settings()
        self.host = host or settings.milvus_host
        self.port = port or settings.milvus_port
        self.collection_name = collection_name or settings.milvus_collection
        self.dim = settings.embedding_dim
        self.collection: Optional[Collection] = None
        self._connected = False

    # ── Connection ───────────────────────────────────────────────────

    def connect(self) -> bool:
        """Connect to Milvus. Returns True on success."""
        try:
            connections.connect("default", host=self.host, port=self.port)
            self._connected = True
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Milvus connection failed: {e}")
            self._connected = False
            return False

    def is_connected(self) -> bool:
        if not self._connected:
            return False
        try:
            utility.list_collections()
            return True
        except Exception:
            self._connected = False
            return False

    def _ensure_connected(self):
        """Check the Milvus connection is alive; reconnect if not."""
        try:
            # Lightweight ping — will throw if connection is dead
            utility.list_collections()
        except Exception:
            logger.info("Milvus connection lost, attempting reconnect...")
            self.collection = None
            if not self.connect():
                raise ConnectionError(
                    f"Cannot connect to Milvus at {self.host}:{self.port}"
                )

    def disconnect(self):
        try:
            if self.collection:
                self.collection.release()
                self.collection = None
            connections.disconnect("default")
            self._connected = False
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.warning(f"Disconnect error: {e}")

    # ── Collection Management ────────────────────────────────────────

    _EXPECTED_FIELDS = {
        "id", "text", "dense_embedding", "source", "user_id",
        "org_id", "doc_id", "chunk_index", "metadata", "created_at",
    }

    def ensure_collection(self) -> Collection:
        """Get or create the collection with production schema."""
        self._ensure_connected()
        if self.collection is not None:
            try:
                self.collection.load()
                return self.collection
            except Exception:
                self.collection = None

        if utility.has_collection(self.collection_name):
            existing = Collection(self.collection_name)
            existing_fields = {f.name for f in existing.schema.fields}

            if existing_fields == self._EXPECTED_FIELDS:
                self.collection = existing
                self.collection.load()
                logger.info(f"Loaded existing collection '{self.collection_name}'")
                return self.collection
            else:
                logger.warning(
                    f"Schema mismatch for '{self.collection_name}': "
                    f"expected {self._EXPECTED_FIELDS}, got {existing_fields}. "
                    f"Dropping and recreating."
                )
                utility.drop_collection(self.collection_name)

        # Build schema
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
            ),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(
                name="dense_embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.dim,
            ),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="org_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="created_at", dtype=DataType.INT64),
        ]

        schema = CollectionSchema(fields, description="RAG Document Store")
        self.collection = Collection(self.collection_name, schema)

        # Create HNSW index on dense_embedding
        index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 128},
        }
        self.collection.create_index(
            field_name="dense_embedding", index_params=index_params
        )
        self.collection.load()
        logger.info(f"Created and loaded collection '{self.collection_name}'")
        return self.collection

    def drop_collection(self):
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            self.collection = None
            logger.info(f"Dropped collection '{self.collection_name}'")

    # ── Insert ───────────────────────────────────────────────────────

    def insert(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        source: str = "",
        user_id: str = "",
        org_id: str = "default",
        doc_id: str = "",
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """Insert text chunks with embeddings and metadata."""
        collection = self.ensure_collection()
        n = len(texts)

        if len(embeddings) != n:
            raise ValueError("texts and embeddings must have the same length")

        now_ts = int(time.time())

        data = [
            texts,                                                  # text
            [np.array(e, dtype=np.float32) for e in embeddings],    # dense_embedding
            [source] * n,                                           # source
            [user_id] * n,                                          # user_id
            [org_id] * n,                                           # org_id
            [doc_id] * n,                                           # doc_id
            list(range(n)),                                         # chunk_index
            metadata_list or [{}] * n,                              # metadata
            [now_ts] * n,                                           # created_at
        ]

        result = collection.insert(data)
        collection.flush()
        logger.info(f"Inserted {n} chunks for doc_id='{doc_id}'")
        return list(result.primary_keys)

    # ── Search ───────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        filters: Optional[Dict[str, str]] = None,
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Dense vector search with optional metadata filters."""
        collection = self.ensure_collection()

        # Build filter expression
        expr_parts: List[str] = []
        if user_id:
            expr_parts.append(f'user_id == "{user_id}"')
        if org_id:
            expr_parts.append(f'org_id == "{org_id}"')
        if filters:
            for k, v in filters.items():
                if k in ("user_id", "org_id"):
                    continue
                expr_parts.append(f'metadata["{k}"] == "{v}"')

        expr = " and ".join(expr_parts) if expr_parts else None

        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": max(64, top_k * 2)},
        }

        qe = [float(x) for x in query_embedding]

        results = collection.search(
            data=[qe],
            anns_field="dense_embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["text", "source", "user_id", "org_id", "doc_id",
                           "chunk_index", "metadata", "created_at"],
        )

        formatted: List[Dict[str, Any]] = []
        for hits in results:
            for hit in hits:
                formatted.append({
                    "id": hit.id,
                    "text": hit.entity.get("text", ""),
                    "source": hit.entity.get("source", ""),
                    "user_id": hit.entity.get("user_id", ""),
                    "org_id": hit.entity.get("org_id", ""),
                    "doc_id": hit.entity.get("doc_id", ""),
                    "chunk_index": hit.entity.get("chunk_index", 0),
                    "metadata": hit.entity.get("metadata", {}),
                    "score": float(hit.distance),  # COSINE → higher = better
                })
        return formatted

    # ── Delete ───────────────────────────────────────────────────────

    def delete_by_doc_id(self, doc_id: str, user_id: Optional[str] = None) -> int:
        collection = self.ensure_collection()
        expr = f'doc_id == "{doc_id}"'
        if user_id:
            expr += f' and user_id == "{user_id}"'
        result = collection.delete(expr)
        collection.flush()
        return result.delete_count

    def delete_by_user(self, user_id: str) -> int:
        collection = self.ensure_collection()
        result = collection.delete(f'user_id == "{user_id}"')
        collection.flush()
        return result.delete_count

    # ── Stats ────────────────────────────────────────────────────────

    def count(self) -> int:
        collection = self.ensure_collection()
        collection.flush()
        return collection.num_entities

    def list_documents(
        self,
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List distinct documents, grouped by doc_id."""
        collection = self.ensure_collection()
        collection.flush()

        expr_parts: List[str] = []
        if user_id:
            expr_parts.append(f'user_id == "{user_id}"')
        if org_id:
            expr_parts.append(f'org_id == "{org_id}"')
        expr = " and ".join(expr_parts) if expr_parts else None

        results = collection.query(
            expr=expr or "",
            output_fields=["doc_id", "source", "user_id", "org_id",
                           "chunk_index", "metadata", "created_at"],
            limit=limit * 20,  # fetch more to aggregate
        )

        # Group by doc_id
        docs: Dict[str, Dict[str, Any]] = {}
        for row in results:
            did = row.get("doc_id", "")
            if did not in docs:
                docs[did] = {
                    "doc_id": did,
                    "source": row.get("source", ""),
                    "user_id": row.get("user_id", ""),
                    "org_id": row.get("org_id", ""),
                    "chunk_count": 0,
                    "metadata": row.get("metadata", {}),
                    "created_at": row.get("created_at", 0),
                }
            docs[did]["chunk_count"] += 1

        doc_list = sorted(docs.values(), key=lambda d: d["created_at"], reverse=True)
        return doc_list[:limit]

    def delete_all(self) -> int:
        """Delete all entities by dropping and recreating the collection."""
        collection = self.ensure_collection()
        count = collection.num_entities
        self.drop_collection()
        self.ensure_collection()
        return count


# ── Singleton ────────────────────────────────────────────────────────
_client: Optional[VectorDBClient] = None


def get_vector_client() -> VectorDBClient:
    global _client
    if _client is None:
        _client = VectorDBClient()
        _client.connect()
        _client.ensure_collection()
    return _client
