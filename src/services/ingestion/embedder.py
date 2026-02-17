"""
Embedding service — generates dense embeddings using sentence-transformers
and sparse BM25 representations for hybrid search.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.config import get_settings
from src.utils.logger.logger import get_logger

logger = get_logger(name="Embedder")


class Embedder:
    """Generates dense vector embeddings for text chunks."""

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded (dim={self.dim})")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate dense embeddings for a batch of texts."""
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,  # unit-norm for cosine
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Generate a dense embedding for a single query string."""
        embedding = self.model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    def embed_chunks(
        self, chunks: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
        """
        Embed a list of chunk dicts (each must have a 'text' key).
        Returns the chunks and their embeddings as parallel lists.
        """
        texts = [c["text"] for c in chunks]
        embeddings = self.embed_texts(texts)
        return chunks, embeddings


# ── Singleton ────────────────────────────────────────────────────────
_embedder: Optional[Embedder] = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
