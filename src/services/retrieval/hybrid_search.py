"""
Hybrid search — combines dense vector search with BM25 sparse keyword search.
Uses Reciprocal Rank Fusion (RRF) to merge results.
"""

from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

from src.services.ingestion.embedder import get_embedder
from src.vector_db.client import get_vector_client
from src.core.config import get_settings
from src.utils.logger.logger import get_logger

logger = get_logger(name="HybridSearch")


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score."""
    return 1.0 / (k + rank)


class HybridSearcher:
    """
    Performs hybrid retrieval:
    1. Dense vector search via Milvus
    2. BM25 keyword search over the dense results (re-score)
    3. Reciprocal Rank Fusion to combine both rankings
    """

    def __init__(self):
        self.embedder = get_embedder()
        self.vector_client = get_vector_client()

    def search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 20,
        final_k: int = 10,
        filters: Optional[Dict[str, str]] = None,
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ) -> List[Dict[str, Any]]:
        """
        Execute hybrid search and return merged results.

        Args:
            query: raw user query text
            query_embedding: pre-computed query embedding (optional)
            top_k: how many candidates to pull from dense search
            final_k: how many results to return after fusion
            filters: metadata filters for Milvus
            user_id: restrict to user's documents
            org_id: restrict to org's documents
            dense_weight: weight for dense rank in RRF
            sparse_weight: weight for BM25 rank in RRF
        """
        # ── 1. Dense search ──────────────────────────────────────────
        if query_embedding is None:
            query_embedding = self.embedder.embed_query(query)

        dense_results = self.vector_client.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
            user_id=user_id,
            org_id=org_id,
        )

        if not dense_results:
            return []

        # ── 2. BM25 sparse re-scoring ───────────────────────────────
        texts = [r["text"] for r in dense_results]
        tokenized_texts = [t.lower().split() for t in texts]
        bm25 = BM25Okapi(tokenized_texts)
        bm25_scores = bm25.get_scores(query.lower().split())

        # Rank by BM25 scores (descending)
        bm25_ranked = sorted(
            enumerate(bm25_scores), key=lambda x: x[1], reverse=True
        )
        bm25_rank_map = {idx: rank for rank, (idx, _) in enumerate(bm25_ranked)}

        # ── 3. Reciprocal Rank Fusion ────────────────────────────────
        for dense_rank, result in enumerate(dense_results):
            idx = dense_rank
            sparse_rank = bm25_rank_map.get(idx, len(dense_results))

            rrf = (
                dense_weight * _rrf_score(dense_rank)
                + sparse_weight * _rrf_score(sparse_rank)
            )
            result["rrf_score"] = rrf
            result["dense_rank"] = dense_rank
            result["sparse_rank"] = sparse_rank

        # Sort by RRF score
        fused = sorted(dense_results, key=lambda x: x["rrf_score"], reverse=True)
        return fused[:final_k]
