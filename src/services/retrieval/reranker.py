"""
Cross-encoder reranking service.
Scores (query, document) pairs using a cross-encoder and reorders results.
"""

from typing import Any, Dict, List, Optional

from sentence_transformers import CrossEncoder

from src.core.config import get_settings
from src.utils.logger.logger import get_logger

logger = get_logger(name="Reranker")


class Reranker:
    """Reranks retrieval results using a cross-encoder model."""

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model
        logger.info(f"Loading reranker model: {self.model_name}")
        self.model = CrossEncoder(self.model_name)
        logger.info("Reranker model loaded")

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Re-score and re-order results using the cross-encoder.

        Args:
            query: the original user query
            results: list of dicts with at least a 'text' key
            top_k: how many to return after reranking (None = all)

        Returns:
            Reranked list of result dicts with 'rerank_score' added.
        """
        if not results:
            return []

        settings = get_settings()
        top_k = top_k or settings.reranker_top_k

        # Build (query, passage) pairs
        pairs = [(query, r["text"]) for r in results]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Attach scores
        for result, score in zip(results, scores):
            result["rerank_score"] = float(score)

        # Sort descending by rerank score
        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

        logger.info(
            f"Reranked {len(results)} results → top-{top_k}, "
            f"best score: {reranked[0]['rerank_score']:.4f}"
        )

        return reranked[:top_k]


# ── Singleton ────────────────────────────────────────────────────────
_reranker: Optional[Reranker] = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
