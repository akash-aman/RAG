"""
RAG Controller — orchestrates the full RAG pipeline end-to-end:
1. Query enhancement (HyDE / sub-query decomposition)
2. Hybrid retrieval + metadata filtering
3. Reranking
4. Prompt optimization
5. LLM generation
6. Self-reflection loop
"""

from typing import Any, Dict, Generator, List, Optional

from src.core.config import get_settings
from src.services.retrieval.hybrid_search import HybridSearcher
from src.services.retrieval.reranker import get_reranker
from src.services.retrieval.hyde import HyDEGenerator
from src.services.retrieval.sub_query import SubQueryDecomposer
from src.services.generator.llm_client import get_llm_client
from src.services.generator.prompt_optimizer import PromptOptimizer
from src.services.generator.self_rag import SelfRAG
from src.services.ingestion.embedder import get_embedder
from src.utils.logger.logger import get_logger

logger = get_logger(name="RAGController")


class RAGController:
    """End-to-end RAG pipeline orchestrator."""

    def __init__(self):
        self.settings = get_settings()
        self.searcher = HybridSearcher()
        self.optimizer = PromptOptimizer()
        self.embedder = get_embedder()
        self.llm = get_llm_client()

        # Lazy-loaded heavy components
        self._reranker = None
        self._hyde = None
        self._sub_query = None
        self._self_rag = None

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = get_reranker()
        return self._reranker

    @property
    def hyde(self):
        if self._hyde is None:
            self._hyde = HyDEGenerator()
        return self._hyde

    @property
    def sub_query_decomposer(self):
        if self._sub_query is None:
            self._sub_query = SubQueryDecomposer()
        return self._sub_query

    @property
    def self_rag(self):
        if self._self_rag is None:
            self._self_rag = SelfRAG()
        return self._self_rag

    def query(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        top_k: Optional[int] = None,
        enable_hyde: Optional[bool] = None,
        enable_reranking: Optional[bool] = None,
        enable_self_rag: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline and return the answer.

        Returns:
            dict with keys: answer, sources, metadata
        """
        settings = self.settings
        top_k = top_k or settings.retrieval_final_k
        use_hyde = enable_hyde if enable_hyde is not None else settings.enable_hyde
        use_reranking = enable_reranking if enable_reranking is not None else settings.enable_reranking
        use_self_rag = enable_self_rag if enable_self_rag is not None else settings.enable_self_rag

        metadata: Dict[str, Any] = {
            "hyde_used": False,
            "reranking_used": False,
            "self_rag_used": False,
            "retrieval_count": 0,
        }

        # ── 1. Query Embedding (optionally enhanced with HyDE) ───────
        query_embedding = None
        if use_hyde:
            try:
                query_embedding, hyp_text = self.hyde.generate_hypothetical_embedding(query)
                metadata["hyde_used"] = True
                metadata["hypothetical_doc"] = hyp_text[:200]
            except Exception as e:
                logger.warning(f"HyDE failed, falling back to direct embedding: {e}")
                query_embedding = self.embedder.embed_query(query)
        else:
            query_embedding = self.embedder.embed_query(query)

        # ── 2. Hybrid Search ─────────────────────────────────────────
        str_filters = {k: str(v) for k, v in (filters or {}).items()}
        results = self.searcher.search(
            query=query,
            query_embedding=query_embedding,
            top_k=settings.retrieval_top_k,
            final_k=settings.retrieval_top_k,  # get more for reranking
            filters=str_filters if str_filters else None,
            user_id=user_id,
            org_id=org_id,
        )
        metadata["retrieval_count"] = len(results)

        if not results:
            return {
                "answer": "I couldn't find any relevant information in the knowledge base to answer your question.",
                "sources": [],
                "metadata": metadata,
            }

        # ── 3. Reranking ─────────────────────────────────────────────
        if use_reranking and results:
            try:
                results = self.reranker.rerank(query, results, top_k=top_k)
                metadata["reranking_used"] = True
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")
                results = results[:top_k]
        else:
            results = results[:top_k]

        # ── 4. Prompt Optimization ───────────────────────────────────
        optimized = self.optimizer.optimize(results)

        # ── 5. LLM Generation ───────────────────────────────────────
        answer = self.llm.generate_with_context(query, optimized)

        # ── 6. Self-RAG Reflection ───────────────────────────────────
        if use_self_rag:
            try:
                evaluation = self.self_rag.evaluate(query, optimized, answer)
                metadata["self_rag_used"] = True
                metadata["self_rag_confidence"] = evaluation.get("confidence", 0)

                should_retry, refined_query = self.self_rag.should_retry(evaluation)
                if should_retry and refined_query:
                    logger.info(f"Self-RAG triggered retry with: {refined_query}")
                    metadata["self_rag_retried"] = True
                    metadata["refined_query"] = refined_query

                    # Re-run with refined query
                    retry_embedding = self.embedder.embed_query(refined_query)
                    retry_results = self.searcher.search(
                        query=refined_query,
                        query_embedding=retry_embedding,
                        top_k=settings.retrieval_top_k,
                        final_k=top_k,
                        filters=str_filters if str_filters else None,
                        user_id=user_id,
                        org_id=org_id,
                    )

                    if retry_results:
                        # Merge with original results (dedup)
                        seen = {r["text"][:100] for r in optimized}
                        for r in retry_results:
                            if r["text"][:100] not in seen:
                                optimized.append(r)
                                seen.add(r["text"][:100])

                        optimized = self.optimizer.optimize(optimized)
                        answer = self.llm.generate_with_context(query, optimized)
            except Exception as e:
                logger.warning(f"Self-RAG failed: {e}")

        # ── Build sources ────────────────────────────────────────────
        sources = [
            {
                "text": r.get("text", "")[:500],
                "source": r.get("source", ""),
                "score": r.get("rerank_score", r.get("rrf_score", r.get("score", 0))),
                "metadata": r.get("metadata", {}),
            }
            for r in optimized
        ]

        return {
            "answer": answer,
            "sources": sources,
            "metadata": metadata,
        }

    def query_stream(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        top_k: Optional[int] = None,
        enable_hyde: Optional[bool] = None,
        enable_reranking: Optional[bool] = None,
    ) -> Generator[str, None, None]:
        """
        Streaming version — yields answer tokens as they arrive.
        Note: Self-RAG is disabled in streaming mode.
        """
        settings = self.settings
        top_k = top_k or settings.retrieval_final_k
        use_hyde = enable_hyde if enable_hyde is not None else settings.enable_hyde
        use_reranking = enable_reranking if enable_reranking is not None else settings.enable_reranking

        # 1. Embedding
        if use_hyde:
            try:
                query_embedding, _ = self.hyde.generate_hypothetical_embedding(query)
            except Exception:
                query_embedding = self.embedder.embed_query(query)
        else:
            query_embedding = self.embedder.embed_query(query)

        # 2. Search
        str_filters = {k: str(v) for k, v in (filters or {}).items()}
        results = self.searcher.search(
            query=query,
            query_embedding=query_embedding,
            top_k=settings.retrieval_top_k,
            final_k=settings.retrieval_top_k,
            filters=str_filters if str_filters else None,
            user_id=user_id,
            org_id=org_id,
        )

        if not results:
            yield "I couldn't find any relevant information to answer your question."
            return

        # 3. Rerank
        if use_reranking:
            try:
                results = self.reranker.rerank(query, results, top_k=top_k)
            except Exception:
                results = results[:top_k]
        else:
            results = results[:top_k]

        # 4. Optimize
        optimized = self.optimizer.optimize(results)

        # 5. Stream
        yield from self.llm.generate_with_context_stream(query, optimized)


# ── Singleton ────────────────────────────────────────────────────────
_controller: Optional[RAGController] = None


def get_rag_controller() -> RAGController:
    global _controller
    if _controller is None:
        _controller = RAGController()
    return _controller
