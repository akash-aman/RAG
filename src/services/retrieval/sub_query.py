"""
Sub-query decomposition — breaks complex questions into simpler retrieval tasks.
"""

from typing import Any, Dict, List

from src.services.generator.llm_client import get_llm_client
from src.utils.logger.logger import get_logger

logger = get_logger(name="SubQuery")

DECOMPOSE_PROMPT = """You are an expert query decomposer. Break the following complex question into 2-4 simpler, self-contained sub-questions that can each be answered independently.

Return ONLY the sub-questions, one per line, numbered like:
1. ...
2. ...

If the question is already simple enough, just return it as-is with number 1.

Complex question: {query}

Sub-questions:"""


class SubQueryDecomposer:
    """Breaks complex queries into simpler sub-queries for parallel retrieval."""

    def __init__(self):
        self.llm = get_llm_client()

    def decompose(self, query: str) -> List[str]:
        """
        Decompose a complex query into simpler sub-queries.
        Returns a list of sub-query strings.
        """
        prompt = DECOMPOSE_PROMPT.format(query=query)

        response = self.llm.generate(
            prompt=prompt,
            system_prompt="You are a helpful query decomposition assistant.",
            max_tokens=256,
            temperature=0.3,
        )

        # Parse numbered list
        sub_queries = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove numbering (1. 2. etc.)
            cleaned = line.lstrip("0123456789.)-) ").strip()
            if cleaned:
                sub_queries.append(cleaned)

        if not sub_queries:
            sub_queries = [query]

        logger.info(
            f"Decomposed query into {len(sub_queries)} sub-queries"
        )
        return sub_queries

    def merge_results(
        self, all_results: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Merge and deduplicate results from multiple sub-queries.
        Uses text content for deduplication.
        """
        seen_texts = set()
        merged = []

        for results in all_results:
            for result in results:
                text_key = result.get("text", "")[:200]  # first 200 chars as key
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    merged.append(result)

        # Sort by best available score
        score_key = "rerank_score" if merged and "rerank_score" in merged[0] else "score"
        merged.sort(key=lambda x: x.get(score_key, 0), reverse=True)

        return merged
