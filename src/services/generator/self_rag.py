"""
Self-RAG (Self-Reflection) — feedback loop where the LLM evaluates
whether retrieved context is sufficient, and triggers re-retrieval if not.
"""

from typing import Any, Dict, List, Optional, Tuple

from src.services.generator.llm_client import get_llm_client
from src.utils.logger.logger import get_logger

logger = get_logger(name="SelfRAG")

REFLECTION_PROMPT = """You are a critical evaluator. Given a question, retrieved context, and a generated answer, evaluate the quality of the answer.

Question: {query}

Retrieved Context:
{context}

Generated Answer:
{answer}

Evaluate on these criteria:
1. Is the answer fully supported by the context? (Yes/Partially/No)
2. Does the answer address the question completely? (Yes/Partially/No)
3. Confidence score (0.0 to 1.0)

If the answer is not well-supported or incomplete, suggest a refined search query that could find the missing information.

Respond in EXACTLY this format:
SUPPORTED: <Yes|Partially|No>
COMPLETE: <Yes|Partially|No>
CONFIDENCE: <0.0-1.0>
REFINED_QUERY: <query or NONE>"""


class SelfRAG:
    """
    Self-reflection loop for RAG quality assurance.
    After generating an answer, evaluates if the context was sufficient.
    If not, suggests a refined query for follow-up retrieval.
    """

    def __init__(self, confidence_threshold: float = 0.6):
        self.llm = get_llm_client()
        self.confidence_threshold = confidence_threshold

    def evaluate(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        answer: str,
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of a RAG response.

        Returns dict with:
            - supported: Yes/Partially/No
            - complete: Yes/Partially/No
            - confidence: float (0-1)
            - needs_refinement: bool
            - refined_query: str or None
        """
        context_text = "\n\n".join(
            c.get("text", "") for c in context_chunks[:5]  # limit for prompt size
        )

        prompt = REFLECTION_PROMPT.format(
            query=query,
            context=context_text,
            answer=answer,
        )

        response = self.llm.generate(
            prompt=prompt,
            system_prompt="You are a precise evaluator. Follow the format exactly.",
            max_tokens=200,
            temperature=0.1,
        )

        return self._parse_evaluation(response)

    def _parse_evaluation(self, response: str) -> Dict[str, Any]:
        """Parse the structured evaluation response."""
        result = {
            "supported": "Unknown",
            "complete": "Unknown",
            "confidence": 0.5,
            "needs_refinement": False,
            "refined_query": None,
            "raw_evaluation": response,
        }

        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("SUPPORTED:"):
                result["supported"] = line.split(":", 1)[1].strip()
            elif line.startswith("COMPLETE:"):
                result["complete"] = line.split(":", 1)[1].strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    result["confidence"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    result["confidence"] = 0.5
            elif line.startswith("REFINED_QUERY:"):
                rq = line.split(":", 1)[1].strip()
                if rq.upper() != "NONE" and rq:
                    result["refined_query"] = rq

        result["needs_refinement"] = (
            result["confidence"] < self.confidence_threshold
            or result["supported"] == "No"
            or result["complete"] == "No"
        )

        logger.info(
            f"Self-RAG evaluation: confidence={result['confidence']:.2f}, "
            f"needs_refinement={result['needs_refinement']}"
        )

        return result

    def should_retry(self, evaluation: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Based on evaluation, decide if we should retry retrieval.
        Returns (should_retry, refined_query_or_none).
        """
        if evaluation["needs_refinement"] and evaluation.get("refined_query"):
            return True, evaluation["refined_query"]
        return False, None
