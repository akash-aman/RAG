"""
HyDE (Hypothetical Document Embeddings) — query enhancement.
Generates a hypothetical answer, embeds it, and uses that for retrieval.
"""

from typing import Any, Dict, List, Optional

from src.services.generator.llm_client import get_llm_client
from src.services.ingestion.embedder import get_embedder
from src.utils.logger.logger import get_logger

logger = get_logger(name="HyDE")

HYDE_PROMPT = """Please write a short, factual passage that would answer the following question.
Write as if you are writing a paragraph from a reference document.
Do not include any preamble like "Here is a passage". Just write the passage directly.

Question: {query}

Passage:"""


class HyDEGenerator:
    """
    Hypothetical Document Embeddings:
    1. Use the LLM to generate a hypothetical answer
    2. Embed the hypothetical answer
    3. Use that embedding for retrieval (instead of raw query embedding)
    """

    def __init__(self):
        self.llm = get_llm_client()
        self.embedder = get_embedder()

    def generate_hypothetical_embedding(
        self, query: str
    ) -> tuple[List[float], str]:
        """
        Generate a hypothetical document for the query, then embed it.
        Returns (embedding, hypothetical_text).
        """
        prompt = HYDE_PROMPT.format(query=query)

        hypothetical_text = self.llm.generate(
            prompt=prompt,
            system_prompt="You are an expert knowledge base writer.",
            max_tokens=256,
            temperature=0.7,
        )

        logger.info(
            f"HyDE generated hypothetical doc ({len(hypothetical_text)} chars)"
        )

        # Embed the hypothetical document
        embedding = self.embedder.embed_query(hypothetical_text)
        return embedding, hypothetical_text
