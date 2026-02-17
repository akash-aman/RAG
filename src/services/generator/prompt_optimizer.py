"""
Prompt optimization:
- Lost-in-the-Middle reordering
- Prompt compression (redundant token removal)
"""

import re
from typing import Any, Dict, List

from src.utils.logger.logger import get_logger

logger = get_logger(name="PromptOptimizer")


class PromptOptimizer:
    """Optimizes retrieved chunks before feeding to the LLM."""

    # ── Lost-in-the-Middle ───────────────────────────────────────────

    @staticmethod
    def reorder_lost_in_middle(
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Reorder chunks so the most relevant are at the BEGINNING and END.
        LLMs pay more attention to the start and end of their context window.

        Input should be sorted by relevance (best first).
        Output interleaves: best → worst → second-best pattern.
        """
        if len(chunks) <= 2:
            return chunks

        # Split into top half and bottom half
        mid = len(chunks) // 2
        top = chunks[:mid]      # most relevant
        bottom = chunks[mid:]   # less relevant

        # Place: top[0], top[2], ... at start; bottom in middle; top[1], top[3], ... at end
        start = top[::2]    # even indices (1st, 3rd, 5th best)
        end = top[1::2]     # odd indices (2nd, 4th, 6th best)

        reordered = start + bottom + list(reversed(end))
        logger.info(
            f"Reordered {len(chunks)} chunks (lost-in-middle optimization)"
        )
        return reordered

    # ── Prompt Compression ───────────────────────────────────────────

    @staticmethod
    def compress_chunk(text: str) -> str:
        """
        Remove redundant tokens and filler content from a chunk.
        Lightweight heuristic compression (no model required).
        """
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove common filler phrases
        fillers = [
            r"\bfor example\b",
            r"\bin other words\b",
            r"\bas mentioned (earlier|above|before|previously)\b",
            r"\bit is (important|worth) (to note|noting) that\b",
            r"\bas we (can|have) see(n)?\b",
            r"\bneedless to say\b",
            r"\bin conclusion\b",
            r"\bto summarize\b",
            r"\bas a matter of fact\b",
        ]
        for filler in fillers:
            text = re.sub(filler, "", text, flags=re.IGNORECASE)

        # Collapse double spaces after removal
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text

    def compress_chunks(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Compress all chunk texts for context-window efficiency."""
        compressed = []
        for chunk in chunks:
            c = chunk.copy()
            original_len = len(c["text"])
            c["text"] = self.compress_chunk(c["text"])
            c["compressed"] = True
            c["compression_ratio"] = (
                len(c["text"]) / original_len if original_len else 1.0
            )
            compressed.append(c)

        total_original = sum(len(c.get("text", "")) for c in chunks)
        total_compressed = sum(len(c["text"]) for c in compressed)
        if total_original > 0:
            logger.info(
                f"Compressed {len(chunks)} chunks: "
                f"{total_original} → {total_compressed} chars "
                f"({total_compressed/total_original:.1%})"
            )
        return compressed

    # ── Combined Optimization ────────────────────────────────────────

    def optimize(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply all optimizations: compress then reorder."""
        compressed = self.compress_chunks(chunks)
        reordered = self.reorder_lost_in_middle(compressed)
        return reordered
