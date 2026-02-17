"""
Document chunking service.
Wraps the existing TextTransformer with file-loading capabilities.
"""

import re
from typing import Any, Dict, List, Optional

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import get_settings
from src.utils.logger.logger import get_logger

logger = get_logger(name="Chunker")


class Chunker:
    """Loads documents and splits them into chunks."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        strategy: Optional[str] = None,
    ):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.strategy = strategy or settings.chunk_strategy

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ── File Loading ────────────────────────────────────────────────

    def load_file(self, file_path: str) -> str:
        """Load text from a file (PDF or plain text)."""
        if file_path.lower().endswith(".pdf"):
            return self._load_pdf(file_path)
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    def _load_pdf(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)

    # ── Text Cleaning ───────────────────────────────────────────────

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize whitespace, strip control chars."""
        # Replace multiple whitespace with single space (preserve newlines)
        text = re.sub(r"[^\S\n]+", " ", text)
        # Collapse 3+ newlines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip leading/trailing
        return text.strip()

    # ── Chunking ────────────────────────────────────────────────────

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Split cleaned text into chunks with metadata."""
        text = self.clean_text(text)
        raw_chunks = self._splitter.split_text(text)

        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunks.append({
                "id": f"chunk_{i}",
                "text": chunk_text,
                "index": i,
                "token_count": len(chunk_text.split()),
            })
        return chunks

    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load file, clean, and chunk in one call."""
        text = self.load_file(file_path)
        return self.chunk_text(text)
