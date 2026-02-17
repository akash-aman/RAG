"""
Indexing service — orchestrates the full ingestion pipeline:
load → clean → chunk → embed → store in Milvus.
"""

import os
import uuid
from typing import Any, Dict, List, Optional

from src.core.config import get_settings
from src.services.ingestion.chunker import Chunker
from src.services.ingestion.embedder import get_embedder
from src.vector_db.client import get_vector_client
from src.utils.logger.logger import get_logger

logger = get_logger(name="Indexer")


class Indexer:
    """Orchestrates document ingestion into the vector store."""

    def __init__(self):
        self.chunker = Chunker()
        self.embedder = get_embedder()
        self.vector_client = get_vector_client()

    def ingest_file(
        self,
        file_path: str,
        user_id: str = "",
        org_id: str = "default",
        tags: Optional[List[str]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Full ingestion pipeline for a single file.
        Returns summary dict with doc_id, chunk count, etc.
        """
        doc_id = str(uuid.uuid4())
        filename = os.path.basename(file_path)
        logger.info(f"Ingesting file: {filename} (doc_id={doc_id})")

        # 1. Chunk
        chunks = self.chunker.process_file(file_path)
        if not chunks:
            return {"doc_id": doc_id, "status": "empty", "chunks": 0}

        # 2. Embed
        _, embeddings = self.embedder.embed_chunks(chunks)

        # 3. Build metadata
        base_meta = {
            "filename": filename,
            "tags": tags or [],
            **(extra_metadata or {}),
        }
        metadata_list = [base_meta.copy() for _ in chunks]

        # 4. Store
        texts = [c["text"] for c in chunks]
        ids = self.vector_client.insert(
            texts=texts,
            embeddings=embeddings,
            source=filename,
            user_id=user_id,
            org_id=org_id,
            doc_id=doc_id,
            metadata_list=metadata_list,
        )

        logger.info(
            f"Ingested {len(chunks)} chunks for '{filename}' "
            f"(doc_id={doc_id})"
        )

        return {
            "doc_id": doc_id,
            "filename": filename,
            "status": "success",
            "chunks": len(chunks),
            "ids": ids,
        }

    def ingest_text(
        self,
        text: str,
        source: str = "direct_input",
        user_id: str = "",
        org_id: str = "default",
        tags: Optional[List[str]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ingest raw text directly (no file loading)."""
        doc_id = str(uuid.uuid4())

        chunks = self.chunker.chunk_text(text)
        if not chunks:
            return {"doc_id": doc_id, "status": "empty", "chunks": 0}

        _, embeddings = self.embedder.embed_chunks(chunks)

        base_meta = {"tags": tags or [], **(extra_metadata or {})}
        metadata_list = [base_meta.copy() for _ in chunks]
        texts = [c["text"] for c in chunks]

        ids = self.vector_client.insert(
            texts=texts,
            embeddings=embeddings,
            source=source,
            user_id=user_id,
            org_id=org_id,
            doc_id=doc_id,
            metadata_list=metadata_list,
        )

        return {
            "doc_id": doc_id,
            "source": source,
            "status": "success",
            "chunks": len(chunks),
            "ids": ids,
        }
