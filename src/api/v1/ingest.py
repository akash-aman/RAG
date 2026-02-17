"""
Ingestion API routes.
POST   /api/v1/ingest           — upload and process documents
GET    /api/v1/ingest           — list ingested documents
DELETE /api/v1/ingest/{doc_id}  — delete a specific document
DELETE /api/v1/ingest           — delete all documents
"""

import os
import uuid
import shutil
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException

from src.core.config import get_settings
from src.core.security import get_current_user
from src.models.schemas import IngestResponse, DocumentInfo, DeleteResponse
from src.services.ingestion.indexer import Indexer
from src.vector_db.client import get_vector_client
from src.utils.logger.logger import get_logger

router = APIRouter(prefix="/ingest", tags=["Ingestion"])
logger = get_logger(name="IngestAPI")


@router.post("", response_model=IngestResponse, status_code=202, summary="Upload document")
async def ingest_document(
    file: UploadFile = File(...),
    tags: Optional[str] = Form(default=""),
    metadata: Optional[str] = Form(default=""),
    user: dict = Depends(get_current_user),
):
    """
    Upload a document for ingestion into the RAG knowledge base.

    Supports: **PDF, TXT, MD, CSV, JSON, HTML**

    The document is chunked, embedded, and stored in the vector database.
    Returns a task_id and chunk count on success.
    """
    settings = get_settings()
    task_id = str(uuid.uuid4())

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    allowed_extensions = {".pdf", ".txt", ".md", ".csv", ".json", ".html"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {allowed_extensions}",
        )

    # Save uploaded file
    upload_dir = settings.upload_dir
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, f"{task_id}_{file.filename}")

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Parse tags
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    # Parse extra metadata
    extra_meta = {}
    if metadata:
        try:
            import json
            extra_meta = json.loads(metadata)
        except Exception:
            extra_meta = {"raw_metadata": metadata}

    try:
        indexer = Indexer()
        result = indexer.ingest_file(
            file_path=file_path,
            user_id=user.get("sub", ""),
            org_id=user.get("org_id", "default"),
            tags=tag_list,
            extra_metadata=extra_meta,
        )

        logger.info(f"Ingested {result.get('chunks', 0)} chunks (task_id={task_id})")

        return IngestResponse(
            task_id=task_id,
            status="completed",
            message=f"Processed {result.get('chunks', 0)} chunks from {file.filename}",
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)


@router.get(
    "",
    response_model=List[DocumentInfo],
    summary="List ingested documents",
)
async def list_documents(
    user: dict = Depends(get_current_user),
    limit: int = 100,
):
    """
    List all documents ingested into the knowledge base.

    Returns grouped document info with chunk counts.
    Filtered to the current user's organization.
    """
    try:
        client = get_vector_client()
        docs = client.list_documents(
            user_id=user.get("user_id"),
            limit=limit,
        )
        return [DocumentInfo(**d) for d in docs]
    except Exception as e:
        logger.error(f"List documents failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {e}")


@router.delete(
    "/{doc_id}",
    response_model=DeleteResponse,
    summary="Delete a document",
)
async def delete_document(
    doc_id: str,
    user: dict = Depends(get_current_user),
):
    """
    Delete a specific document and all its chunks from the knowledge base.
    Only deletes documents owned by the current user.

    - **doc_id**: The document ID to delete (from the list endpoint)
    """
    try:
        client = get_vector_client()
        user_id = user.get("user_id") or user.get("sub", "")
        deleted = client.delete_by_doc_id(doc_id, user_id=user_id)
        if deleted == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Document {doc_id} not found or not owned by you",
            )
        logger.info(f"Deleted doc_id={doc_id} by user={user_id}, {deleted} chunks removed")
        return DeleteResponse(
            deleted=deleted,
            message=f"Deleted document {doc_id} ({deleted} chunks)",
        )
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {e}")


@router.delete(
    "",
    response_model=DeleteResponse,
    summary="Delete all documents",
)
async def delete_all_documents(
    user: dict = Depends(get_current_user),
):
    """
    Delete **all** documents owned by the current user.

    ⚠️ This action is irreversible. Only your documents will be deleted.
    """
    try:
        client = get_vector_client()
        user_id = user.get("user_id") or user.get("sub", "")
        deleted = client.delete_by_user(user_id)
        logger.info(f"Deleted all docs for user={user_id}: {deleted} entities removed")
        return DeleteResponse(
            deleted=deleted,
            message=f"Deleted all your documents ({deleted} entities)",
        )
    except Exception as e:
        logger.error(f"Delete all failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete all: {e}")
