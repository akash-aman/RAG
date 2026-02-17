"""
Query API routes.
POST /api/v1/query — end-to-end RAG query with optional streaming.
"""

import json
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from src.core.security import get_current_user, sanitize_query
from src.models.schemas import QueryRequest, QueryResponse, Source
from src.controllers.rag_controller import get_rag_controller
from src.utils.logger.logger import get_logger

router = APIRouter(prefix="/query", tags=["Query"])
logger = get_logger(name="QueryAPI")


@router.post("", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    user: dict = Depends(get_current_user),
):
    """
    Execute an end-to-end RAG query.

    The pipeline: query enhancement → hybrid search → reranking →
    prompt optimization → LLM generation → self-reflection.

    Set `stream: true` for Server-Sent Events streaming.
    """
    # Sanitize input
    clean_query = sanitize_query(request.query)

    controller = get_rag_controller()

    # ── Streaming mode ───────────────────────────────────────────
    if request.stream:
        def event_stream():
            try:
                for chunk in controller.query_stream(
                    query=clean_query,
                    filters=request.filters,
                    user_id=user.get("sub"),
                    org_id=user.get("org_id"),
                    top_k=request.top_k,
                    enable_hyde=request.enable_hyde,
                    enable_reranking=request.enable_reranking,
                ):
                    data = json.dumps({"content": chunk})
                    yield f"data: {data}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # ── Standard mode ────────────────────────────────────────────
    try:
        result = controller.query(
            query=clean_query,
            filters=request.filters,
            user_id=user.get("sub"),
            org_id=user.get("org_id"),
            top_k=request.top_k,
            enable_hyde=request.enable_hyde,
            enable_reranking=request.enable_reranking,
            enable_self_rag=request.enable_self_rag,
        )

        sources = [
            Source(
                text=s.get("text", ""),
                source=s.get("source", ""),
                score=s.get("score", 0.0),
                metadata=s.get("metadata", {}),
            )
            for s in result.get("sources", [])
        ]

        return QueryResponse(
            answer=result.get("answer", ""),
            sources=sources,
            metadata=result.get("metadata", {}),
        )

    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
