"""
FastAPI dependency injection functions.
"""

from fastapi import Depends

from src.core.security import get_current_user
from src.models.database import get_user_store, UserStore
from src.vector_db.client import get_vector_client, VectorDBClient
from src.controllers.rag_controller import get_rag_controller, RAGController


async def get_db() -> UserStore:
    """Return the user store."""
    return get_user_store()


async def get_vector_store() -> VectorDBClient:
    """Return the vector DB client."""
    return get_vector_client()


async def get_controller() -> RAGController:
    """Return the RAG controller."""
    return get_rag_controller()
