"""
Celery application configuration.
Uses Redis as broker and result backend.
"""

from celery import Celery
from src.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "rag_workers",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # Auto-discover tasks in workers module
    imports=["workers.ingestion_worker"],
)
