"""
Celery worker task for asynchronous document ingestion.
"""

from workers.celery_app import celery_app
from src.utils.logger.logger import get_logger

logger = get_logger(name="IngestionWorker")


@celery_app.task(bind=True, name="workers.process_document")
def process_document(
    self,
    file_path: str,
    user_id: str = "",
    org_id: str = "default",
    tags: list = None,
    extra_metadata: dict = None,
):
    """
    Celery task: runs the full ingestion pipeline asynchronously.

    Args:
        file_path: path to the uploaded file on disk
        user_id: ID of the uploading user
        org_id: organization ID
        tags: list of string tags
        extra_metadata: additional metadata dict
    """
    from src.services.ingestion.indexer import Indexer

    logger.info(f"Worker processing: {file_path} (task_id={self.request.id})")

    try:
        indexer = Indexer()
        result = indexer.ingest_file(
            file_path=file_path,
            user_id=user_id,
            org_id=org_id,
            tags=tags or [],
            extra_metadata=extra_metadata or {},
        )

        logger.info(
            f"Worker completed: {result.get('chunks', 0)} chunks "
            f"(task_id={self.request.id})"
        )
        return result

    except Exception as e:
        logger.error(f"Worker failed: {e} (task_id={self.request.id})")
        raise
