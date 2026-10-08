import os

from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery("transcription", broker=REDIS_URL, backend=REDIS_URL, include=["app.tasks"])

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,              # re-queue the job if a worker dies mid-transcription
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,     # transcription is long-running; take one job at a time
    result_expires=24 * 3600,
)
