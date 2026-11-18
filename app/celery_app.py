import os

from celery import Celery
from redis import Redis

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
    # Survive Redis/DNS blips instead of leaving the worker with a dead connection
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=None,   # retry forever
    broker_transport_options={"socket_keepalive": True, "retry_on_timeout": True},
    redis_backend_health_check_interval=30,
    redis_retry_on_timeout=True,
    redis_socket_keepalive=True,
)


def get_queued_task_count(app=celery_app):
    """Tasks waiting in the Redis broker queues (not yet picked up by a worker)."""
    with Redis.from_url(app.conf.broker_url) as broker:
        queues = app.amqp.queues.keys()
        return sum(broker.llen(queue_name) for queue_name in queues)
