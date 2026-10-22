import logging
import os
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from redis import Redis
from celery import Task

from app.celery_app import celery_app
from transcribe_call import load_model, transcribe_call, turns_to_records

log = logging.getLogger(__name__)

try:  # Silero VAD (onnxruntime, CPU) warns about GPU discovery in containers - harmless
    import onnxruntime

    onnxruntime.set_default_logger_severity(3)
except ImportError:
    pass

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium.en")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE") or None  # unset: float16 on GPU, int8 on CPU
WHISPER_BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", "0"))     # 0 = sequential (no batching)
CALLBACK_TIMEOUT = int(os.getenv("CALLBACK_TIMEOUT", "30"))

_model = None  # loaded once per worker process, reused across tasks


class ShutdownWhenQueueEmptyTask(Task):
    def after_return(self, status, retval, task_id, args, kwargs, einfo):
        super().after_return(status, retval, task_id, args, kwargs, einfo)
        if status == "RETRY":
            return

        shutdown_when_queue_empty.delay()


@celery_app.task(bind=True, name="app.tasks.shutdown_when_queue_empty")
def shutdown_when_queue_empty(self):
    time.sleep(10)

    hostname = self.request.hostname
    if not hostname:
        log.warning("Cannot check queue: worker hostname is unavailable")
        return

    try:
        with Redis.from_url(self.app.conf.broker_url) as broker:
            queues = self.app.amqp.queues.keys()
            pending = sum(broker.llen(queue_name) for queue_name in queues)
    except Exception:
        log.exception("Could not check Celery queue; leaving worker running")
        return

    if pending:
        log.info("Worker remains running; %s Celery task(s) are queued", pending)
        return

    log.info("Celery queues are empty; shutting down worker %s", hostname)
    self.app.control.broadcast("shutdown", destination=[hostname])


def get_model():
    global _model
    if _model is None:
        if WHISPER_DEVICE == "cuda":
            import ctranslate2
            import torch

            if ctranslate2.get_cuda_device_count() == 0 or not torch.cuda.is_available():
                raise RuntimeError("WHISPER_DEVICE=cuda but no GPU is visible to the worker")
        _model = load_model(WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE, WHISPER_BATCH_SIZE)
    return _model


# ---------------------------------------------------------------- download

def download_audio(url, dest_dir):
    """Download a presigned S3 URL to dest_dir."""
    name = Path(urlparse(url).path).name or "audio"
    dest = Path(dest_dir) / name
    log.info("Downloading %s", urlparse(url)._replace(query="").geturl())  # don't log the signature
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)
    return dest


# ---------------------------------------------------------------- callback

def send_callback(callback_url, token, payload):
    r = requests.post(
        callback_url,
        json=payload,
        headers={"Authorization": f"Token {token}"},
        timeout=CALLBACK_TIMEOUT,
    )
    if 400 <= r.status_code < 500:
        # Client errors won't fix themselves on retry - log what the API said and give up
        log.error("Callback rejected for call_id=%s: %s %s",
                  payload.get("call_id"), r.status_code, r.text[:1000])
        return False
    r.raise_for_status()  # 5xx -> RequestException -> retried
    return True


@celery_app.task(bind=True, base=ShutdownWhenQueueEmptyTask,
                 autoretry_for=(requests.RequestException,),
                 retry_backoff=True, retry_backoff_max=600, max_retries=5)
def deliver_callback(self, callback_url, token, payload):
    """Separate task so a failing callback is retried without re-transcribing."""
    if send_callback(callback_url, token, payload):
        log.info("Callback delivered for call_id=%s", payload.get("call_id"))


# ---------------------------------------------------------------- transcription

@celery_app.task(bind=True, base=ShutdownWhenQueueEmptyTask,
                 max_retries=2, default_retry_delay=60)
def transcribe_and_notify(self, audio_url, callback_url, token, call_id, options=None):
    options = options or {}
    try:
        model, device = get_model()
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = download_audio(audio_url, tmp)
            turns = transcribe_call(audio_path, model, device, **options)
        chat = turns_to_records(turns)
        status = "completed"
    except Exception as exc:
        log.exception("Transcription failed for call_id=%s", call_id)
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)
        chat = []  # the callback API records an empty chat as "transcript not available"
        status = "failed"

    # The callback API rejects any keys other than call_id and transcript
    payload = {"call_id": call_id, "transcript": {"chat": chat}}
    deliver_callback.delay(callback_url, token, payload)
    return {"call_id": call_id, "status": status, "segments": len(chat)}
