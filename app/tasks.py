import logging
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests

from app.celery_app import celery_app
from transcribe_call import load_model, transcribe_call, turns_to_records

log = logging.getLogger(__name__)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium.en")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
CALLBACK_TIMEOUT = int(os.getenv("CALLBACK_TIMEOUT", "30"))

_model = None  # loaded once per worker process, reused across tasks


def get_model():
    global _model
    if _model is None:
        if WHISPER_DEVICE == "cuda":
            import ctranslate2
            import torch

            if ctranslate2.get_cuda_device_count() == 0 or not torch.cuda.is_available():
                raise RuntimeError("WHISPER_DEVICE=cuda but no GPU is visible to the worker")
        _model = load_model(WHISPER_MODEL, WHISPER_DEVICE)
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
        headers={"Authorization": f"Bearer {token}"},
        timeout=CALLBACK_TIMEOUT,
    )
    r.raise_for_status()


@celery_app.task(bind=True, autoretry_for=(requests.RequestException,),
                 retry_backoff=True, retry_backoff_max=600, max_retries=5)
def deliver_callback(self, callback_url, token, payload):
    """Separate task so a failing callback is retried without re-transcribing."""
    send_callback(callback_url, token, payload)
    log.info("Callback delivered for call_id=%s", payload.get("call_id"))


# ---------------------------------------------------------------- transcription

@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
def transcribe_and_notify(self, audio_url, callback_url, token, call_id, options=None):
    options = options or {}
    try:
        model, device = get_model()
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = download_audio(audio_url, tmp)
            turns = transcribe_call(audio_path, model, device, **options)
        payload = {
            "call_id": call_id,
            "status": "completed",
            "transcript": turns_to_records(turns),
        }
    except Exception as exc:
        log.exception("Transcription failed for call_id=%s", call_id)
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)
        payload = {"call_id": call_id, "status": "failed", "error": str(exc)}

    deliver_callback.delay(callback_url, token, payload)
    return {"call_id": call_id, "status": payload["status"]}
