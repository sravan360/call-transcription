import logging
import os
from typing import Literal, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, HttpUrl

from app.celery_app import celery_app, get_queued_task_count

log = logging.getLogger(__name__)

app = FastAPI(title="Call Transcription API")

SHUTDOWN_URL = os.getenv(
    "SHUTDOWN_URL", "https://up548gw5y6.execute-api.us-east-2.amazonaws.com/default/BQA_CWA_Stop"
)
CALLBACK_TIMEOUT = int(os.getenv("CALLBACK_TIMEOUT", "30"))
SHUTDOWN_TASK_NAME = "app.tasks.shutdown_when_queue_empty"


class TranscribeOptions(BaseModel):
    language: str = "en"
    agent_channel: Literal["left", "right"] = "left"
    speakers: int = Field(2, ge=1, le=6)
    swap_speakers: bool = False
    force_mono: bool = False


class TranscribeRequest(BaseModel):
    audio_url: str = Field(..., pattern=r"^https?://", description="Presigned S3 URL of the recording")
    api: HttpUrl = Field(..., description="Callback URL that receives the transcript")
    token: str = Field(..., description="Bearer token sent to the callback URL")
    call_id: str
    agent_channel: Optional[Literal["left", "right"]] = Field(
        None, description="Stereo channel carrying the agent; overrides options.agent_channel"
    )
    options: Optional[TranscribeOptions] = None


class TranscribeResponse(BaseModel):
    task_id: str
    call_id: str
    status: str


@app.post("/transcribe", response_model=TranscribeResponse, status_code=202)
def transcribe(req: TranscribeRequest):
    options = req.options or TranscribeOptions()
    if req.agent_channel:
        options.agent_channel = req.agent_channel

    # Enqueue by name so the API process never imports Whisper / torch
    task = celery_app.send_task(
        "app.tasks.transcribe_and_notify",
        args=[
            req.audio_url,  # passed through untouched so the signature stays valid
            str(req.api),
            req.token,
            req.call_id,
            options.model_dump(),
        ],
    )
    return TranscribeResponse(task_id=task.id, call_id=req.call_id, status="queued")


@app.get("/tasks/{task_id}")
def task_status(task_id: str):
    result = celery_app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "state": result.state,
        "result": result.result if result.successful() else None,
    }


def count_worker_tasks(workers):
    # The shutdown-check task itself doesn't count as work
    return sum(1 for tasks in (workers or {}).values() for t in tasks if t.get("name") != SHUTDOWN_TASK_NAME)


def get_queue_status():
    """Queued (in Redis) plus active/reserved (held by workers) transcription tasks."""
    try:
        queued = get_queued_task_count()
        inspect = celery_app.control.inspect(timeout=2)
        active = count_worker_tasks(inspect.active())
        reserved = count_worker_tasks(inspect.reserved())
    except Exception as exc:
        log.exception("Could not check Celery queue")
        raise HTTPException(status_code=503, detail=f"Could not check Celery queue: {exc}")

    return {"queued": queued, "active": active, "reserved": reserved, "empty": not (queued or active or reserved)}


@app.get("/queue")
def queue_status():
    return get_queue_status()


@app.post("/shutdown-if-idle")
def shutdown_if_idle():
    status = get_queue_status()
    if not status["empty"]:
        return {**status, "shutdown": False}

    log.info("Celery queue is empty; stopping server via %s", SHUTDOWN_URL)
    try:
        r = requests.post(SHUTDOWN_URL, timeout=CALLBACK_TIMEOUT)
        r.raise_for_status()
    except requests.RequestException as exc:
        log.exception("Server stop API call failed")
        raise HTTPException(status_code=502, detail=f"Server stop API call failed: {exc}")
    return {**status, "shutdown": True}


@app.get("/health")
def health():
    return {"status": "ok"}
