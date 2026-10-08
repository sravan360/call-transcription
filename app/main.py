from typing import Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field, HttpUrl

from app.celery_app import celery_app

app = FastAPI(title="Call Transcription API")


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
    options: Optional[TranscribeOptions] = None


class TranscribeResponse(BaseModel):
    task_id: str
    call_id: str
    status: str


@app.post("/transcribe", response_model=TranscribeResponse, status_code=202)
def transcribe(req: TranscribeRequest):
    # Enqueue by name so the API process never imports Whisper / torch
    task = celery_app.send_task(
        "app.tasks.transcribe_and_notify",
        args=[
            req.audio_url,  # passed through untouched so the signature stays valid
            str(req.api),
            req.token,
            req.call_id,
            req.options.model_dump() if req.options else None,
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


@app.get("/health")
def health():
    return {"status": "ok"}
