# CUDA 12 + cuDNN 9 runtime: required by faster-whisper (ctranslate2 4.5+) on GPU
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip libgomp1 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY transcribe_call.py .
COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
