FROM runpod/pytorch:2.2.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/app/model_cache

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --default-timeout=200 -r requirements.txt

RUN mkdir -p /app/model_cache

# ── Model download ─────────────────────────────────────────────────────────
# Wav2Vec2-BERT (CTC) model — must match the handler, which uses AutoModelForCTC.
ARG HF_TOKEN
ARG MODEL_ID=BadiniAI/BadiniW2VBert

ENV HF_TOKEN=${HF_TOKEN}
ENV MODEL_ID=${MODEL_ID}

RUN python - << 'PYEOF'
import os, sys, glob
from transformers import AutoProcessor, AutoModelForCTC

token    = os.environ.get("HF_TOKEN", "").strip()
model_id = os.environ.get("MODEL_ID", "BadiniAI/BadiniW2VBert").strip()
cache    = "/app/model_cache"

if not token:
    print("ERROR: HF_TOKEN is empty — make sure the secret is set in GitHub.", flush=True)
    sys.exit(1)

print(f"Downloading model: {model_id}", flush=True)
print(f"Token prefix: {token[:8]}...", flush=True)

AutoProcessor.from_pretrained(
    model_id, token=token, cache_dir=cache
)
AutoModelForCTC.from_pretrained(
    model_id, token=token, cache_dir=cache
)

files = glob.glob(f"{cache}/**/*", recursive=True)
print(f"Done — {len(files)} files in cache.", flush=True)
PYEOF

# ── Scrub the token from the image layer ──────────────────────────────────
ENV HF_TOKEN=""

RUN ls -lh /app/model_cache && du -sh /app/model_cache

COPY runpod_handler.py .

ENV MODEL_DIR=/app/model_cache

CMD ["python", "-u", "runpod_handler.py"]
