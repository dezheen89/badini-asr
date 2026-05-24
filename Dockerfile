FROM python:3.11-slim

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/app/model_cache
ENV TRANSFORMERS_CACHE=/app/model_cache

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
RUN pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN pip install --default-timeout=200 -r requirements.txt

RUN mkdir -p /app/model_cache

# ── Model download ─────────────────────────────────────────────────────────
# ARG is declared first, then promoted to ENV so the RUN step can see it.
ARG HF_TOKEN
ARG MODEL_ID=BadiniAI/whisper-turbo

# Promote to ENV so they are visible inside RUN
ENV HF_TOKEN=${HF_TOKEN}
ENV MODEL_ID=${MODEL_ID}

RUN python - << 'PYEOF'
import os, sys, glob
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

token    = os.environ.get("HF_TOKEN", "").strip()
model_id = os.environ.get("MODEL_ID", "BadiniAI/whisper-turbo").strip()
cache    = "/app/model_cache"

if not token:
    print("ERROR: HF_TOKEN is empty — make sure the secret is set in GitHub.", flush=True)
    sys.exit(1)

print(f"Downloading model: {model_id}", flush=True)
print(f"Token prefix: {token[:8]}...", flush=True)

AutoProcessor.from_pretrained(
    model_id, token=token, cache_dir=cache
)
AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, token=token, cache_dir=cache
)

files = glob.glob(f"{cache}/**/*", recursive=True)
print(f"Done — {len(files)} files in cache.", flush=True)
PYEOF

# ── Scrub the token from the image layer ──────────────────────────────────
# Good practice: unset the secret after it is no longer needed.
ENV HF_TOKEN=""

RUN ls -lh /app/model_cache && du -sh /app/model_cache

COPY runpod_handler.py .

ENV MODEL_DIR=/app/model_cache

CMD ["python", "-u", "runpod_handler.py"]
