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

# Create cache directory
RUN mkdir -p /app/model_cache

# Pre-download the model into the image at build time
ARG HF_TOKEN
RUN HF_TOKEN=${HF_TOKEN} python -c "\
import os; \
from transformers import AutoProcessor, AutoModelForCTC; \
token = os.environ.get('HF_TOKEN'); \
print(f'Using token: {token[:10]}...'); \
processor = AutoProcessor.from_pretrained('BadiniAI/BadiniW2VBert', token=token, cache_dir='/app/model_cache'); \
model = AutoModelForCTC.from_pretrained('BadiniAI/BadiniW2VBert', token=token, cache_dir='/app/model_cache'); \
print('Model pre-downloaded successfully.'); \
import glob; \
files = glob.glob('/app/model_cache/**/*', recursive=True); \
print(f'Total files in cache: {len(files)}'); \
"

# Verify cache contents
RUN ls -lh /app/model_cache && du -sh /app/model_cache

COPY runpod_handler.py .

ENV MODEL_DIR=/app/model_cache

CMD ["python", "-u", "runpod_handler.py"]
