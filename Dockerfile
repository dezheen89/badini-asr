FROM python:3.11-slim

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV MODEL_DIR=/app/model_cache

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

# Pre-download the model into the image at build time
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
RUN python -c "\
import os; \
token = os.environ['HF_TOKEN']; \
from transformers import AutoProcessor, AutoModelForCTC; \
AutoProcessor.from_pretrained('BadiniAI/BadiniW2VBert', token=token, cache_dir='/app/model_cache'); \
AutoModelForCTC.from_pretrained('BadiniAI/BadiniW2VBert', token=token, cache_dir='/app/model_cache'); \
print('Model pre-downloaded successfully.')"

COPY runpod_handler.py .

CMD ["python", "-u", "runpod_handler.py"]
