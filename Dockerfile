FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV MODEL_DIR=/app/model_cache

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --default-timeout=200 -r requirements.txt

RUN mkdir -p /app/model_cache

# Pre-download the model into the image at build time
ARG HF_TOKEN
RUN python -c "\
from transformers import AutoProcessor, AutoModelForCTC; \
AutoProcessor.from_pretrained('BadiniAI/BadiniW2VBert', token='${HF_TOKEN}', cache_dir='/app/model_cache'); \
AutoModelForCTC.from_pretrained('BadiniAI/BadiniW2VBert', token='${HF_TOKEN}', cache_dir='/app/model_cache'); \
print('Model pre-downloaded successfully.')"

COPY runpod_handler.py .

CMD ["python", "-u", "runpod_handler.py"]
