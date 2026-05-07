import base64
import io
import os
import traceback

import numpy as np
import runpod
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoModelForCTC

# ── Configuration (all overridable via environment variables) ──
MODEL_ID = os.getenv("MODEL_ID", "BadiniAI/BadiniW2VBert")
MODEL_CACHE_DIR = os.getenv("MODEL_DIR", "/app/model_cache")
TARGET_SR = 16000
HF_TOKEN = os.getenv("HF_TOKEN")

MAX_SECONDS = 20
MIN_PEAK = 0.005
TRIM_THRESHOLD = 0.001

torch.set_num_threads(1)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {MODEL_ID}")
print(f"Using device: {device}")
print(f"Cache dir: {MODEL_CACHE_DIR}")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is missing. Set it in RunPod environment variables.")

# ── Load model from local cache (pre-downloaded during Docker build) ──
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    cache_dir=MODEL_CACHE_DIR,
    local_files_only=True  # Force local loading — no network download
)
model = AutoModelForCTC.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    cache_dir=MODEL_CACHE_DIR,
    local_files_only=True  # Force local loading — no network download
).to(device)

model.eval()

if device == "cuda":
    try:
        model = torch.compile(model)
        print("torch.compile enabled")
    except Exception as e:
        print(f"torch.compile skipped: {e}")

# ── Warmup: run a dummy inference so CUDA kernels are compiled ──
print("Running warmup inference...")
try:
    dummy_audio = np.zeros(TARGET_SR, dtype=np.float32)  # 1 second of silence
    dummy_inputs = processor(
        dummy_audio,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True
    )
    if "input_values" in dummy_inputs:
        dummy_model_inputs = {"input_values": dummy_inputs["input_values"].to(device)}
    elif "input_features" in dummy_inputs:
        dummy_model_inputs = {"input_features": dummy_inputs["input_features"].to(device)}

    if "attention_mask" in dummy_inputs:
        dummy_model_inputs["attention_mask"] = dummy_inputs["attention_mask"].to(device)

    with torch.inference_mode():
        _ = model(**dummy_model_inputs)

    print("Warmup complete — CUDA kernels ready.")
except Exception as e:
    print(f"Warmup skipped: {e}")

print("Model loaded successfully. Ready for requests.")


# ── Audio helpers ──

def read_audio_bytes(audio_bytes: bytes):
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return np.asarray(audio, dtype=np.float32), sr


def resample_audio(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio.astype(np.float32)

    duration = len(audio) / float(sr)
    target_len = int(duration * target_sr)

    if target_len <= 0:
        raise ValueError("Invalid audio length after resampling.")

    x_old = np.linspace(0, 1, num=len(audio), endpoint=False)
    x_new = np.linspace(0, 1, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def trim_silence(audio: np.ndarray, threshold: float = TRIM_THRESHOLD) -> np.ndarray:
    if audio.size == 0:
        return audio

    mask = np.abs(audio) > threshold
    if not np.any(mask):
        return audio

    start = np.argmax(mask)
    end = len(mask) - np.argmax(mask[::-1])
    return audio[start:end]


def clean_text(text: str) -> str:
    text = text.replace("[PAD]", "").replace("[UNK]", "").strip()
    return " ".join(text.split())


# ── RunPod handler ──

def handler(job):
    try:
        input_data = job.get("input", {})
        wav_base64 = input_data.get("wav_base64")

        if not wav_base64:
            return {"ok": False, "error": "Missing 'wav_base64' in input."}

        try:
            audio_bytes = base64.b64decode(wav_base64)
        except Exception:
            return {"ok": False, "error": "Invalid base64 audio data."}

        audio, sr = read_audio_bytes(audio_bytes)

        if audio.size == 0:
            return {"ok": False, "error": "Decoded audio is empty."}

        audio = trim_silence(audio)
        audio = resample_audio(audio, sr, TARGET_SR)

        if audio.size == 0:
            return {
                "ok": True,
                "text": "",
                "message": "No speech detected",
                "sample_rate": TARGET_SR
            }

        if len(audio) > TARGET_SR * MAX_SECONDS:
            return {
                "ok": False,
                "error": f"Audio too long. Maximum allowed length is {MAX_SECONDS} seconds."
            }

        peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
        if peak < MIN_PEAK:
            return {
                "ok": True,
                "text": "",
                "message": "No speech detected",
                "sample_rate": TARGET_SR
            }

        inputs = processor(
            audio,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True
        )

        if "input_values" in inputs:
            model_inputs = {"input_values": inputs["input_values"].to(device)}
        elif "input_features" in inputs:
            model_inputs = {"input_features": inputs["input_features"].to(device)}
        else:
            return {
                "ok": False,
                "error": f"Unsupported processor output keys: {list(inputs.keys())}"
            }

        if "attention_mask" in inputs:
            model_inputs["attention_mask"] = inputs["attention_mask"].to(device)

        with torch.inference_mode():
            outputs = model(**model_inputs)
            logits = outputs.logits
            pred_ids = torch.argmax(logits, dim=-1)

        text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
        text = clean_text(text)

        return {
            "ok": True,
            "text": text,
            "sample_rate": TARGET_SR,
            "device": device
        }

    except Exception:
        print(traceback.format_exc())
        return {
            "ok": False,
            "error": "Internal server error"
        }


runpod.serverless.start({"handler": handler})
