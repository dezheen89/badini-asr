import base64
import gc
import io
import os
import threading
import time
import traceback

import numpy as np
import runpod
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoModelForCTC

# ════════════════════════════════════════════════════════════════════
#  CONFIGURATION (all overridable via environment variables)
# ════════════════════════════════════════════════════════════════════
MODEL_ID = os.getenv("MODEL_ID", "BadiniAI/BadiniW2VBert")
MODEL_CACHE_DIR = os.getenv("MODEL_DIR", "/app/model_cache")
TARGET_SR = 16000
HF_TOKEN = os.getenv("HF_TOKEN")

MAX_SECONDS = 20
MIN_PEAK = 0.005
TRIM_THRESHOLD = 0.001

# Idle deactivation: free the model from memory after this many idle seconds.
IDLE_TIMEOUT_SECONDS = int(os.getenv("IDLE_TIMEOUT_SECONDS", "180"))   # 3 minutes
IDLE_CHECK_INTERVAL = int(os.getenv("IDLE_CHECK_INTERVAL", "30"))      # check cadence

# CPU thread + CUDA tuning
torch.set_num_threads(1)
if torch.cuda.is_available():
    # Lets cuDNN pick the fastest kernels for the (mostly fixed) input shapes.
    torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"
# fp16 only makes sense (and is only safe) on CUDA. CPU fp16 ops mostly fail.
USE_FP16 = device == "cuda"

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is missing. Set it in RunPod environment variables.")


# ════════════════════════════════════════════════════════════════════
#  AUDIO HELPERS  (unchanged logic)
# ════════════════════════════════════════════════════════════════════

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


def _log_gpu_memory(tag: str):
    if device == "cuda":
        mb = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"[MEM] {tag}: {mb:.1f} MB allocated on GPU", flush=True)


# ════════════════════════════════════════════════════════════════════
#  SINGLETON SERVICE  (thread-safe, lazy-loaded, idle-deactivatable)
# ════════════════════════════════════════════════════════════════════

class TranscribeService:
    _instance = None
    _lock = threading.Lock()          # guards instance creation/teardown
    last_request_time = time.time()   # class-level: survives instance teardown

    # ---- Singleton accessor: double-checked locking ----
    @classmethod
    def get_instance(cls):
        if cls._instance is None:                 # 1st check — no lock (fast path)
            with cls._lock:
                if cls._instance is None:         # 2nd check — under lock
                    print("[COLD START] No live model — loading now...", flush=True)
                    cls._instance = cls._create()
        return cls._instance

    # ---- Real constructor, only ever called once per live cycle ----
    @classmethod
    def _create(cls):
        self = object.__new__(cls)        # bypass __init__ to keep it controlled
        self._load_model()
        return self

    def _load_model(self):
        print(f"[LOAD] Model: {MODEL_ID}", flush=True)
        print(f"[LOAD] Device: {device} | fp16: {USE_FP16}", flush=True)
        print(f"[LOAD] Cache dir: {MODEL_CACHE_DIR}", flush=True)

        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN,
            cache_dir=MODEL_CACHE_DIR,
            local_files_only=True,    # no network — model baked into image
        )

        model = AutoModelForCTC.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN,
            cache_dir=MODEL_CACHE_DIR,
            local_files_only=True,
            torch_dtype=torch.float16 if USE_FP16 else torch.float32,
        ).to(device)
        model.eval()
        self.model = model
        self.dtype = torch.float16 if USE_FP16 else torch.float32

        print("[LOAD] Model loaded.", flush=True)
        _log_gpu_memory("after load")

        self._warmup()
        print("[READY] Service ready for requests.", flush=True)

    # ---- Warmup so CUDA kernels compile before the first real request ----
    def _warmup(self):
        print("[WARMUP] Running dummy inference...", flush=True)
        try:
            dummy_audio = np.zeros(TARGET_SR, dtype=np.float32)  # 1s silence
            model_inputs = self._prepare_inputs(dummy_audio)
            with torch.inference_mode():
                _ = self.model(**model_inputs)
            print("[WARMUP] Done — CUDA kernels ready.", flush=True)
        except Exception as e:
            print(f"[WARMUP] Skipped: {e}", flush=True)

    # ---- Shared input-prep: processor output -> GPU tensors ----
    def _prepare_inputs(self, audio: np.ndarray) -> dict:
        inputs = self.processor(
            audio,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )

        if "input_values" in inputs:
            key = "input_values"
        elif "input_features" in inputs:
            key = "input_features"
        else:
            raise ValueError(f"Unsupported processor output keys: {list(inputs.keys())}")

        # Cast feature tensor to model dtype, async copy to GPU.
        feat = inputs[key].to(device, dtype=self.dtype, non_blocking=True)
        model_inputs = {key: feat}

        if "attention_mask" in inputs:
            # attention mask stays integer/long — do NOT cast to fp16.
            model_inputs["attention_mask"] = inputs["attention_mask"].to(
                device, non_blocking=True
            )
        return model_inputs

    # ---- The actual transcription for one request ----
    def transcribe(self, audio_bytes: bytes) -> dict:
        audio, sr = read_audio_bytes(audio_bytes)
        if audio.size == 0:
            return {"ok": False, "error": "Decoded audio is empty."}

        audio = trim_silence(audio)
        audio = resample_audio(audio, sr, TARGET_SR)

        if audio.size == 0:
            return {"ok": True, "text": "", "message": "No speech detected",
                    "sample_rate": TARGET_SR}

        if len(audio) > TARGET_SR * MAX_SECONDS:
            return {"ok": False,
                    "error": f"Audio too long. Maximum allowed length is {MAX_SECONDS} seconds."}

        peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
        if peak < MIN_PEAK:
            return {"ok": True, "text": "", "message": "No speech detected",
                    "sample_rate": TARGET_SR}

        model_inputs = self._prepare_inputs(audio)

        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(**model_inputs)
            pred_ids = torch.argmax(outputs.logits, dim=-1)
        if device == "cuda":
            torch.cuda.synchronize()   # accurate timing
        inference_ms = round((time.perf_counter() - t0) * 1000, 2)

        text = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
        text = clean_text(text)

        return {
            "ok": True,
            "text": text,
            "sample_rate": TARGET_SR,
            "device": device,
            "inference_ms": inference_ms,
        }

    # ---- Tear down: free model + GPU memory, reset singleton ----
    @classmethod
    def deactivate(cls):
        with cls._lock:
            if cls._instance is None:
                return
            print("[IDLE] Deactivating worker — freeing model from memory...", flush=True)
            inst = cls._instance
            try:
                del inst.model
                del inst.processor
            except Exception:
                pass
            cls._instance = None

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        _log_gpu_memory("after deactivation")
        print("[IDLE] Memory freed. Worker will lazy-reload on next request.", flush=True)


# ════════════════════════════════════════════════════════════════════
#  IDLE MONITOR  (background daemon thread)
# ════════════════════════════════════════════════════════════════════

def _idle_monitor():
    while True:
        time.sleep(IDLE_CHECK_INTERVAL)
        if TranscribeService._instance is None:
            continue  # already deactivated, nothing to free
        idle_for = time.time() - TranscribeService.last_request_time
        if idle_for >= IDLE_TIMEOUT_SECONDS:
            print(f"[IDLE] No requests for {idle_for:.0f}s "
                  f"(threshold {IDLE_TIMEOUT_SECONDS}s).", flush=True)
            TranscribeService.deactivate()


threading.Thread(target=_idle_monitor, daemon=True).start()


# ════════════════════════════════════════════════════════════════════
#  RUNPOD HANDLER
# ════════════════════════════════════════════════════════════════════

def handler(job):
    try:
        # Mark activity FIRST so the idle monitor never frees mid-request.
        TranscribeService.last_request_time = time.time()
        print("[REQUEST] Received.", flush=True)

        input_data = job.get("input", {})
        wav_base64 = input_data.get("wav_base64")

        if not wav_base64:
            return {"ok": False, "error": "Missing 'wav_base64' in input."}

        try:
            audio_bytes = base64.b64decode(wav_base64)
        except Exception:
            return {"ok": False, "error": "Invalid base64 audio data."}

        # First request loads the model; later requests reuse the same object.
        service = TranscribeService.get_instance()

        # Refresh again after a possibly-slow cold load, then transcribe.
        TranscribeService.last_request_time = time.time()
        result = service.transcribe(audio_bytes)
        TranscribeService.last_request_time = time.time()

        if result.get("inference_ms") is not None:
            print(f"[REQUEST] Done in {result['inference_ms']} ms.", flush=True)
        return result

    except Exception:
        print(traceback.format_exc(), flush=True)
        return {"ok": False, "error": "Internal server error"}


runpod.serverless.start({"handler": handler})
