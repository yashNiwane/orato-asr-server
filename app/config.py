"""Central configuration. Everything is overridable via environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _env_int(name: str, default: str) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # --- Model ---
    model_name: str = field(default_factory=lambda: os.getenv("ASR_MODEL", "Qwen/Qwen3-ASR-0.6B"))
    hf_token: str = field(
        default_factory=lambda: os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN", "")
    )
    device: str = field(default_factory=lambda: os.getenv("DEVICE", "auto"))  # auto|cuda|cpu
    dtype: str = field(default_factory=lambda: os.getenv("TORCH_DTYPE", "auto"))
    quantization: str = field(default_factory=lambda: os.getenv("QUANTIZATION", "none").lower())
    backend: str = field(default_factory=lambda: os.getenv("ASR_BACKEND", "auto").lower())

    # --- vLLM ---
    vllm_gpu_memory_utilization: float = field(
        default_factory=lambda: _env_float("VLLM_GPU_MEMORY_UTILIZATION", "0.85")
    )
    vllm_max_new_tokens: int = field(default_factory=lambda: _env_int("VLLM_MAX_NEW_TOKENS", "256"))
    vllm_max_batch_size: int = field(default_factory=lambda: _env_int("VLLM_MAX_BATCH_SIZE", "32"))

    # --- Audio ---
    sample_rate: int = field(default_factory=lambda: _env_int("SAMPLE_RATE", "16000"))
    default_language: str = field(default_factory=lambda: os.getenv("DEFAULT_LANGUAGE", "Hindi"))

    # --- Streaming (telecalling-tuned defaults) ---
    stream_chunk_duration: float = field(
        default_factory=lambda: _env_float("STREAM_CHUNK_DURATION", "0.15")
    )  # interim decode cadence while speaking
    stream_context_duration: float = field(
        default_factory=lambda: _env_float("STREAM_CONTEXT_DURATION", "3.0")
    )  # rolling context window for partials
    silence_flush_duration: float = field(
        default_factory=lambda: _env_float("SILENCE_DURATION_FLUSH", "0.4")
    )  # trailing silence that finalizes an utterance
    vad_energy_threshold: float = field(
        default_factory=lambda: _env_float("VAD_ENERGY_THRESHOLD", "0.003")
    )
    min_utterance_duration: float = field(
        default_factory=lambda: _env_float("MIN_UTTERANCE_DURATION", "0.35")
    )
    max_stream_tokens: int = field(default_factory=lambda: _env_int("MAX_STREAM_TOKENS", "24"))

    # --- Reliability ---
    decode_timeout_sec: float = field(
        default_factory=lambda: _env_float("DECODE_TIMEOUT_SEC", "20")
    )  # hard ceiling per GPU submission; beyond this the GPU is considered wedged
    gpu_exit_on_wedge: bool = field(
        default_factory=lambda: _env_bool("GPU_EXIT_ON_WEDGE", "true")
    )  # exit so a supervisor (e.g. notebook monitor) can restart cleanly
    wedge_exit_delay_sec: float = field(
        default_factory=lambda: _env_float("WEDGE_EXIT_DELAY_SEC", "2")
    )  # grace period to flush an error event to clients before exiting
    gpu_workers: int = field(default_factory=lambda: _env_int("GPU_WORKERS", "1"))

    # --- Server ---
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("PORT", "8000"))
    ws_queue_size: int = field(default_factory=lambda: _env_int("WS_QUEUE_SIZE", "100"))

    @property
    def is_cuda(self) -> bool:
        return self.device_resolved == "cuda"

    device_resolved: str = field(default="cpu", init=False)

    def resolve(self) -> "Settings":
        """Resolve 'auto' values that need torch. Call once at startup."""
        if self.device == "auto" or self.dtype == "auto":
            import torch

            device = self.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = self.dtype
            if dtype == "auto":
                dtype = "float16" if device == "cuda" else "float32"
            object.__setattr__(self, "device_resolved", device)
            object.__setattr__(self, "dtype", dtype)
        else:
            object.__setattr__(self, "device_resolved", self.device)
        return self


settings = Settings().resolve()
