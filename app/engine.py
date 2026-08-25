"""ASR engine: model loading and GPU inference (windowed streaming + batch).

v2 design notes:
- The vLLM native incremental-streaming path (`streaming_transcribe`) is
  intentionally NOT used. It can hang the CUDA worker indefinitely, which
  stalls every queued request. All partials use windowed re-decode through
  the same proven `model.transcribe` call path as batch.
- Every GPU submission made through the server passes under a watchdog
  timeout (see server.run_gpu); a wedged decode triggers a controlled exit so
  the supervisor can restart the process cleanly.
"""

from __future__ import annotations

import importlib.util
import time
from typing import Any, Dict, Optional, Union

import numpy as np
from loguru import logger

from app.config import Settings


class Engine:
    _instance: Optional["Engine"] = None

    @classmethod
    def get(cls, settings: Settings) -> "Engine":
        if cls._instance is None:
            cls._instance = cls(settings)
        return cls._instance

    def __init__(self, settings: Settings):
        self.settings = settings
        self.is_ready = False

        if not settings.hf_token:
            logger.warning("No HF_TOKEN set; private/gated models will fail to download")

        self.backend = self._select_backend()
        logger.info(
            f"Loading model '{settings.model_name}' on device='{settings.device_resolved}' "
            f"dtype={settings.dtype} backend={self.backend} quantization={settings.quantization}"
        )
        start = time.perf_counter()

        import torch  # noqa: F401  (required for dtype objects below)
        import qwen_asr

        kwargs: Dict[str, Any] = {}
        if settings.hf_token:
            kwargs["token"] = settings.hf_token

        if self.backend == "vllm":
            self.model = qwen_asr.Qwen3ASRModel.LLM(
                model=settings.model_name,
                gpu_memory_utilization=settings.vllm_gpu_memory_utilization,
                max_new_tokens=settings.vllm_max_new_tokens,
                max_inference_batch_size=settings.vllm_max_batch_size,
            )
        else:
            dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
            kwargs["dtype"] = dtype_map.get(settings.dtype, torch.float32)
            kwargs["device_map"] = (
                "auto" if settings.device_resolved == "cuda" else settings.device_resolved
            )
            if settings.quantization in ("8bit", "int8"):
                kwargs["load_in_8bit"] = True
            elif settings.quantization in ("4bit", "int4"):
                kwargs["load_in_4bit"] = True

            self.model = qwen_asr.Qwen3ASRModel.from_pretrained(settings.model_name, **kwargs)
            if settings.device_resolved == "cuda" and "load_in" not in "".join(kwargs):
                self.model.model = self.model.model.to("cuda")

        logger.info(f"Model loaded in {time.perf_counter() - start:.1f}s")

        # Warm up before reporting ready: vLLM's first requests pay one-time
        # costs (kernel selection / CUDA graphs) that would otherwise be paid
        # by the first real client, appearing as a multi-second stall.
        try:
            t = time.perf_counter()
            silence = np.zeros(self.settings.sample_rate // 2, dtype=np.float32)
            self.transcribe_window(silence, language=self.settings.default_language)
            logger.info(f"Warmup decode done in {time.perf_counter() - t:.1f}s")
        except Exception as e:
            logger.warning(f"Warmup decode failed (non-fatal): {e}")

        self.is_ready = True

    # ------------------------------------------------------------------ #
    # Backend selection                                                  #
    # ------------------------------------------------------------------ #

    def _select_backend(self) -> str:
        s = self.settings
        if s.backend in ("transformers", "vllm"):
            return s.backend
        if s.device_resolved == "cuda" and importlib.util.find_spec("vllm") is not None:
            return "vllm"
        return "transformers"

    # ------------------------------------------------------------------ #
    # Inference                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rms(chunk: np.ndarray) -> float:
        if len(chunk) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(chunk))))

    def transcribe_window(
        self,
        wav: np.ndarray,
        context: str = "",
        language: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Fast low-latency decode of a short rolling window (streaming partials)."""
        s = self.settings
        t0 = time.perf_counter()
        results = self.model.transcribe(
            audio=(np.asarray(wav, dtype=np.float32), s.sample_rate),
            context=context or "",
            language=language or s.default_language,
        )
        res = results[0] if results else None
        text = (res.text if res else "").strip()
        lang = (res.language if res else None) or language or s.default_language
        return {
            "text": text,
            "language": lang,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    def transcribe_utterance(
        self,
        audio_data: Union[np.ndarray, bytes],
        language: Optional[str] = None,
        context: str = "",
    ) -> Dict[str, Any]:
        """Full-quality batch transcription (utterance finals + REST API)."""
        from app.audio_utils import load_audio

        s = self.settings
        if isinstance(audio_data, bytes):
            wav = load_audio(audio_data, target_sr=s.sample_rate)
        else:
            wav = np.asarray(audio_data, dtype=np.float32)

        duration_sec = len(wav) / s.sample_rate
        t0 = time.perf_counter()
        results = self.model.transcribe(
            audio=(wav, s.sample_rate),
            context=context or "",
            language=language or s.default_language,
        )
        elapsed = time.perf_counter() - t0
        res = results[0] if results else None
        return {
            "text": (res.text if res else "").strip(),
            "language": (res.language if res else None) or language or s.default_language,
            "duration_sec": round(duration_sec, 3),
            "latency_ms": round(elapsed * 1000, 2),
            "rtf": round(elapsed / duration_sec, 3) if duration_sec > 0 else 0.0,
        }

    def model_info(self) -> Dict[str, Any]:
        s = self.settings
        return {
            "model": s.model_name,
            "backend": self.backend,
            "device": s.device_resolved,
            "dtype": str(s.dtype),
            "sample_rate": s.sample_rate,
            "default_language": s.default_language,
            "is_ready": self.is_ready,
        }
