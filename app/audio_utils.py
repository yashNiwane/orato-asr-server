"""Audio decoding helpers (non-hot-path only).

The WebSocket hot path receives raw PCM16 little-endian frames and converts
them directly with numpy -- no container sniffing. These helpers are for the
REST endpoint and tooling where inputs are encoded files.
"""

from __future__ import annotations

import io

import numpy as np


def load_audio(data: bytes, target_sr: int = 16000) -> np.ndarray:
    """Decode any supported audio container to mono float32 at target_sr."""
    try:
        import soundfile as sf

        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
    except Exception:
        wav, sr = _decode_with_av(data)

    if sr != target_sr:
        wav = _resample(wav, sr, target_sr)
    return np.ascontiguousarray(wav, dtype=np.float32)


def _decode_with_av(data: bytes) -> tuple[np.ndarray, int]:
    import av

    container = av.open(io.BytesIO(data))
    stream = container.streams.audio[0]
    sr = stream.rate
    chunks = []
    for frame in container.decode(audio=0):
        arr = frame.to_ndarray()
        if arr.ndim == 2:  # (channels, samples) -> mono
            arr = arr.mean(axis=0)
        chunks.append(arr.reshape(-1))
    container.close()
    if not chunks:
        raise ValueError("no audio frames found")
    dtype = chunks[0].dtype
    wav = (
        np.concatenate(chunks).astype(np.float32) / 32768.0
        if dtype == np.int16
        else np.concatenate(chunks).astype(np.float32)
    )
    return wav, int(sr)


def _resample(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    try:
        import soxr

        return soxr.resample(wav, src_sr, dst_sr)
    except ImportError:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(src_sr, dst_sr)
        return resample_poly(wav, dst_sr // g, src_sr // g)


def pcm16_to_float(raw: bytes) -> np.ndarray:
    """Hot-path conversion of PCM16 little-endian bytes to float32 [-1, 1]."""
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
