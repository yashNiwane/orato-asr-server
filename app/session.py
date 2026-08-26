"""Streaming session: energy VAD, rolling-window partials, silence-flush finals.

Runs synchronously on the GPU worker thread (submitted under a watchdog by the
server). Event dicts are JSON-serializable and protocol-compatible with v1
clients: speech_start / partial / final / speech_end.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from app.config import Settings
from app.engine import Engine


class StreamingSession:
    def __init__(self, session_id: str, engine: Engine, settings: Settings, language: Optional[str] = None):
        self.session_id = session_id
        self.engine = engine
        self.settings = settings
        self.language = language or settings.language

        sr = settings.sample_rate
        self.chunk_samples = int(settings.stream_chunk_duration * sr)
        self.context_samples = int(settings.stream_context_duration * sr)
        self.silence_flush_samples = int(settings.silence_flush_duration * sr)
        self.min_utterance_samples = int(settings.min_utterance_duration * sr)
        self.max_utterance_samples = int(settings.max_utterance_duration * sr)

        # Buffers / state
        self.audio_buffer = np.zeros((0,), dtype=np.float32)
        self.utterance_buffer = np.zeros((0,), dtype=np.float32)
        self.silence_counter = 0
        self.is_speaking = False
        self.confirmed_transcript = ""
        self.last_partial = ""
        self.last_final_text = ""
        self.final_repeat_count = 0
        # Partial-promotion fast path state
        self._last_partial_text = ""
        self._last_partial_buffer_len = -1

        # Stats
        self.chunks_received = 0
        self.partials_sent = 0
        self.partials_skipped = 0
        self.finals_sent = 0
        self.audio_seconds = 0.0

    # ------------------------------------------------------------------ #

    def process_chunk(self, pcm_chunk: np.ndarray, allow_partial: bool = True) -> List[Dict[str, Any]]:
        """Append one PCM float32 chunk; return transcript/VAD events.

        allow_partial=False processes VAD/buffers but skips the expensive
        interim decode - used by the server when the GPU backlog is deep.
        """
        events: List[Dict[str, Any]] = []
        s = self.settings
        if len(pcm_chunk) == 0:
            return events

        self.chunks_received += 1
        self.audio_seconds += len(pcm_chunk) / s.sample_rate
        self.audio_buffer = np.concatenate([self.audio_buffer, pcm_chunk])
        self.utterance_buffer = np.concatenate([self.utterance_buffer, pcm_chunk])

        rms = Engine._rms(pcm_chunk)
        if rms >= s.vad_energy_threshold:
            self.silence_counter = 0
            if not self.is_speaking:
                self.is_speaking = True
                events.append({"type": "speech_start", "session_id": self.session_id, "rms": round(rms, 5)})
        else:
            self.silence_counter += len(pcm_chunk)

        # Utterance complete: enough trailing silence while speaking.
        if self.is_speaking and self.silence_counter >= self.silence_flush_samples:
            events.extend(self._finalize())
            return events

        # Safety valve: energy-VAD stuck on (noise/echo) would otherwise decode
        # the same rolling window forever - hard-cap the utterance length.
        if self.is_speaking and len(self.utterance_buffer) >= self.max_utterance_samples:
            logger.warning(
                f"[{self.session_id}] max utterance duration "
                f"({self.settings.max_utterance_duration}s) reached - finalizing"
            )
            events.extend(self._finalize())
            return events

        # Interim decode on the fixed cadence while speaking.
        if self.is_speaking and len(self.audio_buffer) >= self.chunk_samples:
            event = self._partial() if allow_partial else None
            if not allow_partial:
                self.partials_skipped += 1
            if event:
                events.append(event)
            self.audio_buffer = np.zeros((0,), dtype=np.float32)

        return events

    def flush(self) -> List[Dict[str, Any]]:
        """Force-finalize whatever is buffered (client-requested turn end)."""
        events: List[Dict[str, Any]] = []
        sr = self.settings.sample_rate
        buf_len = len(self.utterance_buffer)

        # Fast path: the latest partial is fresh AND covers everything buffered
        # (utterance fits inside one context window) -> promote it, skip GPU.
        stale_samples = int(self.settings.final_reuse_max_age_sec * sr)
        fresh = (
            self._last_partial_text
            and 0 <= self._last_partial_buffer_len <= buf_len
            and (buf_len - self._last_partial_buffer_len) <= stale_samples
            and buf_len <= self.context_samples
            and buf_len >= int(0.25 * sr)
        )
        if fresh:
            event = self._commit_final(
                text=self._last_partial_text,
                language=self.language,
                duration_sec=round(buf_len / sr, 3),
                latency_ms=0.0,
                reused=True,
            )
            if event:
                events.append(event)
            self._reset_buffers()
            events.append({"type": "speech_end", "session_id": self.session_id})
            return events

        if buf_len >= int(0.25 * sr):
            events.extend(self._finalize(min_duration_sec=0.25))
        else:
            self._reset_buffers()
        return events

    def reset(self, language: Optional[str] = None) -> None:
        self.language = language or self.language
        self._reset_buffers()
        self.last_final_text = ""
        self.final_repeat_count = 0

    def stats(self) -> Dict[str, Any]:
        return {
            "chunks_received": self.chunks_received,
            "partials_sent": self.partials_sent,
            "partials_skipped": self.partials_skipped,
            "finals_sent": self.finals_sent,
            "audio_seconds": round(self.audio_seconds, 2),
        }

    # ------------------------------------------------------------------ #

    def _finalize(self, min_duration_sec: Optional[float] = None) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        duration = (
            min_duration_sec
            if min_duration_sec is not None
            else self.settings.min_utterance_duration
        )
        min_samples = int(duration * self.settings.sample_rate)
        if len(self.utterance_buffer) >= min_samples:
            # No context feedback by default (see Settings.context_feedback):
            # a single hallucination must not anchor every future decode.
            context = (
                self.confirmed_transcript[-100:]
                if self.settings.context_feedback and self.confirmed_transcript
                else ""
            )
            result = self.engine.transcribe_utterance(
                self.utterance_buffer, language=self.language, context=context
            )
            text = result["text"].strip()
            if text and text != "<unintelligible>":
                event = self._commit_final(
                    text=text,
                    language=result["language"],
                    duration_sec=result["duration_sec"],
                    latency_ms=result["latency_ms"],
                )
                if event:
                    events.append(event)

        self._reset_buffers()
        events.append({"type": "speech_end", "session_id": self.session_id})
        return events

    def _commit_final(
        self,
        *,
        text: str,
        language,
        duration_sec: float,
        latency_ms: float,
        reused: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Dedupe-guarded final emission shared by batch decode + fast path."""
        # Anti-hallucination guard: never let an identical final chain more
        # than twice (degenerate repetition loops on noisy audio).
        if text == self.last_final_text:
            self.final_repeat_count += 1
        else:
            self.final_repeat_count = 0
        self.last_final_text = text

        if self.final_repeat_count >= 2:
            logger.warning(
                f"[{self.session_id}] suppressed repeating final #{self.final_repeat_count}: {text[:60]}"
            )
            return None

        self.confirmed_transcript = f"{self.confirmed_transcript} {text}".strip()
        self.finals_sent += 1
        logger.info(
            f"[{self.session_id}] final ({'reused partial' if reused else 'decoded'}, {latency_ms}ms): {text[:60]}"
        )
        return {
            "type": "final",
            "session_id": self.session_id,
            "text": text,
            "cumulative_text": self.confirmed_transcript,
            "language": language,
            "duration_sec": duration_sec,
            "latency_ms": latency_ms,
        }

    def _partial(self) -> Optional[Dict[str, Any]]:
        segment = self.utterance_buffer[-self.context_samples :]
        result = self.engine.transcribe_window(
            segment,
            language=self.language,
            max_tokens=self.settings.max_stream_tokens,
        )
        text = result["text"].strip()
        self._last_partial_text = text or ""
        self._last_partial_buffer_len = len(self.utterance_buffer)
        if not text or text == "<unintelligible>" or text == self.last_partial:
            return None
        self.last_partial = text
        self.partials_sent += 1
        return {
            "type": "partial",
            "session_id": self.session_id,
            "text": text,
            "cumulative_text": f"{self.confirmed_transcript} {text}".strip(),
            "language": result["language"],
            "latency_ms": result["latency_ms"],
        }

    def _reset_buffers(self) -> None:
        z = np.zeros((0,), dtype=np.float32)
        self.audio_buffer = z.copy()
        self.utterance_buffer = z.copy()
        self.is_speaking = False
        self.silence_counter = 0
        self.last_partial = ""
        self._last_partial_text = ""
        self._last_partial_buffer_len = -1
