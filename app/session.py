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
        self.language = language or settings.default_language

        sr = settings.sample_rate
        self.chunk_samples = int(settings.stream_chunk_duration * sr)
        self.context_samples = int(settings.stream_context_duration * sr)
        self.silence_flush_samples = int(settings.silence_flush_duration * sr)
        self.min_utterance_samples = int(settings.min_utterance_duration * sr)

        # Buffers / state
        self.audio_buffer = np.zeros((0,), dtype=np.float32)
        self.utterance_buffer = np.zeros((0,), dtype=np.float32)
        self.silence_counter = 0
        self.is_speaking = False
        self.confirmed_transcript = ""
        self.last_partial = ""
        self.last_final_text = ""
        self.final_repeat_count = 0

        # Stats
        self.chunks_received = 0
        self.partials_sent = 0
        self.finals_sent = 0
        self.audio_seconds = 0.0

    # ------------------------------------------------------------------ #

    def process_chunk(self, pcm_chunk: np.ndarray) -> List[Dict[str, Any]]:
        """Append one PCM float32 chunk; return transcript/VAD events."""
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

        # Interim decode on the fixed cadence while speaking.
        if self.is_speaking and len(self.audio_buffer) >= self.chunk_samples:
            event = self._partial()
            if event:
                events.append(event)
            self.audio_buffer = np.zeros((0,), dtype=np.float32)

        return events

    def flush(self) -> List[Dict[str, Any]]:
        """Force-finalize whatever is buffered (client-requested turn end)."""
        events: List[Dict[str, Any]] = []
        if len(self.utterance_buffer) >= int(0.25 * self.settings.sample_rate):
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
            context = self.confirmed_transcript[-100:] if self.confirmed_transcript else ""
            result = self.engine.transcribe_utterance(
                self.utterance_buffer, language=self.language, context=context
            )
            text = result["text"].strip()
            if text and text != "<unintelligible>":
                # Anti-hallucination guard: LLM-ASR can enter degenerate
                # repetition loops on noisy audio. Never let an identical final
                # chain more than twice, and keep looping garbage out of the
                # rolling context so it cannot feed itself.
                if text == self.last_final_text:
                    self.final_repeat_count += 1
                else:
                    self.final_repeat_count = 0
                self.last_final_text = text

                if self.final_repeat_count < 2:
                    self.confirmed_transcript = f"{self.confirmed_transcript} {text}".strip()
                    self.finals_sent += 1
                    events.append(
                        {
                            "type": "final",
                            "session_id": self.session_id,
                            "text": text,
                            "cumulative_text": self.confirmed_transcript,
                            "language": result["language"],
                            "duration_sec": result["duration_sec"],
                            "latency_ms": result["latency_ms"],
                        }
                    )
                else:
                    logger.warning(
                        f"[{self.session_id}] suppressed repeating final #{self.final_repeat_count}: {text[:60]}"
                    )

        self._reset_buffers()
        events.append({"type": "speech_end", "session_id": self.session_id})
        return events

    def _partial(self) -> Optional[Dict[str, Any]]:
        segment = self.utterance_buffer[-self.context_samples :]
        context = self.confirmed_transcript[-80:] if self.confirmed_transcript else ""
        result = self.engine.transcribe_window(
            segment,
            context=context,
            language=self.language,
            max_tokens=self.settings.max_stream_tokens,
        )
        text = result["text"].strip()
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
