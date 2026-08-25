"""Orato ASR Server v2 - realtime streaming + batch transcription.

Protocol (WS /ws/transcribe?language=<lang>):
  client -> server : binary frames of raw PCM16 LE mono @16kHz
                     text frames {"action": "ping"|"flush"|"reset", ...}
  server -> client : JSON events
    connected     {session_id, model, language, sample_rate, device}
    speech_start  {session_id, rms}
    partial       {text, cumulative_text, language, latency_ms, session_id}
    final         {text, cumulative_text, language, duration_sec, latency_ms, session_id}
    speech_end    {session_id}
    error         {message, session_id}

Reliability: every GPU submission runs under a watchdog. If a decode exceeds
DECODE_TIMEOUT_SEC the CUDA worker is considered wedged; clients get an error
event and (by default) the process exits with code 70 so a supervisor can
restart it cleanly.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.audio_utils import load_audio, pcm16_to_float
from app.config import settings
from app.engine import Engine
from app.session import StreamingSession

STARTED_AT = time.time()


class GPUWedge(Exception):
    """A GPU decode exceeded the watchdog timeout."""


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing ASR engine...")
    app.state.gpu_pool = ThreadPoolExecutor(
        max_workers=settings.gpu_workers, thread_name_prefix="asr-gpu"
    )
    engine = Engine.get(settings)
    app.state.sessions: Dict[str, StreamingSession] = {}
    info = engine.model_info()
    logger.info(f"ASR ready: {info}")
    yield
    logger.info("Shutting down...")
    app.state.gpu_pool.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Orato ASR Server v2", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def run_gpu(fn, *args) -> Any:
    """Run a GPU-bound callable on the worker pool under the watchdog."""
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(app.state.gpu_pool, fn, *args)
    try:
        return await asyncio.wait_for(fut, timeout=settings.decode_timeout_sec)
    except asyncio.TimeoutError:
        logger.critical(
            f"GPU decode exceeded {settings.decode_timeout_sec}s - CUDA worker is wedged"
        )
        if settings.gpu_exit_on_wedge:
            loop.call_later(settings.wedge_exit_delay_sec, os._exit, 70)
        raise GPUWedge("gpu decode timed out")


# ---------------------------------------------------------------------------- #
# Health / introspection                                                       #
# ---------------------------------------------------------------------------- #


def _health_payload() -> Dict[str, Any]:
    engine = Engine.get(settings)
    sessions = getattr(app.state, "sessions", {})
    stats = [s.stats() for s in sessions.values()]
    payload = {
        "status": "healthy" if engine.is_ready else "loading",
        **engine.model_info(),
        "uptime_sec": round(time.time() - STARTED_AT, 1),
        "sessions_active": len(sessions),
        "total_audio_seconds": round(sum(s["audio_seconds"] for s in stats), 2),
        "total_finals": sum(s["finals_sent"] for s in stats),
    }
    pool = getattr(app.state, "gpu_pool", None)
    if pool is not None:
        payload["queue_depth"] = pool._work_queue.qsize()
    return payload


@app.get("/health")
async def health():
    return _health_payload()


@app.get("/api/v1/model-info")
async def model_info():
    return Engine.get(settings).model_info()


# ---------------------------------------------------------------------------- #
# Batch REST                                                                   #
# ---------------------------------------------------------------------------- #


@app.post("/api/v1/transcribe")
async def transcribe_file(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    context: Optional[str] = Form(""),
):
    engine = Engine.get(settings)
    if not engine.is_ready:
        raise HTTPException(status_code=503, detail="ASR model is not yet ready.")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio payload received.")

    try:
        result = await run_gpu(
            engine.transcribe_utterance,
            audio_bytes,
            language or settings.default_language,
            context or "",
        )
        return {"success": True, "filename": file.filename, **result}
    except GPUWedge:
        raise HTTPException(status_code=503, detail="GPU wedged; service restarting.")
    except Exception as e:
        logger.exception(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------- #
# Realtime WebSocket                                                           #
# ---------------------------------------------------------------------------- #


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket, language: Optional[str] = None):
    await websocket.accept()
    session_id = uuid.uuid4().hex[:8]
    engine = Engine.get(settings)

    if not engine.is_ready:
        await websocket.send_json({"type": "error", "message": "model still loading"})
        await websocket.close()
        return

    session = StreamingSession(session_id, engine, settings, language=language)
    app.state.sessions[session_id] = session

    await websocket.send_json(
        {
            "type": "connected",
            "session_id": session_id,
            "model": settings.model_name,
            "language": session.language,
            "sample_rate": settings.sample_rate,
            "device": settings.device_resolved,
        }
    )
    logger.info(f"[ws:{session_id}] connected language={session.language}")

    audio_queue: asyncio.Queue = asyncio.Queue(maxsize=settings.ws_queue_size)

    async def sender_worker():
        while True:
            pcm_chunk = await audio_queue.get()
            try:
                events: List[Dict[str, Any]] = await run_gpu(session.process_chunk, pcm_chunk)
                for event in events:
                    await _send_event(websocket, event)
            except GPUWedge:
                await _send_event(
                    websocket,
                    {
                        "type": "error",
                        "message": "gpu decode timed out; service is restarting",
                        "session_id": session_id,
                    },
                )
                try:
                    await websocket.close(code=1013, reason="gpu wedged")
                except Exception:
                    pass
                return
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.exception(f"[ws:{session_id}] processing error: {e}")

    sender_task = asyncio.create_task(sender_worker())
    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"]:
                pcm_chunk = pcm16_to_float(message["bytes"])
                if audio_queue.full():
                    try:
                        audio_queue.get_nowait()  # drop oldest under congestion
                    except asyncio.QueueEmpty:
                        pass
                    logger.warning(f"[ws:{session_id}] queue full - dropped oldest chunk")
                audio_queue.put_nowait(pcm_chunk)

            elif "text" in message and message["text"]:
                await _handle_control(websocket, message["text"], session, audio_queue, sender_task)

    except WebSocketDisconnect as e:
        logger.info(f"[ws:{session_id}] disconnected code={e.code}")
    except Exception as e:
        logger.exception(f"[ws:{session_id}] unhandled error: {e}")
    finally:
        sender_task.cancel()
        app.state.sessions.pop(session_id, None)
        logger.info(f"[ws:{session_id}] closed | {session.stats()}")


async def _send_event(websocket: WebSocket, event: Dict[str, Any]) -> None:
    if websocket.client_state.name != "CONNECTED":
        return
    etype = event.get("type")
    if etype == "partial":
        logger.debug(f"[ws] <- partial '{event.get('text', '')}' ({event.get('latency_ms')}ms)")
    elif etype == "final":
        logger.info(f"[ws] <- final '{event.get('text', '')}' ({event.get('latency_ms')}ms)")
    await websocket.send_json(event)


async def _handle_control(
    websocket: WebSocket,
    raw: str,
    session: StreamingSession,
    audio_queue: asyncio.Queue,
    sender_task: asyncio.Task,
) -> None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return
    action = payload.get("action")

    if action == "ping":
        await _send_event(websocket, {"type": "pong"})
    elif action == "flush":
        events = await run_gpu(session.flush)
        for event in events:
            await _send_event(websocket, event)
    elif action == "reset":
        session.reset(language=payload.get("language"))
        await _send_event(
            websocket, {"type": "reset_ack", "session_id": session.session_id}
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.server:app",
        host=settings.host,
        port=settings.port,
        ws_ping_interval=20,
        ws_ping_timeout=60,
    )
