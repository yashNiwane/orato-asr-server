"""Stream a WAV file to the ASR websocket and verify finals arrive.

Usage: python scripts/ws_smoke_test.py <ws_url> <wav_path> [--language Hindi]
Exits 0 iff at least one final transcript is received within the timeout.
"""

import argparse
import asyncio
import json
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import soundfile as sf

import websockets


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("wav")
    parser.add_argument("--language", default="Hindi")
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()

    wav, sr = sf.read(args.wav, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != 16000:
        import soxr

        wav = soxr.resample(wav, sr, 16000)

    pcm16 = (np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes()

    url = args.url.strip()
    if url.startswith("https://"):
        url = "wss://" + url[len("https://"):]
    elif url.startswith("http://"):
        url = "ws://" + url[len("http://"):]
    if "/ws/transcribe" not in url:
        url = url.rstrip("/") + "/ws/transcribe"
    if "language=" not in url:
        url += ("&" if "?" in url else "?") + f"language={args.language}"

    finals, partials, got_final = 0, 0, asyncio.Event()

    async with websockets.connect(url, max_size=None, open_timeout=15) as ws:
        print("ack:", (await ws.recv())[:110])

        async def sender():
            chunk_bytes = 3200  # 100ms @16kHz s16le
            for i in range(0, len(pcm16), chunk_bytes):
                await ws.send(pcm16[i : i + chunk_bytes])
                await asyncio.sleep(0.08)
            await ws.send(json.dumps({"action": "flush"}))

        async def receiver():
            try:
                while True:
                    ev = json.loads(await ws.recv())
                    etype = ev.get("type", "")
                    if etype == "partial":
                        partials += 1
                        print(f"  [partial] {str(ev.get('text', ''))[:80]}")
                    elif etype == "final":
                        finals += 1
                        print(f"  [FINAL] {ev.get('text')} ({ev.get('latency_ms')}ms)")
                        got_final.set()
                    elif etype in ("speech_start", "speech_end", "error"):
                        print(f"  [{etype}] {ev.get('message', '')}")
            except (websockets.ConnectionClosed, asyncio.TimeoutError):
                pass

        rt = asyncio.create_task(receiver())
        st = asyncio.create_task(sender())
        await st
        try:
            await asyncio.wait_for(got_final.wait(), timeout=args.timeout)
        except asyncio.TimeoutError:
            pass
        rt.cancel()

    print(f"summary: {partials} partials, {finals} finals")
    return 0 if finals > 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
