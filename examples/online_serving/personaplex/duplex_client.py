# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Headless client for the PersonaPlex duplex server (official Moshi protocol).

Streams a 24 kHz mono WAV to ``/api/chat`` exactly as the browser client does
(Opus over WebSocket, tags ``\\x00`` ready / ``\\x01`` audio / ``\\x02`` text),
collects the agent's reply audio + inner-monologue text, and writes a WAV.

The browser UI at ``http://<host>:<port>/`` is the primary demo; this script is
for headless testing / scripting.

    python duplex_client.py --url ws://localhost:8124/api/chat \
        --input user.wav --out reply.wav

Deps: ``websockets``, ``sphn``, ``soundfile``, ``numpy``.
"""

from __future__ import annotations

import argparse
import asyncio

import numpy as np
import soundfile as sf
import sphn

SR = 24000  # PersonaPlex / Mimi codec rate


async def converse(url: str, pcm: np.ndarray) -> tuple[np.ndarray, str]:
    import websockets

    reader = sphn.OpusStreamReader(SR)  # decode agent audio
    writer = sphn.OpusStreamWriter(SR)  # encode our mic audio
    text: list[str] = []
    ready = asyncio.Event()
    last = {"t": 0.0}

    async with websockets.connect(url, max_size=None) as ws:

        async def receive() -> None:
            loop = asyncio.get_event_loop()
            while True:
                try:
                    msg = await ws.recv()
                except Exception:
                    return
                if not isinstance(msg, (bytes, bytearray)) or not msg:
                    continue
                tag, payload = msg[0], msg[1:]
                if tag == 0:
                    ready.set()
                elif tag == 1:
                    reader.append_bytes(bytes(payload))
                    last["t"] = loop.time()
                elif tag == 2:
                    text.append(payload.decode("utf8", errors="replace"))

        recv_task = asyncio.create_task(receive())
        await asyncio.wait_for(ready.wait(), timeout=120)  # waits for system-prompt prefill

        frame = 1920  # 80 ms
        for i in range(0, len(pcm), frame):
            chunk = pcm[i : i + frame]
            if len(chunk) < frame:
                chunk = np.concatenate([chunk, np.zeros(frame - len(chunk), dtype=np.float32)])
            writer.append_pcm(chunk)
            data = writer.read_bytes()
            if data:
                await ws.send(b"\x01" + data)
            await asyncio.sleep(frame / SR)  # pace at real time

        loop = asyncio.get_event_loop()
        last["t"] = loop.time()
        while loop.time() - last["t"] < 2.5:  # drain until the agent stops talking
            await asyncio.sleep(0.1)
        recv_task.cancel()

    chunks: list[np.ndarray] = []
    while True:
        out = reader.read_pcm()
        if out is None or out.shape[-1] == 0:
            break
        chunks.append(np.asarray(out, dtype=np.float32))
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return audio, "".join(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8124/api/chat")
    ap.add_argument("--input", required=True, help="24 kHz mono WAV of the user turn")
    ap.add_argument("--out", default="personaplex_reply.wav")
    ap.add_argument("--tail", type=float, default=10.0, help="trailing silence (s) for the agent reply")
    ap.add_argument("--persona", default="")
    ap.add_argument("--voice", default="NATF2.pt")
    args = ap.parse_args()

    wav, sr = sf.read(args.input, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        raise SystemExit(f"input must be {SR} Hz mono; got {sr} Hz. Resample it first.")
    wav = np.concatenate([wav, np.zeros(int(args.tail * SR), dtype=np.float32)])

    url = f"{args.url}?text_prompt={args.persona}&voice_prompt={args.voice}"
    audio, text = asyncio.run(converse(url, wav))
    sf.write(args.out, audio if audio.size else np.zeros(SR, dtype=np.float32), SR)
    print(f"agent reply: {audio.size / SR:.2f}s -> {args.out}")
    print(f"inner-monologue: {text.strip()!r}")


if __name__ == "__main__":
    main()
