# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Text SSE -> punctuation chunks -> resident Talker -> ordered audio events."""

import asyncio
import json
import time
import unicodedata

import anyio
import pybase64 as base64


def speech_units(text):
    """Approximate spoken span: one CJK character or half a Latin word."""
    units, in_word = 0, False
    for char in text:
        if char.isalnum() and unicodedata.east_asian_width(char) in ("W", "F"):
            units += 1
            in_word = False
        elif char.isalnum():
            if not in_word:
                units += 2
            in_word = True
        elif char not in "'’":
            in_word = False
    return units


class SentenceBuffer:
    def __init__(self):
        self.pending = ""
        self.first = True

    def add(self, text, final=False):
        self.pending += text
        pieces = []
        start = 0
        for i, ch in enumerate(self.pending):
            n = i + 1 - start
            # Keep short clauses together; first playable chunk stays small.
            hard = ch in "。！？!?；;\n"
            soft = ch in "，,：:" and speech_units(self.pending[start : i + 1]) >= (8 if self.first else 24)
            # A period within 3.14 or an abbreviation should not split speech.
            period = ch == "." and i + 1 < len(self.pending) and self.pending[i + 1].isspace()
            if (hard and n >= 8) or soft or period or n >= 120:
                pieces.append(self.pending[start : i + 1])
                self.first = False
                start = i + 1
        self.pending = self.pending[start:]
        if final and self.pending:
            pieces.append(self.pending)
            self.pending = ""
        return pieces


async def stream_reply(client, backend, payload, audio_output, speech_language="Auto"):
    events: asyncio.Queue[dict[str, object] | None] = asyncio.Queue(maxsize=32)
    sentences: asyncio.Queue[str | None] = asyncio.Queue()  # Bounded by the 65536-character response limit below.
    started = time.monotonic()

    async def generate_text():
        splitter = SentenceBuffer()
        full = ""
        finish = None
        metadata = {}
        async with client.stream("POST", backend + "/v1/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw.strip() == "[DONE]":
                    break
                data = json.loads(raw)
                if "error" in data:
                    raise RuntimeError(str(data["error"]))
                for key in ("decision", "input_transcription", "usage"):
                    if data.get(key) is not None:
                        metadata[key] = data[key]
                        await events.put({"type": key, "value": data[key]})
                for choice in data.get("choices", []):
                    delta = choice.get("delta", {}).get("content") or ""
                    if delta:
                        full += delta
                        if len(full) > 65536:
                            raise RuntimeError("Speech response exceeds 65536 characters")
                        await events.put({"type": "text", "delta": delta})
                        if audio_output:
                            for text in splitter.add(delta):
                                await sentences.put(text)
                    finish = choice.get("finish_reason") or finish
        if finish is None:
            raise RuntimeError("Model stream ended unexpectedly; this turn was not committed")
        if audio_output:
            for text in splitter.add("", final=True):
                await sentences.put(text)
        await sentences.put(None)
        await events.put({"type": "text_done", "text": full, "finish_reason": finish})
        return {"text": full, "finish_reason": finish, **metadata}

    async def synthesize():
        index = 0
        sentence_count = 0
        first_audio = None
        while True:
            sentence = await sentences.get()
            if sentence is None:
                break
            if not sentence.strip():
                continue
            chunk_count = 0
            sentence_index = sentence_count
            sentence_count += 1
            async with client.stream(
                "POST",
                backend + "/v1/drvexo/speech/stream",
                json={
                    "text": sentence,
                    "language": speech_language,
                    "model": payload.get("model", "DRVEXO-speech-10B-General-NVFP4"),
                },
            ) as response:
                response.raise_for_status()
                terminal = False
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if chunk["type"] == "done":
                        terminal = True
                        continue
                    if terminal or chunk["type"] != "audio" or chunk["chunk_index"] != chunk_count:
                        raise RuntimeError("Invalid audio chunk order")
                    raw = base64.b64decode(chunk["audio_data"], validate=True)
                    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
                        raise RuntimeError("Invalid streamed WAV")
                    elapsed = time.monotonic() - started
                    if first_audio is None:
                        first_audio = elapsed
                    await events.put(
                        {
                            "type": "audio",
                            "index": index,
                            "text": sentence if chunk_count == 0 else "",
                            "audio_data": chunk["audio_data"],
                            "elapsed_seconds": round(elapsed, 3),
                            "sentence_index": sentence_index,
                            "chunk_index": chunk_count,
                            "sample_rate": chunk["sample_rate"],
                            "samples": chunk["samples"],
                        }
                    )
                    chunk_count += 1
                    index += 1
                if not terminal or chunk_count == 0:
                    raise RuntimeError("Incomplete native audio stream")
        return {
            "audio_segments": index,
            "first_audio_seconds": round(first_audio, 3) if first_audio is not None else None,
        }

    async def orchestrate():
        workers = [asyncio.create_task(generate_text()), asyncio.create_task(synthesize())]
        try:
            result, audio = await asyncio.gather(*workers)
            await events.put(
                {"type": "done", **result, **audio, "elapsed_seconds": round(time.monotonic() - started, 3)}
            )
        except Exception as error:
            await events.put({"type": "error", "detail": str(error)})
        finally:
            with anyio.CancelScope(shield=True):
                for worker in workers:
                    worker.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
        # A cancellation during an error-event put must skip the terminal put:
        # the consumer has disconnected and a full queue can no longer drain.
        await events.put(None)

    job = asyncio.create_task(orchestrate())
    try:
        while True:
            event = await events.get()
            if event is None:
                break
            yield json.dumps(event, ensure_ascii=False) + "\n"
    finally:
        with anyio.CancelScope(shield=True):
            job.cancel()
            await asyncio.gather(job, return_exceptions=True)
