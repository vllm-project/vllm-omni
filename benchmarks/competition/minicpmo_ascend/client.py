"""Streaming request helpers shared by the MiniCPM Ascend competition tools."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import mimetypes
import time
import wave
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

SYSTEM_PROMPT = (
    "You are MiniCPM-o, a helpful multimodal assistant that can understand "
    "images, audio and video, and respond in text and speech."
)


def media_url(value: str | Path, modality: str) -> str:
    raw = str(value)
    if raw.startswith(("http://", "https://", "data:")):
        return raw
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    mime = (
        mimetypes.guess_type(path.name)[0]
        or {
            "image": "image/png",
            "audio": "audio/wav",
            "video": "video/mp4",
        }[modality]
    )
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def build_payload(
    *,
    model: str,
    prompt: str,
    input_modality: str,
    media: str | Path | None,
    with_audio: bool,
    seed: int,
    thinker_max_tokens: int,
    talker_max_tokens: int,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if input_modality != "text":
        if media is None:
            raise ValueError(f"{input_modality} input requires a media path or URL")
        content.append(
            {
                "type": f"{input_modality}_url",
                f"{input_modality}_url": {"url": media_url(media, input_modality)},
            }
        )
    content.append({"type": "text", "text": prompt})
    modalities = ["text", "audio"] if with_audio else ["text"]
    return {
        "model": model,
        "stream": True,
        "modalities": modalities,
        "chat_template_kwargs": {"use_tts_template": with_audio},
        "sampling_params_list": [
            {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": thinker_max_tokens,
                "seed": seed,
                "detokenize": True,
                "repetition_penalty": 1.1,
            },
            {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": talker_max_tokens,
                "seed": seed,
                "detokenize": False,
            },
        ],
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ],
    }


class WavAccumulator:
    def __init__(self) -> None:
        self.channels: int | None = None
        self.sample_width: int | None = None
        self.sample_rate: int | None = None
        self.frames: list[bytes] = []
        self.chunk_sha256: list[str] = []
        self.boundary_jumps: list[int] = []
        self._last_sample: int | None = None

    def append(self, encoded: str) -> None:
        raw = base64.b64decode(encoded, validate=True)
        with wave.open(io.BytesIO(raw), "rb") as chunk:
            current = (chunk.getnchannels(), chunk.getsampwidth(), chunk.getframerate())
            expected = (self.channels, self.sample_width, self.sample_rate)
            if self.channels is None:
                self.channels, self.sample_width, self.sample_rate = current
            elif current != expected:
                raise ValueError(f"audio format changed from {expected} to {current}")
            frames = chunk.readframes(chunk.getnframes())
        if not frames:
            raise ValueError("empty audio chunk")
        if self.sample_width == 2:
            samples = array("h")
            samples.frombytes(frames)
            if samples.itemsize != 2:
                raise ValueError("host does not provide 16-bit signed samples")
            if self._last_sample is not None:
                self.boundary_jumps.append(abs(int(samples[0]) - self._last_sample))
            self._last_sample = int(samples[-1])
        self.frames.append(frames)
        self.chunk_sha256.append(hashlib.sha256(frames).hexdigest())

    def write(self, path: Path) -> None:
        if not self.frames or self.channels is None or self.sample_width is None or self.sample_rate is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as output:
            output.setnchannels(self.channels)
            output.setsampwidth(self.sample_width)
            output.setframerate(self.sample_rate)
            output.writeframes(b"".join(self.frames))

    def metadata(self) -> dict[str, Any]:
        pcm = b"".join(self.frames)
        return {
            "chunk_count": len(self.frames),
            "pcm_bytes": len(pcm),
            "sample_rate_hz": self.sample_rate,
            "channels": self.channels,
            "sample_width_bytes": self.sample_width,
            "pcm_sha256": hashlib.sha256(pcm).hexdigest() if pcm else None,
            "chunk_sha256": self.chunk_sha256,
            "adjacent_duplicate_chunks": sum(
                left == right for left, right in zip(self.chunk_sha256, self.chunk_sha256[1:], strict=False)
            ),
            "boundary_jump_abs_pcm16": self.boundary_jumps,
        }


async def run_stream_request(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    payload: dict[str, Any],
    request_name: str,
    input_modality: str,
    with_audio: bool,
    output_wav: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    record: dict[str, Any] = {
        "request_name": request_name,
        "input_modality": input_modality,
        "output_mode": "text_audio" if with_audio else "text",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "success": False,
        "complete": False,
        "http_status": None,
        "first_event_s": None,
        "first_text_s": None,
        "first_audio_s": None,
        "audio_chunk_arrival_s": [],
        "audio_chunk_intervals_s": [],
        "e2e_s": None,
        "text": "",
        "finish_reasons": [],
        "errors": [],
    }
    audio = WavAccumulator()
    done = False
    try:
        async with client.stream("POST", endpoint, json=payload) as response:
            record["http_status"] = response.status_code
            if response.status_code != 200:
                body = (await response.aread()).decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {response.status_code}: {body[:2000]}")
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    done = True
                    break
                now = time.perf_counter() - started
                if record["first_event_s"] is None:
                    record["first_event_s"] = now
                event = json.loads(data)
                modality = event.get("modality", "text")
                for choice in event.get("choices", []):
                    finish_reason = choice.get("finish_reason")
                    if finish_reason is not None:
                        record["finish_reasons"].append(finish_reason)
                    content = (choice.get("delta") or {}).get("content")
                    if not content:
                        continue
                    if modality == "audio":
                        if record["first_audio_s"] is None:
                            record["first_audio_s"] = now
                        record["audio_chunk_arrival_s"].append(now)
                        audio.append(content)
                    elif modality == "text":
                        if record["first_text_s"] is None:
                            record["first_text_s"] = now
                        record["text"] += content
    except Exception as exc:
        record["errors"].append(f"{type(exc).__name__}: {exc}")

    record["e2e_s"] = time.perf_counter() - started
    arrivals = record["audio_chunk_arrival_s"]
    record["audio_chunk_intervals_s"] = [right - left for left, right in zip(arrivals, arrivals[1:], strict=False)]
    record["audio"] = audio.metadata()
    record["complete"] = done
    if not record["text"].strip():
        record["errors"].append("empty text output")
    if with_audio:
        if not audio.frames:
            record["errors"].append("empty audio output")
        if audio.sample_rate != 24000:
            record["errors"].append(f"unexpected audio sample rate: {audio.sample_rate}")
        if audio.metadata()["adjacent_duplicate_chunks"]:
            record["errors"].append("adjacent audio chunks are byte-identical")
    if not done:
        record["errors"].append("stream ended without [DONE]")
    record["success"] = not record["errors"]
    if output_wav is not None and audio.frames:
        audio.write(output_wav)
        record["audio_artifact"] = str(output_wav)
    return record


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def metric_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record.get("success")]
    summary: dict[str, Any] = {
        "requests": len(records),
        "successful_requests": len(successful),
        "failed_requests": len(records) - len(successful),
    }
    for metric in ("first_event_s", "first_text_s", "first_audio_s", "e2e_s"):
        values = [float(record[metric]) for record in successful if record.get(metric) is not None]
        summary[metric] = {
            "count": len(values),
            "mean": sum(values) / len(values) if values else None,
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
        }
    intervals = [float(value) for record in successful for value in record.get("audio_chunk_intervals_s", [])]
    summary["audio_chunk_interval_s"] = {
        "count": len(intervals),
        "mean": sum(intervals) / len(intervals) if intervals else None,
        "p50": percentile(intervals, 0.50),
        "p95": percentile(intervals, 0.95),
        "p99": percentile(intervals, 0.99),
    }
    return summary
