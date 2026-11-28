# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""ZONOS2 acceptance oracles shared by CPU and real-weight E2E tests."""

from __future__ import annotations

import base64
import io
import json
import wave
from pathlib import Path

import numpy as np


def pcm16_wav(samples: np.ndarray, sample_rate: int = 44100) -> bytes:
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
        return buffer.getvalue()


def data_uri(audio: bytes) -> str:
    return "data:audio/wav;base64," + base64.b64encode(audio).decode("ascii")


def decode_wav(audio: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(audio), "rb") as wav:
        assert wav.getnchannels() == 1, "Expected mono WAV"
        assert wav.getsampwidth() == 2, "Expected PCM16 WAV"
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
        return samples, wav.getframerate()


def check_audio(samples: np.ndarray, sample_rate: int, *, frames: int | None = None) -> dict:
    samples = np.asarray(samples).reshape(-1)
    assert sample_rate == 44100, f"Unexpected sample rate: {sample_rate}"
    assert samples.size > 0, "Empty audio"
    assert np.isfinite(samples).all(), "NaN/Inf in audio"
    duration = samples.size / sample_rate
    assert 0.1 <= duration <= 20, f"Implausible duration: {duration}"
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    assert rms > 1e-4, f"Silent audio: RMS={rms}"
    if frames is not None:
        assert samples.size == frames * 512, f"Truncated/duplicated audio: {samples.size} vs {frames * 512}"
    return {"samples": int(samples.size), "sample_rate": sample_rate, "duration_s": duration, "rms": rms}


def decode_sse(body: bytes) -> tuple[bytes, int]:
    chunks = []
    done = 0
    for event in body.decode("utf-8").split("\n\n"):
        fields = [line[6:] for line in event.splitlines() if line.startswith("data: ")]
        if not fields or fields == ["[DONE]"]:
            continue
        payload = json.loads("\n".join(fields))
        assert payload.get("type") != "error", payload
        if payload.get("type") == "speech.audio.delta":
            chunks.append(base64.b64decode(payload["audio"]))
        elif payload.get("type") == "speech.audio.done":
            done += 1
    assert done == 1, f"Expected one terminal SSE event, got {done}"
    assert len(chunks) > 1, "Streaming did not produce multiple audio deltas"
    return b"".join(chunks), len(chunks)


def read_trace(directory: Path) -> list[dict]:
    return [json.loads(line) for path in directory.glob("trace-*.jsonl") for line in path.read_text().splitlines()]


def check_lifecycle(rows: list[dict], label: str, *, allow_cap: bool = False) -> dict:
    finishes = [
        row
        for row in rows
        if row["kind"] == "finish" and (row["request"] == label or row["request"].startswith(label + "-"))
    ]
    assert len(finishes) == 1, (label, finishes)
    finish = finishes[0]
    key = finish["request"]
    if not allow_cap:
        assert finish["eos_frame"] > 0 and finish["countdown"] == 0, finish
        assert not finish["reached_cap"], finish
    decoded = [row for row in rows if row["kind"] == "decode" and row["request"] == key]
    target = finish["eos_frame"] if finish["eos_frame"] >= 0 else finish["frames"]
    assert sum(row["samples"] for row in decoded) == target * 512, (label, decoded, finish)
    assert any(row["kind"] == "talker_cleanup" and key in row["ids"] for row in rows), key
    assert any(row["kind"] == "dac_cleanup" and key in row["ids"] for row in rows), key
    assert all(row["finite"] for row in decoded)
    assert all(row["dtype"] == "torch.float32" and row["codec_loaded"] for row in decoded)
    return {**finish, "decoded_samples": target * 512, "decode_calls": len(decoded)}
