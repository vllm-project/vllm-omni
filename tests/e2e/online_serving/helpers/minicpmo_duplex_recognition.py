# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pinned inputs and scene/lifecycle assertions for the #8637 accuracy fixture."""

from __future__ import annotations

import base64
import hashlib
import json
import tarfile
import wave
from collections import Counter
from pathlib import Path

import regex as re

ASSETS = Path(__file__).resolve().parents[3] / "assets/minicpmo_4_5/duplex_recognition"
CHINESE_NUMBERS = ("零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二")


def load_manifest() -> dict:
    return json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))


def unpack_inputs(destination: Path) -> tuple[Path, list[str]]:
    """Read only named regular members; never extract archive-controlled paths."""
    manifest = load_manifest()
    archive = ASSETS / "inputs.tar.xz"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["archive_sha256"]
    destination.mkdir(parents=True, exist_ok=True)
    frames = []
    wav_path = destination / "input_16k.wav"
    with tarfile.open(archive, "r:xz") as tar:
        assert {member.name for member in tar.getmembers()} == set(manifest["files"])
        for name, digest in sorted(manifest["files"].items()):
            member = tar.getmember(name)
            assert member.isfile(), name
            stream = tar.extractfile(member)
            assert stream is not None
            with stream:
                data = stream.read()
            assert hashlib.sha256(data).hexdigest() == digest, name
            if name == "input_16k.wav":
                wav_path.write_bytes(data)
            else:
                frames.append(base64.b64encode(data).decode("ascii"))
    assert len(frames) == 180
    with wave.open(str(wav_path)) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) == (16000, 1, 2, 2880000)
    return wav_path, frames


def covered_scenes(transcript: str, scenes: list[dict]) -> list[int]:
    """Match the reported color/shape/number coverage, not general QA accuracy."""
    sentences = re.split(r"[。！？.!?\n]", transcript)
    covered = []
    for scene in scenes:
        number = scene["number"]
        pattern = rf"数字\s*(?:{number}|{CHINESE_NUMBERS[number]})(?![0-9一二三四五六七八九十])"
        if any(
            scene["color"] in sentence and scene["shape"] in sentence and re.search(pattern, sentence)
            for sentence in sentences
        ):
            covered.append(scene["scene"])
    return covered


def evaluate_session(transcript: str, events: list[dict], scenes: list[dict]) -> dict:
    covered = covered_scenes(transcript, scenes)
    missing = [scene["scene"] for scene in scenes if scene["scene"] not in covered]
    created: list[str] = []
    done: list[str] = []
    errors = []
    for event in events:
        kind = event.get("type", "")
        if kind in {"error", "client.timeout", "client.error"}:
            errors.append(event)
        if kind in {"response.created", "response.done"}:
            response = event.get("response", {})
            response_id = response.get("id") or event.get("response_id")
            if not isinstance(response_id, str) or not response_id:
                errors.append({"missing_response_id": kind})
                continue
            (created if kind == "response.created" else done).append(response_id)
            if kind == "response.done" and response.get("status") != "completed":
                errors.append({"response_not_completed": response_id, "status": response.get("status")})
    counts = Counter(event.get("type") for event in events)
    if not created or Counter(created) != Counter(done) or len(created) != len(set(created)):
        errors.append({"response_id_mismatch": {"created": created, "done": done}})
    for kind in ("input_audio_buffer.committed", "session.closed", "response.output_audio.delta"):
        if not counts[kind]:
            errors.append({"missing_event": kind})
    return {
        "passed": not missing and not errors,
        "covered_scenes": covered,
        "missing_scenes": missing,
        "protocol_errors": errors,
        "event_counts": dict(counts),
        "transcript": transcript,
    }
