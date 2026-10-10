# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Dataset loading for the SocialOmni benchmark."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

PREFIX_ENCODING = {
    "video_codec": "libx264",
    "preset": "fast",
    "crf": "18",
    "audio_codec": "aac",
    "audio_bitrate": "192k",
}

SOCIALOMNI_DATASET_ID = "alexisty/SocialOmni"
SOCIALOMNI_DATASET_REVISION = "3b76009b45090eaa54007454c93a831f3cc8e1e6"
SOCIALOMNI_SUBSET_SIZE = 200
SOCIALOMNI_METADATA_SHA256 = {
    "level1": "051f2e7ca0618de6e78843c64a3800606904be819f47c1e5c771f1c6a49faa23",
    "level2": "87137aa2270f65e9c9e124e54dc5b73d9fe9c4bf9317219a4efe88feb036c058",
}


@dataclass(frozen=True)
class SocialOmniLevel1Sample:
    sample_id: str
    video_path: str
    question: str
    options: tuple[str, str, str, str]
    gold_answer: str
    visibility: str


@dataclass(frozen=True)
class SocialOmniLevel2Sample:
    sample_id: str
    video_path: str
    timestamp_s: float
    target_participant: str
    question_when: str
    question_how: str
    gold_when: str
    reference_response: str
    reference_context: str


def _level_dir(root: Path, level: str, metadata: str) -> Path:
    candidates = [root / "data" / level, root / level]
    if root.name == level:
        candidates.insert(0, root)
    for candidate in candidates:
        if (candidate / metadata).is_file():
            resolved = candidate.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f"SocialOmni {level} directory escapes dataset root")
            return resolved
    raise FileNotFoundError(f"SocialOmni metadata not found: {candidates[0] / metadata}")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid SocialOmni metadata {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_socialomni_dataset(dataset_root: str | Path, levels: Sequence[str]) -> dict[str, Any]:
    """Identify local metadata against the pinned public dataset revision."""
    root = Path(dataset_root).expanduser().resolve()
    metadata_names = {
        "level1": ("level_1", "dataset.json"),
        "level2": ("level_2", "annotations.json"),
    }
    level_dirs = {level: _level_dir(root, *metadata_names[level]) for level in levels}
    observed = {level: _sha256(level_dirs[level] / metadata_names[level][1]) for level in levels}
    metadata_matches_expected_revision = all(observed[level] == SOCIALOMNI_METADATA_SHA256[level] for level in levels)
    return {
        "expected_huggingface_revision": SOCIALOMNI_DATASET_REVISION,
        "verification_scope": "metadata_only",
        "metadata_sha256": observed,
        "metadata_matches_expected_revision": metadata_matches_expected_revision,
    }


def _object(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{description} must be a JSON object")
    return value


def _text(row: dict[str, Any], key: str, description: str) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise ValueError(f"{description} has no non-empty {key!r}")
    return value


def _video_path(level_dir: Path, raw: str, level: str, description: str) -> str:
    if "\\" in raw or "\x00" in raw:
        raise ValueError(f"{description} uses an unsafe media path: {raw!r}")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{description} uses an unsafe media path: {raw!r}")
    parts = list(relative.parts)
    if parts[:2] == ["data", level]:
        parts = parts[2:]
    elif parts[:1] == [level]:
        parts = parts[1:]
    elif parts[:1] in (["data"], ["level_1"], ["level_2"]):
        raise ValueError(f"{description} references the wrong dataset level")
    if parts[:1] == ["videos"]:
        parts = parts[1:]
    if not parts:
        raise ValueError(f"{description} uses an empty media path")
    videos = (level_dir / "videos").resolve()
    path = videos.joinpath(*parts).resolve()
    if not videos.is_relative_to(level_dir) or not path.is_relative_to(videos):
        raise ValueError(f"{description} media path escapes videos/")
    return str(path)


T = TypeVar("T", SocialOmniLevel1Sample, SocialOmniLevel2Sample)


def _mini(samples: Sequence[T], groups: tuple[str, str], group: Callable[[T], str]) -> list[T]:
    selected: dict[str, T] = {}
    for sample in samples:
        selected.setdefault(group(sample), sample)
        if all(name in selected for name in groups):
            return [selected[name] for name in groups]
    raise ValueError(f"SocialOmni mini set is missing groups: {set(groups) - set(selected)}")


def _limit(samples: list[T], max_samples: int | None) -> list[T]:
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative")
    selected = samples if max_samples is None else samples[:max_samples]
    for sample in selected:
        if not Path(sample.video_path).is_file():
            raise FileNotFoundError(f"SocialOmni sample {sample.sample_id} video is missing: {sample.video_path}")
    return selected


def load_socialomni_level1_samples(
    dataset_root: str | Path,
    *,
    max_samples: int | None = None,
    mini: bool = False,
) -> list[SocialOmniLevel1Sample]:
    """Load speaker-attribution items while preserving nested media paths."""
    root = Path(dataset_root).expanduser().resolve()
    level_dir = _level_dir(root, "level_1", "dataset.json")
    payload = _read_json(level_dir / "dataset.json")
    if not isinstance(payload, list):
        raise TypeError("SocialOmni Level 1 dataset.json must contain an array")
    samples: list[SocialOmniLevel1Sample] = []
    seen: set[str] = set()
    for index, raw_row in enumerate(payload):
        description = f"SocialOmni Level 1 row {index}"
        row = _object(raw_row, description)
        sample_id = _text(row, "id", description)
        if sample_id in seen:
            raise ValueError(f"Duplicate SocialOmni Level 1 sample id: {sample_id}")
        seen.add(sample_id)
        raw_options = row.get("options")
        if not isinstance(raw_options, list) or len(raw_options) != 4:
            raise ValueError(f"{description} must contain exactly four options")
        options = tuple(
            re.sub(r"^[A-D][.)]\s*", "", str(option).strip(), flags=re.IGNORECASE) for option in raw_options
        )
        answer = _text(row, "correct_answer", description).upper()
        if answer not in {"A", "B", "C", "D"}:
            raise ValueError(f"{description} has invalid correct_answer {answer!r}")
        metadata = _object(row.get("metadata"), f"{description} metadata")
        consistency = _text(metadata, "consistency", description).lower()
        visibility = {
            "consistent": "speaker_visible",
            "inconsistent": "visibility_mismatch",
        }.get(consistency)
        if visibility is None:
            raise ValueError(f"{description} has invalid consistency {consistency!r}")
        samples.append(
            SocialOmniLevel1Sample(
                sample_id=sample_id,
                video_path=_video_path(
                    level_dir,
                    _text(row, "video_path", description),
                    "level_1",
                    description,
                ),
                question=_text(row, "question", description),
                options=options,  # type: ignore[arg-type]
                gold_answer=answer,
                visibility=visibility,
            )
        )
    if mini:
        samples = _mini(
            samples,
            ("speaker_visible", "visibility_mismatch"),
            lambda sample: sample.visibility,
        )
    return _limit(samples, max_samples)


def parse_socialomni_timestamp(value: Any) -> float:
    """Parse seconds, MM:SS, or the source MM:SS:centiseconds notation."""
    if isinstance(value, bool):
        raise TypeError(f"Invalid SocialOmni timestamp type: {value!r}")
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = str(value or "").strip()
        if re.fullmatch(r"\d+(?:\.\d+)?", text):
            seconds = float(text)
        elif re.fullmatch(r"\d+:\d+(?:\.\d+)?", text):
            minutes, tail = text.split(":")
            seconds = int(minutes) * 60 + float(tail)
        elif re.fullmatch(r"\d+:\d{2}:\d{2}", text):
            minutes, tail, centiseconds = text.split(":")
            seconds = int(minutes) * 60 + int(tail) + int(centiseconds) / 100
        else:
            raise ValueError(f"Invalid SocialOmni timestamp: {value!r}")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"SocialOmni timestamp must be finite and positive: {value!r}")
    return seconds


def _gold_when(question: dict[str, Any], description: str) -> str:
    answer = _text(question, "correct_answer", description).upper()
    if answer in {"YES", "NO"}:
        return answer
    if answer not in {"A", "B"}:
        raise ValueError(f"{description} has invalid correct_answer {answer!r}")
    value = _text(question, f"option_{answer}", description).upper()
    if value not in {"YES", "NO"}:
        raise ValueError(f"{description} maps {answer} to invalid option {value!r}")
    return value


def _participant(question_how: str, question_when: str, description: str) -> str:
    match = re.fullmatch(r"What should (.+?) say\?", question_how, re.IGNORECASE)
    if not match:
        match = re.match(r"(?:Should|Did) (.+?) speak(?:\s|\?)", question_when, re.IGNORECASE)
    if not match:
        raise ValueError(f"{description} does not identify the target participant")
    return match.group(1).strip()


def load_socialomni_level2_samples(
    dataset_root: str | Path,
    *,
    max_samples: int | None = None,
    mini: bool = False,
) -> list[SocialOmniLevel2Sample]:
    """Load all Level 2 items in source order."""
    root = Path(dataset_root).expanduser().resolve()
    level_dir = _level_dir(root, "level_2", "annotations.json")
    payload = _read_json(level_dir / "annotations.json")
    if isinstance(payload, dict):
        declared = payload.get("total_samples")
        payload = payload.get("data")
        if isinstance(payload, list) and declared is not None and declared != len(payload):
            raise ValueError("SocialOmni Level 2 total_samples does not match data length")
    if not isinstance(payload, list):
        raise TypeError("SocialOmni Level 2 annotations must contain an array")
    samples: list[SocialOmniLevel2Sample] = []
    seen: set[str] = set()
    for index, raw_row in enumerate(payload):
        description = f"SocialOmni Level 2 row {index}"
        row = _object(raw_row, description)
        sample_id = _text(row, "video_id", description)
        if sample_id in seen:
            raise ValueError(f"Duplicate SocialOmni Level 2 sample id: {sample_id}")
        seen.add(sample_id)
        when = _object(row.get("question_1"), f"{description} question_1")
        how = _object(row.get("question_2"), f"{description} question_2")
        question_when = _text(when, "question", f"{description} question_1")
        question_how = _text(how, "question", f"{description} question_2")
        gold_when = _gold_when(when, f"{description} question_1")
        reference = str(how.get("answer") or "").strip()
        if gold_when == "YES" and not reference:
            raise ValueError(f"{description} has no reference response")
        samples.append(
            SocialOmniLevel2Sample(
                sample_id=sample_id,
                video_path=_video_path(
                    level_dir,
                    _text(row, "video_file", description),
                    "level_2",
                    description,
                ),
                timestamp_s=parse_socialomni_timestamp(_text(when, "timestamp", f"{description} question_1")),
                target_participant=_participant(question_how, question_when, description),
                question_when=question_when,
                question_how=question_how,
                gold_when=gold_when,
                reference_response=reference,
                reference_context=_text(row, "full_asr", description),
            )
        )
    if mini:
        samples = _mini(samples, ("YES", "NO"), lambda sample: sample.gold_when)
    return _limit(samples, max_samples)


def resolve_ffmpeg_executable() -> str | None:
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    bundled = Path(imageio_ffmpeg.get_ffmpeg_exe())
    return str(bundled) if bundled.is_file() and os.access(bundled, os.X_OK) else None


def build_ffmpeg_prefix_command(ffmpeg: str, source: Path, timestamp_s: float, output: Path) -> list[str]:
    return [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-t",
        f"{timestamp_s:.6f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        PREFIX_ENCODING["video_codec"],
        "-preset",
        PREFIX_ENCODING["preset"],
        "-crf",
        PREFIX_ENCODING["crf"],
        "-c:a",
        PREFIX_ENCODING["audio_codec"],
        "-b:a",
        PREFIX_ENCODING["audio_bitrate"],
        "-movflags",
        "+faststart",
        "-y",
        str(output),
    ]


async def create_video_prefix(
    input_path: str | Path,
    timestamp_s: float,
    cache_dir: str | Path,
    *,
    timeout: float = 300.0,
) -> Path:
    """Re-encode video and audio up to the query time into an atomic cache entry."""
    if not math.isfinite(timestamp_s) or timestamp_s <= 0:
        raise ValueError("timestamp_s must be finite and positive")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    source = Path(input_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    ffmpeg = resolve_ffmpeg_executable()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for SocialOmni Level 2")
    key = hashlib.sha256(
        json.dumps(
            {
                "source_sha256": await asyncio.to_thread(_sha256, source),
                "timestamp_s": f"{timestamp_s:.6f}",
                "encoding": PREFIX_ENCODING,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    cache = Path(cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / f"{key}.mp4"
    if output.is_file() and output.stat().st_size:
        return output
    temporary = cache / f".{key}.{uuid.uuid4().hex}.tmp.mp4"
    process = await asyncio.create_subprocess_exec(
        *build_ffmpeg_prefix_command(ffmpeg, source, timestamp_s, temporary),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        if process.returncode or not temporary.is_file() or not temporary.stat().st_size:
            raise RuntimeError(
                f"ffmpeg prefix generation failed for {source}: {stderr.decode(errors='replace')[:2000]}"
            )
        temporary.replace(output)
        return output
    except asyncio.TimeoutError as exc:
        raise RuntimeError(f"ffmpeg prefix generation timed out after {timeout}s for {source}") from exc
    finally:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.communicate()
        temporary.unlink(missing_ok=True)
