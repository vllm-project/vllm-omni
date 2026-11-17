# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Local Full-Duplex-Bench v1.0 and v1.5 dataset discovery."""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import regex as re
import soundfile

FDB_VERSIONS = ("1.0", "1.5")
FDB_V10_SUBSET_TASKS = {
    "synthetic_pause_handling": "pause_handling",
    "candor_pause_handling": "pause_handling",
    "candor_turn_taking": "turn_taking",
    "synthetic_user_interruption": "user_interruption",
    "icc_backchannel": "backchannel",
}
FDB_V15_SUBSETS = (
    "user_interruption",
    "user_backchannel",
    "talking_to_other",
    "background_speech",
)
FDB_SUBSETS = {
    "1.0": tuple(FDB_V10_SUBSET_TASKS),
    "1.5": FDB_V15_SUBSETS,
}

_V10_ANNOTATIONS = {
    "pause_handling": "pause.json",
    "turn_taking": "turn_taking.json",
    "user_interruption": "interrupt.json",
}
_SAMPLE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class FullDuplexBenchInput:
    variant: str
    path: Path
    duration_s: float


@dataclass(frozen=True)
class FullDuplexBenchSample:
    version: str
    subset: str
    name: str
    inputs: tuple[FullDuplexBenchInput, ...]

    @property
    def id(self) -> str:
        return f"{self.subset}/{self.name}"


def _natural_key(name: str) -> tuple[int, int, str]:
    return (0, int(name), name) if name.isdigit() else (1, 0, name)


def _version_root(root: Path, version: str) -> Path:
    root = root.expanduser().resolve()
    subsets = FDB_SUBSETS[version]
    if any((root / subset).is_dir() for subset in subsets):
        return root
    nested = root / f"v{version}"
    if any((nested / subset).is_dir() for subset in subsets):
        return nested
    raise FileNotFoundError(
        f"Full-Duplex-Bench v{version} subsets were not found under {root} or {nested}; expected one of {subsets}"
    )


def _sample_names(root: Path, subsets: tuple[str, ...]) -> dict[str, list[str]]:
    names: dict[str, list[str]] = {}
    for subset in subsets:
        subset_root = root / subset
        if not subset_root.is_dir() or subset_root.is_symlink():
            names[subset] = []
            continue
        names[subset] = sorted(
            (
                entry.name
                for entry in subset_root.iterdir()
                if entry.is_dir() and not entry.is_symlink() and _SAMPLE_NAME.fullmatch(entry.name)
            ),
            key=_natural_key,
        )
    return names


def _select_ids(
    root: Path,
    subsets: tuple[str, ...],
    sample_ids: tuple[str, ...] | None,
    max_per_subset: int | None,
) -> list[str]:
    if sample_ids is not None and max_per_subset is not None:
        raise ValueError("--fdb-sample-ids and --fdb-max-per-subset are mutually exclusive")
    if max_per_subset is not None and max_per_subset <= 0:
        raise ValueError("--fdb-max-per-subset must be positive")
    names = _sample_names(root, subsets)
    available = [f"{subset}/{name}" for subset in subsets for name in names[subset]]
    if sample_ids is None:
        return [f"{subset}/{name}" for subset in subsets for name in names[subset][:max_per_subset]]
    duplicates = sorted(sample_id for sample_id, count in Counter(sample_ids).items() if count > 1)
    unknown = sorted(set(sample_ids) - set(available))
    if duplicates:
        raise ValueError(f"duplicate requested Full-Duplex-Bench sample IDs: {duplicates}")
    if unknown:
        raise ValueError(f"unknown requested Full-Duplex-Bench sample IDs: {unknown}")
    requested = set(sample_ids)
    return [sample_id for sample_id in available if sample_id in requested]


def _audio_duration(path: Path) -> float:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing regular audio file: {path.name}")
    try:
        info = soundfile.info(str(path))
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"unreadable audio file {path.name}: {exc}") from exc
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError(f"audio file has no frames: {path.name}")
    return info.frames / info.samplerate


def _finite_span(value: object) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    if not all(
        isinstance(boundary, int | float) and not isinstance(boundary, bool) and math.isfinite(boundary)
        for boundary in value
    ):
        return None
    start, end = float(value[0]), float(value[1])
    return (start, end) if 0 <= start <= end else None


def _load_json(path: Path) -> object:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing regular JSON file: {path.name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file {path.name}: {exc}") from exc


def _validate_v10(root: Path, subset: str, name: str) -> FullDuplexBenchSample:
    sample_root = root / subset / name
    audio = sample_root / "input.wav"
    duration = _audio_duration(audio)
    task = FDB_V10_SUBSET_TASKS[subset]
    annotation_name = _V10_ANNOTATIONS.get(task)
    if annotation_name is not None:
        document = _load_json(sample_root / annotation_name)
        if not isinstance(document, list) or not document:
            raise ValueError(f"{annotation_name} must contain a nonempty list")
        if task != "pause_handling" and len(document) != 1:
            raise ValueError(f"{annotation_name} must contain exactly one event")
        for index, annotation in enumerate(document):
            span = _finite_span(annotation.get("timestamp")) if isinstance(annotation, dict) else None
            if span is None or span[1] > duration:
                raise ValueError(f"invalid {annotation_name}[{index}] timestamp for {subset}/{name}")
        if task == "user_interruption":
            annotation = document[0]
            assert isinstance(annotation, dict)
            for key in ("context", "interrupt"):
                if not isinstance(annotation.get(key), str) or not str(annotation[key]).strip():
                    raise ValueError(f"{annotation_name} lacks nonempty {key!r} text")
    return FullDuplexBenchSample("1.0", subset, name, (FullDuplexBenchInput("input", audio, duration),))


def _validate_v15(root: Path, subset: str, name: str) -> FullDuplexBenchSample:
    sample_root = root / subset / name
    input_path = sample_root / "input.wav"
    clean_path = sample_root / "clean_input.wav"
    input_duration = _audio_duration(input_path)
    clean_duration = _audio_duration(clean_path)
    metadata = _load_json(sample_root / "metadata.json")
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain an object")
    span = _finite_span(metadata.get("timestamps"))
    if span is None or span[0] == span[1] or span[1] > input_duration:
        raise ValueError(f"invalid metadata.json timestamps for {subset}/{name}")
    return FullDuplexBenchSample(
        "1.5",
        subset,
        name,
        (
            FullDuplexBenchInput("input", input_path, input_duration),
            FullDuplexBenchInput("clean_input", clean_path, clean_duration),
        ),
    )


def discover_full_duplex_bench(
    dataset_root: str | Path,
    *,
    version: str,
    subsets: tuple[str, ...] | None = None,
    sample_ids: tuple[str, ...] | None = None,
    max_per_subset: int | None = None,
    source_limit: int | None = None,
    shuffle_seed: int | None = None,
) -> list[FullDuplexBenchSample]:
    """Discover selected source samples in stable subset/natural-ID order."""
    if version not in FDB_VERSIONS:
        raise ValueError(f"unsupported Full-Duplex-Bench version {version!r}; choose from {FDB_VERSIONS}")
    if source_limit is not None and source_limit <= 0:
        raise ValueError("Full-Duplex-Bench source_limit must be positive")
    available_subsets = FDB_SUBSETS[version]
    selected_subsets = subsets or available_subsets
    duplicates = sorted(subset for subset, count in Counter(selected_subsets).items() if count > 1)
    unknown = sorted(set(selected_subsets) - set(available_subsets))
    if duplicates:
        raise ValueError(f"duplicate Full-Duplex-Bench subsets: {duplicates}")
    if unknown:
        raise ValueError(f"unsupported Full-Duplex-Bench v{version} subsets: {unknown}")
    root = _version_root(Path(dataset_root), version)
    selected = _select_ids(root, selected_subsets, sample_ids, max_per_subset)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(selected)
    if source_limit is not None:
        selected = selected[:source_limit]
    validator = _validate_v10 if version == "1.0" else _validate_v15
    samples: list[FullDuplexBenchSample] = []
    for sample_id in selected:
        subset, name = sample_id.split("/", 1)
        try:
            samples.append(validator(root, subset, name))
        except ValueError as exc:
            raise ValueError(f"invalid Full-Duplex-Bench v{version} sample {sample_id}: {exc}") from exc
    return samples
