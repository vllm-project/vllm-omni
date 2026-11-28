# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Portions derived from https://github.com/step-out/Duplex-MPE
# (Copyright (c) 2026 MPeval authors, MIT License; see benchmarks/duplex_mpe/LICENSE).
"""Dataset loading, revision pinning, and media validation for Duplex-MPE."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

import soundfile
from huggingface_hub import snapshot_download
from pydantic import BaseModel, ConfigDict, JsonValue, model_validator

logger = logging.getLogger(__name__)

DUPLEX_MPE_DATASET_ID = "ChengqianMa/Duplex-MPE"
DUPLEX_MPE_DATASET_REVISION = "6d5f59fab1237c14ee8229ea36f111b7423b0fe6"
DUPLEX_MPE_PAPER_PAIR_COUNT = 2000
DUPLEX_MPE_MINI_PAIR_COUNT = 20
DUPLEX_MPE_DUTY_PREAMBLE_RELATIVE_PATH = "audio/aria_duty_preamble.wav"
DUPLEX_MPE_METADATA_SHA256: dict[str, str] = {
    "dataset.json": "b3d55f9c194539e6352edbdb76efc68a8815658bbb6289d97ca228d0e17e231b",
    "manifests/explicit.json": "4db23f171e1eb967f67552036cbe2909efc65d605dbe3778e9d7ba791f689007",
    "manifests/implicit.json": "19220dedd3f4f72f9bf0622cd176ce3bb63bd6f2b6bb958435ec192cd99a7128",
    "audio_inventory.json": "f64739344f09fa76468990c3c8982f6a16fccddb0aaa999693233f9896636b17",
    "audio/aria_duty_preamble.wav": "256a3381a7137078e76085dbe6564f5a8a32bddbfc415c1fccd328261b0097f6",
}

AddressingCondition = Literal["explicit", "implicit"]
AddressingSelection = Literal["all", "explicit", "implicit"]
TurnEventType = Literal["T", "N1", "N2", "N3", "N4_Q", "N4_R"]
ADDRESSING_CONDITIONS: tuple[AddressingCondition, AddressingCondition] = (
    "explicit",
    "implicit",
)
SAFE_SCENARIO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")


def validate_safe_identifier(identifier: str, field_label: str = "scenario_id") -> str:
    if not isinstance(identifier, str) or ".." in identifier or SAFE_SCENARIO_ID_RE.fullmatch(identifier) is None:
        raise ValueError(f"Unsafe {field_label} value")
    return identifier


class DuplexMpeTurnRecord(BaseModel):
    model_config = ConfigDict(extra="allow", hide_input_in_errors=True)

    turn_index: int
    event_type: TurnEventType
    speaker: str
    text: str
    audio_path: str


class DuplexMpeScenarioRecord(BaseModel):
    model_config = ConfigDict(extra="allow", hide_input_in_errors=True)

    scenario_id: str
    pair_id: str
    addressing: AddressingCondition
    trigger_index: int
    supporting_indices: list[int]
    gold_answer: str
    turns: list[DuplexMpeTurnRecord]

    @model_validator(mode="after")
    def validate_identifiers(self) -> DuplexMpeScenarioRecord:
        validate_safe_identifier(self.scenario_id, "scenario_id")
        validate_safe_identifier(self.pair_id, "pair_id")
        return self


class AudioInventoryEntry(BaseModel):
    model_config = ConfigDict(extra="allow", hide_input_in_errors=True)

    path: str
    sha256: str
    bytes: int
    sample_rate: int
    frames: int
    channels: int
    duration_s: float


@dataclass(frozen=True, kw_only=True)
class DuplexMpeDatasetIdentity:
    expected_huggingface_revision: str
    verification_scope: str
    metadata_sha256: dict[str, str]
    metadata_matches_expected_revision: bool


def compute_file_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        while chunk := file_handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_file(file_path: Path) -> JsonValue:
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file {file_path} at line {exc.lineno} column {exc.colno}") from None


def write_json_atomic(
    file_path: Path,
    payload: JsonValue | Mapping[str, object] | Sequence[object],
) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = file_path.with_name(f".{file_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(file_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def resolve_dataset_audio_path(dataset_root: Path, relative_audio_path: str) -> Path:
    if (
        not relative_audio_path
        or "\\" in relative_audio_path
        or "\x00" in relative_audio_path
        or any(segment in {"", ".", ".."} for segment in relative_audio_path.split("/"))
    ):
        raise ValueError("Unsafe audio path in dataset")
    posix_path = PurePosixPath(relative_audio_path)
    if posix_path.is_absolute():
        raise ValueError("Audio path must stay relative to dataset root")
    resolved_root = dataset_root.expanduser().resolve()
    resolved_audio = resolved_root.joinpath(*posix_path.parts).resolve()
    if not resolved_audio.is_relative_to(resolved_root):
        raise ValueError("Audio path escapes dataset root")
    return resolved_audio


def resolve_retained_scenario_audio_path(
    audio_directory: Path,
    scenario_id: str,
    *,
    must_exist: bool = True,
) -> Path:
    validate_safe_identifier(scenario_id, "scenario_id")
    resolved_root = audio_directory.expanduser().resolve()
    wav_candidate = resolved_root / f"{scenario_id}.wav"
    candidate = wav_candidate
    if must_exist and not wav_candidate.is_symlink() and not wav_candidate.exists():
        flac_candidate = resolved_root / f"{scenario_id}.flac"
        if flac_candidate.is_symlink() or flac_candidate.exists():
            candidate = flac_candidate
    if candidate.is_symlink():
        raise ValueError(f"Symlink retained audio is not permitted for {scenario_id}: {candidate}")
    resolved_candidate = candidate.resolve()
    if not resolved_candidate.is_relative_to(resolved_root):
        raise ValueError(f"Retained audio path escapes {audio_directory}")
    if must_exist and not resolved_candidate.is_file():
        raise FileNotFoundError(f"Retained assistant audio missing for {scenario_id} in {audio_directory}")
    return resolved_candidate


def inspect_duplex_mpe_dataset(dataset_root: Path) -> DuplexMpeDatasetIdentity:
    """Compare local Duplex-MPE metadata digests against the pinned dataset revision."""
    resolved_root = dataset_root.expanduser().resolve()
    observed_sha256: dict[str, str] = {}
    for relative_name in DUPLEX_MPE_METADATA_SHA256:
        target_path = resolve_dataset_audio_path(resolved_root, relative_name)
        if not target_path.is_file():
            raise FileNotFoundError(f"Duplex-MPE metadata file missing: {target_path}")
        observed_sha256[relative_name] = compute_file_sha256(target_path)
    matches_expected = all(
        observed_sha256[name] == expected_digest for name, expected_digest in DUPLEX_MPE_METADATA_SHA256.items()
    )
    return DuplexMpeDatasetIdentity(
        expected_huggingface_revision=DUPLEX_MPE_DATASET_REVISION,
        verification_scope="metadata_and_preamble",
        metadata_sha256=observed_sha256,
        metadata_matches_expected_revision=matches_expected,
    )


def load_condition_manifest(dataset_root: Path, condition: AddressingCondition) -> list[DuplexMpeScenarioRecord]:
    manifest_path = resolve_dataset_audio_path(dataset_root, f"manifests/{condition}.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Duplex-MPE manifest missing: {manifest_path}")
    raw_manifest = read_json_file(manifest_path)
    if not isinstance(raw_manifest, list):
        raise ValueError(f"Manifest {manifest_path} must be a JSON array")
    scenarios: list[DuplexMpeScenarioRecord] = []
    for raw_entry in raw_manifest:
        scenario = DuplexMpeScenarioRecord.model_validate(raw_entry)
        if scenario.addressing != condition:
            raise ValueError(f"Manifest addressing mismatch in {manifest_path}: {scenario.scenario_id}")
        target_indices = [index for index, turn in enumerate(scenario.turns) if turn.event_type == "T"]
        if target_indices != [scenario.trigger_index]:
            raise ValueError(f"Expected exactly one T turn at trigger_index in {scenario.scenario_id}")
        if not scenario.gold_answer.strip():
            raise ValueError(f"Empty gold_answer in {scenario.scenario_id}")
        if any(
            support_index < 0 or support_index >= scenario.trigger_index
            for support_index in scenario.supporting_indices
        ):
            raise ValueError(f"Non-causal supporting_indices in {scenario.scenario_id}")
        for index, turn in enumerate(scenario.turns):
            if turn.turn_index != index:
                raise ValueError(f"Non-contiguous turn_index at {index} in {scenario.scenario_id}")
            if turn.event_type == "N4_Q" and (
                index + 1 == len(scenario.turns) or scenario.turns[index + 1].event_type != "N4_R"
            ):
                raise ValueError(f"Unpaired N4_Q turn in {scenario.scenario_id}")
            if turn.event_type == "N4_R" and (index == 0 or scenario.turns[index - 1].event_type != "N4_Q"):
                raise ValueError(f"Unpaired N4_R turn in {scenario.scenario_id}")
        scenarios.append(scenario)
    return scenarios


def load_duplex_mpe_scenarios(
    dataset_root: Path,
    *,
    addressing: AddressingSelection = "all",
    mini: bool = False,
    max_scenarios: int | None = None,
) -> list[DuplexMpeScenarioRecord]:
    """Load validated Duplex-MPE scenario manifests in deterministic order."""
    if max_scenarios is not None and max_scenarios < 1:
        raise ValueError("max_scenarios must be >= 1 when provided")
    selected_conditions: tuple[AddressingCondition, ...] = (
        ADDRESSING_CONDITIONS if addressing == "all" else (addressing,)
    )
    by_condition: dict[AddressingCondition, list[DuplexMpeScenarioRecord]] = {
        condition: load_condition_manifest(dataset_root, condition) for condition in selected_conditions
    }
    if mini:
        first_source = by_condition[selected_conditions[0]]
        sorted_first = sorted(first_source, key=lambda row: row.pair_id)[:DUPLEX_MPE_MINI_PAIR_COUNT]
        allowed_pair_ids = {scenario.pair_id for scenario in sorted_first}
        scenarios = [
            scenario
            for condition in selected_conditions
            for scenario in by_condition[condition]
            if scenario.pair_id in allowed_pair_ids
        ]
    else:
        scenarios = [scenario for condition in selected_conditions for scenario in by_condition[condition]]
    seen_ids = {scenario.scenario_id for scenario in scenarios}
    if len(seen_ids) != len(scenarios):
        raise ValueError("Duplicate scenario_id encountered in Duplex-MPE manifests")
    if max_scenarios is not None:
        return scenarios[:max_scenarios]
    return scenarios


def validate_duplex_mpe_dataset(
    dataset_root: Path,
    *,
    mini: bool = False,
    verify_audio_checksums: bool = False,
) -> dict[str, JsonValue]:
    """Verify paired scenario structure, audio inventory, and referenced WAV files."""
    resolved_root = dataset_root.expanduser().resolve()
    scenarios = load_duplex_mpe_scenarios(resolved_root, addressing="all", mini=mini)
    inventory_path = resolve_dataset_audio_path(resolved_root, "audio_inventory.json")
    raw_inventory = read_json_file(inventory_path)
    if not isinstance(raw_inventory, list):
        raise ValueError("audio_inventory.json must contain a JSON array")
    inventory_entries = [AudioInventoryEntry.model_validate(entry) for entry in raw_inventory]
    inventory_by_path = {entry.path: entry for entry in inventory_entries}
    if len(inventory_by_path) != len(inventory_entries):
        raise ValueError("Duplicate audio path in audio_inventory.json")

    pairs: dict[str, dict[AddressingCondition, DuplexMpeScenarioRecord]] = {}
    turn_counts: Counter[str] = Counter()
    referenced_paths: set[str] = {DUPLEX_MPE_DUTY_PREAMBLE_RELATIVE_PATH}
    for scenario in scenarios:
        pair_group = pairs.setdefault(scenario.pair_id, {})
        if scenario.addressing in pair_group:
            raise ValueError(f"Duplicate addressing condition for pair {scenario.pair_id}")
        pair_group[scenario.addressing] = scenario
        for turn in scenario.turns:
            if turn.audio_path not in inventory_by_path:
                raise ValueError(f"Missing audio_inventory.json entry for {turn.audio_path}")
            referenced_paths.add(turn.audio_path)
            turn_counts[f"{scenario.addressing}:{turn.event_type}"] += 1

    for pair_id, pair_group in pairs.items():
        if set(pair_group) != {"explicit", "implicit"}:
            raise ValueError(f"Incomplete paired conditions for {pair_id}")
        if pair_group["explicit"].gold_answer != pair_group["implicit"].gold_answer:
            raise ValueError(f"Mismatched paired gold_answer for {pair_id}")

    entries_to_verify = [inventory_by_path[path] for path in sorted(referenced_paths)] if mini else inventory_entries
    for entry in entries_to_verify:
        audio_path = resolve_dataset_audio_path(resolved_root, entry.path)
        if not audio_path.is_file() or audio_path.stat().st_size != entry.bytes:
            raise ValueError(f"Missing or truncated audio clip: {entry.path}")
        if verify_audio_checksums:
            if compute_file_sha256(audio_path) != entry.sha256:
                raise ValueError(f"Audio SHA-256 mismatch: {entry.path}")
            audio_header = soundfile.info(str(audio_path))
            if (audio_header.samplerate, audio_header.frames, audio_header.channels) != (
                entry.sample_rate,
                entry.frames,
                entry.channels,
            ):
                raise ValueError(f"Audio WAV header mismatch: {entry.path}")

    return {
        "scenarios": len(scenarios),
        "pairs": len(pairs),
        "verified_audio_files": len(entries_to_verify),
        "turn_counts": dict(turn_counts),
        "all_audio_checksums_verified": verify_audio_checksums,
    }


def download_duplex_mpe_dataset(
    *,
    destination_dir: Path,
    revision: str = DUPLEX_MPE_DATASET_REVISION,
    mini: bool = False,
    quiet: bool = False,
) -> None:
    """Download the pinned Duplex-MPE dataset or its deterministic 20-pair mini subset."""
    if revision != DUPLEX_MPE_DATASET_REVISION:
        raise ValueError(f"Duplex-MPE must use the pinned dataset revision {DUPLEX_MPE_DATASET_REVISION}")
    resolved_destination = destination_dir.expanduser().resolve()
    resolved_destination.mkdir(parents=True, exist_ok=True)
    if not quiet:
        subset_label = "mini (20 pairs)" if mini else "full (2000 pairs)"
        logger.info(
            "Downloading Duplex-MPE dataset",
            extra={
                "dataset": DUPLEX_MPE_DATASET_ID,
                "subset": subset_label,
                "revision": revision,
                "destination": str(resolved_destination),
            },
        )

    metadata_patterns = [
        "LICENSE",
        "README.md",
        "SHA256SUMS",
        "dataset.json",
        "audio_inventory.json",
        "manifests/explicit.json",
        "manifests/implicit.json",
        DUPLEX_MPE_DUTY_PREAMBLE_RELATIVE_PATH,
    ]
    if not mini:
        snapshot_download(
            repo_id=DUPLEX_MPE_DATASET_ID,
            repo_type="dataset",
            revision=revision,
            local_dir=str(resolved_destination),
        )
    else:
        snapshot_download(
            repo_id=DUPLEX_MPE_DATASET_ID,
            repo_type="dataset",
            revision=revision,
            local_dir=str(resolved_destination),
            allow_patterns=metadata_patterns,
        )
        mini_scenarios = load_duplex_mpe_scenarios(resolved_destination, addressing="all", mini=True)
        clip_patterns = sorted({turn.audio_path for scenario in mini_scenarios for turn in scenario.turns})
        snapshot_download(
            repo_id=DUPLEX_MPE_DATASET_ID,
            repo_type="dataset",
            revision=revision,
            local_dir=str(resolved_destination),
            allow_patterns=clip_patterns,
        )
    identity = inspect_duplex_mpe_dataset(resolved_destination)
    if not identity.metadata_matches_expected_revision:
        raise ValueError(
            f"Downloaded Duplex-MPE metadata SHA-256 digests do not match pinned revision {DUPLEX_MPE_DATASET_REVISION}"
        )
    summary = validate_duplex_mpe_dataset(resolved_destination, mini=mini, verify_audio_checksums=mini)
    expected_pairs = DUPLEX_MPE_MINI_PAIR_COUNT if mini else DUPLEX_MPE_PAPER_PAIR_COUNT
    if summary.get("pairs") != expected_pairs or summary.get("scenarios") != expected_pairs * 2:
        raise ValueError(
            f"Downloaded Duplex-MPE subset cardinality mismatch: expected {expected_pairs} pairs, "
            f"got {summary.get('pairs')}"
        )
    if not quiet:
        logger.info(
            "Duplex-MPE dataset ready",
            extra={"dataset": DUPLEX_MPE_DATASET_ID, "destination": str(resolved_destination)},
        )
