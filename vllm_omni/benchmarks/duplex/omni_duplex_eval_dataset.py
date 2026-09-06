# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Dataset and split normalization for Omni-DuplexEval.

The loader accepts a Hugging Face dataset id, a JSON/JSONL manifest, or an
already materialized iterable.  Media is deliberately resolved at use time so
the benchmark remains usable in air-gapped environments.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_DATASET = "Hothan/Omni-DuplexEval"
RTD_SPLITS = frozenset(
    {
        "RTD_world_knowledge",
        "RTD_counting",
        "RTD_fine_grained_movement",
        "RTD_interaction_relation",
        "RTD_OCR",
        "RTD_Omni",
    }
)
PR_SPLITS = frozenset({"PR_correction", "PR_event_reminder", "PR_post_event_reminder"})
_TASK_ALIASES = {
    "correction": "correction",
    "pr_correction": "correction",
    "event_reminder": "proactive_reminder",
    "pr_event_reminder": "proactive_reminder",
    "proactive_reminder": "proactive_reminder",
    "post_event_reminder": "post_event_reminder",
    "pr_post_event_reminder": "post_event_reminder",
}


def canonical_task_type(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _TASK_ALIASES:
        raise ValueError(f"unsupported Omni-DuplexEval task type: {value!r}")
    return _TASK_ALIASES[key]


def task_type_for_split(split: str) -> str:
    """Match the task routing used by the official HF batch evaluator."""
    key = str(split).strip().lower()
    if "correction" in key:
        return "correction"
    if "post_event" in key:
        return "post_event_reminder"
    if key.startswith("pr"):
        return "proactive_reminder"
    raise ValueError(f"cannot infer proactive-reminder task from split {split!r}")


def family_for_split(split: str, task_type: str | None = None) -> str:
    if split in RTD_SPLITS or str(split).upper().startswith("RTD"):
        return "rtd"
    if split in PR_SPLITS or str(split).upper().startswith("PR"):
        return "pr"
    if task_type:
        return "rtd" if str(task_type).lower().startswith("rtd") else "pr"
    raise ValueError(f"cannot infer benchmark family from split {split!r}")


def _value(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class DuplexSample:
    id: str
    split: str
    family: str
    task_type: str | None
    video: Any
    question_audio: Any = None
    question_text: str = ""
    answer1: str = ""
    answer2: str = ""
    reminder1: Any = None
    reminder2: Any = None
    video_duration: float | None = None
    video_type: str = ""
    raw: dict[str, Any] | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any], *, split: str | None = None, media_root: Path | None = None) -> DuplexSample:
        chosen_split = str(split or _value(row, "split", "subset", "config", default=""))
        task_value = _value(row, "task_type", "task", "type")
        family = str(_value(row, "family", default="") or family_for_split(chosen_split, task_value))
        task = None
        if family == "pr":
            task = canonical_task_type(str(task_value)) if task_value else task_type_for_split(chosen_split)
        sample_id = str(_value(row, "id", "sample_id", "uid", "name", default=""))
        video = _value(row, "video", "video_path", "video_file")
        audio = _value(row, "question_audio", "audio", "question_wav")
        if media_root:

            def resolve(value: Any) -> Any:
                if not isinstance(value, str):
                    return value
                candidate = Path(value).expanduser()
                return str(candidate if candidate.is_absolute() else media_root / candidate)

            video, audio = resolve(video), resolve(audio)
        return cls(
            id=sample_id,
            split=chosen_split,
            family=family,
            task_type=task,
            video=video,
            question_audio=audio,
            question_text=str(_value(row, "question_text", "question", "instruction", default="") or ""),
            answer1=str(_value(row, "answer1", "answer", "ground_answer", default="") or ""),
            answer2=str(_value(row, "answer2", default="") or ""),
            reminder1=_value(row, "reminder1", "reminder_1"),
            reminder2=_value(row, "reminder2", "reminder_2"),
            video_duration=_float(_value(row, "video_duration", "duration")),
            video_type=str(_value(row, "video_type", default="") or ""),
            raw=dict(row),
        )


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("data", payload.get("samples", [payload]))
    if not isinstance(payload, list):
        raise ValueError("manifest must contain a JSON list")
    return payload


def load_samples(
    dataset: str | Path | Iterable[dict[str, Any]] = DEFAULT_DATASET,
    *,
    split: str | None = None,
    family: str = "all",
    media_root: str | Path | None = None,
    limit: int | None = None,
    ids: Iterable[str] | None = None,
) -> list[DuplexSample]:
    if isinstance(dataset, (str, Path)) and Path(str(dataset)).exists():
        rows = _read_manifest(Path(str(dataset)))
    elif not isinstance(dataset, (str, Path)):
        rows = list(dataset)
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("datasets is required for Hugging Face loading; use a local manifest instead") from exc
        kwargs: dict[str, Any] = {}
        if split and split != "all":
            kwargs["split"] = split
        loaded = load_dataset(str(dataset), **kwargs)
        if isinstance(loaded, dict):
            rows = []
            for name, table in loaded.items():
                rows.extend({**dict(row), "split": name} for row in table)
        else:
            rows = [dict(row) for row in loaded]
    wanted = set(str(item) for item in ids) if ids else None
    root = Path(media_root).expanduser() if media_root else None
    result = []
    for row in rows:
        sample = DuplexSample.from_row(row, split=None if split in (None, "all") else split, media_root=root)
        if split and split != "all" and sample.split != split:
            continue
        if family != "all" and sample.family != family:
            continue
        if wanted is not None and sample.id not in wanted:
            continue
        result.append(sample)
    if limit is not None:
        result = result[: max(0, limit)]
    return result
