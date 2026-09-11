# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Dataset and split normalization for Omni-DuplexEval.

The loader accepts a Hugging Face dataset id, a local Hugging Face dataset
layout (a directory or a ``.parquet`` file), a JSON/JSONL manifest, or an
already materialized iterable.  Media is deliberately resolved at use time so
the benchmark remains usable in air-gapped environments.

Local directory mirrors are expected to use a *single* Hugging Face
configuration whose data files are named after the benchmark splits
(``data/<SPLIT>-00000-of-00001.parquet``, with ``<SPLIT>`` one of the RTD_*/PR_*
names), so ``datasets.load_dataset`` exposes one physical split per benchmark
split.  A single ``.parquet`` file and a mirror that Hugging Face collapses into
a generic ``train`` split are also accepted, but then the requested ``split`` is
applied as a filter on each row's preserved ``split``/``subset``/``config``
identity instead of as a physical split name.  Rows that identify neither their
split nor their family/task raise a clear error that explains the required
layout.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
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


def _identity_error(split: str, sample_id: str, *, family: bool) -> str:
    """Build an actionable error for a row whose family/task cannot be derived."""
    subject = "family" if family else "proactive-reminder task type"
    label = f" for sample {sample_id!r}" if sample_id else ""
    detail = (
        "the split is not an RTD_*/PR_* name and no per-row 'task_type' is present"
        if family
        else "no per-row 'task_type' is present"
    )
    return (
        f"cannot infer the Omni-DuplexEval {subject}{label} from split {split!r}: {detail}. "
        "Load a local Omni-DuplexEval mirror whose data files are named after the "
        "Hugging Face splits (RTD_*/PR_*) or add per-row "
        "'family'/'task_type'/'split' columns."
    )


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
        sample_id = str(_value(row, "id", "sample_id", "uid", "name", default=""))
        task_value = _value(row, "task_type", "task", "type")
        family_value = _value(row, "family", default="")
        if family_value:
            family = str(family_value)
        else:
            try:
                family = family_for_split(chosen_split, task_value)
            except ValueError as exc:
                raise ValueError(_identity_error(chosen_split, sample_id, family=True)) from exc
        task = None
        if family == "pr":
            if task_value:
                task = canonical_task_type(str(task_value))
            else:
                try:
                    task = task_type_for_split(chosen_split)
                except ValueError as exc:
                    raise ValueError(_identity_error(chosen_split, sample_id, family=False)) from exc
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


def _has_split_identity(row: dict[str, Any]) -> bool:
    """Whether a row already identifies the benchmark split it belongs to."""
    return any(row.get(key) for key in ("split", "subset", "config"))


def _stamp_split(rows: Iterable[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    """Stamp a physical Hugging Face split name onto rows that lack identity.

    Hugging Face collapses a config-less local mirror (a bare ``data/``
    directory) or a single ``.parquet`` file into a single generic split
    (``train``).  A row's own split identity (``split``/``subset``/``config``)
    must win so family inference is not poisoned by that generic name; the
    physical split name is only stamped when the row does not identify itself.
    """
    stamped: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if name and not _has_split_identity(item):
            item["split"] = name
        stamped.append(item)
    return stamped


def _read_hf(load_dataset: Callable[..., Any], dataset: str, want: str | None) -> Any:
    """Call ``datasets.load_dataset`` for a local layout or a remote id.

    The physical split is only requested when one was asked for; ``datasets``
    then returns a single ``Dataset`` (an iterable of rows) instead of a
    ``DatasetDict`` (a mapping of split name to such an iterable). The result is
    normalized by :func:`_rows_from_hf`.
    """
    if Path(dataset).suffix.lower() == ".parquet":
        if want:
            return load_dataset("parquet", data_files=dataset, split=want)
        return load_dataset("parquet", data_files=dataset)
    if want:
        return load_dataset(dataset, split=want)
    return load_dataset(dataset)


def _rows_from_hf(dataset: str, *, split: str | None) -> list[dict[str, Any]]:
    """Load rows from a Hugging Face dataset id or a local dataset layout.

    ``datasets.load_dataset`` accepts a remote id, a local dataset directory
    (``data/<SPLIT>-00000-of-00001.parquet``) and, via the ``parquet`` builder,
    a single ``.parquet`` file, which keeps the benchmark usable offline with a
    locally mirrored copy of the dataset.

    The logical benchmark split (``RTD_OCR``, ``PR_correction``, ...) is not
    forwarded blindly as a physical Hugging Face split: a single ``.parquet``
    file and a collapsed local mirror only expose a generic ``train`` split, so
    passing the logical name there raises ``Unknown split``.  The physical
    dataset is loaded instead and the requested split is applied later, on the
    preserved per-row identity (see :func:`load_samples`).
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required to load a Hugging Face dataset id or a local "
            "dataset directory/parquet file; use a local JSON/JSONL manifest instead"
        ) from exc

    requested = split if split and split != "all" else None
    is_local = Path(dataset).exists()
    if Path(dataset).suffix.lower() == ".parquet":
        # A single parquet file is always exposed as one generic ``train``
        # split; the logical benchmark split is filtered from row identity
        # afterwards rather than forwarded as a physical split name.
        loaded = _read_hf(load_dataset, dataset, None)
    else:
        try:
            loaded = _read_hf(load_dataset, dataset, requested)
        except ValueError as exc:
            # A collapsed *local* mirror (bare ``data/`` directory) does not
            # expose the logical split as a physical Hugging Face split; fall
            # back to the whole physical dataset and filter by row identity.
            # A remote id keeps the original error so a mistyped split still
            # fails loudly instead of silently selecting nothing.
            if not is_local or requested is None or "Unknown split" not in str(exc):
                raise
            loaded = _read_hf(load_dataset, dataset, None)
            requested = None

    if isinstance(loaded, dict):
        rows: list[dict[str, Any]] = []
        for name, table in loaded.items():
            rows.extend(_stamp_split((dict(row) for row in table), name))
        return rows
    return _stamp_split((dict(row) for row in loaded), requested or "")


def load_samples(
    dataset: str | Path | Iterable[dict[str, Any]] = DEFAULT_DATASET,
    *,
    split: str | None = None,
    family: str = "all",
    media_root: str | Path | None = None,
    limit: int | None = None,
    ids: Iterable[str] | None = None,
) -> list[DuplexSample]:
    if isinstance(dataset, str | Path):
        path = Path(str(dataset))
        if path.exists():
            # A directory or a ``.parquet`` file is a Hugging Face dataset
            # layout (e.g. a locally mirrored Omni-DuplexEval snapshot); only
            # single JSON/JSONL files are parsed as manifests.  Routing by path
            # shape instead of mere existence lets local Hugging Face datasets
            # load through ``datasets.load_dataset``.
            if path.is_dir() or path.suffix.lower() == ".parquet":
                rows = _rows_from_hf(str(path), split=split)
            else:
                try:
                    rows = _read_manifest(path)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise ValueError(
                        f"cannot load dataset from {str(path)!r}: expected a "
                        "JSON/JSONL manifest file, a local Hugging Face dataset "
                        "directory, or a single .parquet file"
                    ) from exc
        else:
            rows = _rows_from_hf(str(dataset), split=split)
    else:
        rows = list(dataset)
    wanted = set(str(item) for item in ids) if ids else None
    root = Path(media_root).expanduser() if media_root else None
    result = []
    for row in rows:
        # The requested split is a *filter* over each row's own split identity;
        # never pass it as an override, which would relabel every row as the
        # requested split (turning a collapsed local dataset into a full match
        # instead of a filter).
        sample = DuplexSample.from_row(row, media_root=root)
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
