# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Omni-DuplexEval dataset adapter for the standard serving benchmark.

This module only wires the existing ``vllm_omni.benchmarks.duplex`` loader into
the ``BenchmarkDataset`` / ``SampleRequest`` contract used by ``vllm bench serve``.
Split selection, family filtering, media resolution and row normalisation stay in
:func:`vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset.load_samples` so the
standalone CLI and the serving backend share one implementation.

v2 (design §5.1): no judge options — ``DuplexEvalJudgeOptions`` has been removed
from this module; the judge lives exclusively in the DFX runner (Phase 2 / Phase 3).
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regex as re
from vllm.benchmarks.datasets import BenchmarkDataset, SampleRequest
from vllm.tokenizers import TokenizerLike

from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import (
    DEFAULT_DATASET,
    PR_SPLITS,
    RTD_SPLITS,
    DuplexSample,
    load_samples,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DUPLEX_EVAL_FAMILIES: tuple[str, ...] = ("all", "rtd", "pr")

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


_EXCLUDE_RE = re.compile(r"^(?:[^/]+/)?[^/]+$")


def normalize_exclude_ids(raw: Iterable[str]) -> frozenset[str]:
    """Normalise ``--exclude-ids`` entries to ``split/id`` form.

    A bare id (e.g. ``"565"``) is accepted and stored as-is — the per-split
    filter in :meth:`DuplexEvalDataset.sample` checks both ``split/id`` and
    bare ``id`` so that a short form works as a universal exclude across all
    splits.

    Raises ``ValueError`` when a token does not match the expected pattern.
    """
    result: set[str] = set()
    for token in raw:
        token = token.strip()
        if not token:
            continue
        if not _EXCLUDE_RE.match(token):
            raise ValueError(f"invalid exclude-id format {token!r}: expected 'id' or 'split/id'")
        result.add(token)
    return frozenset(result)


def sample_ids(samples: Iterable[DuplexSample]) -> list[str]:
    """Return ``split/id`` keys that identify the selected samples."""
    return [f"{sample.split}/{sample.id}" for sample in samples]


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DuplexEvalSessionOptions:
    """Generation layout shared by every sample in one run (v2: no judge)."""

    response_root: Path
    score_dir: Path
    ref_audio: str
    fps: float = 1.0
    mix: str = "question"
    pace: str = "realtime"
    clock: str = "media"
    unit_ms: int = 1000
    overwrite: bool = False
    exclude_ids: tuple[str, ...] = ()
    write_artifacts: bool = True


# ---------------------------------------------------------------------------
# Sample request
# ---------------------------------------------------------------------------


@dataclass
class DuplexEvalSampleRequest(SampleRequest):
    duplex_eval_sample: DuplexSample | None = None
    duplex_eval_options: DuplexEvalSessionOptions | None = None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _iter_splits(split: str) -> list[str]:
    """Expand ``"all"`` into the known split names; otherwise return [split]."""
    if split == "all":
        return sorted(RTD_SPLITS | PR_SPLITS)
    return [split]


class DuplexEvalDataset(BenchmarkDataset):
    """Adapter that exposes the ``duplex/`` loader via the ``BenchmarkDataset`` contract.

    Parameters
    ----------
    dataset
        Hugging Face dataset id, local path to a dataset directory or
        a ``.parquet`` file, or a JSON/JSONL manifest file path.
    split
        Split name or ``"all"`` (default).  When ``"all"``, samples are loaded
        split by split so that exclude-then-limit semantics work correctly.
    family
        Family filter: ``"all"`` (default), ``"rtd"`` or ``"pr"``.
    media_root
        Optional base directory for resolving relative media paths.
    limit
        Maximum number of samples **per split** (after exclusion).
        ``None`` means no limit.
    ids
        Only include samples whose id appears in this sequence.
    exclude_ids
        Exclude samples whose id (bare or ``split/id``) appears in this
        sequence.  Applied **before** ``limit`` per split.
    random_seed
        Seed for shuffling after loading.
    disable_shuffle
        When ``True``, skip shuffling (deterministic order).
    """

    def __init__(
        self,
        *,
        dataset: str = DEFAULT_DATASET,
        split: str = "all",
        family: str = "all",
        media_root: str | Path | None = None,
        limit: int | None = None,
        ids: Sequence[str] | None = None,
        exclude_ids: Sequence[str] | None = None,
        random_seed: int = 0,
        disable_shuffle: bool = False,
    ) -> None:
        super().__init__(
            dataset_path=str(dataset),
            random_seed=random_seed,
            disable_shuffle=disable_shuffle,
        )
        self.dataset = dataset
        self.split = split or "all"
        self.family = family or "all"
        self.media_root = media_root
        self.limit = limit
        self.ids = tuple(ids) if ids else None
        self.exclude_ids = tuple(exclude_ids) if exclude_ids else ()

    def sample(
        self,
        tokenizer: TokenizerLike | None,
        num_requests: int,
        *,
        request_id_prefix: str = "",
        options: DuplexEvalSessionOptions,
        **_: Any,
    ) -> list[SampleRequest]:
        """Select, shuffle and package samples.

        Semantics (per design §5.1 / §8.5 D-2):

        1. Expand ``split`` to individual split names.
        2. For each split, load samples via ``load_samples``.
        3. **Exclude** matching entries from the loaded list.
        4. **Limit** the remaining entries per split.
        5. Collect across splits, shuffle, slice to ``num_requests``.
        """
        del tokenizer
        exclude = normalize_exclude_ids(self.exclude_ids) if self.exclude_ids else frozenset()
        picked: list[DuplexSample] = []

        for split_name in _iter_splits(self.split):
            rows = load_samples(
                self.dataset,
                split=split_name,
                family=self.family,
                media_root=self.media_root,
                ids=self.ids,
            )
            # Exclude before limit (§8.5 D-2).
            rows = [s for s in rows if f"{s.split}/{s.id}" not in exclude and s.id not in exclude]
            if self.limit is not None:
                rows = rows[: max(0, self.limit)]
            picked.extend(rows)

        if not picked:
            raise ValueError("No Omni-DuplexEval samples were selected")

        if not self.disable_shuffle:
            random.Random(self.random_seed).shuffle(picked)

        # ``num_requests == 0`` means every available sample.
        if num_requests and num_requests > 0:
            picked = picked[:num_requests]

        return [
            DuplexEvalSampleRequest(
                prompt="",
                prompt_len=0,
                expected_output_len=0,
                multi_modal_data=None,
                request_id=f"{request_id_prefix}{index}",
                duplex_eval_sample=sample,
                duplex_eval_options=options,
            )
            for index, sample in enumerate(picked)
        ]


__all__ = [
    "DUPLEX_EVAL_FAMILIES",
    "DuplexEvalDataset",
    "DuplexEvalSampleRequest",
    "DuplexEvalSessionOptions",
    "normalize_exclude_ids",
    "sample_ids",
]
