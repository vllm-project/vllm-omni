# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Interface types for the omni prefix cache.

Naming aligns with vLLM's v1/core KV-cache design.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple, TypeAlias

import torch

# Four identities (still str/int at runtime, not interchangeable):
#   TensorName  which cached tensor (hidden / mm), not a prefix hash
#   ReqId       vLLM request id
#   Tid         WriteTask handle
#   StepId      save_outputs id; materialize or discard exactly once
TensorName: TypeAlias = str
ReqId: TypeAlias = str
Tid: TypeAlias = int
StepId: TypeAlias = int

# Reserved pool key for hidden states (mm keys are flat dotted names).
# Identity only — whether a model caches it lives on ModelCachePolicy.
HIDDEN_KEY: TensorName = "__hidden_states__"


def is_hidden_key(key: TensorName) -> bool:
    return key == HIDDEN_KEY


def without_hidden(keys: Iterable[TensorName]) -> set[TensorName]:
    """Drop the reserved hidden identity; the rest are mm names."""
    return {k for k in keys if k != HIDDEN_KEY}


class WriteSchedule(Enum):
    """Write scheduling policy for one WriteTask."""

    # Immediately-cached keys: device→host launched at save into the
    # staging pool; committer waits that event and writes the CPU pool.
    # The next save waits host_ready.
    JOIN_NEXT_STEP = "join_next_step"
    # Deferred mm: stays on the GPU clone until finish/abort (GPU-byte
    # budget may force a copy earlier). One WriteTask per request.
    JOIN_ON_FINISH = "join_on_finish"


@dataclass(frozen=True)
class PrefixCacheConfig:
    """Sizing and flow-control knobs (mirrors KVCacheConfig)."""

    num_blocks: int
    block_size: int
    # GPU-clone byte budget for JOIN_ON_FINISH; exceeding it forces a copy.
    gpu_staging_bytes: int = 512 * 1024 * 1024
    # Device→host staging: circular slots, one whole step each (not per request).
    # Host memory per key ≈ staging_depth * staging_capacity_tokens * width * dtype.
    # Prefer from_vllm_config so staging_capacity_tokens tracks max_num_batched_tokens.
    staging_depth: int = 4
    staging_capacity_tokens: int = 1024
    # How long save waits for a free staging slot (materialize/discard).
    staging_claim_timeout_s: float = 30.0
    # Device→host chunk size for JOIN_ON_FINISH (copied a piece at a time).
    copy_chunk_bytes: int = 16 * 1024 * 1024

    @classmethod
    def from_vllm_config(
        cls,
        *,
        num_blocks: int,
        block_size: int,
        scheduler_config: Any = None,
        model_config: Any = None,
    ) -> "PrefixCacheConfig":
        """Size device→host staging from the running scheduler.

        A slot holds one *step* (the whole batch), not one request:
        ``staging_capacity_tokens`` is ``max_num_batched_tokens`` (falls back
        to ``max_model_len``); ``staging_depth`` is how many unconsumed
        step ids may exist at once, not ``max_num_seqs``.

        Pinned staging is allocated lazily per key at
        ``depth * capacity_tokens * width * dtype``. There is no clamp: a
        step larger than capacity raises. A 16k-token thinking batch at
        hidden=2048 bf16 is ~256 MiB for hidden alone; each mm key adds
        another pool tensor.

        ``staging_depth`` is the dataclass default (4). There is no CLI or
        deploy YAML knob — changing it is a code change. Every save that
        issues a step id claims one slot, including saves with only leftover
        mm. A full pool waits for materialize/discard;
        ``staging_claim_timeout_s`` then errors.
        """
        batched = getattr(scheduler_config, "max_num_batched_tokens", None)
        model_len = getattr(scheduler_config, "max_model_len", None)
        if not model_len and model_config is not None:
            model_len = getattr(model_config, "max_model_len", None)
        try:
            capacity = int(batched or model_len or 1024)
        except (TypeError, ValueError):
            capacity = 1024
        return cls(
            num_blocks=num_blocks,
            block_size=block_size,
            staging_capacity_tokens=max(1, capacity),
        )


class StageCacheOutputs(NamedTuple):
    """Plain value object: per-request merged stage outputs."""

    # req -> full-prompt hidden states (None when policy skips them)
    hidden_states: dict[ReqId, torch.Tensor] | None
    # tensor name -> req -> payload element
    mm_outputs: dict[TensorName, dict[ReqId, Any]]


class OmniPrefixCacheUnmatchError(RuntimeError):
    """Contract, config, or KV-occupancy error that must raise.

    Includes hit spans that resolve to absent slots (omni cache diverged
    from vLLM KV), a step id consumed twice or never saved, a step larger
    than the staging page, and a failed write. Do not pretend these
    were a miss.
    """


class OmniPrefixCacheStagingTimeoutError(OmniPrefixCacheUnmatchError):
    """Save waited for a free staging slot and timed out."""


@dataclass(frozen=True)
class ModelCachePolicy:
    """Replaces getattr probing on models for cache behavior decisions.

    Hidden's *name* is ``HIDDEN_KEY`` (shared identity). This object
    answers whether this model caches it, and which mm keys are deferred.
    """

    needs_full_hidden_states: bool = True
    # Mm whose first dim is this step's token count; stays on the GPU
    # clone until finish/abort (JOIN_ON_FINISH). Also skipped by the
    # immediate on-device clone / device→host path.
    deferred_keys: frozenset[TensorName] = frozenset()

    @property
    def hidden_key(self) -> TensorName | None:
        """Pool key for hidden, or None when this model opts out."""
        return HIDDEN_KEY if self.needs_full_hidden_states else None

    def get_hit_keys(self, keys: Iterable[TensorName]) -> list[TensorName]:
        """Keys to plan/prefetch for a hit: hidden first (if cached), then mm."""
        mm = sorted(without_hidden(keys))
        return [HIDDEN_KEY, *mm] if self.needs_full_hidden_states else mm

    def skip_immediate_mm(self, key: TensorName) -> bool:
        """Immediate on-device clone must not take hidden or deferred keys from mm."""
        return is_hidden_key(key) or key in self.deferred_keys

    @classmethod
    def from_model(cls, model: Any) -> "ModelCachePolicy":
        """Shim over legacy per-model attributes (deprecation window)."""
        deferred = frozenset(getattr(model, "deferred_prefix_cache_mm_keys", ()) or ())
        return cls(
            needs_full_hidden_states=bool(getattr(model, "requires_full_prefix_cached_hidden_states", True)),
            deferred_keys=deferred,
        )
