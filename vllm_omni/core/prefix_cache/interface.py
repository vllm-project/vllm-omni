# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Interface types for the omni prefix cache.

Naming aligns with vLLM's v1/core KV-cache design.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple

import torch

# Reserved pool key for hidden states (mm keys are flat dotted names).
HIDDEN_KEY = "__hidden_states__"


class WriteSchedule(Enum):
    """Write scheduling policy for one WriteTask."""

    # Immediately-cached keys: D2H launched at save into the staging
    # pool; committer waits that event and scatters. Joined at the next save.
    JOIN_NEXT_STEP = "join_next_step"
    # Deferred mm: stays on the GPU freeze until finish/abort (cap
    # pressure may spill it earlier). One WriteTask per request.
    JOIN_ON_FINISH = "join_on_finish"


@dataclass(frozen=True)
class PrefixCacheConfig:
    """Sizing and flow-control knobs (mirrors KVCacheConfig)."""

    num_blocks: int
    block_size: int
    # GPU freeze byte budget for JOIN_ON_FINISH; exceeding it force-flushes.
    gpu_staging_bytes: int = 512 * 1024 * 1024
    # D2H staging: circular slots, one whole step each (not per request).
    # Host memory per key ≈ staging_depth * staging_capacity_tokens * width * dtype.
    # Prefer from_vllm_config so staging_capacity_tokens tracks max_num_batched_tokens.
    staging_depth: int = 4
    staging_capacity_tokens: int = 1024
    # D2H chunk size for the JOIN_ON_FINISH trickle.
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
        """Size D2H staging from the running scheduler.

        A slot holds one *step* (the whole batch), not one request:
        ``staging_capacity_tokens`` is ``max_num_batched_tokens`` (falls back
        to ``max_model_len``); ``staging_depth`` is the in-flight step bound,
        not ``max_num_seqs``.

        Pinned staging is allocated lazily per key at
        ``depth * capacity_tokens * width * dtype``. There is no clamp: a
        step larger than capacity fail-fasts. A 16k-token thinking batch at
        hidden=2048 bf16 is ~256 MiB for hidden alone; each mm key adds
        another slab.

        ``staging_depth`` is the dataclass default (4). There is no CLI or
        deploy YAML knob — changing it is a code change. The same value
        also bounds unconsumed step contexts; those are different failures
        that share a number today.
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

    # req_id -> full-prompt hidden states (None when policy skips them)
    hidden_states: dict[str, torch.Tensor] | None
    # mm_key -> req_id -> payload element (req-major)
    mm_outputs: dict[str, dict[str, Any]]


class OmniPrefixCacheUnmatchError(RuntimeError):
    """Fail-fast contract, config, or KV-occupancy error.

    Includes hit spans that resolve to absent slots (omni cache diverged
    from vLLM KV), consume-exactly-once violations, staging capacity /
    pool exhaustion, and poisoned saves. Never a degrade path.
    """


@dataclass(frozen=True)
class ModelCachePolicy:
    """Replaces getattr probing on models for cache behavior decisions."""

    needs_full_hidden_states: bool = True
    # Token-major mm that stays on the GPU freeze until finish/abort
    # (JOIN_ON_FINISH). Also skipped by the immediate freeze/D2H path.
    deferred_keys: frozenset[str] = frozenset()

    @classmethod
    def from_model(cls, model: Any) -> "ModelCachePolicy":
        """Shim over legacy per-model attributes (deprecation window)."""
        deferred = frozenset(getattr(model, "deferred_prefix_cache_mm_keys", ()) or ())
        return cls(
            needs_full_hidden_states=bool(getattr(model, "requires_full_prefix_cached_hidden_states", True)),
            deferred_keys=deferred,
        )
