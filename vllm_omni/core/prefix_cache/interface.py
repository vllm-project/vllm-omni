# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Interface types for the omni prefix cache.

Naming aligns with vLLM's v1/core KV-cache design.
"""

from dataclasses import dataclass
from typing import Any, NamedTuple, TypeAlias

import torch

# Three identities (still str/int at runtime, not interchangeable):
#   TensorName  which cached tensor (hidden / mm), not a prefix hash
#   ReqId       vLLM request id
#   PrefixCacheBatchId  save_outputs id; merge_batch or skip_merge exactly once
TensorName: TypeAlias = str
ReqId: TypeAlias = str
PrefixCacheBatchId: TypeAlias = int

# Reserved key for hidden states (mm keys are flat dotted names).
# Identity only — whether a model caches it lives on ModelCachePolicy.
HIDDEN_KEY: TensorName = "__hidden_states__"


@dataclass(frozen=True)
class PrefixCacheConfig:
    """Config for Omni output side (i.e., mm output & hidden state) prefix caching."""

    num_blocks: int
    block_size: int


class StageCacheOutputs(NamedTuple):
    """Plain value object: per-request merged stage outputs."""

    # req -> full-prompt hidden states (None when policy skips them)
    hidden_states: dict[ReqId, torch.Tensor] | None
    # tensor name -> req -> payload element
    mm_outputs: dict[TensorName, dict[ReqId, Any]]


class OmniPrefixCacheUnmatchError(RuntimeError):
    """Contract, config, or KV-occupancy error that must raise.

    Includes hit spans that resolve to absent slots (omni cache diverged
    from vLLM KV) and a batch id consumed twice or never saved. Do not
    pretend these were a miss.
    """


@dataclass(frozen=True)
class ModelCachePolicy:
    """Replaces getattr probing on models for cache behavior decisions.

    Hidden's *name* is ``HIDDEN_KEY`` (shared identity). This object
    answers whether this model caches it.
    """

    needs_full_hidden_states: bool = True
    # The model appends its multimodal outputs in sample(), after the step
    # was saved (YuE2 ships the finished song there), so the step's mm
    # snapshot is missing them. The runner then skips the merge
    # and builds the payload from the live outputs, which carry no
    # full-prompt hidden states.
    mm_outputs_written_in_sample: bool = False

    def __post_init__(self) -> None:
        if self.mm_outputs_written_in_sample and self.needs_full_hidden_states:
            raise OmniPrefixCacheUnmatchError(
                "mm_outputs_written_in_sample needs requires_full_prefix_cached_hidden_states = False: "
                "the payload is built from the live outputs, which carry no full-prompt hidden states"
            )

    @classmethod
    def from_model(cls, model: Any) -> "ModelCachePolicy":
        """Shim over legacy per-model attributes (deprecation window)."""
        return cls(
            needs_full_hidden_states=bool(getattr(model, "requires_full_prefix_cached_hidden_states", True)),
            mm_outputs_written_in_sample=bool(getattr(model, "mm_outputs_written_in_sample", False)),
        )
