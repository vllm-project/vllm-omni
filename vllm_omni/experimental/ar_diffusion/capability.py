# SPDX-License-Identifier: Apache-2.0
"""Typed pipeline contract for the model-neutral AR-Diffusion runtime."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState


@dataclass(frozen=True)
class ARDiffusionKVBranchSpec:
    """One logical model KV branch and its worker-local storage slot.

    Distinct KV branches may share ``local_index`` when distributed execution
    guarantees that at most one of them runs on a worker. For example, two CFG
    branches use indices ``0, 1`` without CFG parallelism and may both use index
    ``0`` when each worker executes only one branch.
    """

    name: str
    local_index: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("AR-Diffusion KV branch names must be non-empty")
        if self.local_index < 0:
            raise ValueError(f"AR-Diffusion KV branch local_index must be non-negative, got {self.local_index}")


@dataclass(frozen=True)
class ARDiffusionCrossAttentionKVSpec:
    """A fixed-length, per-layer cross-attention KV allocation."""

    name: str
    num_tokens: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("AR-Diffusion cross-attention cache names must be non-empty")
        if self.num_tokens <= 0:
            raise ValueError(f"AR-Diffusion cross-attention num_tokens must be positive, got {self.num_tokens}")


@dataclass(frozen=True)
class ARDiffusionKVCacheSpec:
    """KV geometry and session policy supplied by an AR-Diffusion pipeline.

    ``tokens_per_frame`` is also the paged KV block size. A model forward may
    commit ``frames_per_block`` such blocks at once. ``window_frames`` is the
    recent sliding tail and ``sink_frames`` is the separately retained prefix;
    both are expressed in the same frame/block unit.

    ``max_scratch_tokens_per_branch`` is the maximum non-video KV (for example,
    action/state registers) that must coexist with an uncommitted video block.
    ``session_capacity`` is the pipeline's upper bound; a runner may retain
    fewer sessions when its current memory and execution policy is stricter.

    The runner owns all pool allocations and session lifetime. The pipeline
    owns its non-KV model state and receives reset/close notifications through
    :class:`SupportsARDiffusionPipeline`.
    """

    num_layers: int
    num_kv_heads: int
    head_size: int
    tokens_per_frame: int
    frames_per_block: int
    window_frames: int
    kv_branches: tuple[ARDiffusionKVBranchSpec, ...]
    session_capacity: int
    sink_frames: int = 0
    reset_at_boundary: bool = False
    cross_attention: tuple[ARDiffusionCrossAttentionKVSpec, ...] = ()
    max_model_len: int = 1 << 20
    max_scratch_tokens_per_branch: int = 0

    def __post_init__(self) -> None:
        positive_fields = {
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_size": self.head_size,
            "tokens_per_frame": self.tokens_per_frame,
            "frames_per_block": self.frames_per_block,
            "window_frames": self.window_frames,
            "session_capacity": self.session_capacity,
            "max_model_len": self.max_model_len,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"AR-Diffusion {name} must be positive, got {value}")
        if self.sink_frames < 0:
            raise ValueError(f"AR-Diffusion sink_frames must be non-negative, got {self.sink_frames}")
        if self.max_scratch_tokens_per_branch < 0:
            raise ValueError(
                "AR-Diffusion max_scratch_tokens_per_branch must be non-negative, "
                f"got {self.max_scratch_tokens_per_branch}"
            )
        if not self.kv_branches:
            raise ValueError("AR-Diffusion requires at least one KV branch")
        kv_branch_names = [kv_branch.name for kv_branch in self.kv_branches]
        if len(kv_branch_names) != len(set(kv_branch_names)):
            raise ValueError(f"AR-Diffusion KV branch names must be unique, got {kv_branch_names}")
        local_indices = {kv_branch.local_index for kv_branch in self.kv_branches}
        if local_indices != set(range(max(local_indices) + 1)):
            raise ValueError(
                f"AR-Diffusion KV branch local_index values must be contiguous from zero, got {sorted(local_indices)}"
            )
        cross_names = [cache.name for cache in self.cross_attention]
        if len(cross_names) != len(set(cross_names)):
            raise ValueError(f"AR-Diffusion cross-attention cache names must be unique, got {cross_names}")

    @property
    def num_local_kv_branches(self) -> int:
        """Number of KV branch storage slots allocated on this worker."""
        return max(kv_branch.local_index for kv_branch in self.kv_branches) + 1

    @property
    def cross_attention_lengths(self) -> dict[str, int]:
        return {cache.name: cache.num_tokens for cache in self.cross_attention}


@runtime_checkable
class SupportsARDiffusionPipeline(Protocol):
    """Required pipeline capability for :class:`ARDiffusionModelRunner`.

    A session begins when the runner first sees its ``session_id`` and persists
    across requests until reset, explicit close, LRU eviction, or a failed
    forward. ``bind_ar_diffusion_state`` exposes the runner-owned KV state only
    for the duration of one request. The pipeline must not retain the state
    after the context exits.
    """

    def ar_diffusion_kv_cache_spec(self) -> ARDiffusionKVCacheSpec:
        """Return immutable KV geometry and session policy for this worker."""
        ...

    def bind_ar_diffusion_state(
        self,
        session_id: str,
        state: ARDiffusionKVState,
    ) -> AbstractContextManager[None]:
        """Bind ``state`` to model execution for one request."""
        ...

    def reset_ar_diffusion_session(self, session_id: str) -> None:
        """Reset model-owned state for ``session_id`` after KV release."""
        ...

    def close_ar_diffusion_session(self, session_id: str) -> None:
        """Drop model-owned state after close, eviction, or failed forward."""
        ...


@runtime_checkable
class SupportsARDiffusionWarmup(Protocol):
    """Optional provider of model-valid warmup requests.

    Pipelines without this capability are loaded without an AR rollout warmup.
    """

    def ar_diffusion_warmup_requests(self, session_id: str) -> Iterable[OmniDiffusionRequest]:
        """Yield requests for compiled shapes, each carrying ``session_id``."""
        ...
