# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import vllm.ir
from vllm.config import VllmConfig

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionMetadata,
)
from vllm_omni.diffusion.data import OmniDiffusionConfig

if TYPE_CHECKING:
    import torch


@dataclass
class ForwardContext:
    """
    set forward context for diffusion models
    """

    vllm_config: VllmConfig | None = None
    omni_diffusion_config: OmniDiffusionConfig | None = None
    attn_metadata: dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None = None
    # Runner-owned paged execution metadata/runtime. Attention resolves the
    # active Worker adapter from it; model code must not construct BlockTable
    # rows or activate the runtime directly.
    paged_kv_runtime: object | None = None
    # Active Worker-side paged KV adapter.  The adapter is installed only for
    # the duration of a paged forward; dense forwards leave this as ``None``.
    # Keep the field opaque here to avoid coupling the common context module to
    # the diffusion_kv implementation.
    paged_kv_adapter: Any | None = None
    # Startup-only memory profiling runs before Scheduler-owned pages exist.
    # Attention layers use this explicit scope to distinguish that probe from
    # a malformed paged request whose Worker adapter was not activated.
    in_diffusion_kv_memory_profile: bool = False
    split_text_embed_in_sp: bool = False
    denoise_step_idx: int | None = None
    denoise_timestep: float | None = None
    total_denoise_steps: int | None = None
    # Per-request reference latent for img2img DiT models (e.g. Ming)
    ref_latent: torch.Tensor | None = None
    # whether to split the text embed in sequence parallel, if True, the text embed will be split in sequence parallel

    # Sequence Parallel padding support
    # When sequence length is not divisible by SP world size, padding is added
    # These values are used by SequenceParallelGatherHook to remove padding,
    # and by attention layers to create attention masks dynamically
    sp_padding_size: int = 0
    # Original sequence length before padding (for removing padding in gather)
    sp_original_seq_len: int | None = None

    # Set by registry when _sp_plan hooks are applied.
    # When True, sp_active is determined by _sp_shard_depth (for _sp_plan hooks)
    # When False, sp_active defaults to True when sequence_parallel_size > 1 (for manual SP, standalone tests, etc.)
    sp_plan_hooks_applied: bool = False
    # SP active scope tracking within the _sp_plan hook mechanism.
    # Tracks the depth of SP sharding - incremented on shard, decremented on gather
    # Used by attention layers to determine if SP communication should be enabled
    _sp_shard_depth: int = 0
    # One entry per active SP split boundary: whether it auto-pads, which makes
    # every rank's shard the same length and lets attention collectives skip a
    # runtime length all-gather. Pushed on split, popped on gather.
    _sp_equal_pad_stack: list[bool] = field(default_factory=list)

    @property
    def sp_active(self) -> bool:
        """Returns True when SP attention parallelism should be enabled.

        - If _sp_plan hooks are applied: use _sp_shard_depth (0 = outside sharded region).
        - If _sp_plan hooks are NOT applied: default to True when sequence_parallel_size > 1,
          since _sp_shard_depth is only meaningful within the _sp_plan hook mechanism.
        """
        if self.sp_plan_hooks_applied:
            return self._sp_shard_depth > 0
        # No _sp_plan: assume SP active when configured (manual SP, standalone tests)
        if self.omni_diffusion_config is None:
            raise ValueError(
                "omni_diffusion_config is not set when checking sp_active! "
                "This usually means set_forward_context() was not called. "
                "Please call with set_forward_context(omni_diffusion_config=...)."
            )

        sp_size = self.omni_diffusion_config.parallel_config.sequence_parallel_size
        return sp_size is not None and sp_size > 1

    @property
    def sp_rank_local_seq_lens_equal(self) -> bool:
        """Whether every active SP boundary guarantees equal local shard sizes.

        Only framework-managed auto_pad boundaries provide this contract; a
        region that also shards manually keeps the dynamic length exchange.
        """
        return bool(self._sp_equal_pad_stack) and all(self._sp_equal_pad_stack)

    def __post_init__(self):
        pass


_forward_context: ForwardContext | None = None


def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    assert _forward_context is not None, (
        "Forward context is not set. Please use `set_forward_context` to set the forward context."
    )
    return _forward_context


def is_forward_context_available() -> bool:
    return _forward_context is not None


def build_local_sp_padding_mask(
    batch_size: int,
    local_seq_len: int,
    device,
):
    """Build a per-rank SP padding mask that matches the local shard shape.

    Auto-padding is applied before sequence-parallel sharding, so attention on each
    rank must receive a mask for its local shard, not for the global padded sequence.
    """
    if not is_forward_context_available():
        return None

    ctx = get_forward_context()
    if ctx.sp_original_seq_len is None or ctx.sp_padding_size <= 0:
        return None

    from vllm_omni.diffusion.distributed.parallel_state import (
        get_sequence_parallel_rank,
    )
    from vllm_omni.diffusion.distributed.utils import (
        build_local_sp_padding_mask as build_local_sp_padding_mask_for_rank,
    )

    return build_local_sp_padding_mask_for_rank(
        batch_size=batch_size,
        local_seq_len=local_seq_len,
        sp_original_seq_len=ctx.sp_original_seq_len,
        sp_padding_size=ctx.sp_padding_size,
        sequence_parallel_rank=get_sequence_parallel_rank(),
        device=device,
    )


def get_ulysses_mode(*, default: str = "strict") -> str:
    """Resolve the Ulysses-SP mode from the current ForwardContext.

    Returns `default` when ForwardContext is unavailable or the diffusion
    config is not set.
    """
    if not is_forward_context_available():
        return default

    cfg = get_forward_context().omni_diffusion_config
    if cfg is None:
        return default

    parallel_config = cfg.parallel_config
    return str(getattr(parallel_config, "ulysses_mode", default))


def create_forward_context(
    vllm_config: VllmConfig | None = None,
    omni_diffusion_config: OmniDiffusionConfig | None = None,
    attn_metadata: dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None = None,
    paged_kv_runtime: object | None = None,
    in_diffusion_kv_memory_profile: bool = False,
    split_text_embed_in_sp: bool = False,
    denoise_step_idx: int | None = None,
):
    return ForwardContext(
        vllm_config=vllm_config,
        omni_diffusion_config=omni_diffusion_config,
        attn_metadata=attn_metadata,
        paged_kv_runtime=paged_kv_runtime,
        in_diffusion_kv_memory_profile=in_diffusion_kv_memory_profile,
        split_text_embed_in_sp=split_text_embed_in_sp,
        denoise_step_idx=denoise_step_idx,
    )


@contextmanager
def override_forward_context(forward_context: ForwardContext | None):
    """A context manager that overrides the current forward context.
    This is used to override the forward context for a specific
    forward pass.
    """
    global _forward_context
    prev_context = _forward_context
    _forward_context = forward_context
    try:
        yield
    finally:
        _forward_context = prev_context


@contextmanager
def set_forward_context(
    vllm_config: VllmConfig | None = None,
    omni_diffusion_config: OmniDiffusionConfig | None = None,
    attn_metadata: dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None = None,
    paged_kv_runtime: object | None = None,
    in_diffusion_kv_memory_profile: bool = False,
    split_text_embed_in_sp: bool = False,
    denoise_step_idx: int | None = None,
):
    """A context manager that stores the current forward context,
    can be attention metadata, split_text_embed_in_sp, etc.
    Here we can inject common logic for every model forward pass.
    """
    forward_context = create_forward_context(
        vllm_config=vllm_config,
        omni_diffusion_config=omni_diffusion_config,
        attn_metadata=attn_metadata,
        paged_kv_runtime=paged_kv_runtime,
        in_diffusion_kv_memory_profile=in_diffusion_kv_memory_profile,
        split_text_embed_in_sp=split_text_embed_in_sp,
        denoise_step_idx=denoise_step_idx,
    )
    # vLLM CustomOp dispatch (e.g. QKVParallelLinear) requires a global
    # vLLM config set via set_current_vllm_config().
    # Also set priority for vLLM IR ops (e.g. RMSNorm), copied from vllm/forward_context.py
    with override_forward_context(forward_context):
        if vllm_config is None:
            yield
        else:
            # Local import to avoid importing vllm.config.vllm at module import time.
            from vllm.config.vllm import set_current_vllm_config

            with (
                set_current_vllm_config(vllm_config),
                vllm_config.kernel_config.ir_op_priority.set_priority(),
                vllm.ir.enable_torch_wrap(vllm_config.compilation_config.ir_enable_torch_wrap),
            ):
                yield


@contextmanager
def override_paged_kv_adapter(adapter: Any | None):
    """Temporarily expose a Worker paged-KV adapter to Omni Attention.

    This is deliberately a small context override instead of a second global
    forward context.  The model runner owns the outer context and the adapter
    only replaces one opaque field while its prepared native metadata is live.
    """

    if _forward_context is None:
        # Unit-level adapter users can prepare/activate metadata without an
        # Omni model forward context.  In that case there is nothing to
        # override and the adapter's explicit ``forward`` API remains usable.
        yield
        return

    previous = _forward_context.paged_kv_adapter
    _forward_context.paged_kv_adapter = adapter
    try:
        yield
    finally:
        _forward_context.paged_kv_adapter = previous


def set_forward_context_denoise_step_idx(step_idx: int | None) -> None:
    """Set the current diffusion denoise step on the active ForwardContext."""
    if _forward_context is not None:
        _forward_context.denoise_step_idx = step_idx
        if step_idx is not None:
            paged_kv_runtime = getattr(_forward_context, "paged_kv_runtime", None)
            ensure_active = getattr(paged_kv_runtime, "ensure_active", None)
            if callable(ensure_active):
                ensure_active(step_idx)


def set_forward_context_denoise_timestep(timestep: float | None) -> None:
    """Set the normalized (descending, 1 -> 0) denoise timestep.

    Timestep-gated attention features read this; pipelines that drive their own
    denoise loop can publish it directly instead of going through
    :meth:`DenoiseProgressMixin.record_denoise_step`.
    """
    if _forward_context is not None:
        _forward_context.denoise_timestep = None if timestep is None else float(timestep)


def set_forward_context_denoise_total_steps(total_steps: int | None) -> None:
    """Set the total denoise step count on the active ForwardContext.

    Denoise loops publish it so tail-fallback gates (e.g. ``end_step`` in
    RAINFUSION_ATTN) know when the final denoise steps begin.
    """
    if _forward_context is not None:
        _forward_context.total_denoise_steps = total_steps


class DenoiseProgressMixin:
    def record_denoise_step(
        self,
        step_idx: int | None,
        timestep=None,
        scheduler=None,
        normalized_timestep: float | None = None,
        total_steps: int | None = None,
    ) -> None:
        set_forward_context_denoise_step_idx(step_idx)
        if _forward_context is not None:
            _forward_context.total_denoise_steps = total_steps
        if _forward_context is None:
            return
        if normalized_timestep is not None:
            _forward_context.denoise_timestep = float(normalized_timestep)
            return
        if timestep is None:
            return
        scheduler = scheduler if scheduler is not None else getattr(self, "scheduler", None)
        ntt = getattr(getattr(scheduler, "config", None), "num_train_timesteps", None)
        _forward_context.denoise_timestep = float(timestep) / ntt if ntt else None


def set_forward_context_ref_latent(ref_latent: torch.Tensor | None) -> None:
    """Set the per-request reference latent on the active ForwardContext.

    Used by img2img-capable DiT models (e.g. Ming-flash-omni-2.0) so the
    transformer can read the reference latent from request scope instead of
    module instance state.
    """
    if _forward_context is not None:
        _forward_context.ref_latent = ref_latent
