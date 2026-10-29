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
from vllm_omni.diffusion.attention.schedule import (
    AttentionSchedule,
    AttentionSigmaSchedule,
    InvalidAttentionScheduleError,
    normalized_sigma,
    require_attention_schedule_fits,
    select_attention_profile,
    select_attention_profile_by_sigma,
)
from vllm_omni.diffusion.data import OmniDiffusionConfig

if TYPE_CHECKING:
    import torch

    from vllm_omni.diffusion.diffusion_kv.paged_attention_adapter import DiffusionPagedAttentionRuntime


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
    paged_kv_runtime: DiffusionPagedAttentionRuntime | None = None
    # Block-aligned prefix already resident in Scheduler-owned pages for the
    # active request-level prefill. Zero keeps the cold/full-prefill path.
    paged_kv_cached_prefix_len: int = 0
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
    # Normalized scheduler noise in [0, 1], published only while sigma windows are bound.
    denoise_sigma: float | None = None
    total_denoise_steps: int | None = None
    # Bound request schedules. None means unbound. () disables. A non-empty tuple
    # selects prepared profiles while a denoise step is published. A request binds
    # step ranges or sigma windows, never both.
    attention_schedule: AttentionSchedule | None = None
    attention_sigma_schedule: AttentionSigmaSchedule | None = None
    # Whether the bound step ranges are the request's own rather than an inherited service default.
    attention_schedule_from_request: bool = False
    # Profile selected for the bound schedule and the published step or sigma;
    # every scheduled layer in one step reads it instead of selecting again.
    _attention_profile_key: tuple[AttentionSchedule, int, int] | tuple[AttentionSigmaSchedule, float] | None = None
    _attention_profile: str | None = None
    # Per-request reference latent for img2img DiT models (e.g. Ming)
    ref_latent: torch.Tensor | None = None
    # Per-request projected direct-VLM condition (e.g., Ming-Image). For now for bsz 1.
    direct_condition: torch.Tensor | None = None
    # whether to split the text embed in sequence parallel, if True, the text embed will be split in sequence parallel

    # Sequence Parallel padding support
    # When sequence length is not divisible by SP world size, padding is added
    # These values are used by SequenceParallelGatherHook to remove padding,
    # and by attention layers to create attention masks dynamically
    sp_padding_size: int = 0
    # Original sequence length before padding (for removing padding in gather)
    sp_original_seq_len: int | None = None
    # Pre-padding global sequence length per SequenceParallelInput shard_group,
    # for models that auto-pad several independent sequences in one boundary.
    # The singleton fields above remain for single-sequence models.
    sp_shard_metadata: dict[str, int] = field(default_factory=dict)

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


def get_sp_shard_original_seq_len(shard_group: str) -> int | None:
    """Pre-padding global length of `shard_group`, or None if it was not split."""
    if not is_forward_context_available():
        return None
    return get_forward_context().sp_shard_metadata.get(shard_group)


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
    paged_kv_runtime: DiffusionPagedAttentionRuntime | None = None,
    paged_kv_cached_prefix_len: int = 0,
    in_diffusion_kv_memory_profile: bool = False,
    split_text_embed_in_sp: bool = False,
    denoise_step_idx: int | None = None,
):
    return ForwardContext(
        vllm_config=vllm_config,
        omni_diffusion_config=omni_diffusion_config,
        attn_metadata=attn_metadata,
        paged_kv_runtime=paged_kv_runtime,
        paged_kv_cached_prefix_len=paged_kv_cached_prefix_len,
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
    paged_kv_runtime: DiffusionPagedAttentionRuntime | None = None,
    paged_kv_cached_prefix_len: int = 0,
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
        paged_kv_cached_prefix_len=paged_kv_cached_prefix_len,
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


@contextmanager
def bind_attention_schedule(
    schedule: AttentionSchedule | None,
    sigma_schedule: AttentionSigmaSchedule | None = None,
    *,
    from_request: bool = False,
):
    """Install one request's step ranges and sigma windows and restore the previous ones on any exit.

    ``from_request`` marks step ranges the request sent itself; begin_scheduled_denoise checks only those.
    """
    if _forward_context is None:
        raise RuntimeError("bind_attention_schedule requires an active forward context")
    ctx = _forward_context
    previous = (ctx.attention_schedule, ctx.attention_sigma_schedule, ctx.attention_schedule_from_request)
    ctx.attention_schedule = None if schedule is None else tuple(schedule)
    ctx.attention_sigma_schedule = None if sigma_schedule is None else tuple(sigma_schedule)
    ctx.attention_schedule_from_request = from_request
    try:
        yield
    finally:
        ctx.attention_schedule, ctx.attention_sigma_schedule, ctx.attention_schedule_from_request = previous


def get_attention_schedule_profile() -> str | None:
    """Return the profile the bound schedule selects for the published step, or None.

    The selection is computed once per (schedule, step, total) or (sigma windows, sigma)
    and reused by every scheduled layer in that step.
    """
    ctx = _forward_context
    # Check the schedules before the step, so unscheduled compiled code never guards on the step index.
    if ctx is None:
        return None
    if ctx.attention_sigma_schedule:
        if ctx.denoise_step_idx is None:
            return None
        sigma = ctx.denoise_sigma
        if sigma is None:
            raise RuntimeError("attention sigma windows need denoise_sigma published with denoise_step_idx")
        sigma_key = (ctx.attention_sigma_schedule, sigma)
        if sigma_key != ctx._attention_profile_key:
            ctx._attention_profile = select_attention_profile_by_sigma(ctx.attention_sigma_schedule, sigma)
            ctx._attention_profile_key = sigma_key
        return ctx._attention_profile
    if not ctx.attention_schedule or ctx.denoise_step_idx is None:
        return None
    total = ctx.total_denoise_steps
    if total is None:
        raise RuntimeError("an attention schedule needs total_denoise_steps published with denoise_step_idx")
    key = (ctx.attention_schedule, ctx.denoise_step_idx, total)
    if key != ctx._attention_profile_key:
        ctx._attention_profile = select_attention_profile(
            ctx.attention_schedule, ctx.denoise_step_idx, total_steps=total
        )
        ctx._attention_profile_key = key
    return ctx._attention_profile


def set_forward_context_denoise_step_idx(step_idx: int | None) -> None:
    """Set the current diffusion denoise step on the active ForwardContext."""
    if _forward_context is not None:
        _forward_context.denoise_step_idx = step_idx
        if step_idx is not None:
            paged_kv_runtime = getattr(_forward_context, "paged_kv_runtime", None)
            ensure_active = getattr(paged_kv_runtime, "ensure_active", None)
            if callable(ensure_active):
                ensure_active(step_idx)


def begin_scheduled_denoise(total_steps: int) -> int | None:
    """Check the request's own bound schedule against one denoise sequence before its first forward.

    An inherited service default is not checked: it serves requests of any length, and a range that
    ends after this sequence applies up to its last step. Sigma windows do not depend on the length.
    Returns ``total_steps`` when non-empty step ranges or sigma windows are bound, else None.
    A pipeline that publishes a total only for schedules passes the result through, so an
    unscheduled run keeps publishing no total (RAINFUSION with end_step runs dense while the
    total is None). Call it once per sequence that restarts at step 0.
    """
    ctx = _forward_context
    if ctx is None or not (ctx.attention_schedule or ctx.attention_sigma_schedule):
        return None
    if ctx.attention_schedule and ctx.attention_schedule_from_request:
        require_attention_schedule_fits(ctx.attention_schedule, total_steps)
    return total_steps


@contextmanager
def clear_scheduled_denoise_step_on_exit(total_steps: int | None):
    """Clear the step and total a scheduled denoise loop published, on any exit of the loop.

    ``total_steps`` is the result of ``begin_scheduled_denoise``. Work after the loop, such as VAE
    decode, runs outside the denoise schedule, and a forward that raises mid-loop leaves no step
    published either. Without a schedule (None) the context is left as the loop left it.
    Unlike HunyuanImage-3.0's ``_clear_denoise_progress_on_exit``, which restores the timestep and
    total on every run, this keeps unscheduled Wan2.2 runs unchanged; do not merge the two.
    """
    try:
        yield
    finally:
        if total_steps is not None and _forward_context is not None:
            set_forward_context_denoise_step_idx(None)
            _forward_context.total_denoise_steps = None


@contextmanager
def request_denoise_progress(
    step_idx: int, total_steps: int, timestep: float | None = None, *, sigma: float | None = None
):
    """Publish one request's progress around its own forward and restore the previous values on any exit.

    For scheduled requests that share a denoise_step call but not their progress: each request
    is evaluated separately under its own step, total, timestep and sigma. Restoring assigns the
    saved fields directly, so a restore never calls the paged runtime's ensure_active.
    """
    if _forward_context is None:
        raise RuntimeError("request_denoise_progress requires an active forward context")
    ctx = _forward_context
    previous = (ctx.denoise_step_idx, ctx.denoise_timestep, ctx.denoise_sigma, ctx.total_denoise_steps)
    try:
        set_forward_context_denoise_step_idx(step_idx)
        ctx.denoise_timestep = None if timestep is None else float(timestep)
        set_forward_context_denoise_sigma(sigma)
        ctx.total_denoise_steps = total_steps
        yield
    finally:
        ctx.denoise_step_idx, ctx.denoise_timestep, ctx.denoise_sigma, ctx.total_denoise_steps = previous


def get_paged_kv_computed_tokens() -> tuple[int, ...]:
    runtime = _forward_context.paged_kv_runtime if _forward_context is not None else None
    if runtime is None:
        return ()
    return tuple(row.kv_start_pos for row in runtime.metadata.prefill_rows)


@contextmanager
def paged_kv_prefill(sequence_id: int, num_tokens: int):
    from dataclasses import replace

    runtime = get_forward_context().paged_kv_runtime
    if runtime is None:
        raise RuntimeError("Paged KV prefill requires an active runtime")
    rows = runtime.metadata.prefill_rows
    matching_rows = [row for row in rows if row.sequence_id == sequence_id]
    if len(matching_rows) != 1:
        raise ValueError(
            f"Paged KV prefill requires exactly one active row for sequence {sequence_id}; found {len(matching_rows)}"
        )
    row = matching_rows[0]
    if type(num_tokens) is not int or not row.kv_start_pos < num_tokens <= row.seq_len:
        raise ValueError(
            "Paged KV prefill target must extend the active prefix without exceeding its allocation: "
            f"start={row.kv_start_pos}, target={num_tokens!r}, allocated={row.seq_len}"
        )
    prefill = replace(row, query_len=num_tokens - row.kv_start_pos, seq_len=num_tokens)
    batch = runtime.adapter.prepare_batch((prefill,))
    with runtime.adapter.activate(batch):
        yield
    runtime.metadata = replace(
        runtime.metadata,
        prefill_rows=tuple(
            replace(item, kv_start_pos=num_tokens, query_len=item.seq_len - num_tokens)
            if item.sequence_id == sequence_id
            else item
            for item in rows
        ),
    )


def set_forward_context_denoise_sigma(sigma: float | None) -> None:
    """Publish normalized scheduler noise in [0, 1] for bound sigma windows.

    Without bound windows nothing reads the noise level, so the value is neither checked nor stored.
    """
    if _forward_context is not None and _forward_context.attention_sigma_schedule:
        _forward_context.denoise_sigma = None if sigma is None else normalized_sigma(sigma)


def check_scheduler_sigmas(scheduler: object, num_steps: int, *, sigma_scheduled: bool | None = None) -> None:
    """Normalize the sigma of each of the first ``num_steps`` steps once, right after set_timesteps.

    Sigma windows read these sigmas at every step. Checking them first raises the client error the
    step would raise before any forward: in request mode it fails this request's batch, whose
    requests share one sigma ladder, and in step mode, called from prepare_encode, only this request.
    ``sigma_scheduled`` defaults to whether sigma windows are bound; a caller outside the forward
    context passes the request's own answer.
    """
    if sigma_scheduled is None:
        sigma_scheduled = _forward_context is not None and bool(_forward_context.attention_sigma_schedule)
    sigmas = getattr(scheduler, "sigmas", None)
    if not sigma_scheduled or sigmas is None or num_steps <= 0 or len(sigmas) == 0:
        return
    # One host copy instead of a device sync per step; a step past the ladder is not checked, as in _denoise_sigma.
    head = sigmas[:num_steps]
    values = head.tolist() if hasattr(head, "tolist") else list(head)
    for value in values:
        DenoiseProgressMixin._normalize_scheduler_sigma(float(value), float(values[0]))


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
        normalized_sigma: float | None = None,
    ) -> None:
        set_forward_context_denoise_step_idx(step_idx)
        if _forward_context is not None:
            _forward_context.total_denoise_steps = total_steps
        if _forward_context is None:
            return
        scheduler = scheduler if scheduler is not None else getattr(self, "scheduler", None)
        if _forward_context.attention_sigma_schedule:
            # Only sigma windows read the noise level, so unscheduled runs never compute it.
            _forward_context.denoise_sigma = self._denoise_sigma(step_idx, scheduler, normalized_sigma)
        if normalized_timestep is not None:
            _forward_context.denoise_timestep = float(normalized_timestep)
            return
        if timestep is None:
            return
        ntt = getattr(getattr(scheduler, "config", None), "num_train_timesteps", None)
        _forward_context.denoise_timestep = float(timestep) / ntt if ntt else None

    @staticmethod
    def _denoise_sigma(step_idx: int | None, scheduler: object, sigma: float | None) -> float | None:
        """The caller's normalized sigma, else ``scheduler.sigmas[step_idx]`` relative to the first sigma.

        A raw model timestep is never read as noise; without either source the sigma stays None and
        the attention layer rejects the step.
        """
        if step_idx is None:
            return None
        if sigma is not None:
            return normalized_sigma(sigma)
        sigmas = getattr(scheduler, "sigmas", None)
        if sigmas is None or step_idx >= len(sigmas):
            return None
        # Pipelines raise this before their first forward through check_scheduler_sigmas; this repeats
        # the check for any other caller.
        return DenoiseProgressMixin._normalize_scheduler_sigma(float(sigmas[step_idx]), float(sigmas[0]))

    @staticmethod
    def _normalize_scheduler_sigma(current: float, reference: float) -> float:
        try:
            return normalized_sigma(current, reference)
        except ValueError as exc:
            # Wan2.2 T2V, TI2V and I2V build these sigmas from the request's flow_shift, so sigmas that
            # cannot be normalized into [0, 1] are a client error.
            raise InvalidAttentionScheduleError(
                "attention sigma windows cannot select a profile from this request's scheduler sigmas; "
                f"check flow_shift: {exc}"
            ) from exc


def set_forward_context_ref_latent(ref_latent: torch.Tensor | None) -> None:
    """Set the per-request reference latent on the active ForwardContext.

    Used by img2img-capable DiT models (e.g. Ming-flash-omni-2.0) so the
    transformer can read the reference latent from request scope instead of
    module instance state.
    """
    if _forward_context is not None:
        _forward_context.ref_latent = ref_latent


def set_forward_context_direct_condition(direct_condition: torch.Tensor | None) -> None:
    """Set the projected direct-VLM condition on the active context."""
    if _forward_context is not None:
        _forward_context.direct_condition = direct_condition
