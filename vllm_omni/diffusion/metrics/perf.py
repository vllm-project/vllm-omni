# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Estimated/theoretical DiT FLOPs helpers for diffusion MFU metrics."""

from dataclasses import dataclass
from typing import Any

from vllm.v1.metrics.perf import PerfStats


@dataclass(frozen=True)
class DiTForwardStats:
    """Runtime/model shape inputs for the generic DiT FLOPs estimate."""

    batch_size: int
    seq_len: int
    context_len: int
    hidden_dim: int
    ffn_dim: int
    num_layers: int
    num_steps: int
    forwards_per_step_per_gpu: int

    def __post_init__(self) -> None:
        for field_name, value in self.__dict__.items():
            if not isinstance(value, int):
                raise TypeError(f"{field_name} must be int, got {type(value)!r}")
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}")


@dataclass(frozen=True)
class DiTFlopsBreakdown:
    """Componentized per-layer and request-total DiT FLOPs estimate."""

    qkv_out_flops: int
    cross_attn_proj_flops: int
    mlp_flops: int
    self_attn_flops: int
    cross_attn_flops: int
    flops_per_layer: int
    total_flops_per_gpu: int


def estimate_dit_flops_per_gpu(stats: DiTForwardStats) -> DiTFlopsBreakdown:
    """Estimate theoretical forward FLOPs per GPU for a DiT diffusion request.

    The formula intentionally mirrors issue #4077's generic per-layer DiT
    estimate, then makes diffusion request multipliers explicit: denoising
    step count and the number of transformer forwards this GPU executes per
    step, including sequential CFG when applicable.
    """

    qkv_out_flops = 8 * stats.hidden_dim * stats.hidden_dim * stats.seq_len
    cross_attn_proj_flops = (
        4 * stats.hidden_dim * stats.hidden_dim * stats.seq_len
        + 4 * stats.hidden_dim * stats.hidden_dim * stats.context_len
    )
    mlp_flops = 4 * stats.hidden_dim * stats.ffn_dim * stats.seq_len
    self_attn_flops = 4 * stats.seq_len * stats.seq_len * stats.hidden_dim
    cross_attn_flops = 4 * stats.seq_len * stats.context_len * stats.hidden_dim
    flops_per_layer = qkv_out_flops + cross_attn_proj_flops + mlp_flops + self_attn_flops + cross_attn_flops
    total_flops_per_gpu = (
        stats.batch_size * flops_per_layer * stats.num_layers * stats.num_steps * stats.forwards_per_step_per_gpu
    )
    return DiTFlopsBreakdown(
        qkv_out_flops=qkv_out_flops,
        cross_attn_proj_flops=cross_attn_proj_flops,
        mlp_flops=mlp_flops,
        self_attn_flops=self_attn_flops,
        cross_attn_flops=cross_attn_flops,
        flops_per_layer=flops_per_layer,
        total_flops_per_gpu=total_flops_per_gpu,
    )


def to_perf_stats(breakdown: DiTFlopsBreakdown) -> PerfStats:
    """Convert the DiT estimate into upstream vLLM PerfStats."""

    return PerfStats(num_flops_per_gpu=breakdown.total_flops_per_gpu)


def estimate_dit_perf_stats(stats: DiTForwardStats) -> PerfStats:
    """Estimate DiT FLOPs and wrap them in upstream vLLM PerfStats."""

    return to_perf_stats(estimate_dit_flops_per_gpu(stats))


def _positive_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        int_value = int(value)
    except Exception:
        return None
    if int_value > 0:
        return int_value
    return None


def _shape_dim(value: Any, index: int) -> int | None:
    try:
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) <= index:
            return None
        return _positive_int_or_none(shape[index])
    except Exception:
        return None


def collect_flux_dit_forward_stats(
    *,
    latents: Any,
    prompt_embeds: Any,
    timesteps: Any,
    transformer: Any,
    do_true_cfg: bool,
    cfg_parallel_world_size: int = 1,
) -> DiTForwardStats | None:
    """Build the generic DiT estimate input from Flux runtime tensors.

    Flux MVP accounting deliberately uses the packed latent sequence length
    that the transformer executes. This is hardware-MFU oriented shape
    accounting, not useful-token goodput accounting.
    """

    batch_size = _shape_dim(latents, 0)
    seq_len = _shape_dim(latents, 1)
    context_len = _shape_dim(prompt_embeds, 1)
    hidden_dim = _positive_int_or_none(getattr(transformer, "inner_dim", None))
    if hidden_dim is None:
        config = getattr(transformer, "config", None)
        hidden_dim = _positive_int_or_none(getattr(config, "hidden_size", None))
    if hidden_dim is None:
        return None

    transformer_blocks = getattr(transformer, "transformer_blocks", None)
    single_transformer_blocks = getattr(transformer, "single_transformer_blocks", None)
    try:
        num_layers = len(transformer_blocks) + len(single_transformer_blocks)
    except Exception:
        return None
    num_layers = _positive_int_or_none(num_layers)

    try:
        num_steps = len(timesteps)
    except Exception:
        return None
    num_steps = _positive_int_or_none(num_steps)

    cfg_parallel_world_size = _positive_int_or_none(cfg_parallel_world_size)
    if cfg_parallel_world_size is None:
        return None
    forwards_per_step_per_gpu = 1
    if do_true_cfg and cfg_parallel_world_size == 1:
        forwards_per_step_per_gpu = 2

    if None in (batch_size, seq_len, context_len, num_layers, num_steps):
        return None

    try:
        return DiTForwardStats(
            batch_size=batch_size,
            seq_len=seq_len,
            context_len=context_len,
            hidden_dim=hidden_dim,
            ffn_dim=4 * hidden_dim,
            num_layers=num_layers,
            num_steps=num_steps,
            forwards_per_step_per_gpu=forwards_per_step_per_gpu,
        )
    except (TypeError, ValueError):
        return None
