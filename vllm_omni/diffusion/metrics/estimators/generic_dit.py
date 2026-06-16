from __future__ import annotations

from dataclasses import dataclass


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


def estimate_dit_forward_flops_per_gpu(stats: DiTForwardStats) -> int:
    """Estimate theoretical forward FLOPs per GPU as a scalar value."""

    return estimate_dit_flops_per_gpu(stats).total_flops_per_gpu
