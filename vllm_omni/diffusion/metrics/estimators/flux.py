from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm_omni.diffusion.metrics.common import (
    divide_flops_for_parallelism,
    positive_int_or_none,
    require_non_negative,
    require_positive,
    shape_dim,
)


@dataclass(frozen=True)
class FluxDiTForwardStats:
    """Runtime shape for Flux-specific estimated/theoretical DiT FLOPs."""

    batch_size: int
    image_seq_len: int
    text_seq_len: int
    hidden_dim: int
    ffn_dim: int
    num_double_layers: int
    num_single_layers: int
    num_steps: int
    forwards_per_step_per_gpu: int = 1
    tensor_parallel_size: int = 1

    def __post_init__(self) -> None:
        for field_name, value in self.__dict__.items():
            require_non_negative(field_name, value)
        require_positive("forwards_per_step_per_gpu", self.forwards_per_step_per_gpu)
        require_positive("tensor_parallel_size", self.tensor_parallel_size)


def estimate_flux_dit_forward_flops_per_gpu(stats: FluxDiTForwardStats) -> int:
    """Estimate Flux denoiser FLOPs with dual/single block accounting."""

    joint_seq_len = stats.image_seq_len + stats.text_seq_len
    dual_layer_flops = (
        8 * stats.hidden_dim * stats.hidden_dim * stats.image_seq_len
        + 8 * stats.hidden_dim * stats.hidden_dim * stats.text_seq_len
        + 4 * stats.hidden_dim * stats.ffn_dim * stats.image_seq_len
        + 4 * stats.hidden_dim * stats.ffn_dim * stats.text_seq_len
        + 4 * joint_seq_len * joint_seq_len * stats.hidden_dim
    )
    single_layer_flops = (
        8 * stats.hidden_dim * stats.hidden_dim * joint_seq_len
        + 4 * stats.hidden_dim * stats.ffn_dim * joint_seq_len
        + 4 * joint_seq_len * joint_seq_len * stats.hidden_dim
    )
    per_forward_flops = divide_flops_for_parallelism(
        stats.num_double_layers * dual_layer_flops + stats.num_single_layers * single_layer_flops,
        stats.tensor_parallel_size,
    )
    return stats.batch_size * per_forward_flops * stats.num_steps * stats.forwards_per_step_per_gpu


def collect_flux_dit_forward_stats(
    *,
    latents: Any,
    prompt_embeds: Any,
    timesteps: Any,
    transformer: Any,
    do_true_cfg: bool,
    cfg_parallel_world_size: int = 1,
    tensor_parallel_size: int = 1,
) -> FluxDiTForwardStats | None:
    """Build Flux-specific DiT estimate input from Flux runtime tensors."""

    batch_size = shape_dim(latents, 0)
    image_seq_len = shape_dim(latents, 1)
    text_seq_len = shape_dim(prompt_embeds, 1)
    hidden_dim = positive_int_or_none(getattr(transformer, "inner_dim", None))
    if hidden_dim is None:
        config = getattr(transformer, "config", None)
        hidden_dim = positive_int_or_none(getattr(config, "hidden_size", None))
    if hidden_dim is None:
        return None

    transformer_blocks = getattr(transformer, "transformer_blocks", None)
    single_transformer_blocks = getattr(transformer, "single_transformer_blocks", None)
    try:
        num_double_layers = len(transformer_blocks)
        num_single_layers = len(single_transformer_blocks)
    except Exception:
        return None

    num_double_layers = positive_int_or_none(num_double_layers)
    num_single_layers = positive_int_or_none(num_single_layers)
    try:
        num_steps = len(timesteps)
    except Exception:
        return None
    num_steps = positive_int_or_none(num_steps)

    cfg_parallel_world_size = positive_int_or_none(cfg_parallel_world_size)
    tensor_parallel_size = positive_int_or_none(tensor_parallel_size)
    if cfg_parallel_world_size is None or tensor_parallel_size is None:
        return None
    forwards_per_step_per_gpu = 1
    if do_true_cfg and cfg_parallel_world_size == 1:
        forwards_per_step_per_gpu = 2

    if None in (batch_size, image_seq_len, text_seq_len, num_double_layers, num_single_layers, num_steps):
        return None

    try:
        return FluxDiTForwardStats(
            batch_size=batch_size,
            image_seq_len=image_seq_len,
            text_seq_len=text_seq_len,
            hidden_dim=hidden_dim,
            ffn_dim=4 * hidden_dim,
            num_double_layers=num_double_layers,
            num_single_layers=num_single_layers,
            num_steps=num_steps,
            forwards_per_step_per_gpu=forwards_per_step_per_gpu,
            tensor_parallel_size=tensor_parallel_size,
        )
    except (TypeError, ValueError):
        return None
