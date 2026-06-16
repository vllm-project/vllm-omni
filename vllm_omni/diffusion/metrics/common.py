from __future__ import annotations

import math
from typing import Any, TypeAlias

PatchSize3D: TypeAlias = tuple[int, int, int]


def require_non_negative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def require_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}")


def positive_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        int_value = int(value)
    except Exception:
        return None
    if int_value > 0:
        return int_value
    return None


def shape_dim(value: Any, index: int) -> int | None:
    try:
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) <= index:
            return None
        return positive_int_or_none(shape[index])
    except Exception:
        return None


def ceil_to_multiple(value: int, divisor: int) -> int:
    require_non_negative("value", value)
    require_positive("divisor", divisor)
    return math.ceil(value / divisor) * divisor if value else 0


def divide_flops_for_parallelism(value: int, degree: int) -> int:
    require_non_negative("value", value)
    require_positive("degree", degree)
    return math.ceil(value / degree)


def video_patch_seq_len(
    *,
    latent_num_frames: int,
    latent_height: int,
    latent_width: int,
    patch_size: PatchSize3D,
) -> int:
    for field_name, value in (
        ("latent_num_frames", latent_num_frames),
        ("latent_height", latent_height),
        ("latent_width", latent_width),
    ):
        require_non_negative(field_name, value)
    p_t, p_h, p_w = patch_size
    for field_name, value in (
        ("patch_size_t", p_t),
        ("patch_size_h", p_h),
        ("patch_size_w", p_w),
    ):
        require_positive(field_name, value)
    return (latent_num_frames // p_t) * (latent_height // p_h) * (latent_width // p_w)


def estimate_cross_attention_dit_layer_flops(
    *,
    query_seq_len: int,
    key_seq_len: int,
    context_len: int,
    hidden_dim: int,
    ffn_dim: int,
    tensor_parallel_size: int,
) -> int:
    qkv_out_flops = 8 * hidden_dim * hidden_dim * query_seq_len
    cross_attn_proj_flops = (4 * hidden_dim * hidden_dim * query_seq_len) + (4 * hidden_dim * hidden_dim * context_len)
    mlp_flops = 4 * hidden_dim * ffn_dim * query_seq_len
    self_attn_flops = 4 * query_seq_len * key_seq_len * hidden_dim
    cross_attn_flops = 4 * query_seq_len * context_len * hidden_dim
    return divide_flops_for_parallelism(
        qkv_out_flops + cross_attn_proj_flops + mlp_flops + self_attn_flops + cross_attn_flops,
        tensor_parallel_size,
    )
