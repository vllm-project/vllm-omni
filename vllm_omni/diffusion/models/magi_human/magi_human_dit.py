# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 SandAI. All Rights Reserved.
# Ported from daVinci-MagiHuman inference/model/dit/dit_module.py
# Adaptations: removed Ulysses context-parallelism, inlined Modality/VarlenHandler.

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import Parameter
from torch.nn import functional as F
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.vllm_flash_attn import flash_attn_varlen_func as _vllm_fa_varlen

try:
    from magi_compiler.api import magi_register_custom_op
    from magi_compiler.config import CompileConfig
except Exception:

    class CompileConfig:  # type: ignore[no-redef]
        pass

    def magi_register_custom_op(*args, **kwargs):  # type: ignore[no-redef]
        def decorator(func):
            return func

        return decorator


def magi_compile(*args, **kwargs):
    """No-op stub — vllm-omni handles execution; magi compilation is skipped."""

    def decorator(cls_or_fn):
        return cls_or_fn

    return decorator


# ---------------------------------------------------------------------------
# Inlined from inference/common/sequence_schema.py
# ---------------------------------------------------------------------------
class Modality(IntEnum):
    VIDEO = 0
    AUDIO = 1
    TEXT = 2


@dataclass
class VarlenHandler:
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int


def _is_hopper_arch() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 9


# ---------------------------------------------------------------------------
# FFA handler for local / flex attention
# ---------------------------------------------------------------------------
@dataclass
class FFAHandler:
    q_ranges: torch.Tensor
    k_ranges: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    attn_type_map: torch.Tensor
    softmax_scale: float


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------
class MLPActivationType(Enum):
    SWIGLU7 = "swiglu7"
    GELU7 = "gelu7"


def swiglu7(x, alpha: float = 1.702, limit: float = 7.0, out_dtype: torch.dtype | None = None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    x = x.to(torch.float32)
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return (out_glu * (x_linear + 1)).to(out_dtype)


def gelu7(x, alpha: float = 1.702, limit: float = 7.0, out_dtype: torch.dtype | None = None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    x = x.to(torch.float32)
    x_glu = x.clamp(min=None, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu.to(out_dtype)


def create_activation_func(activation_type: MLPActivationType) -> Callable:
    match activation_type:
        case MLPActivationType.SWIGLU7:
            return swiglu7
        case MLPActivationType.GELU7:
            return gelu7
        case _:
            raise ValueError(f"Unknown activation type: {activation_type}")


# ---------------------------------------------------------------------------
# Modality dispatcher (permutation helper)
# ---------------------------------------------------------------------------
class ModalityDispatcher:
    permuted_modality_mapping: torch.Tensor
    group_size: torch.Tensor
    group_size_cpu: list[int]
    num_modalities: int

    def __init__(self, modality_mapping: torch.Tensor, num_modalities: int):
        self.modality_mapping = modality_mapping
        self.num_modalities = num_modalities
        self.permuted_modality_mapping = self._precompute_permute_mapping(modality_mapping)
        self.group_size = torch.bincount(self.permuted_modality_mapping, minlength=num_modalities).to(torch.int32)
        self.group_size_cpu: list[int] = [int(x) for x in self.group_size.to("cpu").tolist()]

    def _precompute_permute_mapping(self, modality_mapping):
        self.permute_mapping = torch.argsort(modality_mapping)
        self.inv_permute_mapping = torch.argsort(self.permute_mapping)
        return modality_mapping[self.permute_mapping]

    def dispatch(self, x: torch.Tensor) -> list[torch.Tensor]:
        return list(torch.split(x, self.group_size_cpu, dim=0))

    def undispatch(self, *processed_groups: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(processed_groups, dim=0)

    @staticmethod
    def permute(x: torch.Tensor, permute_mapping: torch.Tensor) -> torch.Tensor:
        return x[permute_mapping]

    @staticmethod
    def inv_permute(x: torch.Tensor, inv_permute_mapping: torch.Tensor) -> torch.Tensor:
        return x[inv_permute_mapping]


# ---------------------------------------------------------------------------
# Positional / rotary embedding helpers
# ---------------------------------------------------------------------------
def freq_bands(
    num_bands: int, temperature: float = 10000.0, step: int = 2, device: torch.device | None = None
) -> torch.Tensor:
    exp = torch.arange(0, num_bands, step, dtype=torch.int64, device=device).to(torch.float32) / num_bands
    return 1.0 / (temperature**exp)


def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)


def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    return torch.cat([x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin, x[..., ro_dim:]], dim=-1)


# ---------------------------------------------------------------------------
# Fourier positional embedding
# ---------------------------------------------------------------------------
class ElementWiseFourierEmbed(nn.Module):
    def __init__(
        self,
        dim: int,
        max_res: int = 224,
        temperature: float = 10000.0,
        in_pixels: bool = True,
        linear_bands: bool = False,
        learnable: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dim = dim
        self.in_pixels = in_pixels
        self.learnable = learnable
        self.temperature = temperature
        self.max_res = max_res
        self.linear_bands = linear_bands
        self.device = device
        self.dtype = dtype
        bands = self.get_default_bands()
        self.bands = nn.Parameter(bands, requires_grad=self.learnable)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords_xyz = coords[:, :3]
        sizes = coords[:, 3:6]
        refs = coords[:, 6:9]

        scales = (refs - 1) / (sizes - 1)
        scales[(refs == 1) & (sizes == 1)] = 1
        assert not scales.isnan().any(), "scales has nan"
        assert not scales.isinf().any(), "scales has inf"

        centers = (sizes - 1) / 2
        centers[:, 0] = 0
        coords_xyz = coords_xyz - centers

        bands = self.bands.to(coords.device, coords.dtype)
        proj = coords_xyz.unsqueeze(-1) * scales.unsqueeze(-1) * bands
        sin_proj = proj.sin()
        cos_proj = proj.cos()
        return torch.cat((sin_proj, cos_proj), dim=1).flatten(1)

    def reset_parameters(self):
        self.bands.copy_(self.get_default_bands())

    def get_default_bands(self):
        if self.in_pixels:
            raise NotImplementedError("in_pixels are not implemented yet")
        return freq_bands(self.dim // 8, temperature=self.temperature, step=1, device=self.device).to(self.dtype)


# ---------------------------------------------------------------------------
# Multi-modality RMSNorm
# ---------------------------------------------------------------------------
class MultiModalityRMSNorm(nn.Module):
    __constants__ = ["dim", "eps", "num_modality"]

    def __init__(self, dim: int, eps: float = 1e-6, device: torch.device | None = None, num_modality: int = 1):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.num_modality = num_modality
        self.weight = nn.Parameter(torch.zeros(dim * num_modality, device=device, dtype=torch.float32))
        if num_modality > 1:
            self.forward = self.forward_multi_experts
        else:
            self.forward = self.forward_single_expert
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def rms(self, x: torch.Tensor) -> torch.Tensor:
        t = x.float()
        return t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)

    def forward_multi_experts(self, x: torch.Tensor, modality_dispatcher: ModalityDispatcher) -> torch.Tensor:
        original_dtype = x.dtype
        t = self.rms(x)
        weight_chunked = self.weight.chunk(self.num_modality, dim=0)
        t_list = modality_dispatcher.dispatch(t)
        for i in range(self.num_modality):
            t_list[i] = t_list[i] * (weight_chunked[i] + 1)
        t = modality_dispatcher.undispatch(*t_list)
        return t.to(original_dtype)

    def forward_single_expert(
        self, x: torch.Tensor, modality_dispatcher: ModalityDispatcher | None = None
    ) -> torch.Tensor:
        t, original_dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * (self.weight + 1)).to(original_dtype)


# ---------------------------------------------------------------------------
# Linear layers with bf16 compute and MoE dispatch
# ---------------------------------------------------------------------------
class _BF16ComputeLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        output_dtype: torch.dtype | None,
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        input_cast = input.to(compute_dtype)
        weight_cast = weight.to(compute_dtype)
        output = torch.matmul(input_cast, weight_cast.t())
        if bias is not None:
            output = output + bias.to(compute_dtype)
        return output.to(output_dtype)


class BaseLinear(nn.Module):
    __constants__ = ["in_features", "out_features", "num_layers", "num_experts"]

    def __init__(
        self, in_features, out_features, num_layers_for_initialization, num_experts, bias=True, device=None, dtype=None
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": torch.bfloat16}
        self.in_features = in_features
        self.out_features = out_features
        self.num_layers_for_initialization = num_layers_for_initialization
        self.num_experts = num_experts
        self.use_bias = bias
        self.weight = Parameter(torch.empty((out_features * num_experts, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features * num_experts, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        input: torch.Tensor,
        output_dtype: torch.dtype | None = None,
        modality_dispatcher: ModalityDispatcher | None = None,
    ) -> torch.Tensor:
        output_dtype = input.dtype if output_dtype is None else output_dtype
        return _BF16ComputeLinear.apply(input, self.weight, self.bias, output_dtype, torch.bfloat16)


class NativeMoELinear(BaseLinear):
    def forward(
        self,
        input: torch.Tensor,
        output_dtype: torch.dtype | None = None,
        modality_dispatcher: ModalityDispatcher | None = None,
    ) -> torch.Tensor:
        output_dtype = input.dtype if output_dtype is None else output_dtype
        input_list = modality_dispatcher.dispatch(input)  # type: ignore
        weight_chunked = self.weight.chunk(self.num_experts, dim=0)
        if self.bias is not None:
            bias_chunked = self.bias.chunk(self.num_experts, dim=0)
        for i in range(self.num_experts):
            input_list[i] = _BF16ComputeLinear.apply(
                input_list[i],
                weight_chunked[i],
                bias_chunked[i] if self.bias is not None else None,
                output_dtype,
                torch.bfloat16,
            )
        return modality_dispatcher.undispatch(*input_list)  # type: ignore


def create_linear(
    in_features, out_features, num_layers=1, num_experts=1, bias=True, device=None, dtype=None
) -> BaseLinear | NativeMoELinear:
    if num_experts == 1:
        return BaseLinear(in_features, out_features, num_layers, num_experts, bias, device, dtype)
    else:
        return NativeMoELinear(in_features, out_features, num_layers, num_experts, bias, device, dtype)


# ---------------------------------------------------------------------------
# MoE TP parallel linear wrappers: per-expert vLLM parallel layers
# ---------------------------------------------------------------------------
class MoEQKVParallelLinear(nn.Module):
    """Per-expert QKVParallelLinear with modality dispatch.

    Wraps ``num_experts`` independent QKVParallelLinear instances.
    Forward: dispatch tokens by modality → per-expert QKV matmul (TP-sharded)
    → undispatch.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        num_experts: int,
        bias: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [
                QKVParallelLinear(
                    hidden_size=hidden_size,
                    head_size=head_size,
                    total_num_heads=total_num_heads,
                    total_num_kv_heads=total_num_kv_heads,
                    bias=bias,
                    return_bias=False,
                )
                for _ in range(num_experts)
            ]
        )
        # Expose per-rank head info from the first expert (all are identical).
        self.num_heads = self.experts[0].num_heads
        self.num_kv_heads = self.experts[0].num_kv_heads
        self.head_size = head_size

    def forward(
        self,
        x: torch.Tensor,
        modality_dispatcher: ModalityDispatcher,
    ) -> torch.Tensor:
        x_list = modality_dispatcher.dispatch(x)
        out_list: list[torch.Tensor] = []
        for i in range(self.num_experts):
            out = self.experts[i](x_list[i])
            out_list.append(out)
        return modality_dispatcher.undispatch(*out_list)


class MoEColumnParallelLinear(nn.Module):
    """Per-expert ColumnParallelLinear with modality dispatch.

    Forward: dispatch → per-expert column-parallel matmul → undispatch.
    Output stays TP-local (no gather).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_experts: int,
        bias: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [
                ColumnParallelLinear(
                    input_size=input_size,
                    output_size=output_size,
                    bias=bias,
                    gather_output=False,
                    return_bias=False,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        modality_dispatcher: ModalityDispatcher,
    ) -> torch.Tensor:
        x_list = modality_dispatcher.dispatch(x)
        out_list: list[torch.Tensor] = []
        for i in range(self.num_experts):
            out = self.experts[i](x_list[i])
            out_list.append(out)
        return modality_dispatcher.undispatch(*out_list)


class MoERowParallelLinear(nn.Module):
    """Per-expert RowParallelLinear with modality dispatch.

    Forward: dispatch → per-expert row-parallel matmul (includes all-reduce)
    → undispatch.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_experts: int,
        bias: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [
                RowParallelLinear(
                    input_size=input_size,
                    output_size=output_size,
                    bias=bias,
                    input_is_parallel=True,
                    return_bias=False,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        modality_dispatcher: ModalityDispatcher,
    ) -> torch.Tensor:
        x_list = modality_dispatcher.dispatch(x)
        out_list: list[torch.Tensor] = []
        for i in range(self.num_experts):
            out = self.experts[i](x_list[i])
            out_list.append(out)
        return modality_dispatcher.undispatch(*out_list)


def validate_magi_human_tp_constraints(
    *,
    hidden_size: int,
    num_heads_q: int,
    num_heads_kv: int,
    tensor_parallel_size: int,
) -> None:
    """Validate MagiHuman TP divisibility constraints.

    Both shared layers (num_modality == 1) and MoE layers (num_modality == 3)
    support TP via vLLM's parallel linear layers (QKVParallelLinear /
    ColumnParallelLinear / RowParallelLinear).  MoE layers use per-expert
    parallel layers with modality dispatch.

    Supported tp_sizes given default config (hidden=5120, heads_q=40, kv=8): 1, 2, 4.
    """
    tp = tensor_parallel_size
    if tp <= 1:
        return
    errors: list[str] = []
    if num_heads_q % tp != 0:
        errors.append(f"num_heads_q ({num_heads_q}) must be divisible by tensor_parallel_size ({tp})")
    if num_heads_kv % tp != 0:
        errors.append(f"num_heads_kv ({num_heads_kv}) must be divisible by tensor_parallel_size ({tp})")
    # SWIGLU layers use intermediate = int(hidden * 8/3) // 4 * 4
    intermediate_swiglu = int(hidden_size * 4 * 2 / 3) // 4 * 4
    if intermediate_swiglu % tp != 0:
        errors.append(
            f"swiglu intermediate_size ({intermediate_swiglu}) must be divisible by "
            f"tensor_parallel_size ({tp}). Supported tp values: 1, 2, 4"
        )
    # GELU7 MoE layers use intermediate = hidden * 4
    intermediate_gelu = hidden_size * 4
    if intermediate_gelu % tp != 0:
        errors.append(f"gelu intermediate_size ({intermediate_gelu}) must be divisible by tensor_parallel_size ({tp})")
    if errors:
        raise ValueError("MagiHuman TP constraint violations:\n" + "\n".join(f"  - {e}" for e in errors))


# ---------------------------------------------------------------------------
# Flash attention (no context-parallelism) — uses vllm's flash attention
# ---------------------------------------------------------------------------

HAS_MAGI_ATTENTION = importlib.util.find_spec("magi_attention") is not None


def _fa_varlen_simple(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    had_batch = query.ndim == 4
    if had_batch:
        query = query.squeeze(0)
        key = key.squeeze(0)
        value = value.squeeze(0)
    seq_len = query.shape[0]
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=query.device)
    out = _vllm_fa_varlen(
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
    )
    if had_batch:
        out = out.unsqueeze(0)
    return out


@magi_register_custom_op(name="infra::flash_attn_func", is_subgraph_boundary=True)
def flash_attn_func(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return _fa_varlen_simple(query, key, value)


def _split_q_range_with_no_overlap(
    q_ranges: torch.Tensor, k_ranges: torch.Tensor
) -> tuple[list[list[int]], list[list[list[int]]]]:
    range_boundary = torch.unique(q_ranges, sorted=True).tolist()
    candidates = [[start, end, []] for start, end in zip(range_boundary[:-1], range_boundary[1:])]
    q_ranges = q_ranges.tolist()
    k_ranges = k_ranges.tolist()
    for q_range, k_range in zip(q_ranges, k_ranges):
        q_start, q_end = q_range
        for q_range_cand in candidates:
            if q_start <= q_range_cand[0] and q_range_cand[1] <= q_end:
                q_range_cand[2].append(k_range)
    q_ranges_out = []
    k_ranges_out = []
    for q_range_cand in candidates:
        if len(q_range_cand[2]) > 0:
            q_ranges_out.append(q_range_cand[0:2])
            k_ranges_out.append(q_range_cand[2])
    return q_ranges_out, k_ranges_out


def _flash_attn_with_correction(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_ranges: list[list[int]],
    k_range_list: list[list[list[int]]],
):
    output = torch.zeros_like(query)
    output_lse = torch.zeros((query.shape[0], query.shape[1]), dtype=torch.float32, device=query.device)

    for q_range, k_ranges in zip(q_ranges, k_range_list):
        q_start, q_end = q_range
        q_chunk = query[q_start:q_end]
        q_len = q_chunk.shape[0]

        # Concatenate all k_ranges into a single key/value block, then run one
        # flash-attention call.  This avoids the need to merge per-chunk LSEs.
        k_parts = [key[ks:ke] for ks, ke in k_ranges]
        v_parts = [value[ks:ke] for ks, ke in k_ranges]
        k_combined = torch.cat(k_parts, dim=0) if len(k_parts) > 1 else k_parts[0]
        v_combined = torch.cat(v_parts, dim=0) if len(v_parts) > 1 else v_parts[0]
        k_len = k_combined.shape[0]

        cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=query.device)
        cu_k = torch.tensor([0, k_len], dtype=torch.int32, device=query.device)
        qo_out = _vllm_fa_varlen(
            q=q_chunk,
            k=k_combined,
            v=v_combined,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=q_len,
            max_seqlen_k=k_len,
        )
        output[q_start:q_end] = qo_out
    return output, output_lse


def _flex_flash_attn_func_infer_output_meta(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, q_ranges: torch.Tensor, k_ranges: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(query)
    output_lse = torch.empty((query.shape[0], query.shape[1]), dtype=torch.float32, device=query.device)
    return output, output_lse


@magi_register_custom_op(
    name="infra::flex_flash_attn_func",
    mutates_args=(),
    infer_output_meta_fn=_flex_flash_attn_func_infer_output_meta,
    is_subgraph_boundary=True,
)
def flex_flash_attn_func(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, q_ranges: torch.Tensor, k_ranges: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if HAS_MAGI_ATTENTION and _is_hopper_arch():
        from magi_attention.api import flex_flash_attn_func as magi_flex_flash_attn_func

        return magi_flex_flash_attn_func(query, key, value, q_ranges, k_ranges)
    else:
        q_ranges_split, k_range_list = _split_q_range_with_no_overlap(q_ranges, k_ranges)
        return _flash_attn_with_correction(query, key, value, q_ranges_split, k_range_list)


def flash_attn_no_cp(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
    return flash_attn_func(q, k, v).squeeze(0)


def flex_flash_attn_no_cp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
) -> torch.Tensor:
    q, k, v = q.to(torch.bfloat16).squeeze(0), k.to(torch.bfloat16).squeeze(0), v.to(torch.bfloat16).squeeze(0)
    out, _ = flex_flash_attn_func(q, k, v, q_ranges=q_ranges, k_ranges=k_ranges)
    return out


# ---------------------------------------------------------------------------
# Attention module (no context-parallelism)
# ---------------------------------------------------------------------------
@dataclass
class AttentionConfig:
    hidden_size: int
    num_heads_q: int
    num_heads_kv: int
    head_dim: int
    params_dtype: torch.dtype
    checkpoint_qk_layernorm_rope: bool
    num_modality: int
    num_layers: int
    use_local_attn: bool = False
    enable_attn_gating: bool = False


class Attention(torch.nn.Module):
    config: AttentionConfig

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.pre_norm = MultiModalityRMSNorm(config.hidden_size, eps=1e-6, num_modality=config.num_modality)
        self.gating_size = config.num_heads_q if config.enable_attn_gating else 0

        # Both shared blocks (num_modality == 1) and MoE blocks (num_modality > 1)
        # use vLLM's parallel linear layers for TP support.
        # MoE blocks wrap per-expert parallel layers with modality dispatch.
        if config.num_modality == 1:
            # QKVParallelLinear handles GQA head-sharding for any tp_size.
            # The combined checkpoint weight [Q, K, V, G] is split during
            # load_weights: Q+K+V → linear_qkv, G → linear_gating.
            self.linear_qkv = QKVParallelLinear(
                hidden_size=config.hidden_size,
                head_size=config.head_dim,
                total_num_heads=config.num_heads_q,
                total_num_kv_heads=config.num_heads_kv,
                bias=False,
                return_bias=False,
            )
            self.linear_proj = RowParallelLinear(
                input_size=config.num_heads_q * config.head_dim,
                output_size=config.hidden_size,
                bias=False,
                input_is_parallel=True,
                return_bias=False,
            )
            if config.enable_attn_gating:
                self.linear_gating = ColumnParallelLinear(
                    input_size=config.hidden_size,
                    output_size=config.num_heads_q,
                    bias=False,
                    gather_output=False,
                    return_bias=False,
                )
            else:
                self.linear_gating = None
        else:
            # MoE blocks: per-expert TP-sharded parallel layers.
            self.linear_qkv = MoEQKVParallelLinear(
                hidden_size=config.hidden_size,
                head_size=config.head_dim,
                total_num_heads=config.num_heads_q,
                total_num_kv_heads=config.num_heads_kv,
                num_experts=config.num_modality,
                bias=False,
            )
            self.linear_proj = MoERowParallelLinear(
                input_size=config.num_heads_q * config.head_dim,
                output_size=config.hidden_size,
                num_experts=config.num_modality,
                bias=False,
            )
            if config.enable_attn_gating:
                self.linear_gating = MoEColumnParallelLinear(
                    input_size=config.hidden_size,
                    output_size=config.num_heads_q,
                    num_experts=config.num_modality,
                    bias=False,
                )
            else:
                self.linear_gating = None

        self.q_norm = MultiModalityRMSNorm(config.head_dim, num_modality=config.num_modality)
        self.k_norm = MultiModalityRMSNorm(config.head_dim, num_modality=config.num_modality)

        # q_size / kv_size reflect the per-rank head count when tp > 1.
        # Both shared and MoE QKV layers expose .num_heads / .num_kv_heads.
        if config.num_modality == 1:
            self.q_size = self.linear_qkv.num_heads * config.head_dim
            self.kv_size = self.linear_qkv.num_kv_heads * config.head_dim
            self._local_heads_q = self.linear_qkv.num_heads
            self._local_heads_kv = self.linear_qkv.num_kv_heads
        else:
            self.q_size = self.linear_qkv.num_heads * config.head_dim
            self.kv_size = self.linear_qkv.num_kv_heads * config.head_dim
            self._local_heads_q = self.linear_qkv.num_heads
            self._local_heads_kv = self.linear_qkv.num_kv_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        permute_mapping: torch.Tensor,
        inv_permute_mapping: torch.Tensor,
        varlen_handler: VarlenHandler,
        local_attn_handler: FFAHandler | None,
        modality_dispatcher: ModalityDispatcher,
    ) -> torch.Tensor:
        hidden_states = self.pre_norm(hidden_states, modality_dispatcher=modality_dispatcher).to(torch.bfloat16)

        if self.config.num_modality == 1:
            # vLLM parallel layers with return_bias=False return a single tensor.
            qkv = self.linear_qkv(hidden_states).to(torch.float32)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            if self.linear_gating is not None:
                g = self.linear_gating(hidden_states).to(torch.float32)
            else:
                g = hidden_states.new_empty(hidden_states.shape[0], 0)
        else:
            # MoE TP path: per-expert QKV parallel layers.
            qkv = self.linear_qkv(hidden_states, modality_dispatcher=modality_dispatcher).to(torch.float32)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            if self.linear_gating is not None:
                g = self.linear_gating(hidden_states, modality_dispatcher=modality_dispatcher).to(torch.float32)
            else:
                g = hidden_states.new_empty(hidden_states.shape[0], 0)

        q = q.view(-1, self._local_heads_q, self.config.head_dim)
        k = k.view(-1, self._local_heads_kv, self.config.head_dim)
        v = v.view(-1, self._local_heads_kv, self.config.head_dim)
        g = g.view(k.shape[0], self._local_heads_q, -1)

        q = self.q_norm(q, modality_dispatcher=modality_dispatcher)
        k = self.k_norm(k, modality_dispatcher=modality_dispatcher)

        q = ModalityDispatcher.inv_permute(q, inv_permute_mapping).unsqueeze(0)
        k = ModalityDispatcher.inv_permute(k, inv_permute_mapping).unsqueeze(0)
        v = ModalityDispatcher.inv_permute(v, inv_permute_mapping).unsqueeze(0)

        sin_emb, cos_emb = rope.tensor_split(2, -1)
        q = apply_rotary_emb_torch(q, cos_emb, sin_emb)
        k = apply_rotary_emb_torch(k, cos_emb, sin_emb)

        if self.config.use_local_attn and local_attn_handler is not None:
            self_attn_out = flex_flash_attn_no_cp(q, k, v, local_attn_handler.q_ranges, local_attn_handler.k_ranges)
        else:
            self_attn_out = flash_attn_no_cp(q, k, v)
        self_attn_out = ModalityDispatcher.permute(self_attn_out, permute_mapping)

        if self.config.enable_attn_gating:
            self_attn_out = self_attn_out * torch.sigmoid(g)

        self_attn_out = self_attn_out.view(-1, self._local_heads_q * self.config.head_dim).to(torch.bfloat16)
        if self.config.num_modality == 1:
            return self.linear_proj(self_attn_out)
        return self.linear_proj(self_attn_out, modality_dispatcher=modality_dispatcher)


# ---------------------------------------------------------------------------
# MLP module
# ---------------------------------------------------------------------------
@dataclass
class MLPConfig:
    hidden_size: int
    intermediate_size: int
    activation_type: MLPActivationType
    params_dtype: torch.dtype
    num_modality: int = 1
    num_layers: int = 1
    gated_act: bool = False


class MLP(torch.nn.Module):
    config: MLPConfig

    def __init__(self, config: MLPConfig):
        super().__init__()
        num_experts = config.num_modality
        self.pre_norm = MultiModalityRMSNorm(config.hidden_size, num_modality=config.num_modality)
        intermediate_size_up = config.intermediate_size * 2 if config.gated_act else config.intermediate_size

        # Both shared blocks (num_experts == 1) and MoE blocks (num_experts > 1)
        # use vLLM's parallel linear layers for TP support.
        if num_experts == 1:
            # ColumnParallelLinear shards the output dim uniformly.  For
            # SWIGLU7 the interleaved [up0, gate0, up1, gate1, ...] format
            # is preserved within each rank's contiguous slice, so swiglu7
            # (which uses x[..., ::2] / x[..., 1::2]) still works correctly.
            self.up_gate_proj = ColumnParallelLinear(
                input_size=config.hidden_size,
                output_size=intermediate_size_up,
                bias=False,
                gather_output=False,
                return_bias=False,
            )
            self.down_proj = RowParallelLinear(
                input_size=config.intermediate_size,
                output_size=config.hidden_size,
                bias=False,
                input_is_parallel=True,
                return_bias=False,
            )
        else:
            # MoE blocks: per-expert TP-sharded parallel layers.
            self.up_gate_proj = MoEColumnParallelLinear(
                input_size=config.hidden_size,
                output_size=intermediate_size_up,
                num_experts=num_experts,
                bias=False,
            )
            self.down_proj = MoERowParallelLinear(
                input_size=config.intermediate_size,
                output_size=config.hidden_size,
                num_experts=num_experts,
                bias=False,
            )
        self.activation_func = create_activation_func(config.activation_type)

    def forward(self, x: torch.Tensor, modality_dispatcher: ModalityDispatcher) -> torch.Tensor:
        x = self.pre_norm(x, modality_dispatcher=modality_dispatcher).to(torch.bfloat16)
        if isinstance(self.up_gate_proj, ColumnParallelLinear):
            x = self.up_gate_proj(x).to(torch.float32)
            x = self.activation_func(x).to(torch.bfloat16)
            return self.down_proj(x).to(torch.float32)
        # MoE TP path: per-expert column/row parallel layers.
        x = self.up_gate_proj(x, modality_dispatcher=modality_dispatcher).to(torch.float32)
        x = self.activation_func(x).to(torch.bfloat16)
        x = self.down_proj(x, modality_dispatcher=modality_dispatcher).to(torch.float32)
        return x


# ---------------------------------------------------------------------------
# Adapter (per-modality embedders + RoPE)
# ---------------------------------------------------------------------------
@dataclass
class AdapterConfig:
    hidden_size: int
    num_attention_heads: int
    text_in_channels: int
    video_in_channels: int
    audio_in_channels: int
    params_dtype: torch.dtype


class Adapter(torch.nn.Module):
    config: AdapterConfig

    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config
        self.video_embedder = nn.Linear(config.video_in_channels, config.hidden_size, bias=True, dtype=torch.float32)
        self.text_embedder = nn.Linear(config.text_in_channels, config.hidden_size, bias=True, dtype=torch.float32)
        self.audio_embedder = nn.Linear(config.audio_in_channels, config.hidden_size, bias=True, dtype=torch.float32)
        self.rope = ElementWiseFourierEmbed(
            config.hidden_size // config.num_attention_heads, in_pixels=False, learnable=False
        )

    def forward(self, x, coords_mapping, video_mask, audio_mask, text_mask):
        rope = self.rope(coords_mapping)

        text_input = x[text_mask, : self.config.text_in_channels]
        audio_input = x[audio_mask, : self.config.audio_in_channels]
        video_input = x[video_mask, : self.config.video_in_channels]

        text_out = self.text_embedder(text_input)
        audio_out = self.audio_embedder(audio_input)
        video_out = self.video_embedder(video_input)

        output_x = torch.zeros(x.shape[0], self.config.hidden_size, device=x.device, dtype=x.dtype)
        output_x[text_mask] = text_out
        output_x[audio_mask] = audio_out
        output_x[video_mask] = video_out
        return output_x, rope


# ---------------------------------------------------------------------------
# Transformer layer (no CP)
# ---------------------------------------------------------------------------
class TransFormerLayer(torch.nn.Module):
    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        num_modality = 3 if layer_idx in config.mm_layers else 1
        use_local_attn = layer_idx in config.local_attn_layers
        self.post_norm = layer_idx in config.post_norm_layers
        attention_config = AttentionConfig(
            hidden_size=config.hidden_size,
            num_heads_q=config.num_heads_q,
            num_heads_kv=config.num_heads_kv,
            head_dim=config.head_dim,
            params_dtype=config.params_dtype,
            checkpoint_qk_layernorm_rope=config.checkpoint_qk_layernorm_rope,
            num_modality=num_modality,
            num_layers=config.num_layers,
            use_local_attn=use_local_attn,
            enable_attn_gating=config.enable_attn_gating,
        )
        self.attention: Attention = Attention(attention_config)

        activation_type = MLPActivationType.GELU7 if layer_idx in config.gelu7_layers else MLPActivationType.SWIGLU7
        if activation_type == MLPActivationType.SWIGLU7:
            gated_act = True
            intermediate_size = int(config.hidden_size * 4 * 2 / 3) // 4 * 4
        else:
            gated_act = False
            intermediate_size = config.hidden_size * 4
        mlp_config = MLPConfig(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            activation_type=activation_type,
            params_dtype=config.params_dtype,
            num_modality=num_modality,
            num_layers=config.num_layers,
            gated_act=gated_act,
        )
        self.mlp: MLP = MLP(mlp_config)
        if self.post_norm:
            self.attn_post_norm = MultiModalityRMSNorm(config.hidden_size, num_modality=num_modality)
            self.mlp_post_norm = MultiModalityRMSNorm(config.hidden_size, num_modality=num_modality)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        permute_mapping: torch.Tensor,
        inv_permute_mapping: torch.Tensor,
        varlen_handler: VarlenHandler,
        local_attn_handler: FFAHandler | None,
        modality_dispatcher: ModalityDispatcher,
    ) -> torch.Tensor:
        attn_out = self.attention(
            hidden_states,
            rope,
            permute_mapping,
            inv_permute_mapping,
            varlen_handler,
            local_attn_handler,
            modality_dispatcher,
        )
        if self.post_norm:
            attn_out = self.attn_post_norm(attn_out, modality_dispatcher=modality_dispatcher)
        hidden_states = hidden_states + attn_out

        mlp_out = self.mlp(hidden_states, modality_dispatcher)
        if self.post_norm:
            mlp_out = self.mlp_post_norm(mlp_out, modality_dispatcher=modality_dispatcher)
        hidden_states = hidden_states + mlp_out
        return hidden_states


# ---------------------------------------------------------------------------
# TransformerBlock with magi_compile
# ---------------------------------------------------------------------------
is_base_model = True


def config_patch(compile_config: CompileConfig) -> CompileConfig:
    global is_base_model
    if is_base_model:
        is_base_model = False
    else:
        compile_config.offload_config.gpu_resident_weight_ratio = 0.0
    return compile_config


@magi_compile(
    config_patch=config_patch, dynamic_arg_dims={"x": 0, "rope": 0, "permute_mapping": 0, "inv_permute_mapping": 0}
)
class TransformerBlock(torch.nn.Module):
    def __init__(self, model_config: Any):
        super().__init__()
        self.layers: list[TransFormerLayer] = nn.ModuleList()
        for layer_idx in range(model_config.num_layers):
            self.layers.append(TransFormerLayer(model_config, layer_idx))

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        permute_mapping: torch.Tensor,
        inv_permute_mapping: torch.Tensor,
        varlen_handler: VarlenHandler,
        local_attn_handler: FFAHandler | None,
        modality_dispatcher: ModalityDispatcher,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x, rope, permute_mapping, inv_permute_mapping, varlen_handler, local_attn_handler, modality_dispatcher
            )
        return x


# ---------------------------------------------------------------------------
# Internal config for TransformerBlock / DiTModel construction
# ---------------------------------------------------------------------------
@dataclass
class TransformerConfig:
    hidden_size: int
    video_in_channels: int
    audio_in_channels: int
    text_in_channels: int
    params_dtype: torch.dtype
    post_process_dtype: torch.dtype


# ---------------------------------------------------------------------------
# DiTModel (no context-parallelism)
# ---------------------------------------------------------------------------
class DiTModel(torch.nn.Module):
    config: TransformerConfig
    _layerwise_offload_blocks_attr = "blocks"

    @property
    def blocks(self) -> nn.ModuleList:
        return self.block.layers

    def __init__(self, model_config: Any):
        super().__init__()
        validate_magi_human_tp_constraints(
            hidden_size=model_config.hidden_size,
            num_heads_q=model_config.hidden_size // model_config.head_dim,
            num_heads_kv=model_config.num_query_groups,
            tensor_parallel_size=get_tensor_model_parallel_world_size(),
        )
        self.config = TransformerConfig(
            hidden_size=model_config.hidden_size,
            video_in_channels=model_config.video_in_channels,
            audio_in_channels=model_config.audio_in_channels,
            text_in_channels=model_config.text_in_channels,
            params_dtype=model_config.params_dtype,
            post_process_dtype=torch.float32,
        )
        adapter_config = AdapterConfig(
            hidden_size=model_config.hidden_size,
            num_attention_heads=model_config.num_heads_q,
            text_in_channels=model_config.text_in_channels,
            video_in_channels=model_config.video_in_channels,
            audio_in_channels=model_config.audio_in_channels,
            params_dtype=torch.float32,
        )
        self.adapter: Adapter = Adapter(adapter_config)
        self.block: TransformerBlock = TransformerBlock(model_config=model_config)
        self.final_norm_video = MultiModalityRMSNorm(self.config.hidden_size)
        self.final_norm_audio = MultiModalityRMSNorm(self.config.hidden_size)
        self.final_linear_video = nn.Linear(
            self.config.hidden_size, self.config.video_in_channels, bias=False, dtype=torch.float32
        )
        self.final_linear_audio = nn.Linear(
            self.config.hidden_size, self.config.audio_in_channels, bias=False, dtype=torch.float32
        )

    def forward(
        self,
        x: torch.Tensor,
        coords_mapping: torch.Tensor,
        modality_mapping: torch.Tensor,
        varlen_handler: VarlenHandler,
        local_attn_handler: FFAHandler | None,
    ):
        modality_dispatcher = ModalityDispatcher(modality_mapping, 3)
        permute_mapping = modality_dispatcher.permute_mapping
        inv_permute_mapping = modality_dispatcher.inv_permute_mapping
        video_mask = modality_mapping == Modality.VIDEO
        audio_mask = modality_mapping == Modality.AUDIO
        text_mask = modality_mapping == Modality.TEXT

        x, rope = self.adapter(x, coords_mapping, video_mask, audio_mask, text_mask)

        x = x.to(self.config.params_dtype)
        x = ModalityDispatcher.permute(x, permute_mapping)

        x = self.block(
            x,
            rope,
            permute_mapping=permute_mapping,
            inv_permute_mapping=inv_permute_mapping,
            varlen_handler=varlen_handler,
            local_attn_handler=local_attn_handler,
            modality_dispatcher=modality_dispatcher,
        )

        x = ModalityDispatcher.inv_permute(x, inv_permute_mapping)

        x_video = x[video_mask].to(self.final_norm_video.weight.dtype)
        x_video = self.final_norm_video(x_video)
        x_video = self.final_linear_video(x_video)

        x_audio = x[audio_mask].to(self.final_norm_audio.weight.dtype)
        x_audio = self.final_norm_audio(x_audio)
        x_audio = self.final_linear_audio(x_audio)

        x_out = torch.zeros(
            x.shape[0],
            max(self.config.video_in_channels, self.config.audio_in_channels),
            device=x.device,
            dtype=x.dtype,
        )
        x_out[video_mask, : self.config.video_in_channels] = x_video
        x_out[audio_mask, : self.config.audio_in_channels] = x_audio

        return x_out


# ---------------------------------------------------------------------------
# Public config dataclass for building DiTModel from JSON
# ---------------------------------------------------------------------------
@dataclass
class MagiHumanDiTConfig:
    num_layers: int = 40
    hidden_size: int = 5120
    head_dim: int = 128
    num_query_groups: int = 8
    video_in_channels: int = 48 * 4
    audio_in_channels: int = 64
    text_in_channels: int = 3584
    checkpoint_qk_layernorm_rope: bool = False
    params_dtype: torch.dtype = torch.float32
    mm_layers: list = field(default_factory=lambda: [0, 1, 2, 3, 36, 37, 38, 39])
    local_attn_layers: list = field(default_factory=list)
    enable_attn_gating: bool = True
    gelu7_layers: list = field(default_factory=lambda: [0, 1, 2, 3])
    post_norm_layers: list = field(default_factory=list)

    def __post_init__(self):
        self.num_heads_q = self.hidden_size // self.head_dim
        self.num_heads_kv = self.num_query_groups


if TYPE_CHECKING:
    from .pipeline_magi_human import EvalInput


# ===========================================================================
# Data proxy (ported from daVinci-MagiHuman inference/pipeline/data_proxy.py)
# ===========================================================================
def _unfold_3d(
    x: torch.Tensor,
    kernel_size: tuple[int, int, int],
    stride: tuple[int, int, int],
) -> torch.Tensor:
    """Pure-PyTorch 3D unfold matching UnfoldAnd behavior.

    After N unfold ops the shape is (batch, C, oD, oH, oW, kD, kH, kW).
    UnfoldAnd permutes kernel dims next to channel before reshape so that the
    col_dim axis is ordered as (C, kD, kH, kW) -- matching F.unfold semantics.
    Without this permute, .view() interleaves spatial and kernel positions.

    Args:
        x: (N, C, D, H, W)
        kernel_size: (kD, kH, kW)
        stride: (sD, sH, sW)
    Returns:
        (N, C*kD*kH*kW, L) where L = product of output spatial dims.
    """
    ndim = len(kernel_size)
    for d in range(ndim):
        x = x.unfold(d + 2, kernel_size[d], stride[d])
    # x: (N, C, oD, oH, oW, kD, kH, kW)
    # Permute to (N, C, kD, kH, kW, oD, oH, oW) so that view groups correctly
    perm = [0, 1] + list(range(ndim + 2, 2 * ndim + 2)) + list(range(2, ndim + 2))
    x = x.permute(*perm).contiguous()

    batch_size = x.shape[0]
    col_dim = 1
    for i in range(1, ndim + 2):
        col_dim *= x.shape[i]
    spatial = 1
    for i in range(ndim + 2, 2 * ndim + 2):
        spatial *= x.shape[i]
    return x.view(batch_size, col_dim, spatial)


def calc_local_qk_range(
    num_video_tokens,
    num_audio_and_txt_tokens,
    num_frames,
    frame_receptive_field,
):
    token_per_frame = num_video_tokens // num_frames
    total_tokens = num_video_tokens + num_audio_and_txt_tokens

    q_range_list = []
    k_range_list = []
    for i in range(num_frames):
        q_range_list.append(torch.tensor([i * token_per_frame, (i + 1) * token_per_frame]))
        k_range_list.append(
            torch.tensor(
                [
                    (i - frame_receptive_field) * token_per_frame,
                    (i + frame_receptive_field + 1) * token_per_frame,
                ]
            )
        )
    local_q_range = torch.stack(q_range_list, dim=0)
    local_k_range = torch.stack(k_range_list, dim=0)

    local_k_range[local_k_range < 0] = 0
    local_k_range[local_k_range > num_video_tokens] = num_video_tokens

    video_q_range = torch.tensor([[0, num_video_tokens]])
    video_k_range = torch.tensor([[num_video_tokens, num_video_tokens + num_audio_and_txt_tokens]])

    at_q_ranges = torch.tensor([[num_video_tokens, total_tokens]])
    at_k_ranges = torch.tensor([[0, total_tokens]])

    q_ranges = (
        torch.cat([local_q_range, video_q_range, at_q_ranges], dim=0).to(torch.int32).to("cuda", non_blocking=True)
    )
    k_ranges = (
        torch.cat([local_k_range, video_k_range, at_k_ranges], dim=0).to(torch.int32).to("cuda", non_blocking=True)
    )
    return q_ranges, k_ranges


def calc_local_attn_ffa_handler(
    num_video_tokens,
    num_audio_and_txt_tokens,
    num_frames,
    frame_receptive_field,
):
    q_ranges, k_ranges = calc_local_qk_range(
        num_video_tokens,
        num_audio_and_txt_tokens,
        num_frames,
        frame_receptive_field,
    )
    total = num_video_tokens + num_audio_and_txt_tokens
    return FFAHandler(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        max_seqlen_q=total,
        max_seqlen_k=total,
        attn_type_map=torch.zeros([q_ranges.shape[0]], device="cuda", dtype=torch.int32),
        softmax_scale=None,
    )


def get_coords(
    shape: list[int],
    ref_feat_shape: list[int],
    offset_thw: list[int] | None = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
):
    if offset_thw is None:
        offset_thw = [0, 0, 0]
    ori_t, ori_h, ori_w = shape
    ref_t, ref_h, ref_w = ref_feat_shape

    offset_t, offset_h, offset_w = offset_thw
    time_rng = torch.arange(ori_t, device=device, dtype=dtype) + offset_t
    height_rng = torch.arange(ori_h, device=device, dtype=dtype) + offset_h
    width_rng = torch.arange(ori_w, device=device, dtype=dtype) + offset_w

    time_grid, height_grid, width_grid = torch.meshgrid(
        time_rng,
        height_rng,
        width_rng,
        indexing="ij",
    )
    coords_flat = torch.stack([time_grid, height_grid, width_grid], dim=-1).reshape(-1, 3)

    meta = torch.tensor(
        [ori_t, ori_h, ori_w, ref_t, ref_h, ref_w],
        device=device,
        dtype=dtype,
    )
    meta_expanded = meta.expand(coords_flat.size(0), -1)
    return torch.cat([coords_flat, meta_expanded], dim=-1)


@dataclass
class SingleData:
    video_x_t: torch.Tensor
    audio_x_t: torch.Tensor
    audio_feat_len: int
    txt_feat: torch.Tensor
    txt_feat_len: int
    t: int
    h: int
    w: int
    patch_size: int
    t_patch_size: int
    spatial_rope_interpolation: Literal["inter", "extra"]
    ref_audio_offset: int
    text_offset: int
    coords_style: Literal["v1", "v2"] = "v1"

    def __post_init__(self):
        self.video_token_num = self.video_x_t.shape[0]
        self.audio_x_t = self.audio_x_t[: self.audio_feat_len]
        self.txt_feat = self.txt_feat[: self.txt_feat_len]
        self.video_channel = self.video_x_t.shape[-1]
        self.audio_channel = self.audio_x_t.shape[-1]
        self.txt_channel = self.txt_feat.shape[-1]

    @property
    def device(self):
        return self.video_x_t.device

    @property
    def default_dtype(self):
        return self.video_x_t.dtype

    @property
    def total_token_num(self):
        return self.video_token_num + self.audio_feat_len + self.txt_feat_len

    @property
    def token_sequence(self):
        tensors = [self.video_x_t, self.audio_x_t, self.txt_feat]
        max_channel = max(t.shape[-1] for t in tensors)
        padded = [F.pad(t, (0, max_channel - t.shape[-1])) for t in tensors]
        return torch.cat(padded, dim=0)

    @property
    def modality_mapping(self):
        v_map = torch.full((self.video_token_num,), Modality.VIDEO, dtype=torch.int64, device=self.device)
        a_map = torch.full((self.audio_feat_len,), Modality.AUDIO, dtype=torch.int64, device=self.device)
        t_map = torch.full((self.txt_feat_len,), Modality.TEXT, dtype=torch.int64, device=self.device)
        return torch.cat([v_map, a_map, t_map], dim=0)

    def default_coords(self, shape, ref_feat_shape, offset_thw=None):
        if offset_thw is None:
            offset_thw = [0, 0, 0]
        return get_coords(
            shape=shape,
            ref_feat_shape=ref_feat_shape,
            offset_thw=offset_thw,
            device=self.device,
            dtype=self.default_dtype,
        )

    @property
    def coords_mapping(self):
        if self.spatial_rope_interpolation == "inter":
            video_ref_feat_shape = (self.t // self.t_patch_size, 32, 32)
        else:
            video_ref_feat_shape = (
                self.t // self.t_patch_size,
                self.h // self.patch_size,
                self.w // self.patch_size,
            )

        video_coords = self.default_coords(
            shape=(
                self.t // self.t_patch_size,
                self.h // self.patch_size,
                self.w // self.patch_size,
            ),
            ref_feat_shape=video_ref_feat_shape,
        )

        if self.coords_style == "v1":
            audio_coords = self.default_coords(
                shape=(self.audio_feat_len, 1, 1),
                ref_feat_shape=(self.t // self.t_patch_size, 1, 1),
            )
            text_coords = self.default_coords(
                shape=(self.txt_feat_len, 1, 1),
                ref_feat_shape=(2, 1, 1),
                offset_thw=[self.text_offset, 0, 0],
            )
        elif self.coords_style == "v2":
            magic_audio_ref_t = (self.audio_feat_len - 1) // 4 + 1
            audio_coords = self.default_coords(
                shape=(self.audio_feat_len, 1, 1),
                ref_feat_shape=(magic_audio_ref_t // self.t_patch_size, 1, 1),
            )
            text_coords = self.default_coords(
                shape=(self.txt_feat_len, 1, 1),
                ref_feat_shape=(1, 1, 1),
                offset_thw=[-self.txt_feat_len, 0, 0],
            )
        else:
            raise ValueError(f"Unknown coords_style: {self.coords_style}")

        return torch.cat([video_coords, audio_coords, text_coords], dim=0)

    def depack_token_sequence(self, token_sequence):
        video_x_t = token_sequence[: self.video_token_num, : self.video_channel]
        video_x_t = rearrange(
            video_x_t,
            "(T H W) (pT pH pW C) -> C (T pT) (H pH) (W pW)",
            H=self.h // self.patch_size,
            W=self.w // self.patch_size,
            pT=self.t_patch_size,
            pH=self.patch_size,
            pW=self.patch_size,
        ).contiguous()
        audio_x_t = token_sequence[
            self.video_token_num : self.video_token_num + self.audio_feat_len,
            : self.audio_channel,
        ]
        return video_x_t, audio_x_t


@dataclass
class SimplePackedData:
    items: list[SingleData]

    @property
    def token_sequence(self):
        return torch.cat([item.token_sequence for item in self.items], dim=0)

    @property
    def modality_mapping(self):
        return torch.cat([item.modality_mapping for item in self.items], dim=0)

    @property
    def coords_mapping(self):
        return torch.cat([item.coords_mapping for item in self.items], dim=0)

    @property
    def total_token_num(self):
        return sum(item.total_token_num for item in self.items)

    def __getitem__(self, index):
        return self.items[index]

    @property
    def cu_seqlen(self):
        cu = torch.cumsum(
            torch.tensor([item.total_token_num for item in self.items]),
            dim=0,
        )
        return F.pad(cu, (1, 0))

    @property
    def max_seqlen(self):
        return torch.tensor(max(item.total_token_num for item in self.items))

    def depack_token_sequence(self, token_sequence):
        video_list, audio_list = [], []
        parts = torch.split(
            token_sequence,
            [item.total_token_num for item in self.items],
            dim=0,
        )
        for item, part in zip(self.items, parts):
            v, a = item.depack_token_sequence(part)
            video_list.append(v)
            audio_list.append(a)
        return torch.stack(video_list, dim=0), torch.stack(audio_list, dim=0)


class MagiDataProxy:
    def __init__(
        self,
        patch_size: int = 2,
        t_patch_size: int = 1,
        frame_receptive_field: int = 11,
        spatial_rope_interpolation: str = "extra",
        ref_audio_offset: int = 1000,
        text_offset: int = 0,
        coords_style: str = "v2",
    ):
        self.patch_size = patch_size
        self.t_patch_size = t_patch_size
        self.frame_receptive_field = frame_receptive_field
        self.spatial_rope_interpolation = spatial_rope_interpolation
        self.ref_audio_offset = ref_audio_offset
        self.text_offset = text_offset
        self.coords_style = coords_style
        self._kernel = (t_patch_size, patch_size, patch_size)
        self._stride = (t_patch_size, patch_size, patch_size)
        self._saved_data: dict[str, Any] = {}

    def saved_for_output(self, **kwargs):
        self._saved_data.update(kwargs)

    def get_saved_data(self, key: str):
        return self._saved_data[key]

    def img2tokens(self, x_t: torch.Tensor):
        x_t_unfolded = _unfold_3d(x_t, self._kernel, self._stride)
        return rearrange(
            x_t_unfolded,
            "N col_dim num_tokens -> N num_tokens col_dim",
        ).contiguous()

    def process_input(self, transported_data: EvalInput):
        batch_size, _, t, h, w = transported_data.x_t.shape
        x_t = self.img2tokens(transported_data.x_t)
        audio_x_t = transported_data.audio_x_t.contiguous()
        text_in = transported_data.txt_feat.contiguous()

        simple_packed_data = SimplePackedData(items=[])
        for i in range(batch_size):
            single_data = SingleData(
                video_x_t=x_t[i],
                audio_x_t=audio_x_t[i],
                audio_feat_len=transported_data.audio_feat_len[i],
                txt_feat=text_in[i],
                txt_feat_len=transported_data.txt_feat_len[i],
                t=t,
                h=h,
                w=w,
                patch_size=self.patch_size,
                t_patch_size=self.t_patch_size,
                spatial_rope_interpolation=self.spatial_rope_interpolation,
                ref_audio_offset=self.ref_audio_offset,
                text_offset=self.text_offset,
                coords_style=self.coords_style,
            )
            simple_packed_data.items.append(single_data)

        if self.frame_receptive_field != -1:
            assert batch_size == 1, "local attention only supports batch size 1"
            local_attn_handler = calc_local_attn_ffa_handler(
                num_video_tokens=simple_packed_data[0].video_token_num,
                num_audio_and_txt_tokens=(simple_packed_data[0].audio_feat_len + simple_packed_data[0].txt_feat_len),
                num_frames=t,
                frame_receptive_field=self.frame_receptive_field,
            )
            if isinstance(local_attn_handler.max_seqlen_k, torch.Tensor):
                local_attn_handler.max_seqlen_k = local_attn_handler.max_seqlen_k.item()
            if isinstance(local_attn_handler.max_seqlen_q, torch.Tensor):
                local_attn_handler.max_seqlen_q = local_attn_handler.max_seqlen_q.item()
        else:
            local_attn_handler = None

        varlen_handler = VarlenHandler(
            cu_seqlens_q=simple_packed_data.cu_seqlen.to(torch.int32).cuda(),
            cu_seqlens_k=simple_packed_data.cu_seqlen.to(torch.int32).cuda(),
            max_seqlen_q=simple_packed_data.max_seqlen.to(torch.int32).cuda(),
            max_seqlen_k=simple_packed_data.max_seqlen.to(torch.int32).cuda(),
        )

        self.saved_for_output(simple_packed_data=simple_packed_data)

        x = simple_packed_data.token_sequence
        coords_mapping = simple_packed_data.coords_mapping
        modality_mapping = simple_packed_data.modality_mapping
        return (x, coords_mapping, modality_mapping, varlen_handler, local_attn_handler)

    def process_output(self, x: torch.Tensor):
        simple_packed_data: SimplePackedData = self.get_saved_data("simple_packed_data")
        return simple_packed_data.depack_token_sequence(x)
