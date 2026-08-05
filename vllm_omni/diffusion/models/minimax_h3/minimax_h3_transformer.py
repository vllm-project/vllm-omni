# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 packed-token audio/video DiT for vLLM-Omni.

vLLM tensor parallel linears and the unified attention layer provide TP and
Ulysses/Ring sequence parallel execution without changing the checkpoint
layout.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from cache_dit import ForwardPattern
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.cache.cachedit import CacheDiTAdapterConfig
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,
    SequenceParallelOutput,
)

from .fused_ops import (
    fused_qknorm_rope_bf16_,
    fused_rmsnorm_indexed_scale_shift_bf16,
    fused_rope_bf16_,
    indexed_gate_bf16_,
    indexed_scale_shift_bf16_,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )

    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


@dataclass
class MiniMaxH3DiTArchConfig:
    num_layers: int = 50
    token_refiner_num_layers: int = 2
    hidden_size: int = 5376
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336
    latents_dim: int = 24
    audio_latents_dim: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    adaln_out_features: int = 18 * 5376
    final_adaln_out_features: int = 2 * 5376
    rope_inv_freq_len: int = 16
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> MiniMaxH3DiTArchConfig:
        fields = cls.__dataclass_fields__
        values = {name: config[name] for name in fields if name in config}
        if "patch_size" in values:
            values["patch_size"] = tuple(values["patch_size"])
        arch = cls(**values)
        if len(arch.patch_size) != 3:
            raise ValueError(f"patch_size must contain three values, got {arch.patch_size!r}")
        return arch


_ARCH_DEFAULTS = MiniMaxH3DiTArchConfig()
_BF16_DTYPE = torch.bfloat16
_FP32_DTYPE = torch.float32

MINIMAX_H3_FP32_PARAM_NAMES = frozenset(
    {
        "video_patch_proj.weight",
        "video_patch_proj.bias",
        "audio_patch_proj.weight",
        "audio_patch_proj.bias",
        "time_embedder.proj_in.weight",
        "time_embedder.proj_in.bias",
        "time_embedder.proj_out.weight",
        "time_embedder.proj_out.bias",
        "final_layer.video_out.weight",
        "final_layer.video_out.bias",
        "final_layer.audio_out.weight",
        "final_layer.audio_out.bias",
    }
)
MINIMAX_H3_FP32_BUFFER_NAMES = frozenset({"rope.inv_freq"})

# AdaLN modality count: token tags carry -1 for padding and 0/1/2 for
# video/text/audio tokens (padding is clamped to 0 before the embedding
# lookup and masked out afterwards).
MINIMAX_H3_ADALN_MODALITY_NUM = 3


def _required_kwarg(kwargs: dict[str, Any], key: str) -> Any:
    if key not in kwargs or kwargs[key] is None:
        raise ValueError(f"MiniMaxH3DiTModel.forward requires kwarg {key!r}")
    return kwargs[key]


# The exhaustive keyword contract of MiniMaxH3DiTModel.forward. Anything not
# listed here is rejected with a TypeError before any tensor work starts.
_FORWARD_SUPPORTED_KWARGS = frozenset(
    {
        "x",
        "audio_x",
        "img_position_ids",
        "rope_freqs",
        "unique_timesteps",
        "inverse_indices",
        "combined_indices",
        "update_mask",
        "update_audio_mask",
        "token_tags",
        "skip_mask_out_condition",
        "prompt_embeds",
        "prompt_embeds_refined",
        "img_pos_info",
        "audio_pos_info",
        "text_pos_info",
        "img_pos_for_infer_output_info",
        "packed_seq_params",
        "packed_attn_mask",
        "refiner_packed_seq_params",
        "refiner_attn_mask",
    }
)


def _reorder_grouped_qkv_to_qkv(
    weight: torch.Tensor,
    *,
    num_query_groups: int,
    heads_per_group: int,
    head_dim: int,
) -> torch.Tensor:
    per_group = (heads_per_group + 2) * head_dim
    expected_out = num_query_groups * per_group
    if weight.shape[0] != expected_out:
        raise ValueError(
            "qkv weight has incompatible output dim for grouped checkpoint layout: "
            f"got {tuple(weight.shape)}, expected first dim {expected_out}."
        )

    rest_shape = weight.shape[1:]
    grouped = weight.reshape(num_query_groups, per_group, *rest_shape)
    q, k, v = torch.split(
        grouped,
        [heads_per_group * head_dim, head_dim, head_dim],
        dim=1,
    )
    return torch.cat(
        [
            q.reshape(num_query_groups * heads_per_group * head_dim, *rest_shape),
            k.reshape(num_query_groups * head_dim, *rest_shape),
            v.reshape(num_query_groups * head_dim, *rest_shape),
        ],
        dim=0,
    )


def _norm(size: int, *, eps: float, dtype: torch.dtype = _BF16_DTYPE) -> nn.RMSNorm:
    # RMSNorm uses fp32 accumulation with bf16 inputs and outputs.
    # torch.nn.RMSNorm upcasts reduced-precision inputs for the variance
    # reduction, matching that accumulation semantic.
    return nn.RMSNorm(size, eps=eps, dtype=dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _modulate_scale_shift(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Apply per-index affine modulation: x * (1 + scale[idx]) + shift[idx].
    if indexed_scale_shift_bf16_(x, shift, scale, indices):
        return x
    return (x * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)).to(dtype)


def _norm_modulate_scale_shift(
    norm: nn.RMSNorm,
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    fused = fused_rmsnorm_indexed_scale_shift_bf16(
        x,
        norm.weight,
        shift,
        scale,
        indices,
        norm.eps,
    )
    if fused is not None:
        return fused
    return _modulate_scale_shift(norm(x), shift, scale, indices, dtype=_BF16_DTYPE)


def _modulate_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Apply the per-index gated residual: x + gate[idx] * other.
    if indexed_gate_bf16_(x, gate, other, indices):
        return x
    return (x + gate.index_select(0, indices) * other).to(dtype)


def _packed_attention_mask(cu_seqlens: torch.Tensor) -> torch.Tensor | None:
    """Build the optional alignment mask once per packed forward.

    H3 normally has no alignment rows, in which case the attention backend
    takes its mask-free fast path.  When SP padding adds a second, partial
    document, preserve the old mask semantics but keep the scalar CUDA reads
    out of the per-block attention loop.
    """
    if cu_seqlens.numel() < 2:
        return None
    used = int(cu_seqlens[1].item())
    packed_total = int(cu_seqlens[-1].item())
    if used >= packed_total:
        return None
    return torch.arange(packed_total, device=cu_seqlens.device)[None] < used


def _sequence_parallel_local_span(seq_len: int) -> tuple[int, int]:
    """Return this rank's contiguous packed span when strict SP is active."""
    try:
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_sequence_parallel_rank,
            get_sequence_parallel_world_size,
        )

        world_size = int(get_sequence_parallel_world_size())
        rank = int(get_sequence_parallel_rank())
    except AssertionError:
        # Unit tests and single-process callers do not initialize model
        # parallel groups. They retain the original full-sequence path.
        return 0, seq_len

    if world_size <= 1 or seq_len % world_size:
        # The existing SP hook handles non-divisible layouts (when configured
        # with auto padding). Keep the full embedding path in that case so the
        # hook remains the single owner of padding semantics.
        return 0, seq_len
    local_len = seq_len // world_size
    return rank * local_len, local_len


class MiniMaxH3Rope(nn.Module):
    """3D rope over (t, h, w); rotates 96 of 128 head dims (rotary_percent 0.75).

    Frequency layout concatenates temporal, height, and width embeddings twice,
    with 16 frequencies per axis (inv_freq = base^-(arange(0,32,2)/32)).
    """

    def __init__(self, inv_freq_len: int) -> None:
        super().__init__()
        self.register_buffer(
            "inv_freq",
            torch.empty(inv_freq_len, dtype=_FP32_DTYPE),
            persistent=True,
        )

    def forward(self, img_position_ids: torch.Tensor) -> torch.Tensor:
        """img_position_ids: [1, S, 3] (t, h, w) -> freqs [S, rot_dim=96]."""
        if img_position_ids.dim() != 3 or img_position_ids.shape[0] != 1:
            raise ValueError(f"img_position_ids must be [1, S, 3], got {list(img_position_ids.shape)}")
        pos = img_position_ids[0].to(_FP32_DTYPE)  # [S, 3]
        per_axis = pos.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)  # [S, 3, 16]
        t_f, h_f, w_f = per_axis.unbind(dim=1)  # each [S, 16]
        half = torch.cat((t_f, h_f, w_f), dim=-1)  # [S, 48]
        return torch.cat((half, half), dim=-1)  # [S, 96]

    def build_cache(
        self,
        img_position_ids: torch.Tensor,
        *,
        dtype: torch.dtype = _BF16_DTYPE,
    ) -> torch.Tensor:
        """Build the request-static cosine/sine table used by DiT attention.

        The packed positions do not change during denoising.  Keeping the
        trigonometric conversion here lets the pipeline build it once per
        request instead of repeating ``cos``/``sin`` for every DiT step.
        The returned layout is ``[cos, sin]`` and is understood by
        :func:`_apply_rope`.
        """
        freqs = self.forward(img_position_ids)
        return torch.cat((torch.cos(freqs), torch.sin(freqs)), dim=-1).to(dtype)


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Rotate the first rot_dim head dims; pass the rest through.

    x: [T, heads, head_dim]; freqs: [T, rot_dim]. In the unfused path, cos/sin
    are cast to the activation dtype before the elementwise math.
    """
    # The model passes a [cos, sin] cache through the SP boundary when the
    # same positions are reused by all DiT blocks.  Keep accepting raw
    # frequencies for callers/tests that exercise the helper directly.
    cached = freqs.shape[-1] > x.shape[-1]
    if cached:
        rot_dim = freqs.shape[-1] // 2
        cos, sin = freqs.split(rot_dim, dim=-1)
    else:
        rot_dim = freqs.shape[-1]
        cos = torch.cos(freqs).to(x.dtype)
        sin = torch.sin(freqs).to(x.dtype)
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    x_rot = (x_rot * cos.unsqueeze(1)) + (_rotate_half(x_rot) * sin.unsqueeze(1))
    return torch.cat((x_rot, x_pass), dim=-1)


class MiniMaxH3TimeEmbedder(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        *,
        prefix: str,
    ) -> None:
        super().__init__()
        self.frequency_embedding_size = arch.timestep_input_dim
        half = self.frequency_embedding_size // 2
        self.register_buffer(
            "_frequency_cache",
            torch.exp(-math.log(10000.0) * torch.arange(half, dtype=_FP32_DTYPE) / half),
            persistent=False,
        )
        self.proj_in = ColumnParallelLinear(
            arch.timestep_input_dim,
            arch.time_embed_hidden_size,
            bias=True,
            gather_output=True,
            params_dtype=_FP32_DTYPE,
            quant_config=None,
            prefix=f"{prefix}.proj_in",
        )
        self.proj_out = RowParallelLinear(
            arch.time_embed_hidden_size,
            arch.time_embed_dim,
            bias=True,
            input_is_parallel=False,
            params_dtype=_FP32_DTYPE,
            quant_config=None,
            prefix=f"{prefix}.proj_out",
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [M] -> [M, time_embed_dim] fp32.

        The sinusoidal embedding stays fp32 throughout and concatenates cosine
        values before sine values.
        """
        freqs = self._frequency_cache
        if freqs.device != t.device:
            # The normal serving path moves the non-persistent buffer with the
            # module.  Keep a defensive fallback for direct callers that pass
            # a tensor on a different device.
            freqs = freqs.to(device=t.device)
        args = t.to(_FP32_DTYPE)[:, None] * freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        hidden, _ = self.proj_in(t_freq)
        hidden = nn.functional.silu(hidden)
        out, _ = self.proj_out(hidden)
        return out


def _sdpa_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Segment-wise SDPA equivalent of the non-causal varlen FA call.

    Mirrors the generic attention layer's semantics: FA is the fast path,
    SDPA is the correctness fallback when the platform resolves another
    backend. Segments are delimited by ``cu_seqlens`` exactly like the
    varlen kernel, so attention never crosses packed-document boundaries.
    """
    out = torch.empty_like(q)
    bounds = cu_seqlens.tolist()
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if stop == start:
            continue
        seg_q = q[start:stop].transpose(0, 1).unsqueeze(0)
        seg_k = k[start:stop].transpose(0, 1).unsqueeze(0)
        seg_v = v[start:stop].transpose(0, 1).unsqueeze(0)
        seg_out = torch.nn.functional.scaled_dot_product_attention(
            seg_q,
            seg_k,
            seg_v,
            scale=softmax_scale,
        )
        out[start:stop] = seg_out.squeeze(0).transpose(0, 1)
    return out


class MiniMaxH3Attention(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
        *,
        prefix: str,
        skip_sequence_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.total_num_heads = arch.num_attention_heads
        self.head_dim = arch.attention_head_dim
        inner_dim = self.total_num_heads * self.head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.qkv_proj = QKVParallelLinear(
            hidden_size=arch.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_heads,
            bias=False,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            return_bias=True,
        )
        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        self.q_norm = _norm(arch.attention_head_dim, eps=arch.qk_norm_eps)
        self.k_norm = _norm(arch.attention_head_dim, eps=arch.qk_norm_eps)
        self.out_proj = RowParallelLinear(
            inner_dim,
            arch.hidden_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        self.attention = Attention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            softmax_scale=self.softmax_scale,
            causal=False,
            skip_sequence_parallel=skip_sequence_parallel,
        )

    @torch.compiler.disable
    def _run_packed_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        packed_qkv: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run packed attention as a small eager island.

        The scalar packed-layout metadata and the CuTe FlashAttention-4 DSL
        are intentionally opaque to Dynamo. Keeping this boundary narrow lets
        regional compile fuse projections, norms, RoPE, and the surrounding
        DiT block without repeatedly graph-breaking inside the FA4 compiler.
        """
        if self._can_use_direct_fa4(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            attn_mask=attn_mask,
        ):
            return self._run_direct_fa4(
                q,
                k,
                v,
                packed_qkv=packed_qkv,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        if self._can_use_packed_ulysses(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            attn_mask=attn_mask,
        ):
            return self._run_packed_ulysses(
                q,
                k,
                v,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        metadata = AttentionMetadata(
            attn_mask=attn_mask,
            extra={
                "cu_seqlens_q": cu_seqlens,
                "cu_seqlens_k": cu_seqlens,
                "max_seqlen_q": max_seqlen,
                "max_seqlen_k": max_seqlen,
            },
        )
        return self.attention(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            metadata,
        ).squeeze(0)

    @staticmethod
    def _can_use_direct_fa4(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> bool:
        """Check the fixed-shape H3 Ulysses fast path.

        H3's production recipe is strict Ulysses-4 with no ring stage and a
        packed, mask-free sequence.  In that shape the generic Attention
        wrapper adds strategy/metadata dispatch around the same FA4 call.
        Keep the direct path narrowly gated so other parallel layouts and
        alignment-mask cases retain the generic semantics.
        """
        if attn_mask is not None or q.device.type != "cuda":
            return False
        if q.shape != k.shape or q.shape != v.shape:
            return False
        if cu_seqlens.numel() < 2:
            return False
        try:
            from vllm_omni.diffusion.attention.backends.utils.fa import flash_attn_varlen_func
            from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

            sp_group = get_sp_group()
            if sp_group.ulysses_world_size <= 1 or sp_group.ring_world_size != 1:
                return False
            if sp_group.ulysses_world_size != 4:
                return False
            if getattr(flash_attn_varlen_func, "__module__", "") != "flash_attn.cute.interface":
                return False
            # Strict Ulysses uses equal local sequence shards.  This also
            # rules out the advanced-UAA variable-length path.
            return int(cu_seqlens[-1].item()) == int(q.shape[0]) * int(sp_group.ulysses_world_size)
        except (AssertionError, ImportError, RuntimeError):
            return False

    @staticmethod
    def _can_use_packed_ulysses(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> bool:
        """Check H3's packed-QKV Ulysses path.

        A 2-D alignment mask is already redundant when packed FlashAttention
        receives the two document boundaries in ``cu_seqlens``.  Keep the
        branch restricted to that representation; 4-D/piecewise masks stay
        on the generic strategy so their semantics remain unchanged.
        """
        if q.device.type != "cuda" or q.dtype != torch.bfloat16:
            return False
        if q.ndim != 3 or q.shape != k.shape or q.shape != v.shape:
            return False
        if attn_mask is not None and attn_mask.ndim != 2:
            return False
        if cu_seqlens.numel() < 2:
            return False
        try:
            from vllm_omni.diffusion.attention.backends.utils.fa import flash_attn_varlen_func
            from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

            sp_group = get_sp_group()
            if sp_group.ulysses_world_size != 4 or sp_group.ring_world_size != 1:
                return False
            if getattr(flash_attn_varlen_func, "__module__", "") != "flash_attn.cute.interface":
                return False
            return int(cu_seqlens[-1].item()) == int(q.shape[0]) * int(sp_group.ulysses_world_size)
        except (AssertionError, ImportError, RuntimeError):
            return False

    def _run_packed_ulysses(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """Run FA4 with destination-major QKV and fused inverse relayout."""
        from vllm_omni.diffusion.attention.backends.utils.fa import flash_attn_varlen_func
        from vllm_omni.diffusion.distributed.comm import (
            all_to_all_packed_qkv,
            all_to_all_ulysses_output,
        )
        from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

        group = get_sp_group().ulysses_group
        packed = all_to_all_packed_qkv(q, k, v, group=group)
        if packed is None:
            raise RuntimeError("H3 packed Ulysses path was selected for an unsupported QKV layout")
        q, k, v = packed
        out = flash_attn_varlen_func(
            q=q.flatten(0, 1),
            k=k.flatten(0, 1),
            v=v.flatten(0, 1),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=False,
            softmax_scale=self.softmax_scale,
        )
        if isinstance(out, tuple):
            out = out[0]
        out = out.reshape(q.shape)
        merged = all_to_all_ulysses_output(out, group=group)
        if merged is None:
            raise RuntimeError("H3 packed Ulysses output relayout failed")
        return merged.squeeze(0)

    def _run_direct_fa4(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        packed_qkv: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """Run FA4 with the two fixed-shape Ulysses collectives inline."""
        from vllm_omni.diffusion.attention.backends.utils.fa import flash_attn_varlen_func
        from vllm_omni.diffusion.distributed.comm import (
            all_to_all_4D,
            all_to_all_5D,
        )
        from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

        group = get_sp_group().ulysses_group
        # Q, K, and V have identical packed shapes.  Move them through one
        # 5-D all-to-all instead of launching three independent collectives;
        # this keeps the same destination-major layout while removing two
        # NCCL launch/synchronization points from every DiT attention block.
        if packed_qkv is None:
            # Keep the generic fallback for callers whose Q/K/V do not share
            # the original projection buffer (for example, non-fused norm
            # fallbacks and direct unit-test calls).
            qkv = torch.stack((q, k, v), dim=1).unsqueeze(0)
        else:
            # QKVParallelLinear already emits [S, 3*H*D] in Q/K/V order.
            # The fused QKNorm+RoPE path mutates q and k in that buffer, so a
            # reshape is a view of the normalized values and avoids rebuilding
            # the same [1, S, 3, H, D] payload with torch.stack.
            qkv = packed_qkv.view(1, q.shape[0], 3, q.shape[1], q.shape[2])
        qkv = all_to_all_5D(qkv, scatter_idx=3, gather_idx=1, group=group)
        q, k, v = qkv.unbind(dim=2)
        # FA4 requires independently contiguous Q/K/V strides.  The 5-D
        # result is QKV-interleaved along the sequence dimension, so keep the
        # collective packing but materialize the three kernel inputs here.
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out = flash_attn_varlen_func(
            q=q.flatten(0, 1),
            k=k.flatten(0, 1),
            v=v.flatten(0, 1),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=False,
            softmax_scale=self.softmax_scale,
        )
        if isinstance(out, tuple):
            out = out[0]
        out = out.reshape(q.shape)
        return all_to_all_4D(out, scatter_idx=1, gather_idx=2, group=group).squeeze(0)

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_freqs: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        attn_mask: torch.Tensor | None = None,
        sp_seq_lens: list[int] | None = None,
    ) -> torch.Tensor:
        """x: [T, hidden] packed thd rows -> [T, hidden].

        Operation order: fused qkv projection -> per-head q/k RMSNorm -> RoPE
        on q/k -> variable-length non-causal flash attention -> output projection.

        With Ulysses sequence parallelism, x holds this rank's row shard;
        qkv/norm/RoPE run locally, an all-to-all trades sequence for heads.
        Each rank attends the full sequence with heads/world_size local heads,
        so cu_seqlens retains global packed-document semantics. The inverse
        all-to-all restores the row shard before the output projection.
        """
        total = x.shape[0]
        qkv, _ = self.qkv_proj(x)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(total, self.num_heads, self.head_dim)
        k = k.view(total, self.num_kv_heads, self.head_dim)
        v = v.view(total, self.num_kv_heads, self.head_dim)
        fused_qk_rope = False
        if rope_freqs is not None:
            fused_qk_rope = fused_qknorm_rope_bf16_(
                q,
                k,
                self.q_norm.weight,
                self.k_norm.weight,
                rope_freqs,
                eps=self.q_norm.eps,
                rope_dim=rope_freqs.shape[-1] // 2,
            )
        packed_qkv = None
        if fused_qk_rope and qkv.is_contiguous() and q.shape == k.shape == v.shape:
            packed_qkv = qkv
        if not fused_qk_rope:
            q = self.q_norm(q)
            k = self.k_norm(k)
            if rope_freqs is not None:
                rope_dim = rope_freqs.shape[-1] // 2
                if not fused_rope_bf16_(q, k, rope_freqs, rope_dim=rope_dim):
                    q = _apply_rope(q, rope_freqs)
                    k = _apply_rope(k, rope_freqs)

        # The packed layout uses a second document for alignment padding.
        # Local/Ulysses backends unpad it, while Ring keeps aligned rows for
        # fixed-size P2P buffers.
        out = self._run_packed_attention(
            q,
            k,
            v,
            packed_qkv=packed_qkv,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            attn_mask=attn_mask,
        )
        out = out.reshape(total, self.num_heads * self.head_dim)
        out, _ = self.out_proj(out)
        return out


class MiniMaxH3MLP(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
        *,
        prefix: str,
    ) -> None:
        super().__init__()
        self.fc1 = MergedColumnParallelLinear(
            arch.hidden_size,
            [arch.ffn_hidden_size, arch.ffn_hidden_size],
            bias=False,
            gather_output=False,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        # Chunk the fused fc1 output as [gate, up], then compute
        # silu(gate) * up.
        self.fc2 = RowParallelLinear(
            arch.ffn_hidden_size,
            arch.hidden_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )
        self.silu_and_mul = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.fc1(x)
        hidden = self.silu_and_mul(hidden)
        out, _ = self.fc2(hidden)
        return out


class MiniMaxH3AdalnProj(nn.Module):
    """SiLU + zero-init linear over unique condition embeddings.

    Per block, three modalities each produce six H-wide vectors:
    [M, t_dim] -> [M, 3*6H] -> view(M*3, 6H) -> chunk(6).
    The final layer uses one modality and produces two H-wide vectors:
    [M, t_dim] -> [M, 2H] -> chunk(2).
    """

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        out_features: int,
        quant_config: QuantizationConfig | None,
        *,
        expand_ratio: int,
        modality_num: int,
        prefix: str,
    ) -> None:
        super().__init__()
        if out_features != expand_ratio * arch.hidden_size * modality_num:
            raise ValueError(
                f"adaln out_features mismatch: {out_features} != {expand_ratio}*{arch.hidden_size}*{modality_num}"
            )
        self.expand_ratio = expand_ratio
        self.modality_num = modality_num
        self.hidden_size = arch.hidden_size
        self.linear = ColumnParallelLinear(
            arch.time_embed_dim,
            out_features,
            bias=True,
            gather_output=True,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
            prefix=f"{prefix}.linear",
        )

    def forward(self, t_emb: torch.Tensor, *, preactivated: bool = False) -> tuple[torch.Tensor, ...]:
        """t_emb: [M, t_dim] -> expand_ratio tensors of [M*modality_num, H]."""
        if not preactivated:
            t_emb = nn.functional.silu(t_emb).to(_BF16_DTYPE)
        x, _ = self.linear(t_emb)
        m = x.shape[0]
        x = x.view(m * self.modality_num, self.expand_ratio * self.hidden_size)
        return tuple(x.chunk(self.expand_ratio, dim=-1))


class MiniMaxH3TokenRefinerBlock(nn.Module):
    """Standard pre-norm transformer block without AdaLN or RoPE."""

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
        *,
        prefix: str,
    ) -> None:
        super().__init__()
        self.norm1 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.norm2 = _norm(arch.hidden_size, eps=arch.norm_eps)
        # Text refinement runs on replicated rows before ``sp_prepare``.
        # Applying Ulysses here would all-to-all an unsharded sequence while
        # retaining the original packed ``cu_seqlens`` metadata.
        self.attn = MiniMaxH3Attention(
            arch,
            quant_config,
            prefix=f"{prefix}.attn",
            skip_sequence_parallel=True,
        )
        self.mlp = MiniMaxH3MLP(
            arch,
            quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            rope_freqs=None,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            attn_mask=attn_mask,
        )
        x = x + self.mlp(self.norm2(x))
        return x


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
        *,
        prefix: str,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                MiniMaxH3TokenRefinerBlock(
                    arch,
                    quant_config,
                    prefix=f"{prefix}.blocks.{i}",
                )
                for i in range(arch.token_refiner_num_layers)
            ]
        )
        self.final_norm = _norm(arch.hidden_size, eps=arch.final_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_mask is None:
            attn_mask = _packed_attention_mask(cu_seqlens)
        for block in self.blocks:
            x = block(
                x,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                attn_mask=attn_mask,
            )
        return self.final_norm(x)


class MiniMaxH3DiTBlock(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
        *,
        prefix: str,
    ) -> None:
        super().__init__()
        self.norm1 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.norm2 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.attn = MiniMaxH3Attention(
            arch,
            quant_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = MiniMaxH3MLP(
            arch,
            quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.adaln_proj = MiniMaxH3AdalnProj(
            arch,
            arch.adaln_out_features,
            quant_config,
            expand_ratio=6,
            modality_num=MINIMAX_H3_ADALN_MODALITY_NUM,
            prefix=f"{prefix}.adaln_proj",
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        t_emb: torch.Tensor,
        combined_indices: torch.Tensor,
        rope_freqs: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        attn_mask: torch.Tensor | None = None,
        sp_seq_lens: list[int] | None = None,
        t_emb_preactivated: bool = False,
    ) -> torch.Tensor:
        """x: [T, H]; t_emb: [M, t_dim]; combined_indices: [T]
        (= inverse_indices * modality_num + token_tags.clamp(min=0)).

        Each block computes AdaLN parameters once, then applies
        norm1 -> scale/shift -> attention -> gated residual, followed by
        norm2 -> scale/shift -> MLP -> gated residual.
        """
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaln_proj(t_emb, preactivated=t_emb_preactivated)

        residual = x
        h = _norm_modulate_scale_shift(self.norm1, x, shift_msa, scale_msa, combined_indices)
        h = self.attn(
            h,
            rope_freqs=rope_freqs,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            attn_mask=attn_mask,
            sp_seq_lens=sp_seq_lens,
        )
        x = _modulate_gate(residual, gate_msa, h, combined_indices, dtype=_BF16_DTYPE)

        residual = x
        h = _norm_modulate_scale_shift(self.norm2, x, shift_mlp, scale_mlp, combined_indices)
        h = self.mlp(h)
        return _modulate_gate(residual, gate_mlp, h, combined_indices, dtype=_BF16_DTYPE)


class MiniMaxH3FinalLayer(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
        *,
        prefix: str,
    ) -> None:
        super().__init__()
        video_patch_dim = arch.latents_dim * arch.patch_size[0] * arch.patch_size[1] * arch.patch_size[2]
        self.norm = _norm(arch.hidden_size, eps=arch.final_norm_eps)
        self.adaln_proj = MiniMaxH3AdalnProj(
            arch,
            arch.final_adaln_out_features,
            quant_config,
            expand_ratio=2,
            modality_num=1,
            prefix=f"{prefix}.adaln_proj",
        )
        self.video_out = ColumnParallelLinear(
            arch.hidden_size,
            video_patch_dim,
            bias=True,
            gather_output=True,
            params_dtype=_FP32_DTYPE,
            quant_config=None,
            prefix=f"{prefix}.video_out",
        )
        self.audio_out = ColumnParallelLinear(
            arch.hidden_size,
            arch.audio_latents_dim,
            bias=True,
            gather_output=True,
            params_dtype=_FP32_DTYPE,
            quant_config=None,
            prefix=f"{prefix}.audio_out",
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        t_emb: torch.Tensor,
        inverse_indices: torch.Tensor,
        t_emb_preactivated: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [T, H] -> (video_logits [T, 96] fp32, audio_logits [T, 32] fp32).

        Apply single-modality shift/scale AdaLN to the final normalized
        activations, cast to fp32, then apply both output heads to all rows.
        """
        shift, scale = self.adaln_proj(t_emb, preactivated=t_emb_preactivated)
        h = _norm_modulate_scale_shift(self.norm, x, shift, scale, inverse_indices)
        # Preserve full precision through both final output projections.
        h = h.to(_FP32_DTYPE)
        video, _ = self.video_out(h)
        audio, _ = self.audio_out(h)
        return video, audio


class MiniMaxH3SPPrepare(nn.Module):
    """Explicit boundary for sharding packed rows and their metadata together."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_freqs: torch.Tensor,
        combined_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return hidden_states, rope_freqs, combined_indices


class MiniMaxH3SPGather(nn.Module):
    """Explicit boundary for restoring packed rows after the block stack."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class MiniMaxH3DiTModel(nn.Module):
    _cache_dit_adapter_config = CacheDiTAdapterConfig(
        block_forward_patterns={"blocks": ForwardPattern.Pattern_3},
        # H3 is CFG-distilled and performs one transformer forward per step.
        has_separate_cfg=False,
        check_forward_pattern=False,
    )
    # H3's repeated block GEMMs are small enough that Inductor's autotuner
    # materially improves the steady-state kernel choice.  Keep CUDA graphs
    # disabled because the eager FA4/collective island is intentionally
    # outside each compiled block.
    _regional_compile_kwargs = {"mode": "max-autotune-no-cudagraphs"}
    _repeated_blocks = ["MiniMaxH3DiTBlock"]
    _layerwise_offload_blocks_attrs = ["blocks"]

    @staticmethod
    def _is_transformer_block(name: str, module: nn.Module) -> bool:
        del module
        parts = name.split(".")
        return len(parts) == 2 and parts[0] == "blocks" and parts[1].isdigit()

    _hsdp_shard_conditions = [_is_transformer_block]
    _hsdp_ignored_modules = [
        "video_patch_proj",
        "audio_patch_proj",
        "time_embedder",
        "final_layer",
    ]
    _sp_plan = {
        "sp_prepare": {
            2: SequenceParallelInput(
                split_dim=0,
                expected_dims=1,
                split_output=True,
            ),
        },
        "sp_gather": SequenceParallelOutput(gather_dim=0, expected_dims=2),
    }
    packed_modules_mapping = {}

    def _validate_tp_config(self, *, arch: MiniMaxH3DiTArchConfig, tp_size: int) -> None:
        if tp_size < 1:
            raise ValueError(f"tensor_parallel_size must be positive, got {tp_size}")
        if arch.num_attention_heads % tp_size:
            raise ValueError(
                "num_attention_heads must be divisible by tensor_parallel_size: "
                f"{arch.num_attention_heads} % {tp_size} != 0"
            )
        if arch.ffn_hidden_size % tp_size:
            raise ValueError(
                f"ffn_hidden_size must be divisible by tensor_parallel_size: {arch.ffn_hidden_size} % {tp_size} != 0"
            )
        if arch.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive.")
        if arch.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if arch.attention_head_dim <= 0:
            raise ValueError("attention_head_dim must be positive.")
        if arch.ffn_hidden_size <= 0:
            raise ValueError("ffn_hidden_size must be positive.")

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        tf_config = od_config.tf_model_config
        config_mapping = tf_config.to_dict() if hasattr(tf_config, "to_dict") else dict(tf_config)
        arch = MiniMaxH3DiTArchConfig.from_mapping(config_mapping)
        self.arch = arch
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.latents_dim
        self._validate_tp_config(
            arch=arch,
            tp_size=get_tensor_model_parallel_world_size(),
        )
        local_heads = arch.num_attention_heads // get_tensor_model_parallel_world_size()
        ulysses_degree = int(self.parallel_config.ulysses_degree)
        if local_heads % ulysses_degree:
            raise ValueError(
                "MiniMax H3 local attention heads must be divisible by "
                "ulysses_degree: "
                f"({arch.num_attention_heads} / "
                f"{get_tensor_model_parallel_world_size()}) % "
                f"{ulysses_degree} != 0"
            )

        self.video_patch_proj = ColumnParallelLinear(
            arch.latents_dim * arch.patch_size[0] * arch.patch_size[1] * arch.patch_size[2],
            arch.hidden_size,
            bias=True,
            gather_output=True,
            params_dtype=_FP32_DTYPE,
            quant_config=None,
            prefix="video_patch_proj",
        )
        self.audio_patch_proj = ColumnParallelLinear(
            arch.audio_latents_dim,
            arch.hidden_size,
            bias=True,
            gather_output=True,
            params_dtype=_FP32_DTYPE,
            quant_config=None,
            prefix="audio_patch_proj",
        )
        self.condition_proj = ColumnParallelLinear(
            arch.text_dim,
            arch.hidden_size,
            bias=True,
            gather_output=True,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
            prefix="condition_proj",
        )
        self.time_embedder = MiniMaxH3TimeEmbedder(
            arch,
            prefix="time_embedder",
        )
        self.rope = MiniMaxH3Rope(arch.rope_inv_freq_len)
        self.token_refiner = MiniMaxH3TokenRefiner(
            arch,
            quant_config,
            prefix="token_refiner",
        )
        self.blocks = nn.ModuleList(
            [
                MiniMaxH3DiTBlock(
                    arch,
                    quant_config,
                    prefix=f"blocks.{i}",
                )
                for i in range(arch.num_layers)
            ]
        )
        self.sp_prepare = MiniMaxH3SPPrepare()
        self.sp_gather = MiniMaxH3SPGather()
        self.final_layer = MiniMaxH3FinalLayer(
            arch,
            quant_config,
            prefix="final_layer",
        )
        self._mark_missing_params_required()

    def _mark_missing_params_required(self) -> None:
        for _, param in self.named_parameters():
            param.missing_param_init = "error"

    def post_load_weights(self) -> None:
        for name, param in self.named_parameters():
            if name in MINIMAX_H3_FP32_PARAM_NAMES and param.dtype != _FP32_DTYPE:
                raise ValueError(f"{name} must stay fp32 after load, got {param.dtype}.")
        for name, buffer in self.named_buffers():
            if name in MINIMAX_H3_FP32_BUFFER_NAMES and buffer.dtype != _FP32_DTYPE:
                raise ValueError(f"{name} must stay fp32 after load, got {buffer.dtype}.")

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load exact H3 checkpoint names with logical TP-aware loaders."""
        params = dict(self.named_parameters())
        params.update(dict(self.named_buffers()))
        loaded: set[str] = set()
        for name, loaded_weight in weights:
            param = params.get(name)
            if param is None:
                logger.warning("Skipping MiniMax H3 weight not present in model: %s", name)
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            if name.endswith(".attn.qkv_proj.weight"):
                # Transform checkpoint layout before entering vLLM's loader so
                # online FP8 can keep ``online_process_loader`` outermost.
                loaded_weight = _reorder_grouped_qkv_to_qkv(
                    loaded_weight,
                    num_query_groups=self.arch.num_attention_heads,
                    heads_per_group=1,
                    head_dim=self.arch.attention_head_dim,
                )
                weight_loader(param, loaded_weight)
            elif name.endswith(".mlp.fc1.weight"):
                if loaded_weight.shape[0] % 2:
                    raise ValueError(
                        "MiniMax H3 fc1 checkpoint rows must split evenly into "
                        f"gate/up matrices, got {tuple(loaded_weight.shape)}"
                    )
                gate, up = loaded_weight.chunk(2, dim=0)
                weight_loader(param, gate, 0)
                weight_loader(param, up, 1)
            else:
                weight_loader(param, loaded_weight)
            loaded.add(name)
        return loaded

    @staticmethod
    def _pos_ids(pos_info: Any, key: str) -> torch.Tensor:
        if isinstance(pos_info, dict):
            ids = pos_info.get("position_ids")
        else:
            ids = getattr(pos_info, "position_ids", None)
        if ids is None:
            raise ValueError(f"{key}.position_ids is required")
        return ids.view(-1).to(torch.long)

    @staticmethod
    def _psp_field(psp: Any, key: str, field: str) -> Any:
        if isinstance(psp, dict):
            value = psp.get(field)
        else:
            value = getattr(psp, field, None)
        if value is None:
            raise ValueError(f"{key}.{field} is required")
        return value

    def _embed(
        self,
        *,
        x: torch.Tensor,
        audio_x: torch.Tensor,
        text_embeddings_selected: torch.Tensor,
        unique_timesteps: torch.Tensor,
        img_pos: torch.Tensor,
        audio_pos: torch.Tensor,
        text_pos: torch.Tensor,
        refiner_cu_seqlens: torch.Tensor,
        refiner_max_seqlen: int,
        refiner_attn_mask: torch.Tensor | None,
        seq_len: int,
        device: torch.device,
        prompt_embeds_refined: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build packed multimodal embeddings before the SP boundary.

        Returns (decoder_input [S, H] bf16, t_emb [M, t_dim] fp32).
        """
        local_start, local_len = _sequence_parallel_local_span(seq_len)
        local_only = local_len != seq_len
        if local_only:
            local_end = local_start + local_len
            img_row_mask = (img_pos >= local_start) & (img_pos < local_end)
            audio_row_mask = (audio_pos >= local_start) & (audio_pos < local_end)
            text_row_mask = (text_pos >= local_start) & (text_pos < local_end)
            img_global_rows = img_pos[img_row_mask]
            audio_global_rows = audio_pos[audio_row_mask]
            img_rows = img_global_rows - local_start
            audio_rows = audio_global_rows - local_start
            text_rows_idx = torch.nonzero(text_row_mask, as_tuple=False).view(-1)
            text_rows_pos = text_pos[text_row_mask] - local_start
        else:
            img_global_rows = img_pos
            audio_global_rows = audio_pos
            img_rows = img_pos
            audio_rows = audio_pos
            text_rows_idx = None
            text_rows_pos = text_pos

        # Latent embedders stay fp32 in and out; their outputs are cast to the
        # bf16 sequence dtype only during indexed scattering.
        x_rows = x.view(-1, x.shape[-1]).index_select(0, img_global_rows).to(_FP32_DTYPE)
        video_embed, _ = self.video_patch_proj(x_rows)
        audio_input_rows = audio_x.view(-1, audio_x.shape[-1]).index_select(0, audio_global_rows).to(_FP32_DTYPE)
        audio_embed, _ = self.audio_patch_proj(audio_input_rows)

        text_rows = text_embeddings_selected.to(device=device, dtype=_BF16_DTYPE)
        if not prompt_embeds_refined:
            text_embed, _ = self.condition_proj(text_rows)
            text_embed = self.token_refiner(
                text_embed,
                cu_seqlens=refiner_cu_seqlens,
                max_seqlen=refiner_max_seqlen,
                attn_mask=refiner_attn_mask,
            )
        else:
            # The condition projection and token refiner are independent of
            # the denoise timestep.  The pipeline computes this once and
            # passes the refined rows through the static forward kwargs.
            text_embed = text_rows

        # Every packed row has exactly one owner (text, image, audio, or
        # alignment padding), so index_copy avoids the read/modify/write
        # accumulation used by the old full-sequence index_add path. Under
        # strict Ulysses, keep only this rank's span resident; the previous
        # path allocated a full [S, H] buffer and immediately discarded the
        # other ranks' rows at the SP boundary.
        embeddings = torch.empty((local_len, self.hidden_size), device=device, dtype=_BF16_DTYPE)
        embeddings.zero_()
        if local_only:
            text_embed = text_embed.index_select(0, text_rows_idx)
        embeddings.index_copy_(0, text_rows_pos, text_embed.to(_BF16_DTYPE))
        embeddings.index_copy_(0, img_rows, video_embed.to(_BF16_DTYPE))
        embeddings.index_copy_(0, audio_rows, audio_embed.to(_BF16_DTYPE))

        t_emb = self.time_embedder(unique_timesteps)
        return embeddings, t_emb

    @torch.inference_mode()
    def prepare_prompt_embeddings(
        self,
        text_embeddings: torch.Tensor,
        *,
        refiner_cu_seqlens: torch.Tensor,
        refiner_max_seqlen: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Run the timestep-invariant text path once per diffusion request."""
        text_rows = text_embeddings.to(device=device, dtype=_BF16_DTYPE)
        text_embed, _ = self.condition_proj(text_rows)
        refiner_cu_seqlens = refiner_cu_seqlens.to(device=device, dtype=torch.int32)
        return self.token_refiner(
            text_embed,
            cu_seqlens=refiner_cu_seqlens,
            max_seqlen=int(refiner_max_seqlen),
            attn_mask=_packed_attention_mask(refiner_cu_seqlens),
        )

    def forward(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Packed inference forward.

        Keyword names follow the checkpoint's serving contract.
        Returns `(video_logits, audio_logits)` from rows selected by
        `img_pos_for_infer_output_info` and `audio_pos_info`, with condition
        rows zeroed by update masks.
        """
        # Strict keyword contract: refuse any kwarg forward does not consume.
        unexpected = sorted(set(kwargs) - _FORWARD_SUPPORTED_KWARGS)
        if unexpected:
            raise TypeError(
                "MiniMaxH3DiTModel.forward received unexpected kwargs: "
                f"{unexpected}; supported kwargs: "
                f"{sorted(_FORWARD_SUPPORTED_KWARGS)}"
            )

        x = _required_kwarg(kwargs, "x")
        audio_x = _required_kwarg(kwargs, "audio_x")
        img_position_ids = _required_kwarg(kwargs, "img_position_ids")
        unique_timesteps = _required_kwarg(kwargs, "unique_timesteps")
        inverse_indices = _required_kwarg(kwargs, "inverse_indices").view(-1).to(torch.long)
        update_mask = _required_kwarg(kwargs, "update_mask")
        token_tags = _required_kwarg(kwargs, "token_tags").view(-1).to(torch.long)
        skip_mask_out_condition = bool(kwargs.get("skip_mask_out_condition", False))

        text_selected = _required_kwarg(kwargs, "prompt_embeds")
        prompt_embeds_refined = bool(kwargs.get("prompt_embeds_refined", False))

        img_pos = self._pos_ids(_required_kwarg(kwargs, "img_pos_info"), "img_pos_info")
        audio_pos = self._pos_ids(_required_kwarg(kwargs, "audio_pos_info"), "audio_pos_info")
        text_pos = self._pos_ids(
            _required_kwarg(kwargs, "text_pos_info"),
            "text_pos_info",
        )
        infer_out_pos = self._pos_ids(
            _required_kwarg(kwargs, "img_pos_for_infer_output_info"),
            "img_pos_for_infer_output_info",
        )

        psp = _required_kwarg(kwargs, "packed_seq_params")
        cu_seqlens = self._psp_field(psp, "packed_seq_params", "cu_seqlens_q").to(torch.int32)
        max_seqlen = int(self._psp_field(psp, "packed_seq_params", "max_seqlen_q"))
        refiner_psp = _required_kwarg(kwargs, "refiner_packed_seq_params")
        refiner_cu = self._psp_field(refiner_psp, "refiner_packed_seq_params", "cu_seqlens_q").to(torch.int32)
        refiner_max = int(self._psp_field(refiner_psp, "refiner_packed_seq_params", "max_seqlen_q"))

        if x.dim() != 3 or x.shape[0] != 1:
            raise ValueError(f"x must be [1, S, C], got {list(x.shape)}")
        seq_len = int(x.shape[1])
        if token_tags.shape[0] != seq_len:
            raise ValueError(f"token_tags must cover the full packed sequence ({seq_len}), got {token_tags.shape[0]}.")
        if inverse_indices.shape[0] != seq_len:
            raise ValueError(f"inverse_indices must be [{seq_len}], got {list(inverse_indices.shape)}")
        device = x.device
        refiner_cu = refiner_cu.to(device=device)
        # Branches pass a request-static value, including None for a
        # mask-free layout.  Recompute only for direct legacy callers that do
        # not provide the optional key.
        if "refiner_attn_mask" in kwargs:
            refiner_attn_mask = kwargs["refiner_attn_mask"]
        else:
            refiner_attn_mask = _packed_attention_mask(refiner_cu)
        # RoPE depends only on the packed positions and is reused for every
        # denoise timestep when supplied by the pipeline.
        rope_freqs = kwargs.get("rope_freqs")
        if rope_freqs is None:
            rope_freqs = self.rope.build_cache(img_position_ids).to(device)
        else:
            rope_freqs = rope_freqs.to(device=device)
        # Keep accepting raw frequencies for direct callers/tests.  The
        # serving pipeline passes the request-static cache produced by
        # ``MiniMaxH3Rope.build_cache`` and therefore takes this branch only
        # outside the normal serving path.
        if rope_freqs.shape[-1] <= self.arch.attention_head_dim:
            rope_freqs = torch.cat(
                (
                    torch.cos(rope_freqs).to(_BF16_DTYPE),
                    torch.sin(rope_freqs).to(_BF16_DTYPE),
                ),
                dim=-1,
            )

        decoder_input, t_emb = self._embed(
            x=x,
            audio_x=audio_x,
            text_embeddings_selected=text_selected,
            unique_timesteps=unique_timesteps.view(-1).to(device),
            img_pos=img_pos.to(device),
            audio_pos=audio_pos.to(device),
            text_pos=text_pos.to(device),
            refiner_cu_seqlens=refiner_cu,
            refiner_max_seqlen=refiner_max,
            refiner_attn_mask=refiner_attn_mask,
            seq_len=seq_len,
            device=device,
            prompt_embeds_refined=prompt_embeds_refined,
        )

        if "combined_indices" in kwargs:
            combined_indices = kwargs["combined_indices"].view(-1).to(device=device, dtype=torch.long)
        else:
            combined_indices = (inverse_indices * MINIMAX_H3_ADALN_MODALITY_NUM + token_tags.clamp(min=0)).to(device)
        inverse_indices = inverse_indices.to(device)

        hidden = decoder_input
        # All 50 blocks and the final layer consume the same timestep
        # embedding.  Move the fp32 SiLU and BF16 cast out of the repeated
        # block loop; the per-block projections now receive the exact tensor
        # they would have produced independently.
        adaln_input = nn.functional.silu(t_emb).to(_BF16_DTYPE)
        cu_seqlens = cu_seqlens.to(device)
        if "packed_attn_mask" in kwargs:
            block_attn_mask = kwargs["packed_attn_mask"]
        else:
            block_attn_mask = _packed_attention_mask(cu_seqlens)
        local_start, local_len = _sequence_parallel_local_span(seq_len)
        block_rope = rope_freqs.narrow(0, local_start, local_len)
        block_combined = combined_indices

        hidden, block_rope, block_combined = self.sp_prepare(
            hidden,
            block_rope,
            block_combined,
        )
        for block in self.blocks:
            hidden = block(
                hidden,
                t_emb=adaln_input,
                t_emb_preactivated=True,
                combined_indices=block_combined,
                rope_freqs=block_rope,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                attn_mask=block_attn_mask,
            )
        # Keep the hidden state sequence-sharded through the final projection.
        # The previous path gathered [S, hidden_size] here and only then
        # projected to the small video/audio heads.  With H3's 5376-wide
        # hidden state that collective dominates the tail of every denoise
        # step.  The final heads are row-local, so project each shard first
        # and gather only the compact logits ([video_patch_dim + audio_dim]).
        # ``block_combined`` carries the same global timestep/modality index
        # as ``inverse_indices`` and is already sharded by ``sp_prepare``;
        # deriving the local indices from it also handles any SP padding.
        local_inverse_indices = torch.div(
            block_combined,
            MINIMAX_H3_ADALN_MODALITY_NUM,
            rounding_mode="floor",
        )
        video_logits, audio_logits = self.final_layer(
            hidden,
            t_emb=adaln_input,
            inverse_indices=local_inverse_indices,
            t_emb_preactivated=True,
        )
        compact_logits = torch.cat((video_logits, audio_logits), dim=-1)
        compact_logits = self.sp_gather(compact_logits)
        video_width = self.arch.latents_dim * math.prod(self.arch.patch_size)
        video_logits = compact_logits[..., :video_width]
        audio_logits = compact_logits[..., video_width:]

        # Select target and condition rows at inference-output positions, then
        # zero the condition rows.
        video_logits = video_logits.index_select(0, infer_out_pos.to(device))
        audio_logits = audio_logits.index_select(0, audio_pos.to(device))
        if not skip_mask_out_condition:
            update_mask = update_mask.view(-1).to(device)
            if update_mask.shape[0] != video_logits.shape[0]:
                raise ValueError(f"update_mask length mismatch: {update_mask.shape[0]} != {video_logits.shape[0]}")
            video_logits = video_logits * update_mask.unsqueeze(-1)
            # Audio has no condition rows in the supported tasks, so its
            # derived update mask is all ones. Honor an explicit mask when
            # provided.
            update_audio_mask = kwargs.get("update_audio_mask")
            if update_audio_mask is not None:
                audio_logits = audio_logits * update_audio_mask.view(-1).unsqueeze(-1)
        return video_logits, audio_logits


EntryClass = MiniMaxH3DiTModel

__all__ = [
    "MINIMAX_H3_FP32_BUFFER_NAMES",
    "MINIMAX_H3_FP32_PARAM_NAMES",
    "MiniMaxH3DiTModel",
    "_reorder_grouped_qkv_to_qkv",
]
