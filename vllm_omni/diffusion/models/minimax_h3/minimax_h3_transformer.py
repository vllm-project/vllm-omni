# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
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
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionMetadata,
    PackedPaddingMetadata,
    VideoTokenLayout,
)
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.ops.minimax_h3_modulation import (
    indexed_gate,
    indexed_gate_rms_norm_scale_shift,
    indexed_scale_shift_,
    rms_norm_indexed_scale_shift,
)
from vllm_omni.diffusion.cache.cachedit import CacheDiTAdapterConfig
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,
    SequenceParallelOutput,
)
from vllm_omni.diffusion.layers.activation import SiluAndMul
from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_qk_norm_rope
from vllm_omni.diffusion.layers.norm import RMSNorm
from vllm_omni.diffusion.layers.rope import RotaryEmbedding
from vllm_omni.diffusion.models.host_weight_contract import FinalLayoutModelContract
from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )

    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


# Packed multi-request forwards require the attention backend to actually
# consume cu_seqlens as a block-diagonal plan (not a padding-mask rebuild that
# spans the full packed row). The pipeline gates on this capability before
# packing, and ``_run_packed_attention`` re-checks it per forward; a name-only
# gate would let FLASH_ATTN's NPU/XPU code paths through even though those
# variants would silently attend across request boundaries.
def _attention_isolates_packed_requests(attention_layer: Any) -> bool:
    """True if this attention layer keeps N-document packed boundaries.

    Requires a backend advertising ``supports_multi_doc_packed_varlen`` *and*
    that the layer is not running under ring sequence parallelism (the ring
    kernel dispatches through its own attention that ignores the packed
    cu_seqlens regardless of the configured backend).
    """
    backend = getattr(attention_layer, "attn_backend", None)
    if backend is None or not backend.supports_multi_doc_packed_varlen():
        return False
    return not getattr(attention_layer, "use_ring", False)


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
_LOCAL_SP_PREPARE_HOOK = "sp_input---local_sp_prepare"

# Opt-in fp16-range protection for the NPU ascend_laser_attention kernel
# (consumed only via the "laser_input_scale" extra key; other backends and
# platforms ignore it). The kernel stores unscaled QK^T in an fp16 GM
# workspace, and H3's outlier activations (per-element amax in the hundreds)
# push dot products past fp16 max 65504, turning whole 128-row blocks NaN.
# 256 is a power of two, so pre-dividing q/k/v and the compensating
# kernel-scale/output multiplies are exact in floating point.
MINIMAX_H3_LASER_INPUT_SCALE = 256.0


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
        "unique_timesteps",
        "inverse_indices",
        "update_mask",
        "update_audio_mask",
        "token_tags",
        "skip_mask_out_condition",
        "prompt_embeds",
        "img_pos_info",
        "audio_pos_info",
        "text_pos_info",
        "img_pos_for_infer_output_info",
        "packed_seq_params",
        "refiner_packed_seq_params",
        "video_token_layout",
        "rope_table",
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


def _norm(size: int, *, eps: float, dtype: torch.dtype = _BF16_DTYPE) -> RMSNorm:
    # RMSNorm uses fp32 accumulation with bf16 inputs and outputs.
    # torch.nn.RMSNorm upcasts reduced-precision inputs for the variance
    # reduction, matching that accumulation semantic.
    return RMSNorm(size, eps=eps, dtype=dtype)


def _sequence_parallel_local_span(
    seq_len: int,
    *,
    hooks_applied: bool,
) -> tuple[int, int]:
    """Return the packed-row span owned by this sequence-parallel rank."""
    from vllm_omni.diffusion.forward_context import (
        get_ulysses_mode,
        is_forward_context_available,
    )

    if not hooks_applied or not is_forward_context_available():
        return 0, seq_len
    if get_ulysses_mode(default="strict") != "strict":
        return 0, seq_len

    try:
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_allgather_parallel_world_size,
            get_ring_parallel_world_size,
            get_sequence_parallel_rank,
            get_sequence_parallel_world_size,
            get_ulysses_parallel_world_size,
        )

        world_size = int(get_sequence_parallel_world_size())
        rank = int(get_sequence_parallel_rank())
        ulysses_world_size = int(get_ulysses_parallel_world_size())
        ring_world_size = int(get_ring_parallel_world_size())
        allgather_world_size = int(get_allgather_parallel_world_size())
    except AssertionError:
        return 0, seq_len

    if world_size <= 1 or ulysses_world_size != world_size:
        return 0, seq_len
    if ring_world_size != 1 or allgather_world_size != 1:
        return 0, seq_len
    if seq_len < world_size or seq_len % world_size:
        return 0, seq_len

    chunk_size = seq_len // world_size
    start = rank * chunk_size
    return start, chunk_size


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


def _build_rope_table(freqs: torch.Tensor) -> torch.Tensor:
    """Materialize H3's packed ``[cos(freqs[:48]), sin(freqs[:48])]`` table."""
    half = freqs.shape[-1] // 2
    return torch.cat(
        (torch.cos(freqs[..., :half]), torch.sin(freqs[..., :half])),
        dim=-1,
    ).to(_BF16_DTYPE)


class MiniMaxH3TimeEmbedder(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        *,
        prefix: str,
    ) -> None:
        super().__init__()
        self.frequency_embedding_size = arch.timestep_input_dim
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
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=_FP32_DTYPE, device=t.device) / half)
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
        role: str = "self",
        role_category: str | None = None,
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
        self.rot_dim = 6 * arch.rope_inv_freq_len
        self.q_norm = _norm(arch.attention_head_dim, eps=arch.qk_norm_eps)
        self.k_norm = _norm(arch.attention_head_dim, eps=arch.qk_norm_eps)
        self.rope = RotaryEmbedding(is_neox_style=True, half_head_dim=False)
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
            # Packed rows reach the impl as [B, S, N, D].
            qkv_layout="BSND",
            role=role,
            role_category=role_category,
            skip_sequence_parallel=skip_sequence_parallel,
            prefix=prefix,
        )

    def _apply_rope(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Rotate the first rot_dim head dims; pass the rest through.

        x: [T, heads, head_dim]; freqs: [T, rot_dim]. In the unfused path, cos/sin
        are cast to the activation dtype before the elementwise math.
        """
        rot_dim = self.rot_dim
        x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
        cos = torch.cos(freqs).to(x.dtype)  # [T, rot_dim]
        sin = torch.sin(freqs).to(x.dtype)
        x_rot = self.rope(x_rot, cos, sin)
        return torch.cat((x_rot, x_pass), dim=-1)

    @torch.compiler.disable
    def _run_packed_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        packed_total: int,
        num_requests: int = 1,
        video_layout: VideoTokenLayout | None = None,
    ) -> torch.Tensor:
        """Run packed attention as a small eager island.

        The scalar packed-layout metadata and backend-specific attention
        kernels are intentionally opaque to Dynamo. Keeping this boundary
        narrow lets regional compile fuse projections, norms, RoPE, and the
        surrounding DiT block without repeated graph breaks.
        """
        # max_seqlen is already the longest packed document length. Do not read
        # the CUDA cu_seqlens scalars here: this function runs once per layer
        # and .item() would serialize every attention launch. ``num_requests``
        # is carried as a Python int for the same reason.
        if not 0 < max_seqlen <= packed_total:
            raise ValueError(
                f"max_seqlen must be within the packed sequence, got {max_seqlen} for length {packed_total}"
            )
        attn_mask = None
        mask_free_packed_padding = False
        if num_requests > 1:
            # A step-mode batch packs one document per request, so its valid
            # rows are block-diagonal rather than a prefix: neither a KV prefix
            # length nor a 1-D key mask can describe them. Such a layout is
            # only correct on a backend that actually attends by cu_seqlens as
            # a block-diagonal plan. Check the capability (not the backend
            # name): FLASH_ATTN's NPU/XPU variants would otherwise silently
            # fall back to a padding-mask rebuild that spans the whole packed
            # row and attend across request boundaries.
            if not _attention_isolates_packed_requests(self.attention):
                backend_name = self.attention.attn_backend.get_name()
                raise ValueError(
                    f"MiniMax H3 packed a {num_requests}-request batch, but the resolved "
                    f"attention ({backend_name}, use_ring={getattr(self.attention, 'use_ring', False)}) "
                    "does not isolate multi-document packed cu_seqlens. Run one request "
                    "per forward on this backend."
                )
            used = packed_total
        else:
            used = min(max_seqlen, packed_total)
            # Ring attention can dispatch to a different implementation from the
            # configured backend, so the no-mask fast paths are local-only.
            # supports_prefix_kv_slicing: backend slices K/V itself (cuDNN).
            # supports_packed_mask_free: backend consumes the packed metadata
            # without ever reading attn_mask (CUDA packed varlen, NPU
            # npu_attn_varlen opt-in with its own fallback rebuild).
            use_ring = getattr(self.attention, "use_ring", False)
            mask_free_packed_padding = not use_ring and self.attention.attn_backend.supports_packed_mask_free()
            no_mask = not use_ring and (
                self.attention.attn_backend.supports_prefix_kv_slicing or mask_free_packed_padding
            )
            if used < packed_total and not no_mask:
                attn_mask = torch.arange(packed_total, device=q.device)[None] < used
        metadata = AttentionMetadata(
            attn_mask=attn_mask,
            packed_padding=(
                PackedPaddingMetadata(
                    q_length=used,
                    kv_length=used,
                    cu_seqlens_q=cu_seqlens[:2],
                    cu_seqlens_k=cu_seqlens[:2],
                )
                if mask_free_packed_padding
                else None
            ),
            extra={
                "cu_seqlens_q": cu_seqlens,
                "cu_seqlens_k": cu_seqlens,
                "max_seqlen_q": max_seqlen,
                "max_seqlen_k": max_seqlen,
                "valid_kv_length": used,
                # Opt the NPU flash backend into the packed varlen path so the
                # quadratic full_qk mask is never materialized. Ring attention
                # is excluded: it keeps the aligned padding rows for its
                # fixed-size P2P buffers and still needs the mask.
                "npu_attn_varlen": not getattr(self.attention, "use_ring", False),
                # fp16-range protection for the ascend_laser_attention kernel
                # (see MINIMAX_H3_LASER_INPUT_SCALE). Ignored by every other
                # backend/path.
                "laser_input_scale": MINIMAX_H3_LASER_INPUT_SCALE,
            },
            video_layout=video_layout,
        )
        return self.attention(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            metadata,
        ).squeeze(0)

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_table: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        packed_total: int | None = None,
        num_requests: int = 1,
        sp_seq_lens: list[int] | None = None,
        video_layout: VideoTokenLayout | None = None,
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
        if rope_table is None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        else:
            q, k = fused_qk_norm_rope(
                q,
                k,
                self.q_norm.weight,
                self.k_norm.weight,
                rope_table,
                self.q_norm.variance_epsilon,
            )

        # Each request contributes a document for its rows plus one for any
        # nonempty alignment padding. Local/Ulysses backends unpad it, while
        # Ring keeps aligned rows for fixed-size P2P buffers.
        out = self._run_packed_attention(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            # Before Ulysses, q contains only this rank's row shard. The
            # backend receives the global sequence after all-to-all, so carry
            # its Python length explicitly instead of inferring it from q.
            packed_total=packed_total if packed_total is not None else q.shape[0],
            num_requests=num_requests,
            video_layout=video_layout,
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
        self.act_fn = SiluAndMul()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.fc1(x)
        hidden = self.act_fn(hidden)
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

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """t_emb: [M, t_dim] -> expand_ratio tensors of [M*modality_num, H]."""
        x = nn.functional.silu(t_emb)
        x, _ = self.linear(x.to(_BF16_DTYPE))
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
            role="minimax_h3.token_refiner",
            role_category="self",
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
        num_requests: int = 1,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            rope_table=None,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            num_requests=num_requests,
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
        num_requests: int = 1,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, num_requests=num_requests)
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
        # The prefix also carries the block index that block-sparse attention
        # backends match against their skip_layers selector.
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
        rope_table: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        packed_total: int,
        num_requests: int = 1,
        sp_seq_lens: list[int] | None = None,
        video_layout: VideoTokenLayout | None = None,
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
        ) = self.adaln_proj(t_emb)

        residual = x
        h = rms_norm_indexed_scale_shift(
            x,
            self.norm1.weight,
            shift_msa,
            scale_msa,
            combined_indices,
            self.norm1.variance_epsilon,
        )
        h = self.attn(
            h,
            rope_table=rope_table,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            packed_total=packed_total,
            num_requests=num_requests,
            sp_seq_lens=sp_seq_lens,
            video_layout=video_layout,
        )
        x, h = indexed_gate_rms_norm_scale_shift(
            residual,
            gate_msa,
            h,
            self.norm2.weight,
            shift_mlp,
            scale_mlp,
            combined_indices,
            self.norm2.variance_epsilon,
        )
        residual = x
        h = self.mlp(h)
        return indexed_gate(residual, gate_mlp, h, combined_indices)


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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [T, H] -> (video_logits [T, 96] fp32, audio_logits [T, 32] fp32).

        Apply single-modality shift/scale AdaLN to the final normalized
        activations, cast to fp32, then apply both output heads to all rows.
        """
        shift, scale = self.adaln_proj(t_emb)
        h = self.norm(x)
        h = indexed_scale_shift_(h, shift, scale, inverse_indices)
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
        rope_table: torch.Tensor,
        combined_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return hidden_states, rope_table, combined_indices


class MiniMaxH3SPGather(nn.Module):
    """Explicit boundary for restoring packed rows after the block stack."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class MiniMaxH3DiTModel(nn.Module):
    # Loading is tensor-complete: constructor state plus final-layout
    # parameters and persistent buffers is sufficient to reconstruct a ready
    # inference model. The model-specific validator below checks the preserved
    # FP32 portion after a lease-backed restore commits.
    host_weight_restore_contract = FinalLayoutModelContract(
        implementation_id="minimax-h3-dit",
        version="1",
    )

    _cache_dit_adapter_config = CacheDiTAdapterConfig(
        block_forward_patterns={"blocks": ForwardPattern.Pattern_3},
        # H3 is CFG-distilled and performs one transformer forward per step.
        has_separate_cfg=False,
        check_forward_pattern=False,
    )
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
            0: SequenceParallelInput(
                split_dim=0,
                expected_dims=2,
                split_output=True,
            ),
            1: SequenceParallelInput(
                split_dim=0,
                expected_dims=2,
                split_output=True,
            ),
            2: SequenceParallelInput(
                split_dim=0,
                expected_dims=1,
                split_output=True,
            ),
        },
        "local_sp_prepare": {
            2: SequenceParallelInput(
                split_dim=0,
                expected_dims=1,
                split_output=True,
            ),
        },
        "sp_gather": SequenceParallelOutput(gather_dim=0, expected_dims=2),
    }
    # The checkpoint already stores qkv and the MLP gate/up as single tensors
    # (see the reordering in load_weights), so there are no unfused names for
    # quantization or LoRA to map onto. Address the fused layers directly, e.g.
    # ignored_layers=["blocks.0.attn.qkv_proj"].
    packed_modules_mapping = {}
    # Turbo LoRA checkpoints publish separate Q/K/V adapters. This declaration
    # lets the legacy diffusion LoRA manager bind them to the packed QKV layer;
    # it does not change the fused base-checkpoint loading path above.
    stacked_params_mapping = (
        (".attn.qkv_proj", ".attn.to_q", "q"),
        (".attn.qkv_proj", ".attn.to_k", "k"),
        (".attn.qkv_proj", ".attn.to_v", "v"),
    )

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
        self.local_sp_prepare = MiniMaxH3SPPrepare()
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

    def _rope_local_span(self, seq_len: int) -> tuple[int, int]:
        """Return the sequence-parallel rows owned by this DiT rank."""
        local_sp_registry = getattr(self.local_sp_prepare, "_hook_registry", None)
        hooks_applied = local_sp_registry is not None
        if local_sp_registry is not None:
            local_sp_hook = local_sp_registry.get_hook(_LOCAL_SP_PREPARE_HOOK)
            hooks_applied = local_sp_hook is not None
        return _sequence_parallel_local_span(
            seq_len,
            hooks_applied=hooks_applied,
        )

    def prepare_rope_table(
        self,
        img_position_ids: torch.Tensor,
        *,
        seq_len: int,
    ) -> torch.Tensor:
        """Build the static local RoPE table once for one denoise branch.

        A MiniMax-H3 denoise branch reuses its packed position IDs at every
        scheduler step. The returned table is local to the current sequence-
        parallel rank, and therefore must be built by the model that will
        consume it rather than cached globally across requests or ranks.
        """
        local_start, local_len = self._rope_local_span(seq_len)
        rope_position_ids = img_position_ids.narrow(1, local_start, local_len)
        return _build_rope_table(self.rope(rope_position_ids).to(img_position_ids.device))

    def _validate_prepared_rope_table(
        self,
        rope_table: torch.Tensor,
        *,
        local_len: int,
        device: torch.device,
    ) -> None:
        expected_width = 6 * self.arch.rope_inv_freq_len
        if rope_table.dim() != 2 or tuple(rope_table.shape) != (local_len, expected_width):
            raise ValueError(
                "rope_table must be [local_seq_len, rotary_dim] for the current "
                f"sequence-parallel rank, got {list(rope_table.shape)}; expected "
                f"[{local_len}, {expected_width}]."
            )
        if rope_table.device != device:
            raise ValueError(f"rope_table device {rope_table.device} must match x device {device}.")
        if rope_table.dtype != _BF16_DTYPE:
            raise ValueError(f"rope_table must be {_BF16_DTYPE}, got {rope_table.dtype}.")

    def post_load_weights(self) -> None:
        for name, param in self.named_parameters():
            if name in MINIMAX_H3_FP32_PARAM_NAMES and param.dtype != _FP32_DTYPE:
                raise ValueError(f"{name} must stay fp32 after load, got {param.dtype}.")
        for name, buffer in self.named_buffers():
            if name in MINIMAX_H3_FP32_BUFFER_NAMES and buffer.dtype != _FP32_DTYPE:
                raise ValueError(f"{name} must stay fp32 after load, got {buffer.dtype}.")

    def validate_restored_host_weights(self) -> None:
        """Validate mixed-precision invariants after lease-backed restore."""
        self.post_load_weights()

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

    @staticmethod
    def _psp_optional(psp: Any, field: str, default: Any) -> Any:
        value = psp.get(field) if isinstance(psp, dict) else getattr(psp, field, None)
        return default if value is None else value

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
        seq_len: int,
        device: torch.device,
        local_span: tuple[int, int],
        num_requests: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build this rank's packed multimodal embedding rows.

        Returns (decoder_input [S_local, H] bf16, t_emb [M, t_dim] fp32).

        ``num_requests`` defaults to a single packed request so callers that
        pre-date the continuous-batching change (e.g. TeaCache's extractor
        contract) do not silently miss the kwarg. ``forward()`` reads the real
        value from ``packed_seq_params["num_requests"]``.
        """
        local_start, local_len = local_span
        local_end = local_start + local_len
        local_only = local_len != seq_len
        if local_only:
            img_mask = (img_pos >= local_start) & (img_pos < local_end)
            audio_mask = (audio_pos >= local_start) & (audio_pos < local_end)
            text_mask = (text_pos >= local_start) & (text_pos < local_end)
            img_global_pos = img_pos[img_mask]
            audio_global_pos = audio_pos[audio_mask]
            img_local_pos = img_global_pos - local_start
            audio_local_pos = audio_global_pos - local_start
            text_local_pos = text_pos[text_mask] - local_start
            text_local_indices = torch.nonzero(text_mask, as_tuple=False).view(-1)
        else:
            img_global_pos = img_pos
            audio_global_pos = audio_pos
            img_local_pos = img_pos
            audio_local_pos = audio_pos
            text_local_pos = text_pos
            text_local_indices = None

        # Latent embedders stay fp32 in and out; their outputs are cast to the
        # bf16 sequence dtype only during indexed scattering.
        x_rows = x.view(-1, x.shape[-1]).index_select(0, img_global_pos).to(_FP32_DTYPE)
        video_embed, _ = self.video_patch_proj(x_rows)
        audio_rows = audio_x.view(-1, audio_x.shape[-1])
        audio_rows = audio_rows.index_select(0, audio_global_pos).to(_FP32_DTYPE)
        audio_embed, _ = self.audio_patch_proj(audio_rows)

        text_rows = text_embeddings_selected.to(device=device, dtype=_BF16_DTYPE)
        text_embed, _ = self.condition_proj(text_rows)
        text_embed = self.token_refiner(
            text_embed,
            cu_seqlens=refiner_cu_seqlens,
            max_seqlen=refiner_max_seqlen,
            num_requests=num_requests,
        )
        if text_local_indices is not None:
            text_embed = text_embed.index_select(0, text_local_indices)

        embeddings = torch.zeros(
            (local_len, self.hidden_size),
            device=device,
            dtype=_BF16_DTYPE,
        )
        embeddings.index_add_(
            0,
            text_local_pos,
            text_embed.to(_BF16_DTYPE)[: text_local_pos.shape[0]],
        )
        embeddings.index_add_(
            0,
            img_local_pos,
            video_embed.to(_BF16_DTYPE)[: img_local_pos.shape[0]],
        )
        embeddings.index_add_(
            0,
            audio_local_pos,
            audio_embed.to(_BF16_DTYPE)[: audio_local_pos.shape[0]],
        )

        t_emb = self.time_embedder(unique_timesteps)
        return embeddings, t_emb

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
        # How many requests share this packed sequence. Carried as a host int so
        # attention never reads cu_seqlens scalars off the device; a producer
        # that omits it is packing a single request.
        num_requests = int(self._psp_optional(psp, "num_requests", 1))
        refiner_psp = _required_kwarg(kwargs, "refiner_packed_seq_params")
        refiner_cu = self._psp_field(refiner_psp, "refiner_packed_seq_params", "cu_seqlens_q").to(torch.int32)
        refiner_max = int(self._psp_field(refiner_psp, "refiner_packed_seq_params", "max_seqlen_q"))
        video_layout = kwargs.get("video_token_layout")

        if x.dim() != 3 or x.shape[0] != 1:
            raise ValueError(f"x must be [1, S, C], got {list(x.shape)}")
        seq_len = int(x.shape[1])
        if token_tags.shape[0] != seq_len:
            raise ValueError(f"token_tags must cover the full packed sequence ({seq_len}), got {token_tags.shape[0]}.")
        if inverse_indices.shape[0] != seq_len:
            raise ValueError(f"inverse_indices must be [{seq_len}], got {list(inverse_indices.shape)}")
        device = x.device
        local_span = self._rope_local_span(seq_len)
        local_start, local_len = local_span
        rope_table = kwargs.get("rope_table")
        if rope_table is None:
            if current_omni_platform.is_npu():
                rope_table = self.prepare_rope_table(
                    img_position_ids,
                    seq_len=seq_len,
                )
            else:
                # Keep CUDA/CPU numerically and structurally identical to the
                # main-branch reference path used by the H100 accuracy suite.
                rope_position_ids = img_position_ids.narrow(1, local_start, local_len)
                rope_table = _build_rope_table(self.rope(rope_position_ids).to(device))
        else:
            self._validate_prepared_rope_table(
                rope_table,
                local_len=local_len,
                device=device,
            )

        decoder_input, t_emb = self._embed(
            x=x,
            audio_x=audio_x,
            text_embeddings_selected=text_selected,
            unique_timesteps=unique_timesteps.view(-1).to(device),
            img_pos=img_pos.to(device),
            audio_pos=audio_pos.to(device),
            text_pos=text_pos.to(device),
            refiner_cu_seqlens=refiner_cu.to(device),
            refiner_max_seqlen=refiner_max,
            num_requests=num_requests,
            seq_len=seq_len,
            device=device,
            local_span=local_span,
        )

        combined_indices = (inverse_indices * MINIMAX_H3_ADALN_MODALITY_NUM + token_tags.clamp(min=0)).to(device)
        inverse_indices = inverse_indices.to(device)

        hidden = decoder_input
        cu_seqlens = cu_seqlens.to(device)
        block_rope = rope_table
        block_combined = combined_indices

        if local_len == seq_len:
            hidden, block_rope, block_combined = self.sp_prepare(
                hidden,
                block_rope,
                block_combined,
            )
        else:
            hidden, block_rope, block_combined = self.local_sp_prepare(
                hidden,
                block_rope,
                block_combined,
            )
        for block in self.blocks:
            hidden = block(
                hidden,
                t_emb=t_emb,
                combined_indices=block_combined,
                rope_table=block_rope,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                packed_total=seq_len,
                num_requests=num_requests,
                video_layout=video_layout,
            )
        if local_len == seq_len:
            hidden = self.sp_gather(hidden)
            video_logits, audio_logits = self.final_layer(
                hidden,
                t_emb=t_emb,
                inverse_indices=inverse_indices,
            )
        else:
            local_inverse_indices = inverse_indices.narrow(
                0,
                local_start,
                local_len,
            )
            video_logits, audio_logits = self.final_layer(
                hidden,
                t_emb=t_emb,
                inverse_indices=local_inverse_indices,
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
