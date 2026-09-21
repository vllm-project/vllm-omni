# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared architecture and DiT blocks for MiniMax H3 and its control branch."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionMetadata,
    PackedPaddingMetadata,
    VideoTokenLayout,
)
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available
from vllm_omni.diffusion.layers.activation import SiluAndMul
from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_qk_norm_rope
from vllm_omni.diffusion.layers.indexed_modulation import (
    indexed_gate,
    indexed_gate_rms_norm_scale_shift,
    rms_norm_indexed_scale_shift,
)
from vllm_omni.diffusion.layers.norm import RMSNorm
from vllm_omni.diffusion.layers.rope import RotaryEmbedding

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

_BF16_DTYPE = torch.bfloat16
_FP32_DTYPE = torch.float32


# AdaLN modality count: token tags carry -1 for padding and 0/1/2 for
# video/text/audio tokens (padding is clamped to 0 before the embedding
# lookup and masked out afterwards).
MINIMAX_H3_ADALN_MODALITY_NUM = 3


# Opt-in fp16-range protection for the NPU ascend_laser_attention kernel
# (consumed only via the "laser_input_scale" extra key; other backends and
# platforms ignore it). The kernel stores unscaled QK^T in an fp16 GM
# workspace, and H3's outlier activations (per-element amax in the hundreds)
# push dot products past fp16 max 65504, turning whole 128-row blocks NaN.
# 256 is a power of two, so pre-dividing q/k/v and the compensating
# kernel-scale/output multiplies are exact in floating point.
MINIMAX_H3_LASER_INPUT_SCALE = 256.0


# Packed multi-request forwards require the attention backend to actually
# consume cu_seqlens as a block-diagonal plan (not a padding-mask rebuild that
# spans the full packed row). The pipeline gates on this capability before
# packing, and ``_run_packed_attention`` re-checks it per forward; a name-only
# gate would let FLASH_ATTN's NPU/XPU code paths through even though those
# variants would silently attend across request boundaries.
def _ring_sequence_parallel_is_active(attention_layer: Attention) -> bool:
    """Match :meth:`Attention._get_active_parallel_strategy` for Ring."""
    if not getattr(attention_layer, "use_ring", False) or getattr(attention_layer, "skip_sequence_parallel", False):
        return False
    if is_forward_context_available() and not get_forward_context().sp_active:
        return False
    return True


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
    return not _ring_sequence_parallel_is_active(attention_layer)


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
        # The modular Diffusers checkpoint uses these spellings for the same
        # architecture. Normalize before constructing the existing native DiT.
        aliases = {
            "num_refiner_layers": "token_refiner_num_layers",
            "ffn_dim": "ffn_hidden_size",
            "in_channels": "latents_dim",
            "audio_in_channels": "audio_latents_dim",
            "freq_dim": "timestep_input_dim",
            "time_embed_hidden_dim": "time_embed_hidden_size",
            "rope_freq_dim": "rope_inv_freq_len",
        }
        config = {aliases.get(name, name): value for name, value in config.items()}
        fields = cls.__dataclass_fields__
        values = {name: config[name] for name in fields if name in config}
        if "patch_size" in values:
            values["patch_size"] = tuple(values["patch_size"])
        arch = cls(**values)
        if len(arch.patch_size) != 3:
            raise ValueError(f"patch_size must contain three values, got {arch.patch_size!r}")
        return arch


def _norm(size: int, *, eps: float, dtype: torch.dtype = _BF16_DTYPE) -> RMSNorm:
    # RMSNorm uses fp32 accumulation with bf16 inputs and outputs.
    # torch.nn.RMSNorm upcasts reduced-precision inputs for the variance
    # reduction, matching that accumulation semantic.
    return RMSNorm(size, eps=eps, dtype=dtype)


class MiniMaxH3Attention(nn.Module):
    # Full sparse checkpoints pin a ratio; legacy adapters use backend top-k.
    vsa_sparsity: float | None = None

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
        # VSA compression gate. A FastH3 VSA artifact assigns this projection
        # with ``.set_weight``; the dense path never builds it, so the module is
        # created only once the loader knows a VSA artifact is coming.
        self.to_gate_compress: ColumnParallelLinear | None = None
        self._gate_hidden_size = arch.hidden_size
        self._gate_quant_config = quant_config
        self._gate_prefix = f"{prefix}.to_gate_compress"
        from .attention.fastvideo_h3 import MiniMaxH3VSAImpl

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
            impl_overrides={"FASTVIDEO_VSA": MiniMaxH3VSAImpl},
        )

    def enable_vsa_gate(self) -> None:
        """Build the VSA compression gate this attention would otherwise lack.

        Called before ``load_weights`` so the artifact's ``.set_weight`` tensor
        has a parameter to land on. Zero-initialized like the Wan VSA layers, so
        a gate that never receives weights degrades to sparse-only selection
        rather than to garbage.
        """
        if self.to_gate_compress is not None:
            return
        self.to_gate_compress = ColumnParallelLinear(
            self._gate_hidden_size,
            self.total_num_heads * self.head_dim,
            bias=False,
            params_dtype=_BF16_DTYPE,
            quant_config=self._gate_quant_config,
            prefix=self._gate_prefix,
        )
        nn.init.zeros_(self.to_gate_compress.weight)

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
        vsa_prefix_segments: tuple[int, ...] = (),
        gate_compress: torch.Tensor | None = None,
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
        use_ring = _ring_sequence_parallel_is_active(self.attention)
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
            mask_free_packed_padding = not use_ring and self.attention.attn_backend.supports_packed_mask_free()
            no_mask = not use_ring and (
                self.attention.attn_backend.supports_prefix_kv_slicing or mask_free_packed_padding
            )
            # Hybrid Ulysses reshards Q to one ring partition before the ring
            # kernel runs, so a global [packed_total] mask cannot pass its
            # query-length check. Ring consumes valid_kv_length directly and
            # trims the circulated K/V blocks instead.
            if used < packed_total and not no_mask and not use_ring:
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
                "npu_attn_varlen": not use_ring,
                # fp16-range protection for the ascend_laser_attention kernel
                # (see MINIMAX_H3_LASER_INPUT_SCALE). Ignored by every other
                # backend/path.
                "laser_input_scale": MINIMAX_H3_LASER_INPUT_SCALE,
                **({"vsa_h3_sparsity": self.vsa_sparsity} if self.vsa_sparsity is not None else {}),
                # Present only for a VSA artifact; the VSA backend reads it as
                # the learned compression gate and every other backend ignores it.
                **({"gate_compress": gate_compress.unsqueeze(0)} if gate_compress is not None else {}),
                # FastH3 uses segment-pure prefix chunks. The target video and
                # its true 3-D shape remain in the shared typed video layout.
                **(
                    {"vsa_h3_prefix_segments": vsa_prefix_segments}
                    if gate_compress is not None and video_layout is not None and video_layout.video_spans
                    else {}
                ),
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
        vsa_prefix_segments: tuple[int, ...] = (),
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

        # The gate is projected from the same local rows as Q. Pure Ulysses
        # reshards it alongside Q/K/V in UlyssesParallelAttention so each VSA
        # rank receives the full sequence for its local head shard.
        gate_compress = None
        if self.to_gate_compress is not None:
            gate_result = self.to_gate_compress(x)
            gate_compress = gate_result[0] if isinstance(gate_result, tuple) else gate_result
            gate_compress = gate_compress.view(total, self.num_heads, self.head_dim)

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
            vsa_prefix_segments=vsa_prefix_segments,
            gate_compress=gate_compress,
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
        vsa_prefix_segments: tuple[int, ...] = (),
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
            vsa_prefix_segments=vsa_prefix_segments,
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
