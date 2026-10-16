# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flux3 image DiT.

Architecture: 3-phase transformer — per-stream ModeBlocks (Phase 1),
joint SingleStreamBlocks across image streams + text (Phase 2), optional late
ModeBlocks (Phase 3, depth=0 by default).

TP: QKVParallelLinear for attention; ColumnParallelLinear+RowParallelLinear for MLP.
SP: not yet wired (_sp_plan = {}).
"""

import contextlib
import math
from collections.abc import Iterable
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.models.interfaces import SupportsQuant
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.cache.cachedit import CacheDiTAdapterConfig
from vllm_omni.diffusion.cache.cachedit.config import ForwardPattern
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.layers.adalayernorm import AdaLayerNorm
from vllm_omni.diffusion.layers.fused_qk_norm_rope import (
    _fused_cuda_supported,
    fused_qk_norm_rope,
    fused_qk_norm_rope_min_tokens,
)
from vllm_omni.diffusion.models.flux3.flux3_kernel_adapter import Flux3KernelAdapter
from vllm_omni.diffusion.models.flux3.flux3_kernel_adapter_factory import build_flux3_kernel_adapter

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


# Use the fused path for every supported shape by default. The environment
# override can retain the unfused path for numerical validation.
_FUSED_QK_NORM_ROPE_MIN_TOKENS = fused_qk_norm_rope_min_tokens(0)


def _block_range(name: str):
    # NVTX ranges inside block forwards only when eager: torch.compile cannot trace the
    # range context manager, and a graph break inside it makes the whole block eager.
    return contextlib.nullcontext() if torch.compiler.is_compiling() else torch.cuda.nvtx.range(name)


def _join_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


_QKV_PACKED_MODULES_MAPPING = {"qkv": ["q_proj", "k_proj", "v_proj"]}


def _is_nvfp4_quant_config(quant_config: "QuantizationConfig | None") -> bool:
    """Return True for NVFP4 configs that need per-projection Q/K/V scale handling."""
    if quant_config is None:
        return False
    get_name = getattr(quant_config, "get_name", None)
    return callable(get_name) and get_name() == "modelopt_fp4"


def _resolve_fuse_qkv(
    quant_config: "QuantizationConfig | None",
    kernel_backend: str = "vllm",
    prims_ts_target: bool = False,
) -> bool:
    """Return the QKV layout compatible with the selected quantization method."""
    if _is_nvfp4_quant_config(quant_config):
        # PrimsTS preserves the separate projection scales in its fused QKV
        # kernel. Other NVFP4 blocks need split projections for those scales.
        return kernel_backend == "prims_ts" and prims_ts_target
    return True


def _build_qkv_weights_mapper(
    fused_by_prefix: dict[str, bool],
) -> tuple[WeightsMapper, list[tuple[str, str, str]]]:
    """Map checkpoint Q/K/V names to each block's selected projection layout."""
    projections = {"q_proj": "q", "k_proj": "k", "v_proj": "v"}
    stacked: dict[str, tuple[str, str]] = {}
    substr: dict[str, str] = {}
    for prefix, fused in fused_by_prefix.items():
        if fused:
            for name, shard in projections.items():
                stacked[f"{prefix}.{name}."] = (f"{prefix}.qkv.", shard)
            substr[f"{prefix}.to_qkv."] = f"{prefix}.qkv."
        else:
            for name in projections:
                substr[f"{prefix}.{name}."] = f"{prefix}.qkv.{name}."

    mapper = WeightsMapper(
        orig_to_new_stacked=stacked,
        orig_to_new_substr={
            **substr,
            ".query_norm.scale": ".query_norm.weight",
            ".key_norm.scale": ".key_norm.weight",
            ".adaLN_modulation.1.": ".adaLN_modulation.",
        },
    )
    # LoRA discovers the logical Q/K/V slices by module leaf name.
    stacked_params_mapping = [(".qkv.", f".{name}.", shard) for name, shard in projections.items()] if stacked else []
    return mapper, stacked_params_mapping


def _build_split_qkv_checkpoint_mapper(fused_by_prefix: dict[str, bool]) -> WeightsMapper | None:
    """Map unfused checkpoint Q/K/V names into the runtime _SeparateQKV module tree."""
    projections = ("q_proj", "k_proj", "v_proj")
    substr = {
        f"{prefix}.{name}": f"{prefix}.qkv.{name}"
        for prefix, fused in fused_by_prefix.items()
        if not fused
        for name in projections
    }
    return WeightsMapper(orig_to_new_substr=substr) if substr else None


def _build_lora_weights_mapper(fused_by_prefix: dict[str, bool]) -> WeightsMapper:
    """Map raw LoRA Q/K/V names to the selected Flux3 module layout."""
    substr = {
        "diffusion_model.": "",
        "dit.": "",
    }
    substr.update(
        {
            f"{prefix}.{name}.": f"{prefix}.qkv.{name}."
            for prefix, fused in fused_by_prefix.items()
            if not fused
            for name in ("q_proj", "k_proj", "v_proj")
        }
    )
    return WeightsMapper(orig_to_new_substr=substr)


def _set_quant_config_packed_qkv_mapping(
    quant_config: "QuantizationConfig | None",
    packed_mapping: dict[str, list[str]],
) -> None:
    """Keep quantization metadata's packed-module map aligned with Flux3's QKV layout."""
    if quant_config is None:
        return
    quant_packed_mapping = getattr(quant_config, "packed_modules_mapping", None)
    if quant_packed_mapping is None:
        return
    quant_packed_mapping.pop("qkv", None)
    quant_packed_mapping.update(packed_mapping)


def _iter_qkv_prefixes(
    modalities: Iterable[str],
    depth: int,
    depth_single_blocks: int,
    depth_late_blocks: int,
) -> Iterable[str]:
    """Yield every Flux3 block prefix that owns a QKV projection."""
    for mod_key in modalities:
        for i in range(depth):
            yield f"content_mode_blocks.{mod_key}.{i}"
    for i in range(depth):
        yield f"txt_mode_blocks.{i}"
    for i in range(depth_single_blocks):
        yield f"single_blocks.{i}"
    for mod_key in modalities:
        for i in range(depth_late_blocks):
            yield f"late_content_mode_blocks.{mod_key}.{i}"
    for i in range(depth_late_blocks):
        yield f"late_txt_mode_blocks.{i}"


# ── RoPE ──────────


def _rope(pos: Tensor, dim: int, theta: int) -> tuple[Tensor, Tensor]:
    """Compute the cosine and sine halves of an adjacent-pair RoPE table.

    pos: (B, T) integer positions
    Returns: two (B, T, dim//2) float32 tensors
    """
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.float(), omega)
    return torch.cos(out).float(), torch.sin(out).float()


def _apply_rope(xq: Tensor, xk: Tensor, rope_table: Tensor) -> tuple[Tensor, Tensor]:
    """Apply adjacent-pair RoPE from a packed cosine/sine table.

    xq, xk: (B, T, H, head_dim)
    rope_table: (B, T, head_dim), packed as [cosines | sines]
    Returns: rotated xq, xk in same shape and dtype as input.
    """
    rotary_half = rope_table.shape[-1] // 2
    cos = rope_table[..., :rotary_half].unsqueeze(2)
    sin = rope_table[..., rotary_half:].unsqueeze(2)

    def rotate(x: Tensor) -> Tensor:
        pairs = x.float().unflatten(-1, (-1, 2))
        first = pairs[..., 0] * cos - pairs[..., 1] * sin
        second = pairs[..., 1] * cos + pairs[..., 0] * sin
        return torch.stack((first, second), dim=-1).flatten(-2).type_as(x)

    return rotate(xq), rotate(xk)


class _EmbedND(nn.Module):
    """Multi-axis (4D) RoPE positional embedding."""

    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        # ids: (B, T, num_axes) — integer position ids
        # returns: (B, T, dim), packed as [all cosines | all sines]
        axis_tables = [_rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(len(self.axes_dim))]
        cos = torch.cat([table[0] for table in axis_tables], dim=-1)
        sin = torch.cat([table[1] for table in axis_tables], dim=-1)
        return torch.cat((cos, sin), dim=-1)


# ── Sinusoidal timestep features ──────────────────────────────────────────


def _timestep_embedding(t: Tensor, dim: int = 256) -> Tensor:
    """Convert per-token scalar timesteps in [0,1] to sinusoidal features.

    t: (...) float tensor of timestep values
    Returns: (..., dim) tensor of sinusoidal features, cast back to ``t``'s dtype
    """
    assert dim % 2 == 0
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    # Apply the 1000x time-factor in t's dtype before conversion.
    t = t * 1000.0
    args = t.float().unsqueeze(-1) * freqs
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    # Restore the timestep dtype so the embedding feeds a bf16 ``time_in`` Linear
    # without a dtype mismatch.
    if torch.is_floating_point(t):
        embedding = embedding.to(t.dtype)
    return embedding


# ── Plain modules (no TP sharding) ────────────────────────────────────────


class _MLPEmbedder(nn.Module):
    """Two-layer MLP for timestep and vector conditioning: in → SiLU → hidden → hidden."""

    def __init__(self, in_dim: int, hidden_dim: int, quant_config=None, prefix: str = ""):
        super().__init__()
        self.in_layer = ReplicatedLinear(
            in_dim,
            hidden_dim,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "in_layer"),
        )
        self.out_layer = ReplicatedLinear(
            hidden_dim,
            hidden_dim,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "out_layer"),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_layer(x)
        x = self.out_layer(F.silu(x))
        return x


class _QKNorm(nn.Module):
    """Per-head query/key RMS norm applied before RoPE."""

    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim, eps=1e-6)
        self.key_norm = RMSNorm(dim, eps=1e-6)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        return self.query_norm(q).to(v), self.key_norm(k).to(v)


def _qk_norm_rope(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    norm: _QKNorm,
    rope_table: Tensor,
) -> tuple[Tensor, Tensor]:
    """Fuse Q/K RMSNorm and adjacent-pair RoPE when CUDA supports it.

    The separate operations remain as a compatibility fallback for unsupported
    dtypes, platforms, and geometries.
    """
    query_norm = norm.query_norm
    key_norm = norm.key_norm
    head_dim = q.shape[-1]
    if (
        isinstance(query_norm, RMSNorm)
        and isinstance(key_norm, RMSNorm)
        and q.shape[0] * q.shape[1] >= _FUSED_QK_NORM_ROPE_MIN_TOKENS
        and _fused_cuda_supported(q, k, head_dim, rope_table.shape[-1], interleaved=True)
    ):
        batch, tokens, query_heads, _ = q.shape
        key_heads = k.shape[2]
        fused_q, fused_k = fused_qk_norm_rope(
            q.reshape(batch * tokens, query_heads, head_dim),
            k.reshape(batch * tokens, key_heads, head_dim),
            query_norm.weight,
            key_norm.weight,
            rope_table.reshape(batch * tokens, head_dim),
            query_norm.variance_epsilon,
            interleaved=True,
        )
        return (
            fused_q.reshape(batch, tokens, query_heads, head_dim),
            fused_k.reshape(batch, tokens, key_heads, head_dim),
        )

    q, k = norm(q, k, v)
    return _apply_rope(q, k, rope_table)


class _Modulation(nn.Module):
    """AdaLN-Zero modulation: SiLU → Linear(dim, 3*dim) → (shift, scale, gate).

    Inputs a per-token conditioning vector; outputs three modulation tensors
    each with the same shape as the input.
    """

    def __init__(self, dim: int, quant_config=None, prefix: str = ""):
        super().__init__()
        self.lin = ReplicatedLinear(
            dim, 3 * dim, bias=False, return_bias=False, quant_config=quant_config, prefix=_join_prefix(prefix, "lin")
        )

    def forward(self, vec: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # vec: (B, T, dim) or (B, dim)
        out = self.lin(F.silu(vec))
        if out.ndim == 2:
            out = out.unsqueeze(1)  # (B, 1, 3*dim)
        shift, scale, gate = out.chunk(3, dim=-1)
        return shift, scale, gate


class _LastLayer(nn.Module):
    """Per-modality final layer: AdaLN norm + linear projection to latent channels."""

    def __init__(self, hidden_size: int, out_channels: int, quant_config=None, prefix: str = ""):
        super().__init__()
        self.norm_final = AdaLayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = ColumnParallelLinear(
            hidden_size,
            out_channels,
            bias=False,
            gather_output=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "linear"),
        )
        self.adaLN_silu = nn.SiLU()
        self.adaLN_modulation = ReplicatedLinear(
            hidden_size,
            2 * hidden_size,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "adaLN_modulation"),
        )

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        if vec.ndim == 2:
            vec = vec[:, None, :]
        mod = self.adaLN_modulation(self.adaLN_silu(vec))
        shift, scale = mod.chunk(2, dim=-1)
        x = self.norm_final(x, scale, shift)
        return self.linear(x)


# ── Blocks with TP-aware projections ──────────────────────────────────────


class _SeparateQKV(nn.Module):
    """Unfused q/k/v as three separate ``ColumnParallelLinear`` GEMMs.

    Returns the concatenated projection so callers treat it like the fused
    ``QKVParallelLinear`` (``qkv, _ = self.qkv(x); q, k, v = qkv.chunk(3, -1)``).
    Same math as the fused path; differs only by matmul rounding. Used when
    a quantized checkpoint needs separate projection scales.
    """

    def __init__(self, hidden_size, head_dim, num_heads, quant_config, prefix):
        super().__init__()
        out = num_heads * head_dim
        self.q_proj = ColumnParallelLinear(
            hidden_size,
            out,
            bias=False,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "q_proj"),
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            out,
            bias=False,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "k_proj"),
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            out,
            bias=False,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "v_proj"),
        )
        local_out = getattr(self.q_proj, "output_size_per_partition", out)
        self.num_heads = local_out // head_dim
        self.num_kv_heads = self.num_heads

    def forward(self, x: Tensor) -> tuple[Tensor, None]:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return torch.cat([q, k, v], dim=-1), None


def _build_qkv(fused_qkv: bool, hidden_size, head_dim, num_heads, quant_config, prefix):
    """Fused ``QKVParallelLinear`` (default) or three separate ``ColumnParallelLinear``.

    Both expose ``.num_heads`` / ``.num_kv_heads`` and return ``(qkv, bias)`` so the
    block forward is identical; the children live under the ``qkv`` prefix either way.
    """
    if fused_qkv:
        return QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "qkv"),
        )
    return _SeparateQKV(hidden_size, head_dim, num_heads, quant_config, _join_prefix(prefix, "qkv"))


class _ModeBlock(nn.Module):
    """Per-modality self-attention block (Phases 1 and 3).

    Fused attention + SwiGLU FFN gated by a single AdaLN-Zero modulation.
    """

    def __init__(
        self,
        parallel_config: "DiffusionParallelConfig",
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
        fused_qkv: bool = True,
        kernel_adapter: Flux3KernelAdapter | None = None,
    ):
        super().__init__()
        self.kernel_adapter = kernel_adapter or Flux3KernelAdapter(_qk_norm_rope)
        self.head_dim = hidden_size // num_heads
        mlp_hidden = int(hidden_size * mlp_ratio)

        self.norm1 = AdaLayerNorm(hidden_size, elementwise_affine=False, eps=eps)

        self.qkv = _build_qkv(fused_qkv, hidden_size, self.head_dim, num_heads, quant_config, prefix)
        self.num_heads_local: int = self.qkv.num_heads
        self.num_kv_heads_local: int = self.qkv.num_kv_heads

        self.norm = _QKNorm(self.head_dim)

        self.attn_out = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "attn_out"),
        )

        # SwiGLU FFN: mlp_hidden*2 out from mlp_in (gate + up), then mlp_hidden in to mlp_out
        self.mlp_in = ColumnParallelLinear(
            hidden_size,
            mlp_hidden * 2,
            bias=False,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "mlp_in"),
        )
        self.mlp_out = RowParallelLinear(
            mlp_hidden,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "mlp_out"),
        )
        self.mlp_act = SiluAndMul()

        self.attn = Attention(
            num_heads=self.num_heads_local,
            head_size=self.head_dim,
            softmax_scale=self.head_dim**-0.5,
            causal=False,
            num_kv_heads=self.num_kv_heads_local,
            role="self",
        )
        self.kernel_plan = self.kernel_adapter.configure_block(self, prefix)

    def forward(self, x: Tensor, pe: Tensor, mod: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """
        x:   (B, T, hidden_size) — modality tokens
        pe:  (B, T, head_dim) — packed [cosines | sines] RoPE table
        mod: (shift, scale, gate) precomputed AdaLN modulation, each (B, T, hidden_size)
        """
        mod_shift, mod_scale, mod_gate = mod
        x_mod = self.norm1(x, mod_scale, mod_shift)
        prepared_input = self.kernel_adapter.prepare_input(self, x_mod)

        # Attention path
        q, k, v = self.kernel_adapter.qkv_qknorm_rope(self, prepared_input, pe)
        with _block_range("flux3/attn"):
            attn_out = self.attn(q, k, v, None)
        attn_out = attn_out.flatten(2, 3).to(x.dtype)
        attn_out = self.kernel_adapter.attention_output(self, attn_out)

        # SwiGLU FFN path (same x_mod input, parallel with attention)
        mlp_out = self.kernel_adapter.mlp(self, prepared_input)

        return x + mod_gate * (attn_out + mlp_out)


class _SingleStreamBlock(nn.Module):
    """Joint single-stream block (Phase 2).

    Runs shared self-attention across text (ctx) + all content tokens.
    Text and content tokens receive separate AdaLN-Zero modulation parameters.
    """

    def __init__(
        self,
        parallel_config: "DiffusionParallelConfig",
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
        fused_qkv: bool = True,
        kernel_adapter: Flux3KernelAdapter | None = None,
    ):
        super().__init__()
        self.kernel_adapter = kernel_adapter or Flux3KernelAdapter(_qk_norm_rope)
        self.head_dim = hidden_size // num_heads
        mlp_hidden = int(hidden_size * mlp_ratio)

        self.norm1 = AdaLayerNorm(hidden_size, elementwise_affine=False, eps=eps)

        self.qkv = _build_qkv(fused_qkv, hidden_size, self.head_dim, num_heads, quant_config, prefix)
        self.num_heads_local: int = self.qkv.num_heads
        self.num_kv_heads_local: int = self.qkv.num_kv_heads

        self.norm = _QKNorm(self.head_dim)

        self.attn_out = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "attn_out"),
        )
        self.mlp_in = ColumnParallelLinear(
            hidden_size,
            mlp_hidden * 2,
            bias=False,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "mlp_in"),
        )
        self.mlp_out = RowParallelLinear(
            mlp_hidden,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "mlp_out"),
        )
        self.mlp_act = SiluAndMul()
        self.attn = Attention(
            num_heads=self.num_heads_local,
            head_size=self.head_dim,
            softmax_scale=self.head_dim**-0.5,
            causal=False,
            num_kv_heads=self.num_kv_heads_local,
            role="self",
        )
        self.kernel_plan = self.kernel_adapter.configure_block(self, prefix)

    def _forward_joint(
        self,
        joint_states: Tensor,
        mod: tuple[Tensor, Tensor, Tensor],
        pe: Tensor,
        pag: bool = False,
        pag_strength: float = 1.0,
        kv_lens: tuple[int, ...] | None = None,
        mod_index: Tensor | None = None,
    ) -> Tensor:
        """Process an already-concatenated ``[content, text]`` stream.

        ``kv_lens`` gives each sample's valid prefix when the text of a batch was
        padded to a common length: each sample attends only to its own prefix.
        """
        if mod_index is not None:
            # Per-segment modulation rows; the gather fuses into the consumers.
            mod = tuple(m[:, mod_index] for m in mod)
        mod_shift, mod_scale, mod_gate = mod
        modulated_states = self.norm1(joint_states, mod_scale, mod_shift)
        prepared_input = self.kernel_adapter.prepare_input(self, modulated_states)

        # Joint QKV + attention
        q, k, v = self.kernel_adapter.qkv_qknorm_rope(self, prepared_input, pe)
        with _block_range("flux3/attn"):
            if kv_lens is None:
                attn_out = self.attn(q, k, v, None)
            else:
                attn_out = torch.cat(
                    [self.attn(q[i : i + 1], k[i : i + 1, :n], v[i : i + 1, :n], None) for i, n in enumerate(kv_lens)]
                )
        attn_out = attn_out.flatten(2, 3).to(joint_states.dtype)

        # Perturbed Attention Guidance: replace uncond half attn output with passthrough-of-v
        if pag:
            v_flat = v.flatten(2, 3).to(joint_states.dtype)  # v: (B, T, H, D) → (B, T, H*D)
            B = attn_out.shape[0]
            uncond_half = B // 2
            attn_out[:uncond_half] = (1.0 - pag_strength) * attn_out[:uncond_half] + pag_strength * v_flat[:uncond_half]

        with _block_range("flux3/attn_out"):
            attn_out = self.kernel_adapter.attention_output(self, attn_out)

        with _block_range("flux3/mlps"):
            mlp_out = self.kernel_adapter.mlp(self, prepared_input)

        return joint_states + mod_gate * (attn_out + mlp_out)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None,
        mod_x: tuple[Tensor, Tensor, Tensor],
        mod_ctx: tuple[Tensor, Tensor, Tensor] | None,
        pe: Tensor,
        pag: bool = False,
        pag_strength: float = 1.0,
        kv_lens: tuple[int, ...] | None = None,
        mod_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Run separate streams (compatibility) or one pre-joined stream.

        ``encoder_hidden_states=None`` selects the optimized joint-stream path:
        ``hidden_states`` and ``mod_x`` must then be ordered as
        ``[content, text]``. Keeping this mode behind ``forward`` preserves the
        standard module call path used by compile, FSDP, and offload hooks.
        """
        if encoder_hidden_states is None:
            if mod_ctx is not None:
                raise ValueError("mod_ctx must be None for a pre-joined stream")
            return self._forward_joint(hidden_states, mod_x, pe, pag, pag_strength, kv_lens, mod_index), None

        if mod_ctx is None:
            raise ValueError("mod_ctx is required for separate content and text streams")

        T_x = hidden_states.shape[1]
        joint_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
        joint_mod = tuple(torch.cat([mod_x[i], mod_ctx[i]], dim=1) for i in range(3))
        joint_states = self._forward_joint(joint_states, joint_mod, pe, pag, pag_strength)
        return joint_states[:, :T_x], joint_states[:, T_x:]


# ── Top-level model ────────────────────────────────────────────────────────


class Flux3Transformer2DModel(nn.Module, SupportsQuant):
    """Flux3 image DiT.

    Supports image generation and optional image-conditioning streams. Text
    (ctx) is always present but is not predicted.

    Forward call receives active image streams via **kwargs using the
    _SEQUENCE_MAP keys (e.g. x_image, x_image_ids, x_image_timesteps). Inactive
    streams are filled with dummy zero tensors internally so attention kernels
    always have at least two tokens.

    Returns dict[str, Tensor] mapping active kwarg names to velocity predictions
    (same shape as the input for that image stream).
    """

    # Per-stream input channel counts.
    _MODALITY_CHANNELS: dict[str, int] = {
        "image": 128,
        "image_cond": 128,
    }

    # kwarg_name in forward() **kwargs → modality key in _MODALITY_CHANNELS.
    # Order matters: it fixes the phase-2 joint-attention token layout.
    _SEQUENCE_MAP: dict[str, str] = {
        "x_image": "image",
        "x_image_cond": "image_cond",
    }

    _sp_plan: dict = {}

    _repeated_blocks = ["_ModeBlock", "_SingleStreamBlock"]

    # All block-list/dict attributes eligible for layer-wise CPU offload.
    # content_mode_blocks / late_content_mode_blocks are ModuleDicts keyed by
    # modality; txt_mode_blocks / single_blocks / late_txt_mode_blocks are
    # ModuleLists.  late_* attrs are None when depth_late_blocks == 0.
    _layerwise_offload_blocks_attrs = [
        "content_mode_blocks",
        "txt_mode_blocks",
        "single_blocks",
        "late_content_mode_blocks",
        "late_txt_mode_blocks",
    ]

    # Maps fused QKV weight name → constituent projection names for LoRA adapters.
    # Native NVFP4 overrides this at instance level because Q/K/V must remain
    # separate to preserve per-projection global scales.
    packed_modules_mapping = _QKV_PACKED_MODULES_MAPPING

    # vLLM applies this mapper to quantization metadata before __init__ constructs
    # linears. Keep the dynamic load-time mapper below for tensor-layout-specific
    # QKV and norm remapping.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            "diffusion_model.": "",
            "dit.": "",
            ".adaLN_modulation.1": ".adaLN_modulation",
        }
    )

    # Strips the checkpoint-level model prefix from LoRA adapter keys so they
    # match the pipeline component-relative module paths used at runtime.
    lora_weights_mapper = WeightsMapper(
        orig_to_new_substr={
            "diffusion_model.": "",
            "dit.": "",
        }
    )

    _cache_dit_adapter_config = CacheDiTAdapterConfig(
        block_forward_patterns={
            "single_blocks": ForwardPattern.Pattern_0,
        },
        has_separate_cfg=True,
    )

    @staticmethod
    def _is_block(name: str, module: nn.Module) -> bool:
        return ("content_mode_blocks" in name or "txt_mode_blocks" in name or "single_blocks" in name) and name.split(
            "."
        )[-1].isdigit()

    _hsdp_shard_conditions = [_is_block]

    def _maybe_apply_model_mapping(self):
        """Apply Flux3 quant metadata remapping once the quant format is known."""
        if self.quant_config is None:
            return
        if (hf_to_vllm_mapper := self.hf_to_vllm_mapper) is not None:
            self.quant_config.apply_vllm_mapper(hf_to_vllm_mapper.get_rename_mapper())
        if not _is_nvfp4_quant_config(self.quant_config):
            _set_quant_config_packed_qkv_mapping(self.quant_config, _QKV_PACKED_MODULES_MAPPING)

    def __init__(
        self,
        hidden_size: int = 3072,
        num_heads: int = 24,
        depth: int = 5,
        depth_single_blocks: int = 28,
        depth_late_blocks: int = 0,
        context_in_dim: int = 20480,
        vec_in_dim: int | None = None,
        flow_map: bool = False,
        mlp_ratio: float = 3.0,
        axes_dim: tuple[int, ...] = (32, 32, 32, 32),
        theta: int = 10000,
        eps: float = 1e-6,
        out_modes: tuple[str, ...] | list[str] | None = ("image",),
        # Pipeline compatibility: retained for shared transformer config plumbing.
        in_channels: int = 128,
        quant_config: "QuantizationConfig | None" = None,
        od_config: OmniDiffusionConfig | None = None,
    ):
        super().__init__()
        self.kernel_adapter = build_flux3_kernel_adapter(
            qk_norm_rope=_qk_norm_rope,
            od_config=od_config,
            quant_config=quant_config,
        )
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.depth = depth
        self.depth_single_blocks = depth_single_blocks
        self.depth_late_blocks = depth_late_blocks
        self.flow_map = flow_map
        self.out_modes = tuple(self._MODALITY_CHANNELS) if out_modes is None else tuple(out_modes)
        unknown_out_modes = sorted(set(self.out_modes) - set(self._MODALITY_CHANNELS))
        if unknown_out_modes:
            raise ValueError(f"Flux3 out_modes must be image stream names, got: {unknown_out_modes}")
        content_modalities = tuple(self._MODALITY_CHANNELS)
        fused_qkv_by_prefix = {
            prefix: _resolve_fuse_qkv(
                quant_config,
                self.kernel_adapter.backend,
                self.kernel_adapter.uses_fused_qkv(prefix, self.head_dim),
            )
            for prefix in _iter_qkv_prefixes(content_modalities, depth, depth_single_blocks, depth_late_blocks)
        }
        self.packed_modules_mapping = _QKV_PACKED_MODULES_MAPPING if any(fused_qkv_by_prefix.values()) else {}
        _set_quant_config_packed_qkv_mapping(quant_config, self.packed_modules_mapping)
        self._split_qkv_checkpoint_mapper = _build_split_qkv_checkpoint_mapper(fused_qkv_by_prefix)
        self.lora_weights_mapper = _build_lora_weights_mapper(fused_qkv_by_prefix)
        if quant_config is not None and self._split_qkv_checkpoint_mapper is not None:
            apply_vllm_mapper = getattr(quant_config, "apply_vllm_mapper", None)
            if apply_vllm_mapper is not None:
                apply_vllm_mapper(self._split_qkv_checkpoint_mapper.get_rename_mapper())

        def fuse_qkv_for(prefix: str) -> bool:
            return fused_qkv_by_prefix[prefix]

        assert sum(axes_dim) == self.head_dim, f"axes_dim {axes_dim} must sum to head_dim {self.head_dim}"

        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            num_heads=num_heads,
            depth=depth,
            depth_single_blocks=depth_single_blocks,
            depth_late_blocks=depth_late_blocks,
            context_in_dim=context_in_dim,
            vec_in_dim=vec_in_dim,
            flow_map=flow_map,
            mlp_ratio=mlp_ratio,
            axes_dim=axes_dim,
            theta=theta,
            in_channels=in_channels,
            out_modes=self.out_modes,
        )

        self.parallel_config = od_config.parallel_config if od_config is not None else DiffusionParallelConfig()

        # Positional encoding (shared; no learned parameters)
        self.pe_embedder = _EmbedND(dim=self.head_dim, theta=theta, axes_dim=list(axes_dim))

        # Input embedders: per-modality linear projection to hidden_size
        self.emb_in = nn.ModuleDict(
            {
                k: ColumnParallelLinear(
                    c,
                    hidden_size,
                    bias=False,
                    gather_output=True,
                    return_bias=False,
                    quant_config=quant_config,
                    prefix=f"emb_in.{k}",
                )
                for k, c in self._MODALITY_CHANNELS.items()
            }
        )

        # Text input embedder
        self.txt_in = ColumnParallelLinear(
            context_in_dim,
            hidden_size,
            bias=False,
            gather_output=True,
            return_bias=False,
            quant_config=quant_config,
            prefix="txt_in",
        )

        # Timestep and vector conditioning
        self.time_in = _MLPEmbedder(256, hidden_size, quant_config=quant_config, prefix="time_in")
        self.ref_time_in = (
            _MLPEmbedder(256, hidden_size, quant_config=quant_config, prefix="ref_time_in") if flow_map else None
        )
        self.vector_in = (
            None
            if vec_in_dim is None
            else _MLPEmbedder(vec_in_dim, hidden_size, quant_config=quant_config, prefix="vector_in")
        )

        # AdaLN modulation lives at the top level, one _Modulation per stream
        # (each content modality + "txt"), shared across all blocks of a phase.
        content_modalities = sorted(self._MODALITY_CHANNELS.keys())
        self.early_stream_modulations = nn.ModuleDict(
            {
                **{
                    m: _Modulation(hidden_size, quant_config=quant_config, prefix=f"early_stream_modulations.{m}")
                    for m in content_modalities
                },
                "txt": _Modulation(hidden_size, quant_config=quant_config, prefix="early_stream_modulations.txt"),
            }
        )
        self.single_stream_modulations = nn.ModuleDict(
            {
                **{
                    m: _Modulation(hidden_size, quant_config=quant_config, prefix=f"single_stream_modulations.{m}")
                    for m in content_modalities
                },
                "txt": _Modulation(hidden_size, quant_config=quant_config, prefix="single_stream_modulations.txt"),
            }
        )

        # Phase 1: per-modality early ModeBlocks
        self.content_mode_blocks = nn.ModuleDict(
            {
                mod_key: nn.ModuleList(
                    [
                        _ModeBlock(
                            parallel_config=self.parallel_config,
                            hidden_size=hidden_size,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            eps=eps,
                            quant_config=quant_config,
                            prefix=f"content_mode_blocks.{mod_key}.{i}",
                            fused_qkv=fuse_qkv_for(f"content_mode_blocks.{mod_key}.{i}"),
                            kernel_adapter=self.kernel_adapter,
                        )
                        for i in range(depth)
                    ]
                )
                for mod_key in self._MODALITY_CHANNELS
            }
        )
        self.txt_mode_blocks = nn.ModuleList(
            [
                _ModeBlock(
                    parallel_config=self.parallel_config,
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    quant_config=quant_config,
                    prefix=f"txt_mode_blocks.{i}",
                    fused_qkv=fuse_qkv_for(f"txt_mode_blocks.{i}"),
                    kernel_adapter=self.kernel_adapter,
                )
                for i in range(depth)
            ]
        )

        # Phase 2: joint single-stream blocks
        self.single_blocks = nn.ModuleList(
            [
                _SingleStreamBlock(
                    parallel_config=self.parallel_config,
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    quant_config=quant_config,
                    prefix=f"single_blocks.{i}",
                    fused_qkv=fuse_qkv_for(f"single_blocks.{i}"),
                    kernel_adapter=self.kernel_adapter,
                )
                for i in range(depth_single_blocks)
            ]
        )

        # Phase 3: optional late ModeBlocks (depth_late_blocks=0 by default — inert)
        if depth_late_blocks > 0:
            self.late_stream_modulations: nn.ModuleDict | None = nn.ModuleDict(
                {
                    **{
                        m: _Modulation(hidden_size, quant_config=quant_config, prefix=f"late_stream_modulations.{m}")
                        for m in content_modalities
                    },
                    "txt": _Modulation(hidden_size, quant_config=quant_config, prefix="late_stream_modulations.txt"),
                }
            )
            self.late_content_mode_blocks: nn.ModuleDict | None = nn.ModuleDict(
                {
                    mod_key: nn.ModuleList(
                        [
                            _ModeBlock(
                                parallel_config=self.parallel_config,
                                hidden_size=hidden_size,
                                num_heads=num_heads,
                                mlp_ratio=mlp_ratio,
                                eps=eps,
                                quant_config=quant_config,
                                prefix=f"late_content_mode_blocks.{mod_key}.{i}",
                                fused_qkv=fuse_qkv_for(f"late_content_mode_blocks.{mod_key}.{i}"),
                                kernel_adapter=self.kernel_adapter,
                            )
                            for i in range(depth_late_blocks)
                        ]
                    )
                    for mod_key in self._MODALITY_CHANNELS
                }
            )
            self.late_txt_mode_blocks: nn.ModuleList | None = nn.ModuleList(
                [
                    _ModeBlock(
                        parallel_config=self.parallel_config,
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        eps=eps,
                        quant_config=quant_config,
                        prefix=f"late_txt_mode_blocks.{i}",
                        fused_qkv=fuse_qkv_for(f"late_txt_mode_blocks.{i}"),
                        kernel_adapter=self.kernel_adapter,
                    )
                    for i in range(depth_late_blocks)
                ]
            )
        else:
            self.late_stream_modulations = None
            self.late_content_mode_blocks = None
            self.late_txt_mode_blocks = None

        self.kernel_adapter.finalize()
        self._weights_mapper, self.stacked_params_mapping = _build_qkv_weights_mapper(fused_qkv_by_prefix)

        # Final layers: AdaLN + linear back to latent channels, one per modality
        self.final_layer = nn.ModuleDict(
            {
                k: _LastLayer(hidden_size, c, quant_config=quant_config, prefix=f"final_layer.{k}")
                for k, c in self._MODALITY_CHANNELS.items()
                if k in self.out_modes
            }
        )

    def remap_checkpoint_key(self, name: str) -> str | None:
        """Expose split-QKV key remapping to checkpoint adapters that see pipeline keys."""
        split_qkv_mapper = getattr(self, "_split_qkv_checkpoint_mapper", None)
        if split_qkv_mapper is None:
            return None
        mapped = split_qkv_mapper.apply_list([name])
        if not mapped or mapped[0] == name:
            return None
        return mapped[0]

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _timestep_rows(self, timesteps: Tensor) -> Tensor:
        """One timestep row per stream.

        Every token of a stream shares its timestep (the pipeline fills each stream with
        one value), so the timestep embedding and AdaLN modulation run on one row and
        broadcast over the stream's tokens instead of repeating the same row per token.
        """
        return timesteps[:, :1]

    def _vector_embedding(self, vector: Tensor, reference: Tensor) -> Tensor:
        """Embed pooled vector conditioning, or return zeros when the checkpoint omits it."""
        if self.vector_in is None:
            return reference.new_zeros((reference.shape[0], 1, self.hidden_size))
        return self.vector_in(vector).unsqueeze(1)

    def precompute_static_conditioning(
        self,
        ctx: Tensor,
        ctx_ids: Tensor,
        vector: Tensor,
        timesteps_ctx: Tensor,
        ids_by_mod: dict[str, Tensor] | None = None,
        static_content: dict[str, tuple[Tensor, Tensor]] | None = None,
    ) -> dict:
        """Precompute conditioning tensors constant across denoising steps.

        ``static_content`` maps an image-conditioning argument name to its fixed
        ``(tokens, timesteps)``. These Phase 1 outputs do not change during
        denoising.

        Returns a dict of precomputed tensors to be passed as ``precomputed=``
        to forward(). The caller is responsible for separating pos/neg and
        passing the right dict to each forward call.
        """
        vec_emb = self._vector_embedding(vector, timesteps_ctx)
        vec_ctx = self.time_in(_timestep_embedding(self._timestep_rows(timesteps_ctx))) + vec_emb

        ctx_emb = self.txt_in(ctx)
        pe_ctx = self.pe_embedder(ctx_ids)

        early_txt_mod = self.early_stream_modulations["txt"](vec_ctx)
        for d in range(self.depth):
            ctx_emb = self.txt_mode_blocks[d](ctx_emb, pe_ctx, early_txt_mod)

        result: dict = {
            "ctx_emb": ctx_emb,
            "vec_emb": vec_emb,
            "vec_ctx": vec_ctx,
            "pe_ctx": pe_ctx,
            "mod_ctx": self.single_stream_modulations["txt"](vec_ctx),
        }
        pe_by_mod = {k: self.pe_embedder(v) for k, v in (ids_by_mod or {}).items()}
        if ids_by_mod is not None:
            result["pe_by_mod"] = pe_by_mod

        if static_content:
            phase1_content_by_mod: dict[str, Tensor] = {}
            vec_by_mod: dict[str, Tensor] = {}
            for kwarg_name, (tokens, timesteps) in static_content.items():
                mod_key = self._SEQUENCE_MAP[kwarg_name]
                pe = pe_by_mod[kwarg_name]
                content = self.emb_in[mod_key](tokens)
                vec_mod = self.time_in(_timestep_embedding(self._timestep_rows(timesteps))) + vec_emb
                early_mod = self.early_stream_modulations[mod_key](vec_mod)
                for d in range(self.depth):
                    content = self.content_mode_blocks[mod_key][d](content, pe, early_mod)
                phase1_content_by_mod[kwarg_name] = content
                vec_by_mod[kwarg_name] = vec_mod
            result["phase1_content_by_mod"] = phase1_content_by_mod
            result["vec_by_mod"] = vec_by_mod
        return result

    @torch.cuda.nvtx.range("flux3/embed_and_rope")
    def _embed_and_rope(
        self,
        ctx,
        ctx_ids,
        vector,
        x_by_mod,
        ids_by_mod,
        ts_by_mod,
        timesteps_ctx,
        timesteps_ref,
        active_content,
        precomputed: dict | None = None,
    ):
        cached_content = precomputed.get("phase1_content_by_mod", {}) if precomputed is not None else {}
        content_by_mod: dict[str, Tensor] = {}
        for kwarg_name, mod_key in active_content.items():
            content_by_mod[kwarg_name] = (
                cached_content[kwarg_name]
                if kwarg_name in cached_content
                else self.emb_in[mod_key](x_by_mod[kwarg_name])
            )

        if precomputed is not None:
            if self.ref_time_in is not None:
                raise ValueError("Flow-map conditioning cannot be precomputed across denoising steps")
            ctx_emb = precomputed["ctx_emb"]
            vec_emb = precomputed["vec_emb"]
            vec_ctx = precomputed["vec_ctx"]
            pe_ctx = precomputed["pe_ctx"]
            mod_ctx = precomputed["mod_ctx"]
        else:
            ctx_emb = self.txt_in(ctx)
            vec_emb = self._vector_embedding(vector, timesteps_ctx)
            vec_ctx = self.time_in(_timestep_embedding(self._timestep_rows(timesteps_ctx))) + vec_emb
            if self.ref_time_in is not None:
                ref_time = self.ref_time_in(_timestep_embedding(timesteps_ref.to(timesteps_ctx))).unsqueeze(1)
                vec_ctx = vec_ctx + ref_time
            pe_ctx = self.pe_embedder(ctx_ids)
            mod_ctx = None  # computed later in _phase2

        cached_vec = precomputed.get("vec_by_mod", {}) if precomputed is not None else {}
        vec_by_mod: dict[str, Tensor] = {}
        for kwarg_name in active_content:
            if kwarg_name in cached_vec:
                vec_by_mod[kwarg_name] = cached_vec[kwarg_name]
            else:
                t_sin = _timestep_embedding(self._timestep_rows(ts_by_mod[kwarg_name]))
                vec_by_mod[kwarg_name] = self.time_in(t_sin) + vec_emb
                if self.ref_time_in is not None:
                    vec_by_mod[kwarg_name] = vec_by_mod[kwarg_name] + self.ref_time_in(
                        _timestep_embedding(timesteps_ref.to(timesteps_ctx))
                    ).unsqueeze(1)

        cached_pe = precomputed.get("pe_by_mod", {}) if precomputed is not None else {}
        pe_by_mod: dict[str, Tensor] = {
            kwarg_name: (cached_pe[kwarg_name] if kwarg_name in cached_pe else self.pe_embedder(ids_by_mod[kwarg_name]))
            for kwarg_name in active_content
        }

        return content_by_mod, ctx_emb, vec_by_mod, vec_ctx, pe_by_mod, pe_ctx, mod_ctx

    @torch.cuda.nvtx.range("flux3/phase1")
    def _phase1(
        self,
        content_by_mod,
        ctx_emb,
        vec_by_mod,
        vec_ctx,
        pe_by_mod,
        pe_ctx,
        active_content,
        skip_txt: bool = False,
        skip_content: set[str] | None = None,
    ):
        # skip_txt=True when ctx_emb already contains post-Phase-1 text output
        # from precomputed static conditioning passed via forward(precomputed=...).

        dynamic_content = {
            kwarg_name: mod_key
            for kwarg_name, mod_key in active_content.items()
            if skip_content is None or kwarg_name not in skip_content
        }
        early_mods = {
            kwarg_name: self.early_stream_modulations[mod_key](vec_by_mod[kwarg_name])
            for kwarg_name, mod_key in dynamic_content.items()
        }
        if not skip_txt:
            early_txt_mod = self.early_stream_modulations["txt"](vec_ctx)

        for d in range(self.depth):
            with torch.cuda.nvtx.range(f"flux3/phase1/block/{d}"):
                for kwarg_name, mod_key in dynamic_content.items():
                    content_by_mod[kwarg_name] = self.content_mode_blocks[mod_key][d](
                        content_by_mod[kwarg_name],
                        pe_by_mod[kwarg_name],
                        early_mods[kwarg_name],
                    )
                if not skip_txt:
                    ctx_emb = self.txt_mode_blocks[d](ctx_emb, pe_ctx, early_txt_mod)

        return content_by_mod, ctx_emb

    @torch.cuda.nvtx.range("flux3/phase2")
    def _phase2(
        self,
        content_by_mod,
        ctx_emb,
        vec_by_mod,
        vec_ctx,
        pe_by_mod,
        pe_ctx,
        active_kwargs,
        pag_layers_set,
        pag_strength,
        mod_ctx=None,
        text_lens: tuple[int, ...] | None = None,
        static_cache: dict | None = None,
    ):
        x_content = torch.cat([content_by_mod[k] for k in active_kwargs], dim=1)
        # The joint RoPE table and the single-stream modulation of static streams do not
        # change across denoising steps: with precomputed conditioning they are built on
        # the first step and reused (static_cache is that request's precomputed dict).
        cache_key = ("phase2", tuple(active_kwargs))
        cached = static_cache.get(cache_key) if static_cache is not None else None
        if cached is None:
            pe_content = torch.cat([pe_by_mod[k] for k in active_kwargs], dim=1)
            static_mods = {
                k: self.single_stream_modulations[self._SEQUENCE_MAP[k]](vec_by_mod[k])
                for k in active_kwargs
                if static_cache is not None and k in static_cache.get("vec_by_mod", {})
            }
            cached = (torch.cat([pe_content, pe_ctx], dim=1), static_mods)
            if static_cache is not None:
                static_cache[cache_key] = cached
        pe_joint, static_mods = cached

        single_mods = {
            k: static_mods[k]
            if k in static_mods
            else self.single_stream_modulations[self._SEQUENCE_MAP[k]](vec_by_mod[k])
            for k in active_kwargs
        }
        if mod_ctx is None:
            mod_ctx = self.single_stream_modulations["txt"](vec_ctx)

        T_by_mod: dict[str, int] = {k: content_by_mod[k].shape[1] for k in active_kwargs}
        T_ctx = ctx_emb.shape[1]
        mod_index = None
        if all(m[0].shape[1] == 1 for m in (*single_mods.values(), mod_ctx)):
            # One modulation row per stream: blocks gather the rows per token, which
            # fuses into their consumers instead of materializing [B, T, C] tables.
            segments = [T_by_mod[k] for k in active_kwargs] + [T_ctx]
            mod_index = torch.repeat_interleave(
                torch.arange(len(segments), device=ctx_emb.device),
                torch.tensor(segments, device=ctx_emb.device),
                output_size=sum(segments),
            )
            mod_table = tuple(
                torch.cat([single_mods[k][i] for k in active_kwargs] + [mod_ctx[i]], dim=1) for i in range(3)
            )
            single_mods = {k: tuple(m.expand(-1, T_by_mod[k], -1) for m in single_mods[k]) for k in active_kwargs}
            mod_ctx = tuple(m.expand(-1, T_ctx, -1) for m in mod_ctx)
        mod_x = tuple(torch.cat([single_mods[k][i] for k in active_kwargs], dim=1) for i in range(3))

        if getattr(self, "_is_cached", False):
            if text_lens is not None:
                raise NotImplementedError("Cache-DiT does not support padded CFG text")
            # Cache-DiT temporarily replaces this stack with a Pattern_0
            # two-stream wrapper, so retain its established call contract.
            for i, block in enumerate(self.single_blocks):
                with torch.cuda.nvtx.range(f"flux3/phase2/block/{i}"):
                    x_content, ctx_emb = block(
                        hidden_states=x_content,
                        encoder_hidden_states=ctx_emb,
                        mod_x=mod_x,
                        mod_ctx=mod_ctx,
                        pe=pe_joint,
                        pag=(i in pag_layers_set),
                        pag_strength=pag_strength,
                    )
        else:
            # Keep content and text joined across all single-stream blocks.
            # Modulation is constant within this transformer invocation, so it
            # is joined once as well; each block still normalizes its updated
            # residual state independently.
            joint_states = torch.cat([x_content, ctx_emb], dim=1)
            joint_mod = (
                mod_table
                if mod_index is not None
                else tuple(torch.cat([mod_x[i], mod_ctx[i]], dim=1) for i in range(3))
            )
            kv_lens = None if text_lens is None else tuple(x_content.shape[1] + n for n in text_lens)
            del ctx_emb, x_content
            for i, block in enumerate(self.single_blocks):
                with torch.cuda.nvtx.range(f"flux3/phase2/block/{i}"):
                    joint_states, _ = block(
                        hidden_states=joint_states,
                        encoder_hidden_states=None,
                        mod_x=joint_mod,
                        mod_ctx=None,
                        pe=pe_joint,
                        pag=(i in pag_layers_set),
                        pag_strength=pag_strength,
                        kv_lens=kv_lens,
                        mod_index=mod_index,
                    )
            split_idx = joint_states.shape[1] - T_ctx
            x_content = joint_states[:, :split_idx]
            ctx_emb = joint_states[:, split_idx:]

        return x_content, ctx_emb, T_by_mod

    @torch.cuda.nvtx.range("flux3/phase3")
    def _phase3(self, x_content, ctx_emb, vec_by_mod, vec_ctx, pe_by_mod, pe_ctx, active_kwargs, t_by_mod):
        content_by_mod: dict[str, Tensor] = {}
        offset = 0
        for kwarg_name in active_kwargs:
            T = t_by_mod[kwarg_name]
            content_by_mod[kwarg_name] = x_content[:, offset : offset + T]
            offset += T

        late_mods = {
            kwarg_name: self.late_stream_modulations[self._SEQUENCE_MAP[kwarg_name]](vec_by_mod[kwarg_name])
            for kwarg_name in active_kwargs
        }
        late_txt_mod = self.late_stream_modulations["txt"](vec_ctx)

        for d in range(self.depth_late_blocks):
            with torch.cuda.nvtx.range(f"flux3/phase3/block/{d}"):
                for kwarg_name in active_kwargs:
                    mod_key = self._SEQUENCE_MAP[kwarg_name]
                    content_by_mod[kwarg_name] = self.late_content_mode_blocks[mod_key][d](
                        content_by_mod[kwarg_name],
                        pe_by_mod[kwarg_name],
                        late_mods[kwarg_name],
                    )
                ctx_emb = self.late_txt_mode_blocks[d](ctx_emb, pe_ctx, late_txt_mod)

        x_content = torch.cat([content_by_mod[k] for k in active_kwargs], dim=1)
        return x_content, ctx_emb

    def forward(
        self,
        ctx: Tensor,
        ctx_ids: Tensor,
        vector: Tensor,
        timesteps_ctx: Tensor,
        pag_single_layers: list[int] | None = None,
        pag_strength: float = 1.0,
        precomputed: dict | None = None,
        timesteps_ref: Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        """
        ctx:            (B, T_ctx, context_in_dim) — text hidden states
        ctx_ids:        (B, T_ctx, 4)              — text positional ids
        vector:         (B, vec_in_dim)            — pooled text conditioning
        timesteps_ctx:  (B, T_ctx)                 — per-token timestep (typically zeros)
        pag_single_layers: indices of single_blocks to apply PAG on (optional)
        pag_strength:   PAG blend strength (1.0 = full perturbation)
        **kwargs: active image-stream triples (x_image, x_image_ids, x_image_timesteps, ...)

        Returns: dict mapping active kwarg content names → velocity predictions (same shape)
        """
        B = ctx.shape[0]
        device, dtype = ctx.device, ctx.dtype
        if (timesteps_ref is not None) != (self.ref_time_in is not None):
            raise ValueError("timesteps_ref is required by, and only by, a flow-map Flux3 model")
        if timesteps_ref is not None and timesteps_ref.shape != (B,):
            raise ValueError(f"timesteps_ref must have shape ({B},), got {tuple(timesteps_ref.shape)}")

        # ── 1. Collect image streams; fill inactive ones with dummy zeros ─────
        valid_kwargs = {
            name
            for content_name in self._SEQUENCE_MAP
            for name in (content_name, f"{content_name}_ids", f"{content_name}_timesteps")
        }
        unknown_kwargs = sorted(set(kwargs) - valid_kwargs)
        if unknown_kwargs:
            raise ValueError("Flux3 transformer does not support inputs: " + ", ".join(unknown_kwargs))

        active_mods: dict[str, bool] = {}
        x_by_mod: dict[str, Tensor] = {}
        ids_by_mod: dict[str, Tensor] = {}
        ts_by_mod: dict[str, Tensor] = {}

        for kwarg_name, mod_key in self._SEQUENCE_MAP.items():
            c = self._MODALITY_CHANNELS[mod_key]
            if kwarg_name in kwargs:
                missing = [name for name in (f"{kwarg_name}_ids", f"{kwarg_name}_timesteps") if name not in kwargs]
                if missing:
                    raise ValueError(f"Missing required Flux3 image stream inputs: {', '.join(missing)}")
                active_mods[kwarg_name] = True
                x_by_mod[kwarg_name] = kwargs[kwarg_name]
                ids_by_mod[kwarg_name] = kwargs[f"{kwarg_name}_ids"]
                ts_by_mod[kwarg_name] = kwargs[f"{kwarg_name}_timesteps"]
            else:
                active_mods[kwarg_name] = False
                # Dummy: 2 tokens minimum so attention kernels don't break
                x_by_mod[kwarg_name] = torch.zeros(B, 2, c, device=device, dtype=dtype)
                ids_by_mod[kwarg_name] = torch.zeros(B, 2, 4, device=device, dtype=torch.long)
                ts_by_mod[kwarg_name] = torch.zeros(B, 2, device=device, dtype=dtype)

        active_content = {k: v for k, v in self._SEQUENCE_MAP.items() if active_mods[k]}
        if not active_content:
            raise ValueError("Flux3 transformer requires at least one image stream")

        # ── 2–3. Embed + RoPE ────────────────────────────────────────────────
        content_by_mod, ctx_emb, vec_by_mod, vec_ctx, pe_by_mod, pe_ctx, mod_ctx = self._embed_and_rope(
            ctx,
            ctx_ids,
            vector,
            x_by_mod,
            ids_by_mod,
            ts_by_mod,
            timesteps_ctx,
            timesteps_ref,
            active_content,
            precomputed=precomputed,
        )

        # ── 4. Phase 1: early per-modality ModeBlocks ────────────────────────
        content_by_mod, ctx_emb = self._phase1(
            content_by_mod,
            ctx_emb,
            vec_by_mod,
            vec_ctx,
            pe_by_mod,
            pe_ctx,
            active_content,
            skip_txt=precomputed is not None,
            skip_content=(set(precomputed.get("phase1_content_by_mod", {})) if precomputed is not None else None),
        )

        # ── 5. Phase 2: joint SingleStreamBlocks (active modalities only) ─────
        active_kwargs = [k for k in self._SEQUENCE_MAP if active_mods[k]]
        pag_layers_set: set[int] = set(pag_single_layers) if pag_single_layers else set()
        x_content, ctx_emb, T_by_mod = self._phase2(
            content_by_mod,
            ctx_emb,
            vec_by_mod,
            vec_ctx,
            pe_by_mod,
            pe_ctx,
            active_kwargs,
            pag_layers_set,
            pag_strength,
            mod_ctx=mod_ctx,
            text_lens=precomputed.get("text_lens") if precomputed is not None else None,
            static_cache=precomputed,
        )

        # ── 6. Phase 3: optional late ModeBlocks ─────────────────────────────
        if self.depth_late_blocks > 0 and self.late_content_mode_blocks is not None:
            x_content, ctx_emb = self._phase3(
                x_content,
                ctx_emb,
                vec_by_mod,
                vec_ctx,
                pe_by_mod,
                pe_ctx,
                active_kwargs,
                T_by_mod,
            )

        # ── 7. Split back, apply final layers, collect active predictions ─────
        output: dict[str, Tensor] = {}
        offset = 0
        for kwarg_name in active_kwargs:
            mod_key = self._SEQUENCE_MAP[kwarg_name]
            T = T_by_mod[kwarg_name]
            x_k = x_content[:, offset : offset + T]
            offset += T
            if mod_key in self.final_layer:
                output[kwarg_name] = self.final_layer[mod_key](x_k, vec_by_mod[kwarg_name])

        return output

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        ignore_unexpected_prefixes = ["repa_projectors"]
        if self.vector_in is None:
            ignore_unexpected_prefixes.append("vector_in")
        for mode in set(self._MODALITY_CHANNELS) - set(self.final_layer):
            ignore_unexpected_prefixes.append(f"final_layer.{mode}")
        return AutoWeightsLoader(self, ignore_unexpected_prefixes=ignore_unexpected_prefixes).load_weights(
            weights, mapper=self._weights_mapper
        )
