# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Generated standalone Kandinsky 6 vLLM-Omni transformer.

The template records source modules only. ``kandinsky.ports.assemble`` reads
those modules and pastes their current definitions into the generated file;
the generated artifact therefore has no native-package import dependency
(only ``torch`` and ``vllm``/``vllm_omni``, matching vLLM-Omni's other
native diffusion models).
"""

from __future__ import annotations

import contextvars
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelInput, SequenceParallelOutput

# Set while visual blocks run so self-attention can apply Ulysses/Ring only
# to the sequence-sharded visual stream (text self-attention stays local).
_VISUAL_SP: contextvars.ContextVar[bool] = contextvars.ContextVar("k6_visual_sp", default=False)


def _parallel_size(kind: str) -> int:
    """World size / rank for a parallel axis. 1 (or rank 0) when unset."""
    try:
        from vllm_omni.diffusion.distributed import parallel_state as ps
    except Exception:
        return 0 if kind == "pp_rank" else 1
    try:
        if kind == "pp":
            return int(ps.get_pipeline_parallel_world_size())
        if kind == "pp_rank":
            return int(ps.get_pipeline_parallel_rank())
        if kind == "sp":
            return int(ps.get_sequence_parallel_world_size())
        if kind == "ulysses":
            return int(ps.get_ulysses_parallel_world_size())
        if kind == "ring":
            return int(ps.get_ring_parallel_world_size())
    except Exception:
        return 0 if kind == "pp_rank" else 1
    return 1


def get_freqs(dim: int, max_period: float = 10000.0) -> Tensor:
    return torch.exp(-math.log(max_period) * torch.arange(start=0, end=dim, dtype=torch.float32) / dim)


def apply_scale_shift_norm(norm, x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    """AdaLN-style affine in fp32, cast back to ``x.dtype``."""
    return (norm(x.float()) * (scale.float() + 1.0) + shift.float()).to(dtype=x.dtype)


def apply_gate_sum(x: Tensor, out: Tensor, gate: Tensor) -> Tensor:
    """Residual gate in fp32, cast back to ``x.dtype``."""
    return (x.float() + gate.float() * out.float()).to(dtype=x.dtype)


def apply_rotary(x: Tensor, rope: Tensor) -> Tensor:
    """RoPE apply in fp32 (rope tables are fp32), cast back to ``x.dtype``."""
    x_ = x.reshape(*x.shape[:-1], -1, 1, 2).float()
    return (rope.float() * x_).sum(dim=-1).reshape(*x.shape).to(dtype=x.dtype)


def _local_patch(x: Tensor, shape: tuple, group_size: tuple, dim: int = 0) -> Tensor:
    T, H, W = shape
    g1, g2, g3 = group_size
    x = x.reshape(*x.shape[:dim], T // g1, g1, H // g2, g2, W // g3, g3, *x.shape[dim + 3 :])
    d = len(x.shape[:dim])
    x = x.permute(*range(d), d, d + 2, d + 4, d + 1, d + 3, d + 5, *range(d + 6, len(x.shape)))
    return x.flatten(dim, dim + 2).flatten(dim + 1, dim + 3)


def _local_merge(x: Tensor, shape: tuple, group_size: tuple, dim: int = 0) -> Tensor:
    T, H, W = shape
    g1, g2, g3 = group_size
    x = x.reshape(*x.shape[:dim], T // g1, H // g2, W // g3, g1, g2, g3, *x.shape[dim + 2 :])
    d = len(x.shape[:dim])
    x = x.permute(*range(d), d, d + 3, d + 1, d + 4, d + 2, d + 5, *range(d + 6, len(x.shape)))
    return x.flatten(dim, dim + 1).flatten(dim + 1, dim + 2).flatten(dim + 2, dim + 3)


def fractal_flatten(x: Tensor, rope: Tensor, shape: tuple, block_mask: bool = False):
    if block_mask:
        ps = 8
        x = _local_patch(x, shape, (1, ps, ps), dim=0)
        rope = _local_patch(rope, shape, (1, ps, ps), dim=0)
        return x.flatten(0, 1), rope.flatten(0, 1)
    return x.flatten(0, 2), rope.flatten(0, 2)


def fractal_unflatten(x: Tensor, shape: tuple, block_mask: bool = False) -> Tensor:
    if block_mask:
        ps = 8
        x = x.reshape(-1, ps * ps, x.shape[-1])
        return _local_merge(x, shape, (1, ps, ps), dim=0)
    return x.reshape(*shape, x.shape[-1])


def fast_sta_nabla(
    T: int,
    H: int,
    W: int,
    wT: int = 3,
    wH: int = 3,
    wW: int = 3,
    device: str | torch.device = "cuda",
) -> Tensor:
    """Precomputes the Sliding Tile Attention (STA) boolean mask for nabla attention."""
    l = max(T, H, W)
    r = torch.arange(l, dtype=torch.int16, device=device)
    mat = (r.unsqueeze(1) - r.unsqueeze(0)).abs()

    sta_t = mat[:T, :T].flatten() <= wT // 2
    sta_h = mat[:H, :H].flatten() <= wH // 2
    sta_w = mat[:W, :W].flatten() <= wW // 2

    sta_hw = (sta_h.unsqueeze(1) * sta_w.unsqueeze(0)).reshape(H, H, W, W).transpose(1, 2).flatten()
    sta = (sta_t.unsqueeze(1) * sta_hw.unsqueeze(0)).reshape(T, T, H * W, H * W).transpose(1, 2)
    return sta.reshape(T * H * W, T * H * W)


def nabla_block_mask(
    q: Tensor,
    k: Tensor,
    sta: Tensor,
    thr: float = 0.9,
    block_size: int = 64,
) -> BlockMask:
    """Builds a dynamic nabla BlockMask from query/key statistics + STA prior."""
    B, h, S, D = q.shape
    s1 = S // block_size
    qa = q.reshape(B, h, s1, block_size, D).mean(-2)
    ka = k.reshape(B, h, s1, block_size, D).mean(-2).transpose(-2, -1)
    attn_map = torch.softmax((qa @ ka) / math.sqrt(D), dim=-1)

    vals, inds = attn_map.sort(-1)
    mask = (vals.cumsum_(-1) >= 1 - thr).int().gather(-1, inds.argsort(-1))
    mask = torch.logical_or(mask, sta)

    kv_nb = mask.sum(-1).to(torch.int32)
    kv_inds = mask.argsort(dim=-1, descending=True).to(torch.int32)
    return BlockMask.from_kv_blocks(
        torch.zeros_like(kv_nb),
        kv_inds,
        kv_nb,
        kv_inds,
        BLOCK_SIZE=block_size,
        mask_mod=None,
    )


class RoPE1D(nn.Module):
    """1-D Rotary Position Embedding — used for text and audio sequences."""

    def __init__(
        self,
        dim: int,
        max_pos: int = 2048,
        max_period: float = 10000.0,
        freqs_scaling: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_pos = max_pos
        self.max_period = max_period
        self.freqs_scaling = freqs_scaling
        freq = get_freqs(dim // 2, max_period) * freqs_scaling
        self.register_buffer("args", torch.outer(torch.arange(max_pos, dtype=freq.dtype), freq), persistent=False)

    def forward(self, pos: Tensor) -> Tensor:
        # RoPE tables are fp32; keep trig in fp32.
        args = self.args[pos]  # (seq_len, dim//2)
        rope = torch.stack([torch.cos(args), -torch.sin(args), torch.sin(args), torch.cos(args)], dim=-1)
        return rope.view(*rope.shape[:-1], 2, 2).unsqueeze(-4)

    def reset_parameters(self) -> None:
        freq = get_freqs(self.dim // 2, self.max_period).to(self.args.device) * self.freqs_scaling
        self.args = torch.outer(torch.arange(self.max_pos, dtype=freq.dtype, device=freq.device), freq)


class RoPE3D(nn.Module):
    """3-D Rotary Position Embedding — used for video spatial-temporal tokens (T, H, W)."""

    def __init__(
        self,
        axes_dims: tuple[int, int, int],
        max_pos: tuple[int, int, int] = (128, 128, 128),
        max_period: float = 10000.0,
    ):
        super().__init__()
        self.axes_dims = axes_dims
        self.max_pos = max_pos
        self.max_period = max_period
        for i, (d, mp) in enumerate(zip(axes_dims, max_pos)):
            freq = get_freqs(d // 2, max_period)
            self.register_buffer(f"args_{i}", torch.outer(torch.arange(mp, dtype=freq.dtype), freq), persistent=False)

    def forward(
        self,
        shape: tuple,
        pos: list[Tensor],
        scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Tensor:
        T, H, W = shape
        args_t = getattr(self, "args_0")[pos[0]] / scale_factor[0]  # (T, d//2)
        args_h = getattr(self, "args_1")[pos[1]] / scale_factor[1]  # (H, d//2)
        args_w = getattr(self, "args_2")[pos[2]] / scale_factor[2]  # (W, d//2)

        args = torch.cat(
            [
                args_t.view(T, 1, 1, -1).expand(T, H, W, -1),
                args_h.view(1, H, 1, -1).expand(T, H, W, -1),
                args_w.view(1, 1, W, -1).expand(T, H, W, -1),
            ],
            dim=-1,
        )
        cos, sin = torch.cos(args), torch.sin(args)
        rope = torch.stack([cos, -sin, sin, cos], dim=-1)  # (T, H, W, total_dim, 4)
        rope = rope.view(*rope.shape[:-1], 2, 2)  # (T, H, W, total_dim, 2, 2)
        return rope.unsqueeze(-4)  # (T, H, W, 1, total_dim, 2, 2)

    def reset_parameters(self) -> None:
        for i, (d, mp) in enumerate(zip(self.axes_dims, self.max_pos)):
            freq = get_freqs(d // 2, self.max_period).to(getattr(self, f"args_{i}").device)
            setattr(self, f"args_{i}", torch.outer(torch.arange(mp, dtype=freq.dtype, device=freq.device), freq))


"""vLLM-Omni transformer for Kandinsky 6 — tensor-parallel port of ``DiffusionTransformer3D``.

The port assembler extracts every top-level class from this module into the
generated vLLM-Omni pipeline (``inline_module``). The block structure and
AdaLN-modulation math are unchanged from ``core/components/dit.py`` — but
unlike an earlier revision of this file, submodule attribute names and
nesting here match the **Diffusers-exported checkpoint layout** exactly
(``in_layer``/``out_layer``, ``to_query``/``to_key``/``to_value``,
``self_attention``/``cross_attention``, ``text_transformer_blocks``/
``visual_transformer_blocks``, etc. — see
``ports/overrides/diffusers/kandinsky6_transformer.py`` for the reference
this mirrors), not core's terse native names. This is deliberate: real
Kandinsky 6 checkpoints are converted and published in the "patched
Diffusers" bundle layout (``convert_checkpoint.py`` with
``--use-patched-diffusers``, i.e. ``uses_patched_diffusers: true`` in
``model_index.json``), which saves weights under the Diffusers vocabulary.
Matching that vocabulary here — including using SEPARATE
``to_query``/``to_key``/``to_value`` projections rather than a fused QKV
matrix — means ``ModelMixin.from_pretrained()`` loads such a checkpoint
directly via ordinary 1:1 state-dict key matching, with no custom
weight-fusion loader required. (An earlier revision used a fused
``QKVParallelLinear`` for self-attention, which is structurally
incompatible with a checkpoint that stores separate Q/K/V tensors — fixed
here in favor of three separate ``ColumnParallelLinear``s, matching
cross-attention's layout.)

Sharding scheme (Megatron-style, matching vLLM-Omni's other native
transformers — Wan2.2, MiniMax H3):
- A "pair" gets Column -> Row: the first Linear scatters to an
  intermediate that's sharded per rank (``gather_output=False``), the
  second gathers/reduces it back to a full, replicated ``model_dim``
  output that's added straight onto the residual stream
  (``input_is_parallel=True``). Used for ``feed_forward`` and
  ``*_time_embeddings``.
- A standalone Linear whose result must be a full (replicated) tensor
  immediately — because it either seeds the residual stream or its output
  is consumed elementwise (AdaLN shift/scale/gate, applied directly to the
  full residual tensor) rather than matrix-multiplied further — uses
  ``ColumnParallelLinear(..., gather_output=True)``. Used for
  ``*_text_embeddings``/``visual_embeddings`` (seed the residual stream)
  and ``*_modulation``/``out_layer``/``audio_out_layer`` (elementwise-
  consumed or final output).
- Attention: ``to_query``/``to_key``/``to_value`` are three separate
  ``ColumnParallelLinear``s (``gather_output=False``), the output
  projection is a ``RowParallelLinear``.
- ``query_norm``/``key_norm`` stay plain (non-distributed) RMSNorm over
  ``head_dim``: K6 reshapes into per-head vectors *before* normalizing, so
  the norm never crosses a head boundary and needs no cross-rank
  reduction — each rank's local heads are already independent under
  standard head-parallel tensor parallelism.

Attention *computation* itself (dense SDPA/flash dispatch and NABLA
block-sparse attention) is reused unmodified from
``core/components/attention/dispatch.py`` and
``core/components/attention/nabla.py`` (inlined by the template) rather
than routed through vLLM-Omni's pluggable ``Attention``/backend-registry
layer or Diffusers' ``dispatch_attention_fn`` — that deeper integration
(paged KV-cache metadata, a registered NABLA backend analogous to
FASTVIDEO_VSA) is follow-up work once this TP-sharded weight path is
validated, not a prerequisite for correct tensor-parallel serving.
"""

# The assembler inlines core/utils/tensors.py (apply_scale_shift_norm,
# apply_gate_sum, apply_rotary, fractal_flatten, fractal_unflatten,
# get_freqs), core/components/rope.py (RoPE1D, RoPE3D),
# core/components/attention/dispatch.py (SelfAttentionEngine), and
# core/components/attention/nabla.py (nabla_block_mask) into the same
# generated module. They are intentionally resolved there.
# ruff: noqa: F821


# ---------------------------------------------------------------------------
# Small embedding / projection modules
# ---------------------------------------------------------------------------


class Kandinsky6TimeEmbeddings(nn.Module):
    """Sinusoidal time embedding -> tensor-parallel MLP (Column -> Row pair)."""

    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        max_period: float = 10000.0,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        if model_dim % 2:
            raise ValueError("model_dim must be even")
        self.register_buffer("freqs", get_freqs(model_dim // 2, max_period), persistent=False)
        self.in_layer = ColumnParallelLinear(
            model_dim,
            time_dim,
            bias=True,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_layer" if prefix else "in_layer",
        )
        self.activation = nn.SiLU()
        self.out_layer = RowParallelLinear(
            time_dim,
            time_dim,
            bias=True,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out_layer" if prefix else "out_layer",
        )

    def forward(self, time: Tensor) -> Tensor:
        # Sinusoidal embedding is built in fp32 (its args have large
        # magnitude). The MLP itself runs through the layers' own forward
        # (their native, e.g. bf16, dtype) rather than core's manual fp32
        # upcast trick, so tensor-parallel all-reduce/bias-add for
        # `out_layer` (RowParallelLinear) stay correct without hand-rolling
        # its reduction internals.
        freqs = self.freqs.to(device=time.device, dtype=torch.float32)
        args = torch.outer(time.float(), freqs)
        embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        h = self.activation(self.in_layer(embed.to(dtype=self.in_layer.weight.dtype)))
        return self.out_layer(h)


class Kandinsky6TextEmbeddings(nn.Module):
    """Projects raw text/pooled features into the (full, replicated) residual stream."""

    def __init__(
        self,
        text_dim: int,
        model_dim: int,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.in_layer = ColumnParallelLinear(
            text_dim,
            model_dim,
            bias=True,
            gather_output=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_layer" if prefix else "in_layer",
        )
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=True)

    def forward(self, x: Tensor) -> Tensor:
        wdtype = self.in_layer.weight.dtype
        x = self.in_layer(x.to(dtype=wdtype))
        return self.norm(x).to(dtype=x.dtype)


class Kandinsky6VisualEmbeddings(nn.Module):
    """Patchifies video latents and projects into the (full) residual stream."""

    def __init__(
        self,
        visual_dim: int,
        model_dim: int,
        patch_size: tuple,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_layer = ColumnParallelLinear(
            math.prod(patch_size) * visual_dim,
            model_dim,
            bias=True,
            gather_output=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_layer" if prefix else "in_layer",
        )

    def forward(self, x: Tensor) -> Tensor:
        duration, height, width, channels = x.shape
        p_t, p_h, p_w = self.patch_size
        x = (
            x.view(duration // p_t, p_t, height // p_h, p_h, width // p_w, p_w, channels)
            .permute(0, 2, 4, 1, 3, 5, 6)
            .flatten(3, 6)
        )
        return self.in_layer(x.to(dtype=self.in_layer.weight.dtype))


class Kandinsky6Modulation(nn.Module):
    """AdaLN modulation. Always gathers a full output: its ``shift``/``scale``/
    ``gate`` chunks are applied elementwise to the full residual-stream
    tensor, not matrix-multiplied further, so a sharded output would chunk
    incorrectly across ranks."""

    def __init__(
        self,
        time_dim: int,
        model_dim: int,
        num_params: int,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.activation = nn.SiLU()
        self.out_layer = ColumnParallelLinear(
            time_dim,
            num_params * model_dim,
            bias=True,
            gather_output=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out_layer" if prefix else "out_layer",
        )
        nn.init.zeros_(self.out_layer.weight)
        if self.out_layer.bias is not None:
            nn.init.zeros_(self.out_layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        out = self.out_layer(self.activation(x.float()).to(dtype=self.out_layer.weight.dtype))
        return out.to(dtype=x.dtype)


class Kandinsky6FeedForward(nn.Module):
    """Tensor-parallel FFN (Column -> Row pair)."""

    def __init__(
        self,
        dim: int,
        ff_dim: int,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.in_layer = ColumnParallelLinear(
            dim,
            ff_dim,
            bias=False,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_layer" if prefix else "in_layer",
        )
        self.activation = nn.GELU()
        self.out_layer = RowParallelLinear(
            ff_dim,
            dim,
            bias=False,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out_layer" if prefix else "out_layer",
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.activation(self.in_layer(x)))


# ---------------------------------------------------------------------------
# Attention (unified self/cross, matching the Diffusers-exported layout)
# ---------------------------------------------------------------------------


def _partition_visual_blocks(num_blocks: int, factory) -> tuple[int, int, nn.ModuleList]:
    """Keep every rank's block index stable; non-local slots are ``PPMissingLayer``."""
    from vllm.model_executor.models.utils import PPMissingLayer

    world = _parallel_size("pp")
    if world <= 1:
        start, end = 0, num_blocks
    else:
        from vllm.distributed.utils import get_pp_indices

        start, end = get_pp_indices(num_blocks, _parallel_size("pp_rank"), world)
    blocks = [factory(i) if start <= i < end else PPMissingLayer() for i in range(num_blocks)]
    return start, end, nn.ModuleList(blocks)


def _maybe_sequence_parallel_qkv(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    is_self_attention: bool,
) -> tuple[Tensor, Tensor, Tensor, Any, Any]:
    """Ulysses all-to-all (and optional Ring) for sharded visual self-attention.

    Text self-attention is not sequence-sharded, so the all-to-all runs only
    while ``_VISUAL_SP`` is set. Returns ``(q, k, v, ring_group, ulysses_group)``.
    """
    if not is_self_attention or not _VISUAL_SP.get() or query.dim() != 4:
        return query, key, value, None, None
    ulysses = _parallel_size("ulysses")
    ring = _parallel_size("ring")
    if ulysses <= 1 and ring <= 1:
        return query, key, value, None, None
    from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D
    from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

    sp = get_sp_group()
    ulysses_group = None
    if ulysses > 1:
        ulysses_group = sp.ulysses_group
        query = SeqAllToAll4D.apply(ulysses_group, query, 2, 1, False)
        key = SeqAllToAll4D.apply(ulysses_group, key, 2, 1, False)
        value = SeqAllToAll4D.apply(ulysses_group, value, 2, 1, False)
    ring_group = sp.ring_group if ring > 1 else None
    return query, key, value, ring_group, ulysses_group


class Kandinsky6Attention(nn.Module):
    """K6 attention — self-attention when ``encoder_hidden_states`` is
    omitted, cross-attention otherwise. Separate ``to_query``/``to_key``/
    ``to_value`` projections (not a fused QKV matrix) to match the
    Diffusers-exported checkpoint's per-tensor layout."""

    def __init__(
        self,
        num_channels: int,
        head_dim: int,
        kv_dim: int | None = None,
        engine: str = "auto",
        text_token_padding: bool = False,
        visual: bool = False,
        *,
        sequence_parallel: bool = False,
        role: str = "kandinsky6.text_self",
        role_category: str = "self",
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        if num_channels % head_dim:
            raise ValueError("num_channels must be divisible by head_dim")
        kv_dim = kv_dim or num_channels
        tp_size = get_tensor_model_parallel_world_size()
        self.head_dim = head_dim
        self.num_heads = (num_channels // head_dim) // tp_size
        self.to_query = ColumnParallelLinear(
            num_channels,
            num_channels,
            bias=True,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_query" if prefix else "to_query",
        )
        self.to_key = ColumnParallelLinear(
            kv_dim,
            num_channels,
            bias=True,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_key" if prefix else "to_key",
        )
        self.to_value = ColumnParallelLinear(
            kv_dim,
            num_channels,
            bias=True,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_value" if prefix else "to_value",
        )
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)
        self.out_layer = RowParallelLinear(
            num_channels,
            num_channels,
            bias=True,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out_layer" if prefix else "out_layer",
        )
        self.visual = visual
        self.attention_engine = engine
        self.text_token_padding = text_token_padding
        self.sequence_parallel = sequence_parallel
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            causal=False,
            softmax_scale=1.0 / (self.head_dim**0.5),
            num_kv_heads=self.num_heads,
            prefix=prefix,
            role=role,
            role_category=role_category,
            scatter_idx=2,
            gather_idx=1,
            skip_sequence_parallel=not sequence_parallel,
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        attn_mask: Tensor | None = None,
        rotary_emb: Tensor | None = None,
        sparse_params: dict[str, Any] | None = None,
        rope_q: Tensor | None = None,
        rope_kv: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        if attn_mask is None:
            attn_mask = kwargs.get("attention_mask")
        rotary_emb_q = rope_q if rope_q is not None else rotary_emb
        rotary_emb_kv = rope_kv if rope_kv is not None else rotary_emb

        is_self_attention = encoder_hidden_states is None
        kv_input = hidden_states if is_self_attention else encoder_hidden_states

        query = self.to_query(hidden_states).reshape(*hidden_states.shape[:-1], self.num_heads, self.head_dim)
        key = self.to_key(kv_input).reshape(*kv_input.shape[:-1], self.num_heads, self.head_dim)
        value = self.to_value(kv_input).reshape(*kv_input.shape[:-1], self.num_heads, self.head_dim)
        query = self.query_norm(query)
        key = self.key_norm(key)
        if rotary_emb_q is not None:
            query = apply_rotary(query, rotary_emb_q).type_as(query)
        if rotary_emb_kv is not None:
            key = apply_rotary(key, rotary_emb_kv).type_as(key)

        if is_self_attention:
            # FA3/FA2/SDPA dispatch requires exactly (B, S, H, D). Text
            # self-attention's `hidden_states` is unbatched ((S, H, D) after
            # reshape) and needs a batch dim added; visual/audio self-attention
            # is already batched=1 ((1, N, H, D)) because `_embed_visual` /
            # `_embed_audio` unsqueeze upstream. Adding a batch dim
            # unconditionally would hand SDPA a 5-D (1, 1, N, H, D) tensor,
            # which it happily accepts — but then it treats the *head* axis
            # as the sequence and attends across heads instead of tokens
            # (matches core/components/dit.py: MultiheadSelfAttentionEnc
            # unsqueezes, MultiheadSelfAttentionDec does not).
            strip_output_batch = query.dim() == 3
            if strip_output_batch:
                query, key, value = query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0)
        else:
            # Cross-attention: the query side (always visual, or the
            # opposite modality for cross-modal attention) already carries
            # a batch=1 dim; only the kv side may need one added to match
            # (core/components/dit.py's MultiheadCrossAttention: "k/v may
            # lack the batch dim (text has no batch, visual has batch=1)").
            # No dim is added beyond what's needed, so no stripping after.
            if key.dim() < query.dim():
                key, value = key.unsqueeze(0), value.unsqueeze(0)
            strip_output_batch = False

        if sparse_params is not None:
            query, key, value, _ring_group, ulysses_group = _maybe_sequence_parallel_qkv(
                query, key, value, is_self_attention=is_self_attention and self.sequence_parallel
            )
            from torch.nn.attention.flex_attention import flex_attention

            q_ = query.transpose(1, 2).contiguous()
            k_ = key.transpose(1, 2).contiguous()
            v_ = value.transpose(1, 2).contiguous()
            block_mask = nabla_block_mask(q_, k_, sparse_params["sta_mask"], thr=sparse_params["P"])
            out = flex_attention(q_, k_, v_, block_mask=block_mask).transpose(1, 2).contiguous()
            if ulysses_group is not None:
                from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D

                out = SeqAllToAll4D.apply(ulysses_group, out, 1, 2, False)
        else:
            metadata = AttentionMetadata(attn_mask=attn_mask) if attn_mask is not None else None
            out = self.attn(query, key, value, metadata)

        if strip_output_batch:
            out = out[0]
        return self.out_layer(out.flatten(-2, -1))


# ---------------------------------------------------------------------------
# Output layers
# ---------------------------------------------------------------------------


class Kandinsky6OutLayer(nn.Module):
    """Projects model_dim -> pixel patches for the video output (full/gathered)."""

    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        visual_dim: int,
        patch_size: tuple,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.modulation = Kandinsky6Modulation(
            time_dim, model_dim, 2, quant_config=quant_config, prefix=f"{prefix}.modulation" if prefix else "modulation"
        )
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.out_layer = ColumnParallelLinear(
            model_dim,
            math.prod(patch_size) * visual_dim,
            bias=True,
            gather_output=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out_layer" if prefix else "out_layer",
        )

    def forward(self, visual_embed: Tensor, time_embed: Tensor) -> Tensor:
        shift, scale = torch.chunk(self.modulation(time_embed), 2, dim=-1)
        x = apply_scale_shift_norm(self.norm, visual_embed, scale[:, None, None], shift[:, None, None]).type_as(
            visual_embed
        )
        x = self.out_layer(x.to(dtype=self.out_layer.weight.dtype))

        duration, height, width, _ = x.shape
        p_t, p_h, p_w = self.patch_size
        return (
            x.view(duration, height, width, -1, p_t, p_h, p_w)
            .permute(0, 4, 1, 5, 2, 6, 3)
            .flatten(0, 1)
            .flatten(1, 2)
            .flatten(2, 3)
        )


class Kandinsky6OutLayerAudio(nn.Module):
    """Projects model_dim_a -> audio latent channels (full/gathered)."""

    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        audio_dim: int,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.modulation = Kandinsky6Modulation(
            time_dim, model_dim, 2, quant_config=quant_config, prefix=f"{prefix}.modulation" if prefix else "modulation"
        )
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.out_layer = ColumnParallelLinear(
            model_dim,
            audio_dim,
            bias=True,
            gather_output=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out_layer" if prefix else "out_layer",
        )

    def forward(self, audio_embed: Tensor, time_embed: Tensor) -> Tensor:
        shift, scale = torch.chunk(self.modulation(time_embed), 2, dim=-1)
        x = apply_scale_shift_norm(self.norm, audio_embed, scale, shift).type_as(audio_embed)
        x = self.norm(x)  # matches reference training (double norm — do not remove for parity)
        return self.out_layer(x.to(dtype=self.out_layer.weight.dtype))


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------


class Kandinsky6TransformerEncoderBlock(nn.Module):
    """Text-only self-attention + FFN block."""

    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        ff_dim: int,
        head_dim: int,
        engine: str = "auto",
        text_token_padding: bool = False,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.text_modulation = Kandinsky6Modulation(
            time_dim,
            model_dim,
            6,
            quant_config=quant_config,
            prefix=f"{prefix}.text_modulation" if prefix else "text_modulation",
        )
        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = Kandinsky6Attention(
            model_dim,
            head_dim,
            engine=engine,
            text_token_padding=text_token_padding,
            role="kandinsky6.text_self",
            role_category="self",
            quant_config=quant_config,
            prefix=f"{prefix}.self_attention" if prefix else "self_attention",
        )
        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = Kandinsky6FeedForward(
            model_dim, ff_dim, quant_config=quant_config, prefix=f"{prefix}.feed_forward" if prefix else "feed_forward"
        )

    def forward(self, x: Tensor, time_embed: Tensor, rope: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        sa_params, ff_params = torch.chunk(self.text_modulation(time_embed), 2, dim=-1)
        shift, scale, gate = torch.chunk(sa_params, 3, dim=-1)
        x = apply_gate_sum(
            x,
            self.self_attention(
                apply_scale_shift_norm(self.self_attention_norm, x, scale, shift),
                rotary_emb=rope,
                attn_mask=attn_mask,
            ),
            gate,
        )
        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        return apply_gate_sum(
            x, self.feed_forward(apply_scale_shift_norm(self.feed_forward_norm, x, scale, shift)), gate
        )


class Kandinsky6TransformerDecoderBlock(nn.Module):
    """Visual self-attention + cross-attention to text + FFN block."""

    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        ff_dim: int,
        head_dim: int,
        engine: str = "auto",
        text_token_padding: bool = False,
        *,
        self_sequence_parallel: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.visual_modulation = Kandinsky6Modulation(
            time_dim,
            model_dim,
            9,
            quant_config=quant_config,
            prefix=f"{prefix}.visual_modulation" if prefix else "visual_modulation",
        )
        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = Kandinsky6Attention(
            model_dim,
            head_dim,
            engine=engine,
            visual=True,
            sequence_parallel=self_sequence_parallel,
            role="kandinsky6.visual_self" if self_sequence_parallel else "kandinsky6.audio_self",
            role_category="self",
            quant_config=quant_config,
            prefix=f"{prefix}.self_attention" if prefix else "self_attention",
        )
        self.cross_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.cross_attention = Kandinsky6Attention(
            model_dim,
            head_dim,
            kv_dim=model_dim,
            engine=engine,
            text_token_padding=text_token_padding,
            role="kandinsky6.text_cross",
            role_category="cross",
            quant_config=quant_config,
            prefix=f"{prefix}.cross_attention" if prefix else "cross_attention",
        )
        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = Kandinsky6FeedForward(
            model_dim, ff_dim, quant_config=quant_config, prefix=f"{prefix}.feed_forward" if prefix else "feed_forward"
        )

    def forward(
        self,
        vis: Tensor,
        text: Tensor,
        time_embed: Tensor,
        rope: Tensor,
        sparse_params: dict | None,
        attn_mask=None,
    ) -> Tensor:
        sa_params, ca_params, ff_params = torch.chunk(self.visual_modulation(time_embed), 3, dim=-1)
        shift, scale, gate = torch.chunk(sa_params, 3, dim=-1)
        vis = apply_gate_sum(
            vis,
            self.self_attention(
                apply_scale_shift_norm(self.self_attention_norm, vis, scale, shift),
                rotary_emb=rope,
                sparse_params=sparse_params,
            ),
            gate,
        )
        shift, scale, gate = torch.chunk(ca_params, 3, dim=-1)
        vis = apply_gate_sum(
            vis,
            self.cross_attention(
                apply_scale_shift_norm(self.cross_attention_norm, vis, scale, shift),
                encoder_hidden_states=text,
                attn_mask=attn_mask,
            ),
            gate,
        )
        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        return apply_gate_sum(
            vis, self.feed_forward(apply_scale_shift_norm(self.feed_forward_norm, vis, scale, shift)), gate
        )


class Kandinsky6FusedTransformerDecoderBlock(nn.Module):
    """Fused video + audio block with cross-modal attention (T2VA backbone)."""

    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        ff_dim: int,
        head_dim: int,
        model_dim_a: int,
        time_dim_a: int,
        ff_dim_a: int,
        head_dim_a: int,
        engine: str = "auto",
        text_token_padding: bool = False,
        ca_rope: bool = False,
        cross_gates: bool = False,
        fix_modulation: bool = False,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.video_dec_block = Kandinsky6TransformerDecoderBlock(
            model_dim,
            time_dim,
            ff_dim,
            head_dim,
            engine,
            text_token_padding,
            self_sequence_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.video_dec_block" if prefix else "video_dec_block",
        )
        self.audio_dec_block = Kandinsky6TransformerDecoderBlock(
            model_dim_a,
            time_dim_a,
            ff_dim_a,
            head_dim_a,
            engine,
            text_token_padding,
            self_sequence_parallel=False,
            quant_config=quant_config,
            prefix=f"{prefix}.audio_dec_block" if prefix else "audio_dec_block",
        )

        self.va_cross_attention = Kandinsky6Attention(
            model_dim,
            head_dim,
            model_dim_a,
            engine,
            role="kandinsky6.video_audio_cross",
            role_category="cross",
            quant_config=quant_config,
            prefix=f"{prefix}.va_cross_attention" if prefix else "va_cross_attention",
        )
        self.av_cross_attention = Kandinsky6Attention(
            model_dim_a,
            head_dim_a,
            model_dim,
            engine,
            role="kandinsky6.audio_video_cross",
            role_category="cross",
            quant_config=quant_config,
            prefix=f"{prefix}.av_cross_attention" if prefix else "av_cross_attention",
        )

        self.va_modulation = Kandinsky6Modulation(
            time_dim,
            model_dim if not cross_gates else model_dim * 2 + model_dim_a,
            1 if cross_gates else 3,
            quant_config=quant_config,
            prefix=f"{prefix}.va_modulation" if prefix else "va_modulation",
        )
        self.av_modulation = Kandinsky6Modulation(
            time_dim_a,
            model_dim_a if not cross_gates else model_dim_a * 2 + model_dim,
            1 if cross_gates else 3,
            quant_config=quant_config,
            prefix=f"{prefix}.av_modulation" if prefix else "av_modulation",
        )
        self.va_normalization = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.av_normalization = nn.LayerNorm(model_dim_a, elementwise_affine=False)

        self.ca_rope = ca_rope
        self.cross_gates = cross_gates
        self.fix_modulation = fix_modulation
        self.model_dim = model_dim
        self.model_dim_a = model_dim_a

    def forward(
        self,
        vis: Tensor,
        aud: Tensor,
        text_v: Tensor,
        text_a: Tensor,
        time_embed,  # (video_time, audio_time) tuple
        vis_rope: Tensor,
        aud_rope: Tensor,
        sparse_params: dict | None,
        attn_mask=None,
        modality_mask=None,
        av_gate_scale: float = 1.0,
        va_gate_scale: float = 1.0,
    ):
        fake_audio = modality_mask[0] if modality_mask is not None else 0
        fake_video = modality_mask[1] if modality_mask is not None else 0
        t_v, t_a = time_embed

        # ---- video backbone ----
        if vis is not None:
            sa_p, ca_p, ff_p = torch.chunk(self.video_dec_block.visual_modulation(t_v), 3, dim=-1)

            shift, scale, gate = torch.chunk(sa_p, 3, dim=-1)
            vis = apply_gate_sum(
                vis,
                self.video_dec_block.self_attention(
                    apply_scale_shift_norm(self.video_dec_block.self_attention_norm, vis, scale, shift),
                    rotary_emb=vis_rope,
                    sparse_params=sparse_params,
                ),
                gate,
            ).type_as(vis)

            shift, scale, gate_v = torch.chunk(ca_p, 3, dim=-1)
            vis_pre_ca = apply_scale_shift_norm(self.video_dec_block.cross_attention_norm, vis, scale, shift).type_as(
                vis
            )
            vis_out_t = self.video_dec_block.cross_attention(
                vis_pre_ca, encoder_hidden_states=text_v, attn_mask=attn_mask
            )

        # ---- audio backbone ----
        if aud is not None:
            sa_p, ca_p, ff_p_a = torch.chunk(self.audio_dec_block.visual_modulation(t_a), 3, dim=-1)

            shift, scale, gate = torch.chunk(sa_p, 3, dim=-1)
            aud = apply_gate_sum(
                aud,
                self.audio_dec_block.self_attention(
                    apply_scale_shift_norm(self.audio_dec_block.self_attention_norm, aud, scale, shift),
                    rotary_emb=aud_rope,
                ),
                gate,
            ).type_as(aud)

            shift, scale, gate_a = torch.chunk(ca_p, 3, dim=-1)
            aud_pre_ca = apply_scale_shift_norm(self.audio_dec_block.cross_attention_norm, aud, scale, shift).type_as(
                aud
            )
            aud_out_t = self.audio_dec_block.cross_attention(
                aud_pre_ca, encoder_hidden_states=text_a, attn_mask=attn_mask
            )
            aud = apply_gate_sum(aud, aud_out_t, gate_a).type_as(aud)

            # ---- cross-modal attention ----
            if vis is not None:
                t_va_mod = t_a if not self.fix_modulation else t_v
                t_av_mod = t_v if not self.fix_modulation else t_a
                va_params = self.va_modulation(t_va_mod)
                av_params = self.av_modulation(t_av_mod)

                if self.cross_gates:
                    va_shift, va_scale, va_gate = torch.split(
                        va_params, [self.model_dim, self.model_dim, self.model_dim_a], dim=-1
                    )
                    av_shift, av_scale, av_gate = torch.split(
                        av_params, [self.model_dim_a, self.model_dim_a, self.model_dim], dim=-1
                    )
                else:
                    va_shift, va_scale, va_gate = torch.chunk(va_params, 3, dim=-1)
                    av_shift, av_scale, av_gate = torch.chunk(av_params, 3, dim=-1)

                vis = apply_gate_sum(vis, vis_out_t, gate_v).type_as(vis)
                vis_for_va = apply_scale_shift_norm(self.va_normalization, vis, va_scale, va_shift).type_as(vis)
                aud_for_av = apply_scale_shift_norm(self.av_normalization, aud, av_scale, av_shift).type_as(aud)

                # V->A and A->V attention. Visual tokens may be sequence-sharded;
                # audio queries need the full visual key/value sequence.
                rq_v = vis_rope if self.ca_rope else None
                rk_a = aud_rope if self.ca_rope else None
                vis_kv = vis_pre_ca
                vis_rope_kv = rq_v
                if _parallel_size("sp") > 1 and vis_pre_ca is not None:
                    from vllm_omni.diffusion.distributed.sp_sharding import sp_gather

                    vis_kv = sp_gather(vis_pre_ca, dim=1)
                    if vis_rope_kv is not None:
                        vis_rope_kv = sp_gather(vis_rope, dim=0)
                vis_from_aud = (
                    self.va_cross_attention(vis_for_va, encoder_hidden_states=aud_pre_ca, rope_q=rq_v, rope_kv=rk_a)
                    * (1 - fake_audio)
                    * (1 - fake_video)
                )
                aud_from_vis = (
                    self.av_cross_attention(aud_for_av, encoder_hidden_states=vis_kv, rope_q=rk_a, rope_kv=vis_rope_kv)
                    * (1 - fake_audio)
                    * (1 - fake_video)
                )

                va_g = (va_gate if not self.cross_gates else av_gate) * va_gate_scale
                av_g = (av_gate if not self.cross_gates else va_gate) * av_gate_scale
                vis = apply_gate_sum(vis, vis_from_aud, va_g).type_as(vis)
                aud = apply_gate_sum(aud, aud_from_vis, av_g).type_as(aud)
        else:
            vis = apply_gate_sum(vis, vis_out_t, gate_v).type_as(vis)

        # ---- FFN ----
        if vis is not None:
            shift, scale, gate = torch.chunk(ff_p, 3, dim=-1)
            vis = apply_gate_sum(
                vis,
                self.video_dec_block.feed_forward(
                    apply_scale_shift_norm(self.video_dec_block.feed_forward_norm, vis, scale, shift)
                ),
                gate,
            ).type_as(vis)

        if aud is not None:
            shift, scale, gate = torch.chunk(ff_p_a, 3, dim=-1)
            aud = apply_gate_sum(
                aud,
                self.audio_dec_block.feed_forward(
                    apply_scale_shift_norm(self.audio_dec_block.feed_forward_norm, aud, scale, shift)
                ),
                gate,
            ).type_as(aud)

        return vis, aud


# ---------------------------------------------------------------------------
# Unified DiffusionTransformer3D (tensor-parallel)
# ---------------------------------------------------------------------------


class Kandinsky6Transformer3DModel(nn.Module):
    """Kandinsky 6 DiT — handles T2V (is_multimodal=False) and T2VA (is_multimodal=True).

    Tensor-parallel port of ``core/components/dit.py::DiffusionTransformer3D``,
    with submodule names/nesting matching the Diffusers-exported checkpoint
    layout (see module docstring) so ``from_pretrained`` loads a
    ``convert_checkpoint.py --use-patched-diffusers`` bundle directly.

    The class is a plain ``nn.Module``. ``from_diffusers_config`` reads the
    bundle ``config.json``. Safetensors from ``transformer/`` are applied
    later by the pipeline ``load_weights`` path, which narrows full
    checkpoint tensors onto each rank's ``ColumnParallelLinear`` /
    ``RowParallelLinear`` shard.
    """

    _repeated_blocks = [
        "Kandinsky6FusedTransformerDecoderBlock",
        "Kandinsky6TransformerEncoderBlock",
        "Kandinsky6TransformerDecoderBlock",
    ]

    # MagCache must not treat text and visual ModuleLists as one residual chain.
    _magcache_block_attrs = ("visual_transformer_blocks",)

    @classmethod
    def from_diffusers_config(
        cls,
        config: dict,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> Kandinsky6Transformer3DModel:
        """Build the DiT from a Diffusers ``config.json`` dict."""
        fields = (
            "in_visual_dim",
            "out_visual_dim",
            "in_text_dim",
            "in_text_dim2",
            "time_dim",
            "patch_size",
            "model_dim",
            "ff_dim",
            "num_text_blocks",
            "num_visual_blocks",
            "axes_dims",
            "visual_cond",
            "is_multimodal",
            "in_audio_dim",
            "model_dim_a",
            "time_dim_a",
            "ff_dim_a",
            "axes_dims_a",
            "audio_freqs_scaling",
            "attention_engine",
            "text_token_padding",
            "ca_rope",
            "cross_gates",
            "fix_modulation",
            "visual_token_type_num_embeddings",
        )
        kwargs = {key: config[key] for key in fields if key in config}
        for key in ("patch_size", "axes_dims", "axes_dims_a"):
            if isinstance(kwargs.get(key), list):
                kwargs[key] = tuple(kwargs[key])
        return cls(**kwargs, quant_config=quant_config, prefix=prefix)

    @staticmethod
    def _is_transformer_block(name: str, module: nn.Module) -> bool:
        del module
        leaf = name.rsplit(".", 1)[-1]
        if not leaf.isdigit():
            return False
        return any(
            part in name
            for part in (
                "visual_transformer_blocks",
                "text_transformer_blocks",
                "video_text_transformer_blocks",
                "audio_text_transformer_blocks",
            )
        )

    _hsdp_shard_conditions = [_is_transformer_block]

    # Visual tokens are (1, N, D). RoPE after fractal flatten is (N, 1, C, 2, 2).
    # Text and audio stay replicated; cross-modal KV is gathered in the fused block.
    _sp_plan = {
        "_sp_visual_shard": {
            0: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True, auto_pad=True),
        },
        "_sp_visual_rope": {
            0: SequenceParallelInput(split_dim=0, expected_dims=5, split_output=True, auto_pad=True),
        },
        "_sp_visual_gather": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }

    def __init__(
        self,
        in_visual_dim: int = 16,
        out_visual_dim: int = 16,
        in_text_dim: int = 3584,
        in_text_dim2: int = 768,
        time_dim: int = 1024,
        patch_size: tuple = (1, 2, 2),
        model_dim: int = 4096,
        ff_dim: int = 16384,
        num_text_blocks: int = 4,
        num_visual_blocks: int = 60,
        axes_dims: tuple = (32, 48, 48),
        visual_cond: bool = True,
        is_multimodal: bool = False,
        # Audio (T2VA only)
        in_audio_dim: int = 20,
        model_dim_a: int | None = None,
        time_dim_a: int | None = None,
        ff_dim_a: int | None = None,
        axes_dims_a: tuple | None = None,
        audio_freqs_scaling: float = 1.0,
        # Misc
        attention_engine: str = "auto",
        text_token_padding: bool = False,
        ca_rope: bool = False,
        cross_gates: bool = False,
        fix_modulation: bool = False,
        # I2VA: 0 = off; 2 = generated vs reference frame (tail_cond_first_frame)
        visual_token_type_num_embeddings: int = 0,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.visual_cond = visual_cond
        self.is_multimodal = is_multimodal
        self.in_visual_dim = in_visual_dim
        self.in_audio_dim = in_audio_dim
        self.text_token_padding = text_token_padding
        self.visual_token_type_num_embeddings = int(visual_token_type_num_embeddings or 0)
        # Time-independent text/pooled projections (cleared each generation).
        self._text_proj_cache: dict[tuple, object] = {}

        head_dim = sum(axes_dims)

        # Effective audio dims (default to video dims)
        model_dim_a = model_dim_a or model_dim
        time_dim_a = time_dim_a or time_dim
        ff_dim_a = ff_dim_a or ff_dim
        axes_dims_a = axes_dims_a or axes_dims
        head_dim_a = sum(axes_dims_a)

        # ---- visual backbone (shared) ----
        vis_in_dim = (2 * in_visual_dim + 1) if visual_cond else in_visual_dim
        self.visual_embeddings = Kandinsky6VisualEmbeddings(
            vis_in_dim, model_dim, patch_size, quant_config=quant_config, prefix=f"{prefix}.visual_embeddings"
        )
        if self.visual_token_type_num_embeddings > 0:
            self.visual_token_type_embeddings = nn.Embedding(self.visual_token_type_num_embeddings, model_dim)
        self.visual_rope_embeddings = RoPE3D(axes_dims)
        self.out_layer = Kandinsky6OutLayer(
            model_dim, time_dim, out_visual_dim, patch_size, quant_config=quant_config, prefix=f"{prefix}.out_layer"
        )

        if not is_multimodal:
            # T2V: single text/time embedding branch
            self.time_embeddings = Kandinsky6TimeEmbeddings(
                model_dim, time_dim, quant_config=quant_config, prefix=f"{prefix}.time_embeddings"
            )
            self.text_embeddings = Kandinsky6TextEmbeddings(
                in_text_dim, model_dim, quant_config=quant_config, prefix=f"{prefix}.text_embeddings"
            )
            self.pooled_text_embeddings = Kandinsky6TextEmbeddings(
                in_text_dim2, time_dim, quant_config=quant_config, prefix=f"{prefix}.pooled_text_embeddings"
            )
            self.text_rope_embeddings = RoPE1D(head_dim)
            self.text_transformer_blocks = nn.ModuleList(
                [
                    Kandinsky6TransformerEncoderBlock(
                        model_dim,
                        time_dim,
                        ff_dim,
                        head_dim,
                        attention_engine,
                        text_token_padding,
                        quant_config=quant_config,
                        prefix=f"{prefix}.text_transformer_blocks.{i}",
                    )
                    for i in range(num_text_blocks)
                ]
            )

            def _decoder_block(i: int) -> Kandinsky6TransformerDecoderBlock:
                return Kandinsky6TransformerDecoderBlock(
                    model_dim,
                    time_dim,
                    ff_dim,
                    head_dim,
                    attention_engine,
                    text_token_padding,
                    self_sequence_parallel=True,
                    quant_config=quant_config,
                    prefix=f"{prefix}.visual_transformer_blocks.{i}",
                )

            self._pp_block_start, self._pp_block_end, self.visual_transformer_blocks = _partition_visual_blocks(
                num_visual_blocks, _decoder_block
            )
        else:
            # T2VA: dual (video / audio) text+time branches + fused blocks
            self.audio_embeddings = Kandinsky6TextEmbeddings(
                in_audio_dim, model_dim_a, quant_config=quant_config, prefix=f"{prefix}.audio_embeddings"
            )
            self.audio_rope_embeddings = RoPE1D(head_dim_a, freqs_scaling=audio_freqs_scaling)
            self.audio_out_layer = Kandinsky6OutLayerAudio(
                model_dim_a, time_dim_a, in_audio_dim, quant_config=quant_config, prefix=f"{prefix}.audio_out_layer"
            )

            for mod_prefix, md, td, fd, hd in [
                ("video", model_dim, time_dim, ff_dim, head_dim),
                ("audio", model_dim_a, time_dim_a, ff_dim_a, head_dim_a),
            ]:
                setattr(
                    self,
                    f"{mod_prefix}_time_embeddings",
                    Kandinsky6TimeEmbeddings(
                        md, td, quant_config=quant_config, prefix=f"{prefix}.{mod_prefix}_time_embeddings"
                    ),
                )
                setattr(
                    self,
                    f"{mod_prefix}_text_embeddings",
                    Kandinsky6TextEmbeddings(
                        in_text_dim, md, quant_config=quant_config, prefix=f"{prefix}.{mod_prefix}_text_embeddings"
                    ),
                )
                setattr(
                    self,
                    f"{mod_prefix}_pooled_text_embeddings",
                    Kandinsky6TextEmbeddings(
                        in_text_dim2,
                        td,
                        quant_config=quant_config,
                        prefix=f"{prefix}.{mod_prefix}_pooled_text_embeddings",
                    ),
                )
                setattr(self, f"{mod_prefix}_text_rope_embeddings", RoPE1D(hd))
                setattr(
                    self,
                    f"{mod_prefix}_text_transformer_blocks",
                    nn.ModuleList(
                        [
                            Kandinsky6TransformerEncoderBlock(
                                md,
                                td,
                                fd,
                                hd,
                                attention_engine,
                                text_token_padding,
                                quant_config=quant_config,
                                prefix=f"{prefix}.{mod_prefix}_text_transformer_blocks.{i}",
                            )
                            for i in range(num_text_blocks)
                        ]
                    ),
                )

            def _fused_block(i: int) -> Kandinsky6FusedTransformerDecoderBlock:
                return Kandinsky6FusedTransformerDecoderBlock(
                    model_dim,
                    time_dim,
                    ff_dim,
                    head_dim,
                    model_dim_a,
                    time_dim_a,
                    ff_dim_a,
                    head_dim_a,
                    attention_engine,
                    text_token_padding,
                    ca_rope=ca_rope,
                    cross_gates=cross_gates,
                    fix_modulation=fix_modulation,
                    quant_config=quant_config,
                    prefix=f"{prefix}.visual_transformer_blocks.{i}",
                )

            self._pp_block_start, self._pp_block_end, self.visual_transformer_blocks = _partition_visual_blocks(
                num_visual_blocks, _fused_block
            )

        from vllm.model_executor.models.utils import PPMissingLayer

        # Embeddings live on the first PP stage, heads on the last. Text/time
        # modules stay on every rank (they are small next to the visual stack).
        if _parallel_size("pp") > 1 and _parallel_size("pp_rank") != 0:
            self.visual_embeddings = PPMissingLayer()
            if hasattr(self, "audio_embeddings"):
                self.audio_embeddings = PPMissingLayer()
        if _parallel_size("pp") > 1 and _parallel_size("pp_rank") != _parallel_size("pp") - 1:
            self.out_layer = PPMissingLayer()
            if hasattr(self, "audio_out_layer"):
                self.audio_out_layer = PPMissingLayer()
        self._sp_visual_shard = nn.Identity()
        self._sp_visual_rope = nn.Identity()
        self._sp_visual_gather = nn.Identity()
        self._pp_final = None

    # ------------------------------------------------------------------
    # Stage helpers (shared by forward / MagCache) — unchanged from core.
    # ------------------------------------------------------------------

    def clear_text_proj_cache(self) -> None:
        """Drop cached text/pooled projections (call once per generation)."""
        self._text_proj_cache.clear()

    def _project_pooled(self, prefix: str | None, pooled: Tensor) -> Tensor:
        key = ("pe", prefix, pooled.data_ptr(), tuple(pooled.shape))
        hit = self._text_proj_cache.get(key)
        if hit is not None:
            return hit  # type: ignore[return-value]
        if prefix is None:
            pe = self.pooled_text_embeddings(pooled)
        else:
            pe = getattr(self, f"{prefix}_pooled_text_embeddings")(pooled)
        self._text_proj_cache[key] = pe
        return pe

    def _project_text_tokens(
        self,
        prefix: str | None,
        text_embed: Tensor,
        pooled: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Time-independent token + pooled Linears (cached within a generation)."""
        key = (
            "te",
            prefix,
            text_embed.data_ptr(),
            pooled.data_ptr(),
            tuple(text_embed.shape),
            tuple(pooled.shape),
        )
        hit = self._text_proj_cache.get(key)
        if hit is not None:
            return hit  # type: ignore[return-value]
        if prefix is None:
            te = self.text_embeddings(text_embed)
        else:
            te = getattr(self, f"{prefix}_text_embeddings")(text_embed)
        pe = self._project_pooled(prefix, pooled)
        self._text_proj_cache[key] = (te, pe)
        return te, pe

    def _time_embed(self, prefix: str | None, time: Tensor, pooled_proj: Tensor) -> Tensor:
        if prefix is None:
            return self.time_embeddings(time) + pooled_proj
        return getattr(self, f"{prefix}_time_embeddings")(time) + pooled_proj

    @staticmethod
    def _normalize_attn_mask(attn_mask: Tensor | None) -> Tensor | None:
        """HF/K5 key-padding (True=valid): ``(S,)`` -> ``(1, S)`` for SDPA."""
        if attn_mask is None:
            return None
        if attn_mask.dim() == 1:
            return attn_mask.unsqueeze(0)
        return attn_mask

    def _run_text_blocks(
        self,
        prefix: str | None,
        te: Tensor,
        tm: Tensor,
        text_rope: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        blocks = self.text_transformer_blocks if prefix is None else getattr(self, f"{prefix}_text_transformer_blocks")
        for blk in blocks:
            te = blk(te, tm, text_rope, attn_mask)
        return te

    def _encode_text(
        self,
        prefix: str,
        text_embed: Tensor,
        pooled: Tensor,
        time: Tensor,
        text_rope: Tensor,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        te, pe = self._project_text_tokens(prefix, text_embed, pooled)
        tm = self._time_embed(prefix, time, pe)
        te = self._run_text_blocks(prefix, te, tm, text_rope, attn_mask)
        return te, tm

    def _encode_t2v(
        self,
        text_embed: Tensor,
        pooled: Tensor,
        time: Tensor,
        text_rope: Tensor,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        te, pe = self._project_text_tokens(None, text_embed, pooled)
        tm = self._time_embed(None, time, pe)
        te = self._run_text_blocks(None, te, tm, text_rope, attn_mask)
        return te, tm

    def _time_only(self, prefix: str | None, pooled: Tensor, time: Tensor) -> Tensor:
        """Pooled (cached) + time — OutLayer path when MagCache skips text+visual blocks."""
        pe = self._project_pooled(prefix, pooled)
        return self._time_embed(prefix, time, pe)

    def _embed_visual(
        self,
        x_video: Tensor,
        visual_rope: Tensor,
        sparse_params: dict | None,
        *,
        apply_fractal: bool = True,
        visual_token_type_ids: Tensor | None = None,
    ) -> tuple[Tensor, tuple, Tensor]:
        vis_embed = self.visual_embeddings(x_video)
        if hasattr(self, "visual_token_type_embeddings") and visual_token_type_ids is not None:
            if visual_token_type_ids.shape[0] != vis_embed.shape[0]:
                raise ValueError(
                    "visual_token_type_ids length must match visual latent frames: "
                    f"type_ids={tuple(visual_token_type_ids.shape)}, "
                    f"visual={tuple(vis_embed.shape)}"
                )
            vis_embed = (
                vis_embed
                + self.visual_token_type_embeddings(visual_token_type_ids.to(device=vis_embed.device))[:, None, None, :]
            )
        vis_shape = vis_embed.shape[:-1]
        vis_rope = visual_rope
        if apply_fractal:
            to_fractal = sparse_params["to_fractal"] if sparse_params else False
            vis_embed, vis_rope = fractal_flatten(vis_embed, vis_rope, vis_shape, block_mask=to_fractal)
        else:
            vis_embed, vis_rope = fractal_flatten(vis_embed, vis_rope, vis_shape)
        vis_embed = vis_embed.unsqueeze(0)
        return vis_embed, vis_shape, vis_rope

    def _embed_audio(self, x_audio: Tensor, audio_rope: Tensor) -> tuple[Tensor, Tensor]:
        aud_embed = self.audio_embeddings(x_audio).unsqueeze(0)
        return aud_embed, audio_rope

    def _sp_enter(self, vis: Tensor | None, rope: Tensor | None) -> tuple[Tensor | None, Tensor | None]:
        if vis is None or (_parallel_size("pp") > 1 and _parallel_size("pp_rank") != 0):
            return vis, rope
        vis = self._sp_visual_shard(vis)
        if rope is not None:
            rope = self._sp_visual_rope(rope)
        return vis, rope

    def _sp_exit(self, vis: Tensor | None) -> Tensor | None:
        if vis is None:
            return None
        if _parallel_size("pp") > 1 and _parallel_size("pp_rank") != _parallel_size("pp") - 1:
            return vis
        return self._sp_visual_gather(vis)

    def _pp_recv_hidden(
        self,
        vis: Tensor | None,
        aud: Tensor | None,
        vis_rope: Tensor | None = None,
        aud_rope: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        pp = _parallel_size("pp")
        rank = _parallel_size("pp_rank")
        if pp <= 1 or rank == 0:
            return vis, aud, vis_rope, aud_rope
        from vllm_omni.diffusion.distributed.parallel_state import get_pp_group

        payload = get_pp_group().recv_tensor_dict(src=rank - 1)
        assert payload is not None
        if payload.get("shape") is not None:
            self._pp_vis_shape = payload["shape"]
        return (
            payload.get("vis"),
            payload.get("aud"),
            payload.get("vis_rope", vis_rope),
            payload.get("aud_rope", aud_rope),
        )

    def _pp_send_hidden_or_wait(
        self,
        vis: Tensor | None,
        aud: Tensor | None,
        vis_rope: Tensor | None = None,
        aud_rope: Tensor | None = None,
    ):
        """Non-last PP ranks forward hidden states and wait for the velocity."""
        pp = _parallel_size("pp")
        if pp <= 1:
            return None
        rank = _parallel_size("pp_rank")
        if rank == pp - 1:
            return None
        from vllm_omni.diffusion.distributed.parallel_state import get_pp_group

        group = get_pp_group()
        group.send_tensor_dict(
            {
                "vis": vis,
                "aud": aud,
                "vis_rope": vis_rope,
                "aud_rope": aud_rope,
                "shape": getattr(self, "_pp_vis_shape", None),
            },
            dst=rank + 1,
        )
        final = group.recv_tensor_dict(src=pp - 1)
        assert final is not None
        if "pair" in final:
            return final["video"], final["audio"]
        return final["video"]

    def _pp_publish(self, result: Tensor | tuple[Tensor, Tensor]):
        pp = _parallel_size("pp")
        if pp <= 1 or _parallel_size("pp_rank") != pp - 1:
            return result
        from vllm_omni.diffusion.distributed.parallel_state import get_pp_group

        group = get_pp_group()
        if isinstance(result, tuple):
            payload = {"pair": True, "video": result[0], "audio": result[1]}
        else:
            payload = {"video": result}
        for dst in range(pp - 1):
            group.send_tensor_dict(payload, dst=dst)
        return result

    def _iter_local_visual_blocks(self):
        start = getattr(self, "_pp_block_start", 0)
        end = getattr(self, "_pp_block_end", len(self.visual_transformer_blocks))
        for index, block in enumerate(self.visual_transformer_blocks):
            if index < start or index >= end:
                continue
            if type(block).__name__ == "PPMissingLayer":
                continue
            yield block

    def _run_visual_blocks_single(
        self,
        vis_embed: Tensor | None,
        aud_embed: Tensor | None,
        te: Tensor,
        tm: Tensor,
        vis_rope: Tensor | None,
        aud_rope: Tensor | None,
        sparse_params: dict | None,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None]:
        self._pp_final = None
        vis_embed, aud_embed, vis_rope, aud_rope = self._pp_recv_hidden(vis_embed, aud_embed, vis_rope, aud_rope)
        vis_embed, vis_rope = self._sp_enter(vis_embed, vis_rope)
        token = _VISUAL_SP.set(_parallel_size("sp") > 1 and vis_embed is not None)
        try:
            for blk in self._iter_local_visual_blocks():
                if self.is_multimodal:
                    if vis_embed is not None and aud_embed is None:
                        vis_embed, _ = blk(
                            vis_embed,
                            None,
                            te,
                            te,
                            (tm, tm),
                            vis_rope,
                            None,
                            sparse_params,
                            attn_mask,
                        )
                    elif aud_embed is not None and vis_embed is None:
                        _, aud_embed = blk(
                            None,
                            aud_embed,
                            te,
                            te,
                            (tm, tm),
                            None,
                            aud_rope,
                            None,
                            attn_mask,
                        )
                    else:
                        raise RuntimeError("single-modality fused path expects exactly one of video/audio")
                elif vis_embed is not None:
                    vis_embed = blk(vis_embed, te, tm, vis_rope, sparse_params, attn_mask)
                else:
                    aud_embed = blk(aud_embed, te, tm, aud_rope, None, attn_mask)
        finally:
            _VISUAL_SP.reset(token)
        early = self._pp_send_hidden_or_wait(vis_embed, aud_embed, vis_rope, aud_rope)
        if early is not None:
            self._pp_final = early
            return None, None
        return vis_embed, aud_embed

    def _run_visual_blocks_fused(
        self,
        vis_embed: Tensor,
        aud_embed: Tensor,
        video_te: Tensor,
        audio_te: Tensor,
        video_tm: Tensor,
        audio_tm: Tensor,
        vis_rope: Tensor,
        aud_rope: Tensor,
        sparse_params: dict | None,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        self._pp_final = None
        vis_embed, aud_embed, vis_rope, aud_rope = self._pp_recv_hidden(vis_embed, aud_embed, vis_rope, aud_rope)
        vis_embed, vis_rope = self._sp_enter(vis_embed, vis_rope)
        token = _VISUAL_SP.set(_parallel_size("sp") > 1)
        try:
            for blk in self._iter_local_visual_blocks():
                vis_embed, aud_embed = blk(
                    vis_embed,
                    aud_embed,
                    video_te,
                    audio_te,
                    (video_tm, audio_tm),
                    vis_rope,
                    aud_rope,
                    sparse_params,
                    attn_mask,
                )
        finally:
            _VISUAL_SP.reset(token)
        early = self._pp_send_hidden_or_wait(vis_embed, aud_embed, vis_rope, aud_rope)
        if early is not None:
            self._pp_final = early
            return None, None  # type: ignore[return-value]
        return vis_embed, aud_embed

    def _project_video(
        self,
        vis_embed: Tensor,
        vis_shape: tuple,
        tm: Tensor,
        sparse_params: dict | None,
    ) -> Tensor:
        to_fractal = sparse_params["to_fractal"] if sparse_params else False
        if vis_shape is None:
            vis_shape = getattr(self, "_pp_vis_shape", None)
        vis_embed = self._sp_exit(vis_embed)
        vis_embed = fractal_unflatten(vis_embed, vis_shape, block_mask=to_fractal)
        return self._pp_publish(self.out_layer(vis_embed, tm))

    def _project_audio(self, aud_embed: Tensor, tm: Tensor) -> Tensor:
        return self._pp_publish(self.audio_out_layer(aud_embed.squeeze(0), tm))

    def _project_fused(
        self,
        vis_embed: Tensor,
        aud_embed: Tensor,
        vis_shape: tuple,
        video_tm: Tensor,
        audio_tm: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if vis_shape is None:
            vis_shape = self._pp_vis_shape
        vis_embed = self._sp_exit(vis_embed)
        vis_embed = fractal_unflatten(vis_embed, vis_shape)
        video_vel = self.out_layer(vis_embed, video_tm)
        audio_vel = self.audio_out_layer(aud_embed.squeeze(0), audio_tm)
        return self._pp_publish((video_vel, audio_vel))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_video: Tensor | None,
        x_audio: Tensor | None,
        text_embed: Tensor | list[Tensor],
        pooled_text_embed: Tensor | list[Tensor],
        time: Tensor | list[Tensor],
        visual_rope: Tensor | None,
        audio_rope: Tensor | None,
        text_rope: Tensor | list[Tensor],
        sparse_params: dict | None = None,
        attention_mask: Tensor | None = None,
        visual_token_type_ids: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """RoPE tensors are precomputed outside (pipeline / export); not from positions."""
        both = x_video is not None and x_audio is not None and self.is_multimodal
        attn_mask = self._normalize_attn_mask(attention_mask)

        if not both:
            te_in = text_embed[0] if isinstance(text_embed, list) else text_embed
            pe_in = pooled_text_embed[0] if isinstance(pooled_text_embed, list) else pooled_text_embed
            rope_in = text_rope[0] if isinstance(text_rope, list) else text_rope
            t_in = time[0] if isinstance(time, list) else time

            if self.is_multimodal:
                prefix = "audio" if x_audio is not None else "video"
                # Multimodal single-modality path: list text_rope is [video, audio].
                if isinstance(text_rope, list):
                    rope_in = text_rope[1] if prefix == "audio" else text_rope[0]
                te, tm = self._encode_text(prefix, te_in, pe_in, t_in, rope_in, attn_mask)
            else:
                te, tm = self._encode_t2v(te_in, pe_in, t_in, rope_in, attn_mask)

            if x_video is not None:
                if _parallel_size("pp") > 1 and _parallel_size("pp_rank") != 0:
                    vis_embed, vis_shape, vis_rope = None, getattr(self, "_pp_vis_shape", None), visual_rope
                else:
                    vis_embed, vis_shape, vis_rope = self._embed_visual(
                        x_video,
                        visual_rope,
                        sparse_params,
                        apply_fractal=True,
                        visual_token_type_ids=visual_token_type_ids,
                    )
                    self._pp_vis_shape = vis_shape
                vis_embed, _ = self._run_visual_blocks_single(
                    vis_embed,
                    None,
                    te,
                    tm,
                    vis_rope,
                    None,
                    sparse_params,
                    attn_mask,
                )
                if self._pp_final is not None:
                    final = self._pp_final
                    self._pp_final = None
                    return final
                return self._project_video(vis_embed, self._pp_vis_shape, tm, sparse_params)

            aud_embed, aud_rope = self._embed_audio(x_audio, audio_rope)
            _, aud_embed = self._run_visual_blocks_single(
                None,
                aud_embed,
                te,
                tm,
                None,
                aud_rope,
                sparse_params,
                attn_mask,
            )
            if self._pp_final is not None:
                final = self._pp_final
                self._pp_final = None
                return final
            return self._project_audio(aud_embed, tm)

        te_v, pe_v = (
            (text_embed[0], pooled_text_embed[0]) if isinstance(text_embed, list) else (text_embed, pooled_text_embed)
        )
        te_a, pe_a = (
            (text_embed[1], pooled_text_embed[1]) if isinstance(text_embed, list) else (text_embed, pooled_text_embed)
        )
        if isinstance(text_rope, list):
            rope_v, rope_a = text_rope[0], text_rope[1]
        else:
            rope_v = rope_a = text_rope
        t_v, t_a = (time[0], time[1]) if isinstance(time, list) else (time, time)

        video_te, video_tm = self._encode_text("video", te_v, pe_v, t_v, rope_v, attn_mask)
        audio_te, audio_tm = self._encode_text("audio", te_a, pe_a, t_a, rope_a, attn_mask)

        if _parallel_size("pp") > 1 and _parallel_size("pp_rank") != 0:
            vis_embed, vis_shape, vis_rope = None, getattr(self, "_pp_vis_shape", None), visual_rope
            aud_embed, aud_rope = None, audio_rope
        else:
            vis_embed, vis_shape, vis_rope = self._embed_visual(
                x_video,
                visual_rope,
                sparse_params,
                apply_fractal=False,
                visual_token_type_ids=visual_token_type_ids,
            )
            self._pp_vis_shape = vis_shape
            aud_embed, aud_rope = self._embed_audio(x_audio, audio_rope)

        vis_embed, aud_embed = self._run_visual_blocks_fused(
            vis_embed,
            aud_embed,
            video_te,
            audio_te,
            video_tm,
            audio_tm,
            vis_rope,
            aud_rope,
            sparse_params,
            attn_mask,
        )
        if self._pp_final is not None:
            final = self._pp_final
            self._pp_final = None
            return final
        return self._project_fused(
            vis_embed,
            aud_embed,
            self._pp_vis_shape,
            video_tm,
            audio_tm,
        )

    def reset_parameters(self) -> None:
        for m in self.modules():
            if m is not self and hasattr(m, "reset_parameters"):
                m.reset_parameters()
        if hasattr(self, "visual_token_type_embeddings"):
            nn.init.zeros_(self.visual_token_type_embeddings.weight)


__all__ = [
    "Kandinsky6Transformer3DModel",
    "Kandinsky6TransformerEncoderBlock",
    "Kandinsky6TransformerDecoderBlock",
    "Kandinsky6FusedTransformerDecoderBlock",
]
