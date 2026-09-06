# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CausalWanModel — 40-layer DiT with causal attention and KV cache.

Key differences from WanTransformer3DModel:
- Causal self-attention (new frames only see history)
- KV cache for streaming inference
- Action/state token support (appended after video tokens)
- Extended RoPE with action/state-specific frequencies
- Inference-only forward with KV cache
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch._dynamo  # noqa: F401  -- _mark_seq_dynamic needs the submodule bound
import torch.nn as nn
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.conv import Conv3dLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.utils import set_weight_attrs

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.models.dreamzero.action_encoder import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)

# AR-Diffusion paged self-attention (in-tree experimental engine). Import at
# module level so the isinstance check + custom-op call trace cleanly inside
# the fullgraph-compiled DiT block (an import inside the traced region would
# graph-break). The model still works without the engine: the payload type is
# only ever constructed by the AR-Diffusion runner.
try:
    from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import (
        ARDiffusionPagedLayerInputs,
        paged_write_attn,
    )
except ImportError:  # pragma: no cover - experimental package always ships in-tree
    ARDiffusionPagedLayerInputs = None
    paged_write_attn = None

# ── RoPE utilities ──────────────────────────────────────────────────


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Sinusoidal positional embedding for timesteps."""
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}.")
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position,
        torch.pow(10000, -torch.arange(half, dtype=position.dtype, device=position.device).div(half)),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def rope_params(max_seq_len: int, dim: int) -> torch.Tensor:
    """Precompute complex-valued RoPE frequencies (polar form).

    The angles are accumulated in float64 -- at the far end of the table they
    reach 1e4 radians, where float32 would resolve them to only ~1e-3 -- and the
    table is handed out as complex64, since cos and sin land in [-1, 1] and the
    rotation they feed ends in bfloat16.

    Returns: complex tensor [max_seq_len, dim // 2]
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}.")
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(10000, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs.to(torch.complex64)


def rope_apply(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x, with ``freqs`` as a real ``(..., 2)`` cos/sin table.

    Two forms of the same rotation, picked by whether inductor is generating code.

    Compiled: inductor cannot generate code for complex operators, so a complex
    multiply here falls back to eager, and being opaque to the scheduler it also
    splits the surrounding qk-norm and layout work into separate fusion groups.
    Written out in real arithmetic the whole segment fuses instead, measured on
    gfx950 as three kernels collapsing to one.

    Eager: the real form is the slower one by about 4x, because each of its
    multiply-adds becomes its own pass over a temporary while the complex
    multiply is a single elementwise kernel.

    Both branches evaluate the same products in the same order and agree bitwise
    once the caller casts back to bfloat16.

    The rotation runs in float32. Only the angles need anything wider, and they
    are already spent by the time ``rope_params`` hands over cos and sin: those
    live in [-1, 1], where float32 carries four more decimal digits than the
    bfloat16 this result is cast back to.
    """
    B, seq_len, n, _ = x.shape
    pairs = x.to(torch.float32).reshape(B, seq_len, n, -1, 2)
    if not torch.compiler.is_compiling():
        rotated = torch.view_as_complex(pairs) * torch.view_as_complex(freqs).unsqueeze(0)
        return torch.view_as_real(rotated).flatten(3)
    x_re, x_im = pairs[..., 0], pairs[..., 1]
    cos, sin = freqs[..., 0].unsqueeze(0), freqs[..., 1].unsqueeze(0)
    return torch.stack((x_re * cos - x_im * sin, x_re * sin + x_im * cos), dim=-1).flatten(3)


def rope_action_apply(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int = 32,
    num_state_per_block: int = 1,
) -> torch.Tensor:
    """RoPE with action/state frequency tables for multi-step sequences."""
    B, seq_len, n, _ = x.shape
    x = torch.view_as_complex(x.to(torch.float32).reshape(B, seq_len, n, -1, 2))
    if action_register_length is not None:
        if num_action_per_block is None:
            raise ValueError("num_action_per_block is required when action_register_length is set.")
        if num_state_per_block is None:
            raise ValueError("num_state_per_block is required when action_register_length is set.")
        chunk_size = action_register_length // (num_action_per_block + num_state_per_block)
        freqs_1d_action = freqs_action[: chunk_size * num_action_per_block].view(
            chunk_size * num_action_per_block, 1, -1
        )
        freqs_1d_state = freqs_state[: chunk_size * num_state_per_block].view(chunk_size * num_state_per_block, 1, -1)
        freqs = torch.cat([freqs, freqs_1d_action, freqs_1d_state], dim=0)
    freqs = freqs.unsqueeze(0)
    x = torch.view_as_real(x * freqs).flatten(3)
    return x


def _mark_seq_dynamic(t: torch.Tensor | None, dim: int) -> None:
    """Tell Dynamo that ``dim`` varies, so one graph serves every query length.

    ``setup_compile()`` passes ``dynamic=False``, which disables *automatic* dynamic-shape
    inference but does not override an explicit hint: verified on torch 2.12.0+xpu, four
    distinct sequence lengths collapse from four compilations to one, with fusion quality
    and kernel timings within ~2% of the static build.

    Best-effort and silent on purpose. Under ``enforce_eager: true`` there is no
    compilation to influence and marking is a harmless no-op, and ``mark_dynamic`` raises
    if the tensor was already marked or is a view whose base cannot carry the hint. A
    RoPE shape hint is never worth failing a request over.
    """
    if t is None:
        return
    try:
        torch._dynamo.mark_dynamic(t, dim)
    except Exception:  # hint only; correctness never depends on it
        pass


def _materialize_block_freqs(freqs: torch.Tensor) -> torch.Tensor:
    """Real ``(..., 2)`` cos/sin table as a freshly allocated tensor, not a view.

    ``view_as_real`` on its own returns a *view*, and both of the reasons this matters
    are about what the compiled block sees, not about the values:

    1. ``torch._dynamo.mark_dynamic`` cannot always place a hint on a view -- it raises
       when the base cannot carry it. The next commit marks this table's sequence dim,
       and on a view that marking silently does nothing, which quietly costs one graph
       per query length. A real allocation always carries the hint.
    2. Dynamo guards on the tensor's dispatch key set, and the two branches of
       :func:`causal_rope_action_freqs` reach ``view_as_real`` with different bases: the
       no-action branch views the table handed down from ``_create_freqs``, while the
       action branch views a ``torch.cat`` performed here. Under ``inference_mode`` those
       do not reliably agree -- a tensor viewed out of a longer-lived buffer keeps
       ``(XPU, ADInplaceOrView, AutogradXPU, AutocastXPU)`` where a freshly allocated one
       carries ``(XPU, AutocastXPU)`` -- and a mismatch makes prefill and diffuse
       guard-incompatible, costing a second graph:

           tensor 'freqs' dispatch key set mismatch. expected
           DispatchKeySet(XPU, BackendSelect, ADInplaceOrView), actual
           DispatchKeySet(XPU, BackendSelect)

       Allocating on both paths makes the branches hand the block the same flavour of
       tensor by construction.

    ``.contiguous()`` is not a substitute: on an already-contiguous view it returns the
    same view and drops nothing. The copy is of a table four orders of magnitude smaller
    than the activations it rotates, once per forward against the 80 rotations it feeds.
    """
    return torch.view_as_real(freqs).clone()


def causal_rope_action_freqs(
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int,
    num_state_per_block: int,
    action_state_index: int,
) -> torch.Tensor:
    """Video RoPE table with the current step's action/state rows appended.

    Single inference step (causal / KV-cache mode). A function of the step index
    only, so all layers and both q and k share one build -- see ``_forward_blocks``,
    which calls this once per forward and passes the result down.

    Returned as a real ``(..., 2)`` cos/sin table: the complex form is only used
    to build it, so no complex tensor reaches the compiled blocks, where inductor
    would fall back to eager. Converting here costs one view of a table that is
    four orders of magnitude smaller than the tensors it rotates.

    Both exits route through :func:`_materialize_block_freqs`, which is what makes the
    table markable and its dispatch key set branch-independent.
    """
    if action_register_length is None:
        return _materialize_block_freqs(freqs)
    expected_length = num_action_per_block + num_state_per_block
    if action_register_length != expected_length:
        raise ValueError(
            f"action_register_length must equal num_action_per_block + num_state_per_block "
            f"({expected_length}), got {action_register_length}."
        )
    freqs_action = freqs_action[
        action_state_index * num_action_per_block : (action_state_index + 1) * num_action_per_block
    ]
    freqs_state = freqs_state[action_state_index * num_state_per_block : (action_state_index + 1) * num_state_per_block]
    freqs_1d = torch.cat([freqs_action, freqs_state], dim=0).view(action_register_length, 1, -1)
    return _materialize_block_freqs(torch.cat([freqs, freqs_1d], dim=0))


# ── Normalization ───────────────────────────────────────────────────


class WanLayerNorm(nn.LayerNorm):
    """LayerNorm wrapper used by DreamZero blocks."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False) -> None:
        super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine)


class DistributedRMSNorm(nn.Module):
    """RMSNorm that computes global RMS across tensor parallel ranks."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    def weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        if param.shape == loaded_weight.shape:
            param.data.copy_(loaded_weight)
            return

        tp_size = get_tensor_model_parallel_world_size()
        if loaded_weight.shape[0] % tp_size != 0:
            raise ValueError(
                f"Cannot shard RMSNorm weight of shape {tuple(loaded_weight.shape)} across tp_size={tp_size}."
            )

        shard_size = loaded_weight.shape[0] // tp_size
        start_idx = get_tensor_model_parallel_rank() * shard_size
        shard = loaded_weight.narrow(0, start_idx, shard_size)
        if param.shape != shard.shape:
            raise ValueError(f"RMSNorm shard shape mismatch: param={tuple(param.shape)}, shard={tuple(shard.shape)}.")
        param.data.copy_(shard)

    def _local_sum_sq(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Per-rank float32 activation, local sum-of-squares, and local width."""
        x_float = x.float()
        local_sum_sq = x_float.pow(2).sum(dim=-1, keepdim=True)
        return x_float, local_sum_sq, x.shape[-1]

    def _scale(
        self,
        x_float: torch.Tensor,
        global_sum_sq: torch.Tensor,
        global_count: int,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the (already reduced) RMS and the per-rank weight shard."""
        mean_sq = global_sum_sq / global_count
        return (x_float * torch.rsqrt(mean_sq + self.eps)).type_as(x) * self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = get_tensor_model_parallel_world_size()
        x_float, local_sum_sq, local_count = self._local_sum_sq(x)

        if tp_size > 1:
            # Use vLLM's collective (custom all-reduce / symmetric-mem fast path
            # for small tensors) instead of raw torch.distributed.all_reduce, and
            # take the return value so the custom-AR path (which may return a new
            # buffer) is handled correctly. No .clone() needed.
            global_sum_sq = tensor_model_parallel_all_reduce(local_sum_sq)
            global_count = local_count * tp_size
        else:
            global_sum_sq = local_sum_sq
            global_count = local_count

        return self._scale(x_float, global_sum_sq, global_count, x)


def fused_qk_rms_norm(
    norm_q: nn.Module,
    norm_k: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply q/k :class:`DistributedRMSNorm` with a SINGLE fused TP all-reduce.

    WHY: self-attention norms q and k every step; on TP each norm issues its own
    tiny all-reduce of the per-token sum-of-squares. For DreamZero's decode-like
    per-step forward these latency-bound micro-collectives dominate, so we pack
    both sum-of-squares into one tensor and reduce once (2 collectives → 1).

    NUMERICALLY IDENTICAL to ``norm_q(q), norm_k(k)``: all-reduce is elementwise,
    so packing along the last dim reduces each slice independently with the same
    fp32 accumulation. Requires q and k to share the same shape (true for
    self-attention — both come from the same hidden states). Falls back to
    independent application when either norm is not a DistributedRMSNorm
    (e.g. nn.Identity when qk_norm=False).
    """
    if not (isinstance(norm_q, DistributedRMSNorm) and isinstance(norm_k, DistributedRMSNorm)):
        return norm_q(q), norm_k(k)

    # The fused path reduces one packed sum-of-squares and reuses q's token width
    # as the RMS count for BOTH q and k, so q and k must share the same shape.
    # (Self-attention guarantees this: q and k are projected from the same x.)
    assert q.shape == k.shape, "fused_qk_rms_norm requires q and k to have the same shape."

    tp_size = get_tensor_model_parallel_world_size()
    q_float, q_sum_sq, count = norm_q._local_sum_sq(q)
    k_float, k_sum_sq, _ = norm_k._local_sum_sq(k)

    if tp_size > 1:
        packed = torch.cat([q_sum_sq, k_sum_sq], dim=-1)
        packed = tensor_model_parallel_all_reduce(packed)
        q_sum_sq, k_sum_sq = packed[..., 0:1], packed[..., 1:2]
        count = count * tp_size

    q_out = norm_q._scale(q_float, q_sum_sq, count, q)
    k_out = norm_k._scale(k_float, k_sum_sq, count, k)
    return q_out, k_out


# ── Projections ─────────────────────────────────────────────────────


class MLPProj(nn.Module):
    """CLIP feature projection for i2v.
    Uses ColumnParallelLinear + RowParallelLinear (Qwen3_VisionMLP pattern).
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.fc1 = ColumnParallelLinear(
            in_dim,
            in_dim,
            bias=True,
            return_bias=False,
        )
        self.act = nn.GELU()
        self.fc2 = RowParallelLinear(
            in_dim,
            out_dim,
            bias=True,
            return_bias=False,
        )
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        x = self.norm1(image_embeds)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.norm2(x)
        return x


# ── Cross-Attention ─────────────────────────────────────────────────
# T2V and I2V cross-attention variants


class WanT2VCrossAttention(nn.Module):
    """Text-to-video cross-attention.
    Uses vllm-omni Attention for FlashAttn backend.
    """

    def __init__(self, dim: int, num_heads: int, window_size=(-1, -1), qk_norm: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        tp_size = get_tensor_model_parallel_world_size()
        if num_heads % tp_size != 0:
            raise ValueError(f"num_heads={num_heads} must be divisible by tp_size={tp_size}.")
        self.tp_num_heads = num_heads // tp_size
        self.tp_inner_dim = self.tp_num_heads * self.head_dim
        self.q = ColumnParallelLinear(dim, dim, bias=True, gather_output=False, return_bias=False)
        self.k = ColumnParallelLinear(dim, dim, bias=True, gather_output=False, return_bias=False)
        self.v = ColumnParallelLinear(dim, dim, bias=True, gather_output=False, return_bias=False)
        self.o = RowParallelLinear(dim, dim, bias=True, input_is_parallel=True, return_bias=False)
        self.norm_q = DistributedRMSNorm(self.tp_inner_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = DistributedRMSNorm(self.tp_inner_dim, eps=eps) if qk_norm else nn.Identity()
        self.attn = Attention(
            self.tp_num_heads,
            self.head_dim,
            causal=False,
            softmax_scale=self.head_dim**-0.5,
            skip_sequence_parallel=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor | None = None,
        crossattn_cache: dict | None = None,
    ) -> torch.Tensor:
        del context_lens
        n, d = self.tp_num_heads, self.head_dim
        q = self.norm_q(self.q(x)).unflatten(2, (n, d))
        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).unflatten(2, (n, d))
                v = self.v(context).unflatten(2, (n, d))
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).unflatten(2, (n, d))
            v = self.v(context).unflatten(2, (n, d))
        x = self.attn(q, k, v)
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(nn.Module):
    """Image-to-video cross-attention (splits first 257 image tokens).
    Uses vllm-omni Attention for FlashAttn backend.
    """

    def __init__(self, dim: int, num_heads: int, window_size=(-1, -1), qk_norm: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        tp_size = get_tensor_model_parallel_world_size()
        if num_heads % tp_size != 0:
            raise ValueError(f"num_heads={num_heads} must be divisible by tp_size={tp_size}.")
        self.tp_num_heads = num_heads // tp_size
        self.tp_inner_dim = self.tp_num_heads * self.head_dim
        self.q = ColumnParallelLinear(dim, dim, bias=True, gather_output=False, return_bias=False)
        self.k = ColumnParallelLinear(dim, dim, bias=True, gather_output=False, return_bias=False)
        self.v = ColumnParallelLinear(dim, dim, bias=True, gather_output=False, return_bias=False)
        self.o = RowParallelLinear(dim, dim, bias=True, input_is_parallel=True, return_bias=False)
        self.norm_q = DistributedRMSNorm(self.tp_inner_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = DistributedRMSNorm(self.tp_inner_dim, eps=eps) if qk_norm else nn.Identity()
        self.k_img = ColumnParallelLinear(dim, dim, bias=True, gather_output=False, return_bias=False)
        self.v_img = ColumnParallelLinear(dim, dim, bias=True, gather_output=False, return_bias=False)
        self.norm_k_img = DistributedRMSNorm(self.tp_inner_dim, eps=eps) if qk_norm else nn.Identity()
        self.attn = Attention(
            self.tp_num_heads,
            self.head_dim,
            causal=False,
            softmax_scale=self.head_dim**-0.5,
            skip_sequence_parallel=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor | None = None,
        crossattn_cache: dict | None = None,
    ) -> torch.Tensor:
        del context_lens
        context_img = context[:, :257]
        context = context[:, 257:]
        n, d = self.tp_num_heads, self.head_dim
        q = self.norm_q(self.q(x)).unflatten(2, (n, d))
        # context (text) and context_img (clip features) are constant within a
        # session, so k/v and k_img/v_img are cached on first call and reused.
        # context_img == img_emb(clip feature) is session-invariant (state.clip_feas
        # is set once at current_start_frame==0 and only cleared on reset, which also
        # rebuilds crossattn_cache). Caching k_img/v_img removes a per-step
        # norm_k_img all-reduce plus the k_img/v_img projection GEMMs in steady state.
        # Only q depends on the per-step x and is always recomputed.
        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).unflatten(2, (n, d))
                v = self.v(context).unflatten(2, (n, d))
                k_img = self.norm_k_img(self.k_img(context_img)).unflatten(2, (n, d))
                v_img = self.v_img(context_img).unflatten(2, (n, d))
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
                crossattn_cache["k_img"] = k_img
                crossattn_cache["v_img"] = v_img
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
                k_img = crossattn_cache["k_img"]
                v_img = crossattn_cache["v_img"]
        else:
            k = self.norm_k(self.k(context)).unflatten(2, (n, d))
            v = self.v(context).unflatten(2, (n, d))
            k_img = self.norm_k_img(self.k_img(context_img)).unflatten(2, (n, d))
            v_img = self.v_img(context_img).unflatten(2, (n, d))
        x = self.attn(q, k, v)
        img_x = self.attn(q, k_img, v_img)
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    "t2v_cross_attn": WanT2VCrossAttention,
    "i2v_cross_attn": WanI2VCrossAttention,
}


# ── Self-Attention with causal masking + KV cache ───────────────────


class CausalWanSelfAttention(nn.Module):
    """Causal self-attention with KV cache + action/state tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        frame_seqlen: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        num_frame_per_block: int = 1,
        qk_norm: bool = True,
        eps: float = 1e-6,
        num_action_per_block: int = 32,
        num_state_per_block: int = 1,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        tp_size = get_tensor_model_parallel_world_size()
        if num_heads % tp_size != 0:
            raise ValueError(f"num_heads={num_heads} must be divisible by tp_size={tp_size}.")
        self.tp_num_heads = num_heads // tp_size
        self.tp_inner_dim = self.tp_num_heads * self.head_dim
        self.local_attn_size = local_attn_size
        self.num_frame_per_block = num_frame_per_block
        self.frame_seqlen = frame_seqlen
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block
        self.max_attention_size = 21 * frame_seqlen if local_attn_size == -1 else local_attn_size * frame_seqlen
        # Fused QKV projection: q/k/v all come from x, so a single column-parallel
        # GEMM replaces three (fewer launches / better GEMM util in the decode-like
        # per-step forward). No GQA here: total_num_kv_heads defaults to num_heads.
        self.qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            bias=True,
        )
        # The forward splits qkv into three equal q/k/v shards, which is only
        # correct without GQA. total_num_kv_heads defaults to total_num_heads
        # here; fail loud if a future checkpoint introduces GQA so the equal
        # split does not silently misalign k/v.
        if self.qkv.total_num_kv_heads != self.qkv.total_num_heads:
            raise ValueError("Self-attn QKV fusion requires no GQA (total_num_kv_heads == total_num_heads).")
        self.o = RowParallelLinear(dim, dim, bias=True, input_is_parallel=True, return_bias=False)
        self.norm_q = DistributedRMSNorm(self.tp_inner_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = DistributedRMSNorm(self.tp_inner_dim, eps=eps) if qk_norm else nn.Identity()
        self.attn = Attention(
            self.tp_num_heads,
            self.head_dim,
            causal=False,
            softmax_scale=self.head_dim**-0.5,
            skip_sequence_parallel=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        action_register_length: int | None,
        kv_cache: torch.Tensor | Any | None = None,
        is_tf: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Inference-only forward (KV cache path).

        ``freqs`` is the real cos/sin table already carrying this step's
        action/state rows (built once per forward by ``causal_rope_action_freqs``).
        """
        n, d = self.tp_num_heads, self.head_dim

        # Single fused QKV GEMM, then split into the per-rank q/k/v shards.
        qkv, _ = self.qkv(x)
        qk_size = self.tp_num_heads * self.head_dim
        q, k, v = qkv.split([qk_size, qk_size, qk_size], dim=-1)
        # Fuse q/k qk-norm into a single TP all-reduce (both come from x, same shape).
        q, k = fused_qk_rms_norm(self.norm_q, self.norm_k, q, k)
        q = q.unflatten(2, (n, d))
        k = k.unflatten(2, (n, d))
        v = v.unflatten(2, (n, d))

        updated_kv_cache: torch.Tensor | None = None

        if kv_cache is None:
            raise RuntimeError("Inference only: kv_cache is required.")

        roped_query = rope_apply(q, freqs).type_as(v)
        roped_key = rope_apply(k, freqs).type_as(v)

        roped_action_query = None
        roped_action_key = None
        action_v = None

        if action_register_length is not None:
            roped_action_query = roped_query[:, -action_register_length:]
            roped_query = roped_query[:, :-action_register_length]
            roped_action_key = roped_key[:, -action_register_length:]
            roped_key = roped_key[:, :-action_register_length]
            action_v = v[:, -action_register_length:]
            v = v[:, :-action_register_length]

        if ARDiffusionPagedLayerInputs is not None and isinstance(kv_cache, ARDiffusionPagedLayerInputs):
            # Fused write+attend custom op: one opaque node in the compiled
            # graph (slot writes + FlashAttention block-table kernel inside).
            # Metadata tensors were prepared once per forward in _forward_blocks.
            if action_register_length is not None:
                q_cat = torch.cat([roped_query, roped_action_query], dim=1)
                k_act, v_act = roped_action_key[0], action_v[0]
            else:
                q_cat = roped_query
                k_act = v_act = None
            x = paged_write_attn(
                kv_cache,
                q_cat[0],
                roped_key[0],
                v[0],
                k_act,
                v_act,
                self.head_dim**-0.5,
            ).unsqueeze(0)
        else:
            updated_k = kv_cache[0]
            updated_v = kv_cache[1]
            new_k = torch.cat([updated_k, roped_key], dim=1)
            new_v = torch.cat([updated_v, v], dim=1)
            new_k = new_k[:, -self.max_attention_size :]
            new_v = new_v[:, -self.max_attention_size :]

            if action_register_length is not None:
                q_cat = torch.cat([roped_query, roped_action_query], dim=1)
                k_cat = torch.cat([new_k, roped_action_key], dim=1)
                v_cat = torch.cat([new_v, action_v], dim=1)
            else:
                q_cat = roped_query
                k_cat = new_k
                v_cat = new_v

            x = self.attn(q_cat, k_cat, v_cat)
            updated_kv_cache = torch.stack([new_k, new_v], dim=0)

        x = x.flatten(2)
        x = self.o(x)
        return x, updated_kv_cache


# ── Attention Block ─────────────────────────────────────────────────


class CausalWanAttentionBlock(nn.Module):
    """Transformer block: self-attn + cross-attn + FFN with 6-param modulation."""

    def __init__(
        self,
        cross_attn_type: str,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        frame_seqlen: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        num_frame_per_block: int = 1,
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        num_action_per_block: int = 32,
        num_state_per_block: int = 1,
    ) -> None:
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim=dim,
            num_heads=num_heads,
            frame_seqlen=frame_seqlen,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            num_frame_per_block=num_frame_per_block,
            qk_norm=qk_norm,
            eps=eps,
            num_action_per_block=num_action_per_block,
            num_state_per_block=num_state_per_block,
        )
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            ColumnParallelLinear(dim, ffn_dim, bias=True, gather_output=False, return_bias=False),
            nn.GELU(approximate="tanh"),
            RowParallelLinear(ffn_dim, dim, bias=True, input_is_parallel=True, return_bias=False),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        action_register_length: int | None = None,
        kv_cache: torch.Tensor | Any | None = None,
        crossattn_cache: dict | None = None,
        is_tf: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        y, updated_kv_cache = self.self_attn(
            x=(self.norm1(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)),
            freqs=freqs,
            action_register_length=action_register_length,
            kv_cache=kv_cache,
            is_tf=is_tf,
        )
        x = x + (y * e[2].squeeze(2))

        x = x + self.cross_attn(self.norm3(x), context, crossattn_cache=crossattn_cache)
        y = self.ffn(self.norm2(x) * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
        x = x + (y * e[5].squeeze(2))
        return x, updated_kv_cache


# ── Output Head ─────────────────────────────────────────────────────


class CausalHead(nn.Module):
    """Output norm + linear with 2-param modulation.
    Runs once per step (not TP-critical), uses nn.Linear.
    """

    def __init__(self, dim: int, out_dim: int, patch_size: tuple, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        out_channels = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_channels)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L1, C]
            e: [B, F, 1, C]     (time embedding, unsqueezed)
        """
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2))
        return x


# ── Main Model ──────────────────────────────────────────────────────


class CausalWanModel(nn.Module):
    """Causal video diffusion transformer for DreamZero.

    Architecture (14B): 40 layers, dim=5120, heads=40, ffn=13824
    """

    _layerwise_offload_blocks_attrs = ["blocks"]

    def __init__(
        self,
        model_type: str = "t2v",
        patch_size: tuple[int, int, int] = (1, 2, 2),
        frame_seqlen: int = 220,
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 2048,
        ffn_dim: int = 8192,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 16,
        num_layers: int = 32,
        max_chunk_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        num_frame_per_block: int = 1,
        action_dim: int = 32,
        num_registers: int = 8,
        max_state_dim: int = 64,
        max_num_embodiments: int = 32,
        hidden_size: int = 1024,
        diffusion_model_pretrained_path: str | None = None,
        num_action_per_block: int = 32,
        num_state_per_block: int = 1,
    ) -> None:
        super().__init__()
        if model_type not in ["t2v", "i2v", "ti2v"]:
            raise ValueError(f"Unsupported model_type={model_type!r}; expected one of ['t2v', 'i2v', 'ti2v'].")
        self.model_type = model_type
        self.patch_size = patch_size
        self.frame_seqlen = frame_seqlen
        self.text_len = text_len
        self.dim = dim
        self.freq_dim = freq_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = max_chunk_size * num_frame_per_block + 1 if max_chunk_size != -1 else -1
        self.num_frame_per_block = num_frame_per_block
        self.action_dim = action_dim
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block

        max_num_embodiments_local = 1
        self.state_encoder = CategorySpecificMLP(
            num_categories=max_num_embodiments_local,
            input_dim=max_state_dim,
            hidden_dim=hidden_size,
            output_dim=dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=action_dim,
            hidden_size=dim,
            num_embodiments=max_num_embodiments_local,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=max_num_embodiments_local,
            input_dim=dim,
            hidden_dim=hidden_size,
            output_dim=action_dim,
        )

        # Disable the Conv3d GEMM rewrite for patch embedding.
        self.patch_embedding = Conv3dLayer(
            in_dim,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.patch_embedding.enable_linear = False
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )

        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                CausalWanAttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    frame_seqlen,
                    self.local_attn_size,
                    sink_size,
                    num_frame_per_block,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    num_action_per_block,
                    num_state_per_block,
                )
                for _ in range(num_layers)
            ]
        )

        self.head = CausalHead(dim, out_dim, patch_size, eps)

        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        if (dim // num_heads) % 2 != 0:
            raise ValueError(f"dim // num_heads must be even, got {dim // num_heads}.")
        d = dim // num_heads
        self.freqs_action = rope_params(1024 * 10, d)
        self.freqs_state = rope_params(1024, d)
        self.freqs = [
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ]

        if model_type == "i2v":
            self.img_emb = MLPProj(1280, dim)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize parameters."""

        def _init_linear_like(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                return

            if isinstance(module, (ColumnParallelLinear, RowParallelLinear)):
                fan_in = module.input_size
                fan_out = module.output_size
                bound = math.sqrt(6.0 / float(fan_in + fan_out))
                nn.init.uniform_(module.weight, -bound, bound)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for module in self.modules():
            _init_linear_like(module)

        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        if self.patch_embedding.bias is not None:
            fan_in = self.patch_embedding.in_channels * math.prod(self.patch_embedding.kernel_size)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.patch_embedding.bias, -bound, bound)

        for module in self.text_embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)

        for module in self.time_embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)

        nn.init.zeros_(self.head.head.weight)

    def _create_freqs(self, grid_size: torch.Tensor, start_frame: int) -> torch.Tensor:
        """Create 3D RoPE frequency tensor."""
        device = self.patch_embedding.weight.device
        if any(freq.device != device for freq in self.freqs):
            self.freqs = [freq.to(device) for freq in self.freqs]
        if self.freqs_action.device != device:
            self.freqs_action = self.freqs_action.to(device)
        if self.freqs_state.device != device:
            self.freqs_state = self.freqs_state.to(device)

        f, h, w = grid_size.tolist()
        freqs = torch.cat(
            [
                self.freqs[0][start_frame : start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1)
        return freqs

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor) -> torch.Tensor:
        """Reconstruct video from patch embeddings."""
        B = x.shape[0]
        c = self.out_dim
        grid_size = grid_size.tolist()
        expected_seq_len = math.prod(grid_size)
        if x.shape[1] != expected_seq_len:
            raise ValueError(f"x sequence length must equal product(grid_size)={expected_seq_len}, got {x.shape[1]}.")
        x = x.view(B, *grid_size, *self.patch_size, c)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        x = x.reshape(B, c, *[i * j for i, j in zip(grid_size, self.patch_size)])
        return x

    def _forward_blocks(
        self,
        x: torch.Tensor,
        seq_len: int,
        freqs: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: torch.Tensor | None,
        embodiment_id: torch.Tensor | None,
        action: torch.Tensor | None,
        timestep_action: torch.Tensor | None,
        state: torch.Tensor | None,
        kv_cache: list[Any],
        current_start_frame: int,
        crossattn_cache: list[dict] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor | None]]:
        x = x.flatten(start_dim=2).transpose(1, 2)
        B = x.shape[0]
        F_t = timestep.shape[1]

        if action is not None:
            # Current DreamZero checkpoints have one local action/state adapter.
            # Global embodiment IDs are used by transforms and normalization.
            adapter_category_id = torch.zeros(B, dtype=torch.long, device=x.device)
            action_features = self.action_encoder(action, timestep_action, adapter_category_id)
            state_features = self.state_encoder(state, adapter_category_id)
            action_register = torch.cat([action_features, state_features], dim=1)
            action_length = action_features.shape[1]
            action_register_length = action_register.shape[1]
            x = torch.cat([x, action_register], dim=1)
        else:
            action_length = 0
            action_register_length = None

        timestep = timestep.unsqueeze(-1).expand(B, F_t, seq_len // F_t).reshape(B, -1)
        if action is not None:
            if timestep_action is None or state is None:
                raise RuntimeError("timestep_action and state are required when action is provided.")
            state_features_t = self.state_encoder(state, adapter_category_id)
            stride = timestep_action.shape[1] // state_features_t.shape[1]
            timestep_state = timestep_action[:, ::stride]
            timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)

        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x))
        e = e.unflatten(dim=0, sizes=(B, -1))
        e0 = self.time_projection(e)
        e0 = e0.unflatten(dim=2, sizes=(6, self.dim))

        context = self.text_embedding(context)
        if clip_feature is not None:
            clip_embedding = self.img_emb(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)

        # AR-Diffusion paged path: host-side prep ONCE per branch forward, outside
        # the compiled blocks — allocate current video/action slots, build the
        # padded block-table metadata all 40 layers share, and hand the compiled
        # region plain tensors (ARDiffusionPagedLayerInputs) instead of a Python
        # context object. Lazy-allocation contract preserved: only the branch this
        # CFG-parallel rank executes reaches its _forward_blocks.
        if kv_cache and getattr(kv_cache[0], "is_ar_diffusion_paged_context", False):
            if B != 1:
                raise RuntimeError("AR-Diffusion paged self-attention currently expects batch_size=1")
            fctx = kv_cache[0].forward_ctx
            if seq_len != fctx.seq_len:
                raise RuntimeError(
                    f"AR-Diffusion paged context seq_len={fctx.seq_len} but current video KV has {seq_len} tokens"
                )
            fctx.prepare(
                device=x.device,
                action_len=int(action_register_length or 0),
                query_len=int(x.shape[1]),
            )
            kv_cache = [c.to_layer_inputs() for c in kv_cache]

        # Append this step's action/state RoPE rows once, outside the compiled blocks:
        # the table is the same for all 40 layers and for both q and k.
        freqs = causal_rope_action_freqs(
            freqs,
            self.freqs_action,
            self.freqs_state,
            action_register_length,
            self.num_action_per_block,
            self.num_state_per_block,
            max(0, (current_start_frame - 1) // self.num_frame_per_block),
        )

        # One graph for every AR geometry. x, e0 and freqs all carry the query-length
        # dim, which is 880 (first prefill chunk), 1760 (later prefill) or 1785 (diffuse,
        # +25 action/state registers). Unmarked, the block is retraced per length --
        # "tensor 'e' size mismatch at index 1" -- so the table hoist above removes the
        # redundant work but not the recompiles. These three hints are what collapse the
        # lengths, and they need both of the preceding commits to bite: the table must be
        # a real allocation to carry a hint at all, and paged_write_attn must stop passing
        # max_query_len as a Python int or the graph re-specializes on that instead.
        _mark_seq_dynamic(x, 1)
        _mark_seq_dynamic(e0, 1)
        _mark_seq_dynamic(freqs, 0)

        updated_kv_caches: list[torch.Tensor | None] = []
        for block_index, block in enumerate(self.blocks):
            x, updated_kv_cache = block(
                x=x,
                e=e0,
                freqs=freqs,
                context=context,
                action_register_length=action_register_length,
                kv_cache=kv_cache[block_index] if kv_cache else None,
                crossattn_cache=crossattn_cache[block_index] if crossattn_cache else None,
            )
            updated_kv_caches.append(updated_kv_cache)

        if action is not None:
            action_noise_pred = x[:, seq_len : seq_len + action_length]
            action_noise_pred = self.action_decoder(action_noise_pred, adapter_category_id)
        else:
            action_noise_pred = None

        x_video = x[:, :seq_len]
        e_video = e[:, :seq_len]
        x_video = self.head(x_video, e_video.unsqueeze(2))

        return x_video, action_noise_pred, updated_kv_caches

    def _forward_inference(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        seq_len: int,
        kv_cache: list[Any],
        current_start_frame: int,
        crossattn_cache: list[dict] | None = None,
        y: torch.Tensor | None = None,
        clip_feature: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        timestep_action: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        embodiment_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        if self.model_type == "i2v":
            if clip_feature is None or y is None:
                raise RuntimeError("clip_feature and y are required for i2v inference.")
        if context.shape[1] != self.text_len:
            raise ValueError(f"context length must be {self.text_len}, got {context.shape[1]}.")

        if y is not None:
            x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)

        x = self.patch_embedding(x)
        grid_size = torch.tensor(x.shape[2:], dtype=torch.long)
        freqs = self._create_freqs(grid_size, current_start_frame)

        x_video, action_noise_pred, updated_kv_caches = self._forward_blocks(
            x=x,
            seq_len=seq_len,
            freqs=freqs,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            embodiment_id=embodiment_id,
            action=action,
            timestep_action=timestep_action,
            state=state,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start_frame=current_start_frame,
        )

        x_video = x_video.clone()
        if action_noise_pred is not None:
            action_noise_pred = action_noise_pred.clone()

        video_noise_pred = self.unpatchify(x_video, grid_size)
        return video_noise_pred, action_noise_pred, updated_kv_caches

    def forward(self, *args: Any, **kwargs: Any):
        """Inference only. Requires kv_cache."""
        return self._forward_inference(*args, **kwargs)
