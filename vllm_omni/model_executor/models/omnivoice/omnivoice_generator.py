# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OmniVoice Generator (Stage 0) - Iterative unmasking with Qwen3 backbone.

Generates 8-codebook audio tokens from text via 32-step non-autoregressive
iterative masked prediction with classifier-free guidance.

Uses full bidirectional attention computed directly with PyTorch SDPA
(torch.nn.functional.scaled_dot_product_attention); no auto-selected
FlashAttention/SageAttention/DiffusionAttention backend is used.
"""

from __future__ import annotations

import math
import random
import threading
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.model_executor.models.omnivoice.fused_qkv_rope import fused_qkv_norm_rope
from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Triton kernels (inference-only; graceful fallback when triton is absent)
# ---------------------------------------------------------------------------

_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl

    try:
        from triton.language.extra.libdevice import rsqrt  # noqa: F401
    except ModuleNotFoundError:
        from triton.language.extra.cuda.libdevice import rsqrt  # noqa: F401

    def _calculate_settings(n: int) -> tuple[int, int]:
        MAX_FUSED_SIZE = 65536
        BLOCK_SIZE = triton.next_power_of_2(n)
        if BLOCK_SIZE > MAX_FUSED_SIZE:
            raise RuntimeError(f"n={n} exceeds max Triton block size {MAX_FUSED_SIZE}")
        num_warps = 4
        if BLOCK_SIZE >= 32768:
            num_warps = 32
        elif BLOCK_SIZE >= 8192:
            num_warps = 16
        elif BLOCK_SIZE >= 2048:
            num_warps = 8
        return BLOCK_SIZE, num_warps

    @triton.jit
    def _rms_norm_fwd_kernel(
        Y_ptr,  # noqa: N803
        Y_stride,  # noqa: N803
        X_ptr,  # noqa: N803
        X_stride,  # noqa: N803
        W_ptr,  # noqa: N803
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,  # noqa: N803
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(X_ptr + row * X_stride + cols, mask=mask, other=0.0)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        ms = tl.sum(x_f32 * x_f32, axis=0) / n_cols
        rstd = rsqrt(ms + eps)
        y = (x_f32 * rstd).to(x.dtype) * w
        tl.store(Y_ptr + row * Y_stride + cols, y, mask=mask)

    def triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        shape = x.shape
        n_cols = shape[-1]
        x2d = x.contiguous().view(-1, n_cols)
        w = weight.contiguous()
        BLOCK_SIZE, num_warps = _calculate_settings(n_cols)
        y = torch.empty_like(x2d)
        _rms_norm_fwd_kernel[(x2d.shape[0],)](
            y,
            y.stride(0),
            x2d,
            x2d.stride(0),
            w,
            n_cols,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return y.view(*shape)

    @triton.jit
    def _swiglu_fwd_kernel(
        inp_ptr,
        out_ptr,
        in_stride,
        out_stride,
        n_cols: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,  # noqa: N803
    ):
        pid = tl.program_id(0).to(tl.int64)
        inp_ptr += pid * in_stride
        out_ptr += pid * out_stride
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        gate = tl.load(inp_ptr + cols, mask=mask, other=0).to(tl.float32)
        up = tl.load(inp_ptr + n_cols + cols, mask=mask, other=0)
        silu_gate = gate * tl.sigmoid(gate)
        out = silu_gate.cast(up.dtype) * up
        tl.store(out_ptr + cols, out, mask=mask)

    def triton_swiglu(gate_up: torch.Tensor) -> torch.Tensor:
        """SwiGLU over a packed ``[..., 2 * intermediate]`` activation.

        Reading both halves straight out of the fused projection's output keeps
        the packing free: splitting it first would hand this kernel two strided
        views and cost a full copy of each half per layer per step.
        """
        n_cols = gate_up.shape[-1] // 2
        gate_up = gate_up.contiguous()
        x2d = gate_up.view(-1, 2 * n_cols)
        out = torch.empty(x2d.shape[0], n_cols, dtype=gate_up.dtype, device=gate_up.device)
        BLOCK_SIZE, num_warps = _calculate_settings(n_cols)
        _swiglu_fwd_kernel[(x2d.shape[0],)](
            x2d,
            out,
            x2d.stride(0),
            out.stride(0),
            n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return out.view(*gate_up.shape[:-1], n_cols)

    @triton.jit
    def _fused_add_rms_norm_fwd_kernel(
        Y_ptr,  # noqa: N803
        Y_stride,  # noqa: N803
        S_ptr,  # noqa: N803
        S_stride,  # noqa: N803
        X_ptr,  # noqa: N803
        X_stride,  # noqa: N803
        R_ptr,  # noqa: N803
        R_stride,  # noqa: N803
        W_ptr,  # noqa: N803
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,  # noqa: N803
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(X_ptr + row * X_stride + cols, mask=mask, other=0.0)
        r = tl.load(R_ptr + row * R_stride + cols, mask=mask, other=0.0)
        dtype = x.dtype
        s_f32 = x.to(tl.float32) + r.to(tl.float32)
        s = s_f32.to(dtype)
        tl.store(S_ptr + row * S_stride + cols, s, mask=mask)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0)
        ms = tl.sum(s_f32 * s_f32, axis=0) / n_cols
        rstd = rsqrt(ms + eps)
        y = (s_f32 * rstd).to(dtype) * w
        tl.store(Y_ptr + row * Y_stride + cols, y, mask=mask)

    def triton_fused_add_rms_norm(
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape
        n_cols = shape[-1]
        x2d = x.contiguous().view(-1, n_cols)
        r2d = residual.contiguous().view(-1, n_cols)
        w = weight.contiguous()
        BLOCK_SIZE, num_warps = _calculate_settings(n_cols)
        y = torch.empty_like(x2d)
        s = torch.empty_like(x2d)
        _fused_add_rms_norm_fwd_kernel[(x2d.shape[0],)](
            y,
            y.stride(0),
            s,
            s.stride(0),
            x2d,
            x2d.stride(0),
            r2d,
            r2d.stride(0),
            w,
            n_cols,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return y.view(*shape), s.view(*shape)

    _TRITON_AVAILABLE = True
    logger.debug("OmniVoice Triton kernels loaded")
except Exception:
    logger.debug("Triton not available; using PyTorch fallback for OmniVoice kernels")


# ---------------------------------------------------------------------------
# Unmasking schedule helpers
# ---------------------------------------------------------------------------


def _get_time_steps(
    t_start: float,
    t_end: float,
    num_step: int,
    t_shift: float,
) -> torch.Tensor:
    """Compute the unmasking schedule with time shift.

    Returns cumulative proportions [0, ..., 1] of length num_step.
    Formula: r_n = t_shift * (n/N) / (1 + (t_shift - 1) * (n/N))
    """
    steps = torch.linspace(t_start, t_end, num_step)
    shifted = t_shift * steps / (1.0 + (t_shift - 1.0) * steps)
    return shifted


def _gumbel_sample(logits: torch.Tensor, temperature: float, generator: torch.Generator) -> torch.Tensor:
    """Add Gumbel noise for stochastic position selection."""
    noise = -torch.log(
        -torch.log(
            torch.rand(logits.shape, generator=generator, device=logits.device, dtype=logits.dtype).clamp(min=1e-8)
        )
    )
    return logits / max(temperature, 1e-8) + noise


# ---------------------------------------------------------------------------
# Qwen3-style transformer blocks using PyTorch SDPA
# ---------------------------------------------------------------------------


# Subclass keeps .weight name + ctor shape so the state_dict loader stays unchanged.
class OmniVoiceRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _TRITON_AVAILABLE:
            return triton_rms_norm(x, self.weight, self.eps)
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(self.weight.dtype)


class OmniVoiceAttention(nn.Module):
    """Qwen3-style GQA attention using PyTorch SDPA (full bidirectional)."""

    def __init__(self, config: OmniVoiceConfig):
        super().__init__()
        self.hidden_size = config.llm_hidden_size
        self.num_heads = config.llm_num_attention_heads
        self.num_kv_heads = config.llm_num_key_value_heads
        self.head_dim = config.llm_head_dim

        # q/k/v are packed into one projection: the three are sibling GEMMs over
        # the same activation, and at this model's shapes (2 x 44 rows) three
        # small GEMMs cost noticeably more than one wide one.
        self.num_qkv_heads = self.num_heads + 2 * self.num_kv_heads
        self.qkv_proj = nn.Linear(self.hidden_size, self.num_qkv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Qwen3 uses per-head QK norm
        self.q_norm = OmniVoiceRMSNorm(self.head_dim)
        self.k_norm = OmniVoiceRMSNorm(self.head_dim)

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_table: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states).view(batch_size, seq_len, self.num_qkv_heads, self.head_dim)

        # One kernel for the whole prologue: split the packed projection, RMSNorm
        # Q and K per head, rotate both, broadcast K and V across their query
        # groups, and emit SDPA's [batch, heads, positions, head_dim] layout.
        q, k, v = fused_qkv_norm_rope(
            qkv,
            self.q_norm.weight,
            self.k_norm.weight,
            rope_table,
            self.q_norm.eps,
            self.num_heads,
            self.num_kv_heads,
        )

        # Caller passes a float mask; materialize float form if a bool slips through.
        sdpa_mask = attention_mask
        if sdpa_mask is not None and sdpa_mask.dtype == torch.bool:
            sdpa_mask = torch.zeros_like(attention_mask, dtype=q.dtype).masked_fill_(~attention_mask, float("-inf"))

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
            scale=self.scale,
        )

        # Back to (batch, seq, heads * head_dim)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(out)


class OmniVoiceMLP(nn.Module):
    """Qwen3-style MLP with SwiGLU."""

    def __init__(self, config: OmniVoiceConfig):
        super().__init__()
        self.intermediate_size = config.llm_intermediate_size
        self.gate_up_proj = nn.Linear(config.llm_hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, config.llm_hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        if _TRITON_AVAILABLE:
            return self.down_proj(triton_swiglu(gate_up))
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


# Fused parameter -> the HF checkpoint shards it absorbs, in packing order.
# The checkpoint keeps q/k/v and gate/up separate; load_weights packs them.
_FUSED_PROJECTIONS: dict[str, tuple[str, ...]] = {
    "self_attn.qkv_proj": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "mlp.gate_up_proj": ("mlp.gate_proj", "mlp.up_proj"),
}


class OmniVoiceTransformerBlock(nn.Module):
    """Single Qwen3 transformer block with PyTorch SDPA attention."""

    def __init__(self, config: OmniVoiceConfig):
        super().__init__()
        self.input_layernorm = OmniVoiceRMSNorm(config.llm_hidden_size, eps=config.llm_rms_norm_eps)
        self.self_attn = OmniVoiceAttention(config)
        self.post_attention_layernorm = OmniVoiceRMSNorm(config.llm_hidden_size, eps=config.llm_rms_norm_eps)
        self.mlp = OmniVoiceMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        rope_table: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(normed_hidden_states, residual)``.

        The residual is threaded across the block boundary rather than being
        added at the end. The MLP's residual add and the next block's input
        RMSNorm are the same read of the same tensor, so handing the pending
        residual on lets both happen in one fused kernel instead of a bare add
        followed by a separate norm. Only the first block, which has no pending
        residual, still pays for a standalone norm.
        """
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        elif _TRITON_AVAILABLE:
            hidden_states, residual = triton_fused_add_rms_norm(
                hidden_states,
                residual,
                self.input_layernorm.weight,
                self.input_layernorm.eps,
            )
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(hidden_states, rope_table, attention_mask=attention_mask)

        if _TRITON_AVAILABLE:
            # Fused: (attn_out + residual) + RMSNorm in one kernel
            hidden_states, residual = triton_fused_add_rms_norm(
                hidden_states,
                residual,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.eps,
            )
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)

        return self.mlp(hidden_states), residual


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def _precompute_rope_table(
    head_dim: int,
    max_seq_len: int,
    theta: float = 1000000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Precompute the packed ``[max_seq_len, head_dim]`` RoPE table.

    Layout is what ``fused_qkv_norm_rope`` expects: the first half of each row
    holds ``cos(theta)`` and the second half ``sin(theta)``, each of width
    ``head_dim // 2``. Precomputing it keeps the hot path free of any cat.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


# ---------------------------------------------------------------------------
# TF32 opt-in (process-wide; default off)
# ---------------------------------------------------------------------------

_TF32_ENABLED = False


def _maybe_enable_tf32() -> None:
    """Enable TF32 matmuls process-wide (idempotent). Not bit-identical; opt-in via config.enable_tf32."""
    global _TF32_ENABLED
    if _TF32_ENABLED or not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    _TF32_ENABLED = True
    logger.info(
        "OmniVoice TF32 enabled process-wide: matmul.allow_tf32=%s cudnn.allow_tf32=%s float32_matmul_precision=%s",
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    )


# ---------------------------------------------------------------------------
# CUDA Graph wrapper
# ---------------------------------------------------------------------------


def _additive_float_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert a boolean attention mask to its additive float form.

    ``True`` (attend) maps to ``0.0`` and ``False`` (masked) to ``-inf``. A bool
    mask must never be copied straight into a float buffer: the implicit cast
    maps True/False to 1.0/0.0, which leaves masked positions at 0.0 and so
    silently *unmasks* them.

    ``dtype`` is required rather than defaulting to float32: SDPA rejects an
    additive mask whose dtype differs from the query, so the mask has to follow
    the model dtype and a default would just hide that coupling.
    """
    if mask.dtype != torch.bool:
        return mask
    return torch.zeros_like(mask, dtype=dtype).masked_fill_(~mask, float("-inf"))


class _OmniVoiceCUDAGraphForward:
    """Pre-captures CUDA graphs for predefined sequence-length buckets.

    Memory layout: all graphs share a single per-instance pool handle, which
    isolates OmniVoice CUDA memory from other vllm modules.  Sequential replay
    (one step at a time) means pool sharing is safe.
    """

    # Default bucket count is 10; 16 gives modest headroom for edge cases
    # (seq_len > max bucket or non-CFG batch) without unbounded GPU growth.
    _MAX_LAZY_GRAPHS: int = 16

    def __init__(self, generator: OmniVoiceGenerator, capture_sizes: list[int]) -> None:
        self._gen = generator
        self._capture_sizes = sorted(capture_sizes)
        # Pre-warmed graphs keyed by (two_b, bucket); fixed set, never evicted.
        self._graphs: dict[tuple[int, int], dict] = {}
        # Lazy-captured graphs for oversized / non-CFG shapes; capped via LRU.
        self._lazy_graphs: OrderedDict[tuple[int, int], dict] = OrderedDict()
        self._lock = threading.Lock()
        # Per-instance pool handle: isolates OmniVoice CUDA memory from other
        # vllm modules while still allowing safe re-use across sequential replays.
        self._pool_handle: int | None = None

    def _find_bucket(self, seq_len: int) -> int | None:
        for bucket in self._capture_sizes:
            if bucket >= seq_len:
                return bucket
        return None

    def _pad_inputs(
        self,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor | None,
        bucket: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        S = input_ids.shape[-1]
        if S == bucket:
            return input_ids, audio_mask, attention_mask

        two_b = input_ids.shape[0]
        num_cb = input_ids.shape[1]

        ids_padded = torch.zeros(two_b, num_cb, bucket, dtype=input_ids.dtype, device=input_ids.device)
        ids_padded[:, :, :S] = input_ids

        mask_padded = torch.zeros(two_b, bucket, dtype=torch.bool, device=audio_mask.device)
        mask_padded[:, :S] = audio_mask

        if attention_mask is not None:
            # Callers normalize to the additive float form first, so pad with -inf.
            attn_padded = torch.full(
                (two_b, 1, bucket, bucket),
                float("-inf"),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attn_padded[:, :, :S, :S] = attention_mask
        else:
            attn_padded = None

        return ids_padded, mask_padded, attn_padded

    def _capture_for_key(
        self,
        key: tuple[int, int],
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> dict:
        _, bucket = key
        device = input_ids.device

        static_rope_table = self._gen._rope_table_for(bucket, device, self._gen.model_dtype)

        static_input_ids = input_ids.clone()
        static_audio_mask = audio_mask.clone()
        static_attn_mask = attention_mask.clone() if attention_mask is not None else None

        with torch.no_grad():
            _ = self._gen._step_forward(
                static_input_ids,
                static_audio_mask,
                static_attn_mask,
                static_rope_table,
            )
        torch.accelerator.synchronize(device)

        # Lazy-init per-instance pool handle: isolates OmniVoice CUDA Graph
        # memory from other vllm modules (unlike get_global_graph_pool which
        # shares a single pool across all captured graphs and can cause memory
        # aliasing when two graphs replay concurrently).
        if self._pool_handle is None:
            self._pool_handle = torch.cuda.graph_pool_handle()

        graph = torch.cuda.CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph, pool=self._pool_handle):
                static_output = self._gen._step_forward(
                    static_input_ids,
                    static_audio_mask,
                    static_attn_mask,
                    static_rope_table,
                )

        entry = {
            "graph": graph,
            "static_input_ids": static_input_ids,
            "static_audio_mask": static_audio_mask,
            "static_attn_mask": static_attn_mask,
            "static_rope_table": static_rope_table,
            "static_output": static_output,
        }
        logger.info("OmniVoice CUDA Graph captured for key %s", key)
        return entry

    def warmup(self, device: torch.device) -> None:
        """Pre-capture graphs for all bucket sizes with B=1 (two_b=2 for CFG)."""
        if not torch.cuda.is_available():
            return
        logger.info(
            "OmniVoice CUDA Graph warmup: capturing %d bucket sizes %s",
            len(self._capture_sizes),
            self._capture_sizes,
        )
        two_b = 2
        num_cb = self._gen.config.num_audio_codebook
        for bucket in self._capture_sizes:
            key = (two_b, bucket)
            dummy_ids = torch.zeros(two_b, num_cb, bucket, dtype=torch.long, device=device)
            dummy_mask = torch.zeros(two_b, bucket, dtype=torch.bool, device=device)
            # Capture with a float mask to match what forward() feeds at replay time,
            # in the model dtype so replay can copy_ into it without a cast.
            dummy_attn = torch.zeros(two_b, 1, bucket, bucket, dtype=self._gen.model_dtype, device=device)
            self._graphs[key] = self._capture_for_key(key, dummy_ids, dummy_mask, dummy_attn)
        logger.info("OmniVoice CUDA Graph warmup complete (%d graphs)", len(self._graphs))

    def __call__(
        self,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if torch.cuda.is_current_stream_capturing():
            rope_table = self._gen._rope_table_for(input_ids.shape[-1], input_ids.device, self._gen.model_dtype)
            return self._gen._step_forward(input_ids, audio_mask, attention_mask, rope_table)

        seq_len = input_ids.shape[-1]
        two_b = input_ids.shape[0]
        bucket = self._find_bucket(seq_len) if two_b == 2 else None

        # Graphs are captured with (and their static buffers hold) the additive
        # float mask, so normalize here, before padding or any copy_ into them.
        if attention_mask is not None:
            attention_mask = _additive_float_mask(attention_mask, self._gen.model_dtype)

        if bucket is None:
            # Lazy capture: oversized sequence or non-unit batch (no pre-warmed bucket).
            # Lock prevents concurrent threads from double-capturing the same key.
            # _lazy_graphs is capped at _MAX_LAZY_GRAPHS with LRU eviction to
            # prevent unbounded GPU memory growth when seq_len varies widely.
            key = (two_b, seq_len)
            ids_in, mask_in, attn_in = input_ids, audio_mask, attention_mask
            with self._lock:
                entry = self._lazy_graphs.get(key)
                if entry is None:
                    entry = self._capture_for_key(key, ids_in, mask_in, attn_in)
                    if len(self._lazy_graphs) >= self._MAX_LAZY_GRAPHS:
                        evicted_key, _ = self._lazy_graphs.popitem(last=False)
                        logger.warning("OmniVoice CUDA Graph lazy cache full; evicted key %s", evicted_key)
                    self._lazy_graphs[key] = entry
        else:
            key = (two_b, bucket)
            ids_in, mask_in, attn_in = self._pad_inputs(input_ids, audio_mask, attention_mask, bucket)
            with self._lock:
                entry = self._graphs.get(key)
                if entry is None:
                    entry = self._capture_for_key(key, ids_in, mask_in, attn_in)
                    self._graphs[key] = entry

        entry["static_input_ids"].copy_(ids_in)
        entry["static_audio_mask"].copy_(mask_in)
        if attn_in is not None and entry["static_attn_mask"] is not None:
            entry["static_attn_mask"].copy_(attn_in)

        entry["graph"].replay()

        output = entry["static_output"]
        if bucket is not None and bucket != seq_len:
            output = output[:, :, :seq_len, :]
        return output

    def clear(self) -> None:
        with self._lock:
            self._graphs.clear()
            self._lazy_graphs.clear()


# ---------------------------------------------------------------------------
# Generator model
# ---------------------------------------------------------------------------


class OmniVoiceGenerator(nn.Module):
    """OmniVoice Stage 0: Iterative unmasking generator.

    Architecture:
    - Text embedding (from Qwen3 vocab) + Audio embedding (8*1025 entries)
    - 28-layer Qwen3 transformer with full bidirectional attention
    - 8-codebook prediction head (single linear: hidden → 8*1025)
    - 32-step iterative unmasking with classifier-free guidance

    Optimizations:
    - Full bidirectional attention via PyTorch SDPA (no auto-selected
      FlashAttn/SageAttn/DiffusionAttention backend)
    - regionally_compile() compatible for torch.compile on repeated blocks
    """

    # For regionally_compile() support
    _repeated_blocks = ["layers"]

    def __init__(self, config: OmniVoiceConfig):
        super().__init__()
        self.config = config

        # Opt-in TF32; must run before any CUDA-graph capture so captured kernels honour it.
        if getattr(config, "enable_tf32", False):
            _maybe_enable_tf32()

        # Text embedding (shared with LLM)
        self.text_embedding = nn.Embedding(config.llm_vocab_size, config.llm_hidden_size)

        # Audio embedding: 8 codebooks * 1025 tokens
        self.audio_embeddings = nn.Embedding(
            config.num_audio_codebook * config.audio_vocab_size,
            config.llm_hidden_size,
        )
        self.register_buffer(
            "codebook_layer_offsets",
            torch.arange(config.num_audio_codebook) * config.audio_vocab_size,
        )

        # Transformer layers
        self.layers = nn.ModuleList([OmniVoiceTransformerBlock(config) for _ in range(config.llm_num_hidden_layers)])
        self.norm = OmniVoiceRMSNorm(config.llm_hidden_size, eps=config.llm_rms_norm_eps)

        # Prediction head: hidden → 8 * 1025
        self.audio_heads = nn.Linear(
            config.llm_hidden_size,
            config.num_audio_codebook * config.audio_vocab_size,
            bias=False,
        )

        # Precompute RoPE
        self._rope_table = None

        # CUDA Graph (bucket-size pre-capture; lazy fallback for oversized shapes)
        self._cuda_graph_fwd: _OmniVoiceCUDAGraphForward | None = (
            _OmniVoiceCUDAGraphForward(self, config.cuda_graph_capture_sizes) if config.enable_cuda_graph else None
        )

    @property
    def model_dtype(self) -> torch.dtype:
        """The dtype every activation and mask in the generator has to match."""
        return self.text_embedding.weight.dtype

    def _ensure_rope(self, seq_len: int, device: torch.device) -> None:
        """Lazily compute the packed RoPE table if needed."""
        if self._rope_table is None or self._rope_table.shape[0] < seq_len:
            max_len = max(seq_len, 4096)
            self._rope_table = _precompute_rope_table(
                self.config.llm_head_dim,
                max_len,
                theta=self.config.llm_rope_theta,
                device=device,
            )

    def _rope_table_for(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """The packed [seq_len, head_dim] table the fused prologue indexes.

        It depends only on the bucket and dtype, so callers build it once per
        request or per captured graph, never per layer.
        """
        self._ensure_rope(seq_len, device)
        return self._rope_table[:seq_len].to(device=device, dtype=dtype).contiguous()

    def _prepare_embeddings(
        self,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        text_embeds: torch.Tensor | None = None,
        audio_mask_3d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Prepare mixed text+audio embeddings.

        Args:
            input_ids: [B, 8, S] - text tokens replicated across codebooks,
                       audio positions have per-codebook token IDs
            audio_mask: [B, S] - True for audio positions, False for text
            text_embeds: optional cached [B, S, H] text-position embeddings
            audio_mask_3d: optional cached [B, S, 1] audio_mask.unsqueeze(-1)

        Returns:
            embeddings: [B, S, hidden_size]
        """
        # Cached across the denoising loop since text ids don't change.
        if text_embeds is None:
            text_embeds = self.text_embedding(input_ids[:, 0, :])
        if audio_mask_3d is None:
            audio_mask_3d = audio_mask.unsqueeze(-1)

        # Audio embeddings: offset per codebook, then sum across codebooks
        shifted_ids = (input_ids * audio_mask.unsqueeze(1)) + self.codebook_layer_offsets.view(1, -1, 1)
        audio_embeds = self.audio_embeddings(shifted_ids).sum(dim=1)

        # Merge: audio where audio_mask=True, text elsewhere
        return torch.where(audio_mask_3d, audio_embeds, text_embeds)

    def _transformer_forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        rope_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run through transformer layers.

        Args:
            inputs_embeds: [B, S, hidden_size]
            attention_mask: [B, 1, S, S] or None
            rope_table: optional precomputed [B * S, head_dim] RoPE table

        Returns:
            hidden_states: [B, S, hidden_size]
        """
        hidden_states = inputs_embeds
        if rope_table is None:
            rope_table = self._rope_table_for(inputs_embeds.shape[1], inputs_embeds.device, hidden_states.dtype)

        # Safety: convert bool mask if caller hasn't (e.g. external paths beyond forward()).
        if attention_mask is not None and attention_mask.dtype == torch.bool:
            attention_mask = torch.zeros_like(attention_mask, dtype=hidden_states.dtype).masked_fill_(
                ~attention_mask, float("-inf")
            )

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states,
                attention_mask=attention_mask,
                rope_table=rope_table,
                residual=residual,
            )

        return self.norm(hidden_states + residual)

    def _get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to per-codebook logits.

        Args:
            hidden_states: [B, S, hidden_size]

        Returns:
            logits: [B, 8, S, 1025]
        """
        batch_size, seq_len, _ = hidden_states.shape
        logits_flat = self.audio_heads(hidden_states)  # [B, S, 8*1025]
        return logits_flat.view(
            batch_size,
            seq_len,
            self.config.num_audio_codebook,
            self.config.audio_vocab_size,
        ).permute(0, 2, 1, 3)  # [B, 8, S, 1025]

    def _step_forward(
        self,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor | None,
        rope_table: torch.Tensor,
    ) -> torch.Tensor:
        """Single unmasking-step forward using a pre-cast RoPE table (CUDA graph safe)."""
        hidden_states = self._prepare_embeddings(input_ids, audio_mask)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, attention_mask=attention_mask, rope_table=rope_table, residual=residual
            )
        return self._get_logits(self.norm(hidden_states + residual))

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        target_lens: list[int],
        seed: int | None = None,
        num_step: int = 32,
        guidance_scale: float = 2.0,
        t_shift: float = 0.1,
        layer_penalty_factor: float = 5.0,
        position_temperature: float = 5.0,
        class_temperature: float = 0.0,
    ) -> torch.Tensor:
        """Run the full 32-step iterative unmasking generation.

        Args:
            input_ids: [2*B, 8, S] - conditional (0:B) + unconditional (B:2B)
            audio_mask: [2*B, S] - True for audio positions
            attention_mask: [2*B, 1, S, S] - attention mask
            target_lens: List of target audio lengths per batch item
            num_step: Number of unmasking steps
            guidance_scale: CFG scale
            t_shift: Time shift for schedule
            layer_penalty_factor: Penalty for later codebooks
            position_temperature: Gumbel temperature for position selection
            class_temperature: Temperature for token prediction (0=greedy)

        Returns:
            tokens: [B, 8, max_target_len] - generated audio tokens
        """
        B = len(target_lens)
        device = input_ids.device
        max_target_len = max(target_lens)
        mask_id = self.config.audio_mask_id
        num_codebooks = self.config.num_audio_codebook
        if seed is None:
            seed = random.randint(0, 2**63 - 1)
        generator = torch.Generator(device=device).manual_seed(seed)

        # Initialize all target tokens as [MASK]
        tokens = torch.full(
            (B, num_codebooks, max_target_len),
            mask_id,
            dtype=torch.long,
            device=device,
        )

        # Compute unmasking schedule
        timesteps = _get_time_steps(0.0, 1.0, num_step + 1, t_shift).tolist()
        schedules = []
        for t_len in target_lens:
            total_mask = t_len * num_codebooks
            rem = total_mask
            sched = []
            for step in range(num_step):
                num = (
                    rem
                    if step == num_step - 1
                    else min(
                        math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])),
                        rem,
                    )
                )
                sched.append(int(num))
                rem -= int(num)
            schedules.append(sched)

        layer_ids = torch.arange(num_codebooks, device=device).view(1, -1, 1)

        # Single D2H pull for all conditional lengths instead of B per-item .item() syncs.
        c_lens = attention_mask[:B, 0, 0].sum(dim=-1).tolist()

        # Materialize the SDPA float mask once so the captured graph (and eager path) skip per-layer conversion.
        sdpa_attn_mask = _additive_float_mask(attention_mask, self.model_dtype)

        use_cuda_graph = self._cuda_graph_fwd is not None and input_ids.is_cuda
        if not use_cuda_graph:
            # Eager-path-only constants (the cuda-graph captures its own).
            text_embeds_cached = self.text_embedding(input_ids[:, 0, :])
            audio_mask_3d = audio_mask.unsqueeze(-1)
            rope_table = self._rope_table_for(input_ids.shape[-1], device, text_embeds_cached.dtype)

        # Main iterative loop
        for step in range(num_step):
            if use_cuda_graph:
                # Float mask skips per-layer conversion; fp32 cast deferred to the per-item slices below.
                batch_logits = self._cuda_graph_fwd(input_ids, audio_mask, sdpa_attn_mask)
            else:
                # Eager fallback reuses hoisted constants (text embeds, sdpa mask, rope table).
                inputs_embeds = self._prepare_embeddings(
                    input_ids, audio_mask, text_embeds=text_embeds_cached, audio_mask_3d=audio_mask_3d
                )
                hidden_states = self._transformer_forward(inputs_embeds, sdpa_attn_mask, rope_table=rope_table)
                # fp32 cast deferred to the per-item slices below.
                batch_logits = self._get_logits(hidden_states)
            # batch_logits: [2*B, 8, S, 1025]

            for i in range(B):
                k = schedules[i][step]
                if k <= 0:
                    continue

                c_len = c_lens[i]
                t_len = target_lens[i]

                # Extract logits for target region; upcast only the slices we actually consume.
                c_logits = batch_logits[i : i + 1, :, c_len - t_len : c_len, :].to(torch.float32)
                u_logits = batch_logits[B + i : B + i + 1, :, :t_len, :].to(torch.float32)

                # Classifier-free guidance. Fuse the chain: the two inner
                # log_softmax normalizers are per-position scalars that the final
                # shift-invariant log_softmax cancels, so guide on the raw logits
                # with a single softmax: log_softmax((1+s)*c - s*u). Exact.
                if guidance_scale != 0:
                    log_probs = F.log_softmax(
                        (1.0 + guidance_scale) * c_logits - guidance_scale * u_logits,
                        dim=-1,
                    )
                else:
                    log_probs = F.log_softmax(c_logits, dim=-1)

                # Prevent predicting [MASK]
                log_probs[..., mask_id] = -float("inf")

                # Token prediction
                if class_temperature > 0.0:
                    pred_tokens = _gumbel_sample(log_probs, class_temperature, generator).argmax(dim=-1)
                else:
                    pred_tokens = log_probs.argmax(dim=-1)  # [1, 8, T]

                # Confidence scores
                scores = log_probs.max(dim=-1)[0]  # [1, 8, T]

                # Layer penalty (earlier codebooks get higher priority)
                scores = scores - (layer_ids * layer_penalty_factor)

                # Gumbel noise for position selection
                if position_temperature > 0.0:
                    scores = _gumbel_sample(scores, position_temperature, generator)

                # Mask out already unmasked positions
                sample_tokens = tokens[i : i + 1, :, :t_len]
                scores.masked_fill_(sample_tokens != mask_id, -float("inf"))

                # Select top-k positions to unmask. .flatten() on this non-contiguous view already copies.
                _, topk_idx = torch.topk(scores.flatten(), k)
                flat_tokens = sample_tokens.flatten()
                flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
                sample_tokens.copy_(flat_tokens.view_as(sample_tokens))

                # Mirror update into both cond and uncond input_ids halves for the next step.
                input_ids[i, :, c_len - t_len : c_len] = sample_tokens.squeeze(0)
                input_ids[B + i, :, :t_len] = sample_tokens.squeeze(0)

        return tokens

    def _load_fused_projections(self, state_dict: dict[str, torch.Tensor]) -> set[str]:
        """Pack the checkpoint's separate q/k/v and gate/up shards into the fused params.

        This has to happen explicitly. The generic per-tensor path below looks
        the destination up by name, and ``q_proj``/``gate_proj`` no longer exist
        as modules -- so it would find nothing, log a warning nobody reads, and
        leave the fused parameters at their random initialization. A corrupted
        model that still answers requests is worse than a failed load, so a
        missing or wrong-shaped shard raises here.
        """
        loaded: set[str] = set()
        packed_params = 0
        for idx, layer in enumerate(self.layers):
            for fused_path, shard_paths in _FUSED_PROJECTIONS.items():
                keys = [f"llm.layers.{idx}.{path}.weight" for path in shard_paths]
                missing = [k for k in keys if k not in state_dict]
                if len(missing) == len(keys):
                    continue
                if missing:
                    raise ValueError(
                        f"OmniVoice checkpoint is missing {missing} needed to build "
                        f"layers.{idx}.{fused_path}; refusing to load a partially "
                        f"initialized fused projection."
                    )
                module = layer
                for part in fused_path.split("."):
                    module = getattr(module, part)
                packed = torch.cat([state_dict[k] for k in keys], dim=0)
                if packed.shape != module.weight.shape:
                    raise ValueError(
                        f"OmniVoice checkpoint shards {keys} pack to {tuple(packed.shape)} "
                        f"but layers.{idx}.{fused_path}.weight is {tuple(module.weight.shape)}."
                    )
                module.weight.data.copy_(packed)
                loaded.update(keys)
                packed_params += 1

        expected = len(self.layers) * len(_FUSED_PROJECTIONS)
        if loaded and packed_params != expected:
            raise ValueError(
                f"OmniVoice checkpoint filled {packed_params}/{expected} fused projections; "
                f"the remaining ones would stay randomly initialized."
            )
        logger.info("Generator: packed %d/%d fused projections", packed_params, expected)
        return loaded

    def load_weights(self, model_dir: str, device: torch.device) -> None:
        """Load weights from HuggingFace OmniVoice model.safetensors.

        The HF checkpoint contains:
        - llm.* -> Qwen3 transformer weights
        - audio_embeddings.* -> audio embedding table
        - audio_heads.* -> prediction head
        """
        import os

        from safetensors.torch import load_file

        weights_path = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at {weights_path}")

        state_dict = load_file(weights_path, device=str(device))

        # Map HF weight names to our module names
        loaded_keys = set()

        # 1. Text embedding: llm.embed_tokens.weight -> text_embedding.weight
        text_emb_key = "llm.embed_tokens.weight"
        if text_emb_key in state_dict:
            self.text_embedding.weight.data.copy_(state_dict[text_emb_key])
            loaded_keys.add(text_emb_key)

        # 2. Audio embeddings
        for key in ["audio_embeddings.weight"]:
            if key in state_dict:
                self.audio_embeddings.weight.data.copy_(state_dict[key])
                loaded_keys.add(key)

        # 3. Audio heads
        for key in ["audio_heads.weight"]:
            if key in state_dict:
                self.audio_heads.weight.data.copy_(state_dict[key])
                loaded_keys.add(key)

        # 4a. Fused projections, packed from their separate checkpoint shards.
        loaded_keys |= self._load_fused_projections(state_dict)

        # 4b. Remaining transformer weights: llm.layers.N.* -> layers.N.*
        for key, value in state_dict.items():
            if key in loaded_keys:
                continue
            if key.startswith("llm.layers."):
                # llm.layers.0.self_attn.q_proj.weight -> layers.0.self_attn.q_proj.weight
                our_key = key.replace("llm.layers.", "layers.")
                parts = our_key.split(".")
                module = self
                try:
                    for part in parts[:-1]:
                        if part.isdigit():
                            module = module[int(part)]
                        else:
                            module = getattr(module, part)
                    param_name = parts[-1]
                    param = getattr(module, param_name)
                    if isinstance(param, nn.Parameter):
                        param.data.copy_(value)
                    elif isinstance(param, torch.Tensor):
                        param.copy_(value)
                    loaded_keys.add(key)
                except (AttributeError, IndexError, KeyError) as e:
                    logger.warning("Failed to load weight %s: %s", key, e)

        # 5. Final norm: llm.norm.weight -> norm.weight
        norm_key = "llm.norm.weight"
        if norm_key in state_dict:
            self.norm.weight.data.copy_(state_dict[norm_key])
            loaded_keys.add(norm_key)

        unloaded = set(state_dict.keys()) - loaded_keys
        # Filter out audio_tokenizer weights (loaded in decoder stage)
        unloaded = {k for k in unloaded if not k.startswith("audio_tokenizer.")}
        if unloaded:
            logger.info(
                "Generator: %d/%d weights loaded, %d skipped (decoder weights)",
                len(loaded_keys),
                len(state_dict),
                len(unloaded),
            )
        else:
            logger.info("Generator: all %d weights loaded", len(loaded_keys))

        if self._cuda_graph_fwd is not None:
            self._cuda_graph_fwd.warmup(device)
