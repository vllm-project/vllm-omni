# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803 - Triton constexpr parameters use kernel-style capitals.
"""Gated DeltaNet core for Drama 3's non-speculative prefill and mixed steps.

vLLM's standard ``_forward_core`` (kept for pure decode, speculative and
non-FlashInfer paths) handles a step that holds prefills like this, per GDN
layer:

* decode tokens: Q/K/V split and copied into contiguous tensors, then the
  generic gating + recurrent kernel into a temporary output;
* prefill tokens: the requests' states gathered out of the state pool, fresh
  ones zeroed, FlashInfer's chunk kernel (after a separate ``exp(g)`` and an
  ``int64`` copy of the sequence starts), and the final states scattered back;
* both outputs concatenated and copied into the layer's output buffer.

Here the decode tokens go through the packed recurrent kernel the pure-decode
path already uses (it reads the packed convolution output and writes the
output buffer), and FlashInfer reads and writes the state pool in place
through ``state_indices`` -- a small kernel zeroes the rows of requests
without an initial state first -- writing its output straight into the
buffer; ``exp(g)`` comes out of the post-convolution prep kernel and the
``int64`` sequence starts are made once per step. Prefills fall back to the
standard gather / scatter around the chunk kernel when another backend runs
them or the pool's dtype is neither fp32 nor bf16.

The layers also report a metadata builder that skips the FLA chunk indices
and offsets (a CPU pass over the sequence lengths and two host-to-device
copies on every step with prefills) when FlashInfer runs the prefills: its
kernel does not read them.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
    causal_conv1d_fn,
    fused_post_conv_prep,
    fused_recurrent_gated_delta_rule_packed_decode,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend, GDNAttentionMetadataBuilder

_ZERO_BLOCK = 4096


@triton.jit
def _zero_fresh_rows_kernel(pool_ptr, row_stride, slots_ptr, has_state_ptr, row_numel, BLOCK: tl.constexpr):
    seq = tl.program_id(0)
    if tl.load(has_state_ptr + seq) == 0:
        slot = tl.load(slots_ptr + seq).to(tl.int64)
        offs = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        tl.store(
            pool_ptr + slot * row_stride + offs,
            tl.zeros([BLOCK], dtype=pool_ptr.dtype.element_ty),
            mask=offs < row_numel,
        )


def zero_fresh_state_rows_(pool: torch.Tensor, slots: torch.Tensor, has_initial_state: torch.Tensor) -> None:
    """Zero ``pool[slots[i]]`` wherever ``has_initial_state[i]`` is False, in place.

    Rows of requests continuing a chunked prefill keep their state; nothing
    else in the pool is touched. Each row's ``[H, V, K]`` block must be
    contiguous (the first dimension may be padded).
    """
    num = slots.shape[0]
    row_numel = pool[0].numel()
    if num == 0:
        return
    if not pool[0].is_contiguous():
        raise ValueError("GDN state rows must be contiguous within a slot")
    _zero_fresh_rows_kernel[(num, triton.cdiv(row_numel, _ZERO_BLOCK))](
        pool, pool.stride(0), slots, has_initial_state, row_numel, BLOCK=_ZERO_BLOCK
    )


def _prefill_starts_int64(attn_metadata: Any) -> torch.Tensor:
    """The prefill sequence starts as ``int64``, made once per step and shared by every layer."""
    cached = getattr(attn_metadata, "_drama3_prefill_starts_i64", None)
    if cached is None:
        cached = attn_metadata.prefill_query_start_loc.to(torch.int64)
        try:
            attn_metadata._drama3_prefill_starts_i64 = cached
        except AttributeError:
            pass
    return cached


# FlashInfer's GDN prefill (CuTe DSL) compiles one variant for a single
# sequence and one for several on first use, in-process: ~8 s and ~4 s on
# H200, while every stream of the step waits. Each (state dtype, row shape,
# activation dtype) is compiled before serving, from the first GDN call made
# outside graph capture once the state pool is bound (startup warmup).
_FLASHINFER_WARM: set[tuple] = set()


def _flashinfer_prefill_warmup(layer: Any, ssm_state: torch.Tensor, dtype: torch.dtype) -> None:
    key = (ssm_state.device, ssm_state.dtype, tuple(ssm_state.shape[1:]), dtype)
    if key in _FLASHINFER_WARM:
        return
    _FLASHINFER_WARM.add(key)
    from flashinfer.gdn_prefill import chunk_gated_delta_rule

    device = ssm_state.device
    heads = layer.num_k_heads // layer.tp_size
    value_heads = layer.num_v_heads // layer.tp_size
    # Rows of the pool are contiguous (see zero_fresh_state_rows_), so a
    # contiguous scratch pool has the same inner strides: the same variant.
    scratch = torch.zeros((2, *ssm_state.shape[1:]), dtype=ssm_state.dtype, device=device)
    for count in (1, 2):
        tokens = 64 * count
        chunk_gated_delta_rule(
            q=torch.zeros(tokens, heads, layer.head_k_dim, dtype=dtype, device=device),
            k=torch.zeros(tokens, heads, layer.head_k_dim, dtype=dtype, device=device),
            v=torch.zeros(tokens, value_heads, layer.head_v_dim, dtype=dtype, device=device),
            g=torch.ones(tokens, value_heads, dtype=torch.float32, device=device),
            beta=torch.zeros(tokens, value_heads, dtype=torch.float32, device=device),
            initial_state=scratch,
            output_final_state=True,
            cu_seqlens=torch.arange(0, tokens + 1, 64, dtype=torch.int64, device=device),
            use_qk_l2norm_in_kernel=False,
            output=torch.empty(tokens, value_heads, layer.head_v_dim, dtype=dtype, device=device),
            output_state=scratch,
            state_indices=torch.arange(count, dtype=torch.int32, device=device),
            backend="flashinfer",
        )


def _uses_flashinfer_direct(layer: Any, ssm_state: torch.Tensor) -> bool:
    return (
        getattr(layer.chunk_gated_delta_rule, "gdn_prefill_backend", None) == "flashinfer"
        # The standard path casts gathered rows to fp32 and rounds the final
        # state back to the pool's dtype; FlashInfer does the same in place.
        and ssm_state.dtype in (torch.float32, torch.bfloat16)
    )


def mixed_forward_core(
    layer: Any,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    attn_metadata: Any,
) -> None:
    """Non-speculative GDN core of a step holding prefills (decode tokens first)."""
    md = attn_metadata
    kv_cache = layer.kv_cache
    conv_state = kv_cache[0] if is_conv_state_dim_first() else kv_cache[0].transpose(-1, -2)
    ssm_state = kv_cache[1]
    tokens = md.num_actual_tokens
    decode_tokens = md.num_decode_tokens
    mixed_qkv = mixed_qkv[:tokens]
    b = b[:tokens]
    a = a[:tokens]

    # 1. Causal convolution over decode and prefill tokens (updates every conv state once).
    conv_weights = layer.conv1d.weight.view(layer.conv1d.weight.size(0), layer.conv1d.weight.size(2))
    conv_out = causal_conv1d_fn(
        mixed_qkv.transpose(0, 1),
        conv_weights,
        layer.conv1d.bias,
        activation=layer.activation,
        conv_states=conv_state,
        has_initial_state=md.has_initial_state,
        cache_indices=md.non_spec_state_indices_tensor,
        query_start_loc=md.non_spec_query_start_loc,
        metadata=md,
    ).transpose(0, 1)

    # 2. Decode tokens: one token per request, state updated in place by the
    # packed kernel, which reads the convolution output as is.
    if decode_tokens > 0:
        packed = conv_out[:decode_tokens]
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=packed if packed.stride(-1) == 1 else packed.contiguous(),
            a=a[:decode_tokens],
            b=b[:decode_tokens],
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            scale=layer.head_k_dim**-0.5,
            initial_state=ssm_state,
            out=core_attn_out[:decode_tokens].unsqueeze(1),
            ssm_state_indices=md.non_spec_state_indices_tensor[:decode_tokens],
            use_qk_l2norm_in_kernel=True,
        )

    # 3. Prefill tokens through the chunk kernel.
    direct = _uses_flashinfer_direct(layer, ssm_state)
    q, k, v, g, beta = fused_post_conv_prep(
        conv_output=conv_out[decode_tokens:],
        a=a[decode_tokens:],
        b=b[decode_tokens:],
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        num_k_heads=layer.num_k_heads // layer.tp_size,
        head_k_dim=layer.head_k_dim,
        head_v_dim=layer.head_v_dim,
        apply_l2norm=True,
        # FlashInfer takes the decay exp(g); the FLA kernels take g.
        output_g_exp=direct,
    )
    slots = md.prefill_state_indices
    if direct:
        from flashinfer.gdn_prefill import chunk_gated_delta_rule

        zero_fresh_state_rows_(ssm_state, slots, md.prefill_has_initial_state)
        chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=ssm_state,
            output_final_state=True,
            cu_seqlens=_prefill_starts_int64(md),
            use_qk_l2norm_in_kernel=False,
            output=core_attn_out[decode_tokens:tokens],
            output_state=ssm_state,
            state_indices=slots,
            backend="flashinfer",
        )
        return
    initial_state = ssm_state[slots]
    initial_state[~md.prefill_has_initial_state, ...] = 0
    out, final_state = layer.chunk_gated_delta_rule(
        q=q.unsqueeze(0),
        k=k.unsqueeze(0),
        v=v.unsqueeze(0),
        g=g.unsqueeze(0),
        beta=beta.unsqueeze(0),
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=md.prefill_query_start_loc,
        chunk_indices=md.chunk_indices,
        chunk_offsets=md.chunk_offsets,
        use_qk_l2norm_in_kernel=False,
    )
    ssm_state[slots] = final_state.to(ssm_state.dtype)
    core_attn_out[decode_tokens:tokens] = out.squeeze(0)


class Drama3GDNAttentionMetadataBuilder(GDNAttentionMetadataBuilder):
    """GDN metadata without the FLA chunk indices / offsets when FlashInfer runs the prefills."""

    def _build_chunk_metadata(self, prefill_query_start_loc, prefill_query_start_loc_cpu, device):
        if self.gdn_prefill_backend == "flashinfer":
            return None, None
        return super()._build_chunk_metadata(prefill_query_start_loc, prefill_query_start_loc_cpu, device)


class Drama3GDNAttentionBackend(GDNAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[Drama3GDNAttentionMetadataBuilder]:
        return Drama3GDNAttentionMetadataBuilder


class Drama3GatedDeltaNet(QwenGatedDeltaNetAttention):
    """``QwenGatedDeltaNetAttention`` whose non-speculative steps with prefills use :func:`mixed_forward_core`."""

    _drama3_warm_done: bool = False

    def get_attn_backend(self):
        return Drama3GDNAttentionBackend

    def _forward_core(self, mixed_qkv, b, a, core_attn_out):
        if not self._drama3_warm_done:
            self._maybe_warm_prefill(mixed_qkv.dtype)
        attn_metadata = get_forward_context().attn_metadata
        md = attn_metadata.get(self.prefix) if isinstance(attn_metadata, dict) else None
        if md is None or md.spec_sequence_masks is not None or md.num_prefills == 0:
            return super()._forward_core(mixed_qkv, b, a, core_attn_out)
        mixed_forward_core(self, mixed_qkv, b, a, core_attn_out, md)

    def _maybe_warm_prefill(self, dtype: torch.dtype) -> None:
        # Unset until the KV cache is bound (memory profiling runs first).
        kv_cache = getattr(self, "kv_cache", None)
        if not isinstance(kv_cache, (list, tuple)) or len(kv_cache) < 2:
            return
        ssm_state = kv_cache[1]
        if (
            not isinstance(ssm_state, torch.Tensor)
            or ssm_state.dim() != 4
            or ssm_state.numel() == 0
            or not ssm_state.is_cuda
            or torch.cuda.is_current_stream_capturing()
        ):
            return
        if _uses_flashinfer_direct(self, ssm_state):
            _flashinfer_prefill_warmup(self, ssm_state, dtype)
        self._drama3_warm_done = True


def install_drama3_gdn(model: torch.nn.Module) -> int:
    """Switch the backbone's Qwen GDN layers to :class:`Drama3GatedDeltaNet`; returns how many."""
    count = 0
    for module in model.modules():
        if type(module) is QwenGatedDeltaNetAttention:
            module.__class__ = Drama3GatedDeltaNet
            count += 1
    return count


__all__ = [
    "Drama3GDNAttentionBackend",
    "Drama3GDNAttentionMetadataBuilder",
    "Drama3GatedDeltaNet",
    "install_drama3_gdn",
    "mixed_forward_core",
    "zero_fresh_state_rows_",
]
