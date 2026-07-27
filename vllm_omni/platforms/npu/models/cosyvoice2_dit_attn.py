# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU patches for CosyVoice2 / Token2Wav DiT attention.

CosyVoice2's DiT ``Attention.forward`` builds a key-padding style mask
``(B, 1, 1, S)`` via ``mask.unsqueeze(1)`` and feeds it to
``F.scaled_dot_product_attention``. On Ascend that call is routed to
``npu_fusion_attention`` / ``aclnnFlashAttentionScore``, which only accepts
mask shapes ``[B,N,S,S] / [B,1,S,S] / [1,1,S,S] / [S,S]`` and fails with
error **161001** (tiling / parameter invalid) for ``[B,1,1,S]``.

This module:
1. Expands DiT attention masks to ``[B, 1, S, S]`` before SDPA.
2. Provides a MATH-backend SDPA context so inference can avoid the fused FA
   kernel entirely when the platform still incorrectly routes SDPA.
3. Fuses the three gated residual Mul+Add pairs in each streaming DiT block.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext

import torch
import torch.nn.functional as F
from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False
_original_dit_block_forward_chunk = None
_original_modulate = None


def _fused_gated_residual_enabled() -> bool:
    value = os.environ.get("VLLM_OMNI_MINICPMO_FUSED_GATED_RESIDUAL", "1")
    return value.strip().lower() not in {"0", "false", "off", "no"}


def _fused_residual(
    x: torch.Tensor,
    gate: torch.Tensor,
    branch: torch.Tensor,
) -> torch.Tensor:
    """Compute ``x + gate * branch`` as one elementwise NPU operation."""
    return torch.addcmul(x, gate, branch)


def _patched_dit_block_forward_chunk(
    self,
    x: torch.Tensor,
    c: torch.Tensor,
    cnn_cache: torch.Tensor | None = None,
    att_cache: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
):
    assert _original_dit_block_forward_chunk is not None
    assert _original_modulate is not None
    if x.device.type != "npu":
        return _original_dit_block_forward_chunk(self, x, c, cnn_cache, att_cache, mask)

    (
        shift_msa,
        scale_msa,
        gate_msa,
        shift_mlp,
        scale_mlp,
        gate_mlp,
        shift_conv,
        scale_conv,
        gate_conv,
    ) = self.adaLN_modulation(c).chunk(9, dim=-1)
    x_att, new_att_cache = self.attn.forward_chunk(
        _original_modulate(self.norm1(x), shift_msa, scale_msa),
        att_cache,
        mask,
    )
    x = _fused_residual(x, gate_msa, x_att)
    x_conv, new_cnn_cache = self.conv.forward_chunk(
        _original_modulate(self.norm3(x), shift_conv, scale_conv),
        cnn_cache,
    )
    x = _fused_residual(x, gate_conv, x_conv)
    x = _fused_residual(
        x,
        gate_mlp,
        self.mlp(_original_modulate(self.norm2(x), shift_mlp, scale_mlp)),
    )
    return x, new_cnn_cache, new_att_cache


def _expand_attn_mask_for_npu(
    attn_mask: torch.Tensor | None,
    q_len: int,
    kv_len: int | None = None,
) -> torch.Tensor | None:
    """Expand CosyVoice key-padding masks to Ascend FA-compatible shapes."""
    if attn_mask is None:
        return None
    kv_len = kv_len if kv_len is not None else q_len

    # (B, 1, S) key-padding -> (B, S_q, S_kv) then unsqueeze heads below.
    if attn_mask.dim() == 3 and attn_mask.shape[-2] == 1:
        attn_mask = attn_mask.expand(-1, q_len, -1)
    if attn_mask.dim() == 3:
        # (B, S_q, S_kv) -> (B, 1, S_q, S_kv)
        attn_mask = attn_mask.unsqueeze(1)
    if attn_mask.dim() == 4 and attn_mask.shape[-2] == 1 and kv_len > 1:
        # (B, 1, 1, S) -> (B, 1, S_q, S)
        attn_mask = attn_mask.expand(-1, -1, q_len, -1)
    return attn_mask


def _patched_attention_forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    b, t, c = x.shape

    q = self.to_heads(self.to_q(x))
    k = self.to_heads(self.to_k(x))
    v = self.to_heads(self.to_v(x))

    q = self.q_norm(q)
    k = self.k_norm(k)

    attn_mask = _expand_attn_mask_for_npu(attn_mask, q_len=t, kv_len=t)
    x = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=self.attn_drop.p if self.training else 0.0,
    )
    x = x.transpose(1, 2).reshape(b, t, -1)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def _patched_attention_forward_chunk(
    self,
    x: torch.Tensor,
    att_cache: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
):
    b, t, c = x.shape

    q = self.to_heads(self.to_q(x))
    k = self.to_heads(self.to_k(x))
    v = self.to_heads(self.to_v(x))

    q = self.q_norm(q)
    k = self.k_norm(k)

    if att_cache is not None:
        k_cache, v_cache = att_cache.chunk(2, dim=3)
        k = torch.cat([k, k_cache], dim=2)
        v = torch.cat([v, v_cache], dim=2)

    new_att_cache = torch.cat([k, v], dim=3)
    kv_len = k.shape[2]
    if attn_mask is not None:
        attn_mask = _expand_attn_mask_for_npu(attn_mask, q_len=t, kv_len=kv_len)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    x = x.transpose(1, 2).reshape(b, t, -1)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x, new_att_cache


@contextmanager
def npu_math_sdpa_context() -> Iterator[None]:
    """Force SDPA MATH backend so Ascend does not call fused FA."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.MATH):
            yield
    except Exception:
        # Older torch / missing backend enum — just run as-is.
        with nullcontext():
            yield


def _disable_upsample_encoder_compile() -> None:
    """Run CosyVoice2 UpsampleConformerEncoderV2 eagerly on Ascend.

    ``forward_chunk`` is decorated with ``@torch.compile(dynamic=True,
    backend="eager")`` upstream. The backend gives no speedup (still eager),
    but Dynamo tracing of the relative-position attention with a KV cache
    creates a symbolic add ``matrix_ac(..., chunk+cache) + matrix_bd(...,
    (pos//2)+1)`` whose sizes Ascend torch's fake-tensor ``infer_size``
    cannot reconcile, crashing on the 2nd+ streaming chunk. Unwrapping the
    compile and disabling Dynamo keeps concrete shapes (always equal).
    """
    try:
        from cosyvoice2.transformer import upsample_encoder_v2
    except ImportError:
        return

    enc_cls = getattr(upsample_encoder_v2, "UpsampleConformerEncoderV2", None)
    if enc_cls is None:
        return

    fn = enc_cls.forward_chunk
    orig = getattr(fn, "_torchdynamo_orig_callable", None) or getattr(fn, "__wrapped__", fn)
    enc_cls.forward_chunk = torch._dynamo.disable(orig)  # type: ignore[method-assign]
    logger.info("Disabled torch.compile on CosyVoice2 UpsampleConformerEncoderV2.forward_chunk (Ascend eager)")


def apply_cosyvoice2_dit_attn_npu_patch() -> None:
    """Apply CosyVoice2 DiT attention and residual patches for Ascend."""
    global _PATCHED, _original_dit_block_forward_chunk, _original_modulate
    if _PATCHED:
        return

    try:
        from cosyvoice2.flow import decoder_dit
    except ImportError:
        logger.debug("cosyvoice2 not installed; skip DiT attn NPU patch")
        return

    attn_cls = getattr(decoder_dit, "Attention", None)
    dit_block_cls = getattr(decoder_dit, "DiTBlock", None)
    if attn_cls is None:
        return

    attn_cls.forward = _patched_attention_forward  # type: ignore[method-assign]
    attn_cls.forward_chunk = _patched_attention_forward_chunk  # type: ignore[method-assign]
    fused_residuals = dit_block_cls is not None and _fused_gated_residual_enabled()
    if fused_residuals:
        assert dit_block_cls is not None
        _original_dit_block_forward_chunk = dit_block_cls.forward_chunk
        _original_modulate = decoder_dit.modulate
        dit_block_cls.forward_chunk = _patched_dit_block_forward_chunk  # type: ignore[method-assign]
    _disable_upsample_encoder_compile()
    _PATCHED = True
    logger.info(
        "Applied CosyVoice2 DiT NPU patches (attention mask expand%s)",
        " and fused gated residuals" if fused_residuals else "",
    )
