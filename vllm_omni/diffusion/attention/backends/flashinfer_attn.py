# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)

logger = init_logger(__name__)

try:
    from flashinfer.prefill import single_prefill_with_kv_cache

    HAS_FLASHINFER = True
except Exception as e:
    HAS_FLASHINFER = False
    logger.warning(
        "FlashInfer is unavailable; FLASHINFER_ATTN backend will not work. Reason: %s",
        e,
    )


class FlashInferAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @classmethod
    def supports_attention_mask(cls) -> bool:
        # FlashInfer single_prefill_with_kv_cache accepts a ``custom_mask`` kwarg
        # (2D boolean, ``True`` = keep) for non-causal attention. See reviewer
        # comment on #3079 and the flashinfer docs:
        #   https://docs.flashinfer.ai/generated/flashinfer.prefill.single_prefill_with_kv_cache.html
        return True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # FlashInfer dense prefill is well-tested for these head_dims on
        # Ampere/Hopper/Blackwell. Covers the dominant diffusion DiT shapes
        # (SD3 = 64, Flux/HV/Wan = 128, joint-attn = 256).
        return [64, 128, 256]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_ATTN"

    @staticmethod
    def get_impl_cls() -> type["FlashInferAttentionImpl"]:
        return FlashInferAttentionImpl


class FlashInferAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale

    @staticmethod
    def _pack_mask_for_flashinfer(
        attn_mask: torch.Tensor, batch_size: int, qo_len: int, kv_len: int
    ) -> torch.Tensor | None:
        """Convert a diffusion-style attn_mask into the boolean form
        FlashInfer's ``custom_mask`` expects (``True`` = keep).

        Returns either ``(qo_len, kv_len)`` (shared across the batch) or
        ``(batch_size, qo_len, kv_len)`` (per-sample), or ``None`` when the
        mask is all-keep (elide). Only boolean masks are handled here;
        additive/float masks raise ``ValueError`` so the caller falls back to
        SDPA, which applies them with the correct softmax semantics. Shape
        mismatches also raise ``ValueError``.
        """
        mask = attn_mask
        if mask.dtype != torch.bool:
            # Additive masks (0 / -inf / -1e4 / finfo.min) cannot be faithfully
            # reduced to a boolean keep-mask here; SDPA handles them correctly.
            raise ValueError(
                f"non-boolean attn_mask (dtype={mask.dtype}); FlashInfer custom_mask "
                "is boolean-only — deferring to SDPA"
            )
        # Diffusion masks arrive as (qo,kv), (1,1,kv), (B,1,1,kv), (B,1,qo,kv)
        # or (B,H,qo,kv). The mask is identical across heads, so collapse the
        # head dim, but keep a real batch dim — indexing mask[0] would reuse
        # sample 0's padding for every sample (wrong under CFG / mixed lengths).
        if mask.dim() == 4:
            mask = mask[:, 0]  # (B, qo|1, kv)
        if mask.dim() == 3 and mask.shape[0] == 1:
            mask = mask[0]  # (qo|1, kv) — shared across the batch
        try:
            if mask.dim() >= 3:
                mask = mask.broadcast_to((batch_size, qo_len, kv_len))
            else:
                mask = mask.broadcast_to((qo_len, kv_len))
        except RuntimeError as e:
            raise ValueError(
                f"attn_mask shape {tuple(attn_mask.shape)} cannot broadcast to "
                f"(batch={batch_size}, qo_len={qo_len}, kv_len={kv_len})"
            ) from e
        if mask.all():
            return None
        # ``broadcast_to`` returns a non-contiguous view; materialize for the
        # kernel, which reads from GPU memory directly.
        return mask.contiguous()

    def _sdpa_fallback(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> torch.Tensor:
        from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl

        impl = SDPAImpl(
            num_heads=query.shape[2],
            head_size=query.shape[3],
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )
        return impl.forward_cuda(query, key, value, attn_metadata)

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if not HAS_FLASHINFER:
            raise ImportError(
                "FLASHINFER_ATTN backend requires flashinfer. "
                "Install it or set DIFFUSION_ATTENTION_BACKEND to another backend."
            )

        # Try the custom_mask path; if the mask can't be packed into the
        # (qo_len, kv_len) layout FlashInfer expects, fall back to SDPA
        # instead of risking an illegal-memory-access crash in the kernel.
        # Input layout is (B, S, H, D); FlashInfer dense prefill takes (S, H, D).
        batch_size = query.shape[0]

        custom_mask = None
        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            try:
                custom_mask = self._pack_mask_for_flashinfer(
                    attn_metadata.attn_mask,
                    batch_size=batch_size,
                    qo_len=query.shape[1],
                    kv_len=key.shape[1],
                )
            except ValueError as e:
                logger.debug("Falling back to SDPA for mask path: %s", e)
                return self._sdpa_fallback(query, key, value, attn_metadata)
            # FlashInfer cannot combine causal masking with a custom_mask; rather
            # than silently dropping the explicit mask (diverging from SDPA), let
            # SDPA handle the causal+mask case correctly.
            if custom_mask is not None and self.causal:
                logger.debug("causal=True with explicit attn_mask; deferring to SDPA")
                return self._sdpa_fallback(query, key, value, attn_metadata)

        outputs = []
        for b in range(batch_size):
            kwargs: dict = {
                "sm_scale": self.softmax_scale,
                "causal": self.causal,
                "return_lse": False,
            }
            if custom_mask is not None:
                kwargs["custom_mask"] = custom_mask if custom_mask.dim() == 2 else custom_mask[b]
            out = single_prefill_with_kv_cache(query[b], key[b], value[b], **kwargs)
            outputs.append(out)

        if batch_size == 1:
            return outputs[0].unsqueeze(0)
        return torch.stack(outputs, dim=0)
