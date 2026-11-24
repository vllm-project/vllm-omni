# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Patch Qwen-Image-2.1 components for the 310P NPU path."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

if TYPE_CHECKING:
    from torch import nn

    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)

_PATCHED = False


def apply_rotary_patches() -> None:
    """Route interleaved RoPE through the exact fp32 expansion.

    310P's aclop fallback implements only ``rotary_mode="half"``; the model
    default path calls ``npu_rotary_mul(..., rotary_mode="interleave")``.
    """
    global _PATCHED

    if _PATCHED:
        return

    from vllm_omni.diffusion.models.qwen_image_21 import qwen_image_21_transformer

    qwen_image_21_transformer._apply_qwen_image21_rotary_emb_npu = (
        qwen_image_21_transformer._apply_qwen_image21_rotary_emb_native
    )
    _PATCHED = True


def apply_text_encoder_dtype(model: nn.Module, od_config: OmniDiffusionConfig) -> None:
    """Move text-encoder parameters to the runtime dtype.

    ``Qwen3VLForConditionalGeneration.from_pretrained`` takes the dtype from the
    checkpoint config (BF16) and ignores ``od_config.dtype``. 310P has no BF16
    embedding or matmul path, so the encoder must follow the runtime dtype.
    Parameters are converted one at a time to avoid holding two full copies;
    non-floating buffers (RoPE frequencies) keep their FP32 precision.
    """
    from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import QwenImage21Pipeline

    if not isinstance(model, QwenImage21Pipeline):
        return

    dtype = od_config.dtype
    encoder = model.text_encoder
    if next(encoder.parameters()).dtype == dtype:
        return

    for parameter in encoder.parameters():
        if parameter.is_floating_point() and parameter.dtype != dtype:
            parameter.data = parameter.data.to(dtype)


def apply_vae_encode_tiling_patch(model: nn.Module, od_config: OmniDiffusionConfig) -> None:
    """Keep VAE *decode* tiling while encoding the condition image untiled.

    310P enables ``use_tiling`` so that the tile-parallel decode has tiles to
    work on, but the same flag also routes :meth:`encode` through
    ``tiled_encode``. Tiling moves the condition-image latents before denoising
    starts, which costs much more accuracy on this device than it does on the
    decode side: measured against the Atlas 800I A2 BF16 reference, image
    conditioned generation sits at 0.79 SSIM / 17.2 dB with the encoder tiled
    and 0.96 / 28.8 dB with only the decode tiled, at the same noise. The
    untiled encoder is also the faster path here (0.68 s vs 1.04 s for the
    768x1024 condition image).

    The wrapper restores the flag afterwards and falls back to the tiled encode
    if the untiled pass runs out of memory, so this only changes numerics, not
    whether a configuration can run.
    """
    vae = getattr(model, "vae", None)
    if vae is None or not getattr(vae, "use_tiling", False) or getattr(vae, "_310p_untiled_encode", False):
        return

    encode = vae.encode

    def encode_untiled(*args, **kwargs):
        vae.use_tiling = False
        try:
            try:
                return encode(*args, **kwargs)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
        finally:
            vae.use_tiling = True

        # Retry outside the handler: while it is active its traceback keeps the
        # failed encode frames - and the intermediates their locals reference -
        # alive, and empty_cache() cannot free live allocations. Leaving the
        # `except` block releases the exception (the `as` name is cleared with
        # it), so the tiled retry starts from a genuinely empty cache. The flag
        # is already restored above, which is what the tiled retry needs.
        logger.warning(
            "Untiled VAE encode ran out of memory; retrying with tiling. "
            "Image-conditioned output will carry the tiling difference."
        )
        npu = getattr(torch, "npu", None)
        if npu is not None and npu.is_available():
            npu.empty_cache()
        return encode(*args, **kwargs)

    vae.encode = encode_untiled
    vae._310p_untiled_encode = True
    logger.info("310P: VAE tiling applies to decode only; the encoder runs untiled.")


def apply_qwen_image_21_patches(model: nn.Module, od_config: OmniDiffusionConfig) -> None:
    apply_rotary_patches()
    apply_text_encoder_dtype(model, od_config)
    apply_vae_encode_tiling_patch(model, od_config)
