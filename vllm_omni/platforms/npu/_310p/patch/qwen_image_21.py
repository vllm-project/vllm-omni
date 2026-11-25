# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Patch Qwen-Image-2.1 components for the 310P NPU path."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn

    from vllm_omni.diffusion.data import OmniDiffusionConfig

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


def apply_qwen_image_21_patches(model: nn.Module, od_config: OmniDiffusionConfig) -> None:
    apply_rotary_patches()
    apply_text_encoder_dtype(model, od_config)
