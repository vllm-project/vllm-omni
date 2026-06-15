# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online quantization helpers for the Qwen2.5-VL text encoder.

Mirrors the Z-Image path (``create_transformers_model``) but quantizes the
language-model backbone only; the vision tower and ``lm_head`` stay BF16 since
Edit-2509/Layered condition on images and Layered runs ``generate()``.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration
from vllm.model_executor.models.transformers.utils import init_on_device_without_buffers

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.utils import init_parameters, recursive_replace_linear


def text_encoder_quant_enabled(od_config: OmniDiffusionConfig) -> bool:
    """Whether the Qwen2.5-VL text encoder should be online-quantized."""
    return od_config.quantization_config is not None


def remap_qwen_vl_text_encoder_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    prefix: str = "text_encoder.",
) -> Iterable[tuple[str, torch.Tensor]]:
    """Remap legacy checkpoint keys to the transformers>=5 nested layout.

    ``from_pretrained`` does this internally; the meta-init + ``AutoWeightsLoader``
    path bypasses it, so apply the same remapping explicitly.
    """
    rules = (
        ("model.layers.", "model.language_model.layers."),
        ("model.embed_tokens.", "model.language_model.embed_tokens."),
        ("model.norm.", "model.language_model.norm."),
        ("visual.", "model.visual."),
    )
    for name, weight in weights:
        if name.startswith(prefix):
            sub = name[len(prefix) :]
            for legacy, new in rules:
                if sub.startswith(legacy):
                    sub = new + sub[len(legacy) :]
                    break
            name = prefix + sub
        yield name, weight


def get_qwen_vl_language_model(text_encoder: nn.Module) -> nn.Module:
    """Return the language-model backbone of a Qwen2.5-VL encoder."""
    inner = getattr(text_encoder, "model", None)
    if inner is not None and hasattr(inner, "language_model"):
        return inner.language_model
    if hasattr(text_encoder, "language_model"):
        return text_encoder.language_model
    return text_encoder


def build_quantized_qwen_vl_text_encoder(
    model: str,
    od_config: OmniDiffusionConfig,
    device: torch.device,
    *,
    subfolder: str = "text_encoder",
    local_files_only: bool = False,
) -> Qwen2_5_VLForConditionalGeneration:
    """Build a Qwen2.5-VL text encoder with an online-quantized language model.

    Returns a meta-initialized module whose weights are loaded later by
    ``DiffusersPipelineLoader`` via a ``text_encoder.`` ``ComponentSource``.
    """
    config = AutoConfig.from_pretrained(model, subfolder=subfolder, local_files_only=local_files_only)
    with init_on_device_without_buffers("meta"):
        text_encoder = Qwen2_5_VLForConditionalGeneration._from_config(config)

    language_model = get_qwen_vl_language_model(text_encoder)
    recursive_replace_linear(language_model, od_config)

    init_parameters(text_encoder, dtype=od_config.dtype, device=device)

    # init_parameters severs the lm_head/embedding tie; re-tie it (as Z-Image
    # does) so generate() captioning works when lm_head.weight isn't in the ckpt.
    if _tie_word_embeddings(config) and getattr(text_encoder, "lm_head", None) is not None:
        input_embeddings = text_encoder.get_input_embeddings()
        if input_embeddings is not None:
            text_encoder.lm_head.weight = input_embeddings.weight

    return text_encoder


def _tie_word_embeddings(config) -> bool:
    """Resolve ``tie_word_embeddings`` from a (possibly nested) HF config."""
    if getattr(config, "tie_word_embeddings", None) is not None:
        return bool(config.tie_word_embeddings)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return bool(getattr(text_config, "tie_word_embeddings", False))
    return False
