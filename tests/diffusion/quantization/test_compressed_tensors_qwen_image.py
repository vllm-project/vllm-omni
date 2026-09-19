# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression coverage for compressed-tensors MLP-only (ignore attn) on Qwen-Image.

These tests pin the three defects fixed in the compressed-tensors MLP-only
Qwen-Image support so the silent prompt-conditioning failure path is protected
in CI:

1. ``compressed-tensors`` method-name canonicalization (hyphen vs underscore) and
   the ``from_config`` fallback in ``_build_single``.
2. ``QwenImagePipeline`` exposing the transformer's ``packed_modules_mapping`` so
   fused attention projections can be expanded during ignore matching.
3. The ``.attn.to_out.0`` -> ``.attn.to_out`` ignore-list rewrite.
"""

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_compressed_tensors_method_canonicalization_and_from_config(monkeypatch):
    """Hyphenated and underscore spellings resolve to the same canonical method,
    and ``_build_single`` falls back to ``from_config`` (injecting the canonical
    ``quant_method``) when the config is not constructible from raw kwargs.

    CompressedTensorsConfig.from_config expects a full checkpoint config
    (config_groups / format / scheme), so we stub the class to verify the
    canonicalization + fallback contract without depending on that structure.
    """
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from vllm_omni.quantization import build_quant_config

    seen: dict = {}

    def fake_init(self, *args, **kwargs):
        # Simulate "not constructible from raw kwargs" -> triggers from_config.
        raise TypeError("simulated signature mismatch")

    def fake_from_config(cls, config):
        seen["config"] = dict(config)
        return object()  # sentinel; callers only need a returned object

    monkeypatch.setattr(CompressedTensorsConfig, "__init__", fake_init)
    monkeypatch.setattr(
        CompressedTensorsConfig, "from_config", classmethod(fake_from_config)
    )

    # Canonical (hyphenated) spelling reaches from_config with the method set.
    build_quant_config("compressed-tensors", ignore=["lm_head"])
    assert seen["config"]["quant_method"] == "compressed-tensors"

    # Underscore spelling normalizes to the same canonical method.
    seen.clear()
    build_quant_config("compressed_tensors", ignore=["lm_head"])
    assert seen["config"]["quant_method"] == "compressed-tensors"


def test_qwen_image_pipeline_exposes_transformer_packed_modules_mapping():
    """The pipeline must expose the transformer's packed_modules_mapping so that
    configure_quant_config() populates quant_config.packed_modules_mapping and
    should_ignore_layer can expand fused attn projections (to_qkv/add_kv_proj)
    when matching the ignore list on GPU."""
    from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import (
        QwenImagePipeline,
    )
    from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
        QwenImageTransformer2DModel,
    )

    transformer_mapping = QwenImageTransformer2DModel.packed_modules_mapping
    assert transformer_mapping, "transformer must define a packed_modules_mapping"

    pipeline_mapping = QwenImagePipeline.packed_modules_mapping
    assert pipeline_mapping == transformer_mapping

    # Must be a copy, not the transformer's class attribute, so downstream
    # mutations (configure_quant_config) don't leak back onto the transformer.
    assert pipeline_mapping is not transformer_mapping


def test_qwen_image_ct_ignore_name_rewrite():
    """The ``.attn.to_out.0`` -> ``.attn.to_out`` rewrite lets the fused output
    projection be matched against the diffusers-style ignore list, and the
    pipeline opts in via the ``_ct_ignore_name_rewrites`` marker attribute."""
    from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import (
        QwenImagePipeline,
    )
    from vllm_omni.diffusion.registry import _apply_ignore_rewrite

    rewrites = QwenImagePipeline._ct_ignore_name_rewrites
    assert rewrites == {".attn.to_out.0": ".attn.to_out"}

    # Fused output projection (diffusers name) is rewritten to the vLLM fused name.
    assert (
        _apply_ignore_rewrite("transformer_blocks.0.attn.to_out.0", rewrites)
        == "transformer_blocks.0.attn.to_out"
    )
    # Non-matching names are left untouched.
    assert (
        _apply_ignore_rewrite("transformer_blocks.0.ff.net.0_proj", rewrites)
        == "transformer_blocks.0.ff.net.0_proj"
    )
