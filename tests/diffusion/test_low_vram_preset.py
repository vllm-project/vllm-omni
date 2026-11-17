# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Memory composition policy tests; no model downloads or GPU allocations."""

from copy import deepcopy

import pytest

from vllm_omni.diffusion.low_vram import _PIPELINES, expand_low_vram_preset, validate_low_vram_pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_preset_expands_without_mutating_inputs():
    raw = {"low_vram_preset": "fp8"}
    expanded = expand_low_vram_preset(raw)
    assert raw == {"low_vram_preset": "fp8"}
    assert expanded["quantization_config"] == "fp8"
    assert expanded["diffusion_offload_config"] == {"mode": "layer", "components": ["dit", "text_encoder"]}
    assert expanded["cache_backend"] == "none"
    assert expanded["vae_use_tiling"] is True
    assert expand_low_vram_preset(expanded) == expanded
    expanded["diffusion_offload_config"]["components"].clear()
    assert expand_low_vram_preset(raw)["diffusion_offload_config"]["components"] == ["dit", "text_encoder"]


@pytest.mark.parametrize("mode", ["layer"])
@pytest.mark.parametrize("cache", ["none", "tea_cache"])
def test_z_image_composition_matrix(mode, cache):
    raw = {
        "low_vram_preset": "fp8",
        "model_class_name": "ZImagePipeline",
        "diffusion_offload_config": {"mode": mode, "components": ["dit", "text_encoder"]},
        "cache_backend": cache,
        "cache_config": {"rel_l1_thresh": 0.1},
        "quantization_config": {"method": "fp8", "ignored_layers": ["sensitive"]},
    }
    before = deepcopy(raw)
    expanded = expand_low_vram_preset(raw)
    assert expanded["cache_backend"] == cache
    assert expanded["diffusion_offload_config"]["mode"] == mode
    assert expanded["quantization_config"] == raw["quantization_config"]
    assert raw == before


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"low_vram_preset": "typo"}, "Unknown low_vram_preset"),
        ({"quantization_config": "gguf"}, "requires FP8"),
        ({"quantization_config": {"method": "int8"}}, "requires FP8"),
        ({"diffusion_offload_config": {"mode": "module", "components": ["dit"]}}, "components"),
        ({"enable_cpu_offload": True}, "legacy"),
        (
            {"diffusion_offload_config": {"mode": "module", "components": ["dit", "text_encoder"]}},
            "produced invalid images",
        ),
        ({"enable_distributed_layerwise_offload": True}, "legacy"),
        ({"vae_use_tiling": False}, "vae_use_tiling"),
        ({"cache_backend": "cache_dit"}, "cache_backend"),
        ({"cache_config": "{"}, "valid cache_config JSON"),
        ({"cache_config": "[]"}, "JSON object"),
        ({"diffusion_compile_granularity": "full"}, "regional"),
        ({"diffusion_load_format": "diffusers"}, "native diffusion loader"),
        ({"enable_sleep_mode": True}, "sleep mode"),
        ({"model_class_name": "QwenImagePipeline"}, "currently supports only"),
        ({"model_class_name": "WanPipeline"}, "currently supports only"),
        (
            {
                "diffusion_offload_config": {
                    "mode": "layer",
                    "components": ["dit", "text_encoder"],
                    "layer_options": {"dit": {"weight_transfer": "allgather"}},
                }
            },
            "rank-local",
        ),
    ],
)
def test_invalid_combinations_fail_loudly(overrides, message):
    with pytest.raises(ValueError, match=message):
        expand_low_vram_preset({"low_vram_preset": "fp8", **overrides})


def test_unset_preset_preserves_existing_configuration():
    raw = {"cache_backend": "cache_dit", "vae_use_tiling": False, "enable_sleep_mode": True}
    assert expand_low_vram_preset(raw) == raw


def test_late_architecture_resolution_rejects_unsupported_pipeline():
    validate_low_vram_pipeline(None)
    validate_low_vram_pipeline("ZImagePipeline")
    with pytest.raises(ValueError, match="currently supports only"):
        validate_low_vram_pipeline("WanPipeline")
    with pytest.raises(ValueError, match="currently supports only"):
        validate_low_vram_pipeline("DiffusersAdapterPipeline")


def test_null_values_receive_preset_defaults():
    expanded = expand_low_vram_preset(
        {
            "low_vram_preset": "fp8",
            "quantization_config": None,
            "diffusion_offload_config": None,
            "cache_backend": None,
            "vae_use_tiling": None,
        }
    )
    assert expanded["quantization_config"] == "fp8"
    assert expanded["vae_use_tiling"] is True


@pytest.mark.parametrize("model_class_name", sorted(_PIPELINES))
def test_supported_pipelines_declare_streamable_text_encoder(model_class_name):
    # The preset's default selects text_encoder explicitly in layer mode, which
    # the offload resolver rejects unless the pipeline declares its blocks.
    import importlib

    from vllm_omni.diffusion.registry import _DIFFUSION_MODELS

    package, module_name, class_name = _DIFFUSION_MODELS[model_class_name]
    module = importlib.import_module(f"vllm_omni.diffusion.models.{package}.{module_name}")
    plan = getattr(module, class_name)._offload_plan
    assert plan.encoder_block_attrs.get("text_encoder")
