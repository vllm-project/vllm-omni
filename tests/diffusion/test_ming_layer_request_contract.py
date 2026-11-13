# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ming checkpoint discovery and the serving-to-layer-output request contract."""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from vllm.sampling_params import SamplingParams

from vllm_omni.diffusion.data import OmniDiffusionConfig, resolve_model_class_name
from vllm_omni.diffusion.models.ming_image.pipeline import MingImageDiffusionPipeline, _validate_variant_config
from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.entrypoints.openai.diffusion_request_utils import (
    apply_normalized_diffusion_request_extra_args,
    normalize_diffusion_request_args,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.glm_image.pipeline import GLM_IMAGE_PIPELINE
from vllm_omni.model_executor.models.ming_image.pipeline import MING_IMAGE_PIPELINE
from vllm_omni.model_executor.stage_input_processors.ming_image import thinker2image
from vllm_omni.model_extras import get_extra_body_params

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def ming_checkpoint_configs(tmp_path):
    # Design-Layer @ff1b22d has a root config; the older @9fabca8 does not.
    # Both image variants share Bailing metadata with Flash-Omni @c1ed556.
    roots = {}
    for variant, transformer in (
        (
            "layered",
            {"_class_name": "DiffusionTransformer", "alignment_padding_mode": "learned", "multi_frame_output": True},
        ),
        (
            "design",
            {
                "_class_name": "DiffusionTransformer",
                "alignment_padding_mode": "zero_masked",
                "multi_frame_output": False,
            },
        ),
        ("flash", {"_class_name": "ZImageTransformer2DModel"}),
    ):
        root = tmp_path / variant
        (root / "transformer").mkdir(parents=True)
        (root / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["BailingMM2NativeForConditionalGeneration"],
                    "model_type": "bailingmm_moe_v2_lite",
                }
            )
        )
        (root / "transformer/config.json").write_text(json.dumps(transformer))
        roots[variant] = str(root)
    return roots


@dataclass
class CpuCompletion:
    multimodal_output: dict[str, torch.Tensor]


@dataclass
class CpuThinkerOutput:
    prompt_token_ids: list[int]
    outputs: list[CpuCompletion]


def source_output():
    prompt_ids = [11] * 4 + [157158] + [157157] * 256 + [157159]
    hidden = torch.arange(len(prompt_ids) * 2).view(len(prompt_ids), 2)
    payload = {"final_hidden_states": hidden}
    payload.update({f"hidden_states_{layer}": hidden + layer for layer in (5, 12, 20)})
    return CpuThinkerOutput(prompt_ids, [CpuCompletion(payload)])


def frontend_engine(model, pipeline_config=MING_IMAGE_PIPELINE):
    engine = object.__new__(AsyncOmniEngine)
    engine.model = model
    engine.pipeline_config = pipeline_config
    engine._diffusion_od_config_view = None
    return engine


@pytest.mark.parametrize(
    ("variant", "expected"),
    (
        ("layered", "MingImageDiffusionPipeline"),
        ("design", "MingImageDiffusionPipeline"),
        ("flash", "MingImagePipeline"),
    ),
)
def test_checkpoint_resolves_correct_pipeline(ming_checkpoint_configs, variant, expected):
    assert resolve_model_class_name(ming_checkpoint_configs[variant]) == expected


@pytest.mark.parametrize("num_layers", (1, 3, 6))
@pytest.mark.parametrize("request_scope", ("root", "nested"))
def test_request_preserves_layers_through_normalization_bridge_and_output(
    ming_checkpoint_configs, num_layers, request_scope
):
    engine = frontend_engine(ming_checkpoint_configs["layered"])
    od_config = engine.get_diffusion_od_config()
    declared = get_extra_body_params(od_config.model_class_name)
    request = {"num_layers": num_layers, "height": 1024, "width": 1024, "seed": 42}
    normalized, request_args = normalize_diffusion_request_args(
        **{request_scope: request},
        registered_extra_fields=declared,
        serving_root_fields={"height", "width", "seed"},
    )
    ar_sampling = SamplingParams(extra_args={})
    apply_declared_extra_args(ar_sampling, declared, request_args)
    dit_sampling = OmniDiffusionSamplingParams(extra_args={})
    apply_normalized_diffusion_request_extra_args(dit_sampling, normalized)
    result = thinker2image(
        [source_output()], prompt={"multi_modal_data": {"img2img": "reference"}}, sampling_params=ar_sampling
    )
    assert ar_sampling.extra_args["num_layers"] == num_layers
    assert dit_sampling.extra_args["num_layers"] == num_layers
    extra = result[0]["extra"]
    assert extra["num_layers"] == num_layers
    assert extra["query_hidden_states"].shape == (256, 2)
    config = OmniDiffusionConfig(model=engine.model, model_class_name=od_config.model_class_name)
    config.enrich_config()
    pipeline = MingImageDiffusionPipeline.__new__(MingImageDiffusionPipeline)
    pipeline.is_layer_decomposition = _validate_variant_config(None, config.tf_model_config)
    pipeline._configure_output_frames(
        reference=extra["reference_image"],
        num_layers=dit_sampling.extra_args.get("num_layers", extra["num_layers"]),
        is_dummy_run=False,
    )
    assert pipeline._num_frames_per_prompt == num_layers + 1
    assert engine.get_diffusion_od_config() is od_config


@pytest.mark.parametrize("variant", ("layered", "design"))
@pytest.mark.parametrize("explicit", (None, "MingImageDiffusionPipeline", "MingImageLayeredDiffusionPipeline"))
def test_worker_loads_variant_component_config(ming_checkpoint_configs, variant, explicit):
    config = OmniDiffusionConfig(model=ming_checkpoint_configs[variant], model_class_name=explicit)
    config.enrich_config()
    assert config.model_class_name == (explicit or "MingImageDiffusionPipeline")
    assert config.tf_model_config.multi_frame_output is (variant == "layered")
    assert _validate_variant_config(None, config.tf_model_config) is (variant == "layered")
    assert not hasattr(config.tf_model_config, "architectures")


def test_design_keeps_single_frame_contract(ming_checkpoint_configs):
    config = OmniDiffusionConfig(model=ming_checkpoint_configs["design"])
    config.enrich_config()
    pipeline = MingImageDiffusionPipeline.__new__(MingImageDiffusionPipeline)
    pipeline.is_layer_decomposition = _validate_variant_config(None, config.tf_model_config)
    pipeline._configure_output_frames(reference=None, num_layers=1, is_dummy_run=False)
    assert pipeline._num_frames_per_prompt == 1
    with pytest.raises(ValueError, match="exactly one output frame"):
        pipeline._configure_output_frames(reference=None, num_layers=3, is_dummy_run=False)


def test_flash_worker_keeps_zimage_class(ming_checkpoint_configs):
    config = OmniDiffusionConfig(model=ming_checkpoint_configs["flash"])
    config.enrich_config()
    assert config.model_class_name == "MingImagePipeline"
    assert config.tf_model_config.to_dict() == {}
    assert "num_layers" not in get_extra_body_params(config.model_class_name)
    assert {"steps", "cfg", "byte5_text"} <= get_extra_body_params(config.model_class_name)


@pytest.mark.parametrize("contents", ("{invalid json", '["not a config object"]', '"not an object"', "null", ""))
def test_invalid_optional_transformer_metadata_keeps_flash_fallback(ming_checkpoint_configs, contents):
    root = Path(ming_checkpoint_configs["flash"])
    (root / "transformer/config.json").write_text(contents)
    assert resolve_model_class_name(str(root)) == "MingImagePipeline"


def test_missing_optional_transformer_metadata_keeps_flash_fallback(ming_checkpoint_configs):
    root = Path(ming_checkpoint_configs["flash"])
    (root / "transformer/config.json").unlink()
    assert resolve_model_class_name(str(root)) == "MingImagePipeline"


@pytest.mark.parametrize(
    "error", (OSError("unreadable config"), ValueError("invalid config"), TypeError("invalid data"))
)
def test_optional_transformer_probe_handles_read_errors(ming_checkpoint_configs, monkeypatch, error):
    from vllm.transformers_utils import config as hf_config

    read_config = hf_config.get_hf_file_to_dict

    def read_with_component_error(filename, model, revision=None):
        if filename == "transformer/config.json":
            raise error
        return read_config(filename, model, revision=revision)

    monkeypatch.setattr(hf_config, "get_hf_file_to_dict", read_with_component_error)
    assert resolve_model_class_name(ming_checkpoint_configs["flash"]) == "MingImagePipeline"


@pytest.mark.parametrize("pipeline_config", (MING_IMAGE_PIPELINE, GLM_IMAGE_PIPELINE))
def test_frontend_prefers_registered_topology_without_checkpoint_probe(
    ming_checkpoint_configs, monkeypatch, pipeline_config
):
    def unexpected_probe(*args, **kwargs):
        pytest.fail("A resolved topology must not be replaced by checkpoint discovery")

    monkeypatch.setattr("vllm_omni.diffusion.data.resolve_model_class_name", unexpected_probe)
    engine = frontend_engine(ming_checkpoint_configs["flash"], pipeline_config)
    assert engine.get_diffusion_od_config().model_class_name == pipeline_config.diffusers_class_name


def test_frontend_discovers_checkpoint_when_topology_has_no_class(ming_checkpoint_configs):
    engine = frontend_engine(ming_checkpoint_configs["layered"], pipeline_config=None)
    assert engine.get_diffusion_od_config().model_class_name == "MingImageDiffusionPipeline"


def test_legacy_checkpoint_without_root_keeps_explicit_class(ming_checkpoint_configs):
    root = ming_checkpoint_configs["layered"]
    (Path(root) / "config.json").unlink()
    assert frontend_engine(root).get_diffusion_od_config().model_class_name == "MingImageDiffusionPipeline"
    config = OmniDiffusionConfig(model=root, model_class_name="MingImageDiffusionPipeline")
    config.enrich_config()
    assert config.model_class_name == "MingImageDiffusionPipeline"
    assert config.tf_model_config.multi_frame_output is True


def test_missing_component_clears_root_transformer_config(ming_checkpoint_configs):
    root = Path(ming_checkpoint_configs["layered"])
    (root / "transformer/config.json").unlink()
    config = OmniDiffusionConfig(model=str(root), model_class_name="MingImageDiffusionPipeline")
    config.enrich_config()
    assert config.tf_model_config.to_dict() == {}
