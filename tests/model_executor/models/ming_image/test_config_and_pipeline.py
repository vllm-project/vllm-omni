# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.diffusion.data import OmniDiffusionConfig, resolve_model_class_name
from vllm_omni.diffusion.models.ming_image.pipeline import (
    MingImageDiffusionPipeline,
    _validate_variant_config,
)
from vllm_omni.diffusion.utils.hf_utils import is_diffusion_model
from vllm_omni.model_executor.models.ming_flash_omni.ming_flash_omni_thinker import (
    MingFlashOmniThinkerForConditionalGeneration,
)
from vllm_omni.model_executor.models.ming_image import checkpoint
from vllm_omni.model_executor.models.ming_image.model import MingImageForConditionalGeneration
from vllm_omni.model_executor.models.ming_image.pipeline import MING_IMAGE_PIPELINE
from vllm_omni.model_executor.stage_input_processors.ming_image import thinker2image
from vllm_omni.model_extras import (
    get_extra_body_params,
    should_init_extra_args_for_non_diffusion_stages,
)
from vllm_omni.transformers_utils.configs.ming_flash_omni import BailingMM2Config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_CONFIG_TEMPLATES = Path(__file__).parents[4] / "recipes" / "inclusionAI" / "ming-image-configs"


def test_config_selects_qwen25_vision_tower():
    config = BailingMM2Config(vision_config={"model_type": "qwen2_5_vit"}, llm_config={})

    assert type(config.vision_config).__name__ == "Qwen2_5_VLVisionConfig"


@pytest.mark.parametrize(
    ("model_name", "pipeline_class", "padding", "layered"),
    (
        (
            "Ming-Image-0.1-Design",
            "MingImageDiffusionPipeline",
            "zero_masked",
            False,
        ),
        (
            "Ming-Image-0.1-Design-Layer",
            "MingImageLayeredDiffusionPipeline",
            "learned",
            True,
        ),
    ),
)
def test_published_model_metadata_contract(
    model_name,
    pipeline_class,
    padding,
    layered,
):
    model_path = _CONFIG_TEMPLATES / model_name
    model_index = json.loads((model_path / "model_index.json").read_text())
    config = OmniDiffusionConfig(model=str(model_path))
    config.enrich_config()
    is_diffusion_model.cache_clear()

    assert model_index["_class_name"] == pipeline_class
    assert config.model_class_name == pipeline_class
    assert config.tf_model_config.axes_lens == [20480, 512, 512]
    assert config.tf_model_config.alignment_padding_mode == padding
    assert config.tf_model_config.multi_frame_output is layered
    assert StageConfigFactory.try_infer_model_type(str(model_path), False) == "ming_image"
    assert resolve_model_class_name(str(model_path)) == pipeline_class
    assert is_diffusion_model(str(model_path))
    assert _validate_variant_config(model_index, config.tf_model_config) is layered


@pytest.mark.parametrize(
    ("transformer_config", "layered"),
    (
        (
            SimpleNamespace(
                _class_name="DiffusionTransformer",
                alignment_padding_mode="zero_masked",
                multi_frame_output=False,
            ),
            False,
        ),
        (
            SimpleNamespace(
                _class_name="DiffusionTransformer",
                alignment_padding_mode="learned",
                multi_frame_output=True,
            ),
            True,
        ),
    ),
)
def test_variant_config_without_model_index(transformer_config, layered):
    assert _validate_variant_config(None, transformer_config) is layered


def test_variant_config_rejects_mixed_semantics():
    with pytest.raises(ValueError, match="must use either"):
        _validate_variant_config(
            {"_class_name": "MingImageDiffusionPipeline"},
            SimpleNamespace(
                _class_name="DiffusionTransformer",
                alignment_padding_mode="learned",
                multi_frame_output=False,
            ),
        )


def test_layer_latent_frames_flatten_once_in_frame_major_order():
    latents = torch.arange(2 * 4 * 3 * 2 * 2).reshape(2, 4, 3, 2, 2)

    pipeline = MingImageDiffusionPipeline.__new__(MingImageDiffusionPipeline)
    flat = pipeline._flatten_latent_frames(latents)

    assert flat.shape == (6, 4, 2, 2)
    assert torch.equal(flat[0], latents[0, :, 0])
    assert torch.equal(flat[1], latents[1, :, 0])
    assert torch.equal(flat[2], latents[0, :, 1])


def test_layer_pipeline_allows_missing_reference_only_for_dummy_run():
    pipeline = MingImageDiffusionPipeline.__new__(MingImageDiffusionPipeline)
    pipeline.is_layer_decomposition = True
    pipeline._configure_output_frames(
        reference=None,
        num_layers=1,
        is_dummy_run=True,
    )
    assert pipeline._num_frames_per_prompt == 2

    with pytest.raises(ValueError, match="requires a reference image"):
        pipeline._configure_output_frames(
            reference=None,
            num_layers=1,
            is_dummy_run=False,
        )


def _source_output(prefix_len: int = 4):
    prompt_ids = [11] * prefix_len + [157158] + [157157] * 256 + [157159]
    length = len(prompt_ids)
    multimodal = {
        "final_hidden_states": torch.arange(length * 2048).view(length, 2048),
    }
    for layer in (5, 12, 20):
        multimodal[f"hidden_states_{layer}"] = torch.full((length, 2048), layer)
    return SimpleNamespace(
        prompt_token_ids=prompt_ids,
        outputs=[SimpleNamespace(multimodal_output=multimodal)],
    )


def test_bridge_extracts_query_and_direct_conditions():
    result = thinker2image(
        [_source_output()],
        prompt={"multi_modal_data": {"img2img": "reference"}},
        sampling_params=SimpleNamespace(extra_args={"num_layers": 3}),
    )

    extra = result[0]["extra"]
    assert extra["query_hidden_states"].shape == (256, 2048)
    assert extra["direct_hidden_states"].shape == (4, 6144)
    assert extra["reference_image"] == "reference"
    assert extra["num_layers"] == 3
    assert set(extra["direct_hidden_states"][:, :2048].unique().tolist()) == {5}


def test_bridge_rejects_text_negative_condition():
    with pytest.raises(ValueError, match="does not accept negative_prompt"):
        thinker2image(
            [_source_output()],
            prompt={"negative_prompt": "bad"},
            sampling_params=SimpleNamespace(extra_args={}),
        )


def test_two_stage_topology_and_request_metadata():
    assert MING_IMAGE_PIPELINE.model_type == "ming_image"
    assert [stage.model_stage for stage in MING_IMAGE_PIPELINE.stages] == ["mllm", "dit"]
    assert MING_IMAGE_PIPELINE.diffusers_class_name == "MingImageDiffusionPipeline"
    assert MING_IMAGE_PIPELINE.diffusers_class_aliases == ("MingImageLayeredDiffusionPipeline",)
    assert MING_IMAGE_PIPELINE.stages[0].model_subdir == "mllm"
    assert MING_IMAGE_PIPELINE.stages[0].model_path_resolver.endswith(".resolve_ming_image_model_root")
    assert MING_IMAGE_PIPELINE.stages[1].model_arch == "MingImageDiffusionPipeline"
    assert MING_IMAGE_PIPELINE.hf_architectures == ("MingImageForConditionalGeneration",)
    for class_name in MING_IMAGE_PIPELINE.diffusers_class_aliases + (MING_IMAGE_PIPELINE.diffusers_class_name,):
        assert "num_layers" in get_extra_body_params(class_name)
        assert should_init_extra_args_for_non_diffusion_stages(class_name)


def test_stage1_compile_uses_static_regional_cuda_graph(monkeypatch):
    pipeline = MingImageDiffusionPipeline.__new__(MingImageDiffusionPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(
        diffusion_compile_granularity="regional",
        diffusion_compile_dynamic=False,
    )
    pipeline.transformer = torch.nn.Identity()
    captured = {}

    def _regionally_compile(model, **kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.ming_image.pipeline.regionally_compile",
        _regionally_compile,
    )

    pipeline.setup_compile()

    assert captured == {
        "mode": "reduce-overhead",
        "fullgraph": True,
        "dynamic": False,
    }


def test_explicit_pipeline_loads_component_config_without_model_index(tmp_path):
    transformer = tmp_path / "transformer"
    transformer.mkdir()
    (transformer / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "DiffusionTransformer",
                "alignment_padding_mode": "zero_masked",
                "multi_frame_output": False,
            }
        )
    )

    config = OmniDiffusionConfig(
        model=str(tmp_path),
        model_class_name="MingImageDiffusionPipeline",
    )
    config.enrich_config()

    assert config.model_class_name == "MingImageDiffusionPipeline"
    assert config.tf_model_config.alignment_padding_mode == "zero_masked"
    assert config.tf_model_config.multi_frame_output is False


def test_checkpoint_resolver_accepts_root_or_mllm_subfolder(tmp_path):
    root = tmp_path / "checkpoint"
    mllm = root / "mllm"
    mllm.mkdir(parents=True)
    (mllm / "config.json").write_text("{}")

    assert checkpoint.resolve_ming_image_model_root(str(root), None, None) == str(root)
    assert checkpoint.resolve_ming_image_model_root(str(mllm), None, None) == str(root)


def test_checkpoint_resolver_downloads_mllm_and_sibling_mlp(monkeypatch):
    captured = {}

    def _download(**kwargs):
        captured.update(kwargs)
        return "/cache/model"

    monkeypatch.setattr(checkpoint, "download_weights_from_hf_specific", _download)

    assert checkpoint.resolve_ming_image_model_root("org/model", "rev", None) == "/cache/model"
    assert captured["allow_patterns"] == ["mllm/**", "mlp/**"]
    assert captured["revision"] == "rev"
    assert captured["require_all"] is True


def test_compute_logits_accepts_vllm_v1_signature():
    class _LanguageModel(torch.nn.Module):
        def compute_logits(self, hidden_states, sampling_metadata):
            assert sampling_metadata is None
            return hidden_states + 1

    model = MingImageForConditionalGeneration.__new__(MingImageForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.language_model = _LanguageModel()

    hidden_states = torch.zeros(2, 3)
    assert torch.equal(model.compute_logits(hidden_states), hidden_states + 1)


def test_embedding_only_run_uses_empty_modality_masks():
    model = MingImageForConditionalGeneration.__new__(MingImageForConditionalGeneration)
    torch.nn.Module.__init__(model)
    image_mask, audio_mask = model._compute_modality_masks(None, torch.empty(7, 2048))

    assert image_mask.shape == (7,)
    assert not image_mask.any()
    assert torch.equal(audio_mask, image_mask)


def test_mllm_reports_root_mlp_query_tokens_as_loaded(monkeypatch):
    model = MingImageForConditionalGeneration.__new__(MingImageForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.query_tokens_dict = torch.nn.ParameterDict({"16x16": torch.nn.Parameter(torch.empty(1), requires_grad=False)})
    monkeypatch.setattr(
        MingFlashOmniThinkerForConditionalGeneration,
        "load_weights",
        lambda self, weights: {"language_model.weight"},
    )

    assert model.load_weights([]) == {
        "language_model.weight",
        "query_tokens_dict.16x16",
    }
