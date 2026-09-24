# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from vllm.multimodal.processing import ProcessorInputs

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.ming_image.pipeline import (
    MingImageDiffusionPipeline,
    _validate_variant_config,
)
from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.ming_flash_omni.ming_flash_omni_thinker import (
    MingFlashOmniThinkerForConditionalGeneration,
    MingFlashOmniThinkerMultiModalProcessor,
)
from vllm_omni.model_executor.models.ming_image import checkpoint
from vllm_omni.model_executor.models.ming_image.model import (
    MingImageForConditionalGeneration,
    MingImageMultiModalProcessor,
)
from vllm_omni.model_executor.models.ming_image.pipeline import MING_IMAGE_PIPELINE
from vllm_omni.model_executor.stage_input_processors.ming_image import thinker2image
from vllm_omni.model_extras import (
    get_extra_body_params,
    should_init_extra_args_for_non_diffusion_stages,
)
from vllm_omni.transformers_utils.configs.ming_flash_omni import BailingMM2Config
from vllm_omni.transformers_utils.processors.ming import MingImageProcessor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_config_selects_qwen25_vision_tower():
    config = BailingMM2Config(vision_config={"model_type": "qwen2_5_vit"}, llm_config={})

    assert type(config.vision_config).__name__ == "Qwen2_5_VLVisionConfig"


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


def test_pipeline_rejects_multiple_outputs_per_prompt():
    pipeline = MingImageDiffusionPipeline.__new__(MingImageDiffusionPipeline)
    request = DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                prompt="test",
                sampling_params=OmniDiffusionSamplingParams(num_outputs_per_prompt=2),
                request_id="test-request",
            )
        ]
    )

    with pytest.raises(ValueError, match="num_outputs_per_prompt=1 only, got 2"):
        pipeline.forward(request)


@pytest.mark.parametrize(
    ("guidance_scale", "expected"),
    ((0.0, False), (1.0, False), (1.0001, True), (2.0, True)),
)
def test_cfg_is_only_enabled_above_one(guidance_scale, expected):
    pipeline = ZImagePipeline.__new__(ZImagePipeline)
    pipeline._guidance_scale = guidance_scale

    assert pipeline.do_classifier_free_guidance is expected


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


def test_image_generation_template_matches_upstream_layout():
    processor = MingImageProcessor.__new__(MingImageProcessor)
    processor.tokenizer = SimpleNamespace(eos_token="<eos>")

    text = processor._apply_image_generation_template("draw a cabin")
    assert text == (
        "<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off"
        "<eos><role>HUMAN</role>draw a cabin"
        "<eos><role>ASSISTANT</role>"
    )
    assert processor._apply_image_generation_template(text) == text
    assert "<role>HUMAN</role><IMAGE>edit this image<eos>" in processor._apply_image_generation_template(
        "edit this image",
        has_reference_image=True,
    )


def test_multimodal_processor_applies_generation_template(monkeypatch: pytest.MonkeyPatch):
    template_calls = []

    def _format(prompt, *, has_reference_image):
        template_calls.append((prompt, has_reference_image))
        return "formatted prompt"

    tokenizer = SimpleNamespace(
        decode=lambda token_ids, **kwargs: "edit this image",
        encode=lambda text, **kwargs: [101, 102],
    )
    processor = MingImageMultiModalProcessor.__new__(MingImageMultiModalProcessor)
    processor.info = SimpleNamespace(
        get_tokenizer=lambda: tokenizer,
        get_hf_processor=lambda: SimpleNamespace(_apply_image_generation_template=_format),
    )
    captured = {}

    def _parent_apply(_self, inputs, timing_ctx):
        captured["inputs"] = inputs
        return "processed"

    monkeypatch.setattr(MingFlashOmniThinkerMultiModalProcessor, "apply", _parent_apply)
    inputs = ProcessorInputs(
        prompt=[11, 12],
        mm_data_items=None,
        hf_processor_mm_kwargs={"modalities": ["img2img"]},
    )

    assert processor.apply(inputs, None) == "processed"
    assert template_calls == [("edit this image", True)]
    assert captured["inputs"].prompt == [101, 102]
    assert captured["inputs"].hf_processor_mm_kwargs["modalities"] == ["image"]


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
        assert "steps" not in get_extra_body_params(class_name)
        assert "cfg" not in get_extra_body_params(class_name)
        assert should_init_extra_args_for_non_diffusion_stages(class_name)


def test_stage1_compile_uses_configured_dynamic_regional_cuda_graph(monkeypatch):
    pipeline = MingImageDiffusionPipeline.__new__(MingImageDiffusionPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(
        diffusion_compile_granularity="regional",
        diffusion_compile_dynamic=True,
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
        "dynamic": True,
    }
    assert pipeline._uses_cudagraph_trees


def test_explicit_pipeline_loads_component_config_without_model_index(monkeypatch):
    transformer_config = {
        "_class_name": "DiffusionTransformer",
        "alignment_padding_mode": "zero_masked",
        "multi_frame_output": False,
    }

    def _get_hf_file_to_dict(filename, model, revision=None):
        del model, revision
        return transformer_config if filename == "transformer/config.json" else None

    monkeypatch.setattr(
        "vllm_omni.diffusion.utils.hf_utils.get_diffusion_model_index",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "vllm.transformers_utils.config.get_hf_file_to_dict",
        _get_hf_file_to_dict,
    )

    config = OmniDiffusionConfig(
        model="org/model",
        model_class_name="MingImageDiffusionPipeline",
    )
    config.enrich_config()

    assert config.model_class_name == "MingImageDiffusionPipeline"
    assert config.tf_model_config.alignment_padding_mode == "zero_masked"
    assert config.tf_model_config.multi_frame_output is False


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
