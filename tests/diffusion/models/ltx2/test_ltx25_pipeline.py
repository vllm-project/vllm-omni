# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Focused LTX-2.5 pipeline correctness tests."""

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.ltx2 import ltx2_components, ltx2_latents, ltx2_runtime
from vllm_omni.diffusion.models.ltx2.ltx2_components import (
    LTX25_DISTILLED_COMPONENT_PROFILE,
    LTX25_DISTILLED_ONE_STAGE_COMPONENT_PROFILE,
    LTX25_FULL_COMPONENT_PROFILE,
    LTX25_TWO_STAGE_COMPONENT_PROFILE,
    detect_ltx_model_version,
    resolve_ltx_checkpoint_kind,
    resolve_ltx_component_profile,
)
from vllm_omni.diffusion.models.ltx2.ltx2_denoise import (
    LTXPhaseResult,
    _first_frame_keyframes_mask,
    _official_ltx_sigmas,
)
from vllm_omni.diffusion.models.ltx2.ltx2_guidance import (
    LTX_GUIDANCE_EXECUTOR,
    LTXGuidanceSpec,
    LTXModalityGuidance,
    velocity_from_x0,
    x0_from_velocity,
)
from vllm_omni.diffusion.models.ltx2.ltx2_latents import LTXAVState
from vllm_omni.diffusion.models.ltx2.ltx2_recipes import (
    LTX2_DISTILLED_TWO_STAGE_RECIPE,
    LTX2_ONE_STAGE_RECIPE,
    LTX23_ONE_STAGE_RECIPE,
    LTX25_DISTILLED_ONE_STAGE_RECIPE,
    LTX25_DISTILLED_TWO_STAGE_RECIPE,
    LTX25_FULL_RECIPE,
    LTX25_TWO_STAGE_RECIPE,
    LTX_DISTILLED_SIGMAS,
    LTX_POSITIVE_ONLY_RECIPE,
    resolve_ltx_pipeline_recipe,
)
from vllm_omni.diffusion.models.ltx2.ltx2_request import LTXRequestInputs
from vllm_omni.diffusion.models.ltx2.ltx2_runtime import LTXRuntime
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import (
    LTX2DistilledOneStagePipeline,
    LTX2Pipeline,
)
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_two_stage import (
    LTX2DistilledTwoStagePipeline,
    LTX2TwoStagePipeline,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("model_version", ["2", "2.3", "2.5"])
def test_ltx_timestep_adaln_expands_per_token_like_official_without_sp(model_version):
    pipe = object.__new__(LTX2Pipeline)
    pipe.model_version = model_version
    pipe.od_config = SimpleNamespace(parallel_config=SimpleNamespace(sequence_parallel_size=1))
    ts = torch.tensor([750.0, 500.0])

    kwargs = LTXRuntime._denoise_timestep_kwargs(
        pipe,
        ts,
        SimpleNamespace(),
        SimpleNamespace(),
        video_token_count=4,
        audio_token_count=3,
    )

    torch.testing.assert_close(kwargs["timestep"], ts[:, None].expand(-1, 4))
    torch.testing.assert_close(kwargs["audio_timestep"], ts[:, None].expand(-1, 3))
    assert kwargs["sigma"] is ts
    assert kwargs["audio_sigma"] is ts


def test_ltx_ancestral_step_clears_sp_audio_padding_after_scheduler_update(monkeypatch):
    state = LTXAVState(
        video=torch.zeros(1, 2, 2),
        audio=torch.zeros(1, 4, 2),
    )
    forward_ctx = SimpleNamespace(
        original_audio_num_frames=3,
        sampler="euler_ancestral",
    )
    denoise_ctx = SimpleNamespace(latents=None, audio_latents=None)
    pipe = SimpleNamespace(
        _predict_noise_for_step=lambda *_args: (torch.zeros_like(state.video), torch.zeros_like(state.audio))
    )

    def fake_step(_pipeline, actual_forward_ctx, _denoise_ctx, *_args):
        assert actual_forward_ctx.sampler == "euler_ancestral"
        updated_audio = torch.ones_like(state.audio)
        updated_audio[:, 3:] = 99.0
        return torch.ones_like(state.video), updated_audio

    monkeypatch.setattr(ltx2_runtime, "step_denoised_latents", fake_step)

    actual = LTXRuntime._denoise_step(
        pipe,
        0,
        torch.tensor(1.0),
        state,
        forward_ctx,
        denoise_ctx,
    )

    torch.testing.assert_close(actual.audio[:, :3], torch.ones_like(actual.audio[:, :3]))
    torch.testing.assert_close(actual.audio[:, 3:], torch.zeros_like(actual.audio[:, 3:]))


def test_ltx25_first_causal_latent_frame_is_marked_for_keyframe_embedding():
    reference = torch.zeros(2, 12, 8)

    mask = _first_frame_keyframes_mask(reference, latent_num_frames=3)

    assert mask.shape == (2, 12, 1)
    torch.testing.assert_close(mask[:, :4], torch.ones(2, 4, 1))
    torch.testing.assert_close(mask[:, 4:], torch.zeros(2, 8, 1))


def _make_ltx_request_pipe(cls):
    pipe = object.__new__(cls)
    torch.nn.Module.__init__(pipe)
    pipe.device = torch.device("cpu")
    pipe.tokenizer_max_length = 99
    pipe.vae_spatial_compression_ratio = 32
    pipe.vae_temporal_compression_ratio = 8
    return pipe


def _resolve_request_inputs_for_test(
    pipe,
    req,
    *,
    guidance_scale=4.0,
    negative_prompt=None,
):
    return pipe._resolve_request_inputs(
        req,
        prompt=None,
        negative_prompt=negative_prompt,
        height=None,
        width=None,
        num_frames=None,
        frame_rate=None,
        num_inference_steps=None,
        guidance_scale=guidance_scale,
        num_videos_per_prompt=1,
        generator=None,
        latents=None,
        audio_latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        decode_timestep=0.0,
        decode_noise_scale=None,
        output_type="np",
        max_sequence_length=None,
    )


def test_ltx25_missing_gemma4_recommends_supported_transformers_range(monkeypatch):
    pipe = SimpleNamespace(
        component_profile=replace(LTX25_FULL_COMPONENT_PROFILE, text_encoder_cls=None),
        use_diffusion_decoder=True,
    )
    od_config = SimpleNamespace(model="Lightricks/LTX-2.5-Diffusers", dtype=torch.bfloat16)

    monkeypatch.setattr(ltx2_components, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(ltx2_components, "prefetch_subfolders", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ltx2_components.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: object())

    with pytest.raises(ImportError) as exc_info:
        ltx2_components.initialize_pipeline_components(pipe, od_config)

    assert str(exc_info.value) == (
        "LTX-2.5 requires Gemma4UnifiedForConditionalGeneration; install transformers>=5.10.1,<5.15."
    )


def test_ltx_converted_component_loading_propagates_revision(monkeypatch):
    revision = "pinned-revision"
    calls: dict[str, Any] = {"components": []}
    natten_processor = object()

    class FakeDiffusionDecoder:
        def set_attn_processor(self, processor):
            calls["diffusion_decoder_processor"] = processor

        def enable_tiling(self):
            calls["diffusion_decoder_tiling"] = True

        def set_parallel_size(self, parallel_size, mode):
            calls["diffusion_decoder_parallel"] = (parallel_size, mode)

    profile = replace(
        LTX25_DISTILLED_COMPONENT_PROFILE,
        text_encoder_cls=object,
        video_vae_cls=object,
        vocoder_cls=object,
        vocoder_fallback_cls=None,
        scheduler_use_dynamic_shifting=False,
    )
    pipeline = SimpleNamespace(
        component_profile=profile,
        pipeline_kind="distilled_two_stage",
        use_diffusion_decoder=True,
    )
    od_config = SimpleNamespace(
        model="org/converted-ltx25",
        revision=revision,
        dtype=torch.bfloat16,
        quantization_config=None,
        extras={},
        vae_use_tiling=False,
        parallel_config=SimpleNamespace(vae_patch_parallel_size=2, vae_parallel_mode="tile"),
    )

    monkeypatch.setattr(ltx2_components, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(ltx2_components, "_install_connector_attention", lambda _connectors, **_kwargs: None)
    monkeypatch.setattr(ltx2_components, "_place_aux_components", lambda _pipeline: None)

    def fake_prefetch(model, subfolders, **kwargs):
        calls["prefetch"] = (model, tuple(subfolders), kwargs)

    def fake_tokenizer(*args, **kwargs):
        calls["tokenizer"] = (args, kwargs)
        return SimpleNamespace(model_max_length=1024)

    def fake_component(component_cls, model, subfolder, **kwargs):
        calls["components"].append((component_cls, model, subfolder, kwargs))
        if subfolder == "vae":
            return SimpleNamespace(spatial_compression_ratio=32, temporal_compression_ratio=8)
        if subfolder == "audio_vae":
            return SimpleNamespace(
                mel_compression_ratio=4,
                temporal_compression_ratio=4,
                config=SimpleNamespace(sample_rate=16_000, mel_hop_length=160),
            )
        return object()

    def fake_native_diffusion_decoder(model, **kwargs):
        calls["native_diffusion_decoder"] = (model, kwargs)
        return FakeDiffusionDecoder()

    def fake_transformer_config(model, subfolder, local_files_only, *, revision):
        calls["transformer_config"] = (model, subfolder, local_files_only, revision)
        return {"component": "transformer"}

    monkeypatch.setattr(ltx2_components, "prefetch_subfolders", fake_prefetch)
    monkeypatch.setattr(ltx2_components.AutoTokenizer, "from_pretrained", fake_tokenizer)
    monkeypatch.setattr(ltx2_components, "_load_component", fake_component)
    monkeypatch.setattr(ltx2_components, "_load_ltx25_native_diffusion_decoder", fake_native_diffusion_decoder)
    monkeypatch.setattr(ltx2_components, "load_transformer_config", fake_transformer_config)
    monkeypatch.setattr(
        ltx2_components,
        "LTX2VideoVaeNeighborhoodNattenProcessor",
        lambda: natten_processor,
    )
    monkeypatch.setattr(
        ltx2_components,
        "create_transformer_from_config",
        lambda *_args, **_kwargs: SimpleNamespace(config=SimpleNamespace(patch_size=1, patch_size_t=1)),
    )

    def fake_scheduler(*args, **kwargs):
        calls["scheduler"] = (args, kwargs)
        return SimpleNamespace(config={"use_dynamic_shifting": False, "shift_terminal": None})

    monkeypatch.setattr(ltx2_components.FlowMatchEulerDiscreteScheduler, "from_pretrained", fake_scheduler)

    ltx2_components.initialize_pipeline_components(pipeline, od_config)

    assert pipeline.weights_sources[0].revision == revision
    assert calls["prefetch"][2]["revision"] == revision
    assert calls["tokenizer"][1]["revision"] == revision
    assert {call[2] for call in calls["components"]} == {
        "text_encoder",
        "connectors",
        "vae",
        "audio_vae",
        "vocoder",
        "latent_upsampler",
    }
    assert all(call[3]["revision"] == revision for call in calls["components"])
    assert "diffusion_decoder" not in calls["prefetch"][1]
    assert calls["native_diffusion_decoder"] == (
        od_config.model,
        {"local_files_only": False, "dtype": torch.bfloat16, "revision": revision},
    )
    assert calls["diffusion_decoder_processor"] is natten_processor
    assert calls["diffusion_decoder_tiling"] is True
    assert calls["diffusion_decoder_parallel"] == (2, "tile")
    assert calls["transformer_config"] == (
        od_config.model,
        profile.transformer_subfolder,
        False,
        revision,
    )
    assert calls["scheduler"][1]["revision"] == revision


def test_ltx25_native_diffusion_decoder_uses_diffusers_config_and_canonical_artifact(monkeypatch):
    revision = "diffusers-revision"
    config = {"decoder_stage_channels": [8, 8, 8, 8, 8]}
    calls: dict[str, Any] = {}

    class FakeDecoder:
        def init_distributed(self):
            calls["init_distributed"] = True

    expected = FakeDecoder()

    def fake_load_config(model, **kwargs):
        calls["config"] = (model, kwargs)
        return config

    def fake_resolve(model, repo_id, filename, **kwargs):
        calls["artifact"] = (model, repo_id, filename, kwargs)
        return "/models/native-diffvae.safetensors"

    def fake_load(cls, path, passed_config, dtype):
        calls["load"] = (path, passed_config, dtype)
        return expected

    monkeypatch.setattr(
        ltx2_components.DistributedLTX2VideoDiffusionDecoderModel,
        "load_config",
        staticmethod(fake_load_config),
    )
    monkeypatch.setattr(ltx2_components, "resolve_ltx_artifact", fake_resolve)
    monkeypatch.setattr(
        ltx2_components.DistributedLTX2VideoDiffusionDecoderModel,
        "from_ltx25_native_checkpoint",
        classmethod(fake_load),
    )

    actual = ltx2_components._load_ltx25_native_diffusion_decoder(
        "Lightricks/LTX-2.5-Diffusers",
        local_files_only=False,
        dtype=torch.bfloat16,
        revision=revision,
    )

    assert actual is expected
    assert calls["config"] == (
        "Lightricks/LTX-2.5-Diffusers",
        {
            "subfolder": "diffusion_decoder",
            "local_files_only": False,
            "revision": revision,
        },
    )
    assert calls["artifact"] == (
        "Lightricks/LTX-2.5-Diffusers",
        "Lightricks/LTX-2.5",
        "vae/ltx-2.5-video-vae-bf16.safetensors",
        {
            "model_revision": revision,
            "artifact_revision": ltx2_components.LTX25_NATIVE_ARTIFACT_REVISION,
        },
    )
    assert calls["load"] == ("/models/native-diffvae.safetensors", config, torch.bfloat16)
    assert calls["init_distributed"] is True


def test_ltx25_natten_failure_has_actionable_omni_remedy(monkeypatch):
    class MissingNattenProcessor:
        def __init__(self):
            raise ImportError("kernels is unavailable")

    monkeypatch.setattr(
        ltx2_components,
        "LTX2VideoVaeNeighborhoodNattenProcessor",
        MissingNattenProcessor,
    )

    with pytest.raises(RuntimeError, match=r"kernels==0\.14\.1.*supported GPU.*allow Hub access"):
        ltx2_components._create_ltx25_natten_processor()


def test_ltx_checkpoint_explicit_version_precedes_structural_heuristics(tmp_path):
    (tmp_path / "model_index.json").write_text(json.dumps({"model_version": "2.3.1"}))
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text(json.dumps({"ff_bias": False}))

    assert detect_ltx_model_version(str(tmp_path)) == "2.3"


def test_ltx25_checkpoint_selects_full_one_stage_profile(tmp_path, monkeypatch):
    from vllm_omni.diffusion.models.ltx2 import ltx2_runtime

    (tmp_path / "model_index.json").write_text(
        json.dumps({"text_encoder": ["transformers", "Gemma4UnifiedForConditionalGeneration"]})
    )

    def stub_components(pipe, od_config):
        pipe.od_config = od_config
        pipe.vae_spatial_compression_ratio = 32

    monkeypatch.setattr(ltx2_runtime, "initialize_pipeline_components", stub_components)
    monkeypatch.setattr(LTXRuntime, "setup_diffusion_pipeline_profiler", lambda *_args, **_kwargs: None)

    pipe = LTX2Pipeline(od_config=SimpleNamespace(model=str(tmp_path), enable_diffusion_pipeline_profiler=False))

    assert pipe.model_version == "2.5"
    assert pipe.component_profile is LTX25_FULL_COMPONENT_PROFILE
    assert pipe.pipeline_recipe is LTX25_FULL_RECIPE
    assert pipe.preserve_sp_padded_audio_duration
    assert pipe.reports_stage_durations


def test_ltx25_full_recipe_matches_official_sft_defaults():
    (phase,) = LTX25_FULL_RECIPE.phases

    assert (LTX25_FULL_RECIPE.height, LTX25_FULL_RECIPE.width) == (544, 960)
    assert LTX25_FULL_RECIPE.num_frames == 121
    assert LTX25_FULL_RECIPE.frame_rate == 24.0
    assert LTX25_FULL_RECIPE.num_inference_steps == 30
    assert phase.guidance.video == LTXModalityGuidance(
        cfg_scale=3.0,
        stg_scale=1.0,
        modality_scale=3.0,
        rescale_scale=0.7,
        stg_blocks=(28,),
    )
    assert phase.guidance.audio == LTXModalityGuidance(
        cfg_scale=7.0,
        stg_scale=1.0,
        modality_scale=3.0,
        rescale_scale=0.7,
        stg_blocks=(28,),
    )
    assert phase.sigmas is None
    assert phase.noise_scale == 1.0
    assert phase.use_official_sigma_schedule
    assert LTX25_FULL_COMPONENT_PROFILE.transformer_subfolder == "transformer_full"
    assert LTX25_FULL_COMPONENT_PROFILE.scheduler_use_dynamic_shifting
    assert LTX25_FULL_COMPONENT_PROFILE.scheduler_shift_terminal == 0.1


def test_ltx25_full_sigma_schedule_uses_official_default_sequence_anchor():
    scheduler = SimpleNamespace(
        config={
            "base_image_seq_len": 1024,
            "max_image_seq_len": 4096,
            "base_shift": 0.95,
            "max_shift": 2.05,
            "shift_terminal": 0.1,
        }
    )

    sigmas = _official_ltx_sigmas(
        scheduler,
        steps=30,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(
        sigmas[[0, 1, -2, -1]],
        torch.tensor([1.0, 0.9949570, 0.1, 0.0]),
        rtol=1e-5,
        atol=1e-6,
    )


def test_ltx25_checkpoint_selects_distilled_two_stage_profile(tmp_path, monkeypatch):
    from vllm_omni.diffusion.models.ltx2 import ltx2_runtime

    (tmp_path / "model_index.json").write_text(
        json.dumps({"text_encoder": ["transformers", "Gemma4UnifiedForConditionalGeneration"]})
    )

    def stub_components(pipe, od_config):
        pipe.od_config = od_config
        pipe.vae_spatial_compression_ratio = 32

    monkeypatch.setattr(ltx2_runtime, "initialize_pipeline_components", stub_components)
    monkeypatch.setattr(LTXRuntime, "setup_diffusion_pipeline_profiler", lambda *_args, **_kwargs: None)

    pipe = LTX2DistilledTwoStagePipeline(
        od_config=SimpleNamespace(
            model=str(tmp_path),
            enable_diffusion_pipeline_profiler=False,
        )
    )

    assert pipe.component_profile is LTX25_DISTILLED_COMPONENT_PROFILE
    assert pipe.pipeline_recipe is LTX25_DISTILLED_TWO_STAGE_RECIPE


def test_ltx25_full_config_enrichment_uses_selected_transformer_subfolder(tmp_path):
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "LTX2Pipeline", "model_version": "2.5"}))
    for subfolder, marker in (("transformer", "distilled"), ("transformer_full", "full")):
        path = tmp_path / subfolder
        path.mkdir()
        (path / "config.json").write_text(json.dumps({"marker": marker}))

    config = OmniDiffusionConfig(model=str(tmp_path), model_class_name="LTX2Pipeline")
    config.enrich_config()

    assert config.tf_model_config.get("marker") == "full"


def test_ltx25_four_public_pipeline_semantics_are_disjoint():
    assert resolve_ltx_component_profile("one_stage", "2.5") is LTX25_FULL_COMPONENT_PROFILE
    assert resolve_ltx_pipeline_recipe("one_stage", "2.5") is LTX25_FULL_RECIPE
    assert resolve_ltx_component_profile("two_stage", "2.5") is LTX25_TWO_STAGE_COMPONENT_PROFILE
    assert resolve_ltx_pipeline_recipe("two_stage", "2.5") is LTX25_TWO_STAGE_RECIPE
    assert resolve_ltx_component_profile("distilled_one_stage", "2.5") is LTX25_DISTILLED_ONE_STAGE_COMPONENT_PROFILE
    assert resolve_ltx_pipeline_recipe("distilled_one_stage", "2.5") is LTX25_DISTILLED_ONE_STAGE_RECIPE
    assert resolve_ltx_component_profile("distilled_two_stage", "2.5") is LTX25_DISTILLED_COMPONENT_PROFILE
    assert resolve_ltx_pipeline_recipe("distilled_two_stage", "2.5") is LTX25_DISTILLED_TWO_STAGE_RECIPE

    assert LTX2Pipeline.pipeline_kind == "one_stage"
    assert LTX2TwoStagePipeline.pipeline_kind == "two_stage"
    assert LTX2DistilledOneStagePipeline.pipeline_kind == "distilled_one_stage"
    assert LTX2DistilledTwoStagePipeline.pipeline_kind == "distilled_two_stage"


@pytest.mark.parametrize(
    "pipeline_cls",
    [
        LTX2Pipeline,
        LTX2TwoStagePipeline,
        LTX2DistilledOneStagePipeline,
        LTX2DistilledTwoStagePipeline,
    ],
)
@pytest.mark.parametrize(
    ("extras", "use_diffusion_decoder"),
    [({}, True), ({"ltx2_use_conv_vae": True}, False)],
)
def test_ltx25_all_public_pipelines_default_to_diffvae_with_conv_opt_in(
    tmp_path, monkeypatch, pipeline_cls, extras, use_diffusion_decoder
):
    (tmp_path / "model_index.json").write_text(
        json.dumps({"text_encoder": ["transformers", "Gemma4UnifiedForConditionalGeneration"]})
    )

    def stub_components(pipe, od_config):
        pipe.od_config = od_config
        pipe.vae_spatial_compression_ratio = 32

    monkeypatch.setattr(ltx2_runtime, "initialize_pipeline_components", stub_components)
    monkeypatch.setattr(ltx2_runtime, "build_ltx_phase_adapter", lambda _pipe: None)
    monkeypatch.setattr(LTXRuntime, "setup_diffusion_pipeline_profiler", lambda *_args, **_kwargs: None)

    pipe = pipeline_cls(
        od_config=SimpleNamespace(
            model=str(tmp_path),
            extras=extras,
            enable_diffusion_pipeline_profiler=False,
        )
    )

    assert pipe.use_diffusion_decoder is use_diffusion_decoder
    assert pipe._vae_modules.count("diffusion_decoder") == int(use_diffusion_decoder)


@pytest.mark.parametrize(
    ("pipeline_kind", "expected"),
    [
        ("one_stage", "regular"),
        ("two_stage", "regular"),
        ("distilled_one_stage", "distilled"),
        ("distilled_two_stage", "distilled"),
        ("dmd2", None),
    ],
)
def test_ltx_checkpoint_kind_is_derived_from_pipeline_kind(pipeline_kind, expected):
    assert resolve_ltx_checkpoint_kind(pipeline_kind) == expected


def test_ltx_checkpoint_kind_rejects_unknown_pipeline():
    with pytest.raises(ValueError, match="Unsupported LTX pipeline kind"):
        resolve_ltx_checkpoint_kind("unknown")


def test_ltx25_distilled_two_stage_recipe_matches_model_card_resolution():
    stage1, stage2 = LTX25_DISTILLED_TWO_STAGE_RECIPE.phases

    assert (LTX25_DISTILLED_TWO_STAGE_RECIPE.height, LTX25_DISTILLED_TWO_STAGE_RECIPE.width) == (1088, 1920)
    assert stage1.spatial_downscale == 2
    assert stage1.sigmas == LTX_DISTILLED_SIGMAS
    assert stage1.sampler == "euler_ancestral"
    assert stage2.sampler == "euler"
    assert stage2.sigmas == (0.909375, 0.725, 0.421875, 0.0)
    assert stage2.input_transform == "spatial_upsample"

    assert LTX25_DISTILLED_TWO_STAGE_RECIPE.negative_prompt == ""
    assert not LTX25_DISTILLED_TWO_STAGE_RECIPE.allow_negative_prompt


def test_ltx25_distilled_one_stage_uses_native_low_resolution():
    (phase,) = LTX25_DISTILLED_ONE_STAGE_RECIPE.phases

    assert (LTX25_DISTILLED_ONE_STAGE_RECIPE.height, LTX25_DISTILLED_ONE_STAGE_RECIPE.width) == (544, 960)
    assert phase.spatial_downscale == 1
    assert phase.sigmas == LTX_DISTILLED_SIGMAS
    assert phase.sampler == "euler_ancestral"
    assert not LTX25_DISTILLED_ONE_STAGE_RECIPE.supports_cache_dit


def test_ltx25_regular_two_stage_uses_full_transformer_then_distilled_lora():
    stage1, stage2 = LTX25_TWO_STAGE_RECIPE.phases

    assert (LTX25_TWO_STAGE_RECIPE.height, LTX25_TWO_STAGE_RECIPE.width) == (1088, 1920)
    assert LTX25_TWO_STAGE_RECIPE.num_inference_steps == 30
    assert stage1.spatial_downscale == 2
    assert stage1.guidance == LTX25_FULL_RECIPE.request_guidance
    assert stage2.adapter_slot == "ltx_distilled"
    assert LTX25_TWO_STAGE_RECIPE.allow_request_phase_sigmas
    assert LTX25_TWO_STAGE_COMPONENT_PROFILE.transformer_subfolder == "transformer_full"
    assert LTX25_TWO_STAGE_COMPONENT_PROFILE.distilled_lora_filename is not None


@pytest.mark.parametrize(
    "recipe",
    [
        LTX2_ONE_STAGE_RECIPE,
        LTX23_ONE_STAGE_RECIPE,
        LTX_POSITIVE_ONLY_RECIPE,
    ],
)
def test_ltx_one_stage_recipes_declare_cache_dit_supported(recipe):
    assert recipe.supports_cache_dit


@pytest.mark.parametrize(
    "recipe",
    [
        LTX25_FULL_RECIPE,
        LTX25_DISTILLED_ONE_STAGE_RECIPE,
        LTX2_DISTILLED_TWO_STAGE_RECIPE,
        LTX25_DISTILLED_TWO_STAGE_RECIPE,
    ],
)
def test_ltx25_and_multistage_recipes_declare_cache_dit_unsupported(recipe):
    assert not recipe.supports_cache_dit


def test_ltx_positive_only_guidance_preserves_official_x0_roundtrip():
    sample = torch.full((1, 1), 1e8, dtype=torch.float32)
    velocity = torch.ones((1, 1), dtype=torch.bfloat16)
    sigma = torch.tensor(0.5)

    actual = LTX_GUIDANCE_EXECUTOR._guide_modality(
        sample,
        {"cond": velocity},
        sigma,
        LTXModalityGuidance(),
    )
    expected = velocity_from_x0(sample, x0_from_velocity(sample, velocity, sigma), sigma)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_ltx_velocity_from_x0_materializes_official_scalar_sigma():
    item_calls = []

    class TrackingTensor(torch.Tensor):
        @staticmethod
        def __new__(cls):
            return torch.Tensor._make_subclass(cls, torch.tensor(0.725), False)

        def item(self):
            item_calls.append(True)
            return super().item()

    sample = torch.tensor([[1.0]], dtype=torch.bfloat16)
    x0 = torch.tensor([[0.5]], dtype=torch.bfloat16)

    velocity_from_x0(sample, x0, TrackingTensor())

    assert item_calls == [True]


def test_ltx_ancestral_positive_only_guidance_preserves_raw_velocity():
    sample = torch.full((1, 1), 1e8, dtype=torch.float32)
    velocity = torch.ones((1, 1), dtype=torch.bfloat16)

    actual = LTX_GUIDANCE_EXECUTOR._guide_modality(
        sample,
        {"cond": velocity},
        torch.tensor(0.5),
        LTXModalityGuidance(),
        preserve_positive_velocity=True,
    )

    assert actual is velocity


@pytest.mark.parametrize("model_version", ["2.5", "2.3"])
def test_ltx_t2v_denoise_state_uses_official_bfloat16_dtype(model_version):
    expected_dtype = torch.bfloat16
    captured = {}

    def prepare_latents(**kwargs):
        captured["video_dtype"] = kwargs["dtype"]
        return torch.empty(1, 1, 1, dtype=kwargs["dtype"])

    def prepare_audio_latents(*_args, **kwargs):
        captured["audio_dtype"] = kwargs["dtype"]
        return torch.empty(1, 1, 1, dtype=kwargs["dtype"]), 1, 1

    pipeline = SimpleNamespace(
        model_version=model_version,
        transformer=SimpleNamespace(config=SimpleNamespace(in_channels=128)),
        prepare_latents=prepare_latents,
        prepare_audio_latents=prepare_audio_latents,
        audio_sampling_rate=48_000,
        audio_hop_length=160,
        audio_vae_temporal_compression_ratio=4,
        audio_vae_mel_compression_ratio=4,
        audio_vae=SimpleNamespace(config=SimpleNamespace(mel_bins=64, latent_channels=8)),
        _resolve_audio_latent_length=lambda length, _latents: length,
    )
    request = SimpleNamespace(
        num_videos_per_prompt=1,
        height=64,
        width=64,
        num_frames=1,
        frame_rate=24.0,
        generator=None,
        latents=None,
        audio_latents=None,
        audio_latents_normalized=False,
    )
    prompt = SimpleNamespace(
        batch_size=1,
        positive_connector_prompt_embeds=torch.empty(1, dtype=torch.bfloat16),
        positive_connector_audio_prompt_embeds=torch.empty(1, dtype=torch.bfloat16),
    )

    LTXRuntime._prepare_video_latents_stage(
        pipeline,
        request,
        prompt,
        device=torch.device("cpu"),
        noise_scale=1.0,
    )
    LTXRuntime._prepare_audio_latents_stage(
        pipeline,
        request,
        prompt,
        device=torch.device("cpu"),
        noise_scale=1.0,
    )

    assert captured == {"video_dtype": expected_dtype, "audio_dtype": expected_dtype}


def test_ltx_seeded_latents_match_official_packed_rng_layout():
    pipeline = SimpleNamespace(
        model_version="2.5",
        vae_spatial_compression_ratio=2,
        vae_temporal_compression_ratio=2,
        transformer_spatial_patch_size=1,
        transformer_temporal_patch_size=1,
        audio_vae_mel_compression_ratio=2,
        od_config=SimpleNamespace(parallel_config=SimpleNamespace(sequence_parallel_size=1)),
    )
    device = torch.device("cpu")
    actual_generator = torch.Generator(device=device).manual_seed(123)
    expected_generator = torch.Generator(device=device).manual_seed(123)

    actual_video = ltx2_latents.prepare_video_latents(
        pipeline,
        batch_size=1,
        num_channels_latents=2,
        height=4,
        width=6,
        num_frames=3,
        dtype=torch.float32,
        device=device,
        generator=actual_generator,
    )
    actual_audio, original_length, padded_length = ltx2_latents.prepare_audio_latents(
        pipeline,
        batch_size=1,
        num_channels_latents=2,
        audio_latent_length=3,
        num_mel_bins=4,
        dtype=torch.float32,
        device=device,
        generator=actual_generator,
    )

    expected_video = torch.randn((1, 12, 2), generator=expected_generator)
    expected_audio = torch.randn((1, 3, 4), generator=expected_generator)

    torch.testing.assert_close(actual_video, expected_video, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_audio, expected_audio, rtol=0.0, atol=0.0)
    assert (original_length, padded_length) == (3, 3)


def test_ltx25_distilled_two_stage_executes_custom_phase_schedules():
    request_inputs = LTXRequestInputs(
        prompt="prompt",
        negative_prompt="",
        height=64,
        width=64,
        num_frames=1,
        frame_rate=24.0,
        num_inference_steps=8,
        guidance=LTXGuidanceSpec.positive_only(),
        num_videos_per_prompt=1,
        generator=None,
        latents=None,
        audio_latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        decode_timestep=0.25,
        decode_noise_scale=0.5,
        output_type="np",
        max_sequence_length=16,
    )
    prompt_context = object()
    phase_calls: list[Any] = []

    def resolve_request_inputs(req, **kwargs):
        return request_inputs

    def run_phase(req, inputs, *, prompt_context=None, **kwargs):
        phase_recipe = kwargs["phase_recipe"]
        assert kwargs["image"] is source_image
        phase_calls.append((phase_recipe, inputs, prompt_context))
        if len(phase_calls) == 1:
            assert prompt_context is None
            assert inputs.decode_timestep == 0.25
            assert inputs.decode_noise_scale == 0.5
            context = prompt_context_sentinel
            assert phase_recipe.name == "generate_lowres"
            assert (inputs.height, inputs.width) == (32, 32)
            assert inputs.num_inference_steps == 2
            assert kwargs["sigmas"] == stage_1_sigmas
            assert kwargs["noise_scale"] == 1.0
            video = torch.ones(1, 128, 1, 1, 1)
            audio = torch.full((1, 8, 1, 2), 2.0)
            audio_for_next_phase = torch.full((1, 8, 1, 2), 5.0)
        else:
            assert prompt_context is prompt_context_sentinel
            assert phase_recipe.name == "refine"
            assert (inputs.height, inputs.width) == (64, 64)
            assert inputs.latents.shape == (1, 128, 1, 2, 2)
            torch.testing.assert_close(inputs.audio_latents, torch.full((1, 8, 1, 2), 5.0))
            assert inputs.audio_latents_normalized
            assert inputs.guidance_scale == 1.0
            assert inputs.num_inference_steps == 2
            assert kwargs["sigmas"] == stage_2_sigmas
            assert kwargs["noise_scale"] == stage_2_sigmas[0]
            assert inputs.decode_timestep == 0.0
            assert inputs.decode_noise_scale is None
            context = prompt_context
            video = torch.full((1, 128, 1, 2, 2), 3.0)
            audio = torch.full((1, 8, 1, 2), 4.0)
            audio_for_next_phase = None
        return LTXPhaseResult(
            forward_context=SimpleNamespace(prompt_context=context),
            video=video,
            audio=audio,
            audio_for_next_phase=audio_for_next_phase,
        )

    def decode_phase(phase):
        return DiffusionOutput(output=(phase.video, phase.audio))

    class FakeUpsampler(torch.nn.Module):
        dtype = torch.float32

        def forward(self, latents):
            return latents.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)

    prompt_context_sentinel = prompt_context
    stage_1_sigmas = [1.0, 0.5, 0.0]
    stage_2_sigmas = [0.8, 0.2, 0.0]
    pipeline = object.__new__(LTX2TwoStagePipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.pipeline_recipe = LTX25_DISTILLED_TWO_STAGE_RECIPE
    pipeline.device = torch.device("cpu")
    pipeline.vae_spatial_compression_ratio = 32
    pipeline.vae_temporal_compression_ratio = 8
    pipeline.latent_upsampler = FakeUpsampler()
    object.__setattr__(pipeline, "_resolve_request_inputs", resolve_request_inputs)
    object.__setattr__(pipeline, "run_phase", run_phase)
    object.__setattr__(pipeline, "decode_phase", decode_phase)

    source_image = object()
    output = pipeline.forward(
        SimpleNamespace(sampling_params_list=[], prompts=[]),
        image=source_image,
        stage_1_sigmas=stage_1_sigmas,
        stage_2_sigmas=stage_2_sigmas,
    )

    assert len(phase_calls) == 2
    assert phase_calls[1][2] is prompt_context_sentinel
    torch.testing.assert_close(output.output[0], torch.full((1, 128, 1, 2, 2), 3.0))
    torch.testing.assert_close(output.output[1], torch.full((1, 8, 1, 2), 4.0))


class TestLTXRequestParsing:
    def test_request_resolves_custom_sigmas_from_extra_args(self):
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        sigmas = [1.0, 0.5, 0.0]
        req = DiffusionRequestBatch(
            [
                OmniDiffusionRequest(
                    prompt="prompt",
                    sampling_params=OmniDiffusionSamplingParams(extra_args={"sigmas": sigmas}),
                    request_id="ltx-custom-sigmas",
                )
            ]
        )

        assert _make_ltx_request_pipe(LTX2Pipeline)._resolve_request_sigmas(req, None) == sigmas

    def test_request_resolves_per_stage_sigmas_from_extra_args(self):
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        stage_1_sigmas = [1.0, 0.4, 0.0]
        req = DiffusionRequestBatch(
            [
                OmniDiffusionRequest(
                    prompt="prompt",
                    sampling_params=OmniDiffusionSamplingParams(extra_args={"stage_1_sigmas": stage_1_sigmas}),
                    request_id="ltx-custom-phase-sigmas",
                )
            ]
        )

        resolved = _make_ltx_request_pipe(LTX2TwoStagePipeline)._resolve_request_phase_sigmas(
            req, [1.0, 0.0], [0.8, 0.0]
        )

        assert resolved == (stage_1_sigmas, [0.8, 0.0])

    @pytest.mark.parametrize("value", [-1, 52, 1.5, True])
    def test_request_rejects_invalid_image_crf(self, value):
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        req = DiffusionRequestBatch(
            [
                OmniDiffusionRequest(
                    prompt="prompt",
                    sampling_params=OmniDiffusionSamplingParams(extra_args={"image_crf": value}),
                    request_id="ltx-invalid-image-crf",
                )
            ]
        )

        with pytest.raises(ValueError, match="image_crf"):
            _resolve_request_inputs_for_test(_make_ltx_request_pipe(LTX2Pipeline), req)


class TestLTXForwardStages:
    def test_gemma_tokenization_matches_official_bos_and_left_padding(self):
        pipe = _make_ltx_request_pipe(LTX2Pipeline)

        class FakeTokenizer:
            bos_token_id = 2
            eos_token_id = 1
            eos_token = "<eos>"
            pad_token_id = 0
            pad_token = "<pad>"
            padding_side = "right"

            def __init__(self):
                self.calls = []

            def __call__(self, text, **kwargs):
                self.calls.append((text, kwargs))
                ids = [7, 8, 9, self.eos_token_id] if text == "long" else [6, self.eos_token_id]
                return SimpleNamespace(input_ids=ids)

            def pad(self, encoded, *, padding, max_length, return_tensors, return_attention_mask):
                assert padding == "max_length"
                assert return_tensors == "pt"
                assert return_attention_mask is True
                rows = []
                masks = []
                for ids in encoded["input_ids"]:
                    pad = max_length - len(ids)
                    rows.append([self.pad_token_id] * pad + ids)
                    masks.append([0] * pad + [1] * len(ids))
                return SimpleNamespace(input_ids=torch.tensor(rows), attention_mask=torch.tensor(masks))

        class FakeTextEncoder(torch.nn.Module):
            dtype = torch.float32

            def __init__(self):
                super().__init__()
                self.seen_input_ids = None
                self.seen_attention_mask = None

            def forward(self, *, input_ids, attention_mask, output_hidden_states):
                assert output_hidden_states is True
                self.seen_input_ids = input_ids.clone()
                self.seen_attention_mask = attention_mask.clone()
                return SimpleNamespace(hidden_states=(input_ids.unsqueeze(-1).float(),))

        tokenizer = FakeTokenizer()
        text_encoder = FakeTextEncoder()
        object.__setattr__(pipe, "tokenizer", tokenizer)
        object.__setattr__(pipe, "text_encoder", text_encoder)

        pipe._get_gemma_prompt_embeds(
            [" long ", "short"],
            max_sequence_length=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        torch.testing.assert_close(text_encoder.seen_input_ids, torch.tensor([[2, 7, 8, 9], [0, 2, 6, 1]]))
        torch.testing.assert_close(text_encoder.seen_attention_mask, torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]]))
        assert tokenizer.padding_side == "left"
        assert [call[0] for call in tokenizer.calls] == ["long", "short"]
