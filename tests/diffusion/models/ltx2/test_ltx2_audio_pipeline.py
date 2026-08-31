# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for the unified LTX text-to-audio pipeline."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.interface import SupportAudioOutput
from vllm_omni.diffusion.models.ltx2.ltx2_audio_runtime import LTXAudioRuntime
from vllm_omni.diffusion.models.ltx2.ltx2_components import (
    LTX2_T2A_COMPONENT_PROFILE,
    LTX23_T2A_COMPONENT_PROFILE,
    LTX25_T2A_COMPONENT_PROFILE,
    create_audio_transformer_from_config,
    get_ltx2_audio_post_process_func,
    resolve_ltx_component_profile,
)
from vllm_omni.diffusion.models.ltx2.ltx2_guidance import LTXGuidancePlan, LTXGuidanceSpec
from vllm_omni.diffusion.models.ltx2.ltx2_recipes import (
    LTX2_T2A_RECIPE,
    LTX23_T2A_RECIPE,
    LTX25_T2A_RECIPE,
    resolve_ltx_pipeline_recipe,
)
from vllm_omni.diffusion.models.ltx2.ltx2_request import (
    LTX2AudioResourceLimits,
    resolve_ltx_audio_num_frames,
)
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_audio import LTX2TextToAudioPipeline

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("version", "profile", "recipe", "transformer_subfolder"),
    [
        ("2", LTX2_T2A_COMPONENT_PROFILE, LTX2_T2A_RECIPE, "transformer"),
        ("2.3", LTX23_T2A_COMPONENT_PROFILE, LTX23_T2A_RECIPE, "transformer"),
        ("2.5", LTX25_T2A_COMPONENT_PROFILE, LTX25_T2A_RECIPE, "transformer_full"),
    ],
)
def test_ltx_t2a_uses_one_pipeline_with_version_specific_full_profiles(version, profile, recipe, transformer_subfolder):
    assert resolve_ltx_component_profile("text_to_audio", version) is profile
    assert resolve_ltx_pipeline_recipe("text_to_audio", version) is recipe
    assert profile.transformer_subfolder == transformer_subfolder
    assert profile.dit_modules == ("transformer",)
    assert profile.vae_modules == ("audio_vae",)
    assert "vae" not in profile.vae_modules
    assert recipe.supports_cache_dit
    assert recipe.request_guidance.audio.modality_scale == 1.0
    assert LTXGuidancePlan.build(recipe.request_guidance).names == ("cond", "uncond", "ptb")


def test_ltx_t2a_public_contract_is_audio_only():
    assert SupportAudioOutput in LTX2TextToAudioPipeline.__mro__
    assert LTX2TextToAudioPipeline.pipeline_kind == "text_to_audio"
    assert LTX2TextToAudioPipeline.support_audio_output
    assert not LTX2TextToAudioPipeline.support_image_input
    assert not hasattr(LTX2TextToAudioPipeline, "support_video_output")
    assert LTX2TextToAudioPipeline.dummy_run_num_frames == 9


@pytest.mark.parametrize(
    ("seconds", "frame_rate", "expected"),
    [
        (5.0, 24.0, 121),
        (1.0, 24.0, 25),
        (0.5, 24.0, 17),
        (1.0, 25.0, 25),
        (5.1, 24.0, 129),
        (5.21, 24.0, 129),
    ],
)
def test_ltx_t2a_duration_quantizes_up_to_legal_video_clock(seconds, frame_rate, expected):
    assert (
        resolve_ltx_audio_num_frames(
            audio_length=seconds,
            num_frames=None,
            frame_rate=frame_rate,
            default_num_frames=121,
        )
        == expected
    )
    assert (expected - 1) % 8 == 0


@pytest.mark.parametrize(("seconds", "frame_rate"), [(0.5, 24.0), (5.1, 24.0)])
def test_ltx_t2a_audio_length_does_not_resolve_shorter_than_requested(seconds, frame_rate):
    num_frames = resolve_ltx_audio_num_frames(
        audio_length=seconds,
        num_frames=None,
        frame_rate=frame_rate,
        default_num_frames=121,
    )

    assert num_frames / frame_rate >= seconds


def test_ltx_t2a_exact_num_frames_overrides_default_duration():
    assert (
        resolve_ltx_audio_num_frames(
            audio_length=None,
            num_frames=81,
            frame_rate=24.0,
            default_num_frames=121,
        )
        == 81
    )


def test_ltx_t2a_audio_resource_limits_accept_deployment_overrides():
    limits = LTX2AudioResourceLimits.from_additional_config(
        {
            "ltx2_audio_limits": {
                "max_duration_seconds": 12.5,
                "max_latent_frames": 320,
            }
        }
    )

    assert limits.max_duration_seconds == 12.5
    assert limits.max_latent_frames == 320


@pytest.mark.parametrize(
    ("additional_config", "error"),
    [
        ({"ltx2_audio_limits": {"max_duration_seconds": float("inf")}}, "finite and positive"),
        ({"ltx2_audio_limits": {"max_duration_seconds": 0}}, "finite and positive"),
        ({"ltx2_audio_limits": {"max_latent_frames": 0}}, "positive integer"),
        ({"ltx2_audio_limits": {"max_latent_frames": 2.5}}, "positive integer"),
        ({"ltx2_audio_limits": {"unknown": 1}}, "Unknown"),
    ],
)
def test_ltx_t2a_audio_resource_limits_reject_invalid_config(additional_config, error):
    with pytest.raises((TypeError, ValueError), match=error):
        LTX2AudioResourceLimits.from_additional_config(additional_config)


def test_ltx_t2a_audio_graph_bucket_pads_without_consuming_request_rng():
    pipe = object.__new__(LTXAudioRuntime)
    torch.nn.Module.__init__(pipe)
    pipe.device = torch.device("cpu")
    pipe.audio_sampling_rate = 100
    pipe.audio_hop_length = 1
    pipe.audio_vae_temporal_compression_ratio = 1
    pipe.audio_vae_mel_compression_ratio = 2
    pipe.audio_vae = SimpleNamespace(config=SimpleNamespace(mel_bins=8, latent_channels=2))
    pipe.pipeline_recipe = SimpleNamespace(num_frames=121, phases=(SimpleNamespace(noise_scale=1.0),))
    pipe._audio_cuda_graph_config = SimpleNamespace(audio_length_buckets=(1.0, 2.0))
    logical = torch.arange(104 * 8, dtype=torch.float32).reshape(1, 104, 8)

    def prepare_audio_latents(*_args, **_kwargs):
        return logical.clone(), 104, 104

    pipe.prepare_audio_latents = prepare_audio_latents
    inputs = SimpleNamespace(
        num_frames=25,
        frame_rate=24.0,
        num_videos_per_prompt=1,
        generator=None,
        audio_latents=None,
    )
    prompt_context = SimpleNamespace(
        batch_size=1,
        positive_connector_audio_prompt_embeds=torch.zeros(1, 1, 1),
    )

    audio_latents, original, padded, latent_mel_bins = pipe._prepare_audio_state(inputs, prompt_context)

    assert original == 104
    assert padded == 104
    assert latent_mel_bins == 4
    torch.testing.assert_close(audio_latents, logical)


def test_ltx_t2a_audio_graph_warms_first_bucket_without_padding(monkeypatch):
    from vllm_omni.diffusion.models.ltx2 import ltx2_audio_runtime
    from vllm_omni.diffusion.models.ltx2.ltx2_request import LTXRequestInputs

    pipe = object.__new__(LTXAudioRuntime)
    torch.nn.Module.__init__(pipe)
    pipe.pipeline_recipe = SimpleNamespace(frame_rate=24.0, num_frames=121, height=512, width=512)
    pipe._audio_cuda_graph_config = SimpleNamespace(audio_length_buckets=(1.0, 5.0))
    pipe._resolve_request_sigmas = lambda *_args: None

    def resolve_inputs(_req, **kwargs):
        return LTXRequestInputs(
            prompt=None,
            negative_prompt=None,
            height=kwargs["height"],
            width=kwargs["width"],
            num_frames=kwargs["num_frames"],
            frame_rate=kwargs["frame_rate"],
            num_inference_steps=1,
            guidance=LTXGuidanceSpec.positive_only(),
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
            max_sequence_length=1024,
        )

    pipe._resolve_request_inputs = resolve_inputs
    monkeypatch.setattr(ltx2_audio_runtime, "validate_pipeline_request", lambda *_args, **_kwargs: None)
    sampling = SimpleNamespace(extra_args={}, num_frames=9, resolved_frame_rate=24.0, latents=None)
    request = SimpleNamespace(sampling_params_list=[sampling], is_dummy_run=lambda: True)

    inputs = pipe._resolve_audio_request_inputs(request)

    assert inputs.num_frames == 25


def test_ltx_t2a_rejects_cuda_graph_bucket_above_duration_limit():
    pipe = object.__new__(LTXAudioRuntime)
    torch.nn.Module.__init__(pipe)
    pipe.pipeline_recipe = SimpleNamespace(frame_rate=24.0, num_frames=121)
    pipe._audio_cuda_graph_config = SimpleNamespace(audio_length_buckets=(20.11,))
    pipe._audio_resource_limits = LTX2AudioResourceLimits()

    with pytest.raises(ValueError, match="max_duration_seconds"):
        pipe._validate_audio_cuda_graph_buckets()


@pytest.mark.parametrize(
    ("extra_args", "num_frames", "frame_rate", "is_dummy", "error"),
    [
        ({"audio_length": 5.0, "num_frames": 121}, 1, 24.0, False, "mutually exclusive"),
        ({"audio_length": 20.11}, 1, 24.0, False, "max_duration_seconds"),
        ({"audio_length": float("inf")}, 1, 24.0, False, "finite"),
        ({"num_frames": 4097}, 1, 24.0, False, "duration"),
        ({}, 9, 1.0, False, "frame_rate=24"),
        ({}, 4097, 24.0, True, "duration"),
    ],
)
def test_ltx_t2a_production_request_path_rejects_unsafe_shapes(
    extra_args,
    num_frames,
    frame_rate,
    is_dummy,
    error,
):
    pipe = object.__new__(LTXAudioRuntime)
    torch.nn.Module.__init__(pipe)
    pipe.pipeline_recipe = SimpleNamespace(frame_rate=24.0, num_frames=121)
    pipe._audio_cuda_graph_config = SimpleNamespace(audio_length_buckets=())
    pipe._audio_resource_limits = LTX2AudioResourceLimits()
    pipe._reject_video_options = lambda _sampling: None
    pipe._resolve_request_inputs = lambda *_args, **_kwargs: pytest.fail("unsafe request reached normalization")
    sampling = SimpleNamespace(
        extra_args=extra_args,
        num_frames=num_frames,
        resolved_frame_rate=frame_rate,
    )
    request = SimpleNamespace(
        sampling_params_list=[sampling],
        is_dummy_run=lambda: is_dummy,
    )

    with pytest.raises(ValueError, match=error):
        pipe._resolve_audio_request_inputs(request)


def test_ltx_t2a_audio_graph_bucket_appends_zero_padding():
    pipe = object.__new__(LTXAudioRuntime)
    torch.nn.Module.__init__(pipe)
    pipe.device = torch.device("cpu")
    pipe.audio_sampling_rate = 24
    pipe.audio_hop_length = 1
    pipe.audio_vae_temporal_compression_ratio = 1
    pipe.audio_vae_mel_compression_ratio = 2
    pipe.audio_vae = SimpleNamespace(config=SimpleNamespace(mel_bins=8, latent_channels=2))
    pipe.pipeline_recipe = SimpleNamespace(num_frames=121, phases=(SimpleNamespace(noise_scale=1.0),))
    pipe._audio_cuda_graph_config = SimpleNamespace(audio_length_buckets=(2.0,))
    logical = torch.ones(1, 25, 8)
    pipe.prepare_audio_latents = lambda *_args, **_kwargs: (logical.clone(), 25, 25)
    inputs = SimpleNamespace(
        num_frames=25,
        frame_rate=24.0,
        num_videos_per_prompt=1,
        generator=None,
        audio_latents=None,
    )
    prompt_context = SimpleNamespace(
        batch_size=1,
        positive_connector_audio_prompt_embeds=torch.zeros(1, 1, 1),
    )

    audio_latents, original, padded, _latent_mel_bins = pipe._prepare_audio_state(inputs, prompt_context)

    assert original == 25
    assert padded == 49
    torch.testing.assert_close(audio_latents[:, :25], logical)
    torch.testing.assert_close(audio_latents[:, 25:], torch.zeros(1, 24, 8))


def test_ltx_t2a_rejects_latent_budget_before_allocation():
    pipe = object.__new__(LTXAudioRuntime)
    torch.nn.Module.__init__(pipe)
    pipe.audio_sampling_rate = 100
    pipe.audio_hop_length = 1
    pipe.audio_vae_temporal_compression_ratio = 1
    pipe._audio_resource_limits = LTX2AudioResourceLimits(max_latent_frames=100)
    pipe.prepare_audio_latents = lambda *_args, **_kwargs: pytest.fail("latent allocation must not run")
    inputs = SimpleNamespace(num_frames=25, frame_rate=24.0)
    prompt_context = SimpleNamespace(batch_size=1)

    with pytest.raises(ValueError, match="latent frames"):
        pipe._prepare_audio_state(inputs, prompt_context)


def test_ltx_t2a_rejects_sp_padded_latent_budget_before_allocation():
    pipe = object.__new__(LTXAudioRuntime)
    torch.nn.Module.__init__(pipe)
    pipe.audio_sampling_rate = 96
    pipe.audio_hop_length = 1
    pipe.audio_vae_temporal_compression_ratio = 1
    pipe._audio_resource_limits = LTX2AudioResourceLimits(max_latent_frames=100)
    pipe.od_config = SimpleNamespace(parallel_config=SimpleNamespace(sequence_parallel_size=8))
    pipe.prepare_audio_latents = lambda *_args, **_kwargs: pytest.fail("latent allocation must not run")
    inputs = SimpleNamespace(num_frames=25, frame_rate=24.0)
    prompt_context = SimpleNamespace(batch_size=1)

    with pytest.raises(ValueError, match="latent frames"):
        pipe._prepare_audio_state(inputs, prompt_context)


def test_ltx_t2a_rejects_bucket_latent_budget_before_allocation():
    pipe = object.__new__(LTXAudioRuntime)
    torch.nn.Module.__init__(pipe)
    pipe.audio_sampling_rate = 100
    pipe.audio_hop_length = 1
    pipe.audio_vae_temporal_compression_ratio = 1
    pipe.pipeline_recipe = SimpleNamespace(num_frames=121, phases=(SimpleNamespace(noise_scale=1.0),))
    pipe._audio_cuda_graph_config = SimpleNamespace(audio_length_buckets=(2.0,))
    pipe._audio_resource_limits = LTX2AudioResourceLimits(max_latent_frames=150)
    pipe.prepare_audio_latents = lambda *_args, **_kwargs: pytest.fail("latent allocation must not run")
    inputs = SimpleNamespace(num_frames=25, frame_rate=24.0)
    prompt_context = SimpleNamespace(batch_size=1)

    with pytest.raises(ValueError, match="latent frames"):
        pipe._prepare_audio_state(inputs, prompt_context)


@pytest.mark.parametrize(
    ("audio_length", "num_frames", "frame_rate", "error"),
    [
        (1.0, 25, 24.0, "mutually exclusive"),
        (0.0, None, 24.0, "positive"),
        (1.0, None, 0.0, "frame_rate"),
        (1.0, None, float("inf"), "finite"),
        (float("nan"), None, 24.0, "finite"),
        (None, 25.5, 24.0, "integer"),
        (None, "25", 24.0, "integer"),
        (None, True, 24.0, "integer"),
        (None, float("nan"), 24.0, "integer"),
        (None, float("inf"), 24.0, "integer"),
        (None, 24, 24.0, "8 \\* k \\+ 1"),
    ],
)
def test_ltx_t2a_rejects_invalid_duration_inputs(audio_length, num_frames, frame_rate, error):
    with pytest.raises(ValueError, match=error):
        resolve_ltx_audio_num_frames(
            audio_length=audio_length,
            num_frames=num_frames,
            frame_rate=frame_rate,
            default_num_frames=121,
        )


@pytest.mark.parametrize(
    ("num_frames", "frame_rate", "error"),
    [
        (9, 1.0, "frame_rate=24"),
        (4097, 24.0, "duration"),
    ],
)
def test_ltx_t2a_runtime_rejects_unsafe_resolved_duration(num_frames, frame_rate, error):
    limits = LTX2AudioResourceLimits()

    with pytest.raises(ValueError, match=error):
        limits.validate_resolved_duration(
            num_frames=num_frames,
            frame_rate=frame_rate,
            expected_frame_rate=24.0,
        )


def test_ltx_t2a_runtime_rejects_requested_duration_above_limit():
    with pytest.raises(ValueError, match="max_duration_seconds"):
        LTX2AudioResourceLimits().validate_requested_duration(20.11)


def test_ltx_t2a_duration_limit_below_minimum_clock_rejects_minimum_shape():
    with pytest.raises(ValueError, match="max_duration_seconds"):
        LTX2AudioResourceLimits(max_duration_seconds=0.1).validate_resolved_duration(
            num_frames=9,
            frame_rate=24.0,
            expected_frame_rate=24.0,
        )


def test_ltx_t2a_default_resource_limits_accept_twenty_second_boundary():
    limits = LTX2AudioResourceLimits()
    num_frames = resolve_ltx_audio_num_frames(
        audio_length=20.0,
        num_frames=None,
        frame_rate=24.0,
        default_num_frames=121,
    )

    limits.validate_requested_duration(20.0)
    limits.validate_resolved_duration(
        num_frames=num_frames,
        frame_rate=24.0,
        expected_frame_rate=24.0,
    )
    limits.validate_latent_frames(512)
    with pytest.raises(ValueError, match="max_duration_seconds"):
        limits.validate_resolved_duration(
            num_frames=num_frames + 8,
            frame_rate=24.0,
            expected_frame_rate=24.0,
        )


@pytest.mark.parametrize("latent_frames", [0, 513])
def test_ltx_t2a_runtime_rejects_latent_shape_outside_budget(latent_frames):
    with pytest.raises(ValueError, match="latent frames"):
        LTX2AudioResourceLimits().validate_latent_frames(latent_frames)


def test_ltx_t2a_checkpoint_metadata_selects_profile(tmp_path, monkeypatch):
    from vllm_omni.diffusion.models.ltx2 import ltx2_audio_runtime

    (tmp_path / "model_index.json").write_text(json.dumps({"model_version": "2.5"}))

    def stub_components(pipe, od_config):
        pipe.od_config = od_config
        pipe.device = torch.device("cpu")

    monkeypatch.setattr(ltx2_audio_runtime, "initialize_audio_pipeline_components", stub_components)
    monkeypatch.setattr(LTX2TextToAudioPipeline, "setup_diffusion_pipeline_profiler", lambda *_args, **_kwargs: None)

    od_config = SimpleNamespace(
        model=str(tmp_path),
        revision=None,
        parallel_config=SimpleNamespace(ulysses_mode="strict"),
        cache_backend="cache_dit",
        enable_diffusion_pipeline_profiler=False,
    )
    pipe = LTX2TextToAudioPipeline(od_config=od_config)

    assert pipe.model_version == "2.5"
    assert pipe.component_profile is LTX25_T2A_COMPONENT_PROFILE
    assert pipe.pipeline_recipe is LTX25_T2A_RECIPE
    assert pipe.audio_graph_runner is None


def test_ltx_t2a_rejects_cache_dit_for_unqualified_recipe(tmp_path, monkeypatch):
    from vllm_omni.diffusion.models.ltx2 import ltx2_audio_runtime

    (tmp_path / "model_index.json").write_text(json.dumps({"model_version": "2.5"}))

    def stub_components(pipe, od_config):
        pipe.od_config = od_config
        pipe.device = torch.device("cpu")

    unsupported_recipe = replace(LTX25_T2A_RECIPE, supports_cache_dit=False)
    monkeypatch.setattr(ltx2_audio_runtime, "resolve_ltx_pipeline_recipe", lambda *_args: unsupported_recipe)
    monkeypatch.setattr(ltx2_audio_runtime, "initialize_audio_pipeline_components", stub_components)
    monkeypatch.setattr(LTX2TextToAudioPipeline, "setup_diffusion_pipeline_profiler", lambda *_args, **_kwargs: None)

    od_config = SimpleNamespace(
        model=str(tmp_path),
        revision=None,
        parallel_config=SimpleNamespace(ulysses_mode="strict"),
        cache_backend="cache_dit",
        enable_diffusion_pipeline_profiler=False,
    )

    with pytest.raises(ValueError, match="Cache-DiT is not qualified for this LTX recipe"):
        LTX2TextToAudioPipeline(od_config=od_config)


def test_ltx_t2a_rejects_cache_dit_with_manual_cuda_graph():
    od_config = SimpleNamespace(
        parallel_config=SimpleNamespace(ulysses_mode="strict"),
        cache_backend="cache_dit",
        additional_config={"ltx2_audio_cuda_graph": {"enabled": True}},
    )

    with pytest.raises(ValueError, match="cannot be enabled together"):
        LTX2TextToAudioPipeline(od_config=od_config)


def test_ltx_t2a_rejects_other_cache_backends():
    od_config = SimpleNamespace(
        parallel_config=SimpleNamespace(ulysses_mode="strict"),
        cache_backend="tea_cache",
        additional_config=None,
    )

    with pytest.raises(ValueError, match="does not support cache_backend='tea_cache'"):
        LTX2TextToAudioPipeline(od_config=od_config)


def test_ltx_t2a_runtime_owns_configured_audio_graph_runner(tmp_path, monkeypatch):
    from vllm_omni.diffusion.models.ltx2 import ltx2_audio_runtime

    (tmp_path / "model_index.json").write_text(json.dumps({"model_version": "2.5"}))
    transformer = torch.nn.Identity()

    def stub_components(pipe, od_config):
        pipe.od_config = od_config
        pipe.device = torch.device("cuda")
        pipe.transformer = transformer

    monkeypatch.setattr(ltx2_audio_runtime, "get_local_device", lambda: torch.device("cuda"))
    monkeypatch.setattr(ltx2_audio_runtime, "initialize_audio_pipeline_components", stub_components)
    monkeypatch.setattr(LTX2TextToAudioPipeline, "setup_diffusion_pipeline_profiler", lambda *_args, **_kwargs: None)

    od_config = SimpleNamespace(
        model=str(tmp_path),
        revision=None,
        dtype=torch.bfloat16,
        parallel_config=SimpleNamespace(
            ulysses_mode="strict",
            tensor_parallel_size=1,
            sequence_parallel_size=1,
        ),
        cache_backend="none",
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        enable_distributed_layerwise_offload=False,
        quantization_config=None,
        lora_path=None,
        enable_diffusion_pipeline_profiler=False,
        additional_config={"ltx2_audio_cuda_graph": {"enabled": True, "max_entries": 2}},
    )

    pipe = LTX2TextToAudioPipeline(od_config=od_config)

    assert pipe.audio_graph_runner is not None
    assert pipe.audio_graph_runner.transformer is transformer
    assert pipe.audio_graph_runner.max_graphs == 2
    assert pipe.audio_graph_runner.device == torch.device("cuda")


def test_ltx_t2a_transformer_factory_projects_full_config_to_audio_only_model():
    transformer = create_audio_transformer_from_config(
        {
            "in_channels": 128,
            "num_attention_heads": 32,
            "audio_in_channels": 4,
            "audio_out_channels": 4,
            "audio_num_attention_heads": 2,
            "audio_attention_head_dim": 4,
            "audio_cross_attention_dim": 8,
            "caption_channels": 8,
            "num_layers": 0,
            "use_prompt_embeddings": False,
        }
    )

    assert transformer.config.audio_in_channels == 4
    assert not hasattr(transformer, "proj_in")


def test_ltx_t2a_postprocess_emits_audio_payload_and_checkpoint_sample_rate(tmp_path):
    vocoder_dir = tmp_path / "vocoder"
    vocoder_dir.mkdir()
    (vocoder_dir / "config.json").write_text(json.dumps({"output_sampling_rate": 48000}))
    postprocess = get_ltx2_audio_post_process_func(SimpleNamespace(model=str(tmp_path), revision=None))
    waveform = torch.randn(1, 2, 16)

    result = postprocess(waveform)

    assert set(result) == {"audio", "audio_sample_rate"}
    assert result["audio_sample_rate"] == 48000
    assert result["audio"].device.type == "cpu"


def test_ltx_t2a_decode_uses_only_audio_vae_and_vocoder():
    class AudioVAE:
        dtype = torch.float32
        latents_mean = torch.tensor(0.0)
        latents_std = torch.tensor(1.0)

        def decode(self, latents, return_dict=False):
            assert not return_dict
            return (latents + 2,)

    pipe = object.__new__(LTXAudioRuntime)
    torch.nn.Module.__init__(pipe)
    pipe.audio_vae = AudioVAE()
    pipe.vocoder = lambda mel: mel * 3
    pipe.audio_vae_mel_compression_ratio = 2
    packed = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)

    waveform = pipe._decode_audio_latents(packed, original_num_frames=2, latent_mel_bins=4)

    assert waveform.shape == (1, 2, 2, 4)
    assert not hasattr(pipe, "vae")
    assert not hasattr(pipe, "video_processor")


def test_ltx_t2a_decode_runs_bwe_vocoder_in_fp32_and_restores_dtype():
    class AudioVAE:
        dtype = torch.bfloat16
        latents_mean = torch.tensor(0.0)
        latents_std = torch.tensor(1.0)

        def decode(self, latents, return_dict=False):
            assert not return_dict
            return (latents,)

    class BWEVocoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bwe_generator = torch.nn.Identity()
            self.weight = torch.nn.Parameter(torch.ones((), dtype=torch.bfloat16))
            self.input_dtype = None

        def forward(self, mel):
            self.input_dtype = mel.dtype
            return mel.float() * self.weight.float()

    pipe = object.__new__(LTXAudioRuntime)
    torch.nn.Module.__init__(pipe)
    pipe.audio_vae = AudioVAE()
    pipe.vocoder = BWEVocoder()
    packed = torch.arange(16, dtype=torch.bfloat16).reshape(1, 2, 8)

    waveform = pipe._decode_audio_latents(packed, original_num_frames=2, latent_mel_bins=4)

    assert pipe.vocoder.input_dtype == torch.float32
    assert waveform.dtype == torch.bfloat16


def test_ltx_t2a_registry_and_postprocess_entries():
    from vllm_omni.diffusion.registry import (
        _DIFFUSION_MODELS,
        _DIFFUSION_POST_PROCESS_FUNCS,
        _NO_CACHE_ACCELERATION,
    )

    assert _DIFFUSION_MODELS["LTX2TextToAudioPipeline"] == (
        "ltx2",
        "pipeline_ltx2_audio",
        "LTX2TextToAudioPipeline",
    )
    assert _DIFFUSION_POST_PROCESS_FUNCS["LTX2TextToAudioPipeline"] == "get_ltx2_audio_post_process_func"
    assert "LTX2TextToAudioPipeline" not in _NO_CACHE_ACCELERATION


def test_ltx_t2a_weight_source_filters_video_tensors_before_materialization(tmp_path, monkeypatch):
    from vllm_omni.diffusion.models.ltx2 import ltx2_audio_runtime

    captured = {}

    class Source:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(ltx2_audio_runtime.DiffusersPipelineLoader, "ComponentSource", Source)
    monkeypatch.setattr(ltx2_audio_runtime, "prefetch_subfolders", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        ltx2_audio_runtime.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: SimpleNamespace()
    )

    pipe = SimpleNamespace(component_profile=LTX2_T2A_COMPONENT_PROFILE)
    od_config = SimpleNamespace(model=str(tmp_path), revision=None, dtype=torch.float32)
    with pytest.raises(Exception):
        # Later component loading is intentionally unstubbed; the source is
        # constructed first and is the contract under test.
        ltx2_audio_runtime.initialize_audio_pipeline_components(pipe, od_config)

    assert captured["weight_name_patterns"] == (
        "audio_*",
        "transformer_blocks.*.audio_*",
    )


def test_ltx_t2a_rejects_distilled_scheduler_before_large_components(tmp_path, monkeypatch):
    from vllm_omni.diffusion.models.ltx2 import ltx2_audio_runtime

    tokenizer_loaded = False

    class Source:
        def __init__(self, **_kwargs):
            pass

    class Scheduler:
        config = {"use_dynamic_shifting": False, "shift_terminal": None}

    def load_tokenizer(*_args, **_kwargs):
        nonlocal tokenizer_loaded
        tokenizer_loaded = True

    monkeypatch.setattr(ltx2_audio_runtime.DiffusersPipelineLoader, "ComponentSource", Source)
    monkeypatch.setattr(ltx2_audio_runtime, "prefetch_subfolders", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        ltx2_audio_runtime.FlowMatchEulerDiscreteScheduler,
        "from_pretrained",
        lambda *_args, **_kwargs: Scheduler(),
    )
    monkeypatch.setattr(ltx2_audio_runtime.AutoTokenizer, "from_pretrained", load_tokenizer)
    pipe = SimpleNamespace(component_profile=LTX2_T2A_COMPONENT_PROFILE, pipeline_kind="text_to_audio")
    od_config = SimpleNamespace(model=str(tmp_path), revision=None, dtype=torch.float32)

    with pytest.raises(ValueError, match="regular non-distilled"):
        ltx2_audio_runtime.initialize_audio_pipeline_components(pipe, od_config)

    assert not tokenizer_loaded


def test_ltx25_t2a_overrides_shared_distilled_scheduler_before_validation(tmp_path, monkeypatch):
    from vllm_omni.diffusion.models.ltx2 import ltx2_audio_runtime

    tokenizer_loaded = False
    scheduler_from_config = {}

    class Source:
        def __init__(self, **_kwargs):
            pass

    class Scheduler:
        def __init__(self, config):
            self.config = config

    def load_tokenizer(*_args, **_kwargs):
        nonlocal tokenizer_loaded
        tokenizer_loaded = True
        return SimpleNamespace(model_max_length=1024)

    def rebuild_scheduler(_config, **kwargs):
        scheduler_from_config.update(kwargs)
        return Scheduler({"use_dynamic_shifting": True, "shift_terminal": 0.1})

    monkeypatch.setattr(ltx2_audio_runtime.DiffusersPipelineLoader, "ComponentSource", Source)
    monkeypatch.setattr(ltx2_audio_runtime, "prefetch_subfolders", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        ltx2_audio_runtime.FlowMatchEulerDiscreteScheduler,
        "from_pretrained",
        lambda *_args, **_kwargs: Scheduler({"use_dynamic_shifting": False, "shift_terminal": None}),
    )
    monkeypatch.setattr(
        ltx2_audio_runtime.FlowMatchEulerDiscreteScheduler,
        "from_config",
        rebuild_scheduler,
    )
    monkeypatch.setattr(ltx2_audio_runtime.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(
        ltx2_audio_runtime,
        "_load_component",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("stop after scheduler validation")),
    )

    pipe = SimpleNamespace(
        component_profile=LTX25_T2A_COMPONENT_PROFILE,
        pipeline_kind="text_to_audio",
    )
    od_config = SimpleNamespace(model=str(tmp_path), revision=None, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="stop after scheduler validation"):
        ltx2_audio_runtime.initialize_audio_pipeline_components(pipe, od_config)

    assert scheduler_from_config == {"use_dynamic_shifting": True, "shift_terminal": 0.1}
    assert tokenizer_loaded


@pytest.mark.parametrize("request_sigmas", ([1.0, 0.5, 0.0], None))
def test_ltx_t2a_denoise_passes_audio_padding_mask_without_video_inputs(monkeypatch, request_sigmas):
    from vllm_omni.diffusion.models.ltx2 import ltx2_audio_runtime, ltx2_guidance
    from vllm_omni.diffusion.models.ltx2.ltx2_audio_transformer import LTX2AudioStaticConditioning

    calls = []
    conditioning_preparations = []
    perturbation_builds = 0
    sigma_scalars = []
    original_build_perturbation_kwargs = ltx2_audio_runtime.build_perturbation_kwargs
    original_velocity_from_x0 = ltx2_guidance.velocity_from_x0

    def count_perturbation_builds(*args, **kwargs):
        nonlocal perturbation_builds
        perturbation_builds += 1
        return original_build_perturbation_kwargs(*args, **kwargs)

    def track_sigma_scalar(sample, x0, sigma, *, sigma_scalar=None):
        sigma_scalars.append(sigma_scalar)
        return original_velocity_from_x0(sample, x0, sigma, sigma_scalar=sigma_scalar)

    class Rope:
        @staticmethod
        def prepare_audio_coords(batch_size, num_frames, device):
            return torch.zeros(batch_size, 1, num_frames, 2, device=device)

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.audio_rope = Rope()

        def forward(self, **kwargs):
            calls.append(kwargs)
            return torch.zeros_like(kwargs["audio_hidden_states"])

        def prepare_static_conditioning(self, context, coords, *, hidden_dtype):
            conditioning = LTX2AudioStaticConditioning(
                encoder_hidden_states=context,
                rotary_emb=(
                    coords.to(hidden_dtype),
                    coords.to(hidden_dtype),
                ),
            )
            conditioning_preparations.append(conditioning)
            return conditioning

    class Progress:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def update(self):
            pass

    pipe = object.__new__(LTXAudioRuntime)
    torch.nn.Module.__init__(pipe)
    pipe.device = torch.device("cpu")
    pipe.od_config = SimpleNamespace(parallel_config=SimpleNamespace(ring_degree=1))
    pipe.scheduler = SimpleNamespace(config={"num_train_timesteps": 1000})
    pipe.transformer = Transformer()
    pipe._guidance_plan = LTXGuidancePlan.build(LTXGuidanceSpec.positive_only())
    pipe._interrupt = False
    pipe.progress_bar = lambda **_kwargs: Progress()
    monkeypatch.setattr(ltx2_audio_runtime, "get_guidance_parallel_world_size", lambda: 1)
    monkeypatch.setattr(ltx2_audio_runtime, "build_perturbation_kwargs", count_perturbation_builds)
    monkeypatch.setattr(ltx2_guidance, "velocity_from_x0", track_sigma_scalar)
    prompt_context = SimpleNamespace(
        positive_connector_audio_prompt_embeds=torch.zeros(1, 2, 4),
        negative_connector_audio_prompt_embeds=None,
    )
    inputs = SimpleNamespace(num_inference_steps=2)
    latents = torch.ones(1, 3, 4)

    result = pipe._run_audio_denoise(
        latents,
        prompt_context,
        inputs,
        original_num_frames=2,
        padded_num_frames=3,
        request_sigmas=request_sigmas,
    )

    assert len(calls) == 2
    assert len(conditioning_preparations) == 1
    assert calls[0]["audio_static_conditioning"] is conditioning_preparations[0]
    assert calls[1]["audio_static_conditioning"] is conditioning_preparations[0]
    assert perturbation_builds == 1
    assert len(sigma_scalars) == 2
    assert all(isinstance(value, float) for value in sigma_scalars)
    if request_sigmas is not None:
        assert sigma_scalars == [1.0, 0.5]
    assert calls[0]["audio_encoder_hidden_states"] is calls[1]["audio_encoder_hidden_states"]
    assert calls[0]["audio_coords"] is calls[1]["audio_coords"]
    assert calls[0]["audio_attention_mask"] is calls[1]["audio_attention_mask"]
    assert not {"hidden_states", "encoder_hidden_states", "video_coords"} & calls[0].keys()
    torch.testing.assert_close(calls[0]["audio_attention_mask"], torch.tensor([[True, True, False]]))
    torch.testing.assert_close(result[:, :2], latents[:, :2])
    torch.testing.assert_close(result[:, 2:], torch.zeros_like(result[:, 2:]))
