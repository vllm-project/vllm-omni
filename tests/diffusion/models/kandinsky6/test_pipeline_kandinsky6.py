# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("distilled", [False, True])
@pytest.mark.parametrize("entrypoint", ["api", "offline"])
def test_checkpoint_defaults_reach_pipeline_from_shared_entrypoints(monkeypatch, distilled, entrypoint):
    """Shared wrappers must leave checkpoint-dependent fields unspecified."""
    import torch

    from vllm_omni.diffusion.models.kandinsky6 import KandinskyFlowMatchScheduler, KandinskyPiFlowScheduler
    from vllm_omni.diffusion.models.kandinsky6 import pipeline_kandinsky6 as module
    from vllm_omni.entrypoints.openai.protocol.videos import VideoGenerationRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.model_extras import get_video_generation_defaults

    defaults = get_video_generation_defaults("Kandinsky6TI2VAPipeline")
    assert defaults is not None
    assert defaults.max_area is None
    if entrypoint == "api":
        request = VideoGenerationRequest(prompt="A fox runs through snow")
        vp = request.resolve_video_params(default_fps=defaults.fps, default_num_frames=defaults.num_frames)
        sampling = OmniDiffusionSamplingParams(height=vp.height, width=vp.width)
        if vp.num_frames is not None:
            sampling.num_frames = vp.num_frames
        sampling.num_inference_steps = defaults.num_inference_steps
    else:
        preset = defaults.cli_defaults()
        assert "guidance_scale" not in preset
        sampling = OmniDiffusionSamplingParams(
            **{key: preset[key] for key in ("height", "width", "num_frames", "num_inference_steps")}
        )

    monkeypatch.setattr(module, "get_local_device", lambda: torch.device("cpu"))
    scheduler = (KandinskyPiFlowScheduler if distilled else KandinskyFlowMatchScheduler)()
    pipeline = module.Kandinsky6TI2VAPipeline(
        transformer=torch.nn.Identity(), vae=torch.nn.Identity(), text_encoder=torch.nn.Identity(), scheduler=scheduler
    )
    assert pipeline._resolve_generation_settings(sampling) == (
        480,
        864,
        121 if distilled else 125,
        10 if distilled else 50,
        1.0 if distilled else 5.0,
    )
    explicit = OmniDiffusionSamplingParams(
        height=320, width=512, num_frames=25, num_inference_steps=3, guidance_scale=1.0
    )
    # The worker marks explicit guidance before either pipeline entrypoint.
    explicit.guidance_scale_provided = True
    assert pipeline._resolve_generation_settings(explicit) == (320, 512, 25, 3, 1.0)


def test_checkpoint_scheduler_config_overrides_generation_defaults(monkeypatch):
    import torch

    from vllm_omni.diffusion.models.kandinsky6 import KandinskyPiFlowScheduler
    from vllm_omni.diffusion.models.kandinsky6 import pipeline_kandinsky6 as module
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    monkeypatch.setattr(module, "get_local_device", lambda: torch.device("cpu"))
    scheduler = KandinskyPiFlowScheduler()
    scheduler.default_num_frames = 97
    scheduler.default_num_inference_steps = 6
    pipeline = module.Kandinsky6TI2VAPipeline(
        transformer=torch.nn.Identity(), vae=torch.nn.Identity(), text_encoder=torch.nn.Identity(), scheduler=scheduler
    )
    assert pipeline._resolve_generation_settings(OmniDiffusionSamplingParams()) == (480, 864, 97, 6, 1.0)


def test_kandinsky6_pipeline_import_and_registry():
    from vllm_omni.diffusion.models.kandinsky6 import (
        AutoencoderKLHunyuanVideo,
        Kandinsky6AudioVAE,
        Kandinsky6TI2VAPipeline,
        Kandinsky6Transformer3DModel,
        KandinskyFlowMatchScheduler,
        get_kandinsky6_post_process_func,
        get_kandinsky6_pre_process_func,
    )
    from vllm_omni.diffusion.registry import (
        _DIFFUSION_MODELS,
        _DIFFUSION_POST_PROCESS_FUNCS,
        _DIFFUSION_PRE_PROCESS_FUNCS,
        _NO_CACHE_ACCELERATION,
    )

    assert Kandinsky6TI2VAPipeline is not None
    assert Kandinsky6Transformer3DModel is not None
    assert AutoencoderKLHunyuanVideo is not None
    assert Kandinsky6AudioVAE is not None
    assert KandinskyFlowMatchScheduler is not None
    assert get_kandinsky6_post_process_func is not None
    assert get_kandinsky6_pre_process_func is not None

    assert _DIFFUSION_MODELS["Kandinsky6TI2VAPipeline"] == (
        "kandinsky6",
        "pipeline_kandinsky6",
        "Kandinsky6TI2VAPipeline",
    )
    assert _DIFFUSION_POST_PROCESS_FUNCS["Kandinsky6TI2VAPipeline"] == "get_kandinsky6_post_process_func"
    assert _DIFFUSION_PRE_PROCESS_FUNCS["Kandinsky6TI2VAPipeline"] == "get_kandinsky6_pre_process_func"
    assert "Kandinsky6TI2VAPipeline" not in _NO_CACHE_ACCELERATION


def test_kandinsky6_component_discovery_declarations():
    from vllm_omni.diffusion.models.kandinsky6 import Kandinsky6TI2VAPipeline

    assert Kandinsky6TI2VAPipeline._dit_modules == ["transformer"]
    assert Kandinsky6TI2VAPipeline._encoder_modules == ["text_encoder", "text_encoder_2"]
    assert Kandinsky6TI2VAPipeline._vae_modules == ["vae", "audio_vae"]
    assert Kandinsky6TI2VAPipeline.supports_step_execution is True
    assert Kandinsky6TI2VAPipeline.support_audio_output is True
    assert Kandinsky6TI2VAPipeline.support_image_input is True


def test_kandinsky6_pipeline_satisfies_capability_protocols():
    """isinstance checks against the @runtime_checkable protocols the
    framework actually uses to detect these capabilities (io_support.py's
    supports_audio_output, etc.) — stronger than just checking the class
    attribute exists."""
    from vllm_omni.diffusion.models.interface import (
        SupportAudioOutput,
        SupportImageInput,
        SupportsComponentDiscovery,
    )
    from vllm_omni.diffusion.models.kandinsky6 import Kandinsky6TI2VAPipeline

    assert isinstance(Kandinsky6TI2VAPipeline, SupportAudioOutput)
    assert isinstance(Kandinsky6TI2VAPipeline, SupportImageInput)
    assert isinstance(Kandinsky6TI2VAPipeline, SupportsComponentDiscovery)


class _FakeAudioVAE:
    downsample_factor = 1024


def test_kandinsky6_post_process_func_packages_video_and_audio():
    """Pure-function test: no model/weights needed. Confirms the post-process
    payload shape matches what io_support.py -> output_formatter.py ->
    media_utils.py's PyAV muxing expects (the same flat {"video", "audio",
    "audio_sample_rate", "fps"} shape MiniMax H3's own post-process function
    produces), and that int16 PCM from ``postprocess_audio`` is rescaled to
    the float32 [-1, 1] waveform the muxer consumes."""
    from vllm_omni.diffusion.models.kandinsky6 import get_kandinsky6_post_process_func

    post_process = get_kandinsky6_post_process_func(od_config=None)
    video = np.zeros((1, 4, 8, 8, 3), dtype=np.uint8)
    audio = np.array([0, 32767, -32767], dtype=np.int16)

    result = post_process({"video": video, "audio": audio, "audio_sample_rate": 44100}, output_type="np")

    assert result["video"] is video
    assert result["audio"].dtype == np.float32
    np.testing.assert_allclose(result["audio"], [0.0, 1.0, -1.0])
    assert result["audio_sample_rate"] == 44100
    assert result["fps"] == 24.0


def test_kandinsky6_post_process_sample_rate_survives_output_formatter():
    """The formatter only lifts ``audio_sample_rate`` into metadata from the
    flat payload form; an envelope would silently lose it and the MP4 would
    be muxed at the 24 kHz default."""
    from vllm_omni.diffusion.models.kandinsky6 import get_kandinsky6_post_process_func
    from vllm_omni.diffusion.output_formatter import normalize_diffusion_postprocess_output

    post_process = get_kandinsky6_post_process_func(od_config=None)
    video = np.zeros((1, 2, 4, 4, 3), dtype=np.uint8)
    audio = np.zeros((10,), dtype=np.int16)

    normalized = normalize_diffusion_postprocess_output(
        post_process({"video": video, "audio": audio, "audio_sample_rate": 44100}, output_type="np")
    )

    assert normalized.primary_key == "video"
    assert normalized.metadata["audio"]["sample_rate"] == 44100
    assert normalized.metadata["video"]["fps"] == 24.0
    assert "audio_sample_rate" not in normalized.outputs


def test_kandinsky6_post_process_func_omits_audio_when_none():
    from vllm_omni.diffusion.models.kandinsky6 import get_kandinsky6_post_process_func

    post_process = get_kandinsky6_post_process_func(od_config=None)
    video = np.zeros((1, 2, 4, 4, 3), dtype=np.uint8)

    result = post_process({"video": video, "audio": None, "audio_sample_rate": None}, output_type="np")

    assert result["video"] is video
    assert "audio" not in result
    assert "audio_sample_rate" not in result


def test_kandinsky6_post_process_func_unwraps_batched_audio_list():
    from vllm_omni.diffusion.models.kandinsky6 import get_kandinsky6_post_process_func

    post_process = get_kandinsky6_post_process_func(od_config=None)
    video = np.zeros((1, 2, 4, 4, 3), dtype=np.uint8)
    audio_item = np.full((50,), 16384, dtype=np.int16)

    result = post_process({"video": video, "audio": [audio_item], "audio_sample_rate": 44100}, output_type="np")

    assert result["audio"].shape == (50,)
    np.testing.assert_allclose(result["audio"], 16384 / 32767, rtol=1e-6)


class _StubAudioVAE:
    """Audio VAE stand-in: identity "decode" that returns the latent as a waveform."""

    scaling_factor = 0.5
    mean_value = 0.0
    device = "cpu"

    def wrapped_decode(self, latents):
        # (1, audio_dim, A) -> (A,) waveform: take the first channel.
        return latents[0, 0]


def test_postprocess_audio_normalizes_by_default_and_clips_on_request():
    """The default mode must match the k6_video production pipeline, which
    peak-normalizes each decoded waveform to full scale; ``clip`` keeps the
    raw amplitude (saturated to [-1, 1]) for callers that want it."""
    import torch

    from vllm_omni.diffusion.models.kandinsky6.pipeline_kandinsky6 import LatentBundle, postprocess_audio

    # Latent (A=4, audio_dim=1); the stub scales by 1/scaling_factor -> [0.2, -0.4, 0.6, 3.0].
    audio = torch.tensor([[0.1], [-0.2], [0.3], [1.5]], dtype=torch.float32)
    bundle = LatentBundle(
        video=None,
        audio=audio,
        video_cu_seqlens=None,
        audio_cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
    )
    vae = _StubAudioVAE()

    normalized = postprocess_audio(bundle, vae)
    assert normalized is not None and len(normalized) == 1
    assert normalized[0].dtype == np.int16
    np.testing.assert_allclose(
        normalized[0].astype(np.float32) / 32767,
        np.array([0.2, -0.4, 0.6, 3.0], dtype=np.float32) / 3.0,
        atol=1e-4,
    )

    clipped = postprocess_audio(bundle, vae, normalization_mode="clip")
    np.testing.assert_allclose(
        clipped[0].astype(np.float32) / 32767,
        np.array([0.2, -0.4, 0.6, 1.0], dtype=np.float32),
        atol=1e-4,
    )

    assert postprocess_audio(LatentBundle(None, None, None, None), vae) is None
    with pytest.raises(ValueError, match="normalization_mode"):
        postprocess_audio(bundle, vae, normalization_mode="loud")


@pytest.mark.parametrize(
    "scheduler_config,expected_default",
    [
        ({"_class_name": "FlowMatchEulerDiscreteScheduler"}, 50),
        ({"_class_name": "PiflowScheduler"}, 10),
        ({"_class_name": "PiFlowScheduler", "default_num_inference_steps": 7}, 7),
    ],
)
@pytest.mark.parametrize("schedule", ["default", "explicit", "timesteps", "sigmas"])
def test_checkpoint_steps_resolve_before_step_scheduler_admission(
    tmp_path, scheduler_config, expected_default, schedule
):
    import json

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.kandinsky6 import get_kandinsky6_pre_process_func
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.sched.step_scheduler import StepScheduler
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    scheduler_dir = tmp_path / "scheduler"
    scheduler_dir.mkdir()
    (scheduler_dir / "scheduler_config.json").write_text(json.dumps(scheduler_config))
    sampling = OmniDiffusionSamplingParams(num_inference_steps=None)
    expected_steps = expected_default
    if schedule == "explicit":
        sampling.num_inference_steps = expected_steps = 3
    elif schedule == "timesteps":
        sampling.timesteps = [1000.0, 500.0]
        expected_steps = 2
    elif schedule == "sigmas":
        sampling.sigmas = [1.0, 0.5, 0.1]
        expected_steps = 3
    request = OmniDiffusionRequest(prompt={"prompt": "A fox in snow"}, sampling_params=sampling, request_id="k6")
    pre_process = get_kandinsky6_pre_process_func(OmniDiffusionConfig(model=str(tmp_path), step_execution=True))
    scheduler = StepScheduler()
    assert pre_process(request) is request
    assert scheduler.add_request(request) == "k6"
    assert scheduler._request_progress["k6"].total_steps == expected_steps
    if schedule in ("timesteps", "sigmas"):
        assert sampling.num_inference_steps is None


def test_hub_component_keys_map_onto_pipeline_modules():
    """Hub folder keys from Kandinsky-6.0-Pro-5s-Diffusers land on the modules."""
    from vllm_omni.diffusion.models.kandinsky6.pipeline_kandinsky6 import (
        _WEIGHT_SUBFOLDERS,
        _adapt_k6_weight_name,
    )

    assert _WEIGHT_SUBFOLDERS == (
        ("transformer", "transformer."),
        ("vae", "vae."),
        ("text_encoder", "text_encoder."),
        ("text_encoder_2", "text_encoder_2."),
        ("audio_vae", "audio_vae."),
    )
    assert _adapt_k6_weight_name("transformer.visual_transformer_blocks.0.videoT.self_attention.to_query.weight") == (
        "transformer.visual_transformer_blocks.0.video_dec_block.self_attention.to_query.weight"
    )
    assert _adapt_k6_weight_name("transformer.visual_transformer_blocks.0.audioT.feed_forward.in_layer.weight") == (
        "transformer.visual_transformer_blocks.0.audio_dec_block.feed_forward.in_layer.weight"
    )
    assert _adapt_k6_weight_name("transformer.audio_text_transformer_blocks.0.attn.to_query.weight") == (
        "transformer.audio_text_transformer_blocks.0.self_attention.to_query.weight"
    )
    assert _adapt_k6_weight_name(
        "transformer.visual_transformer_blocks.0.video_dec_block.feed_forward.net.0.proj.weight"
    ) == ("transformer.visual_transformer_blocks.0.video_dec_block.feed_forward.in_layer.weight")
    assert _adapt_k6_weight_name(
        "transformer.visual_transformer_blocks.0.video_dec_block.feed_forward.net.2.weight"
    ) == ("transformer.visual_transformer_blocks.0.video_dec_block.feed_forward.out_layer.weight")
    assert _adapt_k6_weight_name("transformer.video_time_embeddings.timestep_embedder.linear_1.weight") == (
        "transformer.video_time_embeddings.in_layer.weight"
    )
    assert _adapt_k6_weight_name("transformer.audio_time_embeddings.timestep_embedder.linear_2.bias") == (
        "transformer.audio_time_embeddings.out_layer.bias"
    )
    assert _adapt_k6_weight_name("vae.decoder.conv_in.conv.weight") == "vae.decoder.conv_in.conv.weight"
    assert _adapt_k6_weight_name("text_encoder_2.encoder.layers.0.mlp.fc1.weight") == (
        "text_encoder_2.encoder.layers.0.mlp.fc1.weight"
    )
    assert (
        _adapt_k6_weight_name("text_encoder.model.layers.0.input_layernorm.weight")
        == "text_encoder.model.language_model.layers.0.input_layernorm.weight"
    )
    assert _adapt_k6_weight_name("text_encoder.visual.blocks.0.norm1.weight") == (
        "text_encoder.model.visual.blocks.0.norm1.weight"
    )
    assert (
        _adapt_k6_weight_name("audio_vae.vae.decoder.conv_in.weight")
        == "audio_vae.native.tod.vae.decoder.conv_in.weight"
    )
    assert _adapt_k6_weight_name("audio_vae.vocoder.conv_pre.weight") == "audio_vae.native.tod.vocoder.conv_pre.weight"
    assert _adapt_k6_weight_name("audio_vae.mel_converter.hann_window") == "audio_vae.native.mel_converter.hann_window"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shift", [1.0, 5.0])
def test_piflow_rollout_matches_diffusers_reference(dtype, shift):
    from vllm_omni.diffusion.models.kandinsky6 import KandinskyPiFlowScheduler
    from vllm_omni.diffusion.models.kandinsky6_sr.scheduling_piflow import PiflowScheduler

    generator = torch.Generator().manual_seed(42)
    sample = torch.randn(2, 3, 4, generator=generator).to(dtype)
    scheduler = KandinskyPiFlowScheduler(scheduler_scale=shift)
    reference = PiflowScheduler(shift=shift)
    scheduler.set_timesteps(10)
    reference.set_timesteps(10)
    for index, timestep in enumerate(reference.timesteps):
        prediction = torch.randn(2, 3, 40, generator=generator).to(dtype)
        expected = reference.step(prediction, timestep, sample).prev_sample
        actual = scheduler._rollout(prediction, sample, index)
        assert actual.dtype == dtype
        # Equivalent FP32 interpolation expressions differ by rounding; BF16
        # results should agree exactly after the final cast.
        tolerance = 2e-6 if dtype == torch.float32 else 0
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        sample = expected
