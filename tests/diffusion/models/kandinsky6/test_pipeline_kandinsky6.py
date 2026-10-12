# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import numpy as np
import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


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


def test_kandinsky6_pre_process_func_is_identity_for_now():
    """v1 scope: the pre-process hook is registered (matching the framework's
    convention for image-conditioned models like SanaImageToVideoPipeline)
    but does not yet validate/transform the request — documented follow-up,
    not a silent gap."""
    from vllm_omni.diffusion.models.kandinsky6 import get_kandinsky6_pre_process_func

    pre_process = get_kandinsky6_pre_process_func(od_config=None)
    sentinel = object()

    assert pre_process(sentinel) is sentinel


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
