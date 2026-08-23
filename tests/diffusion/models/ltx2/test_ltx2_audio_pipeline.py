# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the unified LTX text-to-audio pipeline."""

import json
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
from vllm_omni.diffusion.models.ltx2.ltx2_request import resolve_ltx_audio_num_frames
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
        (0.5, 24.0, 9),
        (1.0, 25.0, 25),
        (5.1, 24.0, 121),
        (5.21, 24.0, 129),
    ],
)
def test_ltx_t2a_duration_quantizes_to_nearest_legal_video_clock(seconds, frame_rate, expected):
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


@pytest.mark.parametrize(
    ("audio_length", "num_frames", "frame_rate", "error"),
    [
        (1.0, 25, 24.0, "mutually exclusive"),
        (0.0, None, 24.0, "positive"),
        (1.0, None, 0.0, "frame_rate"),
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


def test_ltx_t2a_expected_version_is_an_assertion(tmp_path, monkeypatch):
    from vllm_omni.diffusion.models.ltx2 import ltx2_audio_runtime

    (tmp_path / "model_index.json").write_text(json.dumps({"model_version": "2.3"}))
    initialized = False

    def stub_components(*_args, **_kwargs):
        nonlocal initialized
        initialized = True

    monkeypatch.setattr(ltx2_audio_runtime, "initialize_audio_pipeline_components", stub_components)

    od_config = SimpleNamespace(
        model=str(tmp_path),
        revision=None,
        expected_model_version="2.5",
        parallel_config=SimpleNamespace(ulysses_mode="strict"),
        cache_backend="none",
    )
    with pytest.raises(ValueError, match="expected LTX model version '2.5'.*detected '2.3'"):
        LTX2TextToAudioPipeline(od_config=od_config)

    assert not initialized


def test_ltx_t2a_matching_expected_version_keeps_detected_profile(tmp_path, monkeypatch):
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
        expected_model_version="2.5",
        parallel_config=SimpleNamespace(ulysses_mode="strict"),
        cache_backend="none",
        enable_diffusion_pipeline_profiler=False,
    )
    pipe = LTX2TextToAudioPipeline(od_config=od_config)

    assert pipe.model_version == "2.5"
    assert pipe.component_profile is LTX25_T2A_COMPONENT_PROFILE
    assert pipe.pipeline_recipe is LTX25_T2A_RECIPE


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
    assert "LTX2TextToAudioPipeline" in _NO_CACHE_ACCELERATION


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


def test_ltx_t2a_denoise_passes_audio_padding_mask_without_video_inputs(monkeypatch):
    from vllm_omni.diffusion.models.ltx2 import ltx2_audio_runtime

    calls = []

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
    prompt_context = SimpleNamespace(
        positive_connector_audio_prompt_embeds=torch.zeros(1, 2, 4),
        negative_connector_audio_prompt_embeds=None,
    )
    inputs = SimpleNamespace(num_inference_steps=1)
    latents = torch.ones(1, 3, 4)

    result = pipe._run_audio_denoise(
        latents,
        prompt_context,
        inputs,
        original_num_frames=2,
        padded_num_frames=3,
        request_sigmas=[1.0, 0.0],
    )

    assert not {"hidden_states", "encoder_hidden_states", "video_coords"} & calls[0].keys()
    torch.testing.assert_close(calls[0]["audio_attention_mask"], torch.tensor([[True, True, False]]))
    torch.testing.assert_close(result[:, :2], latents[:, :2])
    torch.testing.assert_close(result[:, 2:], torch.zeros_like(result[:, 2:]))
