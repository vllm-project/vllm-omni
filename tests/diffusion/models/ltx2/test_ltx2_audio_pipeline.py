# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for the unified LTX text-to-audio pipeline."""

import fnmatch
import json
import math
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.interface import SupportAudioOutput
from vllm_omni.diffusion.models.ltx2 import ltx2_latents
from vllm_omni.diffusion.models.ltx2.ltx2_audio_runtime import LTXAudioRuntime
from vllm_omni.diffusion.models.ltx2.ltx2_components import (
    LTX2_T2A_COMPONENT_PROFILE,
    LTX23_T2A_COMPONENT_PROFILE,
    LTX25_T2A_COMPONENT_PROFILE,
    create_audio_transformer_from_config,
    get_ltx2_audio_post_process_func,
    resolve_ltx_component_profile,
)
from vllm_omni.diffusion.models.ltx2.ltx2_guidance import LTXGuidancePlan
from vllm_omni.diffusion.models.ltx2.ltx2_recipes import (
    LTX2_T2A_RECIPE,
    LTX23_T2A_RECIPE,
    LTX25_DEFAULT_NEGATIVE_PROMPT,
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
    assert not recipe.supports_cache_dit
    assert recipe.request_guidance.audio.modality_scale == 1.0
    assert LTXGuidancePlan.build(recipe.request_guidance).names == ("cond", "uncond", "ptb")


def test_ltx2_and_ltx23_t2a_defaults_match_official_negative_prompt():
    assert LTX2_T2A_RECIPE.negative_prompt == LTX25_DEFAULT_NEGATIVE_PROMPT
    assert LTX23_T2A_RECIPE.negative_prompt == LTX25_DEFAULT_NEGATIVE_PROMPT


def test_ltx_t2a_public_contract_is_audio_only():
    assert SupportAudioOutput in LTX2TextToAudioPipeline.__mro__
    assert LTX2TextToAudioPipeline.pipeline_kind == "text_to_audio"
    assert LTX2TextToAudioPipeline.support_audio_output
    assert not LTX2TextToAudioPipeline.support_image_input
    assert not hasattr(LTX2TextToAudioPipeline, "support_video_output")
    assert LTX2TextToAudioPipeline.dummy_run_num_frames == 9


def test_ltx_t2a_setup_compile_prepares_norms_before_regional_compile(monkeypatch):
    calls = []
    transformer = SimpleNamespace(prepare_regional_compile=lambda: calls.append("prepare"))
    pipe = object.__new__(LTX2TextToAudioPipeline)
    object.__setattr__(pipe, "transformer", transformer)
    object.__setattr__(pipe, "od_config", SimpleNamespace(diffusion_compile_dynamic=True))

    def fake_compile(model, **kwargs):
        calls.append((model, kwargs))
        return "compiled"

    monkeypatch.setattr("vllm_omni.diffusion.models.ltx2.pipeline_ltx2_audio.regionally_compile", fake_compile)
    pipe.setup_compile()

    assert calls == ["prepare", (transformer, {"dynamic": True, "options": {"emulate_precision_casts": True}})]
    assert pipe.transformer == "compiled"


@pytest.mark.parametrize(
    "parallel_config",
    [
        SimpleNamespace(tensor_parallel_size=2, sequence_parallel_size=1),
        SimpleNamespace(tensor_parallel_size=1, sequence_parallel_size=2),
    ],
)
def test_ltx_t2a_rejects_tensor_or_sequence_parallelism(parallel_config):
    od_config = SimpleNamespace(parallel_config=parallel_config)

    with pytest.raises(
        ValueError,
        match=r"currently supports only tensor_parallel_size=1 and sequence_parallel_size=1; "
        r"TP/SP execution is not supported for audio-only T2A",
    ):
        LTX2TextToAudioPipeline(od_config=od_config)


def test_ltx_t2a_runs_only_audio_connector_for_per_modality_projection():
    class AudioConnector:
        def __init__(self):
            self.calls = []

        def __call__(self, hidden_states, attention_mask):
            self.calls.append((hidden_states, attention_mask))
            return hidden_states + 1, torch.zeros_like(attention_mask)

    audio_connector = AudioConnector()
    connectors = SimpleNamespace(
        config=SimpleNamespace(
            per_modality_projections=True,
            caption_channels=2,
            audio_hidden_dim=4,
        ),
        audio_text_proj_in=torch.nn.Linear(6, 4, bias=True),
        audio_connector=audio_connector,
    )
    pipe = object.__new__(LTXAudioRuntime)
    torch.nn.Module.__init__(pipe)
    object.__setattr__(pipe, "connectors", connectors)
    prompt_embeds = torch.arange(24, dtype=torch.float32).reshape(2, 2, 2, 3)
    attention_mask = torch.tensor([[0, 1], [1, 1]])

    video, audio, output_mask = pipe._run_text_connectors(
        prompt_embeds,
        attention_mask,
        padding_side="left",
    )

    variance = torch.mean(prompt_embeds**2, dim=2, keepdim=True)
    normalized = prompt_embeds * torch.rsqrt(variance + 1e-6)
    normalized = normalized.flatten(2, 3)
    normalized = torch.where(attention_mask.bool().unsqueeze(-1), normalized, torch.zeros_like(normalized))
    expected_projection = connectors.audio_text_proj_in(normalized * math.sqrt(2))
    torch.testing.assert_close(audio_connector.calls[0][0], expected_projection)
    torch.testing.assert_close(video, expected_projection + 1)
    torch.testing.assert_close(audio, expected_projection + 1)
    torch.testing.assert_close(output_mask, torch.ones_like(attention_mask))


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


def test_ltx_t2a_duration_limit_applies_before_grid_alignment():
    limits = LTX2AudioResourceLimits()
    num_frames = resolve_ltx_audio_num_frames(
        audio_length=20.1,
        num_frames=None,
        frame_rate=24.0,
        default_num_frames=121,
    )

    limits.validate_requested_duration(20.1)
    assert limits.validate_resolved_duration(
        num_frames=num_frames,
        frame_rate=24.0,
        expected_frame_rate=24.0,
    ) == pytest.approx(20.375)


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
            num_frames=num_frames + 16,
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
        parallel_config=SimpleNamespace(tensor_parallel_size=1, sequence_parallel_size=1),
        cache_backend="none",
        additional_config=None,
        enable_diffusion_pipeline_profiler=False,
    )
    pipe = LTX2TextToAudioPipeline(od_config=od_config)

    assert pipe.model_version == "2.5"
    assert pipe.component_profile is LTX25_T2A_COMPONENT_PROFILE
    assert pipe.pipeline_recipe is LTX25_T2A_RECIPE
    assert not hasattr(pipe, "audio_graph_runner")


@pytest.mark.parametrize("cache_backend", ["cache_dit", "tea_cache"])
def test_ltx_t2a_foundation_rejects_cache_backends(cache_backend):
    od_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, sequence_parallel_size=1),
        cache_backend=cache_backend,
        additional_config=None,
    )

    with pytest.raises(ValueError, match="does not support cache_backend"):
        LTX2TextToAudioPipeline(od_config=od_config)


@pytest.mark.parametrize(("tp_size", "sp_size"), [(2, 1), (1, 2)])
def test_ltx_t2a_foundation_rejects_distributed_execution(tp_size, sp_size):
    od_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size,
            sequence_parallel_size=sp_size,
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"currently supports only tensor_parallel_size=1 and sequence_parallel_size=1; "
        r"TP/SP execution is not supported for audio-only T2A",
    ):
        LTX2TextToAudioPipeline(od_config=od_config)


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
    packed = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)

    waveform = pipe._decode_audio_latents(packed, latent_mel_bins=4)

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

    waveform = pipe._decode_audio_latents(packed, latent_mel_bins=4)

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
        "transformer_blocks.*.audio_attn1.*",
        "transformer_blocks.*.audio_attn2.*",
        "transformer_blocks.*.audio_ff.*",
        "transformer_blocks.*.audio_prompt_scale_shift_table",
        "transformer_blocks.*.audio_scale_shift_table",
    )
    patterns = captured["weight_name_patterns"]
    assert any(fnmatch.fnmatchcase("transformer_blocks.0.audio_attn1.to_q.weight", pattern) for pattern in patterns)
    assert not any(
        fnmatch.fnmatchcase("transformer_blocks.0.audio_to_video_attn.to_q.weight", pattern) for pattern in patterns
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


def test_ltx_t2a_component_cache_healing_stays_audio_only(tmp_path, monkeypatch):
    from vllm_omni.diffusion.models.ltx2 import ltx2_audio_runtime

    prefetch_lists = []

    class Source:
        def __init__(self, **_kwargs):
            pass

    class Scheduler:
        config = {"use_dynamic_shifting": True, "shift_terminal": 0.1}

    connector = SimpleNamespace(config=SimpleNamespace(per_modality_projections=False))
    audio_vae = SimpleNamespace(
        mel_compression_ratio=4,
        temporal_compression_ratio=8,
        config=SimpleNamespace(sample_rate=24000, mel_hop_length=256),
    )

    def load_component(_cls, _model, subfolder, **kwargs):
        prefetch_lists.append(kwargs["prefetch_list"])
        return {
            "text_encoder": SimpleNamespace(config=SimpleNamespace(max_position_embeddings=1024)),
            "connectors": connector,
            "audio_vae": audio_vae,
            "vocoder": SimpleNamespace(),
        }[subfolder]

    monkeypatch.setattr(ltx2_audio_runtime.DiffusersPipelineLoader, "ComponentSource", Source)
    monkeypatch.setattr(ltx2_audio_runtime, "prefetch_subfolders", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        ltx2_audio_runtime.FlowMatchEulerDiscreteScheduler,
        "from_pretrained",
        lambda *_args, **_kwargs: Scheduler(),
    )
    monkeypatch.setattr(
        ltx2_audio_runtime.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(model_max_length=1024),
    )
    monkeypatch.setattr(ltx2_audio_runtime, "_load_component", load_component)
    monkeypatch.setattr(ltx2_audio_runtime, "_install_connector_attention", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ltx2_audio_runtime, "load_transformer_config", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        ltx2_audio_runtime,
        "create_audio_transformer_from_config",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(ltx2_audio_runtime, "_place_aux_components", lambda _pipe: None)

    pipe = SimpleNamespace(
        component_profile=LTX2_T2A_COMPONENT_PROFILE,
        pipeline_kind="text_to_audio",
    )
    od_config = SimpleNamespace(model=str(tmp_path), revision=None, dtype=torch.float32)

    ltx2_audio_runtime.initialize_audio_pipeline_components(pipe, od_config)

    assert prefetch_lists
    assert all(value == ltx2_audio_runtime._LTX_AUDIO_COMPONENT_SUBFOLDERS for value in prefetch_lists)


def test_ltx_t2a_prepare_latents_rejects_sp_padding(monkeypatch):
    pipe = object.__new__(LTXAudioRuntime)
    torch.nn.Module.__init__(pipe)
    monkeypatch.setattr(
        ltx2_latents,
        "prepare_audio_latents",
        lambda *_args, **_kwargs: (torch.zeros(1, 3, 4), 2, 3),
    )

    with pytest.raises(RuntimeError, match="requires sequence-parallel execution"):
        pipe.prepare_audio_latents(
            1,
            4,
            2,
            8,
            noise_scale=0.0,
            dtype=torch.float32,
            device=torch.device("cpu"),
            generator=None,
            latents=None,
        )


def test_ltx25_t2a_drops_unused_video_connector_modules():
    from vllm_omni.diffusion.models.ltx2.ltx2_audio_runtime import _drop_unused_video_connectors

    connectors = SimpleNamespace(
        config=SimpleNamespace(per_modality_projections=True),
        video_text_proj_in=object(),
        video_connector=object(),
        audio_text_proj_in=object(),
        audio_connector=object(),
    )
    audio_modules = (connectors.audio_text_proj_in, connectors.audio_connector)

    _drop_unused_video_connectors(connectors)

    assert connectors.video_text_proj_in is None
    assert connectors.video_connector is None
    assert (connectors.audio_text_proj_in, connectors.audio_connector) == audio_modules
