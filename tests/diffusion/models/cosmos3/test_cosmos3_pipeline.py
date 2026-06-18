# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image
from torch import nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class StubScheduler:
    def __init__(self, timesteps: list[int] | None = None, *, flow_shift: float = 1.0) -> None:
        self.timesteps = torch.tensor(timesteps or [9, 3], dtype=torch.int64)
        self.config = SimpleNamespace(num_train_timesteps=1000, flow_shift=flow_shift)
        self.set_timesteps_calls: list[tuple[int, torch.device]] = []
        self.step_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def set_timesteps(self, num_steps: int, device: torch.device) -> None:
        self.set_timesteps_calls.append((num_steps, device))
        self.timesteps = torch.arange(num_steps, 0, -1, dtype=torch.int64, device=device)

    def step(self, noise_pred: torch.Tensor, timestep: torch.Tensor, latents: torch.Tensor, **kwargs):
        del kwargs
        self.step_calls.append((noise_pred.clone(), timestep.clone(), latents.clone()))
        return (latents + noise_pred,)


class _ModeLatentDist:
    def __init__(self, latents: torch.Tensor) -> None:
        self._latents = latents

    def mode(self) -> torch.Tensor:
        return self._latents


class StubCosmos3VAE:
    dtype = torch.float32

    def __init__(self, z_dim: int = 2, *, temporal: int = 4, spatial: int = 8) -> None:
        self.config = SimpleNamespace(
            z_dim=z_dim,
            scale_factor_temporal=temporal,
            scale_factor_spatial=spatial,
            latents_mean=[0.0] * z_dim,
            latents_std=[1.0] * z_dim,
        )

    def encode(self, video: torch.Tensor):
        latent_frames = (video.shape[2] - 1) // self.config.scale_factor_temporal + 1
        latent_height = video.shape[-2] // self.config.scale_factor_spatial
        latent_width = video.shape[-1] // self.config.scale_factor_spatial
        latents = torch.ones(
            video.shape[0],
            self.config.z_dim,
            latent_frames,
            latent_height,
            latent_width,
            dtype=video.dtype,
            device=video.device,
        )
        return SimpleNamespace(latent_dist=_ModeLatentDist(latents))

    def decode(self, latents: torch.Tensor, return_dict: bool = False):
        del return_dict
        return (latents,)


class StubCosmos3AVAE:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.sample_rate = int(kwargs["sample_rate"])
        self.audio_channels = int(kwargs["audio_channels"])
        self.latent_ch = int(kwargs["io_channels"])
        self.temporal_compression_factor = int(kwargs["hop_size"])

    def get_latent_num_samples(self, num_audio_samples: int) -> int:
        return int(num_audio_samples) // self.temporal_compression_factor

    def get_audio_num_samples(self, num_latent_samples: int) -> int:
        return int(num_latent_samples) * self.temporal_compression_factor

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.zeros(latents.shape[0], self.audio_channels, 8)


class StubCosmos3Transformer(nn.Module):
    def __init__(
        self,
        *,
        latent_channel_size: int = 2,
        sound_gen: bool = False,
        sound_dim: int = 3,
        sound_latent_fps: float = 25.0,
        action_gen: bool = False,
        action_dim: int = 4,
    ) -> None:
        super().__init__()
        self.latent_channel_size = latent_channel_size
        self.sound_gen = sound_gen
        self.sound_dim = sound_dim
        self.sound_latent_fps = sound_latent_fps
        self.action_gen = action_gen
        self.action_dim = action_dim
        self.cached_kv: Any | None = None
        self.cached_freqs_gen: Any | None = None
        self.calls: list[dict[str, Any]] = []
        self.reset_calls = 0

    def reset_cache(self) -> None:
        self.reset_calls += 1
        self.cached_kv = None
        self.cached_freqs_gen = None

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        token = int(text_ids.reshape(-1)[0].item()) if text_ids.numel() else 0
        sound_latents = kwargs.get("sound_latents")
        self.calls.append(
            {
                "token": token,
                "timestep": timestep.clone(),
                "text_mask": text_mask.clone(),
                "cache_before": self.cached_kv,
                "kwargs": dict(kwargs),
            }
        )
        if self.cached_kv is None:
            marker = torch.tensor([token], dtype=torch.float32)
            self.cached_kv = [(marker, marker + 100)]
            self.cached_freqs_gen = (marker + 200, marker + 300)
        action_latents = kwargs.get("action_latents")
        outputs: list[torch.Tensor] = [torch.full_like(hidden_states, float(token))]
        if action_latents is not None:
            outputs.append(torch.full_like(action_latents, float(token + 20)))
        if sound_latents is not None:
            outputs.append(torch.full_like(sound_latents, float(token + 10)))
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


def passthrough_progress_bar(iterable):
    return iterable


@pytest.fixture(autouse=True)
def fake_cosmos3_guardrails(monkeypatch: pytest.MonkeyPatch):
    module = types.ModuleType("vllm_omni.diffusion.models.cosmos3.guardrails")
    module.is_guardrails_enabled = lambda od_config, sampling_params=None: False
    module.ensure_initialized = lambda od_config: None
    module.check_text_safety = lambda text: None
    module.check_video_safety = lambda video: video
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


@pytest.fixture
def make_cosmos3_pipeline():
    def _make():
        from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
            Cosmos3OmniDiffusersPipeline,
        )

        pipeline = object.__new__(Cosmos3OmniDiffusersPipeline)
        nn.Module.__init__(pipeline)
        pipeline.od_config = SimpleNamespace()
        pipeline.device = torch.device("cpu")
        pipeline.dtype = torch.float32
        pipeline.transformer = StubCosmos3Transformer(latent_channel_size=2)
        pipeline.vae = StubCosmos3VAE(z_dim=2)
        pipeline.vae_scale_factor_temporal = 4
        pipeline.vae_scale_factor_spatial = 8
        pipeline.scheduler = StubScheduler([9, 3], flow_shift=1.0)
        pipeline._base_scheduler_config = pipeline.scheduler.config
        pipeline._engine_init_flow_shift = 1.0
        pipeline._current_flow_shift = 1.0
        pipeline._guidance_scale = None
        pipeline._num_timesteps = None
        pipeline._cache_dit_requires_paired_cfg = False
        pipeline._sound_tokenizer = None
        pipeline.progress_bar = passthrough_progress_bar
        return pipeline

    return _make


def make_sampling_params(**overrides: Any) -> SimpleNamespace:
    values = {
        "height": None,
        "width": None,
        "num_frames": None,
        "num_inference_steps": None,
        "guidance_scale": None,
        "generator": None,
        "seed": 123,
        "num_outputs_per_prompt": 1,
        "frame_rate": None,
        "resolved_frame_rate": None,
        "max_sequence_length": None,
        "extra_args": {},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _ids(value: int) -> torch.Tensor:
    return torch.tensor([[value]], dtype=torch.long)


def _mask() -> torch.Tensor:
    return torch.ones(1, 1, dtype=torch.long)


def test_pipeline_registered_and_exported() -> None:
    from vllm_omni.diffusion.cache.cache_dit_backend import CUSTOM_DIT_ENABLERS
    from vllm_omni.diffusion.models import cosmos3
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline
    from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
    from vllm_omni.diffusion.registry import (
        _DIFFUSION_ACTION_POST_PROCESS_FUNCS,
        _DIFFUSION_IR_OP_PRIORITY_FUNCS,
        _DIFFUSION_MODELS,
        _DIFFUSION_POST_PROCESS_FUNCS,
        _DIFFUSION_PRE_PROCESS_FUNCS,
    )

    assert issubclass(Cosmos3OmniDiffusersPipeline, nn.Module)
    assert issubclass(Cosmos3OmniDiffusersPipeline, ProgressBarMixin)
    assert Cosmos3OmniDiffusersPipeline.support_image_input is True
    assert _DIFFUSION_MODELS["Cosmos3OmniDiffusersPipeline"] == (
        "cosmos3",
        "pipeline_cosmos3",
        "Cosmos3OmniDiffusersPipeline",
    )
    assert _DIFFUSION_PRE_PROCESS_FUNCS["Cosmos3OmniDiffusersPipeline"] == "get_cosmos3_pre_process_func"
    assert _DIFFUSION_POST_PROCESS_FUNCS["Cosmos3OmniDiffusersPipeline"] == "get_cosmos3_post_process_func"
    assert (
        _DIFFUSION_ACTION_POST_PROCESS_FUNCS["Cosmos3OmniDiffusersPipeline"] == "get_cosmos3_action_post_process_func"
    )
    assert _DIFFUSION_IR_OP_PRIORITY_FUNCS["Cosmos3OmniDiffusersPipeline"] == "get_cosmos3_ir_op_priority_func"
    assert "Cosmos3OmniDiffusersPipeline" in CUSTOM_DIT_ENABLERS
    assert "Cosmos3OmniDiffusersPipeline" in cosmos3.__all__


@pytest.fixture
def stub_real_pipeline_init(monkeypatch: pytest.MonkeyPatch):
    from vllm_omni.diffusion.models.cosmos3 import pipeline_cosmos3

    class _StubAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return SimpleNamespace()

    class _StubDiffusersVAE:
        config = SimpleNamespace(scale_factor_temporal=4, scale_factor_spatial=8)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def to(self, _device):
            return self

    class _StubDiffusersScheduler:
        config = SimpleNamespace(flow_shift=1.0)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class _StubVideoProcessor:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(pipeline_cosmos3, "AutoTokenizer", _StubAutoTokenizer)
    monkeypatch.setattr(pipeline_cosmos3, "DistributedAutoencoderKLWan", _StubDiffusersVAE)
    monkeypatch.setattr(pipeline_cosmos3, "UniPCMultistepScheduler", _StubDiffusersScheduler)
    monkeypatch.setattr(pipeline_cosmos3, "VideoProcessor", _StubVideoProcessor)
    monkeypatch.setattr(pipeline_cosmos3, "get_local_device", lambda: torch.device("cpu"))


def _make_od_config(*, sound_gen: bool) -> SimpleNamespace:
    tf_model_config = {
        "hidden_size": 8,
        "num_hidden_layers": 0,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "intermediate_size": 16,
        "vocab_size": 32,
        "latent_patch_size": 1,
        "latent_channel": 2,
        "rope_scaling": {"mrope_section": [1, 1, 0]},
    }
    if sound_gen:
        tf_model_config["sound_gen"] = True
    return SimpleNamespace(
        enable_cpu_offload=False,
        enable_diffusion_pipeline_profiler=False,
        model="/nonexistent/model/path",
        dtype=torch.float32,
        flow_shift=None,
        quantization_config=None,
        custom_pipeline_args={},
        model_config={},
        tf_model_config=tf_model_config,
    )


def test_pipeline_init_skips_tokenizer_when_sound_disabled(stub_real_pipeline_init) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline

    pipeline = Cosmos3OmniDiffusersPipeline(od_config=_make_od_config(sound_gen=False))

    assert pipeline._sound_tokenizer is None
    assert pipeline.transformer.sound_gen is False
    assert not hasattr(pipeline.transformer, "audio_proj_in")
    assert not hasattr(pipeline.transformer, "audio_proj_out")


def test_pipeline_init_passes_tokenizer_attrs_into_transformer(
    stub_real_pipeline_init,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_omni.diffusion.models.cosmos3 import sound_tokenizer
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline

    stub_tokenizer = sound_tokenizer.Cosmos3SoundTokenizer(
        StubCosmos3AVAE(sample_rate=32000, audio_channels=2, io_channels=5, hop_size=800)
    )
    monkeypatch.setattr(
        sound_tokenizer.Cosmos3SoundTokenizer,
        "from_config",
        classmethod(lambda cls, od_config: stub_tokenizer),
    )

    pipeline = Cosmos3OmniDiffusersPipeline(od_config=_make_od_config(sound_gen=True))

    assert pipeline._sound_tokenizer is stub_tokenizer
    assert pipeline.transformer.sound_gen is True
    assert pipeline.transformer.sound_dim == pipeline._sound_tokenizer.latent_ch == 5
    assert pipeline.transformer.sound_latent_fps == pipeline._sound_tokenizer.latent_fps == 40.0
    assert pipeline.transformer.audio_proj_in.in_features == 5
    assert pipeline.transformer.audio_proj_out.out_features == 5


def test_preprocess_i2v_image_and_action_video_inputs() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import get_cosmos3_pre_process_func

    preprocess = get_cosmos3_pre_process_func(SimpleNamespace())
    i2v = SimpleNamespace(
        prompts=[{"prompt": "A slow camera push.", "multi_modal_data": {"image": Image.new("RGB", (320, 160))}}],
        sampling_params=SimpleNamespace(height=None, width=None, extra_args={}),
    )

    result = preprocess(i2v)
    assert (result.sampling_params.height, result.sampling_params.width) == (672, 1344)
    assert tuple(result.prompts[0]["additional_information"]["preprocessed_image"].shape[-2:]) == (672, 1344)

    frames = [Image.new("RGB", (8, 4), color) for color in ("red", "green", "blue")]
    action = SimpleNamespace(
        prompts=[{"prompt": "Move.", "multi_modal_data": {"video": frames}}],
        sampling_params=SimpleNamespace(height=16, width=32, extra_args={"action_mode": "forward_dynamics"}),
    )

    additional = preprocess(action).prompts[0]["additional_information"]
    assert tuple(additional["preprocessed_image"].shape) == (1, 3, 16, 32)
    assert tuple(additional["preprocessed_video"].shape) == (1, 3, 3, 16, 32)

    frames = [Image.new("RGB", (8, 4), color) for color in ("red", "green", "blue", "yellow", "purple", "black")]
    v2v = SimpleNamespace(
        prompts=[{"prompt": "Continue.", "multi_modal_data": {"video": frames}}],
        sampling_params=SimpleNamespace(
            height=16,
            width=32,
            extra_args={"condition_frame_indexes_vision": [0, 1], "condition_video_keep": "last"},
        ),
    )
    additional = preprocess(v2v).prompts[0]["additional_information"]
    assert tuple(additional["preprocessed_video"].shape) == (1, 3, 5, 16, 32)
    assert additional["condition_frame_indexes_vision"] == [0, 1]


def test_postprocess_handles_image_video_audio_and_validation() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import get_cosmos3_post_process_func

    func = get_cosmos3_post_process_func(SimpleNamespace())
    video = torch.zeros(1, 3, 1, 4, 4)

    assert func(video, output_type="latent") is video
    assert func({"image": video})[0].size == (4, 4)
    # Video-only postprocess returns the bare processed video (not a dict),
    # matching the image/latent branches and peer audio-capable pipelines.
    assert not isinstance(func({"video": video}), dict)
    assert (
        func(
            {"video": video, "audio": torch.ones(1, 2, 16), "audio_sample_rate": 48000},
            sampling_params=SimpleNamespace(extra_args={"resolved_frame_rate": 12}),
        )["audio_sample_rate"]
        == 48000
    )

    with pytest.raises(ValueError, match="text-to-image postprocess expects"):
        func({"image": torch.zeros(1, 3, 2, 4, 4)})
    with pytest.raises(ValueError, match="both image and video"):
        func({"image": video, "video": video})


def test_action_postprocess_handles_robolab_policy_outputs() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
        RoboLabPolicyInputs,
        get_cosmos3_action_post_process_func,
        make_robolab_action_postprocess_inputs,
    )

    func = get_cosmos3_action_post_process_func(SimpleNamespace())
    inputs = RoboLabPolicyInputs(
        prompt="Pick the cube.",
        video_tensor=torch.zeros(1, 3, 3, 16, 16),
        action_tensor=torch.zeros(2, 2),
        action_condition_indexes=[0],
        action_start_frame_offset=1,
        raw_action_dim=2,
        domain_id=7,
        fps=15.0,
        height=16,
        width=16,
        image_size=None,
        num_frames=3,
        num_inference_steps=4,
        guidance_scale=3.0,
        flow_shift=5.0,
        seed=11,
        history_length=1,
        action_space="joint_pos",
        observation={},
    )

    action = torch.tensor([[[0.0, 0.25], [1.0, 0.75]]])
    custom_output = {"robolab_action_postprocess": make_robolab_action_postprocess_inputs(inputs)}
    processed = func(action, custom_output=custom_output)

    assert processed.shape == (1, 2)
    assert processed.dtype == torch.zeros((), dtype=torch.float32).numpy().dtype
    torch.testing.assert_close(torch.from_numpy(processed), torch.tensor([[1.0, 0.25]]))
    assert "robolab_action_postprocess" not in custom_output


def test_ir_op_priority_hook_preserves_platform_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import get_cosmos3_ir_op_priority_func

    @dataclass
    class FakeIrOpPriorityConfig:
        rms_norm: list[str]
        fused_add_rms_norm: list[str]
        custom_op: list[str]

    fake_kernel = types.ModuleType("vllm.config.kernel")
    fake_kernel.IrOpPriorityConfig = FakeIrOpPriorityConfig
    monkeypatch.setitem(sys.modules, fake_kernel.__name__, fake_kernel)

    func = get_cosmos3_ir_op_priority_func(SimpleNamespace())
    default_priority = FakeIrOpPriorityConfig(
        rms_norm=["vllm_c", "native"],
        fused_add_rms_norm=["vllm_c", "native"],
        custom_op=["platform_kernel", "native"],
    )

    merged = func(default_priority, vllm_config=SimpleNamespace())

    assert merged.rms_norm == ["native"]
    assert merged.fused_add_rms_norm == ["native"]
    assert merged.custom_op == ["platform_kernel", "native"]


def test_prompt_formatting_and_checkpoint_key_remap(make_cosmos3_pipeline) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline

    pipeline = make_cosmos3_pipeline()
    captured: list[str] = []
    pipeline._tokenize_prompt = lambda text, *args, **kwargs: (captured.append(text) or _ids(len(captured)), _mask())

    pipeline._format_and_tokenize_prompts(
        "A robot",
        "bad",
        num_frames=48,
        frame_rate=24,
        height=720,
        width=1280,
        max_sequence_length=32,
        sp=SimpleNamespace(
            extra_args={
                "negative_metadata_mode": "inverse",
                # Duration/resolution metadata templates are off by default
                # (commit "Change resolution and duration templates to off by
                # default"); enable them explicitly to exercise the formatting.
                "use_duration_template": True,
                "use_resolution_template": True,
            }
        ),
        use_system_prompt=True,
        is_t2i=False,
    )
    assert "The video is 2.0 seconds long" in captured[0]
    assert "The video is not 2.0 seconds long" in captured[1]

    remaps = {
        "embed_tokens.weight": "transformer.language_model.embed_tokens.weight",
        "model.embed_tokens.weight": "transformer.language_model.embed_tokens.weight",
        "norm.weight": "transformer.language_model.norm.weight",
        "norm_moe_gen.weight": "transformer.norm_moe_gen.weight",
        "proj_in.weight": "transformer.proj_in.weight",
        "proj_out.bias": "transformer.proj_out.bias",
        "layers.3.self_attn.to_q.weight": "transformer.language_model.layers.3.self_attn.to_q.weight",
        "layers.3.self_attn.to_out.weight": "transformer.language_model.layers.3.self_attn.to_out.weight",
        "layers.3.self_attn.norm_q.weight": "transformer.language_model.layers.3.self_attn.norm_q.weight",
        "layers.3.self_attn.add_q_proj.weight": "transformer.gen_layers.3.cross_attention.to_q.weight",
        "layers.3.self_attn.to_add_out.weight": "transformer.gen_layers.3.cross_attention.to_out.weight",
        "layers.3.self_attn.norm_added_q.weight": "transformer.gen_layers.3.cross_attention.norm_q.weight",
        "transformer.model.layers.3.self_attn.add_k_proj.weight": (
            "transformer.gen_layers.3.cross_attention.to_k.weight"
        ),
    }
    assert {key: Cosmos3OmniDiffusersPipeline._remap_ckpt_key(key) for key in remaps} == remaps


def test_prepare_latents_for_video_image_sound_and_action(make_cosmos3_pipeline) -> None:
    pipeline = make_cosmos3_pipeline()
    latents = pipeline._prepare_latents(16, 24, 5, torch.Generator(device="cpu").manual_seed(0))
    assert latents.shape == (1, 2, 2, 2, 3)

    pipeline._encode_conditioning_video = lambda *args, **kwargs: torch.full((1, 2, 2, 2, 3), 5.0)
    i2v_latents, velocity_mask, image_latent = pipeline._prepare_latents_i2v(
        torch.zeros(1, 3, 16, 24), 16, 24, 5, torch.Generator(device="cpu").manual_seed(0)
    )
    torch.testing.assert_close(i2v_latents[:, :, 0], torch.full((1, 2, 2, 3), 5.0))
    assert velocity_mask.tolist() == [[[[[0.0]], [[1.0]]]]]
    assert image_latent.shape == (1, 2, 1, 2, 3)

    pipeline._encode_video_tensor = lambda *args, **kwargs: torch.full((1, 2, 3, 2, 3), 6.0)
    v2v_latents, v2v_velocity_mask, v2v_condition = pipeline._prepare_latents_v2v(
        torch.zeros(1, 3, 5, 16, 24),
        16,
        24,
        9,
        torch.Generator(device="cpu").manual_seed(0),
        [0, 1],
    )
    torch.testing.assert_close(v2v_latents[:, :, 0:2], torch.full((1, 2, 2, 2, 3), 6.0))
    assert v2v_velocity_mask.tolist() == [[[[[0.0]], [[0.0]], [[1.0]]]]]
    assert v2v_condition.shape == (1, 2, 3, 2, 3)

    pipeline.transformer = pipeline.transformer.__class__(latent_channel_size=2, sound_gen=True, sound_dim=3)
    pipeline._sound_tokenizer = SimpleNamespace(
        sample_rate=10,
        latent_ch=3,
        hop_size=4,
        decode=lambda x: torch.ones(x.shape[0], 2, 24),
    )
    assert pipeline._resolve_sound_target_samples(SimpleNamespace(extra_args={"sound_duration": 2.0}), 9, 3.0) == (
        20,
        2.0,
        10,
    )
    sound_latents, latent_frames = pipeline._prepare_sound_latents(21, torch.Generator(device="cpu").manual_seed(0))
    assert (sound_latents.shape, latent_frames) == (torch.Size([1, 3, 6]), 6)
    assert pipeline._decode_sound_latents(torch.zeros(1, 3, 6), target_audio_samples=21).shape == (1, 2, 21)

    pipeline.transformer = pipeline.transformer.__class__(action_gen=True, action_dim=4)
    action, action_mask, clean, raw_dim = pipeline._prepare_action_latents(
        mode="forward_dynamics",
        action_chunk_size=2,
        raw_action_dim=None,
        generator=torch.Generator(device="cpu").manual_seed(0),
        sp=SimpleNamespace(extra_args={"action": [[1.0, 2.0], [3.0, 4.0]]}),
    )
    assert raw_dim == 2
    assert action_mask.tolist() == [[[0.0], [0.0]]]
    torch.testing.assert_close(action, clean)


def test_diffuse_covers_cfg_i2v_and_multimodal_steps(make_cosmos3_pipeline) -> None:
    pipeline = make_cosmos3_pipeline()
    latents = torch.zeros(1, 2, 1, 1, 1)

    result = pipeline.diffuse(
        latents=latents,
        timesteps=torch.tensor([900, 100]),
        cond_ids=_ids(2),
        cond_mask=_mask(),
        uncond_ids=_ids(1),
        uncond_mask=_mask(),
        guidance_scale=3.0,
        shared_kwargs={"video_shape": (1, 1, 1), "fps": 24.0},
        guidance_interval=(500.0, 1000.0),
    )
    assert [call["token"] for call in pipeline.transformer.calls] == [2, 1, 2]
    torch.testing.assert_close(result, torch.full_like(latents, 6.0))

    i2v = pipeline.diffuse(
        latents=torch.zeros(1, 2, 2, 1, 1),
        timesteps=torch.tensor([7]),
        cond_ids=_ids(2),
        cond_mask=_mask(),
        uncond_ids=_ids(1),
        uncond_mask=_mask(),
        guidance_scale=1.0,
        shared_kwargs={"video_shape": (2, 1, 1), "fps": 24.0},
        velocity_mask=torch.tensor([[[[[0.0]], [[1.0]]]]]),
        image_latent=torch.full((1, 2, 1, 1, 1), 7.0),
    )
    torch.testing.assert_close(i2v[:, :, 0:1], torch.full((1, 2, 1, 1, 1), 7.0))

    pipeline.transformer = pipeline.transformer.__class__(latent_channel_size=2, action_gen=True, action_dim=4)
    video_result, action_result = pipeline.diffuse(
        latents=latents,
        action_latents=torch.zeros(1, 3, 4),
        action_velocity_mask=torch.ones(1, 3, 1),
        action_condition_latents=torch.zeros(1, 3, 4),
        timesteps=torch.tensor([7, 3]),
        cond_ids=_ids(2),
        cond_mask=_mask(),
        uncond_ids=_ids(1),
        uncond_mask=_mask(),
        guidance_scale=1.0,
        shared_kwargs={"video_shape": (1, 1, 1), "fps": 24.0, "action_domain_ids": torch.tensor([0])},
    )
    torch.testing.assert_close(video_result, torch.full_like(latents, 4.0))
    torch.testing.assert_close(action_result, torch.full((), 44.0).expand_as(action_result))


def test_diffuse_keeps_paired_cfg_when_cache_dit_active(make_cosmos3_pipeline) -> None:
    """With cache-dit active the uncond pass is kept even outside the guidance
    interval (so cache-dit's has_separate_cfg parity stays in phase), and the
    output is numerically identical to the skip path.

    Contrast with ``test_diffuse_covers_cfg_and_i2v_steps`` (no marker), where
    the same inputs skip the out-of-interval uncond pass: calls == [2, 1, 2].
    """
    pipeline = make_cosmos3_pipeline()
    # Marker normally set by ``enable_cache_for_cosmos3`` when cache-dit is on.
    pipeline._cache_dit_requires_paired_cfg = True
    latents = torch.zeros(1, 2, 1, 1, 1)

    result = pipeline.diffuse(
        latents=latents,
        timesteps=torch.tensor([900, 100]),
        cond_ids=_ids(2),
        cond_mask=_mask(),
        uncond_ids=_ids(1),
        uncond_mask=_mask(),
        guidance_scale=3.0,
        shared_kwargs={"video_shape": (1, 1, 1), "fps": 24.0},
        guidance_interval=(500.0, 1000.0),
    )

    # t=900 is inside the interval (cond+uncond); t=100 is outside but the
    # uncond pass is still issued -> paired cond/uncond at every step.
    assert [call["token"] for call in pipeline.transformer.calls] == [2, 1, 2, 1]
    # Identical result to the skip path: out-of-interval combine uses scale=1.0,
    # so combine_cfg_noise(cond=2, uncond=1, 1.0) == 2 == the skipped cond value.
    torch.testing.assert_close(result, torch.full_like(latents, 6.0))


class TestForwardRouting:
    def _install_forward_stubs(self, pipeline):
        captured: dict[str, object] = {"diffuse_calls": [], "prepare_calls": []}

        def fake_format(prompt, negative_prompt, num_frames, frame_rate, height, width, *args, **kwargs):
            captured["format"] = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_frames": num_frames,
                "frame_rate": frame_rate,
                "height": height,
                "width": width,
                "is_t2i": kwargs["is_t2i"],
            }
            return _ids(2), _mask(), _ids(1), _mask()

        def fake_prepare(height, width, num_frames, generator):
            captured["prepare_calls"].append((height, width, num_frames, generator.initial_seed()))
            return torch.zeros(1, 2, 1, 1, 1)

        def fake_diffuse(**kwargs):
            captured["diffuse_calls"].append(kwargs)
            outputs = [kwargs["latents"] + len(captured["diffuse_calls"])]
            if kwargs.get("action_latents") is not None:
                outputs.append(kwargs["action_latents"] + 3.0)
            if kwargs.get("sound_latents") is not None:
                outputs.append(kwargs["sound_latents"] + 2.0)
            return outputs[0] if len(outputs) == 1 else tuple(outputs)

        pipeline._format_and_tokenize_prompts = fake_format
        pipeline._prepare_latents = fake_prepare
        pipeline._set_flow_shift = lambda target: captured.setdefault("flow_shifts", []).append(target)
        pipeline.diffuse = fake_diffuse
        pipeline._decode_latents = lambda latents: latents
        return captured

    @pytest.mark.parametrize(
        ("prompt", "sampling_params", "expected"),
        [
            (
                {"prompt": "A painted robot", "modalities": ["image"]},
                make_sampling_params(num_outputs_per_prompt=2),
                {"key": "image", "is_t2i": True, "flow": [3.0], "steps": [50, 50], "frames": 1},
            ),
            (
                "A warehouse robot",
                make_sampling_params(),
                {"key": "video", "is_t2i": False, "flow": [1.0], "steps": [35], "frames": 189},
            ),
        ],
    )
    def test_forward_defaults_and_mode_selection(
        self,
        make_cosmos3_pipeline,
        prompt,
        sampling_params,
        expected,
    ) -> None:
        pipeline = make_cosmos3_pipeline()
        captured = self._install_forward_stubs(pipeline)

        output = pipeline.forward(SimpleNamespace(prompts=[prompt], sampling_params=sampling_params))

        assert expected["key"] in output.output
        assert captured["format"]["is_t2i"] is expected["is_t2i"]
        assert captured["format"]["num_frames"] == expected["frames"]
        assert captured["flow_shifts"] == expected["flow"]
        assert [call[0] for call in pipeline.scheduler.set_timesteps_calls] == expected["steps"]

    def test_forward_i2v_sound_and_action_routes(self, make_cosmos3_pipeline) -> None:
        pipeline = make_cosmos3_pipeline()
        captured = self._install_forward_stubs(pipeline)
        image_tensor = torch.zeros(1, 3, 16, 16)
        velocity_mask = torch.ones(1, 1, 1, 1, 1)

        pipeline._prepare_latents_i2v = lambda *args, **kwargs: (
            torch.zeros(1, 2, 1, 1, 1),
            velocity_mask,
            torch.zeros(1, 2, 1, 1, 1),
        )
        pipeline.forward(
            SimpleNamespace(
                prompts=[
                    {
                        "prompt": "move",
                        "modalities": ["video"],
                        "additional_information": {"preprocessed_image": image_tensor},
                    }
                ],
                sampling_params=make_sampling_params(height=16, width=16, num_frames=5),
            )
        )
        assert captured["diffuse_calls"][-1]["shared_kwargs"]["noisy_frame_mask"] is velocity_mask

        video_tensor = torch.zeros(1, 3, 5, 16, 16)
        v2v_condition = torch.full((1, 2, 2, 1, 1), 4.0)
        v2v_mask = torch.tensor([[[[[0.0]], [[1.0]]]]])
        pipeline._prepare_latents_v2v = lambda *args, **kwargs: (
            torch.zeros(1, 2, 2, 1, 1),
            v2v_mask,
            v2v_condition,
        )
        pipeline.forward(
            SimpleNamespace(
                prompts=[
                    {
                        "prompt": "continue",
                        "modalities": ["video"],
                        "additional_information": {
                            "preprocessed_video": video_tensor,
                            "condition_frame_indexes_vision": [0],
                        },
                    }
                ],
                sampling_params=make_sampling_params(height=16, width=16, num_frames=5),
            )
        )
        assert captured["flow_shifts"][-1] == 10.0
        assert captured["format"]["negative_prompt"] == ""
        assert captured["diffuse_calls"][-1]["shared_kwargs"]["noisy_frame_mask"] is v2v_mask
        assert captured["diffuse_calls"][-1]["condition_latents"] is v2v_condition

        pipeline.transformer = pipeline.transformer.__class__(latent_channel_size=2, sound_gen=True, sound_dim=3)
        sound_latents = torch.zeros(1, 3, 4)
        pipeline._resolve_sound_target_samples = lambda *args: (20, 2.0, 10)
        pipeline._prepare_sound_latents = lambda *args: (sound_latents, 4)
        pipeline._decode_sound_latents = lambda *args: torch.ones(1, 2, 20)
        output = pipeline.forward(
            SimpleNamespace(
                prompts=[{"prompt": "A robot", "modalities": ["video"], "generate_sound": True}],
                sampling_params=make_sampling_params(num_frames=9, frame_rate=3.0),
            )
        )
        assert captured["diffuse_calls"][-1]["sound_latents"] is sound_latents
        assert output.output["audio_sample_rate"] == 10

        pipeline.transformer = pipeline.transformer.__class__(latent_channel_size=2, action_gen=True, action_dim=4)
        output = pipeline.forward(
            SimpleNamespace(
                prompts=[
                    {
                        "prompt": "Pick the block.",
                        "modalities": ["video"],
                        "additional_information": {"preprocessed_image": image_tensor},
                    }
                ],
                sampling_params=make_sampling_params(
                    height=16,
                    width=16,
                    extra_args={
                        "action_mode": "policy",
                        "action_chunk_size": 2,
                        "raw_action_dim": 2,
                        "domain_name": "bridge_orig_lerobot",
                    },
                ),
            )
        )
        assert captured["diffuse_calls"][-1]["shared_kwargs"]["action_domain_ids"].tolist() == [7]
        assert output.custom_output["action"].shape == (1, 2, 2)
        assert "action_only_output" not in output.custom_output

    def test_forward_dispatches_robolab_policy_flow(
        self,
        make_cosmos3_pipeline,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from vllm_omni.diffusion.models.cosmos3 import pipeline_cosmos3

        pipeline = make_cosmos3_pipeline()
        pipeline.transformer = pipeline.transformer.__class__(latent_channel_size=2, action_gen=True, action_dim=4)
        captured = self._install_forward_stubs(pipeline)
        video_latents = torch.zeros(1, 2, 1, 1, 1)
        velocity_mask = torch.ones(1, 1, 1, 1, 1)
        condition_latents = torch.zeros_like(video_latents)

        inputs = pipeline_cosmos3.RoboLabPolicyInputs(
            prompt="Pick the cube.",
            video_tensor=torch.zeros(1, 3, 3, 16, 16),
            action_tensor=torch.zeros(2, 2),
            action_condition_indexes=[0],
            action_start_frame_offset=1,
            raw_action_dim=2,
            domain_id=7,
            fps=15.0,
            height=16,
            width=16,
            image_size=None,
            num_frames=3,
            num_inference_steps=4,
            guidance_scale=3.0,
            flow_shift=5.0,
            seed=11,
            history_length=1,
            action_space="joint_pos",
            observation={},
        )

        def fake_prepare_action_latents(**kwargs):
            captured["prepare_action"] = kwargs
            action_chunk_size = kwargs["action_chunk_size"]
            raw_action_dim = int(kwargs["raw_action_dim"])
            return (
                torch.zeros(1, action_chunk_size, 4),
                torch.ones(1, action_chunk_size, 1),
                torch.zeros(1, action_chunk_size, 4),
                raw_action_dim,
            )

        def fake_prepare_action_video(*args, **kwargs):
            captured["prepare_action_video"] = {"args": args, "kwargs": kwargs}
            return video_latents, velocity_mask, condition_latents

        monkeypatch.setattr(
            pipeline_cosmos3,
            "build_robolab_unipc_scheduler",
            lambda num_steps, shift, device: StubScheduler(list(range(num_steps, 0, -1)), flow_shift=shift),
        )
        pipeline._build_robolab_policy_inputs = lambda sp, prompt_data, request_id=None: inputs
        pipeline._prepare_action_latents = fake_prepare_action_latents
        pipeline._prepare_latents_action_video = fake_prepare_action_video
        pipeline._decode_latents = lambda latents: (_ for _ in ()).throw(
            AssertionError("RoboLab should not decode video")
        )

        output = pipeline.forward(SimpleNamespace(prompts=["ignored"], sampling_params=make_sampling_params()))

        assert captured["format"] == {
            "prompt": "Pick the cube.",
            "negative_prompt": "",
            "num_frames": 3,
            "frame_rate": 15.0,
            "height": 16,
            "width": 16,
            "is_t2i": False,
        }
        assert "flow_shifts" not in captured
        assert pipeline.scheduler.set_timesteps_calls == []
        assert captured["prepare_action"]["clean_action"] is inputs.action_tensor
        assert captured["prepare_action"]["condition_indexes"] == [0]
        assert captured["prepare_action_video"]["kwargs"] == {"image_size": None}
        assert captured["diffuse_calls"][-1]["shared_kwargs"]["action_domain_ids"].tolist() == [7]
        assert captured["diffuse_calls"][-1]["timesteps"].tolist() == [4, 3, 2, 1]
        assert output.output == {}
        assert output.custom_output["action_only_output"] is True
        assert output.custom_output["action"].shape == (1, 2, 2)
        assert "actions" not in output.custom_output
        assert "robolab_action_postprocess" in output.custom_output
        assert "robolab_policy_inputs" not in output.custom_output

    @pytest.mark.parametrize(
        ("prompt", "sampling_params", "message"),
        [
            (["one", "two"], make_sampling_params(), "single prompt"),
            ([{"prompt": "one", "modalities": ["image", "video"]}], make_sampling_params(), "both image and video"),
            (
                [{"prompt": "x", "modalities": ["image"], "generate_sound": True}],
                make_sampling_params(),
                "only for video",
            ),
        ],
    )
    def test_forward_rejects_invalid_public_requests(
        self,
        make_cosmos3_pipeline,
        prompt,
        sampling_params,
        message,
    ) -> None:
        pipeline = make_cosmos3_pipeline()
        pipeline.transformer = pipeline.transformer.__class__(latent_channel_size=2, sound_gen=True, sound_dim=3)

        with pytest.raises(ValueError, match=message):
            pipeline.forward(SimpleNamespace(prompts=prompt, sampling_params=sampling_params))
