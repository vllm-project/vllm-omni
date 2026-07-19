# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_pipeline_declares_layerwise_offload_components() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline

    assert Cosmos3OmniDiffusersPipeline._dit_modules == ["transformer.language_model", "transformer"]
    assert Cosmos3OmniDiffusersPipeline._encoder_modules == []
    assert Cosmos3OmniDiffusersPipeline._vae_modules == ["vae"]
    assert Cosmos3OmniDiffusersPipeline._resident_modules == []
    assert hasattr(Cosmos3OmniDiffusersPipeline, "enable_omni_model_cpu_offload")


class StubScheduler:
    def __init__(
        self,
        timesteps: list[int] | None = None,
        *,
        flow_shift: float = 1.0,
    ) -> None:
        self.timesteps = torch.tensor(timesteps or [9, 3], dtype=torch.int64)
        self.config = SimpleNamespace(
            num_train_timesteps=1000,
            flow_shift=flow_shift,
        )
        self.set_timesteps_calls: list[dict[str, Any]] = []
        self.step_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        *,
        shift: float | None = None,
        sigmas: list[float] | None = None,
    ) -> None:
        self.set_timesteps_calls.append(
            {
                "num_inference_steps": num_inference_steps,
                "device": device,
                "shift": shift,
                "sigmas": sigmas,
            }
        )
        if sigmas is not None:
            self.timesteps = torch.tensor(sigmas, device=device)
        else:
            assert num_inference_steps is not None
            self.timesteps = torch.arange(num_inference_steps, 0, -1, dtype=torch.int64, device=device)

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
        self.encode_input_shapes: list[tuple[int, ...]] = []

    def encode(self, video: torch.Tensor):
        self.encode_input_shapes.append(tuple(video.shape))
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
        control_latents = kwargs.get("control_latents")
        control_bonus = 100 if control_latents is not None else 0
        self.calls.append(
            {
                "token": token,
                "has_control": control_latents is not None,
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
        outputs: list[torch.Tensor] = [torch.full_like(hidden_states, float(token + control_bonus))]
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
            COSMOS3_VIDEO_DEFAULT_FLOW_SHIFT,
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
        pipeline._engine_init_flow_shift = COSMOS3_VIDEO_DEFAULT_FLOW_SHIFT
        pipeline._current_flow_shift = COSMOS3_VIDEO_DEFAULT_FLOW_SHIFT
        pipeline.is_distilled_model = False
        pipeline.is_edge_model = False
        pipeline._guidance_scale = None
        pipeline._num_timesteps = None
        pipeline._cosmos3_branch_caches = None
        pipeline._cache_dit_requires_paired_cfg = False
        pipeline._sound_tokenizer = None
        pipeline.progress_bar = passthrough_progress_bar
        return pipeline

    return _make


@pytest.fixture
def sequential_cfg_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion.distributed import cfg_parallel

    monkeypatch.setattr(cfg_parallel, "get_classifier_free_guidance_world_size", lambda: 1)


def make_sampling_params(**overrides: Any) -> SimpleNamespace:
    values = {
        "height": None,
        "width": None,
        "num_frames": None,
        "num_inference_steps": None,
        "guidance_scale": None,
        "guidance_scale_provided": False,
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


def make_request_batch(prompt: Any, sampling_params: SimpleNamespace) -> DiffusionRequestBatch:
    if isinstance(prompt, list):
        return DiffusionRequestBatch(
            requests=[
                SimpleNamespace(
                    prompt=item,
                    request_id=f"cosmos3-test-{idx}",
                    sampling_params=sampling_params,
                    kv_sender_info=None,
                )
                for idx, item in enumerate(prompt)
            ]
        )
    return DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                prompt=prompt,
                request_id="cosmos3-test",
                sampling_params=sampling_params,
                kv_sender_info=None,
            )
        ]
    )


def _ids(value: int) -> torch.Tensor:
    return torch.tensor([[value]], dtype=torch.long)


def _mask() -> torch.Tensor:
    return torch.ones(1, 1, dtype=torch.long)


def _capture_tokenize_calls(pipeline: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def _tokenize(
        text: str,
        max_sequence_length: int,
        use_system_prompt: bool = False,
        system_prompt: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        index = len(calls) + 1
        calls.append(
            {
                "text": text,
                "max_sequence_length": max_sequence_length,
                "use_system_prompt": use_system_prompt,
                "system_prompt": system_prompt,
            }
        )
        return _ids(index), torch.full((1, 1), index, dtype=torch.long)

    pipeline._tokenize_prompt = _tokenize
    return calls


@pytest.mark.parametrize(
    ("provided", "value", "default", "is_distilled", "expected"),
    [
        (False, 1.0, 7.0, False, 7.0),
        (True, 1.0, 7.0, False, 1.0),
        (True, 4.5, 7.0, False, 4.5),
        (False, 1.0, 7.0, True, 1.0),
        (True, 4.5, 7.0, True, 1.0),
    ],
)
def test_resolve_guidance_scale(
    make_cosmos3_pipeline,
    provided: bool,
    value: float,
    default: float,
    is_distilled: bool,
    expected: float,
) -> None:
    pipeline = make_cosmos3_pipeline()
    pipeline.is_distilled_model = is_distilled
    sp = make_sampling_params(
        guidance_scale=value,
        guidance_scale_provided=provided,
    )

    assert pipeline._resolve_guidance_scale(sp, default) == expected


def test_distilled_generation_accepts_t2i_and_i2v(make_cosmos3_pipeline) -> None:
    pipeline = make_cosmos3_pipeline()
    pipeline.is_distilled_model = True
    common = {
        "action_enabled": False,
        "transfer_config": None,
        "is_v2v": False,
        "sound_enabled": False,
    }

    pipeline._validate_distilled_generation_mode(
        is_t2i=True,
        image_tensor=None,
        **common,
    )
    pipeline._validate_distilled_generation_mode(
        is_t2i=False,
        image_tensor=torch.zeros(1),
        **common,
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"action_enabled": True}, "action requests are unsupported"),
        ({"transfer_config": SimpleNamespace()}, "transfer requests are unsupported"),
        ({"is_v2v": True}, "video-to-video requests are unsupported"),
        ({"sound_enabled": True}, "sound generation is unsupported"),
        (
            {"is_t2i": False, "image_tensor": None},
            "text-to-video requests are unsupported",
        ),
        (
            {"is_t2i": True, "image_tensor": torch.zeros(1)},
            "image-conditioned image generation is unsupported",
        ),
    ],
)
def test_distilled_generation_rejects_unsupported_modes(
    make_cosmos3_pipeline,
    overrides: dict[str, Any],
    message: str,
) -> None:
    pipeline = make_cosmos3_pipeline()
    pipeline.is_distilled_model = True
    kwargs = {
        "is_t2i": True,
        "image_tensor": None,
        "action_enabled": False,
        "transfer_config": None,
        "is_v2v": False,
        "sound_enabled": False,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        pipeline._validate_distilled_generation_mode(**kwargs)


def test_distilled_generation_rejects_robolab_policy(make_cosmos3_pipeline) -> None:
    pipeline = make_cosmos3_pipeline()
    pipeline.is_distilled_model = True

    with pytest.raises(ValueError, match="RoboLab/action policy requests are unsupported"):
        pipeline._forward_robolab_policy(make_sampling_params(), None, 0.0)


@pytest.mark.parametrize(
    ("prompt", "sampling_params", "message"),
    [
        (
            {"prompt": "x", "modalities": ["video"], "generate_sound": True},
            make_sampling_params(),
            "do not support sound generation",
        ),
        (
            {"prompt": "x", "modalities": ["video"]},
            make_sampling_params(extra_args={"edge": {"control_path": "/tmp/control.mp4"}}),
            "do not support transfer inference",
        ),
        (
            {
                "prompt": "x",
                "modalities": ["video"],
                "additional_information": {
                    "preprocessed_video": torch.zeros(1, 3, 5, 16, 16),
                },
            },
            make_sampling_params(height=16, width=16, num_frames=5),
            "do not support video-to-video generation",
        ),
    ],
)
def test_edge_forward_rejects_unsupported_modes(
    make_cosmos3_pipeline,
    prompt: dict[str, Any],
    sampling_params: SimpleNamespace,
    message: str,
) -> None:
    pipeline = make_cosmos3_pipeline()
    pipeline.is_edge_model = True

    with pytest.raises(ValueError, match=message):
        pipeline.forward(SimpleNamespace(prompts=[prompt], sampling_params=sampling_params))


def test_pipeline_registered_and_exported() -> None:
    from vllm_omni.diffusion.cache.cache_dit_backend import CUSTOM_DIT_ENABLERS
    from vllm_omni.diffusion.models import cosmos3
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline
    from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
    from vllm_omni.diffusion.registry import (
        _DIFFUSION_IR_OP_PRIORITY_FUNCS,
        _DIFFUSION_MODELS,
        _DIFFUSION_POST_PROCESS_FUNCS,
        _DIFFUSION_PRE_PROCESS_FUNCS,
    )

    assert issubclass(Cosmos3OmniDiffusersPipeline, nn.Module)
    assert issubclass(Cosmos3OmniDiffusersPipeline, ProgressBarMixin)
    assert Cosmos3OmniDiffusersPipeline.support_image_input is True
    assert "Cosmos3OmniDiffusersPipeline" in cosmos3.__all__

    for pipeline_name in ("Cosmos3OmniDiffusersPipeline", "Cosmos3OmniPipeline"):
        assert _DIFFUSION_MODELS[pipeline_name] == (
            "cosmos3",
            "pipeline_cosmos3",
            "Cosmos3OmniDiffusersPipeline",
        )
        assert _DIFFUSION_PRE_PROCESS_FUNCS[pipeline_name] == "get_cosmos3_pre_process_func"
        assert _DIFFUSION_POST_PROCESS_FUNCS[pipeline_name] == "get_cosmos3_post_process_func"
        assert _DIFFUSION_IR_OP_PRIORITY_FUNCS[pipeline_name] == "get_cosmos3_ir_op_priority_func"
        assert pipeline_name in CUSTOM_DIT_ENABLERS


@pytest.mark.parametrize(
    "pipeline_name",
    ["Cosmos3OmniDiffusersPipeline", "Cosmos3OmniPipeline"],
)
def test_cosmos3_model_index_resolves_pipeline(
    tmp_path,
    pipeline_name: str,
) -> None:
    import json

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
        Cosmos3OmniDiffusersPipeline,
    )
    from vllm_omni.diffusion.registry import DiffusionModelRegistry

    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": pipeline_name}))

    config = OmniDiffusionConfig(model=str(tmp_path))
    config.enrich_config()

    assert config.model_class_name == pipeline_name
    resolved_pipeline_cls = DiffusionModelRegistry._try_load_model_cls(config.model_class_name)
    assert resolved_pipeline_cls is Cosmos3OmniDiffusersPipeline


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
        load_config_calls: list[dict[str, Any]] = []
        from_config_calls: list[dict[str, Any]] = []

        def __init__(self, *, flow_shift: float = 1.0) -> None:
            self.config = SimpleNamespace(flow_shift=flow_shift)

        @classmethod
        def load_config(cls, *args, **kwargs):
            cls.load_config_calls.append({"args": args, "kwargs": dict(kwargs)})
            return {
                "_class_name": "UniPCMultistepScheduler",
                "flow_shift": 1.0,
            }

        @classmethod
        def from_config(cls, config, **kwargs):
            cls.from_config_calls.append({"config": config, "kwargs": dict(kwargs)})
            return cls(flow_shift=kwargs.get("shift", config.get("flow_shift", 1.0)))

    class _StubVideoProcessor:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(pipeline_cosmos3, "AutoTokenizer", _StubAutoTokenizer)
    monkeypatch.setattr(pipeline_cosmos3, "DistributedAutoencoderKLWan", _StubDiffusersVAE)
    monkeypatch.setattr(pipeline_cosmos3, "FlowUniPCMultistepScheduler", _StubDiffusersScheduler)
    monkeypatch.setattr(pipeline_cosmos3, "VideoProcessor", _StubVideoProcessor)
    monkeypatch.setattr(pipeline_cosmos3, "get_local_device", lambda: torch.device("cpu"))
    return _StubDiffusersScheduler


def _make_od_config(
    *,
    sound_gen: bool,
    tf_model_config_overrides: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
) -> SimpleNamespace:
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
    if tf_model_config_overrides:
        tf_model_config.update(tf_model_config_overrides)
    return SimpleNamespace(
        enable_cpu_offload=False,
        enable_diffusion_pipeline_profiler=False,
        model="/nonexistent/model/path",
        dtype=torch.float32,
        flow_shift=None,
        quantization_config=None,
        custom_pipeline_args={},
        model_config=model_config or {},
        tf_model_config=tf_model_config,
    )


def test_pipeline_init_skips_tokenizer_when_sound_disabled(stub_real_pipeline_init) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline

    pipeline = Cosmos3OmniDiffusersPipeline(od_config=_make_od_config(sound_gen=False))

    assert pipeline._sound_tokenizer is None
    assert pipeline.transformer.sound_gen is False
    assert not hasattr(pipeline.transformer, "audio_proj_in")
    assert not hasattr(pipeline.transformer, "audio_proj_out")


def test_pipeline_init_uses_flow_unipc_with_cosmos3_defaults(stub_real_pipeline_init) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline

    od_config = _make_od_config(sound_gen=False)
    od_config.flow_shift = 2.5

    pipeline = Cosmos3OmniDiffusersPipeline(od_config=od_config)

    assert len(stub_real_pipeline_init.load_config_calls) == 1
    assert len(stub_real_pipeline_init.from_config_calls) == 1
    call = stub_real_pipeline_init.from_config_calls[0]
    assert call["config"]["_class_name"] == "UniPCMultistepScheduler"
    assert call["kwargs"] == {
        "shift": 1.0,
        "use_dynamic_shifting": False,
        "prediction_type": "flow_prediction",
    }
    assert pipeline.is_distilled_model is False
    assert pipeline._engine_init_flow_shift == 2.5


@pytest.mark.parametrize(
    ("scheduler_class_name", "expected_distilled"),
    [
        ("FlowMatchEulerDiscreteScheduler", True),
        ("UniPCMultistepScheduler", False),
    ],
)
def test_pipeline_resolves_scheduler_class_from_checkpoint_file(
    tmp_path,
    stub_real_pipeline_init,
    monkeypatch: pytest.MonkeyPatch,
    scheduler_class_name: str,
    expected_distilled: bool,
) -> None:
    import json

    from vllm_omni.diffusion.models.cosmos3 import pipeline_cosmos3
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline
    from vllm_omni.diffusion.models.schedulers.scheduling_flow_unipc_multistep import (
        FlowUniPCMultistepScheduler as RealFlowUniPCMultistepScheduler,
    )

    t_list = [1.0, 0.75, 0.5, 0.25]
    scheduler_dir = tmp_path / "scheduler"
    scheduler_dir.mkdir()
    scheduler_config = {"_class_name": scheduler_class_name}
    if expected_distilled:
        scheduler_config["fixed_step_sampler_config"] = {"sample_type": "sde", "t_list": t_list}
    (scheduler_dir / "scheduler_config.json").write_text(json.dumps(scheduler_config))

    class StubFlowUniPCScheduler(StubScheduler):
        from_config_calls: list[tuple[Any, dict[str, Any]]] = []

        @classmethod
        def load_config(cls, *args, **kwargs):
            return RealFlowUniPCMultistepScheduler.load_config(*args, **kwargs)

        @classmethod
        def from_config(cls, config, **kwargs):
            cls.from_config_calls.append((config, dict(kwargs)))
            return cls()

    class StubFlowMatchScheduler(StubScheduler):
        from_config_calls: list[tuple[Any, dict[str, Any]]] = []

        @classmethod
        def from_config(cls, config, **kwargs):
            cls.from_config_calls.append((config, dict(kwargs)))
            scheduler = cls()
            scheduler.config.fixed_step_sampler_config = config["fixed_step_sampler_config"]
            return scheduler

    monkeypatch.setattr(pipeline_cosmos3, "FlowUniPCMultistepScheduler", StubFlowUniPCScheduler)
    monkeypatch.setattr(
        pipeline_cosmos3,
        "FlowMatchEulerDiscreteScheduler",
        StubFlowMatchScheduler,
    )

    od_config = _make_od_config(sound_gen=False)
    od_config.model = str(tmp_path)
    pipeline = Cosmos3OmniDiffusersPipeline(od_config=od_config)

    assert pipeline.is_distilled_model is expected_distilled
    if expected_distilled:
        assert len(StubFlowMatchScheduler.from_config_calls) == 1
        assert StubFlowMatchScheduler.from_config_calls[0][1] == {"stochastic_sampling": True}
        assert StubFlowUniPCScheduler.from_config_calls == []
        assert pipeline._scheduler_init_t_list == t_list
    else:
        assert len(StubFlowUniPCScheduler.from_config_calls) == 1
        assert StubFlowMatchScheduler.from_config_calls == []
        assert not hasattr(pipeline, "_scheduler_init_t_list")


def test_distilled_pipeline_initializes_sde_scheduler(
    stub_real_pipeline_init,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
        Cosmos3OmniDiffusersPipeline,
        FlowMatchEulerDiscreteScheduler,
    )

    t_list = [1.0, 0.9375, 0.8333333333333334, 0.625]
    scheduler_config = {
        "_class_name": "FlowMatchEulerDiscreteScheduler",
        "shift": 1.0,
        "stochastic_sampling": False,
        "fixed_step_sampler_config": {
            "sample_type": "sde",
            "t_list": t_list,
        },
    }
    monkeypatch.setattr(
        stub_real_pipeline_init,
        "load_config",
        classmethod(lambda cls, *args, **kwargs: scheduler_config),
    )

    pipeline = Cosmos3OmniDiffusersPipeline(od_config=_make_od_config(sound_gen=False))
    assert pipeline.is_distilled_model is True
    assert isinstance(pipeline.scheduler, FlowMatchEulerDiscreteScheduler)
    assert pipeline.scheduler.config.stochastic_sampling is True
    assert pipeline._scheduler_init_t_list == t_list


@pytest.mark.parametrize(
    ("fixed_step_config", "error_pattern"),
    [
        pytest.param(
            {"t_list": [1.0, 0.5]},
            r"fixed_step_sampler_config\.sample_type=sde",
            id="missing-sample-type",
        ),
        pytest.param(
            {"sample_type": "ode", "t_list": [1.0, 0.5]},
            r"fixed_step_sampler_config\.sample_type=sde",
            id="unsupported-sample-type",
        ),
        pytest.param(
            {"sample_type": "sde"},
            r"non-empty fixed_step_sampler_config\.t_list",
            id="missing-t-list",
        ),
        pytest.param(
            {"sample_type": "sde", "t_list": []},
            r"non-empty fixed_step_sampler_config\.t_list",
            id="empty-t-list",
        ),
    ],
)
def test_distilled_scheduler_validates_fixed_step_config(
    stub_real_pipeline_init,
    monkeypatch: pytest.MonkeyPatch,
    fixed_step_config: dict[str, Any],
    error_pattern: str,
) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
        Cosmos3OmniDiffusersPipeline,
    )

    scheduler_config = {
        "_class_name": "FlowMatchEulerDiscreteScheduler",
        "fixed_step_sampler_config": fixed_step_config,
    }
    monkeypatch.setattr(
        stub_real_pipeline_init,
        "load_config",
        classmethod(lambda cls, *args, **kwargs: scheduler_config),
    )

    with pytest.raises(ValueError, match=error_pattern):
        Cosmos3OmniDiffusersPipeline(od_config=_make_od_config(sound_gen=False))


def test_pipeline_init_selects_edge_transformer_from_backbone_type(stub_real_pipeline_init) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
        COSMOS3_EDGE_VIDEO_DEFAULT_FLOW_SHIFT,
        Cosmos3OmniDiffusersPipeline,
    )
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3_edge import (
        COSMOS3_EDGE_BACKBONE_TYPE,
        Cosmos3EdgeVFMTransformer,
    )

    pipeline = Cosmos3OmniDiffusersPipeline(
        od_config=_make_od_config(
            sound_gen=False,
            tf_model_config_overrides={
                "backbone_type": COSMOS3_EDGE_BACKBONE_TYPE,
                "qk_norm_for_text": False,
                "latent_channel": 48,
                "latent_patch_size": 2,
                "temporal_compression_factor": 4,
                "layer_norm_epsilon": 1e-5,
            },
        )
    )

    assert isinstance(pipeline.transformer, Cosmos3EdgeVFMTransformer)
    assert pipeline.is_edge_model is True
    assert pipeline._engine_init_flow_shift == COSMOS3_EDGE_VIDEO_DEFAULT_FLOW_SHIFT


def test_pipeline_init_does_not_select_edge_from_model_index_class_name(stub_real_pipeline_init) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3_edge import Cosmos3EdgeVFMTransformer

    pipeline = Cosmos3OmniDiffusersPipeline(
        od_config=_make_od_config(
            sound_gen=False,
            model_config={"_class_name": "Cosmos3EdgeOmniDiffusersPipeline"},
        )
    )

    assert isinstance(pipeline.transformer, Cosmos3VFMTransformer)
    assert not isinstance(pipeline.transformer, Cosmos3EdgeVFMTransformer)


def test_flow_unipc_reuses_scheduler_and_forwards_each_request_shift(make_cosmos3_pipeline) -> None:
    pipeline = make_cosmos3_pipeline()
    scheduler = pipeline.scheduler

    assert pipeline._engine_init_flow_shift == 10.0
    assert pipeline._current_flow_shift == 10.0
    assert scheduler.config.flow_shift == 1.0

    for shift in (3.0, 10.0):
        pipeline._set_flow_shift(shift)
        pipeline._set_timesteps(4, torch.device("cpu"), shift=pipeline._current_flow_shift)

    assert pipeline.scheduler is scheduler
    assert pipeline._current_flow_shift == 10.0
    assert [call["shift"] for call in scheduler.set_timesteps_calls] == [3.0, 10.0]
    assert all(call["num_inference_steps"] == 4 for call in scheduler.set_timesteps_calls)
    assert all(call["sigmas"] is None for call in scheduler.set_timesteps_calls)


def test_flow_unipc_reproducible_with_same_seed(make_cosmos3_pipeline) -> None:
    from vllm_omni.diffusion.models.schedulers.scheduling_flow_unipc_multistep import (
        FlowUniPCMultistepScheduler,
    )

    pipeline = make_cosmos3_pipeline()
    pipeline.scheduler = FlowUniPCMultistepScheduler(
        shift=1.0,
        use_dynamic_shifting=False,
        prediction_type="flow_prediction",
    )
    pipeline._format_and_tokenize_prompts = lambda *args, **kwargs: (
        _ids(1),
        _mask(),
        _ids(0),
        _mask(),
    )

    def run(seed: int) -> torch.Tensor:
        request = make_request_batch(
            {"prompt": "A test video.", "modalities": ["video"]},
            make_sampling_params(
                seed=seed,
                guidance_scale=1.0,
                guidance_scale_provided=True,
                num_inference_steps=4,
                num_frames=5,
                height=16,
                width=16,
                extra_args={"flow_shift": 3.0},
            ),
        )
        output = pipeline.forward(request)
        return output.output["video"]

    first = run(123)
    second = run(123)
    different_seed = run(456)

    torch.testing.assert_close(first, second)
    assert not torch.equal(first, different_seed)


def test_distilled_set_timesteps_uses_fixed_sigma_schedule(make_cosmos3_pipeline) -> None:
    pipeline = make_cosmos3_pipeline()
    pipeline.is_distilled_model = True
    pipeline._scheduler_init_t_list = [1.0, 0.75, 0.5, 0.25]

    pipeline._set_timesteps(
        num_inference_steps=99,
        device=torch.device("cpu"),
        shift=7.0,
    )

    assert pipeline.scheduler.set_timesteps_calls == [
        {
            "num_inference_steps": None,
            "device": torch.device("cpu"),
            "shift": None,
            "sigmas": pipeline._scheduler_init_t_list,
        }
    ]


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
        prompt={"prompt": "A slow camera push.", "multi_modal_data": {"image": Image.new("RGB", (320, 160))}},
        sampling_params=make_sampling_params(height=None, width=None, extra_args={}),
    )

    result = preprocess(i2v)
    assert (result.sampling_params.height, result.sampling_params.width) == (672, 1344)
    assert tuple(result.prompt["additional_information"]["preprocessed_image"].shape[-2:]) == (672, 1344)

    frames = [Image.new("RGB", (8, 4), color) for color in ("red", "green", "blue")]
    action = SimpleNamespace(
        prompt={"prompt": "Move.", "multi_modal_data": {"video": frames}},
        sampling_params=make_sampling_params(height=16, width=32, extra_args={"action_mode": "forward_dynamics"}),
    )

    additional = preprocess(action).prompt["additional_information"]
    assert tuple(additional["preprocessed_image"].shape) == (1, 3, 16, 32)
    assert tuple(additional["preprocessed_video"].shape) == (1, 3, 3, 16, 32)

    frames = [Image.new("RGB", (8, 4), color) for color in ("red", "green", "blue", "yellow", "purple", "black")]
    v2v = SimpleNamespace(
        prompt={"prompt": "Continue.", "multi_modal_data": {"video": frames}},
        sampling_params=make_sampling_params(
            height=16,
            width=32,
            extra_args={"condition_frame_indexes_vision": [0, 1], "condition_video_keep": "last"},
        ),
    )
    additional = preprocess(v2v).prompt["additional_information"]
    assert tuple(additional["preprocessed_video"].shape) == (1, 3, 5, 16, 32)
    assert additional["condition_frame_indexes_vision"] == [0, 1]


def test_transfer_config_media_helpers_and_preprocess_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion.models.cosmos3 import transfer
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
        Cosmos3OmniDiffusersPipeline,
        get_cosmos3_pre_process_func,
    )

    cfg = transfer.resolve_transfer_config(make_sampling_params(extra_args={"edge": True}))
    assert cfg is not None
    assert list(cfg.hints) == ["edge"]
    assert cfg.guidance_scale == 3.0
    assert cfg.control_guidance == 1.5
    assert cfg.flow_shift == 10.0
    assert cfg.num_video_frames_per_chunk == 93
    assert cfg.share_vision_temporal_positions is True
    # fps omitted (no fps/frame_rate on the sampling params) -> wsm preset default (10) applies.
    defaulted_fps_cfg = transfer.resolve_transfer_config(make_sampling_params(extra_args={"wsm": True}))
    assert defaulted_fps_cfg is not None
    assert defaulted_fps_cfg.fps == 10
    # fps provided (frame_rate set) -> the user value wins over the preset default.
    explicit_fps_cfg = transfer.resolve_transfer_config(make_sampling_params(frame_rate=24.0, extra_args={"wsm": True}))
    assert explicit_fps_cfg is not None
    assert explicit_fps_cfg.fps == 24.0
    assert (
        Cosmos3OmniDiffusersPipeline.reference_video_decode_spec(extra_args={"edge": True, "max_frames": 4}).max_frames
        == 4
    )
    frames_for_pad = torch.arange(3 * 3, dtype=torch.uint8).reshape(1, 3, 1, 3)
    assert transfer.pad_temporal_frames(frames_for_pad, 5)[0, :, 0, 0].tolist() == [0, 3, 6, 6, 3]

    real_import_module = transfer.importlib.import_module

    def raise_missing_cv2(name: str, *args: Any, **kwargs: Any):
        if name == "cv2":
            raise ImportError("missing cv2")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(transfer.importlib, "import_module", raise_missing_cv2)
    with pytest.raises(ImportError, match="opencv-python"):
        transfer.load_or_compute_control_frames(
            cfg.hints["edge"],
            height=8,
            width=8,
            max_frames=1,
            input_frames=torch.zeros(3, 1, 8, 8, dtype=torch.uint8),
        )

    precomputed = torch.zeros(3, 2, 8, 8, dtype=torch.uint8)
    precomputed_cfg = transfer.resolve_transfer_config(
        make_sampling_params(extra_args={"edge": {"control": precomputed}})
    )
    assert precomputed_cfg is not None
    loaded = transfer.load_or_compute_control_frames(
        precomputed_cfg.hints["edge"],
        height=8,
        width=8,
        max_frames=2,
        input_frames=None,
    )
    assert tuple(loaded.shape) == (3, 2, 8, 8)

    preprocess = get_cosmos3_pre_process_func(SimpleNamespace())

    class FramesWithFps(list):
        fps = 12.5

    frames = FramesWithFps(Image.new("RGB", (8, 4), color) for color in ("red", "green", "blue", "yellow", "black"))
    prompt = {"prompt": "transfer", "multi_modal_data": {"video": frames}}
    request = SimpleNamespace(
        prompt=prompt,
        sampling_params=SimpleNamespace(
            height=16,
            width=32,
            extra_args={"edge": True, "max_frames": 4, "resolution": "256"},
        ),
    )
    additional = preprocess(request).prompt["additional_information"]
    assert (request.sampling_params.height, request.sampling_params.width) == (192, 320)
    assert tuple(additional["preprocessed_transfer_video"].shape) == (1, 3, 4, 192, 320)
    assert additional["transfer_input_fps"] == 12.5
    assert "preprocessed_video" not in additional


def test_transfer_fps_matches_resolved_frame_rate_precedence() -> None:
    """When fps and frame_rate differ, transfer must pick frame_rate -- the same
    precedence as OmniDiffusionSamplingParams.resolved_frame_rate. Uses the real
    sampling-params dataclass so the resolved_frame_rate property is exercised."""
    from vllm_omni.diffusion.models.cosmos3 import transfer
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    sp = OmniDiffusionSamplingParams(fps=24, frame_rate=12.0, extra_args={"edge": True})
    assert sp.resolved_frame_rate == 12.0
    cfg = transfer.resolve_transfer_config(sp)
    assert cfg is not None
    # edge has no preset fps default, so cfg.fps comes straight from fps resolution.
    assert cfg.fps == sp.resolved_frame_rate == 12.0


def test_transfer_edge_uses_rgb_canny(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion.models.cosmos3 import transfer

    class FakeCv2:
        def __init__(self) -> None:
            self.canny_inputs: list[np.ndarray] = []

        def Canny(self, image, lower, upper):
            assert (lower, upper) == (100, 200)
            self.canny_inputs.append(image.copy())
            return np.zeros(image.shape[:2], dtype=np.uint8)

    fake_cv2 = FakeCv2()
    monkeypatch.setattr(transfer, "_import_cv2", lambda _hint_key: fake_cv2)

    frames = torch.zeros(3, 1, 4, 5, dtype=torch.uint8)
    frames[0] = 255
    edge = transfer.make_edge_control(frames, "medium")

    assert tuple(edge.shape) == (3, 1, 4, 5)
    assert len(fake_cv2.canny_inputs) == 1
    assert fake_cv2.canny_inputs[0].shape == (4, 5, 3)


def test_transfer_blur_uses_scaled_bilateral(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion.models.cosmos3 import transfer

    class FakeCv2:
        INTER_AREA = 1
        INTER_LINEAR = 2
        INTER_CUBIC = 3

        def __init__(self) -> None:
            self.bilateral_calls: list[tuple[tuple[int, int, int], int, float, float]] = []

        def resize(self, image, size, interpolation):
            del interpolation
            width, height = size
            return np.zeros((height, width, image.shape[2]), dtype=image.dtype)

        def bilateralFilter(self, image, diameter, sigma_color, sigma_space):
            self.bilateral_calls.append((image.shape, diameter, sigma_color, sigma_space))
            return image

        def GaussianBlur(self, *args, **kwargs):
            raise AssertionError("Cosmos3 transfer blur should use bilateralFilter, not GaussianBlur.")

    fake_cv2 = FakeCv2()
    monkeypatch.setattr(transfer, "_import_cv2", lambda _hint_key: fake_cv2)

    frames = torch.zeros(3, 1, 72, 72, dtype=torch.uint8)
    blurred = transfer.make_blur_control(frames, "high")

    assert tuple(blurred.shape) == (3, 1, 72, 72)
    assert fake_cv2.bilateral_calls == [((72, 72, 3), 3, 15.0, 10.0)]


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
        get_cosmos3_post_process_func,
        make_robolab_action_postprocess_inputs,
    )

    func = get_cosmos3_post_process_func(SimpleNamespace())
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
    processed = func(
        {
            "payload": {
                "actions": action,
            },
            "metadata": {
                "actions": {
                    "raw_action_dim": 2,
                    "action_mode": "policy",
                    "domain_id": 7,
                },
                "common": {
                    "action_only_output": True,
                },
                "internal": {
                    "robolab_action_postprocess": make_robolab_action_postprocess_inputs(inputs),
                },
            },
        }
    )

    processed_action = processed["payload"]["actions"]
    assert processed_action.shape == (1, 2)
    assert processed_action.dtype == torch.zeros((), dtype=torch.float32).numpy().dtype
    torch.testing.assert_close(torch.from_numpy(processed_action), torch.tensor([[1.0, 0.25]]))
    assert processed["metadata"] == {
        "actions": {
            "raw_action_dim": 2,
            "action_mode": "policy",
            "domain_id": 7,
        },
        "common": {
            "action_only_output": True,
        },
    }


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


def test_format_and_tokenize_prompts_leaves_plain_prompts_unchanged_by_default(make_cosmos3_pipeline) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import COSMOS3_SYSTEM_PROMPT

    pipeline = make_cosmos3_pipeline()
    calls = _capture_tokenize_calls(pipeline)

    result = pipeline._format_and_tokenize_prompts(
        "  A robot.  ",
        "  bad.  ",
        num_frames=48,
        frame_rate=24,
        height=720,
        width=1280,
        max_sequence_length=32,
        sp=SimpleNamespace(extra_args={}),
        use_system_prompt=False,
        is_t2i=False,
    )

    assert [call["text"] for call in calls] == ["A robot.", "bad."]
    assert all(call["max_sequence_length"] == 32 for call in calls)
    assert all(call["use_system_prompt"] is False for call in calls)
    assert all(call["system_prompt"] == COSMOS3_SYSTEM_PROMPT for call in calls)
    assert [tensor.item() for tensor in result] == [1, 1, 2, 2]


def test_format_and_tokenize_prompts_applies_video_templates_and_system_override(make_cosmos3_pipeline) -> None:
    pipeline = make_cosmos3_pipeline()
    calls = _capture_tokenize_calls(pipeline)

    pipeline._format_and_tokenize_prompts(
        "A robot",
        "bad",
        num_frames=48,
        frame_rate=24,
        height=720,
        width=1280,
        max_sequence_length=32,
        sp=SimpleNamespace(
            system_prompt="direct system prompt",
            extra_args={
                "use_duration_template": True,
                "use_resolution_template": True,
                "system_prompt": "API system prompt",
            },
        ),
        use_system_prompt=True,
        is_t2i=False,
    )

    assert calls == [
        {
            "text": ("A robot. The video is 2.0 seconds long and is of 24 FPS. This video is of 720x1280 resolution."),
            "max_sequence_length": 32,
            "use_system_prompt": True,
            "system_prompt": "API system prompt",
        },
        {
            "text": (
                "bad. The video is not 2.0 seconds long and is not of 24 FPS. This video is not of 720x1280 resolution."
            ),
            "max_sequence_length": 32,
            "use_system_prompt": True,
            "system_prompt": "API system prompt",
        },
    ]


def test_format_and_tokenize_prompts_uses_image_templates_for_t2i(make_cosmos3_pipeline) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import COSMOS3_T2I_SYSTEM_PROMPT

    pipeline = make_cosmos3_pipeline()
    calls = _capture_tokenize_calls(pipeline)

    pipeline._format_and_tokenize_prompts(
        "A robot",
        "bad",
        num_frames=1,
        frame_rate=24,
        height=1024,
        width=768,
        max_sequence_length=64,
        sp=SimpleNamespace(
            extra_args={
                "use_duration_template": True,
                "use_resolution_template": True,
            }
        ),
        use_system_prompt=True,
        is_t2i=True,
    )

    assert [call["text"] for call in calls] == [
        "A robot. This image is of 1024x768 resolution.",
        "bad. This image is not of 1024x768 resolution.",
    ]
    assert all("seconds" not in call["text"] and "FPS" not in call["text"] for call in calls)
    assert all(call["system_prompt"] == COSMOS3_T2I_SYSTEM_PROMPT for call in calls)


def test_format_and_tokenize_prompts_rewrites_json_object_metadata(make_cosmos3_pipeline) -> None:
    import json

    pipeline = make_cosmos3_pipeline()
    calls = _capture_tokenize_calls(pipeline)

    pipeline._format_and_tokenize_prompts(
        ('{"caption": "A robot", "duration": "old", "fps": 1, "resolution": {"H": 1, "W": 2}, "aspect_ratio": "1:1"}'),
        "bad",
        num_frames=48,
        frame_rate=24,
        height=720,
        width=1280,
        max_sequence_length=32,
        sp=SimpleNamespace(
            extra_args={
                "aspect_ratio": "16:9",
                "use_duration_template": True,
                "use_resolution_template": True,
            }
        ),
        is_t2i=False,
    )

    assert json.loads(calls[0]["text"]) == {
        "caption": "A robot",
        "duration": "2s",
        "fps": 24.0,
        "resolution": {"H": 720, "W": 1280},
        "aspect_ratio": "16:9",
    }
    assert "The video is 2.0 seconds long" not in calls[0]["text"]
    assert calls[1]["text"] == (
        "bad. The video is not 2.0 seconds long and is not of 24 FPS. This video is not of 720x1280 resolution."
    )


def test_format_and_tokenize_prompts_removes_video_metadata_from_t2i_json(make_cosmos3_pipeline) -> None:
    import json

    pipeline = make_cosmos3_pipeline()
    calls = _capture_tokenize_calls(pipeline)

    pipeline._format_and_tokenize_prompts(
        '{"caption": "A robot", "duration": "2s", "fps": 24}',
        "",
        num_frames=1,
        frame_rate=24,
        height=1024,
        width=768,
        max_sequence_length=32,
        sp=SimpleNamespace(extra_args={}),
        is_t2i=True,
    )

    assert json.loads(calls[0]["text"]) == {
        "caption": "A robot",
        "resolution": {"H": 1024, "W": 768},
    }


@pytest.mark.parametrize(
    "prompt",
    [
        "{malformed",
        '["A robot"]',
    ],
)
def test_format_and_tokenize_prompts_falls_back_for_non_object_json(
    make_cosmos3_pipeline,
    prompt: str,
) -> None:
    pipeline = make_cosmos3_pipeline()
    calls = _capture_tokenize_calls(pipeline)

    pipeline._format_and_tokenize_prompts(
        prompt,
        "",
        num_frames=48,
        frame_rate=24,
        height=720,
        width=1280,
        max_sequence_length=32,
        sp=SimpleNamespace(
            extra_args={
                "use_duration_template": True,
                "use_resolution_template": True,
            }
        ),
        is_t2i=False,
    )

    assert calls[0]["text"] == (
        f"{prompt}. The video is 2.0 seconds long and is of 24 FPS. This video is of 720x1280 resolution."
    )


def test_checkpoint_key_remap() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline

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
        "layers.3.self_attn.k_norm_und_for_gen.weight": (
            "transformer.language_model.layers.3.self_attn.k_norm_und_for_gen.weight"
        ),
        "layers.3.self_attn.add_q_proj.weight": "transformer.gen_layers.3.cross_attention.to_q.weight",
        "layers.3.self_attn.to_add_out.weight": "transformer.gen_layers.3.cross_attention.to_out.weight",
        "layers.3.self_attn.norm_added_q.weight": "transformer.gen_layers.3.cross_attention.norm_q.weight",
        "layers.3.mlp_moe_gen.up_proj.weight": "transformer.gen_layers.3.mlp.up_proj.weight",
        "layers.3.mlp_moe_gen.down_proj.weight": "transformer.gen_layers.3.mlp.down_proj.weight",
        "transformer.model.layers.3.self_attn.add_k_proj.weight": (
            "transformer.gen_layers.3.cross_attention.to_k.weight"
        ),
    }
    assert {key: Cosmos3OmniDiffusersPipeline._remap_ckpt_key(key) for key in remaps} == remaps


def test_prepare_latents_for_video_image_sound_and_action(make_cosmos3_pipeline) -> None:
    pipeline = make_cosmos3_pipeline()
    latents = pipeline._prepare_latents(16, 24, 5, torch.Generator(device="cpu").manual_seed(0))
    assert latents.shape == (1, 2, 2, 2, 3)

    pipeline._encode_conditioning_image_latent = lambda *args, **kwargs: torch.full((1, 2, 1, 2, 3), 5.0)
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


def test_prepare_latents_i2v_encodes_only_conditioning_frame(make_cosmos3_pipeline) -> None:
    pipeline = make_cosmos3_pipeline()
    calls: list[tuple[str, tuple[int, ...]]] = []

    def record_to_vae_device(tensor: torch.Tensor, *, pin_cpu: bool = False) -> torch.Tensor:
        assert pin_cpu is True
        calls.append(("to_vae_device", tuple(tensor.shape)))
        return tensor

    class RecordingVAE(StubCosmos3VAE):
        def encode(self, video: torch.Tensor):
            calls.append(("encode", tuple(video.shape)))
            return super().encode(video)

    pipeline._to_vae_device = record_to_vae_device
    pipeline.vae = RecordingVAE(z_dim=2)
    generator = torch.Generator(device="cpu").manual_seed(0)

    latents, velocity_mask, image_latent = pipeline._prepare_latents_i2v(
        torch.zeros(1, 3, 16, 24),
        16,
        24,
        9,
        generator,
    )

    assert calls == [
        ("to_vae_device", (1, 3, 16, 24)),
        ("encode", (1, 3, 1, 16, 24)),
    ]
    assert pipeline.vae.encode_input_shapes[-1] == (1, 3, 1, 16, 24)
    assert latents.shape == (1, 2, 3, 2, 3)
    assert image_latent.shape == (1, 2, 1, 2, 3)
    torch.testing.assert_close(latents[:, :, 0:1], image_latent)
    assert not torch.allclose(latents[:, :, 1:], image_latent.expand(-1, -1, 2, -1, -1))
    assert velocity_mask.tolist() == [[[[[0.0]], [[1.0]], [[1.0]]]]]


@pytest.mark.parametrize("mode", ["policy", "forward_dynamics"])
def test_prepare_action_video_latents_encodes_only_conditioning_frame(make_cosmos3_pipeline, mode: str) -> None:
    pipeline = make_cosmos3_pipeline()
    video = torch.zeros(1, 3, 9, 24, 32)

    latents, velocity_mask, condition_latents = pipeline._prepare_latents_action_video(
        video,
        mode,
        24,
        32,
        9,
        torch.Generator(device="cpu").manual_seed(0),
        image_size=torch.tensor([1, 3, 16, 24]),
    )

    assert pipeline.vae.encode_input_shapes == [(1, 3, 1, 24, 32)]
    assert latents.shape == (1, 2, 3, 2, 3)
    assert condition_latents.shape == latents.shape
    torch.testing.assert_close(condition_latents[:, :, 0], torch.ones(1, 2, 2, 3))
    assert torch.count_nonzero(condition_latents[:, :, 1:]) == 0
    torch.testing.assert_close(latents[:, :, 0:1], condition_latents[:, :, 0:1])
    assert velocity_mask.tolist() == [[[[[0.0]], [[1.0]], [[1.0]]]]]


def test_prepare_inverse_dynamics_latents_encodes_full_video(make_cosmos3_pipeline) -> None:
    pipeline = make_cosmos3_pipeline()
    video = torch.zeros(1, 3, 9, 16, 24)

    latents, velocity_mask, condition_latents = pipeline._prepare_latents_action_video(
        video,
        "inverse_dynamics",
        16,
        24,
        9,
        torch.Generator(device="cpu").manual_seed(0),
    )

    assert pipeline.vae.encode_input_shapes == [(1, 3, 9, 16, 24)]
    assert condition_latents.shape == (1, 2, 3, 2, 3)
    torch.testing.assert_close(latents, condition_latents)
    assert torch.count_nonzero(velocity_mask) == 0


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
    i2v_noise = pipeline.scheduler.step_calls[-1][0]
    torch.testing.assert_close(i2v_noise[:, :, 0:1], torch.zeros(1, 2, 1, 1, 1))
    torch.testing.assert_close(i2v_noise[:, :, 1:2], torch.full((1, 2, 1, 1, 1), 2.0))

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


def test_diffuse_transfer_applies_control_cfg(make_cosmos3_pipeline, sequential_cfg_parallel) -> None:
    pipeline = make_cosmos3_pipeline()
    latents = torch.zeros(1, 2, 1, 1, 1)
    velocity_mask = torch.ones(1, 1, 1, 1, 1)

    result = pipeline.diffuse_transfer(
        latents=latents,
        timesteps=torch.tensor([7]),
        cond_ids=_ids(2),
        cond_mask=_mask(),
        uncond_ids=_ids(1),
        uncond_mask=_mask(),
        guidance_scale=3.0,
        control_guidance=1.5,
        control_guidance_interval=None,
        control_latents=[torch.zeros_like(latents)],
        shared_kwargs={"video_shape": (1, 1, 1), "fps": 24.0, "noisy_frame_mask": velocity_mask},
        velocity_mask=velocity_mask,
        condition_latents=torch.zeros_like(latents),
    )

    assert [(call["token"], call["has_control"]) for call in pipeline.transformer.calls] == [
        (2, True),
        (2, False),
        (1, True),
    ]
    torch.testing.assert_close(result, torch.full_like(latents, 254.0))


def test_diffuse_transfer_skips_idle_cfg_branches(make_cosmos3_pipeline, sequential_cfg_parallel) -> None:
    latents = torch.zeros(1, 2, 1, 1, 1)
    velocity_mask = torch.ones(1, 1, 1, 1, 1)

    control_only = make_cosmos3_pipeline()
    control_result = control_only.diffuse_transfer(
        latents=latents,
        timesteps=torch.tensor([7]),
        cond_ids=_ids(2),
        cond_mask=_mask(),
        uncond_ids=_ids(1),
        uncond_mask=_mask(),
        guidance_scale=1.0,
        control_guidance=1.5,
        control_guidance_interval=None,
        control_latents=[torch.zeros_like(latents)],
        shared_kwargs={"video_shape": (1, 1, 1), "fps": 24.0, "noisy_frame_mask": velocity_mask},
        velocity_mask=velocity_mask,
        condition_latents=torch.zeros_like(latents),
    )
    assert [(call["token"], call["has_control"]) for call in control_only.transformer.calls] == [
        (2, True),
        (2, False),
    ]
    torch.testing.assert_close(control_result, torch.full_like(latents, 152.0))

    text_only = make_cosmos3_pipeline()
    text_result = text_only.diffuse_transfer(
        latents=latents,
        timesteps=torch.tensor([7]),
        cond_ids=_ids(2),
        cond_mask=_mask(),
        uncond_ids=_ids(1),
        uncond_mask=_mask(),
        guidance_scale=3.0,
        control_guidance=1.0,
        control_guidance_interval=None,
        control_latents=[torch.zeros_like(latents)],
        shared_kwargs={"video_shape": (1, 1, 1), "fps": 24.0, "noisy_frame_mask": velocity_mask},
        velocity_mask=velocity_mask,
        condition_latents=torch.zeros_like(latents),
    )
    assert [(call["token"], call["has_control"]) for call in text_only.transformer.calls] == [
        (2, True),
        (1, True),
    ]
    torch.testing.assert_close(text_result, torch.full_like(latents, 104.0))


def test_diffuse_transfer_interval_switches_branch_counts(make_cosmos3_pipeline, sequential_cfg_parallel) -> None:
    pipeline = make_cosmos3_pipeline()
    latents = torch.zeros(1, 2, 1, 1, 1)
    velocity_mask = torch.ones(1, 1, 1, 1, 1)

    result = pipeline.diffuse_transfer(
        latents=latents,
        timesteps=torch.tensor([900, 500, 100]),
        cond_ids=_ids(2),
        cond_mask=_mask(),
        uncond_ids=_ids(1),
        uncond_mask=_mask(),
        guidance_scale=3.0,
        control_guidance=1.5,
        control_guidance_interval=(400.0, 1000.0),
        control_latents=[torch.zeros_like(latents)],
        shared_kwargs={"video_shape": (1, 1, 1), "fps": 24.0, "noisy_frame_mask": velocity_mask},
        velocity_mask=velocity_mask,
        condition_latents=torch.zeros_like(latents),
        guidance_interval=(800.0, 1000.0),
    )

    assert [(call["token"], call["has_control"]) for call in pipeline.transformer.calls] == [
        (2, True),
        (2, False),
        (1, True),
        (2, True),
        (2, False),
        (2, True),
    ]
    torch.testing.assert_close(result, torch.full_like(latents, 508.0))


@pytest.mark.parametrize(("hint_key", "expected_fps"), [("edge", 8.0), ("wsm", 10.0)])
def test_forward_transfer_uses_source_fps_except_wsm(make_cosmos3_pipeline, hint_key: str, expected_fps: float) -> None:
    pipeline = make_cosmos3_pipeline()
    captured: dict[str, Any] = {}

    def fake_format(prompt, negative_prompt, num_frames, frame_rate, height, width, *args, **kwargs):
        del prompt, negative_prompt, num_frames, height, width, args, kwargs
        captured["format_frame_rate"] = frame_rate
        return _ids(2), _mask(), _ids(1), _mask()

    def fake_encode(video: torch.Tensor) -> torch.Tensor:
        latent_frames = (video.shape[2] - 1) // pipeline.vae_scale_factor_temporal + 1
        return torch.ones(
            1,
            2,
            latent_frames,
            max(1, video.shape[-2] // pipeline.vae_scale_factor_spatial),
            max(1, video.shape[-1] // pipeline.vae_scale_factor_spatial),
        )

    def fake_prepare(target_norm, current_conditional_frames, generator):
        del current_conditional_frames, generator
        latent = fake_encode(target_norm)
        velocity_mask = torch.ones(1, 1, latent.shape[2], 1, 1)
        return torch.zeros_like(latent), velocity_mask, torch.zeros_like(latent)

    def fake_diffuse_transfer(**kwargs):
        captured["shared_kwargs"] = kwargs["shared_kwargs"]
        return kwargs["latents"]

    pipeline._transfer_bucket_size = lambda sp, source_hw: (16, 16)
    pipeline._format_and_tokenize_prompts = fake_format
    pipeline._encode_video_tensor = fake_encode
    pipeline._prepare_transfer_latents = fake_prepare
    pipeline.diffuse_transfer = fake_diffuse_transfer

    def fake_set_flow_shift(target):
        captured.setdefault("flow_shifts", []).append(target)
        pipeline._current_flow_shift = float(target)

    pipeline._set_flow_shift = fake_set_flow_shift
    pipeline._decode_latents = lambda latents: torch.zeros(1, 3, 5, 16, 16, device="meta")

    control = torch.zeros(3, 5, 16, 16, dtype=torch.uint8)
    request = SimpleNamespace(
        prompts=[
            {
                "prompt": "transfer",
                "modalities": ["video"],
                "additional_information": {
                    "preprocessed_transfer_video": torch.zeros(1, 3, 5, 16, 16),
                    "transfer_input_fps": 8.0,
                },
            }
        ],
        sampling_params=make_sampling_params(
            height=16,
            width=16,
            # fps omitted (no frame_rate) -> non-wsm uses the source video fps (8), wsm uses its preset (10).
            extra_args={
                hint_key: {"control": control},
                "max_frames": 5,
                "num_video_frames_per_chunk": 5,
                "show_control_condition": True,
            },
        ),
    )

    output = pipeline.forward(request)

    assert captured["format_frame_rate"] == expected_fps
    assert captured["shared_kwargs"]["fps"] == expected_fps
    assert captured["flow_shifts"] == [10.0]
    # Transfer applies the V2V flow shift when building its timestep schedule.
    assert [call["shift"] for call in pipeline.scheduler.set_timesteps_calls] == [10.0]
    assert output.output["metadata"]["video"]["fps"] == expected_fps
    assert output.output["payload"]["video"].device.type == "meta"


def test_forward_transfer_runs_multichunk_overlap_path(
    make_cosmos3_pipeline,
    sequential_cfg_parallel,
) -> None:
    pipeline = make_cosmos3_pipeline()
    captured: dict[str, Any] = {"targets": [], "conditional_frames": []}

    pipeline._transfer_bucket_size = lambda sp, source_hw: (16, 16)
    pipeline._format_and_tokenize_prompts = lambda *args, **kwargs: (_ids(2), _mask(), _ids(1), _mask())
    pipeline._set_flow_shift = lambda target, **_kwargs: captured.setdefault("flow_shifts", []).append(target)

    original_prepare = pipeline._prepare_transfer_latents

    def recording_prepare(target_norm, current_conditional_frames, generator):
        captured["targets"].append(target_norm.detach().clone())
        captured["conditional_frames"].append(current_conditional_frames)
        return original_prepare(target_norm, current_conditional_frames, generator)

    pipeline._prepare_transfer_latents = recording_prepare

    decoded_chunks = [
        torch.tensor([-0.6, -0.5, -0.4, -0.3, -0.2], dtype=torch.float32),
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32),
    ]

    def fake_decode(latents):
        chunk_values = decoded_chunks[len(captured.setdefault("decode_calls", []))]
        captured["decode_calls"].append(latents.detach().clone())
        return chunk_values.view(1, 1, 5, 1, 1).expand(1, 3, 5, 16, 16).clone()

    pipeline._decode_latents = fake_decode

    transfer_video = torch.zeros(1, 3, 8, 16, 16)
    transfer_video[:, :, 1] = 1.0
    control = torch.zeros(3, 8, 16, 16, dtype=torch.uint8)
    request = SimpleNamespace(
        prompts=[
            {
                "prompt": "transfer",
                "modalities": ["video"],
                "additional_information": {
                    "preprocessed_transfer_video": transfer_video,
                    "transfer_input_fps": 8.0,
                },
            }
        ],
        sampling_params=make_sampling_params(
            height=16,
            width=16,
            num_inference_steps=1,
            guidance_scale=1.0,
            extra_args={
                "edge": {"control": control},
                "control_guidance": 1.0,
                "max_frames": 8,
                "num_video_frames_per_chunk": 5,
                "num_conditional_frames": 1,
                "num_first_chunk_conditional_frames": 2,
            },
        ),
    )

    output = pipeline.forward(request)

    assert captured["conditional_frames"] == [2, 1]
    assert len(captured["decode_calls"]) == 2
    assert output.output["payload"]["video"].shape == (1, 3, 8, 16, 16)
    torch.testing.assert_close(
        output.output["payload"]["video"][0, 0, :, 0, 0],
        torch.tensor([-0.6, -0.5, -0.4, -0.3, -0.2, 0.2, 0.3, 0.4]),
    )
    assert output.output["metadata"]["transfer"]["controls"]["edge"].shape == (1, 3, 8, 16, 16)
    torch.testing.assert_close(captured["targets"][0][:, :, 0], torch.full((1, 3, 16, 16), -1.0))
    torch.testing.assert_close(captured["targets"][0][:, :, 1], torch.full((1, 3, 16, 16), 1.0))
    torch.testing.assert_close(captured["targets"][0][:, :, 2:], torch.full((1, 3, 3, 16, 16), 1.0))
    torch.testing.assert_close(captured["targets"][1][:, :, 0], torch.full((1, 3, 16, 16), -0.2))


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

        def fake_format(
            prompt,
            negative_prompt,
            num_frames,
            frame_rate,
            height,
            width,
            max_sequence_length,
            sp,
            use_system_prompt=False,
            is_t2i=False,
        ):
            captured["format"] = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_frames": num_frames,
                "frame_rate": frame_rate,
                "height": height,
                "width": width,
                "use_system_prompt": use_system_prompt,
                "is_t2i": is_t2i,
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

        def fake_set_flow_shift(target):
            captured.setdefault("flow_shifts", []).append(target)
            pipeline._current_flow_shift = float(target)

        pipeline._set_flow_shift = fake_set_flow_shift
        pipeline.diffuse = fake_diffuse
        pipeline._decode_latents = lambda latents: latents
        return captured

    @pytest.mark.parametrize(
        ("prompt", "sampling_params", "expected"),
        [
            (
                {"prompt": "A painted robot", "modalities": ["image"]},
                make_sampling_params(num_outputs_per_prompt=2),
                {
                    "key": "image",
                    "is_t2i": True,
                    "flow": [3.0],
                    "steps": [50, 50],
                    "frames": 1,
                },
            ),
            (
                "A warehouse robot",
                make_sampling_params(),
                {
                    "key": "video",
                    "is_t2i": False,
                    "flow": [10.0],
                    "steps": [35],
                    "frames": 189,
                },
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

        output = pipeline.forward(make_request_batch(prompt, sampling_params))

        assert expected["key"] in output.output
        assert captured["format"]["is_t2i"] is expected["is_t2i"]
        assert captured["format"]["num_frames"] == expected["frames"]
        assert captured["flow_shifts"] == expected["flow"]
        assert [call["num_inference_steps"] for call in pipeline.scheduler.set_timesteps_calls] == expected["steps"]
        assert all(call["shift"] == expected["flow"][0] for call in pipeline.scheduler.set_timesteps_calls)

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
            make_request_batch(
                {
                    "prompt": "move",
                    "modalities": ["video"],
                    "additional_information": {"preprocessed_image": image_tensor},
                },
                make_sampling_params(height=16, width=16, num_frames=5),
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
            make_request_batch(
                {
                    "prompt": "continue",
                    "modalities": ["video"],
                    "additional_information": {
                        "preprocessed_video": video_tensor,
                        "condition_frame_indexes_vision": [0],
                    },
                },
                make_sampling_params(height=16, width=16, num_frames=5),
            )
        )
        assert captured["flow_shifts"][-1] == 10.0
        assert captured["format"]["negative_prompt"] == ""
        assert captured["diffuse_calls"][-1]["shared_kwargs"]["noisy_frame_mask"] is v2v_mask
        assert captured["diffuse_calls"][-1]["condition_latents"] is v2v_condition

        pipeline.transformer = pipeline.transformer.__class__(latent_channel_size=2, sound_gen=True, sound_dim=3)
        sound_latents = torch.zeros(1, 3, 4)
        pipeline._resolve_sound_target_samples = lambda *args: (20, 2.0, 10)
        pipeline._prepare_sound_latents = lambda *args, **kwargs: (sound_latents, 4)
        pipeline._decode_sound_latents = lambda *args: torch.ones(1, 2, 20)
        output = pipeline.forward(
            make_request_batch(
                {"prompt": "A robot", "modalities": ["video"], "generate_sound": True},
                make_sampling_params(num_frames=9, frame_rate=3.0),
            )
        )
        assert captured["diffuse_calls"][-1]["sound_latents"] is sound_latents
        assert output.output["audio_sample_rate"] == 10

        pipeline.transformer = pipeline.transformer.__class__(latent_channel_size=2, action_gen=True, action_dim=4)
        output = pipeline.forward(
            make_request_batch(
                {
                    "prompt": "Pick the block.",
                    "modalities": ["video"],
                    "additional_information": {"preprocessed_image": image_tensor},
                },
                make_sampling_params(
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
        assert output.output["payload"]["actions"].shape == (1, 2, 2)
        assert output.output["metadata"]["actions"] == {
            "raw_action_dim": 2,
            "action_mode": "policy",
            "domain_id": 7,
        }
        assert "common" not in output.output["metadata"]

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

        output = pipeline.forward(make_request_batch("ignored", make_sampling_params()))

        assert captured["format"] == {
            "prompt": "Pick the cube.",
            "negative_prompt": "",
            "num_frames": 3,
            "frame_rate": 15.0,
            "height": 16,
            "width": 16,
            "use_system_prompt": False,
            "is_t2i": False,
        }
        assert "flow_shifts" not in captured
        assert pipeline.scheduler.set_timesteps_calls == []
        assert captured["prepare_action"]["clean_action"] is inputs.action_tensor
        assert captured["prepare_action"]["condition_indexes"] == [0]
        assert captured["prepare_action_video"]["kwargs"] == {"image_size": None}
        assert captured["diffuse_calls"][-1]["shared_kwargs"]["action_domain_ids"].tolist() == [7]
        assert captured["diffuse_calls"][-1]["timesteps"].tolist() == [4, 3, 2, 1]
        assert output.output["payload"]["actions"].shape == (1, 2, 2)
        assert output.output["metadata"]["actions"] == {
            "raw_action_dim": 2,
            "action_mode": "policy",
            "domain_id": 7,
        }
        assert output.output["metadata"]["common"]["action_only_output"] is True
        assert "robolab_action_postprocess" in output.output["metadata"]["internal"]
        assert "robolab_policy_inputs" not in output.output["metadata"]

    @pytest.mark.parametrize(
        ("prompt", "sampling_params", "message"),
        [
            (["one", "two"], make_sampling_params(), "single prompt"),
            ({"prompt": "one", "modalities": ["image", "video"]}, make_sampling_params(), "both image and video"),
            (
                {"prompt": "x", "modalities": ["image"], "generate_sound": True},
                make_sampling_params(),
                "only for video",
            ),
            (
                {"prompt": "x", "modalities": ["image"]},
                make_sampling_params(extra_args={"edge": {"control_path": "/tmp/control.mp4"}}),
                "transfer inference is supported only for video outputs",
            ),
            (
                {"prompt": "x", "modalities": ["video"], "generate_sound": True},
                make_sampling_params(extra_args={"edge": {"control_path": "/tmp/control.mp4"}}),
                "cannot be combined with sound generation",
            ),
            (
                {"prompt": "x", "modalities": ["video"]},
                make_sampling_params(
                    extra_args={
                        "edge": {"control_path": "/tmp/control.mp4"},
                        "action_mode": "policy",
                    }
                ),
                "cannot be combined with action generation",
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
            pipeline.forward(make_request_batch(prompt, sampling_params))
