# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.experimental.world_models.session_state import SessionStateManager

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    ("alias", "canonical_name"),
    [
        ("galbot", "embodiment_b"),
        ("agibot_gear_gripper", "embodiment_c_gripper"),
        ("agibot_gear_gripper_ext", "embodiment_c_gripper_ext"),
    ],
)
def test_action_domain_table_preserves_legacy_aliases(alias: str, canonical_name: str) -> None:
    from vllm_omni.diffusion.models.cosmos3.action import resolve_domain_id

    assert resolve_domain_id(domain_name=alias) == resolve_domain_id(domain_name=canonical_name)


def test_pipeline_declares_layerwise_offload_components() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline

    assert Cosmos3OmniDiffusersPipeline._dit_modules == ["transformer.language_model", "transformer"]
    assert Cosmos3OmniDiffusersPipeline._encoder_modules == []
    assert Cosmos3OmniDiffusersPipeline._vae_modules == ["vae"]
    assert Cosmos3OmniDiffusersPipeline._resident_modules == []
    assert hasattr(Cosmos3OmniDiffusersPipeline, "enable_omni_model_cpu_offload")

    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import (
        Cosmos3LanguageModel,
        Cosmos3VFMTransformer,
    )

    assert Cosmos3LanguageModel._layerwise_offload_blocks_attrs == ["layers"]
    assert Cosmos3VFMTransformer._layerwise_offload_blocks_attrs == ["gen_layers"]


def test_component_selective_model_offload_fails_before_component_loading(monkeypatch) -> None:
    from vllm_omni.diffusion.models.cosmos3 import pipeline_cosmos3 as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "get_local_device",
        lambda: pytest.fail("component loading must not start for an unsupported selector"),
    )
    config = SimpleNamespace(
        diffusion_offload_config={"mode": "module", "components": ["dit"]},
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        enable_distributed_layerwise_offload=False,
        dlo_use_allgather=True,
        dlo_resident_layers=0,
        pin_cpu_memory=True,
    )

    with pytest.raises(ValueError, match="does not support the dit/text_encoder component selector"):
        pipeline_module.Cosmos3OmniDiffusersPipeline(od_config=config)


def test_sampling_dtype_defaults_to_float32() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline

    assert Cosmos3OmniDiffusersPipeline.sampling_dtype == torch.float32


class StubScheduler:
    def __init__(
        self,
        timesteps: list[int] | None = None,
        *,
        flow_shift: float = 1.0,
    ) -> None:
        self.timesteps = torch.tensor(timesteps or [9, 3], dtype=torch.int64)
        self.sigmas = self.timesteps.float() / 10
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
        self.sigmas = self.timesteps.float()

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
                "hidden_states_dtype": hidden_states.dtype,
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
    module: Any = types.ModuleType("vllm_omni.diffusion.models.cosmos3.guardrails")
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
        pipeline._current_step_index = None
        pipeline._current_sigma = None
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
    values: dict[str, Any] = {
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
    ("provided", "value", "default", "is_distilled", "expected", "expected_warning"),
    [
        (False, 1.0, 7.0, False, 7.0, False),
        (True, 1.0, 7.0, False, 1.0, False),
        (True, 4.5, 7.0, False, 4.5, False),
        (False, 1.0, 7.0, True, 1.0, False),
        (True, 1.0, 7.0, True, 1.0, False),
        (True, 4.5, 7.0, True, 1.0, True),
        (True, 0.0, 7.0, True, 1.0, True),
    ],
)
def test_resolve_guidance_scale(
    make_cosmos3_pipeline,
    monkeypatch: pytest.MonkeyPatch,
    provided: bool,
    value: float,
    default: float,
    is_distilled: bool,
    expected: float,
    expected_warning: bool,
) -> None:
    from vllm_omni.diffusion.models.cosmos3 import pipeline_cosmos3

    warning_once = Mock()
    monkeypatch.setattr(pipeline_cosmos3.logger, "warning_once", warning_once)
    pipeline = make_cosmos3_pipeline()
    pipeline.is_distilled_model = is_distilled
    sp = make_sampling_params(
        guidance_scale=value,
        guidance_scale_provided=provided,
    )

    assert pipeline._resolve_guidance_scale(sp, default) == expected
    if expected_warning:
        warning_once.assert_called_once()
        assert "overridden to 1.0" in warning_once.call_args.args[0]
        assert "negative_prompt does not affect generation" in warning_once.call_args.args[0]
    else:
        warning_once.assert_not_called()


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


def test_forward_threads_request_id_to_robolab(make_cosmos3_pipeline) -> None:
    pipeline = make_cosmos3_pipeline()
    robolab_inputs = object()
    captured: dict[str, Any] = {}
    expected = object()
    pipeline._build_robolab_policy_inputs = lambda sp, prompt, request_id: robolab_inputs

    def fake_forward_robolab(sp, inputs, pipeline_start, session_id=None):
        del sp, inputs, pipeline_start
        captured["session_id"] = session_id
        return expected

    pipeline._forward_robolab_policy = fake_forward_robolab
    request = SimpleNamespace(
        prompts=["policy"],
        sampling_params=make_sampling_params(),
        request_id="robolab-request-7",
    )

    assert pipeline.forward(request) is expected
    assert captured["session_id"] == "robolab-request-7"


@pytest.mark.parametrize("format_prompt_as_json", [False, True])
def test_robolab_input_builder_threads_prompt_format_and_uses_wam(
    make_cosmos3_pipeline,
    monkeypatch: pytest.MonkeyPatch,
    format_prompt_as_json: bool,
) -> None:
    from vllm_omni.diffusion.models.cosmos3 import pipeline_cosmos3

    pipeline = make_cosmos3_pipeline()
    pipeline.transformer = StubCosmos3Transformer(action_gen=True, action_dim=64)
    captured: dict[str, Any] = {}

    def fake_transform(sample, resolution):
        captured["sample_mode"] = sample["mode"]
        captured["resolution"] = resolution
        sample["sequence_plan"] = SimpleNamespace(
            condition_frame_indexes_action=[0],
            action_start_frame_offset=1,
        )
        sample["raw_action_dim"] = torch.tensor(8)
        sample["image_size"] = torch.tensor([16, 16, 16, 16])
        if format_prompt_as_json:
            sample["ai_caption"] = {"actions": {"instruction": sample["ai_caption"]}}
        return sample

    def fake_get_transform(*, format_prompt_as_json: bool):
        captured["format_prompt_as_json"] = format_prompt_as_json
        return fake_transform

    pipeline._get_robolab_transform = fake_get_transform
    monkeypatch.setattr(pipeline_cosmos3, "get_robolab_domain_id", lambda name: 8)
    obs = {
        "prompt": "Pick up the cube.",
        "observation/image": np.zeros((16, 16, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros(7, dtype=np.float32),
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
    }
    sampling_params = make_sampling_params(
        extra_args={
            "robot_obs": obs,
            "action_chunk_size": 2,
            "image_height": 16,
            "image_width": 16,
            "format_prompt_as_json": format_prompt_as_json,
        }
    )

    inputs = pipeline._build_robolab_policy_inputs(sampling_params, request_id="request-1")

    assert inputs is not None
    assert captured == {
        "sample_mode": "wam",
        "resolution": "480",
        "format_prompt_as_json": format_prompt_as_json,
    }
    assert inputs.domain_id == 8
    assert inputs.raw_action_dim == 8
    if format_prompt_as_json:
        assert json.loads(inputs.prompt) == {"actions": {"instruction": "Pick up the cube."}}
    else:
        assert inputs.prompt == "Pick up the cube."


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
    from vllm_omni.diffusion.cache.cachedit import CUSTOM_DIT_ENABLERS
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


def test_multiview_pipeline_registers_guardrail_hooks() -> None:
    from vllm_omni.diffusion.registry import (
        _DIFFUSION_POST_PROCESS_FUNCS,
        _DIFFUSION_PRE_PROCESS_FUNCS,
    )
    from vllm_omni.model_extras.cosmos3 import COSMOS3_MULTIVIEW_EXTRA_BODY_PARAMS

    assert _DIFFUSION_PRE_PROCESS_FUNCS["Cosmos3MultiviewPipeline"] == "get_cosmos3_multiview_pre_process_func"
    assert _DIFFUSION_POST_PROCESS_FUNCS["Cosmos3MultiviewPipeline"] == "get_cosmos3_post_process_func"
    assert "guardrails" in COSMOS3_MULTIVIEW_EXTRA_BODY_PARAMS


def _server_and_request_guardrail_gate(od_config: Any, sampling_params: Any = None) -> bool:
    """Same server/per-request resolution as ``guardrails.is_guardrails_enabled``."""
    if not od_config.model_config.get("guardrails", True):
        return False
    per_request = (sampling_params.extra_args or {}).get("guardrails") if sampling_params is not None else None
    return True if per_request is None else bool(per_request)


def _multiview_guardrail_request(extra_args: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        prompt={"prompt": "Drive through the intersection."},
        sampling_params=make_sampling_params(
            extra_args={
                "multiview": {
                    "views": [
                        {"camera_key": "front", "control_path": "front.mp4", "prompt": "The front camera view."},
                        {"camera_key": "rear", "control_path": "rear.mp4"},
                        {"camera_key": "left", "control_path": "left.mp4", "prompt": " "},
                    ]
                },
                **extra_args,
            }
        ),
    )


def test_multiview_preprocess_checks_shared_prompt_and_camera_captions(fake_cosmos3_guardrails) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import (
        get_cosmos3_multiview_pre_process_func,
    )

    initialized = []
    checked = []
    fake_cosmos3_guardrails.is_guardrails_enabled = _server_and_request_guardrail_gate
    fake_cosmos3_guardrails.ensure_initialized = initialized.append
    fake_cosmos3_guardrails.check_text_safety = checked.append
    od_config = SimpleNamespace(model_config={}, tf_model_config=None)

    preprocess = get_cosmos3_multiview_pre_process_func(od_config)
    assert initialized == [od_config]

    request = _multiview_guardrail_request({})
    assert preprocess(request) is request
    assert checked == ["Drive through the intersection.", "The front camera view."]

    checked.clear()
    preprocess(_multiview_guardrail_request({"guardrails": False}))
    assert checked == []


def test_multiview_preprocess_skips_guardrail_load_when_disabled(fake_cosmos3_guardrails) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import (
        get_cosmos3_multiview_pre_process_func,
    )

    fake_cosmos3_guardrails.is_guardrails_enabled = _server_and_request_guardrail_gate
    fake_cosmos3_guardrails.ensure_initialized = Mock()
    fake_cosmos3_guardrails.check_text_safety = Mock()

    preprocess = get_cosmos3_multiview_pre_process_func(SimpleNamespace(model_config={"guardrails": False}))
    # Per-request opt-in cannot enable checks the server never loaded.
    preprocess(_multiview_guardrail_request({"guardrails": True}))

    fake_cosmos3_guardrails.ensure_initialized.assert_not_called()
    fake_cosmos3_guardrails.check_text_safety.assert_not_called()


def test_multiview_postprocess_applies_video_guardrail_per_camera(fake_cosmos3_guardrails) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import get_cosmos3_post_process_func

    checked_shapes = []

    def check_video_safety(video: torch.Tensor) -> torch.Tensor:
        checked_shapes.append(tuple(video.shape))
        return torch.zeros_like(video)

    fake_cosmos3_guardrails.is_guardrails_enabled = lambda od_config, sampling_params=None: True
    fake_cosmos3_guardrails.check_video_safety = check_video_safety
    postprocess = get_cosmos3_post_process_func(SimpleNamespace(model_config={}))

    video = torch.ones(1, 3, 2 * 3, 4, 4)
    result = postprocess(
        {
            "payload": {"video": video},
            "metadata": {"multiview": {"cameras": ["front", "rear"], "frames_per_view": 3}},
        },
        output_type="np",
    )

    assert checked_shapes == [(1, 3, 3, 4, 4), (1, 3, 3, 4, 4)]
    # The guardrail output, not the raw decode, reaches the response.
    assert np.allclose(result["payload"]["video"], 0.5)


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
        enable_session_state_manager=False,
        model="/nonexistent/model/path",
        dtype=torch.float32,
        flow_shift=None,
        quantization_config=None,
        custom_pipeline_args={},
        model_config=model_config or {},
        tf_model_config=tf_model_config,
        parallel_config=SimpleNamespace(cfg_parallel_size=1, ulysses_degree=1),
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


@pytest.mark.parametrize("cfg_parallel_size,ulysses_degree", [(1, 1), (1, 2), (2, 1), (2, 2)])
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
    cfg_parallel_size: int,
    ulysses_degree: int,
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
    scheduler_config: dict[str, Any] = {"_class_name": scheduler_class_name}
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
    od_config.parallel_config.cfg_parallel_size = cfg_parallel_size
    od_config.parallel_config.ulysses_degree = ulysses_degree
    if expected_distilled and cfg_parallel_size > 1:
        monkeypatch.setattr(
            pipeline_cosmos3.AutoTokenizer,
            "from_pretrained",
            lambda *args, **kwargs: pytest.fail("component loading must not start for distilled CFG parallelism"),
        )
        with pytest.raises(ValueError, match="Set --cfg-parallel-size 1 and use --ulysses-degree"):
            Cosmos3OmniDiffusersPipeline(od_config=od_config)
        assert StubFlowMatchScheduler.from_config_calls == []
        assert StubFlowUniPCScheduler.from_config_calls == []
        return

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

    preprocess = get_cosmos3_pre_process_func(SimpleNamespace(model_config={"guardrails": False}, tf_model_config=None))
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


def test_preprocess_v2v_decodes_uploaded_video_path(tmp_path) -> None:
    """Serving may pass multipart uploads as /tmp/...mp4 path lists (#8073)."""
    imageio = pytest.importorskip("imageio.v3")

    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import get_cosmos3_pre_process_func

    frames = [np.full((16, 32, 3), i * 40, dtype=np.uint8) for i in range(6)]
    video_path = tmp_path / "vllm_omni_video_reference_test.mp4"
    imageio.imwrite(video_path, frames, fps=4, codec="libx264")

    preprocess = get_cosmos3_pre_process_func(SimpleNamespace(model_config={"guardrails": False}, tf_model_config=None))
    request = SimpleNamespace(
        prompt={"prompt": "Continue.", "multi_modal_data": {"video": [str(video_path)]}},
        sampling_params=make_sampling_params(
            height=16,
            width=32,
            extra_args={"condition_frame_indexes_vision": [0, 1], "condition_video_keep": "first"},
        ),
    )

    additional = preprocess(request).prompt["additional_information"]
    # condition_frame_indexes_vision=[0,1] => 5 pixel frames after VAE indexing.
    assert tuple(additional["preprocessed_video"].shape) == (1, 3, 5, 16, 32)
    assert additional["condition_frame_indexes_vision"] == [0, 1]


def test_decode_path_video_frames_honors_max_frames_and_keep(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("imageio.v3")
    import imageio.v3 as iio

    from vllm_omni.diffusion.utils.video_decode import decode_path_video_frames

    source = tmp_path / "clip.mp4"
    source.write_bytes(b"placeholder")
    seen = {"n": 0}

    def fake_imiter(_path):
        for idx in range(20):
            seen["n"] += 1
            yield np.full((4, 4, 3), idx, dtype=np.uint8)

    monkeypatch.setattr(iio, "imiter", fake_imiter)

    first = decode_path_video_frames(source, max_frames=5, keep="first")
    assert len(first) == 5
    assert seen["n"] == 5
    assert first[0][0, 0, 0] == 0
    assert first[-1][0, 0, 0] == 4

    seen["n"] = 0
    last = decode_path_video_frames(source, max_frames=5, keep="last")
    assert len(last) == 5
    assert seen["n"] == 20
    assert last[0][0, 0, 0] == 15
    assert last[-1][0, 0, 0] == 19


def test_preprocess_v2v_video_path_honors_prompt_level_keep_and_indexes(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("imageio.v3")
    import imageio.v3 as iio

    from vllm_omni.diffusion.models.cosmos3 import pipeline_cosmos3 as cosmos_pipeline
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import get_cosmos3_pre_process_func
    from vllm_omni.diffusion.utils.video_decode import decode_path_video_frames

    source = tmp_path / "vllm_omni_video_reference_prompt.mp4"
    source.write_bytes(b"placeholder")
    captured: dict[str, Any] = {}

    def fake_imiter(_path):
        for idx in range(20):
            yield np.full((16, 32, 3), idx, dtype=np.uint8)

    def spy_decode(*args, **kwargs):
        frames = decode_path_video_frames(*args, **kwargs)
        captured["keep"] = kwargs.get("keep")
        captured["max_frames"] = kwargs.get("max_frames")
        captured["n"] = len(frames)
        captured["first"] = int(frames[0][0, 0, 0])
        captured["last"] = int(frames[-1][0, 0, 0])
        return frames

    monkeypatch.setattr(iio, "imiter", fake_imiter)
    monkeypatch.setattr(cosmos_pipeline, "decode_path_video_frames", spy_decode)

    preprocess = get_cosmos3_pre_process_func(SimpleNamespace(model_config={"guardrails": False}, tf_model_config=None))
    last_keep = SimpleNamespace(
        prompt={
            "prompt": "Continue.",
            "condition_video_keep": "last",
            "condition_frame_indexes_vision": [0, 1],
            "multi_modal_data": {"video": [str(source)]},
        },
        sampling_params=make_sampling_params(height=16, width=32, extra_args={}),
    )
    additional = preprocess(last_keep).prompt["additional_information"]
    assert captured["keep"] == "last"
    assert captured["max_frames"] == 5
    assert captured["n"] == 5
    assert captured["first"] == 15
    assert captured["last"] == 19
    assert tuple(additional["preprocessed_video"].shape) == (1, 3, 5, 16, 32)
    assert additional["condition_frame_indexes_vision"] == [0, 1]

    wider_indexes = SimpleNamespace(
        prompt={
            "prompt": "Continue.",
            "condition_frame_indexes_vision": [0, 1, 2],
            "multi_modal_data": {"video": [str(source)]},
        },
        sampling_params=make_sampling_params(height=16, width=32, extra_args={}),
    )
    additional = preprocess(wider_indexes).prompt["additional_information"]
    assert captured["keep"] == "first"
    assert captured["max_frames"] == 9
    assert captured["n"] == 9
    assert captured["first"] == 0
    assert captured["last"] == 8
    assert tuple(additional["preprocessed_video"].shape) == (1, 3, 9, 16, 32)
    assert additional["condition_frame_indexes_vision"] == [0, 1, 2]


def test_transfer_config_media_helpers_and_preprocess_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion.models.cosmos3 import transfer
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
        Cosmos3OmniDiffusersPipeline,
        get_cosmos3_pre_process_func,
    )

    cfg = transfer.resolve_transfer_config(make_sampling_params(extra_args={"edge": True}))
    assert cfg is not None
    assert list(cfg.hints) == ["edge"]
    assert cfg.hints["edge"].control_weight == 1.0
    assert cfg.normalized_control_weights == [1.0]
    assert cfg.guidance_scale == 3.0
    assert cfg.control_guidance == 1.5
    assert cfg.flow_shift == 10.0
    assert cfg.num_video_frames_per_chunk == 93
    assert cfg.share_vision_temporal_positions is True
    assert cfg.emphasize_control_in_prompt is True
    prompt_ablation_cfg = transfer.resolve_transfer_config(
        make_sampling_params(extra_args={"edge": True, "emphasize_control_in_prompt": False})
    )
    assert prompt_ablation_cfg is not None
    assert prompt_ablation_cfg.emphasize_control_in_prompt is False
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

    preprocess = get_cosmos3_pre_process_func(SimpleNamespace(model_config={"guardrails": False}, tf_model_config=None))

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
    assert request.sampling_params.extra_args["_cosmos3_transfer_requested_size"] == {"width": 32, "height": 16}
    assert tuple(additional["preprocessed_transfer_video"].shape) == (1, 3, 4, 192, 320)
    assert additional["transfer_input_fps"] == 12.5
    assert "preprocessed_video" not in additional


def test_transfer_resize_antialiases_downscales_like_reference() -> None:
    from vllm_omni.diffusion.models.cosmos3 import transfer

    # One-pixel stripes shrunk 3x. The reference's torchvision resize (and the
    # training loader) averages them; plain bilinear samples single columns,
    # which would return the stripes as alternating 0 and 255.
    stripes = torch.zeros(3, 2, 6, 18, dtype=torch.uint8)
    stripes[..., 1::2] = 255
    resized = transfer.resize_center_crop_uint8_cthw(stripes, 2, 6)
    assert tuple(resized.shape) == (3, 2, 2, 6)
    assert resized.min() >= 100 and resized.max() <= 155


def test_transfer_center_crop_rounds_offsets_like_torchvision() -> None:
    from vllm_omni.diffusion.models.cosmos3 import transfer

    # Cropping 7 rows to 4 leaves 3. torchvision rounds the 1.5-row offset to 2,
    # as for 1720x1080 fisheye inputs at 480p (43 spare rows, offset 22).
    rows = torch.arange(0, 70, 10, dtype=torch.uint8).reshape(1, 1, 7, 1).expand(3, 1, 7, 4).contiguous()
    cropped = transfer.resize_center_crop_uint8_cthw(rows, 4, 4)
    assert cropped[0, 0, :, 0].tolist() == [20, 30, 40, 50]


def test_transfer_control_weight_validation_and_normalization() -> None:
    from vllm_omni.diffusion.models.cosmos3 import transfer

    single = transfer.resolve_transfer_config(make_sampling_params(extra_args={"edge": {"control_weight": 0.4}}))
    assert single is not None
    assert single.hints["edge"].control_weight == 0.4
    assert single.normalized_control_weights == [1.0]

    multiple = transfer.resolve_transfer_config(
        make_sampling_params(
            extra_args={
                "edge": {"control_weight": 1.0},
                "depth": {"control_weight": 3.0, "control": torch.zeros(3, 1, 8, 8, dtype=torch.uint8)},
            }
        )
    )
    assert multiple is not None
    assert multiple.normalized_control_weights == [0.25, 0.75]

    with pytest.raises(ValueError, match="Unsupported.*weight"):
        transfer.resolve_transfer_config(make_sampling_params(extra_args={"edge": {"weight": 0.5}}))
    with pytest.raises(ValueError, match="finite and non-negative"):
        transfer.resolve_transfer_config(make_sampling_params(extra_args={"edge": {"control_weight": -0.5}}))
    with pytest.raises(ValueError, match="positive sum"):
        transfer.resolve_transfer_config(
            make_sampling_params(
                extra_args={
                    "edge": {"control_weight": 0.0},
                    "depth": {"control_weight": 0.0, "control": torch.zeros(3, 1, 8, 8, dtype=torch.uint8)},
                }
            )
        )


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


@pytest.mark.parametrize("use_path", [False, True], ids=["image", "path"])
def test_transfer_pil_conversion_returns_writable_array(tmp_path, use_path: bool) -> None:
    from vllm_omni.diffusion.models.cosmos3 import transfer

    image = Image.new("RGB", (5, 4), "red")
    value = image
    if use_path:
        value = tmp_path / "control.png"
        image.save(value)

    array = transfer._pil_to_uint8_rgb(value)

    assert array.flags.writeable


def test_transfer_vae_executor_requires_distributed_vae() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline

    pipeline = object.__new__(Cosmos3OmniDiffusersPipeline)
    executor = object()
    pipeline.vae = SimpleNamespace(distributed_executor=executor, is_distributed_enabled=lambda: True)
    assert pipeline._transfer_vae_executor() is executor

    pipeline.vae = SimpleNamespace(distributed_executor=executor, is_distributed_enabled=lambda: False)
    assert pipeline._transfer_vae_executor() is None


def test_sync_transfer_overlap_slices_output_rank_and_broadcasts() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline

    class RecordingExecutor:
        rank = 0

        def __init__(self) -> None:
            self.value = None

        def broadcast_tensor(self, value):
            self.value = value
            return value

    pipeline = object.__new__(Cosmos3OmniDiffusersPipeline)
    output = torch.arange(1 * 3 * 5 * 2 * 2).reshape(1, 3, 5, 2, 2)
    executor = RecordingExecutor()

    overlap = pipeline._sync_transfer_overlap(
        output,
        overlap_frames=2,
        reference_video=output,
        vae_executor=executor,
    )

    assert overlap is not None
    torch.testing.assert_close(overlap, output[:, :, -2:])
    assert executor.value is overlap


def test_sync_transfer_overlap_allocates_receive_buffer_on_non_output_rank() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline

    class ReceivingExecutor:
        rank = 1

        def broadcast_tensor(self, value):
            assert value.shape == (1, 3, 2, 4, 5)
            return torch.ones_like(value)

    pipeline = SimpleNamespace(device=torch.device("cpu"), vae=SimpleNamespace(dtype=torch.float32))
    reference = torch.zeros(1, 3, 6, 4, 5)

    overlap = Cosmos3OmniDiffusersPipeline._sync_transfer_overlap(
        pipeline,
        torch.empty(0),
        overlap_frames=2,
        reference_video=reference,
        vae_executor=ReceivingExecutor(),
    )

    assert overlap is not None
    assert torch.equal(overlap, torch.ones_like(overlap))


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

    fake_kernel: Any = types.ModuleType("vllm.config.kernel")
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


def test_format_and_tokenize_prompts_applies_transfer_prompt_contract(make_cosmos3_pipeline) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
        COSMOS3_TRANSFER_CONTROL_DIRECTIVE_TEMPLATE,
        COSMOS3_TRANSFER_SYSTEM_PROMPT,
    )

    pipeline = make_cosmos3_pipeline()
    calls = _capture_tokenize_calls(pipeline)
    directive = COSMOS3_TRANSFER_CONTROL_DIRECTIVE_TEMPLATE.format(hint_names="edge")

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
                "use_resolution_template": True,
                "system_prompt": "request-level system prompt",
            }
        ),
        use_system_prompt=True,
        is_t2i=False,
        system_prompt=COSMOS3_TRANSFER_SYSTEM_PROMPT,
        prompt_suffix=directive,
        use_duration_template=True,
        use_resolution_template=True,
        negative_metadata_mode="same",
    )

    metadata = "The video is 2.0 seconds long and is of 24 FPS. This video is of 720x1280 resolution."
    assert calls[0]["text"] == f"A robot. {metadata} {directive}"
    assert calls[1]["text"] == f"bad. {metadata}"
    assert all(call["use_system_prompt"] is True for call in calls)
    assert all(call["system_prompt"] == COSMOS3_TRANSFER_SYSTEM_PROMPT for call in calls)


def test_format_and_tokenize_prompts_rejects_unknown_negative_metadata_mode(make_cosmos3_pipeline) -> None:
    pipeline = make_cosmos3_pipeline()

    with pytest.raises(ValueError, match="negative_metadata_mode"):
        pipeline._format_and_tokenize_prompts(
            "A robot",
            "bad",
            num_frames=48,
            frame_rate=24,
            height=720,
            width=1280,
            max_sequence_length=32,
            sp=SimpleNamespace(extra_args={}),
            negative_metadata_mode="unexpected",
        )


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


def test_format_and_tokenize_prompts_transfer_aspect_ratio_override(make_cosmos3_pipeline) -> None:
    import json

    pipeline = make_cosmos3_pipeline()
    calls = _capture_tokenize_calls(pipeline)

    pipeline._format_and_tokenize_prompts(
        '{"caption": "A robot", "aspect_ratio": "4,3"}',
        "",
        num_frames=48,
        frame_rate=24,
        height=480,
        width=832,
        max_sequence_length=32,
        sp=SimpleNamespace(extra_args={"aspect_ratio": "4:3"}),
        is_t2i=False,
        aspect_ratio_override="16,9",
    )

    assert json.loads(calls[0]["text"])["aspect_ratio"] == "16,9"


@pytest.mark.parametrize("rank_zero", [True, False])
def test_format_and_tokenize_prompts_corrects_conflicting_aspect_ratio_metadata_on_every_rank(
    make_cosmos3_pipeline,
    monkeypatch: pytest.MonkeyPatch,
    mocker,
    rank_zero: bool,
) -> None:
    import json

    from vllm_omni.diffusion.models.cosmos3 import pipeline_cosmos3

    pipeline = make_cosmos3_pipeline()
    calls = _capture_tokenize_calls(pipeline)
    monkeypatch.setattr(pipeline_cosmos3, "_is_rank_zero", lambda: rank_zero)
    warning = mocker.patch.object(pipeline_cosmos3.logger, "warning")

    pipeline._format_and_tokenize_prompts(
        '{"caption": "A robot", "aspect_ratio": "4,3"}',
        "",
        num_frames=48,
        frame_rate=24,
        height=720,
        width=1280,
        max_sequence_length=32,
        sp=SimpleNamespace(extra_args={}),
        is_t2i=False,
    )

    assert json.loads(calls[0]["text"])["aspect_ratio"] == "16,9"
    if rank_zero:
        rendered_warning = warning.call_args.args[0] % warning.call_args.args[1:]
        assert "JSON prompt aspect_ratio='4,3' conflicts with the generated 1280x720 canvas" in rendered_warning
    else:
        warning.assert_not_called()


@pytest.mark.parametrize(
    ("width", "height", "aspect_ratio"),
    [
        (1104, 816, "4,3"),
        (832, 468, "16,9"),
    ],
)
def test_format_and_tokenize_prompts_keeps_nearest_canonical_aspect_ratio(
    make_cosmos3_pipeline,
    mocker,
    width: int,
    height: int,
    aspect_ratio: str,
) -> None:
    import json

    from vllm_omni.diffusion.models.cosmos3 import pipeline_cosmos3

    pipeline = make_cosmos3_pipeline()
    calls = _capture_tokenize_calls(pipeline)
    warning = mocker.patch.object(pipeline_cosmos3.logger, "warning")

    pipeline._format_and_tokenize_prompts(
        json.dumps({"caption": "A robot", "aspect_ratio": aspect_ratio}),
        "",
        num_frames=48,
        frame_rate=24,
        height=height,
        width=width,
        max_sequence_length=32,
        sp=SimpleNamespace(extra_args={}),
        is_t2i=False,
    )

    assert json.loads(calls[0]["text"])["aspect_ratio"] == aspect_ratio
    warning.assert_not_called()


def test_transfer_bucket_selection_warns_on_size_and_aspect_ratio_conflicts(
    make_cosmos3_pipeline,
    mocker,
) -> None:
    from vllm_omni.diffusion.models.cosmos3 import pipeline_cosmos3

    pipeline = make_cosmos3_pipeline()
    warning = mocker.patch.object(pipeline_cosmos3.logger, "warning")
    sp = make_sampling_params(
        width=832,
        height=480,
        extra_args={
            "resolution": "480",
            "aspect_ratio": "4:3",
            "_cosmos3_transfer_requested_size": {"width": 1280, "height": 720},
        },
    )

    height, width, aspect_ratio = pipeline._transfer_bucket_size(sp, (480, 832))
    assert (height, width, aspect_ratio) == (480, 832, "16,9")

    pipeline._warn_transfer_bucket_conflicts(
        sp,
        "A transfer prompt",
        source_hw=(480, 832),
        height=height,
        width=width,
        aspect_ratio=aspect_ratio,
    )

    rendered_warnings = [call.args[0] % call.args[1:] for call in warning.call_args_list]
    assert any("ignores requested size=1280x720 (WxH)" in message for message in rendered_warnings)
    assert any(
        "requested aspect_ratio='4:3' conflicts with the control-selected 16,9 bucket" in message
        for message in rendered_warnings
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


def test_sampling_state_casts_transformer_execution_to_model_dtype(make_cosmos3_pipeline) -> None:
    pipeline = make_cosmos3_pipeline()
    pipeline.dtype = torch.bfloat16

    latents = pipeline._prepare_latents(16, 24, 5, torch.Generator(device="cpu").manual_seed(0))
    assert pipeline.sampling_dtype == torch.float32
    assert latents.dtype == torch.float32

    prediction = pipeline.predict_noise(
        hidden_states=latents,
        timestep=torch.tensor([1]),
        text_ids=_ids(2),
        text_mask=_mask(),
    )

    assert pipeline.transformer.calls[-1]["hidden_states_dtype"] == torch.bfloat16
    assert prediction.dtype == torch.float32


def test_sampling_dtype_can_use_model_dtype(make_cosmos3_pipeline) -> None:
    pipeline = make_cosmos3_pipeline()
    pipeline.dtype = torch.bfloat16
    pipeline.sampling_dtype = pipeline.dtype

    latents = pipeline._prepare_latents(16, 24, 5, torch.Generator(device="cpu").manual_seed(0))
    prediction = pipeline.predict_noise(
        hidden_states=latents,
        timestep=torch.tensor([1]),
        text_ids=_ids(2),
        text_mask=_mask(),
    )

    assert latents.dtype == torch.bfloat16
    assert pipeline.transformer.calls[-1]["hidden_states_dtype"] == torch.bfloat16
    assert prediction.dtype == torch.bfloat16


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
    pipeline.dtype = torch.bfloat16
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
    assert all(call["hidden_states_dtype"] == torch.bfloat16 for call in pipeline.transformer.calls)
    assert result.dtype == torch.float32
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
    assert pipeline.transformer.calls[-1]["hidden_states_dtype"] == torch.bfloat16
    assert i2v.dtype == torch.float32
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
    assert all(call["hidden_states_dtype"] == torch.bfloat16 for call in pipeline.transformer.calls)
    assert video_result.dtype == torch.float32
    assert action_result.dtype == torch.float32
    torch.testing.assert_close(video_result, torch.full_like(latents, 4.0))
    torch.testing.assert_close(action_result, torch.full((), 44.0).expand_as(action_result))


def test_diffuse_publishes_exact_seacache_metadata_and_cfg_contexts(make_cosmos3_pipeline) -> None:
    pipeline = make_cosmos3_pipeline()
    pipeline.scheduler.sigmas = torch.tensor([0.91, 0.17])
    observations: list[tuple[str, int | None, float | None, int | None]] = []

    class RecordingHook:
        @contextmanager
        def cache_context(self, name: str):
            sigma = pipeline.current_sigma
            observations.append(
                (
                    name,
                    pipeline.current_step_index,
                    None if sigma is None else float(sigma),
                    pipeline.num_timesteps,
                )
            )
            yield

    pipeline._cache_context_factory = RecordingHook().cache_context
    pipeline.diffuse(
        latents=torch.zeros(1, 2, 1, 1, 1),
        timesteps=torch.tensor([900, 100]),
        cond_ids=_ids(2),
        cond_mask=_mask(),
        uncond_ids=_ids(1),
        uncond_mask=_mask(),
        guidance_scale=3.0,
        shared_kwargs={"video_shape": (1, 1, 1), "fps": 24.0},
    )

    assert observations == [
        ("cond", 0, pytest.approx(0.91), 2),
        ("uncond", 0, pytest.approx(0.91), 2),
        ("cond", 1, pytest.approx(0.17), 2),
        ("uncond", 1, pytest.approx(0.17), 2),
    ]
    assert pipeline.current_step_index is None
    assert pipeline.current_sigma is None


def test_diffuse_drops_session_when_progress_iteration_fails(make_cosmos3_pipeline) -> None:
    pipeline = make_cosmos3_pipeline()
    pipeline._use_session_state = True
    pipeline._memory_manager = SessionStateManager()

    class FailingProgress:
        def __iter__(self):
            raise RuntimeError("progress failed before first step")

    pipeline.progress_bar = lambda timesteps: FailingProgress()
    with pytest.raises(RuntimeError, match="progress failed"):
        pipeline.diffuse(
            latents=torch.zeros(1, 2, 1, 1, 1),
            timesteps=torch.tensor([7]),
            cond_ids=_ids(2),
            cond_mask=_mask(),
            uncond_ids=_ids(1),
            uncond_mask=_mask(),
            guidance_scale=1.0,
            shared_kwargs={"video_shape": (1, 1, 1), "fps": 24.0},
            session_id="request-that-fails",
        )

    assert "request-that-fails" not in pipeline._memory_manager


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
        control_weights=[1.0],
        shared_kwargs={"video_shape": (1, 1, 1), "fps": 24.0, "noisy_frame_mask": velocity_mask},
        velocity_mask=velocity_mask,
        condition_latents=torch.zeros_like(latents),
    )

    assert [(call["token"], call["has_control"]) for call in pipeline.transformer.calls] == [
        (2, True),
        (2, False),
        (1, True),
    ]
    assert pipeline.transformer.calls[0]["kwargs"]["control_weights"] == [1.0]
    assert "control_weights" not in pipeline.transformer.calls[1]["kwargs"]
    assert pipeline.transformer.calls[2]["kwargs"]["control_weights"] == [1.0]
    torch.testing.assert_close(result, torch.full_like(latents, 254.0))


def test_diffuse_transfer_uses_named_seacache_contexts(make_cosmos3_pipeline, sequential_cfg_parallel) -> None:
    pipeline = make_cosmos3_pipeline()
    pipeline.scheduler.sigmas = torch.tensor([0.42])
    contexts: list[tuple[str, int | None, float | None, int | None]] = []

    class RecordingHook:
        @contextmanager
        def cache_context(self, name: str):
            sigma = pipeline.current_sigma
            contexts.append(
                (
                    name,
                    pipeline.current_step_index,
                    None if sigma is None else float(sigma),
                    pipeline.num_timesteps,
                )
            )
            yield

    pipeline._cache_context_factory = RecordingHook().cache_context
    latents = torch.zeros(1, 2, 1, 1, 1)
    velocity_mask = torch.ones(1, 1, 1, 1, 1)
    pipeline.diffuse_transfer(
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

    assert contexts == [
        ("cond", 0, pytest.approx(0.42), 1),
        ("cond_no_control", 0, pytest.approx(0.42), 1),
        ("uncond", 0, pytest.approx(0.42), 1),
    ]
    assert pipeline.current_step_index is None
    assert pipeline.current_sigma is None


def test_diffuse_transfer_rejects_session_state_manager(make_cosmos3_pipeline) -> None:
    pipeline = make_cosmos3_pipeline()
    pipeline._use_session_state = True

    with pytest.raises(NotImplementedError, match="richer cache keys"):
        pipeline.diffuse_transfer(
            latents=torch.zeros(1, 2, 1, 1, 1),
            timesteps=torch.tensor([7]),
            cond_ids=_ids(2),
            cond_mask=_mask(),
            uncond_ids=_ids(1),
            uncond_mask=_mask(),
            guidance_scale=1.0,
            control_guidance=1.0,
            control_guidance_interval=None,
            control_latents=[],
            shared_kwargs={},
            velocity_mask=torch.ones(1, 1, 1, 1, 1),
            condition_latents=torch.zeros(1, 2, 1, 1, 1),
        )


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


@pytest.mark.parametrize(
    ("text_cfg_below_one", "expected_calls"),
    [
        (False, [(2, True)]),
        (True, [(2, True), (1, True)]),
    ],
)
def test_diffuse_transfer_guidance_below_one_runs_text_cfg_only_when_opted_in(
    make_cosmos3_pipeline, sequential_cfg_parallel, text_cfg_below_one, expected_calls
) -> None:
    pipeline = make_cosmos3_pipeline()
    latents = torch.zeros(1, 2, 1, 1, 1)
    velocity_mask = torch.ones(1, 1, 1, 1, 1)

    pipeline.diffuse_transfer(
        latents=latents,
        timesteps=torch.tensor([7]),
        cond_ids=_ids(2),
        cond_mask=_mask(),
        uncond_ids=_ids(1),
        uncond_mask=_mask(),
        guidance_scale=0.0,
        control_guidance=1.0,
        control_guidance_interval=None,
        control_latents=[torch.zeros_like(latents)],
        shared_kwargs={"video_shape": (1, 1, 1), "fps": 24.0, "noisy_frame_mask": velocity_mask},
        velocity_mask=velocity_mask,
        condition_latents=torch.zeros_like(latents),
        text_cfg_below_one=text_cfg_below_one,
    )

    assert [(call["token"], call["has_control"]) for call in pipeline.transformer.calls] == expected_calls


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


@pytest.mark.parametrize(
    ("hint_key", "emphasize_control", "expected_fps", "negative_prompt"),
    [
        ("edge", None, 8.0, None),
        ("wsm", None, 10.0, ""),
        ("edge", False, 8.0, "custom negative"),
    ],
)
def test_forward_transfer_uses_transfer_prompt_contract_and_source_fps_except_wsm(
    make_cosmos3_pipeline,
    hint_key: str,
    emphasize_control: bool | None,
    expected_fps: float,
    negative_prompt: str | None,
) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
        COSMOS3_TRANSFER_CONTROL_DIRECTIVE_TEMPLATE,
        COSMOS3_TRANSFER_SYSTEM_PROMPT,
    )

    pipeline = make_cosmos3_pipeline()
    captured: dict[str, Any] = {}

    def fake_format(prompt, negative_prompt, num_frames, frame_rate, height, width, *args, **kwargs):
        del num_frames, height, width, args
        captured["prompt"] = prompt
        captured["negative_prompt"] = negative_prompt
        captured["format_frame_rate"] = frame_rate
        captured["format_kwargs"] = kwargs
        captured["format_aspect_ratio"] = kwargs["aspect_ratio_override"]
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

    pipeline._transfer_bucket_size = lambda sp, source_hw: (16, 16, "1,1")
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
    extra_args = {
        hint_key: {"control": control},
        "max_frames": 5,
        "num_video_frames_per_chunk": 5,
        "show_control_condition": True,
    }
    if emphasize_control is not None:
        extra_args["emphasize_control_in_prompt"] = emphasize_control
    prompt_data = {
        "prompt": "transfer",
        "modalities": ["video"],
        "additional_information": {
            "preprocessed_transfer_video": torch.zeros(1, 3, 5, 16, 16),
            "transfer_input_fps": 8.0,
        },
    }
    if negative_prompt is not None:
        prompt_data["negative_prompt"] = negative_prompt
    request = SimpleNamespace(
        prompts=[prompt_data],
        sampling_params=make_sampling_params(
            height=16,
            width=16,
            # fps omitted (no frame_rate) -> non-wsm uses the source video fps (8), wsm uses its preset (10).
            extra_args=extra_args,
        ),
    )

    output = pipeline.forward(request)

    assert captured["format_frame_rate"] == expected_fps
    assert captured["format_aspect_ratio"] == "1,1"
    expected_negative_prompt = "" if negative_prompt is None else negative_prompt
    assert captured["negative_prompt"] == expected_negative_prompt
    assert captured["format_kwargs"]["use_system_prompt"] is True
    assert captured["format_kwargs"]["system_prompt"] == COSMOS3_TRANSFER_SYSTEM_PROMPT
    assert captured["format_kwargs"]["use_duration_template"] is True
    assert captured["format_kwargs"]["use_resolution_template"] is True
    assert captured["format_kwargs"]["negative_metadata_mode"] == "same"
    expected_suffix = (
        None if emphasize_control is False else COSMOS3_TRANSFER_CONTROL_DIRECTIVE_TEMPLATE.format(hint_names=hint_key)
    )
    assert captured["format_kwargs"]["prompt_suffix"] == expected_suffix
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

    pipeline._transfer_bucket_size = lambda sp, source_hw: (16, 16, "1,1")
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


def test_forward_transfer_non_output_rank_uses_canonical_envelope(
    make_cosmos3_pipeline,
    sequential_cfg_parallel,
) -> None:
    pipeline = make_cosmos3_pipeline()
    pipeline.vae.distributed_executor = SimpleNamespace(rank=1)
    pipeline.vae.is_distributed_enabled = lambda: True
    pipeline._transfer_bucket_size = lambda sp, source_hw: (16, 16, "1,1")
    pipeline._format_and_tokenize_prompts = lambda *args, **kwargs: (_ids(2), _mask(), _ids(1), _mask())
    pipeline._set_flow_shift = lambda *_args, **_kwargs: None
    decoded = torch.zeros(1, 3, 1, 16, 16)
    pipeline._decode_latents = lambda latents: decoded

    request = SimpleNamespace(
        prompts=[{"prompt": "transfer", "modalities": ["video"]}],
        sampling_params=make_sampling_params(
            height=16,
            width=16,
            num_inference_steps=1,
            guidance_scale=1.0,
            extra_args={
                "edge": {"control": torch.zeros(3, 1, 16, 16, dtype=torch.uint8)},
                "max_frames": 1,
                "num_video_frames_per_chunk": 1,
            },
        ),
    )

    output = pipeline.forward(request)

    assert set(output.output) == {"payload", "metadata"}
    assert set(output.output["payload"]) == {"video"}
    torch.testing.assert_close(output.output["payload"]["video"], decoded)
    assert output.output["metadata"] == {"video": {"fps": 24.0}}


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
        captured: dict[str, Any] = {"diffuse_calls": [], "prepare_calls": []}

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


# -- Multiview deployment contract (schema versions 2 and 3) -------------------


def _multiview_lidar_contract(chunk: int = 9, context: int | None = 9) -> dict[str, Any]:
    return {
        "version": "1.2",
        "dtype": "float32",
        "sample_posterior": False,
        "apply_validity_mask": True,
        "fps": 10.0,
        "latent_channels": 128,
        "temporal_compression_factor": 1,
        "spatial_compression": [16, 16],
        "network_config": {
            "resolution": [128, 1808],
            "patch_size": [2, 2],
            "depths": [1, 1, 1, 1],
            "z_dim": 128,
            "in_channels": 3,
            "temporal_downsample": [False, False, False],
        },
        "range_projection": {
            "semantic_width": 1800,
            "model_width": 1808,
            "native_height": 128,
            "model_width_transform": "circular_pad",
            "intensity_encoding": "unit",
            "min_range_m": 0.0,
            "max_range_m": 100.0,
        },
        "streaming_chunk_frames": chunk,
        "streaming_context_frames": context,
    }


def _multiview_contract(model: str) -> dict[str, Any]:
    """Contracts exactly as the fixed imaginaire4 exporter writes them for the three AV models."""
    from vllm_omni.model_extras.cosmos3 import COSMOS3_MADS_CAMERAS

    contract = {
        "schema_version": 2,
        "per_view_captions": True,
        "variable_view_count": True,
        "inference_defaults": {
            "resolution": "480",
            "fps": 30.0,
            "num_steps": 35,
            "guidance": 6.0,
            "shift": 10.0,
            "control_guidance": 1.0,
            "emphasize_control_in_prompt": True,
            "guidance_interval": None,
            "control_guidance_interval": None,
            "sigma_max": 80.0,
            "normalize_cfg": False,
            "negative_metadata_mode": "none",
        },
        "causal_training_strategy": "none",
        "attention_scope": "decomposed",
        "backend": "triton",
        "cameras": list(COSMOS3_MADS_CAMERAS),
        "max_views": len(COSMOS3_MADS_CAMERAS),
        "share_vision_temporal_positions": True,
        "decomposed_temporal_window_seconds": 0.4,
        "control_attends_sensor": True,
        "lidar_attends_captions": True,
        "align_temporal_positions_across_views": True,
        "lidar": _multiview_lidar_contract(),
        "lidar_patch_spatial_hw": [2, 2],
    }
    if model == "baseline":
        return contract
    contract.update(
        schema_version=3,
        lidar=_multiview_lidar_contract(chunk=20, context=21),
        lidar_patch_spatial_hw=[1, 1],
        rig_view_embedding={
            "num_embeddings": 12,
            "camera_ids": {camera: index for index, camera in enumerate(COSMOS3_MADS_CAMERAS)},
            "lidar_id": 11,
        },
    )
    if model == "v2":  # overlapping maskless fold, no window
        contract.update(backend="maskless", decomposed_temporal_window_seconds=None)
    else:
        assert model == "v3"  # exact-count maskless with a window is the Flex windowed mask
    return contract


def _validate_multiview_contract(contract: dict[str, Any], *, camera_patch: int = 2) -> dict[str, Any]:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import (
        _validated_multiview_deployment_config,
    )

    return _validated_multiview_deployment_config(
        {"backbone_type": "cosmos3_multiview", "latent_patch_size": camera_patch, "multiview": contract}
    )


@pytest.mark.parametrize("model", ["baseline", "v2", "v3"])
def test_multiview_contract_accepts_exported_av_models(model: str) -> None:
    validated = _validate_multiview_contract(_multiview_contract(model))

    if model == "baseline":
        assert validated["schema_version"] == 2
        assert validated["lidar_patch_spatial_hw"] == [2, 2]
        assert validated["rig_view_embedding"] is None
    else:
        assert validated["schema_version"] == 3
        assert validated["lidar_patch_spatial_hw"] == [1, 1]
        assert validated["rig_view_embedding"]["lidar_id"] == 11
        assert validated["rig_view_embedding"]["camera_ids"]["camera_front_tele_30fov"] == 6
    assert validated["backend"] == ("maskless" if model == "v2" else "triton")


@pytest.mark.parametrize(
    ("source", "override", "fa_version", "expected"),
    [
        ("triton", None, 4, "fa4"),
        ("triton", None, 3, "triton"),
        ("triton", None, None, "triton"),
        ("triton", "triton", 4, "triton"),
        ("triton", "fa4", 3, "fa4"),
        ("fa4", None, 3, "fa4"),
        ("maskless", None, 4, "maskless"),
    ],
)
def test_multiview_sparse_backend_follows_vllm_flash_attn_version(
    monkeypatch: pytest.MonkeyPatch, source: str, override: str | None, fa_version: int | None, expected: str
) -> None:
    from vllm_omni.diffusion.attention.backends.utils import fa
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import (
        COSMOS3_MULTIVIEW_BACKEND_ENV,
        Cosmos3MultiviewPipeline,
    )

    def resolve_vllm_flash_attn_version() -> int:
        if fa_version is None:
            raise RuntimeError("vLLM-bundled versioned FlashAttention requires CUDA")
        return fa_version

    if override is None:
        monkeypatch.delenv(COSMOS3_MULTIVIEW_BACKEND_ENV, raising=False)
    else:
        monkeypatch.setenv(COSMOS3_MULTIVIEW_BACKEND_ENV, override)
    monkeypatch.setattr(fa, "resolve_vllm_flash_attn_version", resolve_vllm_flash_attn_version)

    assert Cosmos3MultiviewPipeline._resolve_attention_backend({"backend": source}) == expected


@pytest.mark.parametrize("error_type", [ImportError, ModuleNotFoundError])
def test_multiview_sparse_backend_keeps_triton_when_flash_attn_import_fails(
    monkeypatch: pytest.MonkeyPatch, error_type: type[ImportError]
) -> None:
    from vllm_omni.diffusion.attention.backends.utils import fa
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import (
        COSMOS3_MULTIVIEW_BACKEND_ENV,
        Cosmos3MultiviewPipeline,
    )

    monkeypatch.delenv(COSMOS3_MULTIVIEW_BACKEND_ENV, raising=False)
    monkeypatch.setattr(
        fa,
        "resolve_vllm_flash_attn_version",
        Mock(side_effect=error_type("CUDA FlashAttention extensions unavailable")),
    )

    assert Cosmos3MultiviewPipeline._resolve_attention_backend({"backend": "triton"}) == "triton"


def test_multiview_fa4_loads_vllm_bundled_flash_attn(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion.models.cosmos3 import multiview_fa4

    cute = types.ModuleType("cutlass.cute")
    cute.jit = lambda fn: fn
    cutlass = types.ModuleType("cutlass")
    cutlass.cute = cute
    fa_cute = types.ModuleType("vllm.vllm_flash_attn.cute")
    fa_cute.flash_attn_func = object()
    fa_cute.utils = types.ModuleType("vllm.vllm_flash_attn.cute.utils")
    block_sparsity = types.ModuleType("vllm.vllm_flash_attn.cute.block_sparsity")
    block_sparsity.BlockSparseTensorsTorch = object()
    for module in (cutlass, cute, fa_cute, fa_cute.utils, block_sparsity):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    # The standalone flash-attn-4 package is no longer needed.
    monkeypatch.setitem(sys.modules, "flash_attn", None)
    monkeypatch.setitem(sys.modules, "flash_attn.cute", None)
    monkeypatch.setattr(multiview_fa4, "_load_fa4", multiview_fa4._load_fa4.__wrapped__)

    entry = multiview_fa4._load_fa4()

    assert entry.flash_attn_func is fa_cute.flash_attn_func
    assert entry.block_sparse_cls is block_sparsity.BlockSparseTensorsTorch
    assert (entry.mask_mod.__vec_size__, entry.vector_mask_mod.__vec_size__) == (1, 32)


def _fa4_test_sparsity() -> SimpleNamespace:
    return SimpleNamespace(
        partial_counts=torch.ones(1, dtype=torch.int32),
        partial_indices=torch.zeros(1, 1, dtype=torch.int32),
        full_counts=torch.zeros(1, dtype=torch.int32),
        full_indices=torch.zeros(1, 1, dtype=torch.int32),
        q_word_base=torch.zeros(256, dtype=torch.int32),
        k_group_ids=torch.zeros(128, dtype=torch.int32),
        allowed_words=torch.ones(1, dtype=torch.int32),
        q_block_size=256,
        kv_block_size=128,
        q_len=256,
        kv_len=128,
    )


@pytest.mark.parametrize(
    ("capability", "compiled"), [((9, 0), False), ((10, 0), False), ((11, 0), False), ((10, 0), True)]
)
def test_multiview_fa4_launch_preserves_sparse_mask_through_custom_op(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int], compiled: bool
) -> None:
    from vllm_omni.diffusion.models.cosmos3 import multiview_fa4

    sparsity = _fa4_test_sparsity()
    q = torch.zeros(1, sparsity.q_len, 4, 8, dtype=torch.bfloat16)
    k = v = torch.zeros(1, sparsity.kv_len, 2, 8, dtype=torch.bfloat16)
    kernel = Mock(return_value=(q + 1, None))
    entry = multiview_fa4._Fa4Entry(kernel, SimpleNamespace, object(), object())
    monkeypatch.setattr(multiview_fa4, "_load_fa4", lambda: entry)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    attention = multiview_fa4.multiview_fa4_attention
    if compiled:
        attention = torch.compile(attention, backend="eager", fullgraph=True)

    output = attention(q, k, v, sparsity)

    torch.testing.assert_close(output, q + 1)
    kernel.assert_called_once()
    kwargs = kernel.call_args.kwargs
    assert kwargs["mask_mod"] is (entry.vector_mask_mod if capability[0] in (10, 11) else entry.mask_mod)
    assert all(
        actual is expected
        for actual, expected in zip(
            kwargs["aux_tensors"], [sparsity.q_word_base, sparsity.k_group_ids, sparsity.allowed_words], strict=True
        )
    )
    block_sparse = kwargs["block_sparse_tensors"]
    assert block_sparse.block_size == (256, 128)
    for name, expected in (
        ("mask_block_cnt", sparsity.partial_counts),
        ("mask_block_idx", sparsity.partial_indices),
        ("full_block_cnt", sparsity.full_counts),
        ("full_block_idx", sparsity.full_indices),
    ):
        torch.testing.assert_close(getattr(block_sparse, name), expected[None, None])


@pytest.mark.parametrize("field", ["q_len", "kv_len", "q_word_base", "k_group_ids", "q_block_size", "kv_block_size"])
def test_multiview_fa4_rejects_invalid_mask_before_launch(monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    from vllm_omni.diffusion.models.cosmos3 import multiview_fa4

    sparsity = _fa4_test_sparsity()
    q = torch.zeros(1, sparsity.q_len, 4, 8, dtype=torch.bfloat16)
    k = v = torch.zeros(1, sparsity.kv_len, 2, 8, dtype=torch.bfloat16)
    value = getattr(sparsity, field)
    setattr(sparsity, field, value[:-1] if isinstance(value, torch.Tensor) else value - 1)
    launch = Mock(side_effect=AssertionError("Invalid mask must fail before the FA4 launch"))
    monkeypatch.setattr(multiview_fa4, "_cosmos3_multiview_fa4_op", launch)

    with pytest.raises(ValueError, match="Cosmos3 multiview FA4"):
        multiview_fa4.multiview_fa4_attention(q, k, v, sparsity)
    launch.assert_not_called()


def test_multiview_contract_defaults_legacy_lidar_patch_to_camera_patch() -> None:
    contract = _multiview_contract("baseline")
    del contract["lidar_patch_spatial_hw"]
    # Exporters from 2026-09 wrote this field before per_view_captions replaced it.
    contract["separate_view_text_tokenization"] = True

    assert _validate_multiview_contract(contract)["lidar_patch_spatial_hw"] == [2, 2]


def test_multiview_contract_keeps_unversioned_pass_through() -> None:
    contract = _multiview_contract("baseline")
    for field in ("schema_version", "lidar", "lidar_patch_spatial_hw"):
        del contract[field]
    contract["legacy_only_field"] = 1

    assert _validate_multiview_contract(contract)["legacy_only_field"] == 1


def _set(**values):
    return lambda contract: contract.update(values)


def _set_rig(**values):
    return lambda contract: contract["rig_view_embedding"].update(values)


def _drop_camera_id(contract: dict[str, Any]) -> None:
    contract["rig_view_embedding"]["camera_ids"].pop("camera_rear_fisheye_200fov")


@pytest.mark.parametrize(
    ("model", "mutate", "match"),
    [
        ("v3", _set(schema_version=4), "Unsupported Cosmos3 multiview schema_version=4"),
        ("baseline", _set(schema_version=True), "Unsupported Cosmos3 multiview schema_version=True"),
        ("baseline", _set(future_field=1), r"Unknown Cosmos3 multiview contract fields \['future_field'\]"),
        ("v3", _set(future_field=1), r"Unknown Cosmos3 multiview contract fields \['future_field'\]"),
        # Version-2 readers would silently drop these; they need version 3.
        ("v3", _set(schema_version=2, lidar_patch_spatial_hw=[2, 2]), "rig_view_embedding requires schema_version=3"),
        ("baseline", _set(lidar_patch_spatial_hw=[1, 1]), "requires schema_version=3"),
        (
            "v3",
            _set(rig_view_embedding=None, lidar_patch_spatial_hw=[2, 2]),
            "schema_version=3 requires rig_view_embedding or a LiDAR patch",
        ),
        ("baseline", _set(schema_version=3), "schema_version=3 requires rig_view_embedding or a LiDAR patch"),
        ("v3", _set(lidar_patch_spatial_hw=[1]), "lidar_patch_spatial_hw must be two positive integers"),
        ("v3", _set(lidar_patch_spatial_hw=[1, 0]), "lidar_patch_spatial_hw must be two positive integers"),
        ("v3", _set(lidar_patch_spatial_hw=[True, 1]), "lidar_patch_spatial_hw must be two positive integers"),
        ("v3", _set(lidar_patch_spatial_hw=1), "lidar_patch_spatial_hw must be two positive integers"),
        ("v3", _set(lidar=None), "lidar_patch_spatial_hw requires a lidar block"),
        ("v3", _set_rig(num_embeddings=1), "num_embeddings must be an integer >= 2"),
        ("v3", _set_rig(num_embeddings=12.0), "num_embeddings must be an integer >= 2"),
        ("v3", _drop_camera_id, "camera_ids must name exactly the exported cameras"),
        ("v3", _set_rig(camera_ids=None), "camera_ids must be an object"),
        ("v3", lambda c: c["rig_view_embedding"]["camera_ids"].update(camera_front_wide_120fov=11), r"in \[0, 10\]"),
        ("v3", lambda c: c["rig_view_embedding"]["camera_ids"].update(camera_front_wide_120fov=-1), r"in \[0, 10\]"),
        ("v3", lambda c: c["rig_view_embedding"]["camera_ids"].update(camera_front_wide_120fov=True), r"in \[0, 10\]"),
        ("v3", _set_rig(lidar_id=10), "lidar_id must be the final row 11"),
        ("v3", _set_rig(lidar_id=11.0), "lidar_id must be the final row 11"),
        ("v3", _set_rig(extra=1), "Unknown Cosmos3 multiview rig_view_embedding fields"),
        ("v3", _set(rig_view_embedding=[1]), "rig_view_embedding must be an object"),
    ],
)
def test_multiview_contract_rejects_invalid_versioned_metadata(model: str, mutate, match: str) -> None:
    contract = _multiview_contract(model)
    mutate(contract)

    with pytest.raises((TypeError, ValueError), match=match):
        _validate_multiview_contract(contract)


def test_multiview_contract_v3_accepts_lidar_patch_without_rig_embedding() -> None:
    contract = _multiview_contract("v3")
    del contract["rig_view_embedding"]

    validated = _validate_multiview_contract(contract)

    assert validated["lidar_patch_spatial_hw"] == [1, 1]
    assert validated["rig_view_embedding"] is None


def test_multiview_contract_v3_accepts_camera_only_rig_embedding() -> None:
    contract = _multiview_contract("v3")
    for field in ("lidar", "lidar_patch_spatial_hw"):
        del contract[field]

    validated = _validate_multiview_contract(contract)

    assert "lidar_patch_spatial_hw" not in validated
    assert validated["rig_view_embedding"]["num_embeddings"] == 12


@pytest.mark.parametrize(("version", "accepted"), [(2, True), (3, True), (None, False)])
def test_multiview_variable_view_count_applies_to_versioned_contracts(version: int | None, accepted: bool) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import Cosmos3MultiviewPipeline
    from vllm_omni.model_extras.cosmos3 import COSMOS3_MADS_CAMERAS

    pipeline = object.__new__(Cosmos3MultiviewPipeline)
    pipeline.multiview_cameras = COSMOS3_MADS_CAMERAS
    pipeline.multiview_config = {"schema_version": version, "per_view_captions": True, "variable_view_count": True}
    # A reordered subset: physical rig IDs, not request positions, identify cameras.
    views = [
        {"camera_key": "camera_front_tele_30fov", "prompt": "A truck merges."},
        {"camera_key": "camera_front_wide_120fov", "prompt": "A car drives."},
    ]
    sp = SimpleNamespace(extra_args={"multiview": {"views": views}})

    if accepted:
        _, parsed = pipeline._parse_multiview_request(sp)
        assert [view["camera_key"] for view in parsed] == ["camera_front_tele_30fov", "camera_front_wide_120fov"]
    else:
        with pytest.raises(ValueError, match="full exported camera order"):
            pipeline._parse_multiview_request(sp)


# -- Multiview per-camera caption parity with imaginaire4 ---------------------


def test_multiview_rig_view_caption_matches_reference_single_camera() -> None:
    from vllm_omni.diffusion.models.cosmos3.multiview_prompts import format_rig_view_captions

    assert format_rig_view_captions(["A car drives."], ["camera_front_wide_120fov"]) == [
        "This multiview driving sequence contains time-aligned recordings from 1 vehicle-mounted camera: "
        "front wide-angle camera (forward-facing, 120° FOV).\n\n"
        "The description below is for the front wide-angle camera mounted on the vehicle. "
        "This camera is facing forward and has a 120° field of view:\n\nA car drives."
    ]


def test_multiview_rig_view_captions_list_request_cameras_in_order() -> None:
    from vllm_omni.diffusion.models.cosmos3.multiview_prompts import format_rig_view_captions

    captions = format_rig_view_captions(
        ["A truck passes.", "Rain falls."], ["camera_rear_right_70fov", "camera_front_tele_30fov"]
    )

    rig = (
        "This multiview driving sequence contains time-aligned recordings from 2 vehicle-mounted cameras: "
        "rear-right camera (rear-right-facing, 70° FOV); front telephoto camera (forward-facing, 30° FOV)."
    )
    assert captions == [
        f"{rig}\n\nThe description below is for the rear-right camera mounted on the vehicle. "
        "This camera is facing rear-right and has a 70° field of view:\n\nA truck passes.",
        f"{rig}\n\nThe description below is for the front telephoto camera mounted on the vehicle. "
        "This camera is facing forward and has a 30° field of view:\n\nRain falls.",
    ]
    with pytest.raises(ValueError, match="must match the cameras"):
        format_rig_view_captions(["only one"], ["camera_rear_right_70fov", "camera_front_tele_30fov"])


def test_multiview_rig_view_attributes_cover_every_mads_camera() -> None:
    from vllm_omni.diffusion.models.cosmos3.multiview_prompts import MADS_CAMERA_ATTRIBUTES
    from vllm_omni.model_extras.cosmos3 import COSMOS3_MADS_CAMERAS

    assert tuple(MADS_CAMERA_ATTRIBUTES) == COSMOS3_MADS_CAMERAS


@pytest.mark.parametrize(("truncate", "expected"), [(False, "6.7"), (True, "6.0")])
def test_metadata_duration_truncates_to_whole_seconds_for_per_view_captions(truncate: bool, expected: str) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
        COSMOS3_DURATION_TEMPLATE,
        COSMOS3_RESOLUTION_TEMPLATE,
        Cosmos3OmniDiffusersPipeline,
    )

    prompt = Cosmos3OmniDiffusersPipeline._apply_metadata_templates(
        "A car drives.",
        201,
        30.0,
        480,
        832,
        duration_template=COSMOS3_DURATION_TEMPLATE,
        resolution_template=COSMOS3_RESOLUTION_TEMPLATE,
        truncate_duration=truncate,
    )

    assert prompt == (
        f"A car drives. The video is {expected} seconds long and is of 30 FPS. This video is of 480x832 resolution."
    )


@pytest.mark.parametrize(
    ("per_view_captions", "transfer", "joint", "expected"),
    [
        (True, True, True, "joint"),
        (True, True, False, "multiview"),
        (False, True, True, "transfer"),
        (False, True, False, "transfer"),
        (True, False, False, "transfer"),
    ],
)
def test_multiview_system_prompt_matches_reference_selection(
    per_view_captions: bool, transfer: bool, joint: bool, expected: str
) -> None:
    from vllm_omni.diffusion.models.cosmos3.multiview_prompts import (
        COSMOS3_AV_JOINT_CAMERA_LIDAR_TRANSFER_SYSTEM_PROMPT,
        COSMOS3_AV_MULTIVIEW_TRANSFER_SYSTEM_PROMPT,
    )
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import COSMOS3_TRANSFER_SYSTEM_PROMPT
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import _multiview_system_prompt

    prompts = {
        "joint": COSMOS3_AV_JOINT_CAMERA_LIDAR_TRANSFER_SYSTEM_PROMPT,
        "multiview": COSMOS3_AV_MULTIVIEW_TRANSFER_SYSTEM_PROMPT,
        "transfer": COSMOS3_TRANSFER_SYSTEM_PROMPT,
    }
    actual = _multiview_system_prompt(per_view_captions=per_view_captions, transfer=transfer, joint=joint)

    assert actual == prompts[expected]


# -- Explicit per-view negative captions --------------------------------------


@pytest.mark.parametrize("model", ["baseline", "v2", "v3"])
@pytest.mark.parametrize(
    ("source", "explicit", "separate", "expected"),
    [
        ("extra_args", "Textures crawl.", True, "Textures crawl."),
        ("prompt", "Textures crawl.", True, "Textures crawl."),
        ("attribute", "Textures crawl.", True, "Textures crawl."),
        ("prompt", "", True, ""),
        ("extra_args", "", True, ""),
        ("extra_args", None, True, ""),
        ("extra_args", "Textures crawl.", False, "Legacy negative."),
    ],
)
def test_multiview_forward_per_view_negative_captions(
    make_cosmos3_pipeline, model, source, explicit, separate, expected
) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import Cosmos3MultiviewPipeline
    from vllm_omni.model_extras.cosmos3 import COSMOS3_MADS_CAMERAS

    pipeline = make_cosmos3_pipeline()
    pipeline.__class__ = Cosmos3MultiviewPipeline
    pipeline.multiview_config = _multiview_contract(model)
    pipeline.multiview_config["per_view_captions"] = separate
    pipeline.multiview_cameras = COSMOS3_MADS_CAMERAS
    pipeline.multiview_align_temporal_positions_across_views = True
    pipeline.multiview_attention_scope = "decomposed"
    pipeline.multiview_decomposed_temporal_window_seconds = None
    pipeline.multiview_control_attends_sensor = True
    pipeline.multiview_backend = "triton"
    pipeline.transformer._pad_to_patch_size = lambda h, w: (1, 1, 0, 0)
    pipeline._prepare_camera_major_pixels = lambda *args, **kwargs: torch.zeros(1)
    pipeline._encode_multiview_video = lambda *args, **kwargs: torch.zeros(1, 2, 150, 1, 1)
    pipeline._prepare_multiview_latents = lambda **kwargs: (
        torch.zeros(1, 2, 150, 1, 1),
        torch.ones(1, 1, 150, 1, 1),
        torch.zeros(1, 2, 150, 1, 1),
    )
    diffuse_calls = []

    def diffuse(**kwargs):
        diffuse_calls.append(kwargs)
        return kwargs["latents"]

    pipeline.diffuse_transfer = diffuse
    pipeline._decode_multiview_latents = lambda *args, **kwargs: torch.zeros(1)
    tokenizations = _capture_tokenize_calls(pipeline)
    views = [
        {"camera_key": COSMOS3_MADS_CAMERAS[1], "prompt": "A truck passes.", "control_path": "right.mp4"},
        {"camera_key": COSMOS3_MADS_CAMERAS[0], "prompt": "A car drives.", "control_path": "front.mp4"},
    ]
    extra = {"multiview": {"views": views}, "wsm": {}, "aspect_ratio": "16,9"}
    prompt = {"prompt": "Driving.", "negative_prompt": "Legacy negative."}
    sp = make_sampling_params(num_frames=297, num_inference_steps=1, latents=None, extra_args=extra)
    if explicit is not None:
        if source == "prompt":
            prompt["per_view_negative_prompt"] = explicit
            extra["per_view_negative_prompt"] = "Must lose to the prompt value."
        elif source == "attribute":
            sp.per_view_negative_prompt = explicit
        else:
            extra["per_view_negative_prompt"] = explicit
    pipeline.forward(make_request_batch(prompt, sp))

    # Capture the real formatter's input to the tokenizer, and the separate
    # unconditional segments delivered to CFG, rather than just its options.
    negatives = tokenizations[1::2]
    assert len(negatives) == (2 if separate else 1)
    formatted = (
        f"{expected} The video is 9.0 seconds long and is of 30 FPS. This video is of 480x832 resolution."
        if separate and expected
        else expected
    )
    assert all(call["text"] == formatted for call in negatives)
    assert all(call["system_prompt"] == tokenizations[0]["system_prompt"] for call in negatives)
    assert diffuse_calls[0]["uncond_mask"].tolist() == ([[1, 2]] if separate else [[1]])


@pytest.mark.parametrize("negative", [[], {}, 42, True])
def test_multiview_admission_rejects_non_string_per_view_negative_prompt(negative) -> None:
    from vllm_omni.model_extras.cosmos3 import validate_multiview_request

    with pytest.raises(ValueError, match="per_view_negative_prompt must be a string"):
        validate_multiview_request({**_joint_multiview_extra(), "per_view_negative_prompt": negative})


@pytest.mark.parametrize("nested", [False, True])
def test_multiview_clients_forward_per_view_negative_prompt(tmp_path, monkeypatch, nested) -> None:
    import importlib.util
    from pathlib import Path

    from vllm_omni.model_extras.cosmos3 import COSMOS3_MULTIVIEW_EXTRA_BODY_PARAMS

    assert "per_view_negative_prompt" in COSMOS3_MULTIVIEW_EXTRA_BODY_PARAMS
    root = Path(__file__).resolve().parents[4]

    def load_client(name, relative):
        spec = importlib.util.spec_from_file_location(name, root / relative)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    # The example's engine is replaced at its public generate boundary; the
    # request preparation code still uses real sampling params.
    engine_module = types.ModuleType("vllm_omni.entrypoints.omni")
    engine_module.Omni = SimpleNamespace
    monkeypatch.setitem(sys.modules, engine_module.__name__, engine_module)
    online = load_client(
        "negative_online_client", "examples/online_serving/multiview_video/cosmos3_multiview_client.py"
    )
    offline = load_client("negative_offline_client", "examples/offline_inference/multiview_video/cosmos3_multiview.py")
    params = {
        "per_view_negative_prompt": "Textures crawl.",
        "multiview": {"views": [{"camera_key": "camera_front_wide_120fov", "prompt": "A car drives."}]},
    }
    manifest = {"prompt": "Driving.", **({"extra_params": params} if nested else params)}
    form, paths = online.prepare_request(manifest, tmp_path)
    assert json.loads(form["extra_params"])["per_view_negative_prompt"] == "Textures crawl."
    assert paths == []

    class GeneratedError(Exception):
        pass

    def generate(prompt, sampling_params):
        assert sampling_params.extra_args["per_view_negative_prompt"] == "Textures crawl."
        raise GeneratedError

    with pytest.raises(GeneratedError):
        offline._run_request(
            SimpleNamespace(generate=generate), manifest, output_dir=tmp_path, seed=0, fallback_negative_prompt=None
        )


# -- Multiview LiDAR prefix conditioning ---------------------------------------


def _joint_multiview_extra(**lidar) -> dict[str, Any]:
    return {
        "multiview": {
            "views": [
                {"camera_key": "camera_front_wide_120fov", "control_path": "front.mp4", "prompt": "A car drives."}
            ]
        },
        "wsm": True,
        "lidar": {"control_path": "hdmap.safetensors", **lidar},
    }


def test_multiview_admission_accepts_lidar_condition_prefix() -> None:
    from vllm_omni.model_extras.cosmos3 import validate_multiview_request

    for lidar in (
        {"condition_path": "measured.safetensors"},
        {"condition_path": "measured.safetensors", "num_conditional_sweeps": 3, "return_output": True},
    ):
        validate_multiview_request(_joint_multiview_extra(**lidar), variable_view_count=True)


@pytest.mark.parametrize(
    ("lidar", "match"),
    [
        ({"condition_path": "measured.pt"}, "condition_path must be a .safetensors file"),
        ({"num_conditional_sweeps": 1}, "requires lidar.condition_path"),
        ({"condition_path": "measured.safetensors", "num_conditional_sweeps": 0}, "positive integer"),
        ({"condition_path": "measured.safetensors", "num_conditional_sweeps": True}, "positive integer"),
        ({"condition_path": "measured.safetensors", "num_conditional_sweeps": 1.0}, "positive integer"),
        ({"condition_frames": 1}, r"Unsupported Cosmos3 lidar fields: \['condition_frames'\]"),
    ],
)
def test_multiview_admission_rejects_invalid_lidar_condition(lidar: dict[str, Any], match: str) -> None:
    from vllm_omni.model_extras.cosmos3 import validate_multiview_request

    with pytest.raises(ValueError, match=match):
        validate_multiview_request(_joint_multiview_extra(**lidar), variable_view_count=True)


def _multiview_pipeline_with_packed_camera_and_lidar():
    from vllm_omni.diffusion.models.cosmos3.multiview_packing import pack_state
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import Cosmos3MultiviewPipeline

    pipeline = object.__new__(Cosmos3MultiviewPipeline)
    camera = torch.full((1, 2, 3, 1, 2), 5.0)
    lidar = torch.full((1, 4, 3, 1, 1), 7.0)
    velocity_mask = torch.tensor([0.0, 1.0, 1.0]).reshape(1, 1, 3, 1, 1)  # camera frame 0 conditioned
    shared_kwargs = {
        "packed_shapes": (tuple(camera.shape[1:]), tuple(lidar.shape[1:])),
        "lidar_condition_frames": 2,
        "lidar_condition_latents": torch.full((1, 4, 2, 1, 1), -1.0),
    }
    return pipeline, pack_state([camera, lidar]), velocity_mask, shared_kwargs


def test_multiview_lidar_condition_prefix_masks_velocity_and_restores_sample() -> None:
    from vllm_omni.diffusion.models.cosmos3.multiview_packing import unpack_state

    pipeline, packed, velocity_mask, shared_kwargs = _multiview_pipeline_with_packed_camera_and_lidar()

    noise = pipeline._mask_transfer_noise(packed.clone(), velocity_mask, shared_kwargs)
    camera, lidar = unpack_state(noise, shared_kwargs["packed_shapes"])
    assert camera[:, :, 0].eq(0).all() and camera[:, :, 1:].eq(5).all()
    assert lidar[:, :, :2].eq(0).all() and lidar[:, :, 2:].eq(7).all()

    condition = torch.full((1, 2, 3, 1, 2), 9.0)
    latents = pipeline._apply_transfer_condition(packed.clone(), velocity_mask, condition, shared_kwargs)
    camera, lidar = unpack_state(latents, shared_kwargs["packed_shapes"])
    assert camera[:, :, 0].eq(9).all() and camera[:, :, 1:].eq(5).all()
    assert lidar[:, :, :2].eq(-1).all() and lidar[:, :, 2:].eq(7).all()

    # Without a LiDAR prefix the LiDAR stream is untouched.
    shared_kwargs["lidar_condition_frames"] = 0
    noise = pipeline._mask_transfer_noise(packed.clone(), velocity_mask, shared_kwargs)
    assert unpack_state(noise, shared_kwargs["packed_shapes"])[1].eq(7).all()


def test_multiview_encode_lidar_condition_bounds_and_shape(tmp_path) -> None:
    from safetensors.torch import save_file

    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import Cosmos3MultiviewPipeline

    path = tmp_path / "measured.safetensors"
    frames = torch.zeros(3, 4, 128, 1800)
    frames[0] = 10.0
    save_file({"frames": frames}, str(path))
    encoded: list[tuple[int, ...]] = []

    class FakeEncoder:
        config = {"streaming_chunk_frames": 20}

        def __call__(self, sweeps: torch.Tensor) -> torch.Tensor:
            encoded.append(tuple(sweeps.shape))
            return torch.ones(1, 128, sweeps.shape[1], 8, 113)

    pipeline = object.__new__(Cosmos3MultiviewPipeline)
    pipeline.lidar_encoder = FakeEncoder()
    pipeline.device = torch.device("cpu")
    pipeline.dtype = torch.bfloat16
    target = (1, 128, 5, 8, 113)

    latents = pipeline._encode_lidar_condition({"condition_path": str(path)}, 5, target)
    assert tuple(latents.shape) == (1, 128, 1, 8, 113)  # reference default: one measured sweep
    assert latents.dtype == torch.float32  # part of the float32 denoising state
    latents = pipeline._encode_lidar_condition({"condition_path": str(path), "num_conditional_sweeps": 3}, 5, target)
    assert tuple(latents.shape) == (1, 128, 3, 8, 113)
    # Zero-padded to the chunk boundary, capped at the request's 5 sweeps.
    assert encoded == [(3, 5, 128, 1800), (3, 5, 128, 1800)]

    with pytest.raises(ValueError, match="must leave at least one"):
        pipeline._encode_lidar_condition({"condition_path": str(path), "num_conditional_sweeps": 5}, 5, target)
    with pytest.raises(ValueError, match="requires 5 sweeps, but contains 4"):
        pipeline._encode_lidar_condition({"condition_path": str(path), "num_conditional_sweeps": 5}, 6, target)


def test_multiview_denoising_state_is_float32_with_bf16_model() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import Cosmos3MultiviewPipeline

    pipeline = object.__new__(Cosmos3MultiviewPipeline)
    pipeline.device = torch.device("cpu")
    pipeline.dtype = torch.bfloat16
    pipeline.transformer = SimpleNamespace(latent_channel_size=2)
    pipeline.vae_scale_factor_temporal = 4
    pipeline.vae_scale_factor_spatial = 8
    # Two cameras with two latent frames each; latent frame 0 of each camera is conditioned.
    encoded = torch.full((1, 2, 4, 2, 2), 0.3, dtype=torch.bfloat16)
    pipeline._encode_multiview_video = lambda video, **kwargs: encoded
    # 1 + 2**-12 rounds to 1 in BF16; injected reference noise must survive unrounded.
    injected = torch.full((1, 2, 4, 2, 2), 1.0 + 2**-12)
    prepare = dict(
        target_pixels=torch.zeros(1),
        condition_indexes=[0, 2],
        num_views=2,
        num_frames=5,
        height=16,
        width=16,
        generator=torch.Generator().manual_seed(0),
    )

    latents, velocity_mask, condition = pipeline._prepare_multiview_latents(injected_latents=injected, **prepare)
    assert latents.dtype == velocity_mask.dtype == condition.dtype == torch.float32
    assert velocity_mask.flatten().tolist() == [0.0, 1.0, 0.0, 1.0]
    assert latents[:, :, [0, 2]].eq(encoded[:, :, [0, 2]].float()).all()
    assert latents[:, :, [1, 3]].eq(1.0 + 2**-12).all()
    sampled, _, _ = pipeline._prepare_multiview_latents(injected_latents=None, **prepare)
    assert sampled.dtype == torch.float32

    # The transformer's BF16 velocity is masked in place by the float32 mask.
    pipeline, packed, velocity_mask, shared_kwargs = _multiview_pipeline_with_packed_camera_and_lidar()
    noise = pipeline._mask_transfer_noise(packed.to(torch.bfloat16), velocity_mask, shared_kwargs)
    assert noise.dtype == torch.bfloat16


def test_diffuse_transfer_runs_model_dtype_transformer_on_float32_state(make_cosmos3_pipeline) -> None:
    pipeline = make_cosmos3_pipeline()
    pipeline.dtype = torch.bfloat16
    transformer_input_dtypes: list[torch.dtype] = []
    stub_forward = pipeline.transformer.forward

    def recording_forward(*, hidden_states: torch.Tensor, **kwargs: Any):
        transformer_input_dtypes.append(hidden_states.dtype)
        return stub_forward(hidden_states=hidden_states, **kwargs)

    pipeline.transformer.forward = recording_forward
    # 1 + 2**-12 rounds to 1 in BF16; the float32 sampler state must keep it.
    latents = torch.full((1, 2, 1, 1, 1), 1.0 + 2**-12)
    velocity_mask = torch.ones(1, 1, 1, 1, 1)

    result = pipeline.diffuse_transfer(
        latents=latents,
        timesteps=torch.tensor([7]),
        cond_ids=_ids(2),
        cond_mask=_mask(),
        uncond_ids=_ids(1),
        uncond_mask=_mask(),
        guidance_scale=1.0,
        control_guidance=1.0,
        control_guidance_interval=None,
        control_latents=[torch.zeros_like(latents)],
        shared_kwargs={"video_shape": (1, 1, 1), "fps": 24.0, "noisy_frame_mask": velocity_mask},
        velocity_mask=velocity_mask,
        condition_latents=torch.zeros_like(latents),
    )

    assert transformer_input_dtypes == [torch.bfloat16]
    assert result.dtype == torch.float32
    # The stub velocity is its cond token plus the control bonus: 2 + 100.
    assert torch.equal(result, latents + 102.0)


def _chunked_lidar_encoder(chunk: int, context: int):
    """The real streaming loop of Cosmos3LidarEncoder.forward around a toy frame-causal block.

    Each output frame is the mean of every input frame the block can see (the
    kept cache plus the causal part of its chunk), so latents change whenever
    the retained history does.
    """
    from vllm_omni.diffusion.models.cosmos3.lidar import Cosmos3LidarEncoder

    class CausalMean:
        def forward_stream(self, pixels, coords, cache):
            history = pixels if cache is None else torch.cat([cache["x"][0], pixels], dim=2)
            total, new = history.shape[2], pixels.shape[2]
            running = history.mean(dim=(1, 3, 4), keepdim=True).cumsum(2)
            running = running / torch.arange(1, total + 1).view(1, 1, total, 1, 1)
            return running[:, :, total - new :].expand(-1, 6, -1, -1, -1), {"x": (history, history)}

    encoder = object.__new__(Cosmos3LidarEncoder)
    nn.Module.__init__(encoder)
    encoder.config = {
        "streaming_chunk_frames": chunk,
        "streaming_context_frames": context,
        "range_projection": {"semantic_width": 1800, "model_width": 1808, "min_range_m": 0.0, "max_range_m": 100.0},
    }
    encoder.encoder = CausalMean()
    encoder.quant_conv = nn.Identity()
    encoder.coords = torch.zeros(1)
    encoder.latent_mean = torch.zeros(1, 3, 1, 1, 1)
    encoder.latent_std = torch.ones(1, 3, 1, 1, 1)
    return encoder


@pytest.mark.parametrize("count", [3, 4, 5, 9, 10])
def test_multiview_lidar_condition_matches_reference_full_clip_encoding(tmp_path, count: int) -> None:
    """The reference encodes the prefix inside an empty full-length target clip."""
    from safetensors.torch import save_file

    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import Cosmos3MultiviewPipeline

    num_sweeps, chunk, context = 11, 4, 5  # the 20/21 streaming geometry, scaled down
    generator = torch.Generator().manual_seed(0)
    measured = torch.rand(3, count, 128, 1800, generator=generator)
    measured[0] *= 50.0
    path = tmp_path / "measured.safetensors"
    save_file({"frames": measured}, str(path))
    encoder = _chunked_lidar_encoder(chunk, context)
    pipeline = object.__new__(Cosmos3MultiviewPipeline)
    nn.Module.__init__(pipeline)
    pipeline.lidar_encoder = encoder
    pipeline.device = torch.device("cpu")
    pipeline.dtype = torch.float32

    actual = pipeline._encode_lidar_condition(
        {"condition_path": str(path), "num_conditional_sweeps": count}, num_sweeps, (1, 3, num_sweeps, 1, 1)
    )

    full_target = torch.zeros(3, num_sweeps, 128, 1800)
    full_target[:, :count] = measured
    torch.testing.assert_close(actual, encoder(full_target)[:, :, :count], rtol=0, atol=0)
    if count > chunk and count % chunk:
        # Encoding the bare prefix keeps a different history for its trailing partial chunk.
        assert not torch.equal(encoder(measured)[:, :, :count], actual)


# -- Multiview LiDAR condition uploads -----------------------------------------


def _uploaded_joint_manifest(**lidar) -> dict[str, Any]:
    return {
        "multiview": {
            "views": [{"camera_key": "camera_front_wide_120fov", "control_reference_index": 0, "prompt": "A car."}]
        },
        "wsm": True,
        "lidar": {"control_reference_index": 1, **lidar},
    }


def test_multiview_uploads_resolve_lidar_condition_reference() -> None:
    from vllm_omni.model_extras.cosmos3 import (
        has_multiview_upload_indexes,
        multiview_lidar_upload_indexes,
        resolve_multiview_uploads,
    )

    extra = _uploaded_joint_manifest(condition_reference_index=2, num_conditional_sweeps=3, return_output=True)
    assert multiview_lidar_upload_indexes(extra) == {1, 2}
    assert has_multiview_upload_indexes({"lidar": {"condition_reference_index": 0}})

    resolved = resolve_multiview_uploads(extra, ["front.mp4", "hdmap.safetensors", "measured.safetensors"])

    assert resolved["lidar"] == {
        "control_path": "hdmap.safetensors",
        "condition_path": "measured.safetensors",
        "num_conditional_sweeps": 3,
        "return_output": True,
    }
    assert resolved["multiview"]["views"][0]["control_path"] == "front.mp4"
    assert "condition_reference_index" in extra["lidar"]  # the caller's manifest is not mutated


@pytest.mark.parametrize(
    ("lidar", "paths", "match"),
    [
        ({"condition_reference_index": 1}, 2, "referenced more than once"),
        ({"condition_reference_index": 2, "condition_path": "x.safetensors"}, 3, "cannot be combined"),
        ({"condition_reference_index": 5}, 3, "condition_reference_index must be an integer index"),
        ({"condition_reference_index": "2"}, 3, "condition_reference_index must be an integer index"),
        ({"condition_reference_index": 2}, 3 + 1, "must be referenced exactly once"),
    ],
)
def test_multiview_uploads_reject_invalid_lidar_condition_reference(lidar, paths: int, match: str) -> None:
    from vllm_omni.model_extras.cosmos3 import multiview_lidar_upload_indexes, resolve_multiview_uploads

    extra = _uploaded_joint_manifest(**lidar)
    multiview_lidar_upload_indexes(extra)  # malformed indexes never raise here
    upload_paths = ["front.mp4", "hdmap.safetensors", "measured.safetensors", "extra.safetensors"][:paths]

    with pytest.raises(ValueError, match=match):
        resolve_multiview_uploads(extra, upload_paths)


def test_multiview_client_uploads_lidar_condition(tmp_path) -> None:
    import importlib.util
    from pathlib import Path

    client_path = (
        Path(__file__).resolve().parents[4] / "examples/online_serving/multiview_video/cosmos3_multiview_client.py"
    )
    spec = importlib.util.spec_from_file_location("cosmos3_multiview_client", client_path)
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    for name in ("front.mp4", "hdmap.safetensors", "measured.safetensors"):
        (tmp_path / name).write_bytes(b"x")
    manifest = {
        "prompt": "",
        "multiview": {
            "views": [{"camera_key": "camera_front_wide_120fov", "control_path": "front.mp4", "prompt": "A car."}]
        },
        "wsm": True,
        "lidar": {
            "control_path": "hdmap.safetensors",
            "condition_path": "measured.safetensors",
            "num_conditional_sweeps": 2,
        },
    }

    data, paths = client.prepare_request(manifest, tmp_path, resolution_override="480", aspect_ratio_override="16,9")

    lidar = json.loads(data["extra_params"])["lidar"]
    assert lidar == {"control_reference_index": 1, "condition_reference_index": 2, "num_conditional_sweeps": 2}
    assert [path.name for path in paths] == ["front.mp4", "hdmap.safetensors", "measured.safetensors"]
