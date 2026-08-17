# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import gc
import weakref
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import pytest
import torch
from diffusers.utils.torch_utils import randn_tensor as _diffusers_randn_tensor
from PIL import Image
from torch import nn

import vllm_omni.diffusion.models.lingbot_world.pipeline as lingbot_pipeline
from tests.diffusion.models.wan2_2.conftest import noop_progress_bar
from vllm_omni.diffusion.models.lingbot_world.actions import (
    integrate_lingbot_camera_actions,
)
from vllm_omni.diffusion.models.lingbot_world.camera import CameraTrajectory as _CameraTrajectory
from vllm_omni.diffusion.models.lingbot_world.camera import (
    build_plucker_embedding as _real_build_plucker_embedding,
)
from vllm_omni.experimental.ar_diffusion.tick_protocol import (
    ARDiffusionControlInput,
    ARDiffusionTickRequest,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_ROOT = Path(__file__).parents[4]
_LINGBOT_INIT_PATH = _ROOT / "vllm_omni/diffusion/models/lingbot_world/__init__.py"
_MODEL_INDEX_FIXTURE = Path(__file__).with_name("fixtures") / "lingbot_world_model_index.json.fixture"
_SCHEDULER_FIXTURE = Path(__file__).with_name("fixtures") / "lingbot_world_scheduler_config.json.fixture"


def _scheduler_config(**overrides):
    values = {
        "_class_name": "UniPCMultistepScheduler",
        "num_train_timesteps": 1000,
        "flow_shift": 5.0,
        "prediction_type": "flow_prediction",
        "predict_x0": True,
        "use_flow_sigmas": True,
        "use_dynamic_shifting": False,
        "use_beta_sigmas": False,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
        "final_sigmas_type": "zero",
        "timestep_spacing": "linspace",
        "solver_order": 2,
        "solver_type": "bh2",
        "lower_order_final": True,
        "disable_corrector": [],
        "time_shift_type": "exponential",
    }
    values.update(overrides)
    return values


class _AutoWeightsLoader:
    def __init__(self, module):
        self.module = module

    def load_weights(self, weights):
        loaded = self.module.transformer.load_weights(
            (name.removeprefix("transformer."), value) for name, value in weights if name.startswith("transformer.")
        )
        return {f"transformer.{name}" for name in loaded}


class _FakePretrained:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        del args, kwargs
        return cls()

    def to(self, *args, **kwargs):
        self.to_calls = getattr(self, "to_calls", [])
        self.to_calls.append((args, kwargs))
        return self


class _FakeScheduler:
    def __init__(self, **kwargs):
        self.config = SimpleNamespace(**kwargs)


class _FakeTransformerFactory:
    @classmethod
    def from_config(cls, config, *, quant_config=None, prefix=""):
        cls.last_call = (config, quant_config, prefix)
        return _RecordingTransformer()


class _Cache:
    def __init__(self, *, num_layers: int, max_tokens: int, num_local_heads: int, head_dim: int):
        shape = (1, max_tokens, num_local_heads, head_dim)
        self.self_attention = [
            SimpleNamespace(key=torch.zeros(shape), value=torch.zeros(shape)) for _ in range(num_layers)
        ]
        self.cross_attention = [None] * num_layers


class _RecordingTransformer(nn.Module):
    def __init__(self, *, raise_on_call: int | None = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.config = SimpleNamespace(
            patch_size=(1, 2, 2),
            in_channels=36,
            out_channels=16,
            text_dim=8,
            num_layers=2,
            num_attention_heads=2,
            attention_head_dim=4,
            num_frames_per_block=3,
            sliding_window_num_frames=6,
            local_attn_size=-1,
            sink_size=3,
        )
        self.blocks = nn.ModuleList([nn.Identity(), nn.Identity()])
        for block in self.blocks:
            block.self_attn = SimpleNamespace(num_local_heads=2, head_dim=4)
        self.calls: list[dict] = []
        self.cache_allocations: list[dict] = []
        self.raise_on_call = raise_on_call
        self._dtype = dtype
        self.loaded_weights: list[tuple[str, torch.Tensor]] = []

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def forward(self, **kwargs):
        call = {
            "hidden_states": kwargs["hidden_states"].detach().clone(),
            "timestep": kwargs["timestep"].detach().clone(),
            "encoder_hidden_states": kwargs["encoder_hidden_states"].detach().clone(),
            "camera_hidden_states": kwargs["camera_hidden_states"].detach().clone(),
            "cache_id": id(kwargs["cache"]),
            "start_frame": kwargs["start_frame"],
            "update_cache": kwargs["update_cache"],
        }
        self.calls.append(call)
        if self.raise_on_call == len(self.calls):
            raise RuntimeError("forced transformer failure")
        return torch.ones_like(kwargs["hidden_states"][:, :16])

    def allocate_cache(self, **kwargs):
        self.cache_allocations.append(dict(kwargs))
        patch_height, patch_width = self.config.patch_size[1:]
        window_frames = (
            self.config.local_attn_size if self.config.local_attn_size != -1 else self.config.sliding_window_num_frames
        )
        max_tokens = window_frames * (kwargs["latent_height"] // patch_height) * (kwargs["latent_width"] // patch_width)
        return _Cache(
            num_layers=self.config.num_layers,
            max_tokens=max_tokens,
            num_local_heads=2,
            head_dim=4,
        )

    def load_weights(self, weights):
        self.loaded_weights = list(weights)
        return {name for name, _ in self.loaded_weights}


class _StubVAE(_FakePretrained):
    dtype = torch.float32

    def __init__(self):
        self.config = SimpleNamespace(
            z_dim=16,
            scale_factor_temporal=4,
            scale_factor_spatial=8,
            latents_mean=[float(index) - 3.0 for index in range(16)],
            latents_std=[1.0 + index / 4.0 for index in range(16)],
        )
        self.encode_inputs: list[torch.Tensor] = []
        self.decode_inputs: list[torch.Tensor] = []
        self.on_decode = None

    def encode(self, video: torch.Tensor):
        self.encode_inputs.append(video.detach().clone())
        latent_frames = (video.shape[2] - 1) // 4 + 1
        latents = torch.zeros(
            video.shape[0],
            16,
            latent_frames,
            video.shape[-2] // 8,
            video.shape[-1] // 8,
            dtype=video.dtype,
            device=video.device,
        )
        latents[:, :, 0] = 2.0
        return SimpleNamespace(latents=latents)

    def decode(self, latents: torch.Tensor, return_dict: bool = False):
        del return_dict
        if self.on_decode is not None:
            self.on_decode()
        self.decode_inputs.append(latents.detach().clone())
        pixel_frames = (latents.shape[2] - 1) * 4 + 1
        decoded = torch.zeros(
            latents.shape[0],
            3,
            pixel_frames,
            latents.shape[-2] * 8,
            latents.shape[-1] * 8,
            dtype=latents.dtype,
            device=latents.device,
        )
        return (decoded,)


class _SamplingParams:
    def __init__(
        self,
        *,
        height: int | None = 16,
        width: int | None = 16,
        num_frames: int = 9,
        num_inference_steps: int | None = 4,
        num_outputs_per_prompt: int = 1,
        seed: int | None = 17,
        generator: torch.Generator | None = None,
        output_type: str | None = "latent",
        max_sequence_length: int | None = 512,
        extra_args: dict | None = None,
        include_action: bool = True,
    ) -> None:
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.num_inference_steps = num_inference_steps
        self.num_outputs_per_prompt = num_outputs_per_prompt
        self.seed = seed
        self.generator = (
            generator
            if generator is not None
            else (torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None)
        )
        self.output_type = output_type
        self.max_sequence_length = max_sequence_length
        self.extra_args = {"action_path": "."} if include_action else {}
        self.extra_args["_lingbot_camera_trajectory"] = _CameraTrajectory(
            poses=torch.eye(4).repeat(32, 1, 1),
            intrinsics=torch.tensor([[100.0, 100.0, 8.0, 8.0]]).repeat(32, 1),
        )
        if extra_args is not None:
            self.extra_args.update(extra_args)
        self.latents = None


class _RequestBatch:
    def __init__(self, prompt, sampling_params: _SamplingParams, *, num_reqs: int = 1):
        self._prompts = [prompt] * num_reqs
        self._sampling = sampling_params
        self.num_reqs = num_reqs

    @property
    def prompts(self):
        return self._prompts

    @property
    def sampling_params(self):
        return self._sampling

    @property
    def request_id(self):
        return "legacy-lingbot-request"


def _load_pipeline_module():
    return lingbot_pipeline


@pytest.fixture(autouse=True)
def _stub_pipeline_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace only heavyweight component construction, not the import graph."""

    loader_state = SimpleNamespace(prefetch_calls=[])

    def prefetch_subfolders(model, subfolders, *, local_files_only):
        loader_state.prefetch_calls.append((model, tuple(subfolders), local_files_only))

    def from_pretrained_with_prefetch(callable_, model, **kwargs):
        kwargs.pop("prefetch_list", None)
        return callable_(model, **kwargs)

    def load_transformer_config(model, subfolder, local_files_only):
        del model, subfolder, local_files_only
        return {
            "_class_name": "CausalLingBotWorldTransformer3DModel",
            "patch_size": [1, 2, 2],
            "in_channels": 36,
            "out_channels": 16,
            "text_dim": 8,
            "num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 4,
            "num_frames_per_block": 3,
            "sliding_window_num_frames": 6,
            "local_attn_size": -1,
        }

    def retrieve_latents(value, sample_mode="argmax"):
        assert sample_mode == "argmax"
        return value.latents

    def load_json(model, filename, local_files_only):
        del model, local_files_only
        assert filename == "scheduler/scheduler_config.json"
        return _scheduler_config()

    trajectory = _CameraTrajectory(
        poses=torch.eye(4).repeat(32, 1, 1),
        intrinsics=torch.tensor([[100.0, 100.0, 8.0, 8.0]]).repeat(32, 1),
    )

    def load_camera_trajectory(action_path):
        assert action_path
        return trajectory

    def interpolate_camera_trajectory(value, num_frames):
        return SimpleNamespace(
            poses=value.poses[:num_frames],
            intrinsics=value.intrinsics[:num_frames],
        )

    def build_plucker_embedding(value, *, height, width, target_height, target_width, device, dtype):
        del target_height, target_width
        frames = value.poses.shape[0]
        data = torch.arange(frames * 6 * height * width, device=device, dtype=torch.float32)
        return data.reshape(frames, 6, height, width).to(dtype=dtype)

    replacements = {
        "AutoTokenizer": _FakePretrained,
        "UMT5EncoderModel": _FakePretrained,
        "AutoWeightsLoader": _AutoWeightsLoader,
        "DistributedAutoencoderKLWan": _StubVAE,
        "FlowUniPCMultistepScheduler": _FakeScheduler,
        "CausalLingBotWorldTransformer3DModel": _FakeTransformerFactory,
        "get_local_device": lambda: torch.device("cpu"),
        "set_forward_context_denoise_step_idx": lambda index: None,
        "prefetch_subfolders": prefetch_subfolders,
        "from_pretrained_with_prefetch": from_pretrained_with_prefetch,
        "load_transformer_config": load_transformer_config,
        "retrieve_latents": retrieve_latents,
        "_load_json": load_json,
        "load_camera_trajectory": load_camera_trajectory,
        "interpolate_camera_trajectory": interpolate_camera_trajectory,
        "build_plucker_embedding": build_plucker_embedding,
        "randn_tensor": _diffusers_randn_tensor,
    }
    for name, value in replacements.items():
        monkeypatch.setattr(lingbot_pipeline, name, value)
    monkeypatch.setattr(lingbot_pipeline, "_loader_state", loader_state, raising=False)


def _od_config(**overrides):
    values = {
        "model": "checkpoint",
        "dtype": torch.float32,
        "flow_shift": None,
        "quantization_config": None,
        "enable_diffusion_pipeline_profiler": False,
        "model_config": {
            "lingbot_action_root": str(_ROOT),
            "ar_diffusion_height": 16,
            "ar_diffusion_width": 16,
        },
        "enable_cpu_offload": False,
        "enable_layerwise_offload": False,
        "enforce_eager": True,
        "parallel_config": SimpleNamespace(
            pipeline_parallel_size=1,
            sequence_parallel_size=1,
            cfg_parallel_size=1,
            vae_patch_parallel_size=1,
            use_hsdp=False,
            enable_expert_parallel=False,
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _prompt(*, action_path: str | None = None, images=None):
    if images is None:
        images = Image.new("RGB", (16, 16), color=(255, 128, 0))
    prompt = {
        "prompt": "move through the room",
        "multi_modal_data": {"image": images},
        "additional_information": {},
    }
    if action_path is not None:
        prompt["additional_information"]["action_path"] = action_path
    return prompt


def _pipeline(module, *, transformer=None, od_config=None):
    pipeline = module.LingBotWorldCausalDMDPipeline(od_config=od_config or _od_config())
    if transformer is not None:
        pipeline.transformer = transformer
    pipeline.encode_prompt = lambda *args, **kwargs: torch.ones(1, 512, 8)
    pipeline.progress_bar = noop_progress_bar
    return pipeline


def _preprocess_request(module, *, prompt=None, sampling=None, od_config=None):
    request = SimpleNamespace(
        prompt=_prompt() if prompt is None else prompt,
        sampling_params=_SamplingParams() if sampling is None else sampling,
    )
    preprocess = module.get_lingbot_world_pre_process_func(od_config or _od_config())
    return preprocess(request)


@pytest.mark.parametrize("offload_field", ["enable_cpu_offload", "enable_layerwise_offload"])
def test_pipeline_respects_loader_managed_component_placement(offload_field: str) -> None:
    module = _load_pipeline_module()

    pipeline = _pipeline(module, od_config=_od_config(**{offload_field: True}))

    assert getattr(pipeline.text_encoder, "to_calls", []) == []
    assert getattr(pipeline.vae, "to_calls", []) == []


def test_ar_diffusion_capability_uses_fixed_tp_local_lingbot_geometry() -> None:
    module = _load_pipeline_module()
    module.get_tensor_model_parallel_world_size = lambda: 1
    pipeline = _pipeline(module)

    spec = pipeline.ar_diffusion_kv_cache_spec()

    assert spec.num_layers == 2
    assert spec.num_kv_heads == 2
    assert spec.head_size == 4
    assert spec.tokens_per_frame == 1
    assert spec.frames_per_block == 3
    assert spec.window_frames == 3
    assert spec.sink_frames == 3
    assert [(branch.name, branch.local_index) for branch in spec.kv_branches] == [("main", 0)]
    assert spec.cross_attention_lengths == {"text": 512}
    assert spec.model_owned_state_bytes_per_session == 9_600


def test_preprocess_materializes_external_inputs_before_worker_execution(tmp_path: Path) -> None:
    module = _load_pipeline_module()
    action_root = tmp_path / "trusted-actions"
    action_dir = action_root / "forward"
    action_dir.mkdir(parents=True)
    image_path = tmp_path / "first-frame.png"
    Image.new("RGBA", (16, 16), color=(1, 2, 3, 4)).save(image_path)
    sampling = _SamplingParams(extra_args={"action_path": "forward"})
    sampling.extra_args.pop("_lingbot_camera_trajectory")
    request = SimpleNamespace(
        prompt={"prompt": "move", "multi_modal_data": {"image": str(image_path)}},
        sampling_params=sampling,
    )
    trajectory = _CameraTrajectory(
        poses=torch.eye(4).repeat(9, 1, 1),
        intrinsics=torch.ones(9, 4),
    )
    load_calls = []
    module.load_camera_trajectory = lambda action: load_calls.append(action) or trajectory

    preprocess = module.get_lingbot_world_pre_process_func(
        _od_config(model_config={"lingbot_action_root": str(action_root)})
    )
    result = preprocess(request)

    assert result is request
    assert isinstance(request.prompt["multi_modal_data"]["image"], Image.Image)
    assert request.prompt["multi_modal_data"]["image"].mode == "RGB"
    assert sampling.extra_args["action_path"] == "forward"
    assert sampling.extra_args["_lingbot_camera_trajectory"] is trajectory
    assert len(load_calls) == 1


def test_preprocess_materializes_camera_from_typed_tick_without_action_path() -> None:
    module = _load_pipeline_module()
    poses = torch.eye(4).repeat(9, 1, 1)
    intrinsics = torch.tensor([[100.0, 100.0, 8.0, 8.0]]).repeat(9, 1)
    tick = ARDiffusionTickRequest(
        session_id="world-1",
        request_id="tick-request-0",
        chunk_index=0,
        controls=(
            ARDiffusionControlInput(
                track="camera",
                schema="lingbot.camera_trajectory.v1",
                data={
                    "poses": poses.tolist(),
                    "intrinsics": intrinsics.tolist(),
                },
            ),
        ),
    )
    sampling = _SamplingParams(include_action=False)
    sampling.extra_args.update(tick.to_extra_args())
    request = SimpleNamespace(
        request_id="tick-request-0",
        prompt=_prompt(),
        sampling_params=sampling,
    )

    result = module.get_lingbot_world_pre_process_func(_od_config())(request)

    trajectory = result.sampling_params.extra_args["_lingbot_camera_trajectory"]
    torch.testing.assert_close(trajectory.poses, poses)
    torch.testing.assert_close(trajectory.intrinsics, intrinsics)


def test_preprocess_materializes_chunk_sized_actions_from_typed_tick() -> None:
    module = _load_pipeline_module()
    tick = ARDiffusionTickRequest(
        session_id="world-actions",
        request_id="tick-request-0",
        chunk_index=0,
        controls=(
            ARDiffusionControlInput(
                track="camera",
                schema="lingbot.camera_actions.v1",
                data={
                    "mode": "frames",
                    "frames": [["w"], ["w", "j"], []],
                },
            ),
        ),
    )
    sampling = _SamplingParams(include_action=False)
    sampling.extra_args.update(tick.to_extra_args())
    request = SimpleNamespace(
        request_id="tick-request-0",
        prompt=_prompt(),
        sampling_params=sampling,
    )

    result = module.get_lingbot_world_pre_process_func(_od_config())(request)

    assert result.sampling_params.extra_args["_lingbot_camera_trajectory"] is None
    assert result.sampling_params.extra_args["_lingbot_camera_actions"] == (
        ("w",),
        ("w", "j"),
        (),
    )


def _request(*, sampling=None, prompt=None, num_reqs: int = 1):
    return _RequestBatch(
        _prompt() if prompt is None else prompt,
        _SamplingParams() if sampling is None else sampling,
        num_reqs=num_reqs,
    )


def _resolve_pipeline_through_real_registry(pipeline_module):
    from vllm_omni.diffusion import registry as registry_module

    resolved = registry_module.DiffusionModelRegistry._try_load_model_cls("LingBotWorldCausalDMDPipeline")
    entry = registry_module._DIFFUSION_MODELS["LingBotWorldCausalDMDPipeline"]
    cache_acceleration_disabled = "LingBotWorldCausalDMDPipeline" in registry_module._NO_CACHE_ACCELERATION
    preprocess_name = registry_module._DIFFUSION_PRE_PROCESS_FUNCS["LingBotWorldCausalDMDPipeline"]
    preprocess = registry_module.get_diffusion_pre_process_func(
        SimpleNamespace(
            model_class_name="LingBotWorldCausalDMDPipeline",
            model_config={"lingbot_action_root": str(_ROOT)},
        )
    )
    return resolved, entry, cache_acceleration_disabled, preprocess_name, preprocess


def test_component_discovery_uses_official_checkpoint_contract() -> None:
    module = _load_pipeline_module()
    pipeline = module.LingBotWorldCausalDMDPipeline(od_config=_od_config())

    assert pipeline._dit_modules == ["transformer"]
    assert pipeline._encoder_modules == ["text_encoder"]
    assert pipeline._vae_modules == ["vae"]
    assert pipeline.dummy_run_num_frames == 0
    assert pipeline.weights_sources == [
        module.DiffusersPipelineLoader.ComponentSource("checkpoint", "transformer", None, "transformer.", True)
    ]
    assert _FakeTransformerFactory.last_call[1:] == (None, "transformer")
    assert pipeline.scheduler.config.shift == 5.0
    assert pipeline.scheduler.config.num_train_timesteps == 1000
    assert module._loader_state.prefetch_calls == [("checkpoint", ("tokenizer", "text_encoder", "vae"), False)]


@pytest.mark.parametrize(
    ("field", "value", "feature"),
    [
        ("pipeline_parallel_size", 2, "pipeline parallelism"),
        ("sequence_parallel_size", 2, "sequence parallelism"),
        ("cfg_parallel_size", 2, "CFG parallelism"),
        ("vae_patch_parallel_size", 2, "VAE parallelism"),
        ("use_hsdp", True, "HSDP"),
        ("enable_expert_parallel", True, "expert parallelism"),
    ],
)
def test_unsupported_parallel_modes_fail_before_component_loading(field: str, value: object, feature: str) -> None:
    module = _load_pipeline_module()
    parallel_config = _od_config().parallel_config
    setattr(parallel_config, field, value)

    with pytest.raises(NotImplementedError, match=feature):
        module.LingBotWorldCausalDMDPipeline(od_config=_od_config(parallel_config=parallel_config))

    assert module._loader_state.prefetch_calls == []


def test_unsupported_quantization_fails_before_component_loading() -> None:
    module = _load_pipeline_module()

    with pytest.raises(NotImplementedError, match="quantization"):
        module.LingBotWorldCausalDMDPipeline(od_config=_od_config(quantization_config=object()))

    assert module._loader_state.prefetch_calls == []


def test_official_scheduler_config_matches_fixed_dmd_contract() -> None:
    import json

    module = _load_pipeline_module()
    scheduler_config = json.loads(_SCHEDULER_FIXTURE.read_text())
    module._load_json = lambda *args, **kwargs: scheduler_config.copy()

    pipeline = _pipeline(module)

    assert pipeline.scheduler.config.num_train_timesteps == 1000
    assert pipeline.scheduler.config.shift == 5.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("_class_name", "FlowMatchEulerDiscreteScheduler"),
        ("num_train_timesteps", 999),
        ("prediction_type", "epsilon"),
        ("predict_x0", False),
        ("use_flow_sigmas", False),
        ("use_dynamic_shifting", True),
        ("final_sigmas_type", "sigma_min"),
    ],
)
def test_scheduler_config_rejects_semantic_drift(field: str, value: object) -> None:
    module = _load_pipeline_module()
    module._load_json = lambda *args, **kwargs: _scheduler_config(**{field: value})

    with pytest.raises(ValueError, match=field):
        _pipeline(module)


def test_official_model_index_discovers_only_declared_components() -> None:
    import json

    module = _load_pipeline_module()
    model_index = json.loads(_MODEL_INDEX_FIXTURE.read_text())
    resolved, entry, cache_acceleration_disabled, preprocess_name, preprocess = _resolve_pipeline_through_real_registry(
        module
    )

    assert model_index["_class_name"] == "LingBotWorldCausalDMDPipeline"
    assert resolved is module.LingBotWorldCausalDMDPipeline
    assert entry == ("lingbot_world", "pipeline", "LingBotWorldCausalDMDPipeline")
    assert cache_acceleration_disabled
    assert preprocess_name == "get_lingbot_world_pre_process_func"
    assert callable(preprocess)
    assert model_index["tokenizer"] == ["transformers", "T5TokenizerFast"]
    assert model_index["text_encoder"] == ["transformers", "UMT5EncoderModel"]
    assert model_index["vae"] == ["diffusers", "AutoencoderKLWan"]
    assert model_index["scheduler"] == ["diffusers", "UniPCMultistepScheduler"]
    assert model_index["transformer"] == ["diffusers", "CausalLingBotWorldTransformer3DModel"]
    assert model_index["image_encoder"] == [None, None]
    assert model_index["image_processor"] == [None, None]
    assert model_index["transformer_2"] == [None, None]


def test_postprocess_returns_latents_without_video_conversion() -> None:
    module = _load_pipeline_module()
    postprocess = module.get_lingbot_world_post_process_func(_od_config())
    latents = torch.randn(1, 16, 3, 2, 2)

    output = postprocess(latents, sampling_params=SimpleNamespace(output_type="latent"))

    assert output is latents


def test_postprocess_uses_the_standard_diffusion_output_envelope(monkeypatch) -> None:
    import diffusers.video_processor as video_processor_module

    module = _load_pipeline_module()
    processed_video = object()
    calls = []

    class VideoProcessor:
        def __init__(self, *, vae_scale_factor):
            assert vae_scale_factor == 8

        def postprocess_video(self, video, *, output_type):
            calls.append((video, output_type))
            return processed_video

    monkeypatch.setattr(video_processor_module, "VideoProcessor", VideoProcessor)
    postprocess = module.get_lingbot_world_post_process_func(_od_config())
    video = torch.randn(1, 3, 9, 8, 8)

    output = postprocess(video, output_type="np")

    assert output == {"payload": {"video": processed_video}, "metadata": {}}
    assert calls == [(video, "np")]


def test_forward_propagates_pipeline_profiler_stage_durations() -> None:
    module = _load_pipeline_module()
    pipeline = _pipeline(module)
    pipeline._profiler_lock = Lock()
    pipeline._stage_durations = {"_generate_block": 0.25, "vae.decode": 0.5}

    output = pipeline(_request())

    assert output.stage_durations == pipeline.stage_durations


def test_pipeline_weight_loader_preserves_component_parameter_prefixes() -> None:
    module = _load_pipeline_module()
    pipeline = _pipeline(module)
    weight = torch.randn(2, 2)

    loaded = pipeline.load_weights([("transformer.blocks.0.weight", weight)])

    assert loaded == {"transformer.blocks.0.weight"}
    assert pipeline.transformer.loaded_weights == [("blocks.0.weight", weight)]


def test_request_requires_runner_provided_generator() -> None:
    module = _load_pipeline_module()
    sampling = _SamplingParams(seed=None, generator=None)

    with pytest.raises(ValueError, match="runner-provided torch.Generator"):
        _pipeline(module)._parse_request(_RequestBatch(_prompt(), sampling))


@pytest.mark.parametrize("unsupported", ["generator-list", "caller-latents"])
def test_request_rejects_unsupported_sampling_state(unsupported: str) -> None:
    module = _load_pipeline_module()
    sampling = _SamplingParams()
    if unsupported == "generator-list":
        sampling.generator = [sampling.generator]
        message = "generator list"
    else:
        sampling.latents = torch.zeros(1, 16, 3, 2, 2)
        message = "caller-provided latents"

    with pytest.raises(ValueError, match=message):
        _pipeline(module)._parse_request(_RequestBatch(_prompt(), sampling))


def test_denoise_state_stays_fp32_while_transformer_inputs_use_model_dtype() -> None:
    module = _load_pipeline_module()
    transformer = _RecordingTransformer(dtype=torch.bfloat16)
    pipeline = _pipeline(module, transformer=transformer)
    requested_noise_dtypes: list[torch.dtype] = []

    def randn(shape, *, generator, device, dtype):
        del generator
        assert device == pipeline.device
        requested_noise_dtypes.append(dtype)
        return torch.zeros(shape, device=device, dtype=dtype)

    module.randn_tensor = randn
    result = pipeline(_request())

    assert requested_noise_dtypes == [torch.float32] * 4
    assert result.output.dtype == torch.float32
    assert all(call["hidden_states"].dtype == torch.bfloat16 for call in transformer.calls)


def test_path_image_rejects_oversized_source_before_decode_or_convert(monkeypatch, tmp_path: Path) -> None:
    module = _load_pipeline_module()
    source_path = tmp_path / "oversized-compressed.png"
    Image.new("1", (4097, 4097)).save(source_path)
    decode_calls: list[str] = []

    def forbidden_convert(*args, **kwargs):
        del args, kwargs
        decode_calls.append("convert")
        raise AssertionError("oversized source reached convert")

    def forbidden_load(*args, **kwargs):
        del args, kwargs
        decode_calls.append("load")
        raise AssertionError("oversized source reached load")

    monkeypatch.setattr(module.PIL.Image.Image, "convert", forbidden_convert)
    monkeypatch.setattr(module.PIL.Image.Image, "load", forbidden_load)

    with pytest.raises(ValueError, match="source image.*4096.*4096"):
        _preprocess_request(module, prompt=_prompt(images=str(source_path)))

    assert decode_calls == []


def test_normal_path_image_is_decoded_after_source_size_validation(tmp_path: Path) -> None:
    module = _load_pipeline_module()
    source_path = tmp_path / "normal.png"
    Image.new("RGBA", (32, 24), color=(1, 2, 3, 128)).save(source_path)

    request = _preprocess_request(module, prompt=_prompt(images=str(source_path)))
    parsed = _pipeline(module)._parse_request(_RequestBatch(request.prompt, request.sampling_params))

    assert module._MAX_SOURCE_IMAGE_PIXELS == 4096 * 4096
    assert isinstance(parsed.image, Image.Image)
    assert parsed.image.mode == "RGB"
    assert parsed.image.size == (32, 24)


@pytest.mark.parametrize(
    ("source_size", "expected_size"),
    [
        ((832, 480), (464, 832)),
        ((480, 832), (832, 480)),
        ((512, 512), (624, 624)),
    ],
)
def test_default_resolution_matches_official_480p_aspect_derivation(source_size, expected_size) -> None:
    module = _load_pipeline_module()
    sampling = _SamplingParams(height=None, width=None)

    parsed = _pipeline(module)._parse_request(_RequestBatch(_prompt(images=Image.new("RGB", source_size)), sampling))

    assert (parsed.height, parsed.width) == expected_size
    assert parsed.height % 16 == 0
    assert parsed.width % 16 == 0
    assert parsed.height * parsed.width <= 480 * 832


def test_pil_and_tensor_images_share_official_bicubic_preprocess() -> None:
    module = _load_pipeline_module()
    height, width = 7, 11
    source = torch.arange(3 * height * width, dtype=torch.uint8).reshape(3, height, width)
    pil_image = Image.fromarray(source.permute(1, 2, 0).numpy(), mode="RGB")
    sampling = _SamplingParams(height=16, width=32)

    pil_pipeline = _pipeline(module)
    tensor_pipeline = _pipeline(module)
    pil_inputs = pil_pipeline._parse_request(_RequestBatch(_prompt(images=pil_image), sampling))
    tensor_inputs = tensor_pipeline._parse_request(
        _RequestBatch(_prompt(images=source.clone()), _SamplingParams(height=16, width=32))
    )
    pil_condition = pil_pipeline._prepare_image_tensor(pil_inputs.image, height=16, width=32)
    tensor_condition = tensor_pipeline._prepare_image_tensor(tensor_inputs.image, height=16, width=32)
    expected = torch.nn.functional.interpolate(
        source.unsqueeze(0).float() / 255.0,
        size=(16, 32),
        mode="bicubic",
        align_corners=False,
    )
    expected = expected * 2.0 - 1.0

    torch.testing.assert_close(pil_condition, expected)
    torch.testing.assert_close(tensor_condition, expected)


def test_path_image_decode_error_is_sanitized(monkeypatch) -> None:
    module = _load_pipeline_module()
    unsafe_path = "/private/source/customer-secret.png"
    monkeypatch.setattr(
        module.PIL.Image,
        "open",
        lambda path: (_ for _ in ()).throw(module.PIL.Image.DecompressionBombError(f"unsafe {path}")),
    )

    with pytest.raises(ValueError, match="Unable to load multi_modal_data.image") as exc_info:
        _preprocess_request(module, prompt=_prompt(images=unsafe_path))

    assert unsafe_path not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize("source_size", [(4096, 4096), (4097, 4097)])
def test_supplied_pil_image_obeys_documented_source_pixel_ceiling(source_size) -> None:
    module = _load_pipeline_module()
    source_image = Image.new("1", source_size)
    close_calls: list[bool] = []
    source_image.close = lambda: close_calls.append(True)

    if source_size[0] * source_size[1] <= 4096 * 4096:
        request = _preprocess_request(module, prompt=_prompt(images=source_image))
        parsed = _pipeline(module)._parse_request(_RequestBatch(request.prompt, request.sampling_params))
        assert parsed.image is not source_image
        assert parsed.image.mode == "RGB"
        assert close_calls == []
    else:
        with pytest.raises(ValueError, match="source image.*4096.*4096"):
            _preprocess_request(module, prompt=_prompt(images=source_image))
        assert close_calls == []


def test_supplied_pil_decode_error_is_sanitized_without_closing_caller(monkeypatch) -> None:
    module = _load_pipeline_module()
    source_image = Image.new("RGB", (16, 16))
    unsafe_detail = "/private/source/customer-secret.png"
    close_calls: list[bool] = []
    monkeypatch.setattr(source_image, "convert", lambda mode: (_ for _ in ()).throw(OSError(unsafe_detail)))
    monkeypatch.setattr(source_image, "close", lambda: close_calls.append(True))

    with pytest.raises(ValueError, match="Unable to load multi_modal_data.image") as exc_info:
        _preprocess_request(module, prompt=_prompt(images=source_image))

    assert unsafe_detail not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert close_calls == []


def test_action_path_resolves_from_sampling_extra_args(tmp_path: Path) -> None:
    module = _load_pipeline_module()
    action_root = tmp_path / "trusted-actions"
    contained_action = action_root / "actions"
    contained_action.mkdir(parents=True)
    sampling = _SamplingParams(extra_args={"action_path": "actions"})
    prompt = _prompt()
    original_prompt = prompt.copy()
    trajectory = _CameraTrajectory(torch.eye(4).repeat(9, 1, 1), torch.ones(9, 4))
    resolved_actions = []
    module.load_camera_trajectory = lambda action: resolved_actions.append(action) or trajectory

    request = _preprocess_request(
        module,
        prompt=prompt,
        sampling=sampling,
        od_config=_od_config(model_config={"lingbot_action_root": str(action_root)}),
    )
    parsed = _pipeline(module)._parse_request(_RequestBatch(request.prompt, request.sampling_params))

    assert resolved_actions[0].root == action_root.resolve()
    assert resolved_actions[0].relative == Path("actions")
    assert parsed.camera_trajectory is trajectory
    assert sampling.extra_args["action_path"] == "actions"
    assert sampling.extra_args["_lingbot_camera_trajectory"] is trajectory
    assert prompt == original_prompt


@pytest.mark.parametrize("with_extra", [False, True], ids=["additional-only", "extra-and-additional"])
def test_additional_information_action_path_is_not_an_input_source(with_extra: bool) -> None:
    module = _load_pipeline_module()
    sampling = _SamplingParams(
        extra_args={"action_path": "."} if with_extra else None,
        include_action=with_extra,
    )

    with pytest.raises(ValueError, match="additional_information.*action_path.*not supported"):
        _preprocess_request(module, prompt=_prompt(action_path="legacy-actions"), sampling=sampling)


def test_action_path_is_required_in_sampling_extra_args() -> None:
    module = _load_pipeline_module()

    with pytest.raises(ValueError, match="sampling_params.extra_args.action_path"):
        _preprocess_request(module, sampling=_SamplingParams(include_action=False))


@pytest.mark.parametrize("escape_kind", ["traversal", "absolute", "symlink"])
def test_online_action_path_rejects_escape_from_trusted_root(escape_kind: str, tmp_path: Path) -> None:
    module = _load_pipeline_module()
    root = tmp_path / "trusted"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    if escape_kind == "traversal":
        action_path = "../outside"
    elif escape_kind == "absolute":
        action_path = str(outside)
    else:
        (root / "escape-link").symlink_to(outside, target_is_directory=True)
        action_path = "escape-link"
    with pytest.raises(ValueError, match="trusted.*root|contained"):
        _preprocess_request(
            module,
            sampling=_SamplingParams(extra_args={"action_path": action_path}),
            od_config=_od_config(model_config={"lingbot_action_root": str(root)}),
        )


def test_online_action_path_uses_environment_root_fallback(monkeypatch, tmp_path: Path) -> None:
    module = _load_pipeline_module()
    root = tmp_path / "trusted"
    action_dir = root / "forward"
    action_dir.mkdir(parents=True)
    monkeypatch.setenv("VLLM_OMNI_LINGBOT_ACTION_ROOT", str(root))
    resolved_actions = []
    module.load_camera_trajectory = lambda action: (
        resolved_actions.append(action) or _CameraTrajectory(torch.eye(4).repeat(9, 1, 1), torch.ones(9, 4))
    )

    request = _preprocess_request(
        module,
        sampling=_SamplingParams(extra_args={"action_path": "forward"}),
        od_config=_od_config(model_config={}),
    )

    assert request.sampling_params.extra_args["_lingbot_camera_trajectory"] is not None
    assert resolved_actions[0].root == root.resolve()
    assert resolved_actions[0].relative == Path("forward")


def test_online_action_path_error_suppresses_path_bearing_filesystem_cause(tmp_path: Path) -> None:
    module = _load_pipeline_module()
    root = tmp_path / "trusted"
    root.mkdir()

    with pytest.raises(ValueError, match="trusted action root") as exc_info:
        _preprocess_request(
            module,
            sampling=_SamplingParams(extra_args={"action_path": "does-not-exist"}),
            od_config=_od_config(model_config={"lingbot_action_root": str(root)}),
        )

    assert str(root) not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize("source", ["root", "candidate"])
def test_online_action_path_unknown_user_is_sanitized(source: str, tmp_path: Path) -> None:
    module = _load_pipeline_module()
    unknown_user_path = "~__vllm_omni_user_that_does_not_exist__/actions"
    root = tmp_path / "trusted"
    root.mkdir()
    configured_root = unknown_user_path if source == "root" else str(root)
    action_path = "actions" if source == "root" else unknown_user_path

    with pytest.raises(ValueError, match="trusted action root") as exc_info:
        _preprocess_request(
            module,
            sampling=_SamplingParams(extra_args={"action_path": action_path}),
            od_config=_od_config(model_config={"lingbot_action_root": configured_root}),
        )

    assert "__vllm_omni_user_that_does_not_exist__" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_action_path_requires_a_trusted_root_for_every_request() -> None:
    module = _load_pipeline_module()

    with pytest.raises(ValueError, match="lingbot_action_root|VLLM_OMNI_LINGBOT_ACTION_ROOT"):
        _preprocess_request(module, od_config=_od_config(model_config={}))


@pytest.mark.parametrize(
    ("request_batch", "message"),
    [
        (_request(num_reqs=2), "single prompt"),
        (_request(prompt=_prompt(images=[Image.new("RGB", (16, 16))])), "image.*list"),
        (_request(prompt=_prompt(images=[])), "image.*list"),
        (
            _request(prompt=_prompt(images=[Image.new("RGB", (16, 16)), Image.new("RGB", (16, 16))])),
            "image.*list",
        ),
        (_request(sampling=_SamplingParams(num_outputs_per_prompt=2)), "num_outputs_per_prompt"),
        (_request(sampling=_SamplingParams(num_inference_steps=5)), "num_inference_steps"),
        (_request(sampling=_SamplingParams(height=15)), "height.*divisible"),
        (_request(sampling=_SamplingParams(width=15)), "width.*divisible"),
        (_request(sampling=_SamplingParams(height=None, width=16)), "height and width.*both"),
        (_request(sampling=_SamplingParams(num_frames=13)), "num_frames.*three-frame"),
    ],
)
def test_request_validation_rejects_unsupported_contracts(request_batch, message: str) -> None:
    module = _load_pipeline_module()

    with pytest.raises(ValueError, match=message):
        _pipeline(module)._parse_request(request_batch)


def test_resource_limits_accept_exact_documented_boundaries() -> None:
    module = _load_pipeline_module()
    sampling = _SamplingParams(height=480, width=832, num_frames=117, max_sequence_length=512)

    parsed = _pipeline(module)._parse_request(_RequestBatch(_prompt(), sampling))

    assert (parsed.height, parsed.width, parsed.num_frames, parsed.max_sequence_length) == (480, 832, 117, 512)


@pytest.mark.parametrize(
    ("sampling", "message"),
    [
        (_SamplingParams(height=480, width=848), "pixel area|480.*832"),
        (_SamplingParams(num_frames=129), "num_frames.*117"),
        (_SamplingParams(max_sequence_length=511), "max_sequence_length.*512"),
        (_SamplingParams(max_sequence_length=513), "max_sequence_length.*512"),
        (_SamplingParams(max_sequence_length=512.0), "max_sequence_length.*512"),
    ],
)
def test_resource_limits_reject_oversize_before_any_component_call(sampling, message: str) -> None:
    module = _load_pipeline_module()
    pipeline = _pipeline(module)
    calls: list[str] = []
    pipeline.encode_prompt = lambda *args, **kwargs: calls.append("text")
    pipeline._prepare_condition = lambda *args, **kwargs: calls.append("vae")
    pipeline._prepare_camera = lambda *args, **kwargs: calls.append("camera")
    pipeline.transformer.allocate_cache = lambda *args, **kwargs: calls.append("cache")

    with pytest.raises(ValueError, match=message):
        pipeline(_RequestBatch(_prompt(), sampling))

    assert calls == []


def test_request_validation_rejects_insufficient_camera_frames() -> None:
    module = _load_pipeline_module()
    pipeline = _pipeline(module)
    short_trajectory = _CameraTrajectory(
        poses=torch.eye(4).repeat(8, 1, 1),
        intrinsics=torch.ones(8, 4),
    )
    sampling = _SamplingParams(extra_args={"_lingbot_camera_trajectory": short_trajectory})

    with pytest.raises(ValueError, match="camera.*frames.*num_frames"):
        pipeline(_request(sampling=sampling))


def test_camera_load_error_is_actionable_without_echoing_path_contents() -> None:
    module = _load_pipeline_module()
    module.load_camera_trajectory = lambda path: (_ for _ in ()).throw(FileNotFoundError(f"missing {path}/poses.npy"))

    with pytest.raises(ValueError, match="camera trajectory.*action_path") as exc_info:
        _preprocess_request(module)

    assert str(_ROOT) not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_first_frame_condition_and_camera_fold_match_transformer_contract() -> None:
    module = _load_pipeline_module()
    transformer = _RecordingTransformer()
    pipeline = _pipeline(module, transformer=transformer)
    module.randn_tensor = lambda shape, **kwargs: torch.full(
        shape,
        -99.0,
        device=kwargs["device"],
        dtype=kwargs["dtype"],
    )

    result = pipeline(_request())

    assert result.output.shape == (1, 16, 3, 2, 2)
    first_input = transformer.calls[0]["hidden_states"]
    assert first_input.shape == (1, 36, 3, 2, 2)
    mean = torch.tensor(pipeline.vae.config.latents_mean).view(1, 16, 1, 1)
    std = torch.tensor(pipeline.vae.config.latents_std).view(1, 16, 1, 1)
    expected_first = (torch.full((1, 16, 2, 2), 2.0) - mean) / std
    expected_future = (torch.zeros(1, 16, 2, 2, 2) - mean.unsqueeze(2)) / std.unsqueeze(2)
    torch.testing.assert_close(first_input[:, :16], torch.full((1, 16, 3, 2, 2), -99.0))
    torch.testing.assert_close(first_input[:, 16:20, 0], torch.ones(1, 4, 2, 2))
    torch.testing.assert_close(first_input[:, 16:20, 1:], torch.zeros(1, 4, 2, 2, 2))
    torch.testing.assert_close(first_input[:, 20:36, 0], expected_first)
    torch.testing.assert_close(first_input[:, 20:36, 1:], expected_future)

    raw_camera = module.build_plucker_embedding(
        SimpleNamespace(poses=torch.empty(3, 4, 4)),
        height=16,
        width=16,
        target_height=16,
        target_width=16,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected_camera = module._fold_camera_embedding(raw_camera)
    reference_camera = torch.nn.functional.pixel_unshuffle(raw_camera, 8).permute(1, 0, 2, 3).unsqueeze(0)
    torch.testing.assert_close(expected_camera, reference_camera)
    torch.testing.assert_close(transformer.calls[0]["camera_hidden_states"], expected_camera)
    assert expected_camera.shape == (1, 384, 3, 2, 2)


def test_fixed_dmd_transition_and_cache_commit_trace() -> None:
    module = _load_pipeline_module()
    transformer = _RecordingTransformer()
    pipeline = _pipeline(module, transformer=transformer)

    result = pipeline(_request())

    generator = torch.Generator(device="cpu").manual_seed(17)
    current = torch.randn((1, 16, 3, 2, 2), generator=generator)
    warped_schedule = (
        (1000.0, 1.0),
        (937.5, 0.9375),
        (2500.0 / 3.0, 5.0 / 6.0),
        (625.0, 0.625),
    )
    for index, (_, sigma) in enumerate(warped_schedule):
        x0 = current - sigma
        if index + 1 < len(module.LINGBOT_DMD_TIMESTEPS):
            next_sigma = warped_schedule[index + 1][1]
            noise = torch.randn(current.shape, generator=generator)
            current = (1.0 - next_sigma) * x0 + next_sigma * noise
        else:
            current = x0
    torch.testing.assert_close(result.output, current)

    torch.testing.assert_close(
        torch.cat([call["timestep"] for call in transformer.calls]),
        torch.tensor([*(timestep for timestep, _ in warped_schedule), 0.0]),
    )
    assert [call["update_cache"] for call in transformer.calls] == [False, False, False, False, True]
    assert [call["start_frame"] for call in transformer.calls] == [0, 0, 0, 0, 0]
    assert len({call["cache_id"] for call in transformer.calls}) == 1
    torch.testing.assert_close(transformer.calls[-1]["hidden_states"][:, :16], result.output)


@pytest.mark.parametrize(
    ("flow_shift", "expected_timesteps"),
    [
        (5.0, (1000.0, 937.5, 2500.0 / 3.0, 625.0)),
        (2.5, (1000.0, 15000.0 / 17.0, 5000.0 / 7.0, 5000.0 / 11.0)),
    ],
)
def test_request_flow_shift_warps_transformer_timesteps(flow_shift: float, expected_timesteps) -> None:
    module = _load_pipeline_module()
    transformer = _RecordingTransformer()
    pipeline = _pipeline(module, transformer=transformer)

    pipeline(
        _request(
            sampling=_SamplingParams(extra_args={"action_path": ".", "flow_shift": flow_shift}),
        )
    )

    torch.testing.assert_close(
        torch.cat([call["timestep"] for call in transformer.calls[:4]]),
        torch.tensor(expected_timesteps),
    )


def test_request_tiny_positive_flow_shift_keeps_schedule_finite() -> None:
    module = _load_pipeline_module()
    transformer = _RecordingTransformer()
    pipeline = _pipeline(module, transformer=transformer)

    pipeline(
        _request(
            sampling=_SamplingParams(extra_args={"action_path": ".", "flow_shift": 1e-20}),
        )
    )

    timesteps = torch.cat([call["timestep"] for call in transformer.calls[:4]])
    assert torch.isfinite(timesteps).all()
    assert timesteps[0].item() == 1000.0


def test_request_flow_shift_override_has_precedence_without_mutating_scheduler() -> None:
    module = _load_pipeline_module()
    pipeline = _pipeline(module, od_config=_od_config(flow_shift=3.0))

    default_inputs = pipeline._parse_request(_request())
    sampling = _SamplingParams(extra_args={"flow_shift": 2.0})
    override_inputs = pipeline._parse_request(_RequestBatch(_prompt(), sampling))

    assert default_inputs.flow_shift == 3.0
    assert override_inputs.flow_shift == 2.0
    assert sampling.extra_args["action_path"] == "."
    assert sampling.extra_args["flow_shift"] == 2.0
    assert pipeline.scheduler.config.shift == 3.0


def test_checkpoint_flow_shift_default_and_engine_request_override_precedence() -> None:
    module = _load_pipeline_module()
    checkpoint_scheduler_config = _scheduler_config(flow_shift=6.25, shift=7.5)
    module._load_json = lambda *args, **kwargs: checkpoint_scheduler_config.copy()

    checkpoint_default = _pipeline(module)
    engine_override = _pipeline(module, od_config=_od_config(flow_shift=3.5))
    request_override = checkpoint_default._parse_request(
        _RequestBatch(
            _prompt(),
            _SamplingParams(extra_args={"flow_shift": 2.25}),
        )
    )

    assert checkpoint_default.scheduler.config.shift == 6.25
    assert checkpoint_default._parse_request(_request()).flow_shift == 6.25
    assert engine_override.scheduler.config.shift == 3.5
    assert engine_override._parse_request(_request()).flow_shift == 3.5
    assert request_override.flow_shift == 2.25


@pytest.mark.parametrize(
    ("scheduler_config", "expected"),
    [
        ({"flow_shift": 6.0, "shift": 7.0}, 6.0),
        ({"shift": 7.0}, 7.0),
        ({}, 5.0),
    ],
)
def test_checkpoint_scheduler_shift_fallback_order(scheduler_config, expected: float) -> None:
    module = _load_pipeline_module()
    checkpoint_config = _scheduler_config()
    checkpoint_config.pop("flow_shift")
    checkpoint_config.update(scheduler_config)
    module._load_json = lambda *args, **kwargs: checkpoint_config.copy()

    pipeline = _pipeline(module)

    assert pipeline.scheduler.config.shift == expected


@pytest.mark.parametrize("flow_shift", [0, -1, float("nan"), float("inf"), "invalid"])
def test_request_flow_shift_must_be_positive_and_finite(flow_shift) -> None:
    module = _load_pipeline_module()
    sampling = _SamplingParams(extra_args={"flow_shift": flow_shift})

    with pytest.raises(ValueError, match="flow_shift.*positive.*finite"):
        _pipeline(module)._parse_request(_RequestBatch(_prompt(), sampling))


@pytest.mark.parametrize(
    "flow_shift",
    [True, False, 10**10000],
    ids=["true", "false", "overflowing-integer"],
)
def test_request_flow_shift_rejects_booleans_and_normalizes_integer_overflow(flow_shift) -> None:
    module = _load_pipeline_module()
    sampling = _SamplingParams(extra_args={"flow_shift": flow_shift})

    with pytest.raises(ValueError, match="flow_shift.*positive.*finite"):
        _pipeline(module)._parse_request(_RequestBatch(_prompt(), sampling))


def test_flow_shift_is_request_local_without_mutating_scheduler() -> None:
    module = _load_pipeline_module()
    pipeline = _pipeline(module)

    def generate(flow_shift=None):
        extra_args = {} if flow_shift is None else {"flow_shift": flow_shift}
        return pipeline(
            _RequestBatch(
                _prompt(),
                _SamplingParams(num_frames=21, extra_args=extra_args),
            )
        ).output

    default_before = generate()
    shifted_two = generate(2.0)
    shifted_seven = generate(7.0)
    default_after = generate()

    torch.testing.assert_close(default_before, default_after)
    assert not torch.equal(default_before, shifted_two)
    assert not torch.equal(shifted_two, shifted_seven)
    assert pipeline.scheduler.config.shift == 5.0


def test_encode_prompt_zeroes_padded_umt5_states_to_exactly_512_tokens() -> None:
    module = _load_pipeline_module()
    pipeline = _pipeline(module)
    attention_mask = torch.zeros(1, 512, dtype=torch.long)
    attention_mask[:, :3] = 1
    pipeline.tokenizer = lambda *args, **kwargs: SimpleNamespace(
        input_ids=torch.arange(512).view(1, 512),
        attention_mask=attention_mask,
    )
    raw_states = torch.arange(512 * 8, dtype=torch.float32).view(1, 512, 8) + 1.0
    pipeline.text_encoder = lambda input_ids, mask: SimpleNamespace(last_hidden_state=raw_states.clone())

    encoded = module.LingBotWorldCausalDMDPipeline.encode_prompt(
        pipeline,
        "move",
        max_sequence_length=512,
        dtype=torch.float32,
    )

    assert encoded.shape == (1, 512, 8)
    torch.testing.assert_close(encoded[:, :3], raw_states[:, :3])
    torch.testing.assert_close(encoded[:, 3:], torch.zeros_like(encoded[:, 3:]))


def test_multi_chunk_generation_uses_one_request_local_cache_and_decodes_accumulated_latents() -> None:
    module = _load_pipeline_module()
    transformer = _RecordingTransformer()
    pipeline = _pipeline(module, transformer=transformer)
    allocations = []
    original_allocate = transformer.allocate_cache

    def allocate(**kwargs):
        cache = original_allocate(**kwargs)
        allocations.append((kwargs, weakref.ref(cache)))
        return cache

    transformer.allocate_cache = allocate
    sampling = _SamplingParams(num_frames=21, output_type="np")

    result = pipeline(_request(sampling=sampling))

    assert result.output.shape == (1, 3, 21, 16, 16)
    assert len(transformer.calls) == 10
    assert [call["start_frame"] for call in transformer.calls] == [0] * 5 + [3] * 5
    expected_timesteps = torch.tensor([1000.0, 937.5, 2500.0 / 3.0, 625.0, 0.0] * 2)
    torch.testing.assert_close(
        torch.cat([call["timestep"] for call in transformer.calls]),
        expected_timesteps,
    )
    assert len(allocations) == 1
    kwargs, cache_ref = allocations[0]
    assert kwargs == {
        "batch_size": 1,
        "latent_height": 2,
        "latent_width": 2,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }
    assert not hasattr(pipeline, "cache")
    assert not hasattr(pipeline, "transformer_cache")
    assert pipeline.vae.decode_inputs[0].shape == (1, 16, 6, 2, 2)
    normalized_latents = torch.cat(
        (transformer.calls[4]["hidden_states"][:, :16], transformer.calls[9]["hidden_states"][:, :16]),
        dim=2,
    )
    mean = torch.tensor(pipeline.vae.config.latents_mean).view(1, 16, 1, 1, 1)
    std = torch.tensor(pipeline.vae.config.latents_std).view(1, 16, 1, 1, 1)
    torch.testing.assert_close(pipeline.vae.decode_inputs[0], normalized_latents * std + mean)
    transformer.calls.clear()
    gc.collect()
    assert cache_ref() is None


def _tick_extra_args(*, chunk_index: int, prompt: str = "move through the room"):
    return ARDiffusionTickRequest(
        session_id="world-1",
        request_id="legacy-lingbot-request",
        chunk_index=chunk_index,
        applied_event_ids=(chunk_index,),
        prompt=prompt,
        controls=(
            ARDiffusionControlInput(
                track="camera",
                schema="lingbot.camera_trajectory.v1",
                data={"poses": [], "intrinsics": []},
            ),
        ),
    ).to_extra_args()


def test_typed_ticks_generate_one_global_block_and_return_standard_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_pipeline_module()
    pipeline = _pipeline(module)
    pipeline._ar_height = 16
    pipeline._ar_width = 16
    pipeline._ar_diffusion_kv_state = object()
    cross_calls = []
    generated = []

    def ar_text_caches(prompt_embeds, *, invalidate):
        del prompt_embeds
        cross_calls.append(invalidate)
        return [SimpleNamespace()]

    def generate_block(**kwargs):
        block = torch.randn(
            (1, 16, 3, 2, 2),
            generator=kwargs["generator"],
        )
        generated.append(
            {
                "start_frame": kwargs["start_frame"],
                "condition": kwargs["condition"].clone(),
                "block": block.clone(),
            }
        )
        return block

    monkeypatch.setattr(pipeline, "_ar_text_caches", ar_text_caches)
    monkeypatch.setattr(pipeline, "_generate_block", generate_block)

    first_sampling = _SamplingParams(extra_args=_tick_extra_args(chunk_index=0))
    first = pipeline(_request(sampling=first_sampling))
    switched_prompt = _prompt()
    switched_prompt["prompt"] = "enter the snowy valley"
    second_sampling = _SamplingParams(
        extra_args=_tick_extra_args(
            chunk_index=1,
            prompt=switched_prompt["prompt"],
        )
    )
    second = pipeline(_request(sampling=second_sampling, prompt=switched_prompt))

    assert generated[0]["start_frame"] == 0
    assert generated[1]["start_frame"] == 3
    assert torch.count_nonzero(generated[0]["condition"]) > 0
    torch.testing.assert_close(
        generated[1]["condition"][:, :4],
        torch.zeros_like(generated[1]["condition"][:, :4]),
    )
    assert torch.count_nonzero(generated[1]["condition"][:, 4:]) > 0
    assert cross_calls == [False, True]
    assert first.output["payload"]["latents"].shape == (1, 16, 3, 2, 2)
    assert second.output["metadata"]["ar_diffusion"] == {
        "session_id": "world-1",
        "request_id": "legacy-lingbot-request",
        "chunk_index": 1,
        "applied_event_ids": [1],
    }
    assert pipeline._ar_sessions["world-1"].next_chunk_index == 2
    assert pipeline._ar_sessions["world-1"].prompt == "enter the snowy valley"
    assert pipeline._ar_sessions["world-1"].camera_tail is not None
    assert pipeline._ar_sessions["world-1"].camera_tail.poses.shape == (1, 4, 4)
    expected_generator = torch.Generator(device="cpu").manual_seed(17)
    expected_first = torch.randn((1, 16, 3, 2, 2), generator=expected_generator)
    expected_second = torch.randn((1, 16, 3, 2, 2), generator=expected_generator)
    torch.testing.assert_close(generated[0]["block"], expected_first)
    torch.testing.assert_close(generated[1]["block"], expected_second)


def test_typed_action_ticks_integrate_camera_across_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_pipeline_module()
    pipeline = _pipeline(module)
    pipeline._ar_height = 16
    pipeline._ar_width = 16
    pipeline._ar_diffusion_kv_state = object()
    monkeypatch.setattr(
        pipeline,
        "_ar_text_caches",
        lambda *args, **kwargs: [SimpleNamespace()],
    )
    monkeypatch.setattr(
        pipeline,
        "_generate_block",
        lambda **kwargs: torch.zeros_like(kwargs["condition"][:, :16]),
    )

    def action_sampling(chunk_index: int, frames: list[list[str]]) -> _SamplingParams:
        tick = ARDiffusionTickRequest(
            session_id="world-actions",
            request_id="legacy-lingbot-request",
            chunk_index=chunk_index,
            controls=(
                ARDiffusionControlInput(
                    track="camera",
                    schema="lingbot.camera_actions.v1",
                    data={"mode": "frames", "frames": frames},
                ),
            ),
        )
        sampling = _SamplingParams(extra_args=tick.to_extra_args())
        sampling.extra_args["_lingbot_camera_trajectory"] = None
        sampling.extra_args["_lingbot_camera_actions"] = tuple(tuple(frame) for frame in frames)
        return sampling

    pipeline(_request(sampling=action_sampling(0, [["w"], ["w"], ["w"]])))
    first_tail = pipeline._ar_sessions["world-actions"].camera_tail.poses.clone()
    pipeline(_request(sampling=action_sampling(1, [["d"], ["d"], ["d"]])))
    state = pipeline._ar_sessions["world-actions"]

    assert state.next_chunk_index == 2
    assert state.camera_tail is not None
    torch.testing.assert_close(state.camera_tail.poses[0, 2, 3], first_tail[0, 2, 3])
    assert state.camera_tail.poses[0, 0, 3] > first_tail[0, 0, 3]


def test_first_typed_yaw_action_uses_pre_action_identity_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_pipeline_module()
    pipeline = _pipeline(module)
    monkeypatch.setattr(
        module,
        "build_plucker_embedding",
        _real_build_plucker_embedding,
    )
    action_trajectory, _ = integrate_lingbot_camera_actions(
        [["j"], [], []],
        width=16,
        height=16,
    )
    neutral_trajectory, _ = integrate_lingbot_camera_actions(
        [[], [], []],
        width=16,
        height=16,
    )

    def inputs(trajectory, actions):
        return SimpleNamespace(
            camera_trajectory=trajectory,
            camera_actions=actions,
            num_latent_frames=3,
            num_frames=9,
            height=16,
            width=16,
        )

    action_embedding, tail = pipeline._prepare_camera(
        inputs(action_trajectory, (("j",), (), ())),
        dtype=torch.float32,
    )
    neutral_embedding, _ = pipeline._prepare_camera(
        inputs(neutral_trajectory, ((), (), ())),
        dtype=torch.float32,
    )

    identity_anchor = torch.eye(4, dtype=action_trajectory.poses.dtype).unsqueeze(0)
    explicit_trajectory = _CameraTrajectory(
        poses=torch.cat((identity_anchor, action_trajectory.poses), dim=0),
        intrinsics=torch.cat(
            (action_trajectory.intrinsics[:1], action_trajectory.intrinsics),
            dim=0,
        ),
    )
    explicit = _real_build_plucker_embedding(
        explicit_trajectory,
        height=16,
        width=16,
        target_height=16,
        target_width=16,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )[1:]

    assert not torch.equal(action_embedding, neutral_embedding)
    torch.testing.assert_close(action_embedding, module._fold_camera_embedding(explicit))
    torch.testing.assert_close(tail.poses, action_trajectory.poses[-1:])


def test_typed_tick_rejects_non_contiguous_chunk_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_pipeline_module()
    pipeline = _pipeline(module)
    pipeline._ar_height = 16
    pipeline._ar_width = 16
    pipeline._ar_diffusion_kv_state = object()
    monkeypatch.setattr(
        pipeline,
        "_ar_text_caches",
        lambda *args, **kwargs: [SimpleNamespace()],
    )

    sampling = _SamplingParams(extra_args=_tick_extra_args(chunk_index=2))
    with pytest.raises(ValueError, match="must be contiguous"):
        pipeline(_request(sampling=sampling))


def test_typed_tick_rejects_chunk_beyond_realtime_condition_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_pipeline_module()
    pipeline = _pipeline(module)
    pipeline._ar_height = 16
    pipeline._ar_width = 16
    pipeline._ar_diffusion_kv_state = object()
    pipeline._ar_sessions["world-1"] = module._LingBotARSessionState(
        next_chunk_index=10,
    )
    monkeypatch.setattr(
        pipeline,
        "encode_prompt",
        lambda *args, **kwargs: pytest.fail("the realtime horizon must be validated before prompt encoding"),
    )

    sampling = _SamplingParams(extra_args=_tick_extra_args(chunk_index=10))
    with pytest.raises(
        ValueError,
        match=r"at most 10 ticks.*chunk_index 0 through 9.*117 pixel frames",
    ):
        pipeline(_request(sampling=sampling))


def test_request_cache_is_released_before_vae_decode() -> None:
    module = _load_pipeline_module()
    transformer = _RecordingTransformer()
    pipeline = _pipeline(module, transformer=transformer)
    cache_refs: list[weakref.ReferenceType] = []
    original_allocate = transformer.allocate_cache

    def allocate(**kwargs):
        cache = original_allocate(**kwargs)
        cache_refs.append(weakref.ref(cache))
        return cache

    transformer.allocate_cache = allocate

    def assert_cache_released() -> None:
        gc.collect()
        assert len(cache_refs) == 1
        assert cache_refs[0]() is None

    pipeline.vae.on_decode = assert_cache_released

    result = pipeline(_request(sampling=_SamplingParams(num_frames=21, output_type="np")))

    assert result.output.shape[2] == 21


def test_117_frame_request_generates_ten_complete_latent_blocks() -> None:
    module = _load_pipeline_module()
    transformer = _RecordingTransformer()
    pipeline = _pipeline(module, transformer=transformer)
    trajectory = _CameraTrajectory(
        poses=torch.eye(4).repeat(117, 1, 1),
        intrinsics=torch.tensor([[100.0, 100.0, 8.0, 8.0]]).repeat(117, 1),
    )
    sampling = _SamplingParams(
        num_frames=117,
        output_type="latent",
        extra_args={"_lingbot_camera_trajectory": trajectory},
    )

    result = pipeline(_request(sampling=sampling))

    assert result.output.shape == (1, 16, 30, 2, 2)
    assert len(transformer.calls) == 50
    assert [call["start_frame"] for call in transformer.calls[::5]] == list(range(0, 30, 3))


def test_request_cache_becomes_unreachable_after_transformer_error() -> None:
    module = _load_pipeline_module()
    transformer = _RecordingTransformer(raise_on_call=2)
    pipeline = _pipeline(module, transformer=transformer)
    cache_refs = []
    original_allocate = transformer.allocate_cache

    def allocate(**kwargs):
        cache = original_allocate(**kwargs)
        cache_refs.append(weakref.ref(cache))
        return cache

    transformer.allocate_cache = allocate

    with pytest.raises(RuntimeError, match="forced transformer failure") as exc_info:
        pipeline(_request())

    exc_info.value.__traceback__ = None
    del exc_info
    transformer.calls.clear()
    gc.collect()
    assert len(cache_refs) == 1
    assert cache_refs[0]() is None
    assert not hasattr(pipeline, "cache")
    assert not hasattr(pipeline, "transformer_cache")


def test_registry_and_model_exports_resolve_official_pipeline_class_name() -> None:
    module = _load_pipeline_module()
    resolved, entry, cache_acceleration_disabled, preprocess_name, preprocess = _resolve_pipeline_through_real_registry(
        module
    )

    assert entry == ("lingbot_world", "pipeline", "LingBotWorldCausalDMDPipeline")
    assert resolved is module.LingBotWorldCausalDMDPipeline
    assert cache_acceleration_disabled
    assert preprocess_name == "get_lingbot_world_pre_process_func"
    request = SimpleNamespace(prompt=_prompt(), sampling_params=_SamplingParams())
    assert preprocess(request) is request
    assert isinstance(request.sampling_params.extra_args["_lingbot_camera_trajectory"], _CameraTrajectory)
    assert module.LingBotWorldCausalDMDPipeline.__name__ == "LingBotWorldCausalDMDPipeline"
    lingbot_init = _LINGBOT_INIT_PATH.read_text()
    assert "from .pipeline import" in lingbot_init
    assert '"LingBotWorldCausalDMDPipeline"' in lingbot_init
    assert '"CausalLingBotWorldTransformer3DModel"' in lingbot_init
