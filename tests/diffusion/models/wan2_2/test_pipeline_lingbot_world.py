# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import gc
import importlib.util
import sys
import types
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_ROOT = Path(__file__).parents[4]
_MODULE_PATH = _ROOT / "vllm_omni/diffusion/models/wan2_2/pipeline_lingbot_world.py"
_REGISTRY_PATH = _ROOT / "vllm_omni/diffusion/registry.py"
_WAN_INIT_PATH = _ROOT / "vllm_omni/diffusion/models/wan2_2/__init__.py"
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


@dataclass
class _DiffusionOutput:
    output: torch.Tensor
    error: str | None = None
    finished: bool = True
    stage_durations: dict | None = None


@dataclass
class _ComponentSource:
    model_or_path: str
    subfolder: str
    revision: str | None
    prefix: str
    fall_back_to_pt: bool


class _SupportImageInput:
    pass


class _SupportsComponentDiscovery:
    pass


class _ProgressBarMixin:
    @contextmanager
    def progress_bar(self, total: int):
        del total

        class _Bar:
            def update(self) -> None:
                return None

        yield _Bar()


class _DiffusionPipelineProfilerMixin:
    def setup_diffusion_pipeline_profiler(self, *, profiler_targets=None, enable_diffusion_pipeline_profiler=False):
        self.profiler_setup = (tuple(profiler_targets or ()), enable_diffusion_pipeline_profiler)


class _AutoWeightsLoader:
    def __init__(self, module):
        self.module = module

    def load_weights(self, weights):
        return self.module.transformer.load_weights(
            (name.removeprefix("transformer."), value) for name, value in weights if name.startswith("transformer.")
        )


class _FakePretrained:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        del args, kwargs
        return cls()

    def to(self, *args, **kwargs):
        del args, kwargs
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


def _make_package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _load_pipeline_module():
    assert _MODULE_PATH.is_file(), f"Task 4 pipeline module is missing: {_MODULE_PATH}"

    stub_modules: dict[str, types.ModuleType] = {}
    for package in (
        "diffusers",
        "diffusers.utils",
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.models",
        "vllm_omni",
        "vllm_omni.diffusion",
        "vllm_omni.diffusion.distributed",
        "vllm_omni.diffusion.distributed.autoencoders",
        "vllm_omni.diffusion.model_loader",
        "vllm_omni.diffusion.models",
        "vllm_omni.diffusion.models.wan2_2",
        "vllm_omni.diffusion.profiler",
        "vllm_omni.diffusion.worker",
    ):
        stub_modules[package] = _make_package(package)

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

    trajectory = SimpleNamespace(
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

    class CameraTrajectory:
        def __init__(self, poses, intrinsics):
            self.poses = poses
            self.intrinsics = intrinsics

    @dataclass(frozen=True)
    class TrustedActionDirectory:
        root: Path
        relative: Path
        root_device: int
        root_inode: int

    def resolve_trusted_action_directory(action_path, trusted_root):
        try:
            root = Path(trusted_root).expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            raise ValueError("The configured LingBot trusted action root is unavailable.") from None
        try:
            candidate = Path(action_path).expanduser()
            if not candidate.is_absolute():
                candidate = root / candidate
            candidate = candidate.resolve(strict=True)
            relative = candidate.relative_to(root)
        except (OSError, RuntimeError, ValueError):
            raise ValueError("action_path must be contained by the trusted action root.") from None
        if not candidate.is_dir():
            raise ValueError("action_path must identify a directory in the trusted action root.")
        stat_result = root.stat()
        return TrustedActionDirectory(root, relative, stat_result.st_dev, stat_result.st_ino)

    def allocate_lingbot_cache(**kwargs):
        return _Cache(
            num_layers=kwargs["num_layers"],
            max_tokens=kwargs["max_tokens"],
            num_local_heads=kwargs["num_local_heads"],
            head_dim=kwargs["head_dim"],
        )

    def default_randn_tensor(shape, *, generator, device, dtype):
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)

    stub_modules.update(
        {
            "diffusers.utils.torch_utils": _module(
                "diffusers.utils.torch_utils",
                randn_tensor=default_randn_tensor,
            ),
            "vllm.model_executor.models.utils": _module(
                "vllm.model_executor.models.utils",
                AutoWeightsLoader=_AutoWeightsLoader,
            ),
            "vllm_omni.diffusion.data": _module(
                "vllm_omni.diffusion.data",
                DiffusionOutput=_DiffusionOutput,
                OmniDiffusionConfig=object,
            ),
            "vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan": _module(
                "vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan",
                DistributedAutoencoderKLWan=_StubVAE,
            ),
            "vllm_omni.diffusion.distributed.utils": _module(
                "vllm_omni.diffusion.distributed.utils",
                get_local_device=lambda: torch.device("cpu"),
            ),
            "vllm_omni.diffusion.forward_context": _module(
                "vllm_omni.diffusion.forward_context",
                set_forward_context_denoise_step_idx=lambda index: None,
            ),
            "vllm_omni.diffusion.model_loader.diffusers_loader": _module(
                "vllm_omni.diffusion.model_loader.diffusers_loader",
                DiffusersPipelineLoader=SimpleNamespace(ComponentSource=_ComponentSource),
            ),
            "vllm_omni.diffusion.model_loader.hub_prefetch": _module(
                "vllm_omni.diffusion.model_loader.hub_prefetch",
                from_pretrained_with_prefetch=from_pretrained_with_prefetch,
                prefetch_subfolders=prefetch_subfolders,
            ),
            "vllm_omni.diffusion.models.interface": _module(
                "vllm_omni.diffusion.models.interface",
                SupportImageInput=_SupportImageInput,
                SupportsComponentDiscovery=_SupportsComponentDiscovery,
            ),
            "vllm_omni.diffusion.models.progress_bar": _module(
                "vllm_omni.diffusion.models.progress_bar",
                ProgressBarMixin=_ProgressBarMixin,
            ),
            "vllm_omni.diffusion.profiler.diffusion_pipeline_profiler": _module(
                "vllm_omni.diffusion.profiler.diffusion_pipeline_profiler",
                DiffusionPipelineProfilerMixin=_DiffusionPipelineProfilerMixin,
            ),
            "vllm_omni.diffusion.models.schedulers": _module(
                "vllm_omni.diffusion.models.schedulers",
                FlowUniPCMultistepScheduler=_FakeScheduler,
            ),
            "vllm_omni.diffusion.models.utils": _module(
                "vllm_omni.diffusion.models.utils",
                _load_json=load_json,
            ),
            "vllm_omni.diffusion.models.wan2_2.lingbot_world_camera": _module(
                "vllm_omni.diffusion.models.wan2_2.lingbot_world_camera",
                CameraTrajectory=CameraTrajectory,
                TrustedActionDirectory=TrustedActionDirectory,
                build_plucker_embedding=build_plucker_embedding,
                interpolate_camera_trajectory=interpolate_camera_trajectory,
                load_camera_trajectory=load_camera_trajectory,
                resolve_trusted_action_directory=resolve_trusted_action_directory,
            ),
            "vllm_omni.diffusion.models.wan2_2.lingbot_world_transformer": _module(
                "vllm_omni.diffusion.models.wan2_2.lingbot_world_transformer",
                CausalLingBotWorldTransformer3DModel=_FakeTransformerFactory,
                allocate_lingbot_cache=allocate_lingbot_cache,
            ),
            "vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2": _module(
                "vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2",
                load_transformer_config=load_transformer_config,
                retrieve_latents=retrieve_latents,
            ),
            "vllm_omni.diffusion.worker.request_batch": _module(
                "vllm_omni.diffusion.worker.request_batch",
                DiffusionRequestBatch=_RequestBatch,
            ),
            "transformers": _module(
                "transformers",
                AutoTokenizer=_FakePretrained,
                UMT5EncoderModel=_FakePretrained,
            ),
        }
    )

    previous = {name: sys.modules.get(name) for name in stub_modules}
    sys.modules.update(stub_modules)
    spec = importlib.util.spec_from_file_location("_lingbot_world_pipeline_under_test", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module
    module._loader_state = loader_state
    return module


def _od_config(**overrides):
    values = {
        "model": "checkpoint",
        "dtype": torch.float32,
        "flow_shift": None,
        "quantization_config": None,
        "enable_diffusion_pipeline_profiler": False,
        "model_config": {"lingbot_action_root": str(_ROOT)},
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
    return pipeline


def test_pipeline_registers_lingbot_profiler_targets() -> None:
    module = _load_pipeline_module()

    pipeline = _pipeline(module, od_config=_od_config(enable_diffusion_pipeline_profiler=True))

    assert isinstance(pipeline, _DiffusionPipelineProfilerMixin)
    assert pipeline.profiler_setup == (
        ("vae.encode", "vae.decode", "_generate_block", "text_encoder.forward", "tokenizer.forward"),
        True,
    )


def _request(*, sampling=None, prompt=None, num_reqs: int = 1):
    return _RequestBatch(
        _prompt() if prompt is None else prompt,
        _SamplingParams() if sampling is None else sampling,
        num_reqs=num_reqs,
    )


def _resolve_pipeline_through_real_registry(pipeline_module):
    @dataclass(frozen=True)
    class LazyRegisteredModel:
        module_name: str
        class_name: str

    class ModelRegistry:
        def __init__(self, models):
            self.models = models

        def _try_load_model_cls(self, architecture):
            registered = self.models.get(architecture)
            if registered is None:
                return None
            module = importlib.import_module(registered.module_name)
            return getattr(module, registered.class_name)

    stub_modules: dict[str, types.ModuleType] = {}
    for package in (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.model_loader",
        "vllm.model_executor.models",
        "vllm_omni",
        "vllm_omni.diffusion",
        "vllm_omni.diffusion.distributed",
        "vllm_omni.diffusion.distributed.autoencoders",
        "vllm_omni.diffusion.hooks",
        "vllm_omni.diffusion.utils",
        "vllm_omni.diffusion.models",
        "vllm_omni.diffusion.models.wan2_2",
    ):
        stub_modules[package] = _make_package(package)

    @contextmanager
    def no_op_context(*args, **kwargs):
        del args, kwargs
        yield

    stub_modules.update(
        {
            "vllm.logger": _module("vllm.logger", init_logger=lambda name: SimpleNamespace()),
            "vllm.model_executor.model_loader.utils": _module(
                "vllm.model_executor.model_loader.utils", configure_quant_config=lambda *args: None
            ),
            "vllm.model_executor.models.registry": _module(
                "vllm.model_executor.models.registry",
                _LazyRegisteredModel=LazyRegisteredModel,
                _ModelRegistry=ModelRegistry,
            ),
            "vllm_omni.diffusion.config": _module(
                "vllm_omni.diffusion.config", set_current_diffusion_config=no_op_context
            ),
            "vllm_omni.diffusion.data": _module("vllm_omni.diffusion.data", OmniDiffusionConfig=object),
            "vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor": _module(
                "vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor",
                DistributedVaeMixin=object,
            ),
            "vllm_omni.diffusion.distributed.sp_plan": _module(
                "vllm_omni.diffusion.distributed.sp_plan",
                SequenceParallelConfig=SimpleNamespace,
                get_sp_plan_from_model=lambda model: None,
            ),
            "vllm_omni.diffusion.forward_context": _module(
                "vllm_omni.diffusion.forward_context",
                get_forward_context=lambda: SimpleNamespace(),
            ),
            "vllm_omni.diffusion.hooks.sequence_parallel": _module(
                "vllm_omni.diffusion.hooks.sequence_parallel",
                apply_sequence_parallel=lambda *args: None,
            ),
            "vllm_omni.diffusion.utils.tf_utils": _module(
                "vllm_omni.diffusion.utils.tf_utils", find_module_with_attr=lambda *args: None
            ),
            "vllm_omni.platforms": _module(
                "vllm_omni.platforms",
                current_omni_platform=SimpleNamespace(get_diffusion_packed_modules_mapping=lambda model: None),
            ),
            "vllm_omni.diffusion.models.wan2_2.pipeline_lingbot_world": pipeline_module,
        }
    )
    previous = {name: sys.modules.get(name) for name in stub_modules}
    sys.modules.update(stub_modules)
    spec = importlib.util.spec_from_file_location("_lingbot_registry_under_test", _REGISTRY_PATH)
    assert spec is not None and spec.loader is not None
    registry_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = registry_module
    try:
        spec.loader.exec_module(registry_module)
        resolved = registry_module.DiffusionModelRegistry._try_load_model_cls("LingBotWorldCausalDMDPipeline")
        entry = registry_module._DIFFUSION_MODELS["LingBotWorldCausalDMDPipeline"]
        cache_acceleration_disabled = "LingBotWorldCausalDMDPipeline" in registry_module._NO_CACHE_ACCELERATION
    finally:
        sys.modules.pop(spec.name, None)
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module
    return resolved, entry, cache_acceleration_disabled


def test_component_discovery_uses_official_checkpoint_contract() -> None:
    module = _load_pipeline_module()
    pipeline = module.LingBotWorldCausalDMDPipeline(od_config=_od_config())

    assert pipeline._dit_modules == ["transformer"]
    assert pipeline._encoder_modules == ["text_encoder"]
    assert pipeline._vae_modules == ["vae"]
    assert pipeline.dummy_run_num_frames == 0
    assert pipeline.weights_sources == [_ComponentSource("checkpoint", "transformer", None, "transformer.", True)]
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
    resolved, entry, cache_acceleration_disabled = _resolve_pipeline_through_real_registry(module)

    assert model_index["_class_name"] == "LingBotWorldCausalDMDPipeline"
    assert resolved is module.LingBotWorldCausalDMDPipeline
    assert entry == ("wan2_2", "pipeline_lingbot_world", "LingBotWorldCausalDMDPipeline")
    assert cache_acceleration_disabled
    assert model_index["tokenizer"] == ["transformers", "T5TokenizerFast"]
    assert model_index["text_encoder"] == ["transformers", "UMT5EncoderModel"]
    assert model_index["vae"] == ["diffusers", "AutoencoderKLWan"]
    assert model_index["scheduler"] == ["diffusers", "UniPCMultistepScheduler"]
    assert model_index["transformer"] == ["diffusers", "CausalLingBotWorldTransformer3DModel"]
    assert model_index["image_encoder"] == [None, None]
    assert model_index["image_processor"] == [None, None]
    assert model_index["transformer_2"] == [None, None]


def test_postprocess_reads_output_type_from_sampling_params() -> None:
    module = _load_pipeline_module()
    postprocess = module.get_lingbot_world_post_process_func(_od_config())
    latents = torch.randn(1, 16, 3, 2, 2)

    output = postprocess(latents, sampling_params=SimpleNamespace(output_type="latent"))

    assert output is latents


def test_noise_uses_device_safe_diffusers_helper() -> None:
    module = _load_pipeline_module()
    pipeline = _pipeline(module)
    generator = torch.Generator(device="cpu").manual_seed(4)
    calls = []

    def randn_tensor(shape, *, generator, device, dtype):
        calls.append((shape, generator, device, dtype))
        return torch.full(shape, 7.0, device=device, dtype=dtype)

    module.randn_tensor = randn_tensor
    output = pipeline._randn((1, 2, 3), generator=generator, dtype=torch.float32)

    torch.testing.assert_close(output, torch.full((1, 2, 3), 7.0))
    assert calls == [((1, 2, 3), generator, torch.device("cpu"), torch.float32)]


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

    def randn(shape, *, generator, dtype):
        del generator
        requested_noise_dtypes.append(dtype)
        return torch.zeros(shape, dtype=dtype)

    pipeline._randn = randn
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
        _pipeline(module)._parse_request(_RequestBatch(_prompt(images=str(source_path)), _SamplingParams()))

    assert decode_calls == []


def test_normal_path_image_is_decoded_after_source_size_validation(tmp_path: Path) -> None:
    module = _load_pipeline_module()
    source_path = tmp_path / "normal.png"
    Image.new("RGBA", (32, 24), color=(1, 2, 3, 128)).save(source_path)

    parsed = _pipeline(module)._parse_request(_RequestBatch(_prompt(images=str(source_path)), _SamplingParams()))

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
        _pipeline(module)._parse_request(_RequestBatch(_prompt(images=unsafe_path), _SamplingParams()))

    assert unsafe_path not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize("source_size", [(4096, 4096), (4097, 4097)])
def test_supplied_pil_image_obeys_documented_source_pixel_ceiling(source_size) -> None:
    module = _load_pipeline_module()
    source_image = Image.new("1", source_size)
    close_calls: list[bool] = []
    source_image.close = lambda: close_calls.append(True)

    if source_size[0] * source_size[1] <= 4096 * 4096:
        parsed = _pipeline(module)._parse_request(_RequestBatch(_prompt(images=source_image), _SamplingParams()))
        assert parsed.image is not source_image
        assert parsed.image.mode == "RGB"
        assert close_calls == []
    else:
        with pytest.raises(ValueError, match="source image.*4096.*4096"):
            _pipeline(module)._parse_request(_RequestBatch(_prompt(images=source_image), _SamplingParams()))
        assert close_calls == []


def test_supplied_pil_decode_error_is_sanitized_without_closing_caller(monkeypatch) -> None:
    module = _load_pipeline_module()
    source_image = Image.new("RGB", (16, 16))
    unsafe_detail = "/private/source/customer-secret.png"
    close_calls: list[bool] = []
    monkeypatch.setattr(source_image, "convert", lambda mode: (_ for _ in ()).throw(OSError(unsafe_detail)))
    monkeypatch.setattr(source_image, "close", lambda: close_calls.append(True))

    with pytest.raises(ValueError, match="Unable to load multi_modal_data.image") as exc_info:
        _pipeline(module)._parse_request(_RequestBatch(_prompt(images=source_image), _SamplingParams()))

    assert unsafe_detail not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert close_calls == []


def test_action_path_resolves_from_sampling_extra_args(tmp_path: Path) -> None:
    module = _load_pipeline_module()
    action_root = tmp_path / "trusted-actions"
    contained_action = action_root / "actions"
    contained_action.mkdir(parents=True)
    pipeline = _pipeline(module, od_config=_od_config(model_config={"lingbot_action_root": str(action_root)}))
    sampling = _SamplingParams(extra_args={"action_path": "actions"})
    prompt = _prompt()
    original_extra = dict(sampling.extra_args)
    original_prompt = prompt.copy()

    parsed = pipeline._parse_request(_RequestBatch(prompt, sampling))

    assert parsed.action_path.root == action_root.resolve()
    assert parsed.action_path.relative == Path("actions")
    assert sampling.extra_args == original_extra
    assert prompt == original_prompt


@pytest.mark.parametrize("with_extra", [False, True], ids=["additional-only", "extra-and-additional"])
def test_additional_information_action_path_is_not_an_input_source(with_extra: bool) -> None:
    module = _load_pipeline_module()
    sampling = _SamplingParams(
        extra_args={"action_path": "."} if with_extra else None,
        include_action=with_extra,
    )

    with pytest.raises(ValueError, match="additional_information.*action_path.*not supported"):
        _pipeline(module)._parse_request(_RequestBatch(_prompt(action_path="legacy-actions"), sampling))


def test_action_path_is_required_in_sampling_extra_args() -> None:
    module = _load_pipeline_module()

    with pytest.raises(ValueError, match="sampling_params.extra_args.action_path"):
        _pipeline(module)._parse_request(_RequestBatch(_prompt(), _SamplingParams(include_action=False)))


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
    pipeline = _pipeline(module, od_config=_od_config(model_config={"lingbot_action_root": str(root)}))

    with pytest.raises(ValueError, match="trusted.*root|contained"):
        pipeline._parse_request(_RequestBatch(_prompt(), _SamplingParams(extra_args={"action_path": action_path})))


def test_online_action_path_uses_environment_root_fallback(monkeypatch, tmp_path: Path) -> None:
    module = _load_pipeline_module()
    root = tmp_path / "trusted"
    action_dir = root / "forward"
    action_dir.mkdir(parents=True)
    monkeypatch.setenv("VLLM_OMNI_LINGBOT_ACTION_ROOT", str(root))
    pipeline = _pipeline(module, od_config=_od_config(model_config={}))

    parsed = pipeline._parse_request(_RequestBatch(_prompt(), _SamplingParams(extra_args={"action_path": "forward"})))

    assert parsed.action_path.root == root.resolve()
    assert parsed.action_path.relative == Path("forward")


def test_online_action_path_error_suppresses_path_bearing_filesystem_cause(tmp_path: Path) -> None:
    module = _load_pipeline_module()
    root = tmp_path / "trusted"
    root.mkdir()
    pipeline = _pipeline(module, od_config=_od_config(model_config={"lingbot_action_root": str(root)}))

    with pytest.raises(ValueError, match="trusted action root") as exc_info:
        pipeline._parse_request(_RequestBatch(_prompt(), _SamplingParams(extra_args={"action_path": "does-not-exist"})))

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
    pipeline = _pipeline(module, od_config=_od_config(model_config={"lingbot_action_root": configured_root}))

    with pytest.raises(ValueError, match="trusted action root") as exc_info:
        pipeline._parse_request(_RequestBatch(_prompt(), _SamplingParams(extra_args={"action_path": action_path})))

    assert "__vllm_omni_user_that_does_not_exist__" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_action_path_requires_a_trusted_root_for_every_request() -> None:
    module = _load_pipeline_module()
    pipeline = _pipeline(module, od_config=_od_config(model_config={}))

    with pytest.raises(ValueError, match="lingbot_action_root|VLLM_OMNI_LINGBOT_ACTION_ROOT"):
        pipeline._parse_request(_RequestBatch(_prompt(), _SamplingParams()))


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
    pipeline._allocate_request_cache = lambda *args, **kwargs: calls.append("cache")

    with pytest.raises(ValueError, match=message):
        pipeline(_RequestBatch(_prompt(), sampling))

    assert calls == []


def test_request_validation_rejects_insufficient_camera_frames() -> None:
    module = _load_pipeline_module()
    pipeline = _pipeline(module)
    module.load_camera_trajectory = lambda path: SimpleNamespace(
        poses=torch.eye(4).repeat(8, 1, 1),
        intrinsics=torch.ones(8, 4),
    )

    with pytest.raises(ValueError, match="camera.*frames.*num_frames"):
        pipeline(_request())


def test_camera_load_error_is_actionable_without_echoing_path_contents() -> None:
    module = _load_pipeline_module()
    module.load_camera_trajectory = lambda path: (_ for _ in ()).throw(FileNotFoundError(f"missing {path}/poses.npy"))

    with pytest.raises(ValueError, match="camera trajectory.*action_path") as exc_info:
        _pipeline(module)(_request())

    assert str(_ROOT) not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_first_frame_condition_and_camera_fold_match_transformer_contract() -> None:
    module = _load_pipeline_module()
    transformer = _RecordingTransformer()
    pipeline = _pipeline(module, transformer=transformer)
    pipeline._randn = lambda shape, **kwargs: torch.full(shape, -99.0, dtype=kwargs["dtype"])

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
    expected_camera = module._fold_camera_embedding(raw_camera, spatial_fold=8)
    reference_camera = torch.nn.functional.pixel_unshuffle(raw_camera, 8).permute(1, 0, 2, 3).unsqueeze(0)
    torch.testing.assert_close(expected_camera, reference_camera)
    torch.testing.assert_close(transformer.calls[0]["camera_hidden_states"], expected_camera)
    assert expected_camera.shape == (1, 384, 3, 2, 2)


def test_vae_latent_stats_helper_is_shared_by_encode_and_decode() -> None:
    module = _load_pipeline_module()
    pipeline = _pipeline(module)
    original_stats = pipeline._vae_latent_stats
    references: list[torch.Tensor] = []

    def recording_stats(reference: torch.Tensor):
        references.append(reference)
        return original_stats(reference)

    pipeline._vae_latent_stats = recording_stats

    pipeline(_request(sampling=_SamplingParams(output_type="np")))

    assert len(references) == 2
    assert all(reference.shape[1] == 16 for reference in references)
    assert all(reference.dtype == torch.float32 for reference in references)


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
    assert sampling.extra_args == {"action_path": ".", "flow_shift": 2.0}
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
    module.load_camera_trajectory = lambda path: SimpleNamespace(
        poses=torch.eye(4).repeat(117, 1, 1),
        intrinsics=torch.tensor([[100.0, 100.0, 8.0, 8.0]]).repeat(117, 1),
    )

    result = pipeline(_request(sampling=_SamplingParams(num_frames=117, output_type="latent")))

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


def test_registry_and_wan_exports_resolve_official_pipeline_class_name() -> None:
    module = _load_pipeline_module()
    resolved, entry, cache_acceleration_disabled = _resolve_pipeline_through_real_registry(module)

    assert entry == ("wan2_2", "pipeline_lingbot_world", "LingBotWorldCausalDMDPipeline")
    assert resolved is module.LingBotWorldCausalDMDPipeline
    assert cache_acceleration_disabled
    assert module.LingBotWorldCausalDMDPipeline.__name__ == "LingBotWorldCausalDMDPipeline"
    wan_init = _WAN_INIT_PATH.read_text()
    assert "from .pipeline_lingbot_world import" in wan_init
    assert '"LingBotWorldCausalDMDPipeline"' in wan_init
    assert '"CausalLingBotWorldTransformer3DModel"' in wan_init
