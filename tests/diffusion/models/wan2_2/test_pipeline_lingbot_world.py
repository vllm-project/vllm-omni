# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import ast
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
    def __init__(self, *, raise_on_call: int | None = None):
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
        self.raise_on_call = raise_on_call
        self.loaded_weights: list[tuple[str, torch.Tensor]] = []

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

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
            latents_mean=[0.0] * 16,
            latents_std=[1.0] * 16,
        )
        self.encode_inputs: list[torch.Tensor] = []
        self.decode_inputs: list[torch.Tensor] = []

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
        height: int = 16,
        width: int = 16,
        num_frames: int = 9,
        num_inference_steps: int | None = 4,
        num_outputs_per_prompt: int = 1,
        seed: int | None = 17,
        generator: torch.Generator | None = None,
        output_type: str | None = "latent",
        max_sequence_length: int | None = 8,
        extra_args: dict | None = None,
    ) -> None:
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.num_inference_steps = num_inference_steps
        self.num_outputs_per_prompt = num_outputs_per_prompt
        self.seed = seed
        self.generator = generator
        self.output_type = output_type
        self.max_sequence_length = max_sequence_length
        self.extra_args = {} if extra_args is None else extra_args
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
        return {
            "num_train_timesteps": 1000,
            "prediction_type": "flow_prediction",
            "solver_order": 2,
            "predict_x0": True,
            "solver_type": "bh2",
            "lower_order_final": True,
            "disable_corrector": [],
            "use_dynamic_shifting": False,
            "time_shift_type": "exponential",
            "final_sigmas_type": "zero",
        }

    trajectory = SimpleNamespace(
        poses=torch.eye(4).repeat(32, 1, 1),
        intrinsics=torch.tensor([[100.0, 100.0, 8.0, 8.0]]).repeat(32, 1),
    )

    def load_camera_trajectory(action_path):
        assert action_path in {"actions", "prompt-actions"}
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
                build_plucker_embedding=build_plucker_embedding,
                interpolate_camera_trajectory=interpolate_camera_trajectory,
                load_camera_trajectory=load_camera_trajectory,
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


def _pipeline(module, *, transformer=None):
    pipeline = module.LingBotWorldCausalDMDPipeline(od_config=_od_config())
    if transformer is not None:
        pipeline.transformer = transformer
    pipeline.encode_prompt = lambda *args, **kwargs: torch.ones(1, 8, 8)
    return pipeline


def _request(*, sampling=None, prompt=None, num_reqs: int = 1):
    return _RequestBatch(
        _prompt(action_path="prompt-actions") if prompt is None else prompt,
        _SamplingParams() if sampling is None else sampling,
        num_reqs=num_reqs,
    )


def _registry_models() -> dict:
    tree = ast.parse(_REGISTRY_PATH.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_DIFFUSION_MODELS" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("_DIFFUSION_MODELS assignment not found")


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


@pytest.mark.parametrize("source", ["extra", "prompt"])
def test_action_path_resolves_from_exactly_one_supported_source(source: str) -> None:
    module = _load_pipeline_module()
    pipeline = _pipeline(module)
    sampling = _SamplingParams(extra_args={"action_path": "actions"} if source == "extra" else {})
    prompt = _prompt(action_path="prompt-actions" if source == "prompt" else None)
    original_extra = dict(sampling.extra_args)
    original_prompt = prompt.copy()

    parsed = pipeline._parse_request(_RequestBatch(prompt, sampling))

    assert parsed.action_path == ("actions" if source == "extra" else "prompt-actions")
    assert sampling.extra_args == original_extra
    assert prompt == original_prompt


@pytest.mark.parametrize(
    ("extra_action", "prompt_action", "message"),
    [(None, None, "action_path"), ("actions", "prompt-actions", "ambiguous.*action_path")],
)
def test_action_path_rejects_missing_or_ambiguous_sources(extra_action, prompt_action, message) -> None:
    module = _load_pipeline_module()
    sampling = _SamplingParams(extra_args={} if extra_action is None else {"action_path": extra_action})

    with pytest.raises(ValueError, match=message):
        _pipeline(module)._parse_request(_RequestBatch(_prompt(action_path=prompt_action), sampling))


@pytest.mark.parametrize(
    ("request_batch", "message"),
    [
        (_request(num_reqs=2), "single prompt"),
        (_request(prompt=_prompt(images=[])), "exactly one image"),
        (
            _request(prompt=_prompt(images=[Image.new("RGB", (16, 16)), Image.new("RGB", (16, 16))])),
            "exactly one image",
        ),
        (_request(sampling=_SamplingParams(num_outputs_per_prompt=2)), "num_outputs_per_prompt"),
        (_request(sampling=_SamplingParams(num_inference_steps=5)), "num_inference_steps"),
        (_request(sampling=_SamplingParams(height=15)), "height.*divisible"),
        (_request(sampling=_SamplingParams(width=15)), "width.*divisible"),
        (_request(sampling=_SamplingParams(num_frames=13)), "num_frames.*three-frame"),
    ],
)
def test_request_validation_rejects_unsupported_contracts(request_batch, message: str) -> None:
    module = _load_pipeline_module()

    with pytest.raises(ValueError, match=message):
        _pipeline(module)._parse_request(request_batch)


def test_request_validation_rejects_insufficient_camera_frames() -> None:
    module = _load_pipeline_module()
    pipeline = _pipeline(module)
    module.load_camera_trajectory = lambda path: SimpleNamespace(
        poses=torch.eye(4).repeat(8, 1, 1),
        intrinsics=torch.ones(8, 4),
    )

    with pytest.raises(ValueError, match="camera.*frames.*num_frames"):
        pipeline(_request())


def test_camera_load_error_names_action_path() -> None:
    module = _load_pipeline_module()
    module.load_camera_trajectory = lambda path: (_ for _ in ()).throw(FileNotFoundError(f"missing {path}/poses.npy"))

    with pytest.raises(ValueError, match="action_path.*prompt-actions"):
        _pipeline(module)(_request())


def test_first_frame_condition_and_camera_fold_match_transformer_contract() -> None:
    module = _load_pipeline_module()
    transformer = _RecordingTransformer()
    pipeline = _pipeline(module, transformer=transformer)

    result = pipeline(_request())

    assert result.output.shape == (1, 16, 3, 2, 2)
    first_input = transformer.calls[0]["hidden_states"]
    assert first_input.shape == (1, 36, 3, 2, 2)
    torch.testing.assert_close(first_input[:, 16:32, 0], torch.full((1, 16, 2, 2), 2.0))
    torch.testing.assert_close(first_input[:, 16:32, 1:], torch.zeros(1, 16, 2, 2, 2))
    torch.testing.assert_close(first_input[:, 32:36, 0], torch.ones(1, 4, 2, 2))
    torch.testing.assert_close(first_input[:, 32:36, 1:], torch.zeros(1, 4, 2, 2, 2))

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


def test_fixed_dmd_transition_and_cache_commit_trace() -> None:
    module = _load_pipeline_module()
    transformer = _RecordingTransformer()
    pipeline = _pipeline(module, transformer=transformer)

    result = pipeline(_request())

    generator = torch.Generator(device="cpu").manual_seed(17)
    current = torch.randn((1, 16, 3, 2, 2), generator=generator)
    for index, timestep in enumerate(module.LINGBOT_DMD_TIMESTEPS):
        sigma = module._shifted_flow_sigma(timestep, flow_shift=5.0, num_train_timesteps=1000)
        x0 = current - sigma
        if index + 1 < len(module.LINGBOT_DMD_TIMESTEPS):
            next_sigma = module._shifted_flow_sigma(
                module.LINGBOT_DMD_TIMESTEPS[index + 1],
                flow_shift=5.0,
                num_train_timesteps=1000,
            )
            noise = torch.randn(current.shape, generator=generator)
            current = (1.0 - next_sigma) * x0 + next_sigma * noise
        else:
            current = x0
    torch.testing.assert_close(result.output, current)

    assert [int(call["timestep"].item()) for call in transformer.calls] == [1000, 750, 500, 250, 0]
    assert [call["update_cache"] for call in transformer.calls] == [False, False, False, False, True]
    assert [call["start_frame"] for call in transformer.calls] == [0, 0, 0, 0, 0]
    assert len({call["cache_id"] for call in transformer.calls}) == 1
    torch.testing.assert_close(transformer.calls[-1]["hidden_states"][:, :16], result.output)


def test_dmd_sigma_lookup_does_not_apply_flow_shift_twice() -> None:
    module = _load_pipeline_module()

    assert module._shifted_flow_sigma(1000, flow_shift=5.0, num_train_timesteps=1000) == 1.0
    assert module._shifted_flow_sigma(750, flow_shift=5.0, num_train_timesteps=1000) == pytest.approx(0.75)


def test_multi_chunk_generation_uses_one_request_local_cache_and_decodes_accumulated_latents() -> None:
    module = _load_pipeline_module()
    transformer = _RecordingTransformer()
    pipeline = _pipeline(module, transformer=transformer)
    allocations = []
    original_allocate = module.allocate_lingbot_cache

    def allocate(**kwargs):
        cache = original_allocate(**kwargs)
        allocations.append((kwargs, weakref.ref(cache)))
        return cache

    module.allocate_lingbot_cache = allocate
    sampling = _SamplingParams(num_frames=21, output_type="np")

    result = pipeline(_request(sampling=sampling))

    assert result.output.shape == (1, 3, 21, 16, 16)
    assert len(transformer.calls) == 10
    assert [call["start_frame"] for call in transformer.calls] == [0] * 5 + [3] * 5
    assert [int(call["timestep"].item()) for call in transformer.calls] == [1000, 750, 500, 250, 0] * 2
    assert len(allocations) == 1
    kwargs, cache_ref = allocations[0]
    assert kwargs["max_tokens"] == 6
    assert not hasattr(pipeline, "cache")
    assert not hasattr(pipeline, "transformer_cache")
    assert pipeline.vae.decode_inputs[0].shape == (1, 16, 6, 2, 2)
    transformer.calls.clear()
    gc.collect()
    assert cache_ref() is None


def test_request_cache_becomes_unreachable_after_transformer_error() -> None:
    module = _load_pipeline_module()
    transformer = _RecordingTransformer(raise_on_call=2)
    pipeline = _pipeline(module, transformer=transformer)
    cache_refs = []
    original_allocate = module.allocate_lingbot_cache

    def allocate(**kwargs):
        cache = original_allocate(**kwargs)
        cache_refs.append(weakref.ref(cache))
        return cache

    module.allocate_lingbot_cache = allocate

    with pytest.raises(RuntimeError, match="forced transformer failure"):
        pipeline(_request())

    transformer.calls.clear()
    gc.collect()
    assert len(cache_refs) == 1
    assert cache_refs[0]() is None
    assert not hasattr(pipeline, "cache")
    assert not hasattr(pipeline, "transformer_cache")


def test_registry_and_wan_exports_resolve_official_pipeline_class_name() -> None:
    module = _load_pipeline_module()
    entry = _registry_models().get("LingBotWorldCausalDMDPipeline")

    assert entry == ("wan2_2", "pipeline_lingbot_world", "LingBotWorldCausalDMDPipeline")
    assert module.LingBotWorldCausalDMDPipeline.__name__ == "LingBotWorldCausalDMDPipeline"
    wan_init = _WAN_INIT_PATH.read_text()
    assert "from .pipeline_lingbot_world import" in wan_init
    assert '"LingBotWorldCausalDMDPipeline"' in wan_init
    assert '"CausalLingBotWorldTransformer3DModel"' in wan_init
