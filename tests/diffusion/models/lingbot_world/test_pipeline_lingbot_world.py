# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

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

import vllm_omni.diffusion.models.lingbot_world.dmd_block as lingbot_dmd_block
import vllm_omni.diffusion.models.lingbot_world.pipeline as lingbot_pipeline
from tests.diffusion.models.wan2_2.conftest import noop_progress_bar
from vllm_omni.diffusion.models.interface import SupportsStepExecution, supports_step_execution
from vllm_omni.diffusion.models.lingbot_world.actions import (
    integrate_lingbot_camera_actions,
)
from vllm_omni.diffusion.models.lingbot_world.camera import CameraTrajectory as _CameraTrajectory
from vllm_omni.diffusion.models.lingbot_world.camera import (
    build_plucker_embedding as _real_build_plucker_embedding,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.utils import StepRequestState
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


class _StubCausalDecoder:
    """Stand-in for the Wan causal decoder's cache protocol.

    Reproduces only what streaming depends on: each call walks ``feat_idx``
    across one cache slot per causal convolution, and the session's opening
    frame (``first_chunk``) expands to a single raw frame while every later
    frame expands by the temporal factor.
    """

    NUM_CONVS = 3

    def __init__(self) -> None:
        self.first_chunk_flags: list[bool] = []
        # Identity of the cache each call wrote through, so an interleaved run
        # can assert that a session never advanced another session's context.
        self.cache_ids: list[int] = []

    def __call__(self, x: torch.Tensor, *, feat_cache, feat_idx, first_chunk: bool = False):
        self.first_chunk_flags.append(bool(first_chunk))
        self.cache_ids.append(id(feat_cache))
        for _ in range(self.NUM_CONVS):
            index = feat_idx[0]
            feat_cache[index] = x.detach().clone()
            feat_idx[0] = index + 1
        num_frames = 1 if first_chunk else 4
        return x.new_zeros(x.shape[0], 3, num_frames, x.shape[-2] * 8, x.shape[-1] * 8)


class _StubCausalEncoder(nn.Module):
    """Minimal Wan cache layout; the real tiny VAE below checks numerical parity."""

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv3d(3, 16, 1)
        self.down_blocks = nn.ModuleList([lingbot_pipeline.WanResample(16, mode="downsample2d") for _ in range(3)])
        self.mid_block = SimpleNamespace(resnets=[])
        self.conv_out = nn.Conv3d(16, 32, 1)
        self.inputs: list[torch.Tensor] = []

    def forward(self, video, *, feat_cache, feat_idx):
        self.inputs.append(video.detach().clone())
        first = feat_cache[0] is None
        tail = video[:, :, -2:].clone()
        if tail.shape[2] == 1 and feat_cache[0] is not None:
            tail = torch.cat((feat_cache[0][:, :, -1:], tail), dim=2)
        feat_cache[0] = tail
        height, width = video.shape[-2] // 8, video.shape[-1] // 8
        previous_frames = 0 if feat_cache[1] is None else feat_cache[1].shape[2]
        feat_cache[1] = video.new_zeros(1, 16, min(2, previous_frames + 1), height, width)
        feat_idx[0] += 2
        moments = video.new_zeros(1, 32, 1, height, width)
        if first:
            moments[:, :16] = 2.0
        return moments


class _StubVAE(_FakePretrained):
    dtype = torch.float32

    def __init__(self, *, streaming: bool = True):
        self.encoder = _StubCausalEncoder()
        self.quant_conv = nn.Identity()
        self._cached_conv_counts = {"encoder": 2}
        self._enc_feat_map = ["module-owned"]
        self._enc_conv_idx = [71]
        if streaming:
            self.decoder = _StubCausalDecoder()
            # Recorded, not discarded: on the streaming branch this is the only
            # place the tensor handed to the decoder can be observed, and it is
            # what carries the checkpoint's latent rescale.
            self.post_quant_inputs: list[torch.Tensor] = []
            self.post_quant_conv = self._record_post_quant
            self._cached_conv_counts["decoder"] = _StubCausalDecoder.NUM_CONVS
            # Module-owned cache the shared VAE keeps for whole-clip decode;
            # streaming must never write through it.
            self._feat_map = ["module-owned"]
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

    def _record_post_quant(self, latent: torch.Tensor) -> torch.Tensor:
        self.post_quant_inputs.append(latent.detach().clone())
        return latent

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
        self.guidance_scale = None
        self.guidance_scale_2 = None


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
    # The DMD block math lives in its own module; stub the symbols it imports.
    monkeypatch.setattr(lingbot_dmd_block, "set_forward_context_denoise_step_idx", lambda index: None)
    monkeypatch.setattr(lingbot_dmd_block, "randn_tensor", _diffusers_randn_tensor)
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
            ulysses_degree=1,
            ring_degree=1,
            allgather_degree=1,
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


def test_ar_diffusion_capability_uses_transformer_local_head_geometry() -> None:
    module = _load_pipeline_module()
    pipeline = _pipeline(module)
    # Use the constructed head count even when the config describes a different geometry.
    pipeline.transformer.blocks[0].self_attn = SimpleNamespace(num_sp_heads=1)
    spec = pipeline.ar_diffusion_kv_cache_spec()

    assert spec.num_layers == 2
    assert spec.num_kv_heads == 1
    assert spec.head_size == 4
    assert spec.tokens_per_frame == 1
    assert spec.frames_per_block == 3
    assert spec.window_frames == 3
    assert spec.sink_frames == 3
    assert [(branch.name, branch.local_index) for branch in spec.kv_branches] == [("main", 0)]
    assert spec.cross_attention_lengths == {"text": 512}
    # One condition block, committed + pending encoder caches, and decoder state.
    # The stub encoder retains 2 RGB frames at 16x16 and 2 feature frames at 2x2.
    encoder_bytes = (2 * 3 * 16 * 16 + 2 * 16 * 2 * 2) * 4
    assert spec.model_owned_state_bytes_per_session == 960 + 2 * encoder_bytes + 2_421_248


def test_session_admission_accounts_for_the_streaming_decoder_cache() -> None:
    """Admission includes encoder state even when the decoder cannot stream."""
    module = _load_pipeline_module()

    def spec_pipeline():
        pipeline = _pipeline(module)
        # The spec reads the constructed attention's head count; the head count
        # plays no part in model-owned bytes, so any value will do here.
        pipeline.transformer.blocks[0].self_attn = SimpleNamespace(num_sp_heads=1)
        return pipeline

    small = spec_pipeline().ar_diffusion_kv_cache_spec().model_owned_state_bytes_per_session

    wide = spec_pipeline()
    wide._ar_width = wide._ar_width * 2
    widened = wide.ar_diffusion_kv_cache_spec().model_owned_state_bytes_per_session
    # Twice the area: the decoder cache doubles, so the total grows by far more
    # than the image condition alone could account for.
    assert widened - small > 2_000_000

    # Removing decoder support leaves the condition and both encoder histories.
    bare = spec_pipeline()
    for attribute in ("decoder", "post_quant_conv"):
        if hasattr(bare.vae, attribute):
            delattr(bare.vae, attribute)
    bare.vae._cached_conv_counts.pop("decoder")
    assert bare.ar_diffusion_kv_cache_spec().model_owned_state_bytes_per_session == 960 + 2 * 6_656


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


def _ulysses_config(size: int):
    parallel_config = _od_config().parallel_config
    parallel_config.sequence_parallel_size = size
    parallel_config.ulysses_degree = size
    return _od_config(parallel_config=parallel_config)


def _record_shard_install(monkeypatch, module, *, world_size: int):
    """Stand in for the Ulysses group and the decoder patch; return what the install was asked for."""
    installs: list[dict] = []
    group = object()

    def install(vae, group_arg, split_dim, *, dst):
        installs.append({"vae": vae, "group": group_arg, "split_dim": split_dim, "dst": dst})

    monkeypatch.setattr(module, "install_wan_spatial_shard_decode", install)
    monkeypatch.setattr(module.LingBotWorldCausalDMDPipeline, "_vae_shard_group", lambda self: (group, world_size))
    return installs, group


def test_a_multi_rank_deployment_shards_the_decoder_across_the_ulysses_ranks(monkeypatch) -> None:
    """No switch: running the DiT on several Ulysses ranks is what shards the decode across them."""
    module = _load_pipeline_module()
    installs, group = _record_shard_install(monkeypatch, module, world_size=2)

    pipeline = module.LingBotWorldCausalDMDPipeline(od_config=_ulysses_config(2))

    # Along the width, and assembled on every rank: each rank's post_decode consumes the frame.
    assert installs == [{"vae": pipeline.vae, "group": group, "split_dim": "width", "dst": None}]
    assert pipeline._vae_shard_split_dim == "width"


def test_a_single_rank_deployment_leaves_the_decoder_alone(monkeypatch) -> None:
    module = _load_pipeline_module()
    installs, _ = _record_shard_install(monkeypatch, module, world_size=1)

    pipeline = module.LingBotWorldCausalDMDPipeline(od_config=_ulysses_config(1))

    assert installs == [] and pipeline._vae_shard_split_dim is None


def test_vae_shard_refuses_a_group_of_the_wrong_size(monkeypatch) -> None:
    module = _load_pipeline_module()
    installs, _ = _record_shard_install(monkeypatch, module, world_size=4)

    with pytest.raises(RuntimeError, match="sequence_parallel_size=2 but the Ulysses group has 4 ranks"):
        module.LingBotWorldCausalDMDPipeline(od_config=_ulysses_config(2))
    assert installs == []


@pytest.mark.parametrize(
    ("field", "value", "feature"),
    [
        ("pipeline_parallel_size", 2, "pipeline parallelism"),
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


def test_pure_ulysses_parallel_config_is_supported(monkeypatch) -> None:
    module = _load_pipeline_module()
    _record_shard_install(monkeypatch, module, world_size=2)
    parallel_config = _od_config().parallel_config
    parallel_config.sequence_parallel_size = 2
    parallel_config.ulysses_degree = 2

    pipeline = module.LingBotWorldCausalDMDPipeline(od_config=_od_config(parallel_config=parallel_config))

    assert pipeline.transformer is not None


@pytest.mark.parametrize(
    "overrides",
    [
        {"sequence_parallel_size": 4, "ring_degree": 2},  # Normalized hybrid.
        {"ulysses_degree": 1, "allgather_degree": 2},  # Normalized AllGather-KV.
        {"ring_degree": 2},  # Isolate each clause from the SP-size mismatch.
        {"allgather_degree": 2},
        {"ulysses_mode": "advanced_uaa"},
        {"ulysses_a2a_permute": True},
        {"ulysses_degree": None},  # Missing degree must not imply pure Ulysses.
    ],
)
def test_unsupported_sp_config_fails_before_component_loading(overrides):
    module = _load_pipeline_module()
    config = _od_config().parallel_config
    config.sequence_parallel_size = config.ulysses_degree = 2
    for name, value in overrides.items():
        if value is None:
            delattr(config, name)
        else:
            setattr(config, name, value)
    with pytest.raises(NotImplementedError, match="pure Ulysses"):
        module.LingBotWorldCausalDMDPipeline(od_config=_od_config(parallel_config=config))
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
    lingbot_dmd_block.randn_tensor = randn
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


def test_request_validation_accepts_frames_beyond_previous_limit() -> None:
    module = _load_pipeline_module()
    sampling = _SamplingParams(height=480, width=832, num_frames=129, max_sequence_length=512)

    parsed = _pipeline(module)._parse_request(_RequestBatch(_prompt(), sampling))

    assert (parsed.height, parsed.width, parsed.num_frames, parsed.max_sequence_length) == (480, 832, 129, 512)


@pytest.mark.parametrize(
    ("sampling", "message"),
    [
        (_SamplingParams(height=480, width=848), "pixel area|480.*832"),
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
    module.randn_tensor = lingbot_dmd_block.randn_tensor = lambda shape, **kwargs: torch.full(
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


def _tick_extra_args(*, chunk_index: int, prompt: str = "move through the room", session_id: str = "world-1"):
    return ARDiffusionTickRequest(
        session_id=session_id,
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


def test_realtime_rejects_pixel_output_before_condition_encoding() -> None:
    pipeline = _pipeline(_load_pipeline_module())
    pipeline._ar_diffusion_kv_state = object()
    sampling = _SamplingParams(output_type="np", extra_args=_tick_extra_args(chunk_index=0))
    with pytest.raises(ValueError, match="require output_type='latent'"):
        pipeline(_request(sampling=sampling))
    assert pipeline._ar_sessions == {}
    assert pipeline.vae.encoder.inputs == []


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


def test_typed_ticks_keep_bounded_encoder_state_beyond_ten_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_pipeline_module()
    pipeline = _pipeline(module)
    pipeline._ar_height = 16
    pipeline._ar_width = 16
    pipeline._ar_diffusion_kv_state = object()
    generated = []
    monkeypatch.setattr(
        pipeline,
        "_ar_text_caches",
        lambda *args, **kwargs: [SimpleNamespace()],
    )

    def generate_block(**kwargs):
        generated.append(
            {
                "condition": kwargs["condition"].clone(),
                "start_frame": kwargs["start_frame"],
            }
        )
        return torch.zeros_like(kwargs["condition"][:, :16])

    monkeypatch.setattr(pipeline, "_generate_block", generate_block)

    for chunk_index in range(11):
        sampling = _SamplingParams(
            extra_args=_tick_extra_args(chunk_index=chunk_index),
        )
        pipeline(_request(sampling=sampling))

    assert pipeline.vae.encode_inputs == []
    assert [video.shape[2] for video in pipeline.vae.encoder.inputs] == [1] + [4] * 32
    assert all(torch.count_nonzero(video) == 0 for video in pipeline.vae.encoder.inputs[1:])
    encoder_cache = pipeline._ar_sessions["world-1"].encoder_cache
    assert sum(t.numel() * t.element_size() for t in encoder_cache) == pipeline._condition_encoder_cache_bytes()
    assert pipeline.vae._enc_feat_map == ["module-owned"]
    assert pipeline.vae._enc_conv_idx == [71]
    assert pipeline._ar_sessions["world-1"].next_chunk_index == 11
    assert [item["start_frame"] for item in generated] == list(range(0, 33, 3))
    assert torch.count_nonzero(generated[0]["condition"][:, :4]) > 0
    torch.testing.assert_close(
        generated[1]["condition"][:, :4],
        torch.zeros_like(generated[1]["condition"][:, :4]),
    )
    for item in generated[2:]:
        torch.testing.assert_close(item["condition"], generated[1]["condition"])


def _tiny_condition_pipeline(monkeypatch: pytest.MonkeyPatch):
    from diffusers import AutoencoderKLWan

    from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import retrieve_latents

    pipeline = _pipeline(_load_pipeline_module())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        pipeline.vae = AutoencoderKLWan(
            base_dim=4,
            z_dim=16,
            dim_mult=[1, 1, 1, 1],
            num_res_blocks=1,
            temperal_downsample=[False, True, True],
        ).eval()
    monkeypatch.setattr(lingbot_pipeline, "retrieve_latents", retrieve_latents)
    monkeypatch.setattr(pipeline, "_ar_text_caches", lambda *args, **kwargs: [SimpleNamespace()])
    return pipeline


@pytest.mark.parametrize("mode", ["realtime", "stepwise"])
def test_condition_continues_real_causal_vae_beyond_initial_horizon(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    pipeline = _tiny_condition_pipeline(monkeypatch)
    blocks = 12
    inputs = pipeline._parse_request(_request(sampling=_SamplingParams(num_frames=(blocks * 3 - 1) * 4 + 1)))
    with torch.inference_mode():
        expected = pipeline._prepare_condition(inputs, dtype=torch.float32)
    module_cache = pipeline.vae._enc_feat_map
    seen = []

    def check(condition):
        index = len(seen)
        torch.testing.assert_close(condition, expected[:, :, index * 3 : (index + 1) * 3], rtol=1e-6, atol=1e-6)
        seen.append(condition.clone())

    if mode == "realtime":
        pipeline._ar_diffusion_kv_state = object()

        def generate_block(**kwargs):
            check(kwargs["condition"])
            return torch.zeros_like(kwargs["condition"][:, :16])

        monkeypatch.setattr(pipeline, "_generate_block", generate_block)
        session_id = "world-1"
        outputs = (
            pipeline(_request(sampling=_SamplingParams(extra_args=_tick_extra_args(chunk_index=index))))
            for index in range(blocks)
        )
    else:

        def probe_step(**kwargs):
            if kwargs["step_index"] == 0:
                check(kwargs["condition"])
            return torch.ones_like(kwargs["current_latents"])

        monkeypatch.setattr(pipeline._dmd_blocks, "probe_step", probe_step)
        monkeypatch.setattr(pipeline._dmd_blocks, "commit_block_kv", lambda **kwargs: None)
        state = _stepwise_state(num_frames=(blocks * 3 - 1) * 4 + 1)
        session_id = state.request_id
        outputs = _stepwise_chunks(pipeline, state, _FakeARState(session_id))

    cache_sizes = []
    for _ in outputs:
        state = pipeline._ar_sessions.get(session_id)
        if state is not None:
            for cache in (state.encoder_cache, state.pending_encoder_cache):
                if cache is not None:
                    size = sum(t.numel() * t.element_size() for t in cache if t is not None)
                    assert size == pipeline._condition_encoder_cache_bytes()
                    assert all(t.grad_fn is None for t in cache if t is not None)
                    cache_sizes.append(size)
    assert len(seen) == blocks and len(set(cache_sizes)) == 1
    assert pipeline.vae._enc_feat_map is module_cache and all(t is None for t in module_cache)
    pipeline.close_ar_diffusion_session(session_id)
    assert pipeline._ar_sessions == {}


def test_condition_encoder_interleaving_retry_and_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _tiny_condition_pipeline(monkeypatch)
    pipeline._ar_diffusion_kv_state = object()
    prompts = {
        "a": _prompt(images=Image.new("RGB", (16, 16), (255, 0, 0))),
        "b": _prompt(images=Image.new("RGB", (16, 16), (0, 255, 0))),
    }
    expected = {}
    with torch.inference_mode():
        for name, prompt in prompts.items():
            inputs = pipeline._parse_request(_request(prompt=prompt, sampling=_SamplingParams(num_frames=45)))
            expected[name] = pipeline._prepare_condition(inputs, dtype=torch.float32)
    assert not torch.equal(expected["a"], expected["b"])
    active = None
    failure = None
    original_encoder = pipeline.vae.encoder.forward

    def encoder(*args, **kwargs):
        output = original_encoder(*args, **kwargs)
        if failure == "encoder":
            raise RuntimeError("injected encoder failure")
        return output

    def generate_block(**kwargs):
        index = kwargs["start_frame"]
        torch.testing.assert_close(kwargs["condition"], expected[active][:, :, index : index + 3], rtol=1e-6, atol=1e-6)
        if failure == "dit":
            raise RuntimeError("injected DiT failure")
        return torch.zeros_like(kwargs["condition"][:, :16])

    monkeypatch.setattr(pipeline.vae.encoder, "forward", encoder)
    monkeypatch.setattr(pipeline, "_generate_block", generate_block)

    def run(name, index):
        nonlocal active
        active = name
        sampling = _SamplingParams(extra_args=_tick_extra_args(chunk_index=index, session_id=name))
        return pipeline(_request(prompt=prompts[name], sampling=sampling))

    run("a", 0)
    for index, kind in [(1, "encoder"), (2, "dit")]:
        state = pipeline._ar_sessions["a"]
        committed = state.encoder_cache
        saved = [t.clone() if t is not None else None for t in committed]
        failure = kind
        with pytest.raises(RuntimeError, match="injected"):
            run("a", index)
        assert state.next_chunk_index == index and state.encoder_cache is committed
        assert state.pending_encoder_cache is None
        for before, after in zip(saved, committed, strict=True):
            if before is not None:
                torch.testing.assert_close(before, after, rtol=0, atol=0)
        failure = None
        run("b", index - 1)
        run("a", index)

    old_a = pipeline._ar_sessions["a"]
    pipeline.reset_ar_diffusion_session("a")
    assert old_a.encoder_cache is None and old_a.pending_encoder_cache is None
    assert "a" not in pipeline._ar_sessions and "b" in pipeline._ar_sessions
    run("a", 0)
    old_b = pipeline._ar_sessions["b"]
    pipeline.close_ar_diffusion_session("b")
    assert old_b.encoder_cache is None and old_b.pending_encoder_cache is None
    assert "a" in pipeline._ar_sessions and "b" not in pipeline._ar_sessions
    pipeline.close_ar_diffusion_session("a")
    assert pipeline._ar_sessions == {}


def test_stepwise_failed_chunk_releases_pending_encoder_on_close(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _pipeline(_load_pipeline_module())
    state = _stepwise_state()
    transformer = pipeline.transformer
    transformer.raise_on_call = 1
    with pipeline.bind_ar_diffusion_state(state.request_id, _FakeARState(state.request_id)):
        pipeline.prepare_encode(state)
        session = pipeline._ar_sessions[state.request_id]
        assert session.encoder_cache is None and session.pending_encoder_cache is not None
        with pytest.raises(RuntimeError, match="forced transformer failure"):
            pipeline.denoise_step(None, states=[state])
        assert session.next_chunk_index == 0 and session.encoder_cache is None
    pipeline.close_ar_diffusion_session(state.request_id)
    assert session.encoder_cache is None and session.pending_encoder_cache is None
    assert state.request_id not in pipeline._ar_sessions


@pytest.mark.parametrize("mode", ["realtime", "stepwise"])
def test_stateful_condition_rejects_tiled_encoder(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    pipeline = _pipeline(_load_pipeline_module())
    _enable_vae_tiling(pipeline, tile_sample_min=8)
    with pytest.raises(ValueError, match="does not support tiled VAE encoding"):
        if mode == "realtime":
            pipeline._ar_diffusion_kv_state = object()
            pipeline(_request(sampling=_SamplingParams(extra_args=_tick_extra_args(chunk_index=0))))
        else:
            state = _stepwise_state()
            with pipeline.bind_ar_diffusion_state(state.request_id, _FakeARState(state.request_id)):
                pipeline.prepare_encode(state)
    assert pipeline.vae.encoder.inputs == []
    assert all(s.encoder_cache is None and s.pending_encoder_cache is None for s in pipeline._ar_sessions.values())


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


def test_129_frame_request_generates_eleven_complete_latent_blocks() -> None:
    module = _load_pipeline_module()
    transformer = _RecordingTransformer()
    pipeline = _pipeline(module, transformer=transformer)
    trajectory = _CameraTrajectory(
        poses=torch.eye(4).repeat(129, 1, 1),
        intrinsics=torch.tensor([[100.0, 100.0, 8.0, 8.0]]).repeat(129, 1),
    )
    sampling = _SamplingParams(
        num_frames=129,
        output_type="latent",
        extra_args={"_lingbot_camera_trajectory": trajectory},
    )

    result = pipeline(_request(sampling=sampling))

    assert result.output.shape == (1, 16, 33, 2, 2)
    assert len(transformer.calls) == 55
    assert [call["start_frame"] for call in transformer.calls[::5]] == list(range(0, 33, 3))


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


class _FakeARState:
    def __init__(self, session_id: str = "req-1") -> None:
        self.session_id = session_id
        self.commits: list[str] = []

    def get_kv_caches(self, branch, *, seq_len, commit_current):
        del branch, seq_len, commit_current
        return [SimpleNamespace()]

    def commit_paged_context(self, branch):
        self.commits.append(branch)

    def clear_cross_attention(self):
        return None

    def is_cross_attention_populated(self, branch, name):
        del branch, name
        return True

    def get_cross_attention_kv(self, branch, name):
        del branch, name
        zeros = torch.zeros(1, 1, 2, 4)
        return [{"k": zeros, "v": zeros} for _ in range(2)]


def _empty_action_script(num_chunks: int):
    return tuple(((), (), ()) for _ in range(num_chunks))


def _stepwise_state(
    *,
    request_id: str = "req-1",
    num_frames: int = 21,
    script=None,
    seed: int = 17,
):
    num_chunks = ((num_frames - 1) // 4 + 1) // 3
    sampling = _SamplingParams(num_frames=num_frames, seed=seed)
    sampling.extra_args["_lingbot_camera_trajectory"] = None
    sampling.extra_args["_lingbot_camera_actions"] = None
    sampling.extra_args["_lingbot_camera_action_script"] = (
        script if script is not None else _empty_action_script(num_chunks)
    )
    return StepRequestState(
        request_id=request_id,
        sampling=sampling,
        prompt=_prompt(),
    )


def _run_stepwise(pipeline, state):
    outputs = []
    pipeline.prepare_encode(state)
    while not state.request_denoise_completed:
        noise = pipeline.denoise_step(None, states=[state])
        pipeline.step_scheduler(state, noise)
        if state.chunk_denoise_completed:
            outputs.append(pipeline.post_decode(state))
    return outputs


def _stepwise_chunks(pipeline, state, ar_state):
    """Yield one chunk at a time, binding AR state around each invocation.

    The runner binds runner-owned KV for the duration of one stepwise
    invocation and releases it again, so a generator that suspends at each
    chunk boundary is what lets two requests be driven the way the scheduler
    drives them at session_capacity > 1: A-chunk0, B-chunk0, A-chunk1, ...
    """
    with pipeline.bind_ar_diffusion_state(state.request_id, ar_state):
        pipeline.prepare_encode(state)
    while not state.request_denoise_completed:
        output = None
        with pipeline.bind_ar_diffusion_state(state.request_id, ar_state):
            noise = pipeline.denoise_step(None, states=[state])
            pipeline.step_scheduler(state, noise)
            if state.chunk_denoise_completed:
                output = pipeline.post_decode(state)
        if output is not None:
            yield output


def test_pipeline_declares_step_execution_support() -> None:
    module = _load_pipeline_module()
    pipeline = _pipeline(module)

    assert pipeline.supports_step_execution is True
    assert isinstance(pipeline, SupportsStepExecution)
    assert supports_step_execution(pipeline) is True


def test_preprocess_materializes_camera_action_script_without_action_path() -> None:
    module = _load_pipeline_module()
    sampling = _SamplingParams(include_action=False)
    sampling.extra_args["camera_action_script"] = [[["w"], ["w"], ["w"]], [["a"], [], []]]
    request = OmniDiffusionRequest(prompt=_prompt(), sampling_params=sampling, request_id="req-1")

    result = module.get_lingbot_world_pre_process_func(_od_config())(request)

    assert result.sampling_params.extra_args["_lingbot_camera_trajectory"] is None
    assert result.sampling_params.extra_args["_lingbot_camera_actions"] is None
    assert result.sampling_params.extra_args["_lingbot_camera_action_script"] == (
        (("w",), ("w",), ("w",)),
        (("a",), (), ()),
    )


@pytest.mark.parametrize(
    "extra_args",
    [
        pytest.param({}, id="request_mode"),
        pytest.param(_tick_extra_args(chunk_index=0), id="tick_mode"),
    ],
)
def test_forward_rejects_a_stepwise_camera_action_script(extra_args) -> None:
    """A script only steers step execution, so request mode must not drop it silently."""
    module = _load_pipeline_module()
    pipeline = _pipeline(module)
    sampling = _SamplingParams(
        include_action=False,
        extra_args={
            **extra_args,
            "_lingbot_camera_trajectory": None,
            "_lingbot_camera_action_script": _empty_action_script(1),
        },
    )

    with pytest.raises(ValueError, match="camera_action_script is read only by LingBot step execution"):
        pipeline(_request(sampling=sampling))


def test_stepwise_progress_metadata_and_commit_trace() -> None:
    module = _load_pipeline_module()
    transformer = _RecordingTransformer()
    pipeline = _pipeline(module, transformer=transformer)
    pipeline._ar_height = 16
    pipeline._ar_width = 16
    state = _stepwise_state(num_frames=21)
    fake = _FakeARState(state.request_id)

    with pipeline.bind_ar_diffusion_state(state.request_id, fake):
        outputs = _run_stepwise(pipeline, state)

    assert pipeline._ar_diffusion_kv_state is None
    assert len(outputs) == 2
    assert [output.chunk_index for output in outputs] == [0, 1]
    assert all(output.total_chunks == 2 for output in outputs)
    assert outputs[-1].finished is True
    assert [call["update_cache"] for call in transformer.calls] == [False, False, False, False, True] * 2
    assert [call["start_frame"] for call in transformer.calls] == [0] * 5 + [3] * 5
    assert fake.commits == ["main", "main"]
    for chunk_index, output in enumerate(outputs):
        metadata = output.output["metadata"]["ar_diffusion"]
        assert metadata == {
            "session_id": "req-1",
            "request_id": "req-1",
            "chunk_index": chunk_index,
            "applied_event_ids": [],
        }
        assert output.output["payload"]["latents"].shape == (1, 16, 3, 2, 2)


@pytest.mark.parametrize("num_chunks", [2, 12])
def test_stepwise_matches_tick_transformer_trace_and_latents(num_chunks) -> None:
    module = _load_pipeline_module()
    transformer = _RecordingTransformer()
    pipeline = _pipeline(module, transformer=transformer)
    pipeline._ar_height = 16
    pipeline._ar_width = 16
    script = ((("w",), ("w",), ("w",)), (("d",), ("d",), ("d",))) * (num_chunks // 2)
    fake = _FakeARState("world-1")

    with pipeline.bind_ar_diffusion_state("world-1", fake):
        tick_latents = []
        for chunk_index, frames in enumerate(script):
            tick = ARDiffusionTickRequest(
                session_id="world-1",
                request_id="legacy-lingbot-request",
                chunk_index=chunk_index,
                controls=(
                    ARDiffusionControlInput(
                        track="camera",
                        schema="lingbot.camera_actions.v1",
                        data={"mode": "frames", "frames": [list(frame) for frame in frames]},
                    ),
                ),
            )
            sampling = _SamplingParams(extra_args=tick.to_extra_args(), seed=17)
            sampling.extra_args["_lingbot_camera_trajectory"] = None
            sampling.extra_args["_lingbot_camera_actions"] = frames
            result = pipeline(_request(sampling=sampling))
            tick_latents.append(result.output["payload"]["latents"].clone())
    tick_calls = list(transformer.calls)
    transformer.calls.clear()
    pipeline.close_ar_diffusion_session("world-1")

    stepwise_state = _stepwise_state(request_id="req-1", num_frames=12 * num_chunks - 3, script=script, seed=17)
    stepwise_fake = _FakeARState(stepwise_state.request_id)
    with pipeline.bind_ar_diffusion_state(stepwise_state.request_id, stepwise_fake):
        stepwise_outputs = _run_stepwise(pipeline, stepwise_state)

    assert [call["update_cache"] for call in transformer.calls] == [call["update_cache"] for call in tick_calls]
    assert [call["start_frame"] for call in transformer.calls] == [call["start_frame"] for call in tick_calls]
    for tick_latent, output in zip(tick_latents, stepwise_outputs, strict=True):
        torch.testing.assert_close(output.output["payload"]["latents"], tick_latent)
    for step_call, tick_call in zip(transformer.calls, tick_calls, strict=True):
        torch.testing.assert_close(step_call["hidden_states"], tick_call["hidden_states"])
        torch.testing.assert_close(step_call["camera_hidden_states"], tick_call["camera_hidden_states"])
    assert stepwise_state.extra["condition"].shape[2] == 3
    assert pipeline.vae.encode_inputs == []
    assert [video.shape[2] for video in pipeline.vae.encoder.inputs] == ([1] + [4] * (num_chunks * 3 - 1)) * 2
    assert stepwise_fake.commits == ["main"] * num_chunks
    assert len(stepwise_outputs) == num_chunks and stepwise_outputs[-1].finished
    assert pipeline.vae.decode_inputs == []


def test_stepwise_trajectory_camera_matches_request_mode_under_non_uniform_speed(monkeypatch) -> None:
    """A trajectory that slows down between blocks must condition every stepwise
    chunk exactly as request mode does. Request mode embeds the whole trajectory
    once; embedding per chunk would re-normalize each block's framewise
    translations by its own largest step and present a slow block as full-speed
    motion."""
    from vllm_omni.diffusion.models.lingbot_world import camera as camera_module

    module = _load_pipeline_module()
    # The fixture stubs the ray embedding with a pose-blind ramp; parity needs
    # the real geometry so a scale mismatch actually shows up.
    monkeypatch.setattr(module, "build_plucker_embedding", camera_module.build_plucker_embedding)
    transformer = _RecordingTransformer()
    pipeline = _pipeline(module, transformer=transformer)
    pipeline._ar_height = 16
    pipeline._ar_width = 16

    # Six latent frames -> two 3-frame blocks. The first block moves at full
    # speed and everything after it at a tenth of that, so a per-chunk
    # normalization would rescale the second block by 10x.
    steps = torch.tensor([1.0] * 3 + [0.1] * 18)
    poses = torch.eye(4).repeat(21, 1, 1)
    poses[:, 2, 3] = torch.cumsum(steps, dim=0) - steps[0]
    trajectory = _CameraTrajectory(
        poses=poses,
        intrinsics=torch.tensor([[100.0, 100.0, 8.0, 8.0]]).repeat(21, 1),
    )

    sampling = _SamplingParams(num_frames=21)
    sampling.extra_args["_lingbot_camera_trajectory"] = trajectory
    pipeline(_request(sampling=sampling))
    request_cameras = [call["camera_hidden_states"] for call in transformer.calls]
    transformer.calls.clear()

    state = _stepwise_state(num_frames=21)
    state.sampling.extra_args["_lingbot_camera_trajectory"] = trajectory
    state.sampling.extra_args.pop("_lingbot_camera_action_script")
    with pipeline.bind_ar_diffusion_state(state.request_id, _FakeARState(state.request_id)):
        _run_stepwise(pipeline, state)
    stepwise_cameras = [call["camera_hidden_states"] for call in transformer.calls]

    # Two blocks x (four probes + one commit), in the same order on both paths.
    assert len(request_cameras) == len(stepwise_cameras) == 10
    for stepwise, request in zip(stepwise_cameras, request_cameras, strict=True):
        torch.testing.assert_close(stepwise, request)


def test_stepwise_emits_latents_without_touching_the_vae() -> None:
    """Latent mode must stay latent: streaming decode is opt-in per request."""
    module = _load_pipeline_module()
    pipeline = _pipeline(module, transformer=_RecordingTransformer())
    pipeline._ar_height = 16
    pipeline._ar_width = 16
    state = _stepwise_state(num_frames=21)
    fake = _FakeARState(state.request_id)

    with pipeline.bind_ar_diffusion_state(state.request_id, fake):
        outputs = _run_stepwise(pipeline, state)

    assert [set(output.output["payload"]) for output in outputs] == [{"latents"}, {"latents"}]
    assert pipeline.vae.decode_inputs == []


def _capture_video_processor(monkeypatch) -> list[tuple[tuple[int, ...], str]]:
    """Record what reaches postprocess_video, and name each chunk's output."""
    import diffusers.video_processor as video_processor_module

    processed: list[tuple[tuple[int, ...], str]] = []

    class VideoProcessor:
        def __init__(self, *, vae_scale_factor):
            assert vae_scale_factor == 8

        def postprocess_video(self, video, *, output_type):
            processed.append((tuple(video.shape), output_type))
            return f"frames-{len(processed)}"

    monkeypatch.setattr(video_processor_module, "VideoProcessor", VideoProcessor)
    return processed


def _streaming_pipeline(module):
    pipeline = _pipeline(module, transformer=_RecordingTransformer())
    pipeline._ar_height = 16
    pipeline._ar_width = 16
    return pipeline


def test_stepwise_decodes_each_chunk_for_streaming_consumers(monkeypatch) -> None:
    """A non-latent request must stream pixels under the primary "video" key."""
    module = _load_pipeline_module()
    processed = _capture_video_processor(monkeypatch)
    pipeline = _streaming_pipeline(module)
    state = _stepwise_state(num_frames=21)
    state.sampling.output_type = "np"
    fake = _FakeARState(state.request_id)

    with pipeline.bind_ar_diffusion_state(state.request_id, fake):
        outputs = _run_stepwise(pipeline, state)

    assert [output_type for _, output_type in processed] == ["np", "np"]
    assert [output.output["payload"] for output in outputs] == [{"video": "frames-1"}, {"video": "frames-2"}]
    # Identity metadata survives the switch to pixel output.
    assert [output.output["metadata"]["ar_diffusion"]["chunk_index"] for output in outputs] == [0, 1]


def test_stepwise_chunks_continue_one_sessions_temporal_decode(monkeypatch) -> None:
    """Chunk N + 1 must continue chunk N instead of restarting the decoder.

    Only the session's opening latent frame collapses to a single raw frame,
    so the streamed timeline carries the same frame count an offline decode of
    the same latents would: 9 frames, then 12.
    """
    module = _load_pipeline_module()
    processed = _capture_video_processor(monkeypatch)
    pipeline = _streaming_pipeline(module)
    state = _stepwise_state(num_frames=21)
    state.sampling.output_type = "np"
    fake = _FakeARState(state.request_id)

    with pipeline.bind_ar_diffusion_state(state.request_id, fake):
        _run_stepwise(pipeline, state)

    assert [shape for shape, _ in processed] == [(1, 3, 9, 16, 16), (1, 3, 12, 16, 16)]
    # One opening frame for the session, not one per chunk.
    assert pipeline.vae.decoder.first_chunk_flags == [True] + [False] * 5
    # The whole-clip decode path is not involved, and the module-owned cache
    # the shared VAE keeps for it is untouched.
    assert pipeline.vae.decode_inputs == []
    assert pipeline.vae._feat_map == ["module-owned"]


def test_streaming_decode_state_is_owned_by_the_session(monkeypatch) -> None:
    """Decoder state is keyed by request id and released with the AR session."""
    module = _load_pipeline_module()
    _capture_video_processor(monkeypatch)
    pipeline = _streaming_pipeline(module)
    state = _stepwise_state(request_id="req-stream", num_frames=21)
    state.sampling.output_type = "np"
    fake = _FakeARState(state.request_id)

    with pipeline.bind_ar_diffusion_state(state.request_id, fake):
        _run_stepwise(pipeline, state)

    decode_state = pipeline._streaming_decode_states["req-stream"]
    assert decode_state.session_id == "req-stream"
    assert decode_state.frames_decoded == 6
    assert decode_state.nbytes() > 0

    pipeline.close_ar_diffusion_session("req-stream")
    assert "req-stream" not in pipeline._streaming_decode_states
    # Release drops the cache itself, not just the pipeline's reference to it.
    assert decode_state.nbytes() == 0
    assert decode_state.frames_decoded == 0


def test_streaming_decode_state_is_dropped_on_session_reset(monkeypatch) -> None:
    """A reset session must not resume the temporal context it just abandoned."""
    module = _load_pipeline_module()
    _capture_video_processor(monkeypatch)
    pipeline = _streaming_pipeline(module)
    state = _stepwise_state(request_id="req-reset", num_frames=21)
    state.sampling.output_type = "np"
    fake = _FakeARState(state.request_id)

    with pipeline.bind_ar_diffusion_state(state.request_id, fake):
        _run_stepwise(pipeline, state)

    pipeline.reset_ar_diffusion_session("req-reset")
    assert pipeline._streaming_decode_states == {}
    # Releasing a session that never decoded is a no-op, not an error.
    pipeline.reset_ar_diffusion_session("req-reset")


def test_streaming_decode_keeps_interleaved_sessions_isolated(monkeypatch) -> None:
    """Two sessions ticking alternately must not share one temporal cache.

    Running one rollout to completion before the other starts would only show
    that the dict is keyed by request id. The failure worth catching is chunk N
    of one session advancing the other's cache mid-rollout, which needs the
    interleaving the scheduler actually produces.
    """
    module = _load_pipeline_module()
    processed = _capture_video_processor(monkeypatch)
    pipeline = _streaming_pipeline(module)
    states = []
    for request_id in ("req-a", "req-b"):
        state = _stepwise_state(request_id=request_id, num_frames=21)
        state.sampling.output_type = "np"
        states.append(state)
    runs = [_stepwise_chunks(pipeline, state, _FakeARState(state.request_id)) for state in states]

    chunks: list[tuple[str, int]] = []
    while runs:
        for run in list(runs):
            try:
                output = next(run)
            except StopIteration:
                runs.remove(run)
                continue
            metadata = output.output["metadata"]["ar_diffusion"]
            chunks.append((metadata["session_id"], metadata["chunk_index"]))

    # A-chunk0, B-chunk0, A-chunk1, B-chunk1: the sessions really are interleaved.
    assert chunks == [("req-a", 0), ("req-b", 0), ("req-a", 1), ("req-b", 1)]

    caches = pipeline._streaming_decode_states
    assert set(caches) == {"req-a", "req-b"}
    assert caches["req-a"].feat_map is not caches["req-b"].feat_map
    # Every frame of a session was decoded through that session's own cache,
    # three frames per chunk, and each session opened exactly one rollout.
    cache_a, cache_b = (id(caches["req-a"].feat_map), id(caches["req-b"].feat_map))
    assert pipeline.vae.decoder.cache_ids == [cache_a] * 3 + [cache_b] * 3 + [cache_a] * 3 + [cache_b] * 3
    assert pipeline.vae.decoder.first_chunk_flags == (
        [True, False, False] + [True, False, False] + [False] * 3 + [False] * 3
    )
    assert [shape for shape, _ in processed] == [
        (1, 3, 9, 16, 16),
        (1, 3, 9, 16, 16),
        (1, 3, 12, 16, 16),
        (1, 3, 12, 16, 16),
    ]


def test_streaming_decode_failure_drops_the_half_advanced_cache(monkeypatch) -> None:
    """A failed chunk must not leave a cache a later chunk would continue from."""
    module = _load_pipeline_module()
    _capture_video_processor(monkeypatch)
    pipeline = _streaming_pipeline(module)
    state = _stepwise_state(request_id="req-boom", num_frames=21)
    state.sampling.output_type = "np"

    def explode(*args, **kwargs):
        raise RuntimeError("decoder blew up")

    pipeline.vae.decoder = explode

    with pytest.raises(RuntimeError, match="decoder blew up"):
        with pipeline.bind_ar_diffusion_state(state.request_id, _FakeARState(state.request_id)):
            _run_stepwise(pipeline, state)

    assert pipeline._streaming_decode_states == {}


def _enable_vae_tiling(pipeline, *, tile_sample_min: int) -> None:
    """Configure the stub VAE the way registry.py configures the real one."""
    pipeline.vae.use_tiling = True
    pipeline.vae.spatial_compression_ratio = 8
    pipeline.vae.tile_sample_min_height = tile_sample_min
    pipeline.vae.tile_sample_min_width = tile_sample_min


def test_streaming_decode_receives_rescaled_latents(monkeypatch) -> None:
    """The streaming branch must invert the checkpoint's latent statistics.

    Nothing downstream can catch a miss here: the pixels stay finite and
    correctly shaped, only wrongly scaled. Request mode and the fallback path
    observe the rescale through ``vae.decode``'s recorded input, which streaming
    never reaches, so this asserts on what ``post_quant_conv`` received.
    """
    module = _load_pipeline_module()
    _capture_video_processor(monkeypatch)
    pipeline = _streaming_pipeline(module)
    state = _stepwise_state(num_frames=21)
    state.sampling.output_type = "np"

    with pipeline.bind_ar_diffusion_state(state.request_id, _FakeARState(state.request_id)):
        pipeline.prepare_encode(state)
        while not state.chunk_denoise_completed:
            pipeline.step_scheduler(state, pipeline.denoise_step(None, states=[state]))
        model_space = state.latents.clone()
        pipeline.post_decode(state)

    shape = (1, -1, 1, 1, 1)
    latent_mean = torch.as_tensor(pipeline.vae.config.latents_mean, dtype=model_space.dtype).view(*shape)
    latent_std = torch.as_tensor(pipeline.vae.config.latents_std, dtype=model_space.dtype).view(*shape)
    expected = (model_space * latent_std + latent_mean).to(dtype=pipeline.vae.dtype)

    recorded = torch.cat(pipeline.vae.post_quant_inputs, dim=2)
    assert recorded.shape == expected.shape, "one recorded frame per latent frame in the chunk"
    torch.testing.assert_close(recorded, expected, rtol=0, atol=0)
    # And the rescale is not a no-op on this fixture, so the assertion bites.
    assert not torch.equal(expected, model_space.to(dtype=pipeline.vae.dtype))


def test_streaming_decode_is_reported_to_the_profiler(monkeypatch) -> None:
    """A streamed rollout must report decode time, not zero.

    The profiler wraps ``vae.decode``, which streaming never calls, so without
    its own stage the one stage this path changes would be invisible -- and a
    perf regression in the streaming decoder would be too.
    """
    module = _load_pipeline_module()
    _capture_video_processor(monkeypatch)
    pipeline = _pipeline(
        module,
        transformer=_RecordingTransformer(),
        od_config=_od_config(
            enable_diffusion_pipeline_profiler=True,
        ),
    )
    pipeline._ar_height = 16
    pipeline._ar_width = 16
    state = _stepwise_state(num_frames=21)
    state.sampling.output_type = "np"

    with pipeline.bind_ar_diffusion_state(state.request_id, _FakeARState(state.request_id)):
        outputs = _run_stepwise(pipeline, state)

    stage = f"{type(pipeline).__name__}._streaming_decode_chunk"
    durations = pipeline.stage_durations
    assert stage in durations, f"streamed decode is unprofiled; stages seen: {sorted(durations)}"
    assert durations[stage] > 0.0
    assert durations[f"{type(pipeline).__name__}._prepare_condition_chunk"] > 0.0
    # The stage reaches the consumer on every streamed chunk, not just at the end.
    assert all(stage in (output.stage_durations or {}) for output in outputs)


def test_streaming_decode_falls_back_when_the_vae_would_tile(monkeypatch) -> None:
    """Tiled decode drives its own cache, so a tiling shape keeps per-chunk decode."""
    module = _load_pipeline_module()
    processed = _capture_video_processor(monkeypatch)
    pipeline = _streaming_pipeline(module)
    # A latent of 2x2 exceeds a one-latent-cell tile, so vae.decode would tile.
    _enable_vae_tiling(pipeline, tile_sample_min=8)
    outputs = [
        pipeline._decode_chunk_to_pixels(torch.zeros(1, 16, 3, 2, 2), output_type="np", session_id="tiled")
        for _ in range(2)
    ]

    # One whole-clip decode per AR block, each seeing only that block's frames.
    assert [tuple(latents.shape) for latents in pipeline.vae.decode_inputs] == [(1, 16, 3, 2, 2)] * 2
    assert pipeline.vae.decoder.first_chunk_flags == []
    assert pipeline._streaming_decode_states == {}
    # Nine frames per chunk instead of 9 then 12: the frame loss being removed.
    assert [shape for shape, _ in processed] == [(1, 3, 9, 16, 16), (1, 3, 9, 16, 16)]
    assert outputs == ["frames-1", "frames-2"]


def test_streaming_decode_survives_tiling_enabled_below_the_tile_threshold(monkeypatch) -> None:
    """``use_tiling`` alone does not tile, so such a shape must still stream.

    ``AutoencoderKLWan._decode`` tiles only when the latent exceeds a tile. A
    gate on the flag alone would give up streaming -- and a frame per block --
    on a configuration that would never have tiled.
    """
    module = _load_pipeline_module()
    processed = _capture_video_processor(monkeypatch)
    pipeline = _streaming_pipeline(module)
    # A 2x2 latent is within a tile this size, so vae.decode would not tile.
    _enable_vae_tiling(pipeline, tile_sample_min=256)
    state = _stepwise_state(num_frames=21)
    state.sampling.output_type = "np"

    with pipeline.bind_ar_diffusion_state(state.request_id, _FakeARState(state.request_id)):
        _run_stepwise(pipeline, state)

    assert pipeline.vae.decode_inputs == []
    assert pipeline.vae.decoder.first_chunk_flags == [True] + [False] * 5
    assert [shape for shape, _ in processed] == [(1, 3, 9, 16, 16), (1, 3, 12, 16, 16)]


def test_registry_and_model_exports_resolve_official_pipeline_class_name() -> None:
    module = _load_pipeline_module()
    resolved, entry, cache_acceleration_disabled, preprocess_name, preprocess = _resolve_pipeline_through_real_registry(
        module
    )

    assert entry == ("lingbot_world", "pipeline", "LingBotWorldCausalDMDPipeline")
    assert resolved is module.LingBotWorldCausalDMDPipeline
    assert cache_acceleration_disabled
    assert preprocess_name == "get_lingbot_world_pre_process_func"
    request = OmniDiffusionRequest(prompt=_prompt(), sampling_params=_SamplingParams(), request_id="req-1")
    assert preprocess(request) is request
    assert isinstance(request.sampling_params.extra_args["_lingbot_camera_trajectory"], _CameraTrajectory)
    assert module.LingBotWorldCausalDMDPipeline.__name__ == "LingBotWorldCausalDMDPipeline"
    lingbot_init = _LINGBOT_INIT_PATH.read_text()
    assert "from .pipeline import" in lingbot_init
    assert '"LingBotWorldCausalDMDPipeline"' in lingbot_init
    assert '"CausalLingBotWorldTransformer3DModel"' in lingbot_init


# ---------------------------------------------------------------------------
# a prompt is encoded once, whichever path asks for it
# ---------------------------------------------------------------------------
def _counting_text_encoder(pipeline) -> list[str]:
    """Give ``pipeline`` its real encode_prompt over stubs that record each encode."""
    del pipeline.encode_prompt  # _pipeline() replaces it; these tests need the real one
    encoded: list[str] = []

    def tokenizer(texts, **kwargs):
        del kwargs
        encoded.extend(texts)
        return SimpleNamespace(
            input_ids=torch.arange(512).view(1, 512),
            attention_mask=torch.ones(1, 512, dtype=torch.long),
        )

    def text_encoder(input_ids, attention_mask):
        del input_ids, attention_mask
        return SimpleNamespace(last_hidden_state=torch.full((1, 512, 8), float(len(encoded))))

    pipeline.tokenizer = tokenizer
    pipeline.text_encoder = text_encoder
    return encoded


def test_the_text_encoder_runs_once_per_distinct_prompt() -> None:
    module = _load_pipeline_module()
    pipeline = _pipeline(module)
    encoded = _counting_text_encoder(pipeline)

    def encode(prompt, *, length=512, dtype=torch.float32):
        return pipeline.encode_prompt(prompt, max_sequence_length=length, dtype=dtype)

    first = encode("move through the room")
    assert encode("move through the room") is first
    # The key is the text the tokenizer actually sees, so spacing alone is not new.
    assert encode("  move   through the room ") is first
    assert encoded == ["move through the room"]

    encode("enter the snowy valley")
    # dtype and sequence length change the tensor, so each is its own entry.
    encode("move through the room", dtype=torch.float16)
    encode("move through the room", length=256)
    assert encoded == [
        "move through the room",
        "enter the snowy valley",
        "move through the room",
        "move through the room",
    ]


def test_the_prompt_encode_cache_is_bounded() -> None:
    module = _load_pipeline_module()
    pipeline = _pipeline(module)
    encoded = _counting_text_encoder(pipeline)
    size = module._PROMPT_EMBEDS_CACHE_SIZE

    for index in range(size + 1):
        pipeline.encode_prompt(f"prompt {index}", max_sequence_length=512, dtype=torch.float32)
    assert len(pipeline._prompt_embeds_cache) == size

    # The oldest entry was the one evicted, so it is the one encoded again.
    pipeline.encode_prompt("prompt 0", max_sequence_length=512, dtype=torch.float32)
    pipeline.encode_prompt(f"prompt {size}", max_sequence_length=512, dtype=torch.float32)
    assert encoded == [f"prompt {index}" for index in range(size + 1)] + ["prompt 0"]


def test_realtime_ticks_reuse_the_encode_and_still_invalidate_on_a_prompt_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_pipeline_module()
    pipeline = _pipeline(module)
    pipeline._ar_height = 16
    pipeline._ar_width = 16
    pipeline._ar_diffusion_kv_state = object()
    encoded = _counting_text_encoder(pipeline)
    invalidations: list[bool] = []

    def ar_text_caches(prompt_embeds, *, invalidate):
        del prompt_embeds
        invalidations.append(invalidate)
        return [SimpleNamespace()]

    monkeypatch.setattr(pipeline, "_ar_text_caches", ar_text_caches)
    monkeypatch.setattr(
        pipeline,
        "_generate_block",
        lambda **kwargs: torch.randn((1, 16, 3, 2, 2), generator=kwargs["generator"]),
    )

    for chunk_index in range(3):
        pipeline(_request(sampling=_SamplingParams(extra_args=_tick_extra_args(chunk_index=chunk_index))))
    switched = _prompt()
    switched["prompt"] = "enter the snowy valley"
    pipeline(
        _request(
            sampling=_SamplingParams(extra_args=_tick_extra_args(chunk_index=3, prompt=switched["prompt"])),
            prompt=switched,
        )
    )

    assert encoded == ["move through the room", "enter the snowy valley"]
    # A new prompt must still drop the cross-attention K/V built from the old one.
    assert invalidations == [False, False, False, True]


def test_stepwise_requests_with_the_same_prompt_share_one_encode() -> None:
    module = _load_pipeline_module()
    pipeline = _pipeline(module, transformer=_RecordingTransformer())
    pipeline._ar_height = 16
    pipeline._ar_width = 16
    encoded = _counting_text_encoder(pipeline)

    states = [_stepwise_state(request_id=request_id, num_frames=21) for request_id in ("req-a", "req-b")]
    for state in states:
        with pipeline.bind_ar_diffusion_state(state.request_id, _FakeARState(state.request_id)):
            pipeline.prepare_encode(state)

    assert encoded == ["move through the room"]
    assert states[0].prompt_embeds is states[1].prompt_embeds
