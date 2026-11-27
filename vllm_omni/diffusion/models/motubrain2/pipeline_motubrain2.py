# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Diffusion serving adapter for the external Motubrain 2.0 policy."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from copy import copy
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from vllm.logger import init_logger
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from .obs_adapter import build_observation, camera_sources
from .state_motubrain2 import SessionStore, build_rtc, request_rtc, validate_rtc_config

if TYPE_CHECKING:
    from motubrain2.inference.policy import RawPolicy

logger = init_logger(__name__)


@contextmanager
def _native_sdpa(device: torch.device) -> Iterator[None]:
    # vLLM disables cuDNN SDPA at CUDA platform initialization. Motubrain's
    # native encoders and denoiser rely on PyTorch's default kernel selection.
    if device.type != "cuda" or torch.backends.cuda.cudnn_sdp_enabled():
        yield
        return
    torch.backends.cuda.enable_cudnn_sdp(True)
    try:
        yield
    finally:
        torch.backends.cuda.enable_cudnn_sdp(False)


def _check_native_dit_cache(args: SimpleNamespace) -> None:
    enabled = args.dit_cache_enabled
    if enabled is not None and type(enabled) is not bool:
        raise ValueError("dit_cache_enabled must be true, false or null.")
    try:
        from motubrain2.inference.dit_cache import DiTCacheConfig
    except ImportError as exc:
        if exc.name != "motubrain2.inference.dit_cache":
            raise
        DiTCacheConfig = None
    if hasattr(DiTCacheConfig, "for_recipe"):
        return  # RawPolicy resolves defaults from the saved recipe, before serving overrides.

    if enabled is False:
        # Older cache implementations enable solely by the presence of thresholds.
        args.dit_cache_threshold = args.dit_cache_threshold_2 = None
        return
    if args.dit_cache_threshold is not None or args.dit_cache_threshold_2 is not None:
        if DiTCacheConfig is not None:
            return  # A threshold-aware native version can honor this explicit setting.
    elif enabled is None:
        from motubrain2.inference.sampling import sampling_options

        recipe = sampling_options(args.model)
        if (recipe["sampler"], recipe["steps"], recipe["video_shift"], recipe["action_shift"]) != ("euler", 30, 6, 1):
            return
    raise ImportError("Install a Motubrain revision with native DiT cache support for Base/Distill defaults.")


def _load_policy(config: Mapping[str, Any], model_root: str, device: torch.device, dtype: torch.dtype) -> RawPolicy:
    """Load the external policy and keep its CPU-side controller transforms on the host."""
    try:
        from motubrain2.inference.policy import RawPolicy
    except ModuleNotFoundError as exc:
        if exc.name != "motubrain2":
            raise
        raise ImportError("Install the Motubrain 2.0 inference package to serve Motubrain2Pipeline.") from exc

    root = Path(model_root)
    bundle = Path(config.get("bundle") or root / "policy")
    embodiment = config.get("embodiment")
    profile = config.get("profile")
    if profile is not None and embodiment is not None:
        raise ValueError("Choose either model_config.profile or model_config.embodiment.")
    if embodiment is None:
        profile = Path(profile or root / "profile.json")
    files = [bundle] if bundle.is_file() else [bundle / "config.json", bundle / "model.safetensors"]
    if profile is not None:
        files.append(profile)
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Motubrain serving bundle is missing {path}")
    for name in ("wan", "text_encoder"):
        if not config.get(name):
            raise ValueError(f"Motubrain requires model_config.{name} for its frozen encoders.")
    encoder_device = torch.device(config.get("encoder_device") or device)
    if encoder_device.type == "cuda" and encoder_device.index is not None:
        if encoder_device.index >= torch.accelerator.device_count():
            raise ValueError(f"Motubrain encoder_device {encoder_device} is not visible to this worker.")

    args = SimpleNamespace(
        model=bundle,
        profile=profile,
        embodiment=embodiment,
        device=device,
        encoder_device=encoder_device,
        wan=config["wan"],
        wan_subfolder=config.get("wan_subfolder"),
        wan_config=config.get("wan_config"),
        text_encoder=config["text_encoder"],
        da3=config.get("da3"),
        da3_config=config.get("da3_config"),
        action_stats=config.get("action_stats"),
        sampler=config.get("sampler"),
        steps=config.get("steps"),
        video_shift=config.get("video_shift"),
        action_shift=config.get("action_shift"),
        dit_cache_enabled=config.get("dit_cache_enabled"),
        dit_cache_threshold=config.get("dit_cache_threshold"),
        dit_cache_threshold_2=config.get("dit_cache_threshold_2"),
        dit_cache_skip_steps=config.get("dit_cache_skip_steps", 4),
        dit_cache_skip_steps_2=config.get("dit_cache_skip_steps_2", 2),
        dit_cache_damping=config.get("dit_cache_damping", 0.25),
        bf16=dtype == torch.bfloat16,
        video_mxfp8=config.get("video_mxfp8", False),
        video_mxfp8_overlay=config.get("video_mxfp8_overlay"),
        seed=0,
        compile=config.get("compile", False),
        multistream=config.get("multistream", "off"),
        async_io=config.get("async_io", False),
    )
    _check_native_dit_cache(args)
    # The diffusion loader sets the default device and dtype around pipeline
    # construction. Native preprocessing owns several CPU/FP32 tensors.
    with torch.device("cpu"), set_default_torch_dtype(torch.float32):
        try:
            policy = RawPolicy.from_options(args)
        except ModuleNotFoundError as exc:
            if not exc.name or exc.name.split(".")[0] != "depth_anything_3":
                raise
            raise ImportError("This Motubrain profile requires the Depth Anything 3 DINOv2 encoder.") from exc
        # Native loading preserves the checkpoint dtype when bf16=False. An
        # explicit Omni FP32 deployment also promotes lower-precision weights.
        if dtype == torch.float32 and any(
            p.is_floating_point() and p.dtype != dtype for p in policy.model.parameters()
        ):
            policy.model.float()
            if policy.streams is not None:
                from motubrain2.inference.streams import InferenceStreams

                # The scheduler registers each parameter's final CUDA storage.
                with torch.inference_mode():
                    policy.streams = InferenceStreams(policy.model, device, args.multistream)
        return policy


class Motubrain2Pipeline(nn.Module):
    """Produce a single controller action chunk for one observation."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        config = dict(od_config.model_config or {})
        if od_config.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"Motubrain supports float32 and bfloat16 denoiser weights, got {od_config.dtype}.")
        if od_config.quantization_config:
            raise ValueError("Use model_config.video_mxfp8 for native Motubrain weight quantization.")
        if config.get("video_mxfp8") and od_config.dtype != torch.bfloat16:
            raise ValueError("Motubrain video MXFP8 requires a bfloat16 denoiser.")
        if od_config.parallel_config.world_size != 1:
            raise ValueError("Motubrain2Pipeline requires a single diffusion worker without model parallelism.")
        if config.get("compile") and od_config.enforce_eager:
            raise ValueError("Motubrain model_config.compile requires enforce_eager=false.")

        self.policy = _load_policy(config, od_config.model, get_local_device(), od_config.dtype)
        self.model = self.policy.model
        self.processor = self.policy.processor
        self.horizon = self.processor.action_steps
        self.action_dim = self.processor.actions.wire_dim
        self.camera_keys = camera_sources(self.processor.views)
        fields = self.processor.profile["action"]["fields"]
        self._needs_state = any(
            field["kind"] in {"relative", "eef_relative", "eef_camera"} and field.get("model_used", True)
            for field in fields
        )
        self._holds_state = any(not field.get("model_used", True) for field in fields)
        self._needs_extrinsic = any(field["kind"] == "eef_camera" and field.get("model_used", True) for field in fields)
        self.default_num_inference_steps = self.policy.sampling["steps"]
        self._validate_handshake(config.get("motubrain2_metadata"))

        self.rtc_config = validate_rtc_config(config.get("rtc"), self.horizon)
        self.sessions = SessionStore(config.get("max_sessions", 64))
        self._lock = threading.Lock()
        cache_size = config.get("text_cache_size", 64)
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("text_cache_size must be a nonnegative integer.")
        if cache_size:
            # RawProcessor only reads the frozen encoder's result.
            self.processor.text.encode = lru_cache(maxsize=cache_size)(self.processor.text.encode)
        logger.info(
            "Loaded Motubrain policy for %s with %d steps, action shape (%d, %d) and cameras %s.",
            self.policy.device,
            self.default_num_inference_steps,
            self.horizon,
            self.action_dim,
            self.camera_keys,
        )

    def _validate_handshake(self, handshake: Mapping[str, Any] | None) -> None:
        if handshake is None:
            return
        fields = self.processor.profile["action"]["fields"]
        if any(field["kind"].startswith("eef_") for field in fields):
            has_joints = any(
                field["kind"] == "relative" or (field["kind"] == "linear" and "gripper" not in field["name"])
                for field in fields
            )
            action_space = "cartesian_pose_xyzw_joint_position_gripper" if has_joints else "cartesian_pose_xyzw_gripper"
        else:
            action_space = "joint_position"
        expected = {
            "action_horizon": self.horizon,
            "action_dim": self.action_dim,
            "camera_sources": self.camera_keys,
            "action_space": action_space,
        }
        for field, value in expected.items():
            if handshake.get(field) != value:
                raise ValueError(f"motubrain2_metadata.{field} must match the robot profile: expected {value!r}.")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]] = ()) -> None:
        """The native policy already loaded and validated its own checkpoints."""
        if next(iter(weights), None) is not None:
            raise ValueError("Motubrain loads its native bundle and cannot accept additional weight sources.")

    def drop_session_state(self, session_id: str) -> None:
        with self._lock:
            self.sessions.reset(session_id)

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch, **kwargs: Any) -> DiffusionOutput:
        if getattr(req, "num_reqs", 1) != 1:
            return DiffusionOutput(error="Motubrain accepts one observation per forward.", error_status_code=400)
        if req.is_dummy_run():
            return DiffusionOutput(output={"actions": np.zeros((self.horizon, self.action_dim), dtype=np.float32)})

        params = req.sampling_params
        extra = params.extra_args or {}
        session_id = extra.get("session_id")
        # Anonymous WebSocket connections all use the same "default" key.
        if not isinstance(session_id, str) or session_id.strip() in {"", "default"}:
            session_id = None
        if self.rtc_config and session_id is None:
            return DiffusionOutput(
                error="Motubrain RTC requires a unique session_id for each rollout. 'default' is reserved.",
                error_status_code=400,
            )
        with self._lock:
            # The frontend sends reset once. Even an invalid first observation
            # must invalidate the previous episode's action chunk.
            if extra.get("reset"):
                self.sessions.reset(session_id)
            if "robot_obs" not in extra or extra["robot_obs"] is None:
                return DiffusionOutput(error="Motubrain requires extra_args['robot_obs'].", error_status_code=400)
            return self._predict(extra["robot_obs"], params, session_id)

    def _predict(self, robot_obs: dict, params: OmniDiffusionSamplingParams, session_id: str | None) -> DiffusionOutput:
        steps = params.num_inference_steps
        if steps is not None and (isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1):
            return DiffusionOutput(error="num_inference_steps must be a positive integer.", error_status_code=400)

        # RawPolicy stores its generator and schedule on the policy object. A
        # shallow copy shares all weights and encoders but owns request settings.
        policy = copy(self.policy)
        if steps is not None:
            policy.sampling = {**policy.sampling, "steps": int(steps)}
        extra = params.extra_args or {}
        session = self.sessions.peek(session_id)
        session_seed = extra.get("motubrain2_session_seed")
        generator = params.generator
        if isinstance(generator, list) and len(generator) == 1:
            generator = generator[0]
        if session_seed is not None:
            # Use a private copy so failed calls never advance the committed RNG.
            generator = torch.Generator(device=policy.device).manual_seed(session_seed)
            if session is not None and session.rng_state is not None:
                generator.set_state(session.rng_state)
        elif generator is None:
            generator = torch.Generator(device=policy.device)
            if params.seed is None:
                generator.seed()
            else:
                generator.manual_seed(params.seed)
        if not isinstance(generator, torch.Generator) or generator.device != policy.generator.device:
            return DiffusionOutput(error=f"Motubrain requires one generator on {policy.device}.", error_status_code=400)
        policy.generator = generator

        sparse = self.processor.sparse_history
        try:
            observation = build_observation(
                robot_obs,
                self.processor.views,
                wire_dim=self.action_dim,
                observed_frames=self.processor.observed_frames,
                temporal_stride=self.processor.video.temporal_stride,
                history_frames=None if sparse is None else sparse.rgb_frames,
            )
        except (TypeError, ValueError) as exc:
            return DiffusionOutput(error=str(exc), error_status_code=400)

        native_outputs = robot_obs.get("return_native_outputs", False)
        if not isinstance(native_outputs, (bool, np.bool_)):
            return DiffusionOutput(error="return_native_outputs must be a boolean.", error_status_code=400)
        if robot_obs.get("rtc") is not None:
            try:
                rtc = request_rtc(robot_obs["rtc"], self.action_dim, self.horizon)
            except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
                return DiffusionOutput(error=str(exc), error_status_code=400)
        else:
            rtc = build_rtc(session, self.rtc_config, self.horizon)
        has_state = "state" in observation or (rtc is not None and rtc.anchor is not None)
        if (self._needs_state or (self._holds_state and not native_outputs)) and not has_state:
            return DiffusionOutput(error="This Motubrain action profile requires robot state.", error_status_code=400)
        if self._needs_extrinsic and "camera_extrinsic" not in observation:
            return DiffusionOutput(
                error="This Motubrain action profile requires camera_extrinsic.", error_status_code=400
            )
        # The native denoiser also wraps its local SwiGLU with torch.compile.
        with (
            _native_sdpa(policy.device),
            torch.compiler.set_stance("force_eager") if self.od_config.enforce_eager else nullcontext(),
        ):
            result = policy.predict(observation, rtc=rtc)
        actions = np.asarray(result["decoded_action"], dtype=np.float32)
        valid = np.asarray(result["decoded_valid"])
        shape = (self.horizon, self.action_dim)
        if valid.shape != shape or valid.dtype != np.bool_ or (not native_outputs and not valid.all()):
            return DiffusionOutput(error="Motubrain returned invalid controller coordinates.")
        if actions.shape != shape or not np.isfinite(actions[valid]).all():
            return DiffusionOutput(error=f"Motubrain must return finite valid controller actions shaped {shape}.")
        if native_outputs:
            # The engine forwards custom_output when no regular output or media is set.
            tensors = {key: np.asarray(result[key]) for key in ("video", "action", "decoded_action", "decoded_valid")}
            output = DiffusionOutput(custom_output=tensors)
        else:
            output = DiffusionOutput(output={"actions": actions})
        if session_id is not None and (session_seed is not None or self.rtc_config):
            # A masked native diagnostic can advance the RNG, but cannot seed RTC.
            last_action = actions.copy() if self.rtc_config and valid.all() else None
            rng_state = session.rng_state if session is not None else None
            if session_seed is not None:
                rng_state = generator.get_state()
            if last_action is None and rng_state is None:
                self.sessions.reset(session_id)
            else:
                session = self.sessions.get(session_id)
                session.last_action = last_action
                session.rng_state = rng_state
        return output
