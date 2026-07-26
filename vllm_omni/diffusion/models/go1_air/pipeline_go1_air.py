# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import json
import os
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
    wrap_methods_by_paths,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest

from .config import (
    DEFAULT_INTERNVL_PROCESSOR,
    OBS_IMAGES,
    OBS_STATE,
    OBS_TASK,
    Go1AirConfig,
)
from .model_go1_air import Go1AirPolicy

logger = init_logger(__name__)


def get_go1_air_post_process_func(od_config: OmniDiffusionConfig):
    del od_config

    def post_process_func(x):
        if isinstance(x, dict):
            return x
        # DiffusionEngine forwards this key to OmniRequestOutput.multimodal_output,
        # which is what the OpenPI robot serving layer consumes.
        return {"actions": x, "video": []}

    return post_process_func


def _as_1d_float_tensor(value: Any, *, key: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(np.asarray(value))
    tensor = tensor.flatten().to(dtype=torch.float32)
    if tensor.numel() == 0:
        raise ValueError(f"Go1AirPipeline robot_obs['{key}'] must not be empty.")
    return tensor


def _pad_state(state: torch.Tensor, *, state_dim: int) -> torch.Tensor:
    if state.numel() > state_dim:
        raise ValueError(f"Go1AirPipeline robot state has {state.numel()} dims, expected at most {state_dim}.")
    padded = torch.zeros((state_dim,), dtype=torch.float32)
    padded[: state.numel()] = state
    return padded.unsqueeze(0)


def _normalize_robot_image(value: Any, *, image_size: tuple[int, int]) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(np.asarray(value))
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(
            "Go1AirPipeline robot images must be HWC, CHW, THWC, or TCHW tensors/arrays; "
            f"got shape {tuple(tensor.shape)}."
        )

    if tensor.shape[-1] == 3:
        tensor = tensor.permute(0, 3, 1, 2)
    elif tensor.shape[1] != 3:
        raise ValueError(f"Go1AirPipeline robot image channel dimension must be 3, got shape {tuple(tensor.shape)}.")

    tensor = tensor.contiguous().to(dtype=torch.float32)
    if tensor.max().item() > 1.0:
        tensor = tensor / 255.0

    height, width = image_size
    if tuple(tensor.shape[-2:]) != (height, width):
        tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return tensor.unsqueeze(0)


def _robot_obs_image_items(robot_obs: dict[str, Any]) -> list[tuple[str, Any]]:
    native_prefix = f"{OBS_IMAGES}."
    native_items = [
        (key, value)
        for key, value in robot_obs.items()
        if key.startswith(native_prefix) and not key.endswith("_mask") and value is not None
    ]
    if native_items:
        return sorted(native_items)

    return sorted(
        (key, value)
        for key, value in robot_obs.items()
        if key.startswith("observation/") and "image" in key and value is not None
    )


def build_go1_air_batch_inputs_from_robot_obs(
    robot_obs: dict[str, Any],
    *,
    config: Go1AirConfig,
    device: str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Convert one OpenPI robot observation into GO-1-Air batch_inputs."""

    if OBS_STATE in robot_obs:
        state = _as_1d_float_tensor(robot_obs[OBS_STATE], key=OBS_STATE)
    elif "observation/state" in robot_obs:
        state = _as_1d_float_tensor(robot_obs["observation/state"], key="observation/state")
    else:
        parts = [
            _as_1d_float_tensor(robot_obs[key], key=key)
            for key in (
                "observation/joint_position",
                "observation/cartesian_position",
                "observation/gripper_position",
            )
            if key in robot_obs
        ]
        if not parts:
            raise KeyError(
                "Go1AirPipeline robot_obs must include 'observation.state', 'observation/state', "
                "or at least one OpenPI state key."
            )
        state = torch.cat(parts)

    batch: dict[str, Any] = {
        OBS_STATE: _pad_state(state, state_dim=config.max_state_dim).to(device=device, dtype=dtype),
        OBS_TASK: [str(robot_obs.get(OBS_TASK) or robot_obs.get("prompt") or robot_obs.get("task") or "")],
    }

    image_items = _robot_obs_image_items(robot_obs)
    if not image_items:
        raise KeyError("Go1AirPipeline robot_obs must include at least one observation image.")
    for idx, (_, value) in enumerate(image_items):
        image_key = f"{OBS_IMAGES}.image{idx}"
        images = _normalize_robot_image(value, image_size=config.image_resolution).to(device=device, dtype=dtype)
        batch[image_key] = images
        batch[f"{image_key}_mask"] = torch.ones(
            (1, images.shape[1]),
            device=device,
            dtype=torch.bool,
        )

    if "control_freq" in robot_obs:
        batch["control_freq"] = _as_1d_float_tensor(robot_obs["control_freq"], key="control_freq").to(device=device)

    return batch


def get_go1_air_actions(output: DiffusionOutput) -> Any:
    if isinstance(output.output, dict):
        return output.output.get("actions")
    return output.output


class Go1AirPipeline(nn.Module, DiffusionPipelineProfilerMixin, SupportsComponentDiscovery):
    """GO-1-Air pipeline wrapper for the diffusion-policy implementation."""

    _dit_modules: ClassVar[list[str]] = ["policy.model.action_model"]
    _encoder_modules: ClassVar[list[str]] = ["policy.model.vision_model", "policy.model.language_model"]
    _vae_modules: ClassVar[list[str]] = []
    _resident_modules: ClassVar[list[str]] = [
        "policy.model.mlp1",
        "policy.model.k_proj_layers",
        "policy.model.v_proj_layers",
        "policy.model.state_adaptor",
        "policy.model.action_adaptor",
        "policy.model.time_embedder",
        "policy.model.freq_embedder",
        "policy.model.final_layer",
    ]

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.prefix = prefix
        self.model_dir = od_config.model
        self.config = self._build_config(od_config)
        self._ensure_policy_server_config()
        custom_args = od_config.custom_pipeline_args or {}
        self.strict_load = bool(custom_args.get("strict_load", True))
        self.processor_model_name = str(custom_args.get("processor_model_name", DEFAULT_INTERNVL_PROCESSOR))
        enable_warmup = custom_args.get("enable_warmup")
        self.enable_warmup = bool(enable_warmup) if isinstance(enable_warmup, bool) else False

        self.setup_diffusion_pipeline_profiler(
            profiler_targets=["forward"],
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler,
        )
        self.policy = self._initialize_policy()
        self._setup_policy_profiler_targets()
        if self.enable_warmup:
            self._warmup()

    def _build_config(self, od_config: OmniDiffusionConfig) -> Go1AirConfig:
        config_dict = self._load_config_dict(od_config)
        config = Go1AirConfig.from_model_config(config_dict)

        custom_args = od_config.custom_pipeline_args or {}
        device = custom_args.get("device")
        if isinstance(device, str):
            config.device = device

        dtype = custom_args.get("dtype")
        if isinstance(dtype, str):
            config.dtype = dtype
        elif od_config.dtype is not None:
            config.dtype = str(od_config.dtype).split(".")[-1]

        compile_model = custom_args.get("compile_model")
        if isinstance(compile_model, bool):
            config.compile_model = compile_model

        attn_implementation = custom_args.get("attn_implementation")
        if isinstance(attn_implementation, str):
            config.attn_implementation = attn_implementation

        enable_regional_compile = custom_args.get("enable_regional_compile")
        if isinstance(enable_regional_compile, bool):
            config.enable_regional_compile = enable_regional_compile

        regional_compile_dynamic = custom_args.get("regional_compile_dynamic")
        if isinstance(regional_compile_dynamic, bool):
            config.regional_compile_dynamic = regional_compile_dynamic

        return config

    def _ensure_policy_server_config(self) -> None:
        self.od_config.model_config = dict(self.od_config.model_config or {})
        policy_server_config = dict(self.config.policy_server_config)
        policy_server_config["image_resolution"] = list(self.config.image_resolution)
        policy_server_config["action_horizon"] = self.config.chunk_size
        policy_server_config["action_dim"] = self.config.max_action_dim
        self.od_config.model_config.setdefault("policy_server_config", policy_server_config)

    def _load_config_dict(self, od_config: OmniDiffusionConfig) -> dict[str, Any]:
        if od_config.model_config:
            return dict(od_config.model_config)

        model_path = od_config.model
        if not model_path:
            return {}

        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            logger.info("Go1AirPipeline config.json not found under %s; using defaults.", model_path)
            return {}

        with open(config_path, encoding="utf-8") as f:
            return json.load(f)

    def has_real_checkpoint(self) -> bool:
        if not self.model_dir:
            return False
        # Sharded weights ship as model-XXXXX-of-YYYYY.safetensors plus an index.
        index_path = os.path.join(self.model_dir, "model.safetensors.index.json")
        single_path = os.path.join(self.model_dir, "model.safetensors")
        return os.path.exists(index_path) or os.path.exists(single_path)

    def runtime_mode(self) -> str:
        return "real_checkpoint_loaded" if self.has_real_checkpoint() else "no_checkpoint_policy"

    def _setup_policy_profiler_targets(self) -> None:
        if not self.od_config.enable_diffusion_pipeline_profiler:
            return

        wrap_methods_by_paths(
            self,
            [
                "policy.model.sample_actions",
                "policy.model.encode_vision",
                "policy.model.encode_prefix",
            ],
        )

    def _apply_policy_optimizations(self, policy: Go1AirPolicy) -> None:
        policy.model.set_attention_implementation(self.config.attn_implementation)
        if not self.config.enable_regional_compile:
            return

        compile_targets = [
            "vision_model",
            "language_model",
            "action_model",
        ]

        for path in compile_targets:
            current = policy.model
            for part in path.split("."):
                current = getattr(current, part, None)
                if current is None:
                    break
            if current is None:
                continue
            try:
                regionally_compile(current, dynamic=self.config.regional_compile_dynamic)
                logger.info("Go1AirPipeline regional compile applied to %s", path)
            except Exception as exc:
                logger.warning("Go1AirPipeline regional compile failed for %s: %s", path, exc)

    def _initialize_policy(self) -> Go1AirPolicy:
        if self.has_real_checkpoint():
            logger.info("Loading GO-1-Air weights from %s", self.model_dir)
            policy = Go1AirPolicy.from_pretrained(
                self.model_dir,
                config=self.config,
                processor_model_name=self.processor_model_name,
                strict=self.strict_load,
            )
        else:
            logger.info("Initializing GO-1-Air policy without checkpoint weights.")
            policy = Go1AirPolicy(
                self.config,
                processor_model_name=self.processor_model_name,
            )

        policy.to(self.config.device)
        policy.to(getattr(torch, self.config.dtype))
        policy.eval()
        self._apply_policy_optimizations(policy)
        return policy

    def _build_fake_batch_inputs(self) -> dict[str, torch.Tensor]:
        device = torch.device(self.config.device)
        dtype = getattr(torch, self.config.dtype)
        history = 1
        channels = 3
        h, w = self.config.image_resolution
        fake_image = torch.zeros((1, history, channels, h, w), device=device, dtype=dtype)
        fake_mask = torch.ones((1,), device=device, dtype=torch.bool)
        return {
            OBS_STATE: torch.zeros(
                (1, self.config.max_state_dim),
                device=device,
                dtype=dtype,
            ),
            OBS_TASK: [""],
            f"{OBS_IMAGES}.image0": fake_image.clone(),
            f"{OBS_IMAGES}.image0_mask": fake_mask.clone(),
        }

    def _warmup(self) -> None:
        logger.info("Go1AirPipeline warmup started")
        try:
            batch_inputs = self._build_fake_batch_inputs()
            noise = torch.zeros(
                (1, self.config.chunk_size, self.config.max_action_dim),
                device=self.config.device,
                dtype=torch.float32,
            )
            with torch.inference_mode():
                self.policy.forward(batch_inputs, noise=noise)
        except Exception as exc:
            logger.warning("Go1AirPipeline warmup failed: %s", exc)
            return
        logger.info("Go1AirPipeline warmup finished")

    def _predict_actions(
        self,
        batch_inputs: dict[str, Any],
        *,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logger.debug("Go1AirPipeline forward mode=%s", self.runtime_mode())
        return self.policy.forward(batch_inputs, noise=noise)

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        if len(req.prompts) > 1:
            logger.warning("Go1AirPipeline only supports a single prompt/request; taking the first sample.")
        extra_args = getattr(req.sampling_params, "extra_args", {}) or {}
        batch_inputs = extra_args.get("batch_inputs")
        if batch_inputs is None and "robot_obs" in extra_args:
            batch_inputs = build_go1_air_batch_inputs_from_robot_obs(
                extra_args["robot_obs"],
                config=self.config,
                device=self.config.device,
                dtype=getattr(torch, self.config.dtype),
            )
        if batch_inputs is None:
            return DiffusionOutput(
                error=(
                    "Go1AirPipeline.forward expects sampling_params.extra_args['batch_inputs'] "
                    "with pre-built repo-side inputs, or extra_args['robot_obs'] from OpenPI serving."
                ),
                post_process_func=get_go1_air_post_process_func(self.od_config),
            )

        output = self._predict_actions(
            batch_inputs,
            noise=extra_args.get("noise"),
        )
        return DiffusionOutput(
            output={"actions": output, "video": []},
            custom_output={},
            post_process_func=get_go1_air_post_process_func(self.od_config),
        )
