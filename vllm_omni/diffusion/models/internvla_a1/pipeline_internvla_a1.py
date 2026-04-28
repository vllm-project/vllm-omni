# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import json
import os
from typing import Any

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
    wrap_methods_by_paths,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest

from .config import (
    DEFAULT_QWEN3_VL_MODEL,
    OBS_IMAGES,
    OBS_STATE,
    OBS_TASK,
    InternVLAA1Config,
)
from .model_internvla_a1 import InternVLAA1Policy, resolve_cosmos_checkpoint_paths

logger = init_logger(__name__)


def get_internvla_a1_post_process_func(od_config: OmniDiffusionConfig):
    del od_config

    def post_process_func(x):
        return x

    return post_process_func


class InternVLAA1Pipeline(nn.Module, DiffusionPipelineProfilerMixin):
    """InternVLA-A1 pipeline wrapper for the policy implementation."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.prefix = prefix
        self.model_dir = od_config.model
        self.config = self._build_config(od_config)
        custom_args = od_config.custom_pipeline_args or {}
        self.strict_load = bool(custom_args.get("strict_load", False))
        self.processor_model_name = str(custom_args.get("processor_model_name", DEFAULT_QWEN3_VL_MODEL))
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

    def _build_config(self, od_config: OmniDiffusionConfig) -> InternVLAA1Config:
        config_dict = self._load_config_dict(od_config)
        config = InternVLAA1Config.from_model_config(config_dict)

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

    def _load_config_dict(self, od_config: OmniDiffusionConfig) -> dict[str, Any]:
        if od_config.model_config:
            return dict(od_config.model_config)

        model_path = od_config.model
        if not model_path:
            return {}

        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            logger.info("InternVLAA1Pipeline config.json not found under %s; using defaults.", model_path)
            return {}

        with open(config_path, encoding="utf-8") as f:
            return json.load(f)

    def has_real_checkpoint(self) -> bool:
        return bool(self.model_dir) and os.path.exists(os.path.join(self.model_dir, "model.safetensors"))

    def runtime_mode(self) -> str:
        return "real_checkpoint_loaded" if self.has_real_checkpoint() else "no_checkpoint_policy"

    def _setup_policy_profiler_targets(self) -> None:
        if not self.od_config.enable_diffusion_pipeline_profiler:
            return

        wrap_methods_by_paths(
            self,
            [
                "policy.model.sample_actions",
                "policy.model.embed_prefix",
                "policy.model.embed_middle",
                "policy.model.denoise_step",
            ],
        )

    def _apply_policy_optimizations(self, policy: InternVLAA1Policy) -> None:
        policy.model.set_attention_implementation(self.config.attn_implementation)
        if not self.config.enable_regional_compile:
            return

        compile_targets = [
            "qwen3_vl_with_expert.und_expert.visual",
            "qwen3_vl_with_expert.und_expert.language_model",
            "qwen3_vl_with_expert.gen_expert",
            "qwen3_vl_with_expert.act_expert",
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
                logger.info("InternVLAA1Pipeline regional compile applied to %s", path)
            except Exception as exc:
                logger.warning("InternVLAA1Pipeline regional compile failed for %s: %s", path, exc)

    def _initialize_policy(self) -> InternVLAA1Policy:
        resolve_cosmos_checkpoint_paths()
        if self.has_real_checkpoint():
            logger.info("Loading InternVLA-A1 weights from %s", self.model_dir)
            policy = InternVLAA1Policy.from_pretrained(
                self.model_dir,
                config=self.config,
                processor_model_name=self.processor_model_name,
                strict=self.strict_load,
            )
        else:
            logger.info("Initializing InternVLA-A1 policy without checkpoint weights.")
            policy = InternVLAA1Policy(self.config, processor_model_name=self.processor_model_name)

        policy.to(self.config.device)
        policy.to(getattr(torch, self.config.dtype))
        policy.eval()
        self._apply_policy_optimizations(policy)
        return policy

    def _build_fake_batch_inputs(self) -> dict[str, torch.Tensor]:
        device = torch.device(self.config.device)
        dtype = getattr(torch, self.config.dtype)
        history = 2
        channels = 3
        image_size = self.config.image_resolution
        fake_image = torch.zeros((1, history, channels, image_size[0], image_size[1]), device=device, dtype=dtype)
        fake_mask = torch.ones((1,), device=device, dtype=torch.bool)
        return {
            OBS_STATE: torch.zeros((1, self.config.max_state_dim), device=device, dtype=dtype),
            OBS_TASK: [""],
            f"{OBS_IMAGES}.image0": fake_image.clone(),
            f"{OBS_IMAGES}.image1": fake_image.clone(),
            f"{OBS_IMAGES}.image2": fake_image.clone(),
            f"{OBS_IMAGES}.image0_mask": fake_mask.clone(),
            f"{OBS_IMAGES}.image1_mask": fake_mask.clone(),
            f"{OBS_IMAGES}.image2_mask": fake_mask.clone(),
        }

    def _warmup(self) -> None:
        logger.info("InternVLAA1Pipeline warmup started")
        try:
            batch_inputs = self._build_fake_batch_inputs()
            noise = torch.zeros(
                (1, self.config.chunk_size, self.config.max_action_dim),
                device=self.config.device,
                dtype=torch.float32,
            )
            with torch.inference_mode():
                self.policy.forward(batch_inputs, noise=noise, decode_image=False)
        except Exception as exc:
            logger.warning("InternVLAA1Pipeline warmup failed: %s", exc)
            return
        logger.info("InternVLAA1Pipeline warmup finished")

    def _predict_actions(
        self,
        batch_inputs: dict[str, Any],
        *,
        noise: torch.Tensor | None = None,
        decode_image: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logger.debug("InternVLAA1Pipeline forward mode=%s", self.runtime_mode())
        return self.policy.forward(
            batch_inputs,
            noise=noise,
            decode_image=decode_image,
        )

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        if len(req.prompts) > 1:
            logger.warning("InternVLAA1Pipeline only supports a single prompt/request; taking the first sample.")
        extra_args = getattr(req.sampling_params, "extra_args", {}) or {}
        batch_inputs = extra_args.get("batch_inputs")
        if batch_inputs is None:
            return DiffusionOutput(
                error=(
                    "InternVLAA1Pipeline.forward expects sampling_params.extra_args['batch_inputs'] "
                    "with pre-built repo-side inputs."
                ),
                post_process_func=get_internvla_a1_post_process_func(self.od_config),
            )

        output, decoded = self._predict_actions(
            batch_inputs,
            noise=extra_args.get("noise"),
            decode_image=bool(extra_args.get("decode_image", False)),
        )
        custom_output: dict[str, Any] = {}
        if decoded is not None:
            custom_output["decoded"] = decoded
        return DiffusionOutput(
            output=output,
            custom_output=custom_output,
            post_process_func=get_internvla_a1_post_process_func(self.od_config),
        )
