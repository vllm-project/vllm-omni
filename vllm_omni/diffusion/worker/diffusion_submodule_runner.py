# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight runner for diffusion submodule stages."""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any

import torch
from vllm.config import LoadConfig
from vllm.logger import init_logger
from vllm.utils.mem_utils import DeviceMemoryProfiler, GiB_bytes

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)


class DiffusionSubmoduleRunner:
    """Single-forward runner for diffusion submodule stages."""

    def __init__(
        self,
        vllm_config: Any,
        od_config: OmniDiffusionConfig,
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.od_config = od_config
        self.device = device
        self.pipeline = None

    def load_model(
        self,
        memory_pool_context_fn: Any | None = None,
        load_format: str | None = None,
        custom_pipeline_name: str | None = None,
    ) -> None:
        if load_format == "dummy":
            return

        load_device = (
            "cpu" if self.od_config.enable_cpu_offload or self.od_config.enable_layerwise_offload else str(self.device)
        )

        def get_memory_context():
            if memory_pool_context_fn is not None:
                return memory_pool_context_fn(tag="weights")
            return nullcontext()

        load_config = LoadConfig()
        model_loader = DiffusersPipelineLoader(load_config, od_config=self.od_config)
        t0 = time.perf_counter()
        with get_memory_context():
            with DeviceMemoryProfiler() as mem:
                self.pipeline = model_loader.load_model(
                    od_config=self.od_config,
                    load_device=load_device,
                    load_format=load_format,
                    custom_pipeline_name=custom_pipeline_name,
                    device=self.device,
                )

        logger.info(
            "DiffusionSubmoduleRunner[stage=%s]: loaded in %.3fs, %.3f GiB GPU",
            self.od_config.model_stage,
            time.perf_counter() - t0,
            mem.consumed_memory / GiB_bytes,
        )

    def execute_model(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        assert self.pipeline is not None, "Model not loaded. Call load_model() first."
        if not req.prompts:
            raise ValueError("Cannot execute diffusion submodule runner on an empty request.")

        stage = getattr(self.od_config, "model_stage", None)
        if stage not in ("encode", "decode"):
            raise ValueError(f"DiffusionSubmoduleRunner requires model_stage encode/decode, got {stage!r}.")

        sampling = req.sampling_params
        if sampling.generator is None and sampling.seed is not None:
            if sampling.generator_device is not None:
                gen_device = sampling.generator_device
            elif self.device.type == "cpu":
                gen_device = "cpu"
            else:
                gen_device = self.device
            sampling.generator = torch.Generator(device=gen_device).manual_seed(sampling.seed)

        with torch.inference_mode():
            with set_forward_context(
                vllm_config=self.vllm_config,
                omni_diffusion_config=self.od_config,
            ):
                if stage == "encode":
                    results = self.pipeline.execute_encode([req])
                else:
                    results = self.pipeline.execute_decode([req])

        if not results:
            return DiffusionOutput(error=f"{stage} stage returned no outputs")

        stage_out = results[0]
        stage_items = stage_out.items() if isinstance(stage_out, dict) else vars(stage_out).items()
        payload = {k: v for k, v in stage_items if k != "req_id" and v is not None and not (k == "metadata" and not v)}

        if stage == "decode":
            return DiffusionOutput(output=payload.get("image"), multimodal_output=payload)

        return DiffusionOutput(output=None, multimodal_output=payload)
