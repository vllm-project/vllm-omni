# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Installation ownership and request isolation for AlignCache."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from vllm_omni.diffusion.cache.base import CacheBackend

from .adapters.wan21 import validate_pipeline
from .config import WAN22_EXPERT_PROFILES, AlignCacheConfig
from .multicomponent import WanRuntime
from .state import ModelBinding

if TYPE_CHECKING:
    from .flow import FlowRuntime


def is_online_fp8(quant_config) -> bool:
    """Online FP8 only: checkpoint-serialized or other methods stay rejected."""
    if getattr(quant_config, "is_checkpoint_fp8_serialized", False):
        return False
    components = getattr(quant_config, "component_configs", None)
    if components is not None:
        configs = [*components.values(), getattr(quant_config, "default_config", None)]
        return all(c is None or is_online_fp8(c) for c in configs)
    return hasattr(quant_config, "get_name") and quant_config.get_name() == "fp8"


def validate_execution_config(config, quantization="none", policy="wan21_feedback_v1") -> None:
    if getattr(config, "stage_role", None) not in (None, "full"):
        raise ValueError("align_cache.stage_role: disaggregated stages are not supported")
    if getattr(config, "diffusion_kv_mode", "dense_legacy") != "dense_legacy":
        raise ValueError("align_cache.diffusion_kv_mode: requires dense_legacy")
    if not config.enforce_eager:
        raise ValueError("align_cache.enforce_eager: must be True")
    if getattr(config, "kv_offload_config", None) is not None:
        raise ValueError(f"align_cache.kv_offload_config: unsupported for {policy}")
    for name in (
        "enable_cpu_offload",
        "enable_layerwise_offload",
        "enable_distributed_layerwise_offload",
        "diffusion_offload_config",
        "streaming_output",
        "step_execution",
        "lora_path",
        "enable_sleep_mode",
    ):
        if getattr(config, name, None):
            raise ValueError(f"align_cache.{name}: unsupported for {policy}")
    quant_config = getattr(config, "quantization_config", None)
    if quant_config and not (quantization == "fp8" and is_online_fp8(quant_config)):
        raise ValueError("align_cache.quantization_config: unsupported unless align_cache.quantization is fp8")
    if quantization == "fp8" and not quant_config:
        raise ValueError("align_cache.quantization: fp8 requires the engine's online FP8 quantization")
    if config.dtype not in (torch.bfloat16, "bfloat16", "bf16"):
        raise ValueError("align_cache.dtype: requires bfloat16")
    if getattr(config, "num_gpus", 1) not in (None, 1):
        raise ValueError("align_cache.num_gpus: requires 1")
    parallel = config.parallel_config
    if getattr(parallel, "enable_expert_parallel", False):
        raise ValueError("align_cache.parallel_config.enable_expert_parallel: unsupported")
    for name in (
        "pipeline_parallel_size",
        "data_parallel_size",
        "tensor_parallel_size",
        "sequence_parallel_size",
        "ulysses_degree",
        "ring_degree",
        "allgather_degree",
        "cfg_parallel_size",
        "vae_patch_parallel_size",
    ):
        if getattr(parallel, name, 1) != 1:
            raise ValueError(f"align_cache.parallel_config.{name}: requires 1")
    if parallel.use_hsdp:
        raise ValueError("align_cache.parallel_config.use_hsdp: unsupported")


class AlignCacheBackend(CacheBackend):
    def __init__(self, config=None):
        super().__init__(config)
        self.align_config = AlignCacheConfig.parse(config)
        self.binding: ModelBinding | None = None
        self.runtime: FlowRuntime | WanRuntime | None = None
        self.component_binding: tuple[object, ...] = ()

    def _components(self, pipeline):
        if self.align_config.profile in WAN22_EXPERT_PROFILES:
            return (pipeline.transformer, pipeline.transformer_2)
        return (pipeline.transformer,)

    def _validate_pipeline(self, pipeline):
        if self.align_config.policy in ("qwen_flow_v1", "minimax_h3_flow_v1"):
            from .adapters.flow import validate_pipeline as validate_flow

            validate_flow(pipeline, self.align_config.policy, self.align_config.quantization == "fp8")
        elif self.align_config.policy == "wan22_feedback_v1":
            from .adapters.wan22 import validate_pipeline as validate_wan22

            validate_wan22(pipeline, self.align_config.profile)
        elif self.align_config.profile == "wan21_14b_reference_v1":
            validate_pipeline(pipeline, large=True)
        else:
            validate_pipeline(pipeline)

    def _check_binding(self, pipeline) -> FlowRuntime | WanRuntime:
        if (
            self.runtime is None
            or self.binding is None
            or pipeline is not self.binding.pipeline
            or pipeline.transformer is not self.binding.transformer
            or getattr(pipeline, "_aligncache_runtime", None) is not self.runtime
            or getattr(self.runtime, "owner", None) is not self.binding.token
            or any(a is not b for a, b in zip(self.component_binding, self._components(pipeline)))
        ):
            raise RuntimeError("align_cache: pipeline/transformer/runtime ownership conflict")
        return self.runtime

    def enable(self, pipeline):
        if self.enabled:
            self._check_binding(pipeline)
            return
        if hasattr(pipeline, "_aligncache_runtime"):
            raise RuntimeError("align_cache: foreign runtime already installed")
        validate_execution_config(pipeline.od_config, self.align_config.quantization, self.align_config.policy)
        self._validate_pipeline(pipeline)
        runtime: FlowRuntime | WanRuntime
        if self.align_config.policy == "qwen_flow_v1":
            from .adapters.flow import QwenFlowRuntime

            runtime = QwenFlowRuntime(self.align_config)
        elif self.align_config.policy == "minimax_h3_flow_v1":
            from .flow import FlowRuntime

            runtime = FlowRuntime(self.align_config)
        else:
            runtime = WanRuntime(self.align_config)
        token = object()
        setattr(runtime, "owner", token)
        binding = ModelBinding(pipeline, pipeline.transformer, token)
        try:
            pipeline._aligncache_runtime = runtime
        except BaseException:
            if getattr(pipeline, "_aligncache_runtime", None) is runtime:
                del pipeline._aligncache_runtime
            raise
        self.component_binding = self._components(pipeline)
        self.binding, self.runtime, self.enabled = binding, runtime, True

    def disable(self, pipeline):
        if not self.enabled:
            return
        assert self.binding is not None and self.runtime is not None
        if pipeline is not self.binding.pipeline:
            raise RuntimeError("align_cache: disable called for different pipeline")
        if self.runtime.active:
            raise RuntimeError("align_cache: cannot disable an active request")
        self.runtime.reset()
        if (
            getattr(pipeline, "_aligncache_runtime", None) is self.runtime
            and getattr(self.runtime, "owner", None) is self.binding.token
        ):
            del pipeline._aligncache_runtime
        self.runtime = self.binding = None
        self.component_binding = ()
        self.enabled = False

    def refresh(self, pipeline, num_inference_steps=0, verbose=True):
        self._check_binding(pipeline).reset()

    @contextmanager
    def request_context(self, pipeline, req):
        runtime = self._check_binding(pipeline)
        if runtime.active:
            raise RuntimeError("align_cache: concurrent/reentrant request forbidden")
        runtime.reset()
        if getattr(req, "skip_cache_refresh", False):
            raise ValueError("align_cache.skip_cache_refresh: unsupported")
        if req.prompt is None or req.sampling_params.num_outputs_per_prompt != 1:
            raise ValueError("align_cache: requires one prompt and one output")
        if req.sampling_params.lora_request is not None:
            raise ValueError("align_cache: request LoRA is unsupported")
        # Re-check admission: settings or hooks may change between requests.
        validate_execution_config(pipeline.od_config, self.align_config.quantization, self.align_config.policy)
        self._validate_pipeline(pipeline)
        with runtime.request_context():
            yield
