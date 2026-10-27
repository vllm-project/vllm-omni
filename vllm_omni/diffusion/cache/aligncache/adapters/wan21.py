# SPDX-License-Identifier: Apache-2.0
"""Wan2.1 T2V admission on the shared Wan pipeline; Wan2.2 reuses its structural checks."""

import math

import torch


def validate_pipeline(pipeline, large=False):
    from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler
    from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import Wan22Pipeline
    from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel

    if type(pipeline) is not Wan22Pipeline or pipeline.expand_timesteps or getattr(pipeline, "is_dmd", False):
        raise ValueError("align_cache: Wan2.1 requires the native WanPipeline T2V path, not I2V/TI2V or subclasses")
    if (
        getattr(pipeline, "has_transformer_2", False)
        or pipeline.transformer_2 is not None
        or type(pipeline.transformer) is not WanTransformer3DModel
    ):
        raise ValueError("align_cache: Wan2.1 requires one native WanTransformer3DModel")
    if type(pipeline.scheduler) is not FlowUniPCMultistepScheduler:
        raise ValueError("align_cache: requires native FlowUniPCMultistepScheduler")
    validate_transformer(pipeline.transformer, large=large)
    validate_scheduler(pipeline.scheduler)


def validate_transformer(transformer, large=False, overrides=None):
    expected = {
        "in_channels": 16,
        "out_channels": 16,
        "num_layers": 40 if large else 30,
        "num_attention_heads": 40 if large else 12,
        "attention_head_dim": 128,
        "text_dim": 4096,
        "ffn_dim": 13824 if large else 8960,
        "freq_dim": 256,
        "image_dim": None,
        "added_kv_proj_dim": None,
        **(overrides or {}),
    }
    for key, value in expected.items():
        if getattr(transformer.config, key, None) != value:
            raise ValueError(f"align_cache.transformer.{key}: T2V structure requires {value}")
    if tuple(transformer.config.patch_size) != (1, 2, 2):
        raise ValueError("align_cache.transformer.patch_size: requires (1, 2, 2)")
    for module in transformer.modules():
        if any(
            getattr(module, key, None)
            for key in ("_forward_hooks", "_forward_pre_hooks", "_hf_hook", "_hook_registry", "_diffusers_hook")
        ):
            raise ValueError("align_cache: transformer hooks require separate capability validation")
        if "forward" in vars(module):
            raise ValueError("align_cache: replaced transformer forward is unsupported")
    parameter = next(transformer.parameters())
    if parameter.dtype != torch.bfloat16 or parameter.device.type != "cuda":
        raise ValueError("align_cache: native CUDA BF16 weights required")


def validate_scheduler(scheduler):
    if scheduler.config.prediction_type != "flow_prediction":
        raise ValueError("align_cache.scheduler.prediction_type: requires flow_prediction")
    scheduler_defaults = {
        "num_train_timesteps": 1000,
        "solver_order": 2,
        "solver_type": "bh2",
        "predict_x0": True,
        "use_dynamic_shifting": False,
        "thresholding": False,
        "lower_order_final": True,
        "timestep_spacing": "linspace",
        "steps_offset": 0,
        "final_sigmas_type": "zero",
    }
    for key, value in scheduler_defaults.items():
        if scheduler.config.get(key) != value:
            raise ValueError(f"align_cache.scheduler.{key}: native configuration requires {value}")
    if scheduler.solver_p is not None or scheduler.disable_corrector:
        raise ValueError("align_cache.scheduler: custom corrector/solver unsupported")
    if not math.isfinite(scheduler.config.shift) or scheduler.config.shift <= 0:
        raise ValueError("align_cache.scheduler.shift: requires finite positive value")
