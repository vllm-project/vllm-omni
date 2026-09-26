# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Supported execution contract for the native whole-clip SeedVR2 pipeline."""

import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.seedvr2.parallel import validate_seedvr2_parallel_config


def validate_seedvr2_config(config: OmniDiffusionConfig) -> None:
    size = config.additional_config.get("seedvr2_model_size", "3b")
    if size not in {"3b", "7b"}:
        raise ValueError("seedvr2_model_size must be '3b' or '7b'")
    dtype = torch.bfloat16 if size == "7b" else torch.float16
    if config.dtype != dtype:
        raise ValueError(f"SeedVR2 {size} requires dtype={dtype}")
    if not config.enforce_eager:
        raise ValueError("SeedVR2 requires enforce_eager=True; compiled execution is not supported")
    if config.cache_backend != "none" or config.quantization_config is not None:
        raise ValueError("SeedVR2 requires unquantized weights and cache_backend=none for its single Euler step")
    if config.vae_use_slicing:
        raise ValueError("SeedVR2 whole-clip VAE does not support batch slicing")
    if config.enable_cpu_offload or config.enable_layerwise_offload or config.enable_distributed_layerwise_offload:
        raise ValueError("SeedVR2 native whole-clip execution does not support CPU offload")
    parallel = config.parallel_config
    validate_seedvr2_parallel_config(parallel)
    if parallel.data_parallel_size is not None and parallel.data_parallel_size > 1:
        raise ValueError("SeedVR2 does not support data_parallel_size > 1")
    heads = 24 if size == "7b" else 20
    if parallel.ulysses_degree > heads:
        raise ValueError(f"SeedVR2 {size} Ulysses requires at least one of its {heads} heads per rank")
    degrees = {
        "cfg_parallel_size": parallel.cfg_parallel_size,
        "tensor_parallel_size": parallel.tensor_parallel_size,
        "pipeline_parallel_size": parallel.pipeline_parallel_size,
        "text_encoder_tp_size": parallel.text_encoder_tp_size,
        "ring_degree": parallel.ring_degree,
        "allgather_degree": parallel.allgather_degree,
    }
    for name, degree in degrees.items():
        if degree != 1:
            raise ValueError(f"SeedVR2 requires {name}=1; use ulysses_degree for its model-owned SP")
    if parallel.vae_patch_parallel_size > 1 and parallel.vae_patch_parallel_size != parallel.ulysses_degree:
        raise ValueError("SeedVR2 requires vae_patch_parallel_size to match ulysses_degree")
    if parallel.vae_parallel_mode == "spatial_shard_width":
        raise ValueError("SeedVR2 VAE supports spatial_shard_height or tile mode")
    if parallel.use_hsdp or parallel.enable_expert_parallel:
        raise ValueError("SeedVR2 does not support HSDP or expert parallelism")
    if config.lora_path is not None:
        raise ValueError("SeedVR2 does not support LoRA adapters")
