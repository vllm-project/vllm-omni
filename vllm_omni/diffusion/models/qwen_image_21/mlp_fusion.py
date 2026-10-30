# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Configuration eligibility for Qwen-Image-2.1's opt-in FFN backend."""

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.layers.fused_swiglu import fused_swiglu_available
from vllm_omni.diffusion.offloader.config import OffloadStrategy, resolve_offload_strategy

logger = init_logger(__name__)


def select_mlp_backend(od_config: OmniDiffusionConfig, *, unquantized: bool) -> str:
    backend = od_config.additional_config.get("qwen21_mlp_backend", "native")
    if backend not in ("native", "eagle"):
        raise ValueError("qwen21_mlp_backend must be native or eagle")
    if backend == "native":
        return backend
    parallel = od_config.parallel_config
    if (
        not unquantized
        or not od_config.enforce_eager
        or od_config.dtype != torch.bfloat16
        or parallel.tensor_parallel_size != 1
        or parallel.sequence_parallel_size != 1
        or parallel.use_hsdp
        or od_config.lora_path
        or getattr(od_config, "lora_config", None) is not None
        or resolve_offload_strategy(od_config) != OffloadStrategy.NONE
        or not fused_swiglu_available()
    ):
        logger.warning(
            "Qwen-Image-2.1 Eagle FFN requires the public MindIE-SD API on A5, BF16 eager, "
            "TP1/SP1, resident unquantized weights and no LoRA/HSDP; using native MLP"
        )
        return "native"
    return backend
