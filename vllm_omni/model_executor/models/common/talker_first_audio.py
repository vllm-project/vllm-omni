# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared eligibility; each model separately controls its first-audio opt-in."""

from typing import Any

import torch

from vllm_omni.platforms import current_omni_platform


def supports_talker_first_audio(vllm_config: Any) -> bool:
    """Require CUDA MRv2 streaming, local single-rank execution and no prefix replay.

    Other configurations retain the regular codec path. These conditions do
    not enable the feature: model-specific options must also accept it.
    """
    model = vllm_config.model_config
    if not (getattr(model, "use_v2_model_runner", False) and getattr(model, "async_chunk", False)):
        return False
    parallel = vllm_config.parallel_config
    return (
        current_omni_platform.is_cuda()
        and torch.device(vllm_config.device_config.device).type == "cuda"
        and parallel.tensor_parallel_size == 1
        and parallel.pipeline_parallel_size == 1
        and parallel.distributed_executor_backend in (None, "uni")
        and not vllm_config.cache_config.enable_prefix_caching
    )
