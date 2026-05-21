# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch

from vllm_omni.platforms import current_omni_platform


def get_local_device() -> torch.device:
    """Return the torch device for the current rank based on detected device type."""
    if current_omni_platform.is_initialized():
        return current_omni_platform.get_torch_device(current_omni_platform.current_device())
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return current_omni_platform.get_torch_device(local_rank)
