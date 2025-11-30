# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch

from vllm_omni.utils.platform_utils import detect_device_type


def get_local_device() -> torch.device:
    """Return the torch device for the current rank based on detected device type."""
    device_type = detect_device_type()
    local_rank = os.environ.get("LOCAL_RANK", 0)
    return torch.device(f"{device_type}:{local_rank}")
