# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch


def get_local_device() -> torch.device:
    """Return the torch device for the current rank."""
    return torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
