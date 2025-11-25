import os

import torch


def get_local_device() -> torch.device:
    """Return the torch device for the current rank."""
    return torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
