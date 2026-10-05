# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager

import torch


@contextmanager
def cudnn_settings(benchmark: bool, deterministic: bool):
    """Temporarily set cuDNN backend flags, restoring the old values on exit.

    Used to scope ``torch.backends.cudnn.benchmark``/``deterministic`` to the
    few blocks whose convolutions actually benefit from autotune, instead of
    setting them globally at import time (which would silently change every
    other model's conv behaviour in the same process).
    """
    old_benchmark = torch.backends.cudnn.benchmark
    old_deterministic = torch.backends.cudnn.deterministic

    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = old_benchmark
        torch.backends.cudnn.deterministic = old_deterministic
