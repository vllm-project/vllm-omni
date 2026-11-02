# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""cuDNN numeric policy for CUDA-graph parity tests.

Graph replay and eager convolution can disagree by about one ulp when TF32
or cuDNN benchmark autotune is left on. These helpers set the conv-level
flag (not the backend-wide one) and restore the previous values.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def ieee_cudnn_fp32() -> Iterator[None]:
    """Force IEEE FP32 conv math, then restore the previous conv precision."""
    conv = torch.backends.cudnn.conv
    previous = conv.fp32_precision
    conv.fp32_precision = "ieee"
    try:
        yield
    finally:
        conv.fp32_precision = previous


@contextmanager
def deterministic_cudnn() -> Iterator[None]:
    """Turn benchmark off and deterministic algorithms on, then restore both."""
    benchmark = torch.backends.cudnn.benchmark
    deterministic = torch.backends.cudnn.deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic
