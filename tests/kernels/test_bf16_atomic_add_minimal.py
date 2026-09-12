# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""No vLLM/model/MATE dependency: python -m pytest -q this_file.py."""

import importlib

import pytest
import torch

if hasattr(torch.version, "musa") and torch.version.musa is not None:
    importlib.import_module("torchada")

from vllm.triton_utils import tl, triton

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.musa]


@triton.jit
def _atomic_sum(values, output, count: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    increments = tl.load(values + offsets, mask=offsets < count, other=0)
    tl.atomic_add(output + offsets % 32, increments, mask=offsets < count)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_atomic_add(dtype):
    cuda = getattr(torch, "cuda")
    available = cuda.is_available() or (hasattr(torch, "musa") and torch.musa.is_available())
    if not available:
        pytest.skip("No CUDA/MUSA device")
    # Each destination receives eight exactly representable increments from
    # two independent blocks. Reset between runs to exercise real atomics.
    values = torch.ones(256, device="cuda", dtype=dtype)
    output = torch.zeros(32, device="cuda", dtype=dtype)
    for _ in range(5):
        output.zero_()
        _atomic_sum[(2,)](values, output, 256, 128)
        cuda.synchronize()
        torch.testing.assert_close(output, torch.full_like(output, 8), rtol=0, atol=0)
