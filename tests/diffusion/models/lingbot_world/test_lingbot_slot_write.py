# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""KV row writes retain slot ordering, conversion and untouched pool rows."""

import importlib.util
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def load():
    path = Path(__file__).resolve().parents[4] / "vllm_omni/experimental/ar_diffusion/kv_cache/slot_write.py"
    spec = importlib.util.spec_from_file_location("_slot_write", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.write_kv_slots


def check(device, empty):
    write = load()
    pool = torch.full((32, 2, 8), -7, dtype=torch.bfloat16, device=device)
    expected = pool.clone()
    slots = torch.tensor([] if empty else [19, 3, 8, 0], dtype=torch.long, device=device)
    packed = torch.arange(slots.numel() * 3 * 2 * 8, device=device, dtype=torch.float32).reshape(-1, 3, 2, 8)
    source = packed[:, 1]
    expected[slots] = source.to(expected.dtype)
    write(pool, slots, source)
    assert torch.equal(pool, expected)
    expected[slots] = -source.to(expected.dtype)
    write(pool, slots, -source)
    assert torch.equal(pool, expected)


@pytest.mark.cpu
@pytest.mark.parametrize("empty", [False, True])
def test_cpu_slot_write(empty):
    check("cpu", empty)


@pytest.mark.npu
@pytest.mark.A5
@pytest.mark.parametrize("empty", [False, True])
def test_npu_slot_write(empty):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    check("npu", empty)
