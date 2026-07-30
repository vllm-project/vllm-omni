# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.parallel import ring

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((8, 0), True),
        ((8, 9), True),
        ((9, 0), True),
        ((10, 0), False),
        ((10, 3), False),
        ((12, 0), False),
    ],
)
def test_can_use_fa3_checks_device_arch(monkeypatch, capability, expected):
    monkeypatch.setattr(ring, "HAS_FA3", True)
    monkeypatch.setattr(ring, "FA3_SUPPORTED_CUDA_MAJORS", frozenset({8, 9}))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)

    assert ring._can_use_fa3(torch.device("cuda")) is expected


def test_can_use_fa3_preserves_unknown_extension_contract(monkeypatch):
    monkeypatch.setattr(ring, "HAS_FA3", True)
    monkeypatch.setattr(ring, "FA3_SUPPORTED_CUDA_MAJORS", None)

    assert ring._can_use_fa3(torch.device("cuda"))


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((8, 0), True),
        ((9, 0), True),
        ((10, 0), False),
        ((10, 3), False),
        ((12, 0), False),
    ],
)
def test_can_use_fa2_checks_device_arch(monkeypatch, capability, expected):
    monkeypatch.setattr(ring, "HAS_FLASH_ATTN", True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)

    assert ring._can_use_fa2(torch.device("cuda")) is expected


def test_ring_falls_back_to_sdpa_when_fa3_extension_has_no_device_kernel(monkeypatch):
    monkeypatch.setattr(ring, "HAS_FA3", True)
    monkeypatch.setattr(ring, "HAS_FLASH_ATTN", False)
    monkeypatch.setattr(ring, "HAS_AITER", False)
    monkeypatch.setattr(ring, "_can_use_fa3", lambda _device: False)

    expected = torch.randn(1, 2, 1, 8)

    def fake_ring_sdpa(*args, **kwargs):
        return expected

    module_name = "vllm_omni.diffusion.attention.backends.ring_pytorch_attn"
    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(ring_pytorch_attn_func=fake_ring_sdpa))

    strategy = ring.RingParallelAttention(SimpleNamespace(ring_group=object()))
    query = torch.randn(1, 2, 1, 8)
    actual = strategy.run_attention(query, query, query, attn_metadata=None)

    assert actual is expected
