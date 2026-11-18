# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage memory budgets for discrete and integrated GPUs, without GPU / NVML."""

import math

import pytest
from vllm.config import CacheConfig
from vllm.platforms import current_platform
from vllm.utils.mem_utils import MemorySnapshot

from vllm_omni.worker.memory_utils import request_memory_tolerant

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

GIB = 1024**3


@pytest.fixture(autouse=True)
def discrete_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(current_platform, "is_integrated_gpu", lambda device_id: False)


def _snapshot(total: int, free: int, device: str = "cuda:0") -> MemorySnapshot:
    return MemorySnapshot(total_memory=total, free_memory=free, device=device, auto_measure=False)


def _cache_config(util: float) -> CacheConfig:
    return CacheConfig(gpu_memory_utilization=util)


def test_passes_through_requested_when_free_suffices():
    snap = _snapshot(total=40 * GIB, free=40 * GIB)
    cfg = _cache_config(0.5)

    out = request_memory_tolerant(snap, cfg)

    assert out == math.ceil(40 * GIB * 0.5)


@pytest.mark.parametrize("free_memory", [0, 10 * GIB])
def test_caps_to_free_when_insufficient(free_memory: int):
    cfg = _cache_config(0.9)
    requested = math.ceil(40 * GIB * 0.9)
    snap = _snapshot(total=40 * GIB, free=free_memory)

    out = request_memory_tolerant(snap, cfg)

    assert out == free_memory
    assert out < requested


def test_boundary_free_equals_requested_is_not_capped():
    util = 0.5
    total = 40 * GIB
    requested = math.ceil(total * util)
    snap = _snapshot(total=total, free=requested)  # strict `<`, so no cap at equality

    out = request_memory_tolerant(snap, _cache_config(util))

    assert out == requested


def test_requested_uses_ceil_of_total_times_util():
    # 7 GiB * 0.3333... rounds UP (math.ceil), not down.
    total = 7 * GIB
    util = 1 / 3
    snap = _snapshot(total=total, free=total)

    out = request_memory_tolerant(snap, _cache_config(util))

    assert out == math.ceil(total * util)


@pytest.mark.parametrize("free_memory", [0, 10 * GIB, 36 * GIB - 1])
def test_integrated_gpu_rejects_insufficient_shared_memory(monkeypatch: pytest.MonkeyPatch, free_memory: int):
    # A nonzero device index catches accidentally checking the default GPU.
    monkeypatch.setattr(current_platform, "is_integrated_gpu", lambda device_id: device_id == 3)
    snap = _snapshot(total=40 * GIB, free=free_memory, device="cuda:3")

    with pytest.raises(ValueError, match="Decrease GPU memory utilization"):
        request_memory_tolerant(snap, _cache_config(0.9))


@pytest.mark.parametrize(
    ("total", "free", "util"),
    [(40 * GIB, 40 * GIB, 0.5), (40 * GIB, 20 * GIB, 0.5), (7 * GIB, 7 * GIB, 1 / 3)],
    ids=["sufficient", "exact-boundary", "rounding"],
)
def test_integrated_gpu_preserves_requested_budget(monkeypatch: pytest.MonkeyPatch, total: int, free: int, util: float):
    monkeypatch.setattr(current_platform, "is_integrated_gpu", lambda device_id: True)
    snap = _snapshot(total=total, free=free)

    assert request_memory_tolerant(snap, _cache_config(util)) == math.ceil(total * util)
