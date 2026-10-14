# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Characterization tests for ``vllm_omni.worker.memory_utils``.

``request_memory_tolerant`` is still live in both workers
(``gpu_ar_worker.py`` / ``gpu_generation_worker.py``); a later change makes its cap loud
but must not change the numeric contract. These tests pin that contract with a
mocked ``MemorySnapshot`` — pure CPU, no GPU / NVML.
"""

import math
from types import SimpleNamespace

import pytest

from vllm_omni.worker.memory_utils import request_memory_tolerant

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

GIB = 1024**3


def _snapshot(total: int, free: int) -> SimpleNamespace:
    # Only the fields request_memory_tolerant reads (incl. device_ for the log).
    return SimpleNamespace(total_memory=total, free_memory=free, device_="cuda:0")


def _cache_config(util: float) -> SimpleNamespace:
    return SimpleNamespace(gpu_memory_utilization=util)


def test_passes_through_requested_when_free_suffices():
    snap = _snapshot(total=40 * GIB, free=40 * GIB)
    cfg = _cache_config(0.5)

    out = request_memory_tolerant(snap, cfg)

    assert out == math.ceil(40 * GIB * 0.5)


def test_caps_to_free_when_insufficient():
    cfg = _cache_config(0.9)
    requested = math.ceil(40 * GIB * 0.9)
    snap = _snapshot(total=40 * GIB, free=10 * GIB)  # free < requested

    out = request_memory_tolerant(snap, cfg)

    assert out == 10 * GIB
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
