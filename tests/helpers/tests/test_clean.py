# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import nullcontext

import pytest

from tests.helpers import clean

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_GIB = 2**30


def _patch_wait_platform(mocker, monkeypatch, *, queried: list[int], mem_get_info):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    monkeypatch.setattr(clean.current_omni_platform, "device_control_env_var", "CUDA_VISIBLE_DEVICES")

    def fake_device(idx):
        queried.append(idx)
        return nullcontext()

    mocker.patch.object(clean.current_omni_platform, "device", side_effect=fake_device)
    mocker.patch.object(clean.current_omni_platform, "mem_get_info", side_effect=mem_get_info)
    mocker.patch.object(clean.current_omni_platform, "empty_cache")
    mocker.patch.object(clean.time, "sleep")


def test_wait_queries_logical_ordinals_when_visible_set_is_physical(mocker, monkeypatch):
    """#6523 kept a physical remap after switching off NVML; device(4) is invalid."""
    queried: list[int] = []

    def mem_get_info():
        return (99 * _GIB, 100 * _GIB)

    _patch_wait_platform(mocker, monkeypatch, queried=queried, mem_get_info=mem_get_info)
    monkeypatch.setattr(clean, "_whisper_vram_allowance", lambda: (0.0, None))

    clean.wait_for_gpu_memory_to_clear(devices=[0, 1], threshold_ratio=0.05, timeout_s=1)

    assert queried == [0, 1]
    assert 4 not in queried
    assert 5 not in queried


def test_wait_matches_whisper_on_physical_id_not_logical(mocker, monkeypatch):
    queried: list[int] = []

    def mem_get_info():
        logical = queried[-1]
        if logical == 0:
            return (99 * _GIB, 100 * _GIB)
        return (50 * _GIB, 100 * _GIB)

    _patch_wait_platform(mocker, monkeypatch, queried=queried, mem_get_info=mem_get_info)
    # logical 1 → physical 5; without physical matching this 50% used card fails 15%.
    monkeypatch.setattr(clean, "_whisper_vram_allowance", lambda: (40.0, 5))

    clean.wait_for_gpu_memory_to_clear(devices=[0, 1], threshold_ratio=0.15, timeout_s=1)

    assert queried == [0, 1]
