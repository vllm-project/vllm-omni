# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU coverage of connector rank resolution before distributed initialization."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.parallel]

LOCAL_RANK_ENVS = (
    "LOCAL_RANK",
    "VLLM_LOCAL_RANK",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "MV2_COMM_WORLD_LOCAL_RANK",
)
GLOBAL_RANK_ENVS = ("RANK", "RANK_ID")
DEVICE_ENVS = ("ASCEND_RT_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")


@pytest.fixture
def local_rank(monkeypatch, mocker) -> ModuleType:
    # Load the stdlib-only module without importing the optional connector backends.
    path = Path(__file__).resolve().parents[3] / "vllm_omni/distributed/omni_connectors/utils/local_rank.py"
    spec = importlib.util.spec_from_file_location("connector_local_rank", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in (*LOCAL_RANK_ENVS, *GLOBAL_RANK_ENVS, *DEVICE_ENVS):
        monkeypatch.delenv(name, raising=False)
    mocker.patch.object(module, "_get_tensor_model_parallel_rank", side_effect=AssertionError("TP is not initialized"))
    return module


@pytest.mark.parametrize("tp_rank", [0, 3])
def test_initialized_tp_rank_takes_precedence(local_rank, monkeypatch, mocker, tp_rank):
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,6")
    mocker.patch.object(local_rank, "_get_tensor_model_parallel_rank", return_value=tp_rank)

    assert local_rank.get_connector_local_rank() == tp_rank


@pytest.mark.parametrize("env_name", LOCAL_RANK_ENVS)
@pytest.mark.parametrize("rank", [0, 5])
def test_launcher_local_rank_overrides_global_rank(local_rank, monkeypatch, env_name, rank):
    for name in LOCAL_RANK_ENVS:
        monkeypatch.setenv(name, "invalid")
    monkeypatch.setenv(env_name, str(rank))
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3,5,7")

    assert local_rank.get_connector_local_rank() == rank


@pytest.mark.parametrize("preferred,fallback", list(zip(LOCAL_RANK_ENVS, LOCAL_RANK_ENVS[1:])))
def test_launcher_local_rank_priority(local_rank, monkeypatch, preferred, fallback):
    monkeypatch.setenv(preferred, "1")
    monkeypatch.setenv(fallback, "2")

    assert local_rank.get_connector_local_rank() == 1


@pytest.mark.parametrize("invalid_rank", ["", "not-a-rank", "1.5"])
def test_invalid_local_ranks_fall_back_to_global_rank(local_rank, monkeypatch, invalid_rank):
    for name in LOCAL_RANK_ENVS:
        monkeypatch.setenv(name, invalid_rank)
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,6")

    assert local_rank.get_connector_local_rank() == 1


@pytest.mark.parametrize("rank_env", GLOBAL_RANK_ENVS)
@pytest.mark.parametrize("device_env", DEVICE_ENVS)
def test_global_rank_maps_to_visible_device_ordinal(local_rank, monkeypatch, rank_env, device_env):
    monkeypatch.setenv(rank_env, "6")
    monkeypatch.setenv(device_env, " 1, ,3,5,7, ")

    # The local ordinal is 2, not physical device ID 5 or global rank 6.
    assert local_rank.get_connector_local_rank() == 2


def test_ascend_device_mask_takes_precedence(local_rank, monkeypatch):
    monkeypatch.setenv("RANK", "6")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "1,3,5,7")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    assert local_rank.get_connector_local_rank() == 2


def test_empty_ascend_mask_falls_back_to_cuda(local_rank, monkeypatch):
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")

    assert local_rank.get_connector_local_rank() == 1


@pytest.mark.parametrize("mask", [None, "", " , , "])
def test_global_rank_is_preserved_without_visible_devices(local_rank, monkeypatch, mask):
    monkeypatch.setenv("RANK", "6")
    if mask is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)

    assert local_rank.get_connector_local_rank() == 6


@pytest.mark.parametrize("rank,expected", [("0", 0), ("6", 2), ("invalid", 3)])
def test_global_rank_priority_and_invalid_fallback(local_rank, monkeypatch, rank, expected):
    monkeypatch.setenv("RANK", rank)
    monkeypatch.setenv("RANK_ID", "7")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3,5,7")

    assert local_rank.get_connector_local_rank() == expected


@pytest.mark.parametrize("rank", [None, "invalid"])
def test_no_usable_rank_defaults_to_zero(local_rank, monkeypatch, rank):
    if rank is not None:
        for name in (*LOCAL_RANK_ENVS, *GLOBAL_RANK_ENVS):
            monkeypatch.setenv(name, rank)

    assert local_rank.get_connector_local_rank() == 0


def test_environment_is_resolved_on_each_call(local_rank, monkeypatch):
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3,5,7")
    assert local_rank.get_connector_local_rank() == 1

    monkeypatch.setenv("RANK", "6")
    assert local_rank.get_connector_local_rank() == 2

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
    assert local_rank.get_connector_local_rank() == 0
