# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for connector rank resolution before and after TP initialization."""

import pytest
from pytest_mock import MockerFixture
from vllm.distributed import parallel_state

from vllm_omni.distributed.omni_connectors.utils.local_rank import get_connector_local_rank

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

LOCAL_RANK_ENV_NAMES = (
    "LOCAL_RANK",
    "VLLM_LOCAL_RANK",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "MV2_COMM_WORLD_LOCAL_RANK",
)


@pytest.fixture(autouse=True)
def isolate_rank_sources(monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture):
    # Do not inherit the developer's or CI worker's distributed launch settings.
    for name in (*LOCAL_RANK_ENV_NAMES, "RANK", "RANK_ID", "CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    mocker.patch.object(
        parallel_state,
        "get_tensor_model_parallel_rank",
        side_effect=AssertionError("tensor model parallel group is not initialized"),
    )


@pytest.mark.parametrize("tp_rank", [0, 3])
def test_tp_rank_takes_precedence(tp_rank: int, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture):
    monkeypatch.setenv("LOCAL_RANK", "5")
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    mocker.patch.object(parallel_state, "get_tensor_model_parallel_rank", return_value=tp_rank)

    assert get_connector_local_rank() == tp_rank


def test_tp_query_error_falls_back(monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture):
    mocker.patch.object(parallel_state, "get_tensor_model_parallel_rank", side_effect=RuntimeError("TP unavailable"))
    monkeypatch.setenv("LOCAL_RANK", "2")

    assert get_connector_local_rank() == 2


@pytest.mark.parametrize("env_name", LOCAL_RANK_ENV_NAMES)
def test_local_rank_sources(env_name: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(env_name, "3")
    monkeypatch.setenv("RANK", "7")
    # Local ranks must not be reduced modulo the number of visible devices.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    assert get_connector_local_rank() == 3


@pytest.mark.parametrize(
    ("preferred", "fallback"),
    [
        ("LOCAL_RANK", "VLLM_LOCAL_RANK"),
        ("VLLM_LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK"),
        ("OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK"),
    ],
)
def test_local_rank_precedence(preferred: str, fallback: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(preferred, "0")
    monkeypatch.setenv(fallback, "3")

    assert get_connector_local_rank() == 0


@pytest.mark.parametrize("invalid_rank", ["", "invalid", "1.5"])
def test_invalid_local_rank_is_skipped(invalid_rank: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOCAL_RANK", invalid_rank)
    monkeypatch.setenv("VLLM_LOCAL_RANK", "2")

    assert get_connector_local_rank() == 2


@pytest.mark.parametrize(
    "rank_env",
    [
        {"RANK": "7", "RANK_ID": "2"},
        {"RANK_ID": "7"},
        {"RANK": "invalid", "RANK_ID": "7"},
    ],
    ids=["rank-precedes-rank-id", "rank-id-only", "invalid-rank-falls-back"],
)
def test_global_rank_fallback(rank_env: dict[str, str], monkeypatch: pytest.MonkeyPatch):
    for name, value in rank_env.items():
        monkeypatch.setenv(name, value)

    assert get_connector_local_rank() == 7


@pytest.mark.parametrize(
    ("device_env", "expected_rank"),
    [
        ({"CUDA_VISIBLE_DEVICES": "0,1,2,3"}, 3),
        ({"ASCEND_RT_VISIBLE_DEVICES": "4,5", "CUDA_VISIBLE_DEVICES": "0,1,2,3"}, 1),
        ({"ASCEND_RT_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "0,1,2,3"}, 3),
        ({"CUDA_VISIBLE_DEVICES": " 2, , 5, 7, "}, 1),
        ({"CUDA_VISIBLE_DEVICES": ""}, 7),
        ({"CUDA_VISIBLE_DEVICES": " , , "}, 7),
    ],
    ids=["cuda", "ascend-precedes-cuda", "empty-ascend", "whitespace-and-empty-entries", "empty-list", "blank-entries"],
)
def test_global_rank_maps_to_visible_devices(
    device_env: dict[str, str], expected_rank: int, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("RANK", "7")
    for name, value in device_env.items():
        monkeypatch.setenv(name, value)

    assert get_connector_local_rank() == expected_rank


@pytest.mark.parametrize("invalid_values", [False, True], ids=["missing-ranks", "invalid-ranks"])
def test_no_usable_rank_defaults_to_zero(invalid_values: bool, monkeypatch: pytest.MonkeyPatch):
    if invalid_values:
        for name in (*LOCAL_RANK_ENV_NAMES, "RANK", "RANK_ID"):
            monkeypatch.setenv(name, "invalid")

    assert get_connector_local_rank() == 0
