# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm_ascend")

from vllm_ascend.ascend_forward_context import MoECommType

from vllm_omni.platforms.npu.layers import fused_moe

pytestmark = [pytest.mark.core_model, pytest.mark.npu]


def _vllm_config(additional_config=None, *, enable_expert_parallel=True):
    return SimpleNamespace(
        additional_config=additional_config,
        parallel_config=SimpleNamespace(enable_expert_parallel=enable_expert_parallel),
    )


def _set_ep_size(monkeypatch, ep_size):
    monkeypatch.setattr(
        fused_moe,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=ep_size),
    )


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("allgather", MoECommType.ALLGATHER),
        ("ALLTOALL", MoECommType.ALLTOALL),
    ],
)
def test_additional_config_override(monkeypatch, configured, expected):
    _set_ep_size(monkeypatch, 2)

    result = fused_moe._select_moe_comm_method(_vllm_config({"npu_moe_comm_method": configured}))

    assert result is expected


def test_invalid_override_fails_fast(monkeypatch):
    _set_ep_size(monkeypatch, 2)

    with pytest.raises(ValueError, match="Supported values"):
        fused_moe._select_moe_comm_method(_vllm_config({"npu_moe_comm_method": "mc2"}))


@pytest.mark.parametrize("configured", [False, 0, []])
def test_falsey_non_string_override_fails_fast(monkeypatch, configured):
    _set_ep_size(monkeypatch, 2)

    with pytest.raises(TypeError, match="must be a string"):
        fused_moe._select_moe_comm_method(_vllm_config({"npu_moe_comm_method": configured}))


def test_empty_string_override_fails_fast(monkeypatch):
    _set_ep_size(monkeypatch, 2)

    with pytest.raises(ValueError, match="Supported values"):
        fused_moe._select_moe_comm_method(_vllm_config({"npu_moe_comm_method": ""}))


@pytest.mark.parametrize(
    ("enable_expert_parallel", "ep_size"),
    [(False, 1), (True, 1)],
)
def test_alltoall_override_requires_multi_rank_ep(
    monkeypatch,
    enable_expert_parallel,
    ep_size,
):
    _set_ep_size(monkeypatch, ep_size)

    with pytest.raises(ValueError, match="requires expert parallelism"):
        fused_moe._select_moe_comm_method(
            _vllm_config(
                {"npu_moe_comm_method": "alltoall"},
                enable_expert_parallel=enable_expert_parallel,
            )
        )
