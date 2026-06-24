# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


def _install_fake_vllm_ascend(monkeypatch):
    fake_modules = {
        "vllm_ascend": ModuleType("vllm_ascend"),
        "vllm_ascend.ascend_forward_context": ModuleType("vllm_ascend.ascend_forward_context"),
        "vllm_ascend.distributed": ModuleType("vllm_ascend.distributed"),
        "vllm_ascend.distributed.parallel_state": ModuleType("vllm_ascend.distributed.parallel_state"),
        "vllm_ascend.ops": ModuleType("vllm_ascend.ops"),
        "vllm_ascend.ops.fused_moe": ModuleType("vllm_ascend.ops.fused_moe"),
        "vllm_ascend.ops.fused_moe.fused_moe": ModuleType("vllm_ascend.ops.fused_moe.fused_moe"),
        "vllm_ascend.ops.fused_moe.moe_comm_method": ModuleType("vllm_ascend.ops.fused_moe.moe_comm_method"),
        "vllm_ascend.utils": ModuleType("vllm_ascend.utils"),
    }

    class FakeMoECommType:
        ALLGATHER = "allgather"
        ALLTOALL = "alltoall"

    class FakeAscendDeviceType:
        A2 = "A2"
        A3 = "A3"
        A5 = "A5"
        _310P = "310P"

    class FakeAscendFusedMoE:
        pass

    class FakeMoECommMethods:
        @staticmethod
        def get(_comm_type):
            return None

    fake_modules["vllm_ascend.ascend_forward_context"].MoECommType = FakeMoECommType
    fake_modules["vllm_ascend.distributed.parallel_state"]._EP = None
    fake_modules["vllm_ascend.distributed.parallel_state"]._MC2 = None
    fake_modules["vllm_ascend.ops.fused_moe.fused_moe"].AscendFusedMoE = FakeAscendFusedMoE
    fake_modules["vllm_ascend.ops.fused_moe.moe_comm_method"]._MoECommMethods = FakeMoECommMethods
    fake_modules["vllm_ascend.utils"].AscendDeviceType = FakeAscendDeviceType
    fake_modules["vllm_ascend.utils"].get_ascend_device_type = lambda: FakeAscendDeviceType.A2

    fake_modules["vllm_ascend"].distributed = fake_modules["vllm_ascend.distributed"]
    fake_modules["vllm_ascend.distributed"].parallel_state = fake_modules["vllm_ascend.distributed.parallel_state"]
    fake_modules["vllm_ascend"].ops = fake_modules["vllm_ascend.ops"]
    fake_modules["vllm_ascend.ops"].fused_moe = fake_modules["vllm_ascend.ops.fused_moe"]
    fake_modules["vllm_ascend.ops.fused_moe"].fused_moe = fake_modules["vllm_ascend.ops.fused_moe.fused_moe"]
    fake_modules["vllm_ascend.ops.fused_moe"].moe_comm_method = fake_modules[
        "vllm_ascend.ops.fused_moe.moe_comm_method"
    ]
    fake_modules["vllm_ascend"].utils = fake_modules["vllm_ascend.utils"]

    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    sys.modules.pop("vllm_omni.platforms.npu.models.hunyuan_fused_moe", None)
    return fake_modules["vllm_ascend.distributed.parallel_state"]


@pytest.mark.cpu
@pytest.mark.core_model
def test_hunyuan_mc2_uses_ep_group_ranks(monkeypatch):
    """MC2 must use the same non-contiguous EP layout as vLLM, not contiguous rank chunks."""
    ascend_parallel_state = _install_fake_vllm_ascend(monkeypatch)
    hunyuan_moe = importlib.import_module("vllm_omni.platforms.npu.models.hunyuan_fused_moe")

    captured = {}

    def fake_vllm_init_model_parallel_group(group_ranks, local_rank, backend, group_name=None):
        captured["group_ranks"] = group_ranks
        captured["local_rank"] = local_rank
        captured["backend"] = backend
        captured["group_name"] = group_name
        return SimpleNamespace(ranks=group_ranks[0], world_size=len(group_ranks[0]))

    monkeypatch.setattr(hunyuan_moe, "vllm_init_model_parallel_group", fake_vllm_init_model_parallel_group)

    ep_group_ranks = [[0, 1, 4, 5], [2, 3, 6, 7]]
    hunyuan_moe._init_mc2_group_for_diffusion(
        world_size=8,
        expert_parallel_size=4,
        backend="hccl",
        local_rank=0,
        group_ranks=ep_group_ranks,
    )

    assert captured == {
        "group_ranks": ep_group_ranks,
        "local_rank": 0,
        "backend": "hccl",
        "group_name": "mc2",
    }
    assert ascend_parallel_state._MC2.ranks == ep_group_ranks[0]
