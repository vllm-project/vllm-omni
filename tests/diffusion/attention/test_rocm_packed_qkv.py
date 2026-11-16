# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU coverage for platform selection and safe packed-QKV eligibility."""

import pytest
import torch

from vllm_omni.diffusion import envs
from vllm_omni.diffusion.attention.parallel import ulysses
from vllm_omni.diffusion.distributed.group_coordinator import SequenceParallelGroupCoordinator
from vllm_omni.diffusion.distributed.packed_qkv import packed_qkv_eligible
from vllm_omni.platforms.interface import OmniPlatform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("flag", [None, "0", "1", "other"])
@pytest.mark.parametrize("rocm", [False, True])
def test_rocm_packed_qkv_selection(monkeypatch, mocker, flag, rocm):
    if flag is None:
        monkeypatch.delenv("VLLM_OMNI_ROCM_PACKED_QKV", raising=False)
    else:
        monkeypatch.setenv("VLLM_OMNI_ROCM_PACKED_QKV", flag)
    platform = mocker.Mock(spec=OmniPlatform)
    platform.is_rocm.return_value = rocm
    monkeypatch.setattr(ulysses, "current_omni_platform", platform)
    group = mocker.Mock(spec=SequenceParallelGroupCoordinator)
    group.ulysses_group = object()
    group.ulysses_world_size = 4
    group.ring_world_size = group.allgather_world_size = 1
    strategy = ulysses.UlyssesParallelAttention(group, 2, 1, False)
    assert envs.VLLM_OMNI_ROCM_PACKED_QKV is (flag == "1")
    assert strategy._rocm_packed_qkv is (rocm and flag == "1")


def test_packed_qkv_eligibility_preserves_unsupported_paths():
    q, k, v = (torch.randn(2, 11, 8, 4) for _ in range(3))
    assert packed_qkv_eligible(q, k, v, 4)
    assert not packed_qkv_eligible(q, k, v, 1)
    assert not packed_qkv_eligible(q, k[:, :, :4], v[:, :, :4], 4)
    assert not packed_qkv_eligible(q, k.to(torch.float64), v, 4)
    assert not packed_qkv_eligible(q, k, v, 3)
    assert not packed_qkv_eligible(q[:, 0], k[:, 0], v[:, 0], 4)
    assert not packed_qkv_eligible(q[:, :0], k[:, :0], v[:, :0], 4)
    assert not packed_qkv_eligible(q.requires_grad_(), k, v, 4)
