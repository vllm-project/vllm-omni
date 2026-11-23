# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU coverage for platform selection and safe packed-QKV eligibility."""

import pytest
import torch

from vllm_omni.diffusion import envs
from vllm_omni.diffusion.attention.parallel import ulysses
from vllm_omni.diffusion.distributed import packed_qkv
from vllm_omni.diffusion.distributed.group_coordinator import SequenceParallelGroupCoordinator
from vllm_omni.diffusion.distributed.packed_qkv import packed_qkv_eligible
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("flag", [None, "0", "1", "other"])
@pytest.mark.parametrize("platform_kind", list(OmniPlatformEnum))
def test_packed_qkv_selection(monkeypatch, mocker, flag, platform_kind):
    if flag is None:
        monkeypatch.delenv("VLLM_OMNI_PACKED_QKV", raising=False)
    else:
        monkeypatch.setenv("VLLM_OMNI_PACKED_QKV", flag)
    platform = OmniPlatform()
    platform._omni_enum = platform_kind
    monkeypatch.setattr(ulysses, "current_omni_platform", platform)
    group = mocker.Mock(spec=SequenceParallelGroupCoordinator)
    group.ulysses_group = object()
    group.ulysses_world_size = 4
    group.ring_world_size = group.allgather_world_size = 1
    strategy = ulysses.UlyssesParallelAttention(group, 2, 1, False)
    assert envs.VLLM_OMNI_PACKED_QKV is (flag == "1")
    assert strategy._packed_qkv is (flag == "1")


@pytest.mark.parametrize("device", ["cpu", "cuda", "xpu", "privateuseone"])
@pytest.mark.parametrize("use_sync", [False, True])
def test_packed_qkv_sync_uses_platform_backend(monkeypatch, mocker, device, use_sync):
    platform = mocker.Mock(spec=OmniPlatform)
    monkeypatch.setattr(packed_qkv, "current_omni_platform", platform)
    tensor = mocker.Mock(spec=torch.Tensor)
    tensor.device = torch.device(device)
    packed_qkv._synchronize(tensor, use_sync)
    if use_sync and device != "cpu":
        platform.synchronize.assert_called_once_with()
    else:
        platform.synchronize.assert_not_called()


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
