# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.distributed import a2a_permute

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_jit_build_includes_cuda_headers_from_nvidia_wheels(tmp_path, monkeypatch) -> None:
    cu13_include = tmp_path / "nvidia" / "cu13" / "include"
    nccl_include = tmp_path / "nvidia" / "nccl" / "include"
    nccl_lib = tmp_path / "nvidia" / "nccl" / "lib" / "libnccl.so.2"
    cu13_include.mkdir(parents=True)
    nccl_include.mkdir(parents=True)
    nccl_lib.parent.mkdir(parents=True)
    (cu13_include / "cusparse.h").touch()
    (nccl_include / "nccl.h").touch()
    nccl_lib.touch()

    load_kwargs = {}
    monkeypatch.setattr(a2a_permute.sysconfig, "get_paths", lambda: {"purelib": str(tmp_path)})
    monkeypatch.setattr(
        torch.utils.cpp_extension,
        "load",
        lambda **kwargs: load_kwargs.update(kwargs),
    )
    monkeypatch.setattr(a2a_permute.symm_mem, "set_backend", lambda _backend: None)
    monkeypatch.setattr(a2a_permute, "_BUILT", False)

    a2a_permute.ensure_a2a_permute_available()

    assert set(load_kwargs["extra_include_paths"]) == {str(cu13_include), str(nccl_include)}
    assert load_kwargs["extra_ldflags"] == [str(nccl_lib)]


@dataclass
class _FakeAllocation:
    size: int
    device: torch.device


class _FakeHandle:
    rank = 0

    def __init__(self, allocation: _FakeAllocation) -> None:
        self.allocation = allocation
        self.views: list[tuple[tuple[int, ...], torch.dtype]] = []

    def get_buffer(self, rank, shape, dtype):
        assert rank == self.rank
        self.views.append((shape, dtype))
        return self.allocation, shape, dtype


def test_workspace_reuses_peak_capacity_across_shapes(monkeypatch) -> None:
    allocations: list[_FakeAllocation] = []
    all_reduce_calls: list[object] = []
    synchronize_calls: list[torch.device] = []
    stream_id = [1]
    a2a_permute._SYMM_WORKSPACES.clear()

    monkeypatch.setattr(a2a_permute.torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        a2a_permute.torch.cuda,
        "current_stream",
        lambda _device: SimpleNamespace(cuda_stream=stream_id[0]),
    )
    monkeypatch.setattr(a2a_permute, "_resolve_process_group", lambda _name: object())
    monkeypatch.setattr(a2a_permute.torch, "ones", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        a2a_permute.dist,
        "all_reduce",
        lambda _tensor, group: all_reduce_calls.append(group),
    )
    monkeypatch.setattr(
        a2a_permute.torch.accelerator,
        "synchronize",
        lambda device: synchronize_calls.append(device),
    )

    def empty(size, *, dtype, device):
        assert dtype == torch.uint8
        allocation = _FakeAllocation(size=size, device=torch.device(device))
        allocations.append(allocation)
        return allocation

    monkeypatch.setattr(a2a_permute.symm_mem, "empty", empty)
    monkeypatch.setattr(a2a_permute.symm_mem, "rendezvous", lambda allocation, _group: _FakeHandle(allocation))

    device = torch.device("cuda:0")
    first = a2a_permute._get_symm_buffer((2, 3), torch.float16, device, "group")
    smaller = a2a_permute._get_symm_buffer((1, 4), torch.float16, device, "group")
    larger = a2a_permute._get_symm_buffer((4, 3), torch.float16, device, "group")

    assert first[0].size == 12
    assert smaller[0] is first[0]
    assert larger[0].size == 24
    assert [allocation.size for allocation in allocations] == [12, 24]
    assert len(a2a_permute._SYMM_WORKSPACES) == 1
    assert len(all_reduce_calls) == 1
    assert synchronize_calls == [device, device]

    stream_id[0] = 2
    with pytest.raises(RuntimeError, match="single CUDA stream"):
        a2a_permute._get_symm_buffer((1, 4), torch.float16, device, "group")

    a2a_permute.clear_a2a_permute_workspaces()
    assert not a2a_permute._SYMM_WORKSPACES
    assert synchronize_calls == [device, device, device]
