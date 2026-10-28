# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

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
    monkeypatch.setattr(a2a_permute, "current_omni_platform", SimpleNamespace(is_npu=lambda: False))

    a2a_permute.ensure_a2a_permute_available()

    assert set(load_kwargs["extra_include_paths"]) == {str(cu13_include), str(nccl_include)}
    assert load_kwargs["extra_ldflags"] == [str(nccl_lib)]


def test_npu_initialization_never_builds_cuda(monkeypatch):
    import sys

    calls = []
    module = SimpleNamespace(ensure_a2a_permute_available=lambda: calls.append("npu"))
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion.distributed.ulysses_shmem_npu", module)
    monkeypatch.setattr(a2a_permute, "current_omni_platform", SimpleNamespace(is_npu=lambda: True))

    def unexpected():
        pytest.fail("NPU initialization must not compile CUDA")

    monkeypatch.setattr(a2a_permute, "_ensure_built", unexpected)
    a2a_permute.ensure_a2a_permute_available()
    assert calls == ["npu"]


def test_grouped_npu_dispatch_preserves_fused_qkv(monkeypatch):
    import sys

    query, key, value = (SimpleNamespace(device=SimpleNamespace(type="npu")) for _ in range(3))
    outputs = (object(), object(), object())

    def forward(q, k, v, group_name, world_size):
        assert (q, k, v) == (query, key, value)
        assert (group_name, world_size) == ("ulysses", 4)
        return outputs

    module = SimpleNamespace(ulysses_qkv_fwd_group=forward)
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion.distributed.ulysses_shmem_npu", module)
    assert a2a_permute.ulysses_qkv_fwd_group(query, key, value, "ulysses", 4) is outputs


def test_shared_fake_metadata_supports_gqa_and_batch():
    with FakeTensorMode():
        query = torch.empty(2, 17, 8, 32, dtype=torch.bfloat16)
        key = torch.empty(2, 17, 4, 32, dtype=torch.bfloat16)
        value = torch.empty_like(key)
        q, k, v = a2a_permute.ulysses_qkv_fwd_group(query, key, value, "ulysses", 4)
        assert q.shape == (2, 68, 2, 32)
        assert k.shape == v.shape == (2, 68, 1, 32)
        output = a2a_permute.ulysses_o_rev(q, "ulysses", 4)
        assert output.shape == query.shape
        assert output.dtype == query.dtype


@dataclass
class _FakeAllocation:
    """Byte workspace stand-in that slices and reinterprets like ``symm_mem.empty()``."""

    size: int
    device: torch.device

    def __post_init__(self) -> None:
        self.buffer = torch.empty(self.size, dtype=torch.uint8)

    def __getitem__(self, item: slice) -> torch.Tensor:
        return self.buffer[item]


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
    # The handle is only retained to keep the rendezvous registration alive.
    monkeypatch.setattr(a2a_permute.symm_mem, "rendezvous", lambda _allocation, _group: object())

    device = torch.device("cuda:0")
    first = a2a_permute._get_symm_buffer((2, 3), torch.float16, device, "group")
    smaller = a2a_permute._get_symm_buffer((1, 4), torch.float16, device, "group")
    larger = a2a_permute._get_symm_buffer((4, 3), torch.float16, device, "group")

    assert (first.shape, first.dtype) == ((2, 3), torch.float16)
    assert (smaller.shape, smaller.dtype) == ((1, 4), torch.float16)
    assert (larger.shape, larger.dtype) == ((4, 3), torch.float16)
    # A sub-capacity request reuses the workspace; only growth allocates again.
    assert smaller.untyped_storage().data_ptr() == first.untyped_storage().data_ptr()
    assert larger.untyped_storage().data_ptr() != first.untyped_storage().data_ptr()
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
