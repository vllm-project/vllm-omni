# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.distributed import a2a_permute

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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

    monkeypatch.setattr(
        a2a_permute.symm_mem,
        "_SymmetricMemory",
        SimpleNamespace(
            empty_strided_p2p=lambda shape, stride, dtype, device: empty(shape[0], dtype=dtype, device=device)
        ),
    )
    monkeypatch.setattr(a2a_permute.symm_mem, "rendezvous", lambda allocation, _group: _FakeHandle(allocation))

    device = torch.device("cuda:0")
    first = a2a_permute._get_symm_buffer((2, 3), torch.float16, device, "group")
    handle = a2a_permute._SYMM_WORKSPACES[(device, "group")].handle
    assert a2a_permute._get_symm_buffer((2, 3), torch.float16, device, "group") is first
    assert handle.views == [((2, 3), torch.float16)]
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


@pytest.mark.parametrize("size,batch", [(2, 1), (3, 2), (4, 2)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ce_direct_copies_match_independent_4d_exchange(monkeypatch, size, batch, dtype):
    """Execute peer writes on CPU, including batched, strided and GQA regions."""

    class Stream:
        cuda_stream = 17

        def wait_event(self, event):
            pass

    class Event:
        def record(self, stream):
            pass

    def copy2d(dst, dpitch, src, spitch, width, height, kind, stream):
        assert kind == 4 and stream == 17
        for row in range(height):
            ctypes.memmove(dst + row * dpitch, src + row * spitch, width)
        return 0

    def copy3d(pointer, stream):
        args = pointer._obj
        for plane in range(args.extent.depth):
            copy2d(
                args.dstPtr.ptr + plane * args.dstPtr.pitch * args.dstPtr.ysize,
                args.dstPtr.pitch,
                args.srcPtr.ptr + plane * args.srcPtr.pitch * args.srcPtr.ysize,
                args.srcPtr.pitch,
                args.extent.width,
                args.extent.height,
                args.kind,
                stream,
            )
        return 0

    def check(result):
        assert result == 0

    monkeypatch.setattr(
        a2a_permute,
        "_ce_runtime",
        lambda: SimpleNamespace(
            lib=SimpleNamespace(cudaMemcpy2DAsync=copy2d, cudaMemcpy3DAsync=copy3d), CUDART_CHECK=check
        ),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: Stream())
    monkeypatch.setattr(torch.cuda, "Stream", lambda device: Stream())
    monkeypatch.setattr(torch.cuda, "Event", Event)
    assert ctypes.sizeof(a2a_permute._Memcpy3DParams) == 160
    for reverse in (False, True):
        shapes = [(batch, 7 * size, 2, 3)] if reverse else [(batch, 7, h * size, 3) for h in (2, 1, 1)]
        shared = [torch.empty(sum(torch.Size(shape).numel() for shape in shapes), dtype=dtype) for _ in range(size)]
        plans = []
        # Reuse plans with new source pointers and values on the second exchange.
        for iteration in range(2):
            inputs = [
                tuple(
                    (
                        torch.arange(torch.Size(shape).numel() * 3).reshape(*shape[:2], 3, *shape[2:]) % 97
                        + rank
                        + index
                        + iteration * 10
                    ).to(dtype)[:, :, 1]
                    for index, shape in enumerate(shapes)
                )
                for rank in range(size)
            ]
            for rank in range(size):
                if not iteration:
                    handle = SimpleNamespace(
                        rank=rank, buffer_ptrs=[x.data_ptr() for x in shared], barrier=lambda: None
                    )
                    plans.append(a2a_permute._CEPlan(inputs[rank], shared[rank], handle, size, reverse, Stream()))
                # Graph capture records transport only; eager calls own their output.
                result = plans[rank]._run_eager(inputs[rank], copy_out=bool(iteration))
                if iteration:
                    assert result.data_ptr() != shared[rank].data_ptr()
                else:
                    assert result is None
            for rank in range(size):
                for index, (part, shape) in enumerate(
                    zip(shared[rank].split([torch.Size(s).numel() for s in shapes]), shapes)
                ):
                    b, sequence, heads, dim = shape
                    axis, gather = (1, 2) if reverse else (2, 1)
                    expected = torch.cat([x[index].chunk(size, axis)[rank] for x in inputs], dim=gather)
                    output_shape = (
                        (b, sequence // size, heads * size, dim)
                        if reverse
                        else (b, sequence * size, heads // size, dim)
                    )
                    torch.testing.assert_close(part.view(output_shape), expected, rtol=0, atol=0)


def test_ce_fake_ops_preserve_functional_output_metadata():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        x = torch.empty(2, 17, 8, 64, dtype=torch.bfloat16, device="cuda")
        packed = a2a_permute.ulysses_qkv_ce(x, x, x, "fake-only", 2)
        assert packed.shape == (3 * x.numel(),) and packed.dtype == x.dtype
        y = packed[: x.numel()].view(2, 34, 4, 64)
        z = a2a_permute.ulysses_o_rev(y, "fake-only", 2)
        assert y.shape == (2, 34, 4, 64)
        assert z.shape == x.shape and z.dtype == x.dtype and z.device == x.device
