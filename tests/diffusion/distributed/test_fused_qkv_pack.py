# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Layout-only fusion eligibility and exact four-rank exchange coverage."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch._subclasses.fake_tensor import FakeTensorMode

from tests.helpers.distributed import start_collective_workers
from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import get_open_port
from vllm_omni.diffusion.distributed import packed_qkv
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.diffusion, pytest.mark.parallel]


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.int32])
def test_fused_eligibility_accepts_independent_positive_strides(monkeypatch, dtype):
    monkeypatch.setattr(torch.version, "hip", "test-rocm")
    with FakeTensorMode():
        q = torch.empty((2, 7, 8, 4), dtype=dtype, device="cuda")
        k = torch.empty((2, 14, 8, 4), dtype=dtype, device="cuda")[:, ::2]
        v = torch.empty((2, 7, 8, 4, 3), dtype=dtype, device="cuda")[..., 2]
        assert packed_qkv._fused_qkv_eligible(q, k, v, 4)


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.parametrize("case", ["negative", "zero_stride", "wire_overflow", "stride_overflow", "double", "grad"])
def test_fused_eligibility_rejects_unsafe_views(monkeypatch, case):
    monkeypatch.setattr(torch.version, "hip", "test-rocm")
    with FakeTensorMode():
        q = torch.empty((2, 7, 8, 4), dtype=torch.float32, device="cuda")
        if case == "negative":
            q = torch._neg_view(q)
            assert q.is_neg()
        elif case == "zero_stride":
            q = q[:1].expand(2, -1, -1, -1)
        elif case == "wire_overflow":
            q = torch.empty((1, 1, 8, 2**27), dtype=torch.float32, device="cuda")
        elif case == "stride_overflow":
            q = torch.empty_strided((1, 3, 8, 4), (64, 2**30, 4, 1), dtype=torch.float32, device="cuda")
        elif case == "double":
            q = q.to(torch.float64)
        else:
            q.requires_grad_()
        assert not packed_qkv._fused_qkv_eligible(q, q, q, 4)


@pytest.mark.core_model
@pytest.mark.cpu
def test_fused_eligibility_rejects_cpu_and_non_rocm(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "test-rocm")
    q = torch.empty((1, 7, 8, 4))
    assert not packed_qkv._fused_qkv_eligible(q, q, q, 4)
    monkeypatch.setattr(torch.version, "hip", None)
    with FakeTensorMode():
        q = torch.empty((1, 7, 8, 4), device="cuda")
        assert not packed_qkv._fused_qkv_eligible(q, q, q, 4)


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.parametrize("slot", [0, 1, 2])
def test_fused_eligibility_checks_each_lazy_producer(monkeypatch, slot):
    monkeypatch.setattr(torch.version, "hip", "test-rocm")
    with FakeTensorMode():
        producers = [torch.empty((2, 7, 8, 4), device="cuda") for _ in range(3)]
        producers[slot] = torch._neg_view(producers[slot])
        assert not packed_qkv._fused_qkv_eligible(*producers, 4)


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.parametrize("flag,eligible,has_triton", [("0", True, True), ("1", False, True), ("1", True, False)])
def test_disabled_ineligible_or_missing_triton_retains_copy_path(monkeypatch, flag, eligible, has_triton):
    from vllm_omni.diffusion import distributed

    monkeypatch.setenv("VLLM_OMNI_ROCM_FUSED_QKV_PACK", flag)
    monkeypatch.setattr(packed_qkv, "_fused_qkv_eligible", lambda *args: eligible)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(dist, "all_to_all_single", lambda receive, send, group: receive.copy_(send))

    def unexpected_kernel(*args):
        pytest.fail("fallback must not invoke a Triton kernel")

    module = SimpleNamespace(HAS_TRITON=has_triton, _pack_qkv=unexpected_kernel)
    monkeypatch.setattr(distributed, "fused_qkv_pack", module, raising=False)
    q = torch.arange(2 * 7 * 8 * 4, dtype=torch.float32).reshape(2, 7, 8, 4)
    actual = packed_qkv.packed_qkv_all_to_all(object(), q, q + 1, q + 2)
    for slot, output in enumerate(actual):
        expected = torch.cat((q + slot).chunk(2, dim=2), dim=1)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.parametrize("flag", ["0", "1"])
def test_lazy_negative_cpu_fallback_copies_logical_values(monkeypatch, flag):
    """A local wire stub checks fallback values; this is not a collective test."""
    monkeypatch.setenv("VLLM_OMNI_ROCM_FUSED_QKV_PACK", flag)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    calls = []

    def copy_local_wire(receive, send, group):
        calls.append(1)
        receive.copy_(send)

    monkeypatch.setattr(dist, "all_to_all_single", copy_local_wire)
    q = torch.arange(2 * 7 * 8 * 4, dtype=torch.float32).reshape(2, 7, 8, 4)
    producers = (q, torch._neg_view(q + 1), q + 2)
    assert producers[1].is_neg()
    actual = packed_qkv.packed_qkv_all_to_all(object(), *producers)
    assert len(calls) == 1
    for output, producer in zip(actual, producers):
        expected = torch.cat(producer.chunk(2, dim=2), dim=1)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


def _assert_bits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def _inputs(shape, dtype, layout, device, rank, raw_bits=False):
    elements = 3
    for size in shape:
        elements *= size
    if raw_bits:
        encodings = {
            torch.bfloat16: [0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0xFFC2, 0x7F81, 0xFF82, 0x0001, 0x8001, 0x7F7F],
            torch.float16: [0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E01, 0xFE02, 0x7C01, 0xFC02, 0x0001, 0x8001, 0x7BFF],
            torch.float32: [
                0x00000000,
                0x80000000,
                0x7F800000,
                0xFF800000,
                0x7FC00001,
                0xFFC00002,
                0x7F800001,
                0xFF800002,
                0x00000001,
                0x80000001,
                0x7F7FFFFF,
            ],
        }[dtype]
        bits = torch.arange(elements, dtype=torch.int64) * 104729 + rank * 17
        for index, encoding in enumerate(encodings):
            bits[index::16] = encoding
        words = bits.to(torch.int32 if dtype == torch.float32 else torch.int16)
        data = words.view(dtype).reshape(*shape, 3).to(device)
    else:
        data = (torch.arange(elements, dtype=torch.int32) + rank * 1000000).reshape(*shape, 3).to(device, dtype)
    if layout == "contiguous":
        return tuple(data[..., slot].contiguous() for slot in range(3))
    if layout == "projection":
        data = data.permute(0, 1, 2, 4, 3).contiguous()
        return tuple(data[..., slot, :] for slot in range(3))
    if layout == "interleaved":
        return tuple(data[..., slot] for slot in range(3))
    if layout == "row_strided":
        storage = torch.empty((shape[0], 2 * shape[1], shape[2], shape[3], 3), dtype=dtype, device=device)
        storage[:, ::2].copy_(data)
        return tuple(storage[:, ::2, ..., slot] for slot in range(3))
    if layout == "head_transposed":
        data = data.permute(0, 2, 1, 3, 4).contiguous().permute(0, 2, 1, 3, 4)
        return tuple(data[..., slot] for slot in range(3))
    assert layout == "distinct_strides"
    q = data[..., 0].contiguous()
    storage = torch.empty((shape[0], 2 * shape[1], shape[2], shape[3]), dtype=dtype, device=device)
    k = storage[:, ::2]
    k.copy_(data[..., 1])
    return q, k, data[..., 2]


def _reference_wire(producers, world_size):
    batch, tokens, heads, width = producers[0].shape
    wire = torch.empty(
        (world_size, batch, tokens, heads // world_size, 3, width),
        dtype=producers[0].dtype,
        device=producers[0].device,
    )
    for slot, tensor in enumerate(producers):
        wire[..., slot, :].copy_(
            tensor.reshape(batch, tokens, world_size, heads // world_size, width).permute(2, 0, 1, 3, 4)
        )
    return wire


def _run_fused_exchange(rank, world_size, port):
    from vllm_omni.diffusion.distributed import fused_qkv_pack

    device = torch.device(f"{current_omni_platform.device_type}:{rank}")
    current_omni_platform.set_device(device)
    os.environ["VLLM_OMNI_ROCM_FUSED_QKV_PACK"] = "1"
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)
    cases = [
        ((1, 7, 8, 5), torch.int32, "contiguous", False),
        ((2, 11, 8, 7), torch.int32, "row_strided", False),
        ((1, 13, 56, 128), torch.bfloat16, "projection", False),
        ((2, 17, 56, 16), torch.bfloat16, "interleaved", False),
        ((2, 19, 56, 16), torch.float16, "row_strided", False),
        ((2, 23, 8, 7), torch.float32, "head_transposed", False),
        ((2, 17, 8, 7), torch.bfloat16, "distinct_strides", False),
        ((1, 19, 8, 7), torch.bfloat16, "interleaved", True),
        ((1, 19, 8, 7), torch.float16, "projection", True),
        ((1, 19, 8, 7), torch.float32, "row_strided", True),
        ((1, 1032, 56, 128), torch.bfloat16, "projection", False),
    ]
    try:
        with torch.inference_mode():
            for shape, dtype, layout, bits in cases:
                qkv = _inputs(shape, dtype, layout, device, rank, bits)
                assert packed_qkv._fused_qkv_eligible(*qkv, world_size)
                wire = _reference_wire(qkv, world_size)
                _assert_bits(fused_qkv_pack._pack_qkv(*qkv, world_size), wire)
                collective = dist.all_to_all_single
                calls = []

                def counted(*args, **kwargs):
                    calls.append(1)
                    return collective(*args, **kwargs)

                with pytest.MonkeyPatch.context() as patch:
                    patch.setattr(dist, "all_to_all_single", counted)
                    actual = packed_qkv.packed_qkv_all_to_all(dist.group.WORLD, *qkv)
                    assert len(calls) == 1
                    reverse = packed_qkv.packed_output_all_to_all(dist.group.WORLD, actual[0])
                    assert len(calls) == 2
                assert len({tensor.untyped_storage().data_ptr() for tensor in actual}) == 3
                for output, producer in zip(actual, qkv):
                    assert output.is_contiguous()
                    gathered = [torch.empty_like(producer.contiguous()) for _ in range(world_size)]
                    dist.all_gather(gathered, producer.contiguous())
                    local_heads = shape[2] // world_size
                    expected = torch.cat(
                        [tensor[:, :, rank * local_heads : (rank + 1) * local_heads] for tensor in gathered], dim=1
                    )
                    _assert_bits(output, expected)
                _assert_bits(reverse, qkv[0])
                retained = tuple(tensor.clone() for tensor in actual)
                packed_qkv.packed_qkv_all_to_all(dist.group.WORLD, *qkv)
                for output, saved in zip(actual, retained):
                    _assert_bits(output, saved)

            # The guarded fallback must resolve lazy negative producer views.
            q, k, v = _inputs((2, 17, 8, 7), torch.float32, "distinct_strides", device, rank)
            k = torch._neg_view(k)
            assert not packed_qkv._fused_qkv_eligible(q, k, v, world_size)
            outputs = packed_qkv.packed_qkv_all_to_all(dist.group.WORLD, q, k, v)
            gathered = [torch.empty_like(k.contiguous()) for _ in range(world_size)]
            dist.all_gather(gathered, k.contiguous())
            expected = torch.cat([tensor[:, :, rank * 2 : (rank + 1) * 2] for tensor in gathered], dim=1)
            _assert_bits(outputs[1], expected)

            # Producer, collective, consumer and inverse all execute on a
            # nondefault stream; synchronization happens only after use.
            stream = torch.Stream(device=device)
            with stream:
                qkv = _inputs((1, 13, 8, 7), torch.int32, "distinct_strides", device, rank)
                outputs = packed_qkv.packed_qkv_all_to_all(dist.group.WORLD, *qkv)
                reverse = packed_qkv.packed_output_all_to_all(dist.group.WORLD, outputs[0])
            stream.synchronize()
            _assert_bits(reverse, qkv[0])
    finally:
        dist.destroy_process_group()


@pytest.mark.full_model
@pytest.mark.parametrize("world_size", [pytest.param(4, marks=hardware_marks(res={"rocm": "MI325"}, num_cards=4))])
def test_fused_packed_qkv_four_rank_exactness(world_size):
    if not current_omni_platform.is_rocm():
        pytest.skip("fused packed QKV requires ROCm")
    if current_omni_platform.get_device_count() < world_size:
        pytest.skip(f"requires {world_size} ROCm GPUs")
    start_collective_workers(
        _run_fused_exchange,
        args=(world_size, get_open_port()),
        nprocs=world_size,
        device_kind="cuda",
    )
