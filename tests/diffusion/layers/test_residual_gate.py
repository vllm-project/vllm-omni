# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.diffusion,
    pytest.mark.skipif(not current_platform.is_cuda(), reason="NVIDIA CUDA platform required"),
]


@pytest.fixture(autouse=True)
def _clear_runtime_failure_cache():
    import vllm_omni.diffusion.layers.residual_gate as residual_gate

    residual_gate._FAILED_RUNTIME_KEYS.clear()
    yield
    residual_gate._FAILED_RUNTIME_KEYS.clear()


def _make_residual(shape, layout, dtype, *, storage_offset=0):
    batch, tokens, hidden_size = shape
    if layout == "contiguous":
        if storage_offset:
            storage = torch.randn(storage_offset + batch * tokens * hidden_size, device="cuda", dtype=dtype)
            return torch.as_strided(
                storage,
                shape,
                (tokens * hidden_size, hidden_size, 1),
                storage_offset,
            )
        return torch.randn(shape, device="cuda", dtype=dtype)

    if layout == "transposed":
        if storage_offset:
            storage = torch.randn(storage_offset + batch * tokens * hidden_size, device="cuda", dtype=dtype)
            return torch.as_strided(
                storage,
                shape,
                (tokens * hidden_size, 1, tokens),
                storage_offset,
            )
        return torch.randn(batch, hidden_size, tokens, device="cuda", dtype=dtype).permute(0, 2, 1)
    raise ValueError(layout)


def _make_update(shape, dtype, *, storage_offset=0):
    if not storage_offset:
        return torch.randn(shape, device="cuda", dtype=dtype)
    storage = torch.randn(storage_offset + torch.Size(shape).numel(), device="cuda", dtype=dtype)
    return torch.as_strided(
        storage,
        shape,
        (shape[1] * shape[2], shape[2], 1),
        storage_offset,
    )


def _make_gate(shape, mode, dtype):
    batch, tokens, hidden_size = shape
    if mode == "shared":
        return torch.randn(1, 1, hidden_size, device="cuda", dtype=dtype)

    # SANA's gates are slices of a contiguous [B, N, 6, D] modulation
    # tensor, so consecutive rows are separated by 6 * D elements.
    gate_batch = 1 if mode == "shared_tokenwise" else batch
    gate_tokens = 1 if mode == "batch" else tokens
    modulation = torch.randn(gate_batch, gate_tokens, 6, hidden_size, device="cuda", dtype=dtype)
    return modulation.unbind(dim=2)[2]


def test_int32_indexing_guard_rejects_oversized_gate_stride():
    import vllm_omni.diffusion.layers.residual_gate as residual_gate

    residual = torch.empty(1, 2, 1, device="meta")
    safe_gate = torch.empty_strided((1, 2, 1), (2, 1, 1), device="meta")
    oversized_gate = torch.empty_strided(
        (1, 2, 1),
        (2**32, 2**31, 1),
        device="meta",
    )

    assert residual_gate._fits_int32_indexing(residual, safe_gate)
    assert not residual_gate._fits_int32_indexing(residual, oversized_gate)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["contiguous", "transposed"])
@pytest.mark.parametrize("gate_mode", ["shared", "batch", "shared_tokenwise", "tokenwise"])
@pytest.mark.parametrize("shape", [(2, 37, 68), (2, 65, 96)])
@pytest.mark.parametrize("storage_offset", [0, 7])
@pytest.mark.parametrize("update_storage_offset", [0, 11])
def test_fused_residual_gate_add_is_bit_exact_and_preserves_stride(
    dtype,
    layout,
    gate_mode,
    shape,
    storage_offset,
    update_storage_offset,
):
    import vllm_omni.diffusion.layers.residual_gate as residual_gate

    torch.manual_seed(17)
    residual = _make_residual(shape, layout, dtype, storage_offset=storage_offset)
    update = _make_update(shape, dtype, storage_offset=update_storage_offset)
    gate = _make_gate(shape, gate_mode, dtype)

    expected = residual + gate * update
    actual = residual_gate._launch_fused_residual_gate_add(residual, update, gate)

    assert residual_gate._can_use_fused_residual_gate_add(residual, update, gate)
    assert actual.stride() == residual.stride()
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["contiguous", "transposed"])
def test_fused_residual_gate_add_is_bit_exact_for_special_values(dtype, layout):
    import vllm_omni.diffusion.layers.residual_gate as residual_gate

    shape = (2, 3, 8)
    smallest_subnormal = 2**-133 if dtype == torch.bfloat16 else 2**-24
    residual_row = torch.tensor(
        [-0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0],
        device="cuda",
        dtype=dtype,
    )
    update_row = torch.tensor(
        [
            -0.0,
            0.0,
            smallest_subnormal,
            -smallest_subnormal,
            float("inf"),
            float("-inf"),
            torch.finfo(dtype).max,
            float("nan"),
        ],
        device="cuda",
        dtype=dtype,
    )
    gate_row = torch.tensor(
        [1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 2.0, 1.0],
        device="cuda",
        dtype=dtype,
    )
    residual = residual_row.expand(shape).clone()
    if layout == "transposed":
        residual = residual.transpose(1, 2).contiguous().transpose(1, 2)
    update = update_row.expand(shape).contiguous()
    gate = gate_row.expand(shape)

    expected = residual + gate * update
    actual = residual_gate._launch_fused_residual_gate_add(residual, update, gate)

    assert residual_gate._can_use_fused_residual_gate_add(residual, update, gate)
    # Compare representations so signed zeros and NaN payloads are checked.
    assert torch.equal(actual.contiguous().view(torch.int16), expected.contiguous().view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("unsupported_input", ["update", "gate"])
def test_residual_gate_add_falls_back_for_unsupported_layout(monkeypatch, unsupported_input):
    import vllm_omni.diffusion.layers.residual_gate as residual_gate

    residual = torch.randn(2, 19, 32, device="cuda", dtype=torch.bfloat16)
    update = torch.randn_like(residual)
    gate = torch.randn(2, 1, 32, device="cuda", dtype=torch.bfloat16)
    if unsupported_input == "update":
        update_storage = torch.randn(2, 19, 64, device="cuda", dtype=torch.bfloat16)
        update = update_storage[..., ::2]
        assert not update.is_contiguous()
    else:
        gate_storage = torch.randn(2, 1, 64, device="cuda", dtype=torch.bfloat16)
        gate = gate_storage[..., ::2]
        assert gate.stride(2) == 2

    launch_calls = 0

    def counting_launch(residual, update, gate):
        nonlocal launch_calls
        launch_calls += 1
        return residual + gate * update

    monkeypatch.setattr(residual_gate, "_launch_fused_residual_gate_add", counting_launch)
    assert not residual_gate._can_use_fused_residual_gate_add(residual, update, gate)
    expected = residual + gate * update
    actual = residual_gate.residual_gate_add(residual, update, gate)
    assert launch_calls == 0
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_residual_gate_add_dispatches_supported_inputs(monkeypatch):
    import vllm_omni.diffusion.layers.residual_gate as residual_gate

    shape = (2, 37, 68)
    residual = _make_residual(shape, "transposed", torch.bfloat16)
    update = _make_update(shape, torch.bfloat16, storage_offset=11)
    gate = _make_gate(shape, "tokenwise", torch.bfloat16)
    original_launch = residual_gate._launch_fused_residual_gate_add
    launch_calls = 0

    def counting_launch(residual, update, gate):
        nonlocal launch_calls
        launch_calls += 1
        return original_launch(residual, update, gate)

    monkeypatch.setattr(residual_gate, "_launch_fused_residual_gate_add", counting_launch)
    expected = residual + gate * update
    actual = residual_gate.residual_gate_add(residual, update, gate)

    assert launch_calls == 1
    assert not residual_gate._FAILED_RUNTIME_KEYS
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_residual_gate_add_caches_runtime_failure(monkeypatch):
    import vllm_omni.diffusion.layers.residual_gate as residual_gate

    residual = torch.randn(2, 19, 32, device="cuda", dtype=torch.bfloat16)
    update = torch.randn_like(residual)
    gate = torch.randn(2, 1, 32, device="cuda", dtype=torch.bfloat16)
    launch_calls = 0

    def failing_launch(residual, update, gate):
        nonlocal launch_calls
        launch_calls += 1
        raise RuntimeError("synthetic Triton launch failure")

    monkeypatch.setattr(residual_gate, "_launch_fused_residual_gate_add", failing_launch)
    expected = residual + gate * update

    first = residual_gate.residual_gate_add(residual, update, gate)
    second = residual_gate.residual_gate_add(residual, update, gate)

    assert launch_calls == 1
    assert (residual.device.index, residual.dtype) in residual_gate._FAILED_RUNTIME_KEYS
    assert torch.equal(first, expected)
    assert torch.equal(second, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_residual_gate_add_falls_back_when_grad_is_required():
    import vllm_omni.diffusion.layers.residual_gate as residual_gate

    residual = torch.randn(2, 19, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    update = torch.randn_like(residual)
    gate = torch.randn(2, 1, 32, device="cuda", dtype=torch.bfloat16)

    assert not residual_gate._can_use_fused_residual_gate_add(residual, update, gate)
    actual = residual_gate.residual_gate_add(residual, update, gate)
    actual.float().sum().backward()
    assert residual.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("layout", ["contiguous", "transposed"])
@pytest.mark.parametrize("gate_mode", ["batch", "tokenwise"])
def test_residual_gate_add_compiles_fullgraph(layout, gate_mode):
    from vllm_omni.diffusion.layers.residual_gate import residual_gate_add

    shape = (2, 37, 68)
    residual = _make_residual(shape, layout, torch.bfloat16)
    update = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    gate = _make_gate(shape, gate_mode, torch.bfloat16)
    expected = residual + gate * update

    compiled = torch.compile(
        residual_gate_add,
        fullgraph=True,
        options={"emulate_precision_casts": True},
    )
    actual = compiled(residual, update, gate)

    assert torch.equal(actual, expected)
