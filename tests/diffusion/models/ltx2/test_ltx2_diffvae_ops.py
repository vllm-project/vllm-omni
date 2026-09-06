# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.ltx2.ops.diffvae import qk_rms_norm as qk_rms_norm_ops
from vllm_omni.diffusion.models.ltx2.ops.diffvae import residual_adaln as residual_adaln_ops
from vllm_omni.diffusion.models.ltx2.ops.diffvae import swiglu as swiglu_ops
from vllm_omni.diffusion.models.ltx2.ops.diffvae import (
    try_qk_rms_norm_scale_rope_3d_exact,
    try_residual_add3_exact,
    try_residual_rms_norm_modulate_exact,
    try_swiglu_tiled_exact,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cuda, pytest.mark.diffusion]

_DIM_SPLIT = (16, 24, 24)


def _sm90_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


@pytest.fixture
def isolated_qk_rms_norm_runtime_state():
    failed_rope_keys = set(qk_rms_norm_ops._FAILED_ROPE_KEYS)
    verified_rope_keys = set(qk_rms_norm_ops._VERIFIED_ROPE_KEYS)
    qk_rms_norm_ops._FAILED_ROPE_KEYS.clear()
    qk_rms_norm_ops._VERIFIED_ROPE_KEYS.clear()
    try:
        yield
    finally:
        qk_rms_norm_ops._FAILED_ROPE_KEYS.clear()
        qk_rms_norm_ops._FAILED_ROPE_KEYS.update(failed_rope_keys)
        qk_rms_norm_ops._VERIFIED_ROPE_KEYS.clear()
        qk_rms_norm_ops._VERIFIED_ROPE_KEYS.update(verified_rope_keys)


@pytest.fixture
def isolated_swiglu_runtime_state():
    failed_keys = set(swiglu_ops._FAILED_KEYS)
    verified_keys = set(swiglu_ops._VERIFIED_KEYS)
    swiglu_ops._FAILED_KEYS.clear()
    swiglu_ops._VERIFIED_KEYS.clear()
    try:
        yield
    finally:
        swiglu_ops._FAILED_KEYS.clear()
        swiglu_ops._FAILED_KEYS.update(failed_keys)
        swiglu_ops._VERIFIED_KEYS.clear()
        swiglu_ops._VERIFIED_KEYS.update(verified_keys)


@pytest.mark.skipif(not _sm90_available(), reason="CUDA SM90 required")
@pytest.mark.parametrize("rows", [257, 16387])
def test_diffvae_swiglu_tiled_is_bit_exact(rows: int, isolated_swiglu_runtime_state) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260829)
    hidden_states = torch.randn((1, 1, 1, rows, 256), device="cuda", dtype=torch.bfloat16, generator=generator)
    gate_weight = torch.randn((1024, 256), device="cuda", dtype=torch.bfloat16, generator=generator)
    up_weight = torch.randn_like(gate_weight)
    down_weight = torch.randn((256, 1024), device="cuda", dtype=torch.bfloat16, generator=generator)
    expected = torch.nn.functional.linear(
        torch.nn.functional.silu(torch.nn.functional.linear(hidden_states, gate_weight))
        * torch.nn.functional.linear(hidden_states, up_weight),
        down_weight,
    )

    with torch.inference_mode():
        actual = try_swiglu_tiled_exact(hidden_states, gate_weight, up_weight, down_weight)

    assert actual is not None
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not _sm90_available(), reason="CUDA SM90 required")
def test_diffvae_swiglu_tiled_falls_back_while_compiling(monkeypatch: pytest.MonkeyPatch) -> None:
    hidden_states = torch.randn(1, 1, 1, 17, 256, device="cuda", dtype=torch.bfloat16)
    gate_weight = torch.randn(1024, 256, device="cuda", dtype=torch.bfloat16)
    up_weight = torch.randn_like(gate_weight)
    down_weight = torch.randn(256, 1024, device="cuda", dtype=torch.bfloat16)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    with torch.inference_mode():
        assert try_swiglu_tiled_exact(hidden_states, gate_weight, up_weight, down_weight) is None


@pytest.mark.skipif(not _sm90_available(), reason="CUDA SM90 required")
def test_diffvae_swiglu_tiled_permanently_falls_back_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    isolated_swiglu_runtime_state,
) -> None:
    hidden_states = torch.randn(1, 1, 1, 17, 256, device="cuda", dtype=torch.bfloat16)
    gate_weight = torch.randn(1024, 256, device="cuda", dtype=torch.bfloat16)
    up_weight = torch.randn_like(gate_weight)
    down_weight = torch.randn(256, 1024, device="cuda", dtype=torch.bfloat16)
    calls = 0

    def fail_launch(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("injected launch failure")

    monkeypatch.setattr(swiglu_ops, "_launch", fail_launch)
    with torch.inference_mode():
        assert try_swiglu_tiled_exact(hidden_states, gate_weight, up_weight, down_weight) is None
        assert try_swiglu_tiled_exact(hidden_states, gate_weight, up_weight, down_weight) is None

    assert calls == 1


@pytest.fixture
def isolated_residual_adaln_runtime_state():
    state = (
        set(residual_adaln_ops._FAILED_ADALN_KEYS),
        set(residual_adaln_ops._VERIFIED_ADALN_KEYS),
        set(residual_adaln_ops._FAILED_ADD_DEVICES),
        set(residual_adaln_ops._VERIFIED_ADD_DEVICES),
    )
    residual_adaln_ops._FAILED_ADALN_KEYS.clear()
    residual_adaln_ops._VERIFIED_ADALN_KEYS.clear()
    residual_adaln_ops._FAILED_ADD_DEVICES.clear()
    residual_adaln_ops._VERIFIED_ADD_DEVICES.clear()
    try:
        yield
    finally:
        residual_adaln_ops._FAILED_ADALN_KEYS.clear()
        residual_adaln_ops._FAILED_ADALN_KEYS.update(state[0])
        residual_adaln_ops._VERIFIED_ADALN_KEYS.clear()
        residual_adaln_ops._VERIFIED_ADALN_KEYS.update(state[1])
        residual_adaln_ops._FAILED_ADD_DEVICES.clear()
        residual_adaln_ops._FAILED_ADD_DEVICES.update(state[2])
        residual_adaln_ops._VERIFIED_ADD_DEVICES.clear()
        residual_adaln_ops._VERIFIED_ADD_DEVICES.update(state[3])


@pytest.mark.skipif(not _sm90_available(), reason="CUDA SM90 required")
@pytest.mark.parametrize("residual_count", [1, 2])
def test_diffvae_residual_rms_norm_modulate_is_bit_exact(
    residual_count: int,
    isolated_residual_adaln_runtime_state,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260829)
    shape = (2, 3, 7, 7, 256)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    residual_a = torch.randn_like(x)
    residual_b = torch.randn_like(x) if residual_count == 2 else None
    norm_weight = torch.randn((256,), device="cuda", dtype=torch.bfloat16, generator=generator)
    scale = torch.randn((2, 1, 1, 1, 256), device="cuda", dtype=torch.bfloat16, generator=generator)
    shift = torch.randn_like(scale)
    hidden_states = x + residual_a
    if residual_b is not None:
        hidden_states = hidden_states + residual_b
    expected = torch.nn.functional.rms_norm(hidden_states, (256,), norm_weight, eps=1e-6) * (1 + scale) + shift

    with torch.inference_mode():
        actual = try_residual_rms_norm_modulate_exact(
            x,
            residual_a,
            residual_b,
            norm_weight,
            scale,
            shift,
            1e-6,
        )

    assert actual is not None
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not _sm90_available(), reason="CUDA SM90 required")
def test_diffvae_residual_add3_is_bit_exact(isolated_residual_adaln_runtime_state) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260829)
    tensors = [
        torch.randn((2, 3, 7, 7, 256), device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(4)
    ]
    expected = tensors[0] + tensors[1]
    expected = expected + tensors[2]
    expected = expected + tensors[3]

    with torch.inference_mode():
        actual = try_residual_add3_exact(*tensors)

    assert actual is not None
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not _sm90_available(), reason="CUDA SM90 required")
def test_diffvae_residual_fusions_fall_back_while_compiling(monkeypatch: pytest.MonkeyPatch) -> None:
    shape = (1, 3, 7, 7, 256)
    tensors = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    weight = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(1, 1, 1, 1, 256, device="cuda", dtype=torch.bfloat16)
    shift = torch.randn_like(scale)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    with torch.inference_mode():
        assert (
            try_residual_rms_norm_modulate_exact(tensors[0], tensors[1], tensors[2], weight, scale, shift, 1e-6) is None
        )
        assert try_residual_add3_exact(*tensors) is None


@pytest.mark.skipif(not _sm90_available(), reason="CUDA SM90 required")
def test_diffvae_residual_adaln_permanently_falls_back_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    isolated_residual_adaln_runtime_state,
) -> None:
    shape = (1, 3, 7, 7, 256)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(1, 1, 1, 1, 256, device="cuda", dtype=torch.bfloat16)
    shift = torch.randn_like(scale)
    calls = 0

    def fail_launch(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("injected launch failure")

    monkeypatch.setattr(residual_adaln_ops, "_launch_adaln", fail_launch)
    with torch.inference_mode():
        for _ in range(2):
            assert try_residual_rms_norm_modulate_exact(x, residual, None, weight, scale, shift, 1e-6) is None

    assert calls == 1


def _tables(
    frames: int,
    height: int,
    width: int,
    dim_split: tuple[int, int, int] = _DIM_SPLIT,
    base: float = 10000.0,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    tables = []
    for length, dim in zip((frames, height, width), dim_split, strict=True):
        exponents = torch.arange(0, dim, 2, dtype=torch.float64, device="cuda") / dim
        inv_freqs = (1.0 / base**exponents).to(torch.float32)
        positions = torch.arange(length, dtype=torch.float32, device="cuda")
        angles = positions[:, None] * inv_freqs[None, :]
        tables.append((angles.cos(), angles.sin()))
    return tuple(tables)


def _apply_rope_reference(
    hidden_states: torch.Tensor,
    tables: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    dim_split: tuple[int, int, int] = _DIM_SPLIT,
) -> torch.Tensor:
    outputs = []
    offset = 0
    for axis, (dim, (cos, sin)) in enumerate(zip(dim_split, tables, strict=True), 1):
        chunk = hidden_states[..., offset : offset + dim]
        pairs = chunk.reshape(*chunk.shape[:-1], dim // 2, 2)
        even = pairs[..., 0].float()
        odd = pairs[..., 1].float()
        shape = [1, 1, 1, 1, 1, dim // 2]
        shape[axis] = cos.shape[0]
        cos = cos.reshape(shape)
        sin = sin.reshape(shape)
        rotated = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1)
        outputs.append(rotated.reshape(chunk.shape).to(hidden_states.dtype))
        offset += dim
    return torch.cat(outputs, dim=-1)


@pytest.mark.skipif(not _sm90_available(), reason="CUDA SM90 required")
@pytest.mark.parametrize(
    ("frames", "height", "width", "heads"),
    [(3, 7, 7, 1), (31, 136, 192, 4)],
)
def test_diffvae_qk_rms_norm_scale_rope_3d_is_bit_exact(
    frames: int,
    height: int,
    width: int,
    heads: int,
    isolated_qk_rms_norm_runtime_state,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260829)
    shape = (1, frames, height, width, heads, 64)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    key = torch.randn_like(query)
    query_weight = torch.randn((64,), device="cuda", dtype=torch.bfloat16, generator=generator)
    key_weight = torch.randn((64,), device="cuda", dtype=torch.bfloat16, generator=generator)
    tables = _tables(frames, height, width)
    expected = (
        _apply_rope_reference(torch.nn.functional.rms_norm(query, (64,), query_weight, 1e-6) * 0.125, tables),
        _apply_rope_reference(torch.nn.functional.rms_norm(key, (64,), key_weight, 1e-6), tables),
    )

    with torch.inference_mode():
        actual = try_qk_rms_norm_scale_rope_3d_exact(
            query,
            key,
            query_weight,
            key_weight,
            1e-6,
            0.125,
            _DIM_SPLIT,
            10000.0,
        )

    assert actual is not None
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


@pytest.mark.skipif(not _sm90_available(), reason="CUDA SM90 required")
def test_diffvae_qk_rms_norm_scale_rope_3d_falls_back_while_compiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = torch.randn(1, 3, 7, 7, 1, 64, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    weight = torch.ones(64, device="cuda", dtype=torch.bfloat16)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    with torch.inference_mode():
        assert try_qk_rms_norm_scale_rope_3d_exact(query, key, weight, weight, 1e-6, 0.125, _DIM_SPLIT, 10000.0) is None


@pytest.mark.skipif(not _sm90_available(), reason="CUDA SM90 required")
def test_diffvae_qk_rms_norm_scale_rope_3d_permanently_falls_back_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    isolated_qk_rms_norm_runtime_state,
) -> None:
    query = torch.randn(1, 3, 7, 7, 1, 64, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    weight = torch.ones(64, device="cuda", dtype=torch.bfloat16)
    calls = 0

    def fail_launch(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("injected launch failure")

    monkeypatch.setattr(qk_rms_norm_ops, "_launch_combined", fail_launch)
    with torch.inference_mode():
        for _ in range(2):
            assert (
                try_qk_rms_norm_scale_rope_3d_exact(query, key, weight, weight, 1e-6, 0.125, _DIM_SPLIT, 10000.0)
                is None
            )

    assert calls == 1
