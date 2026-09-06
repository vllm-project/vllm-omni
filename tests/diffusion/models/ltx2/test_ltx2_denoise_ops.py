# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.models.ltx2.ops.denoise import (
    try_attention_gate_exact,
    try_masked_residual_gate_add_exact,
    try_perturbation_blend_attention_gate_exact,
    try_qknorm_split_rope_exact,
    try_residual_gate_add_exact,
    try_rms_norm_dual_modulate_exact,
    try_rms_norm_modulate_exact,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cuda, pytest.mark.diffusion]


def _sm90_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _split_rope(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    output_dtype = hidden_states.dtype
    batch = hidden_states.shape[0]
    _, heads, tokens, _ = cos.shape
    hidden_states = hidden_states.reshape(batch, tokens, heads, -1).swapaxes(1, 2)
    head_dim = hidden_states.shape[-1]
    half_dim = head_dim // 2
    split_states = hidden_states.reshape(*hidden_states.shape[:-1], 2, half_dim)
    first = split_states[..., :1, :]
    second = split_states[..., 1:, :]
    output = split_states * cos.unsqueeze(-2)
    output[..., :1, :].addcmul_(-sin.unsqueeze(-2), second)
    output[..., 1:, :].addcmul_(sin.unsqueeze(-2), first)
    output = output.reshape(*output.shape[:-2], head_dim)
    return output.swapaxes(1, 2).reshape(batch, tokens, -1).to(output_dtype)


def _qknorm_split_rope_reference(
    states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    normalized = F.rms_norm(
        states.float(),
        (states.shape[-1],),
        None if weight is None else weight.float(),
        eps,
    ).to(states.dtype)
    return _split_rope(normalized, cos, sin)


@pytest.mark.skipif(not _sm90_available(), reason="CUDA SM90 required")
@pytest.mark.parametrize(
    ("hidden", "heads", "batch", "tokens", "rope_dtype", "weighted"),
    [
        (4096, 32, 1, 64, torch.float32, False),
        (4096, 32, 1, 64, torch.float32, True),
        (4096, 32, 1, 64, torch.bfloat16, True),
        (2048, 16, 2, 17, torch.bfloat16, True),
    ],
)
def test_qknorm_split_rope_is_bit_exact(
    hidden: int,
    heads: int,
    batch: int,
    tokens: int,
    rope_dtype: torch.dtype,
    weighted: bool,
) -> None:
    torch.manual_seed(20260825)
    head_dim = hidden // heads
    packed = torch.randn(
        batch,
        tokens,
        hidden * 3,
        device="cuda",
        dtype=torch.bfloat16,
    )
    query, key, _ = packed.split(hidden, dim=-1)
    rope_shape = (batch, tokens, heads, head_dim // 2)
    query_cos = torch.randn(rope_shape, device="cuda", dtype=rope_dtype).transpose(1, 2)
    query_sin = torch.randn_like(query_cos)
    key_cos = torch.randn(rope_shape, device="cuda", dtype=rope_dtype).transpose(1, 2)
    key_sin = torch.randn_like(key_cos)
    query_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16) if weighted else None
    key_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16) if weighted else None

    expected_query = _qknorm_split_rope_reference(query, query_cos, query_sin, query_weight, 1e-6)
    expected_key = _qknorm_split_rope_reference(key, key_cos, key_sin, key_weight, 1e-6)
    with torch.inference_mode():
        actual = try_qknorm_split_rope_exact(
            query,
            query_cos,
            query_sin,
            query_weight,
            key,
            key_cos,
            key_sin,
            key_weight,
            1e-6,
            heads,
            head_dim,
        )

    assert actual is not None
    actual_query, actual_key = actual
    assert torch.equal(actual_query, expected_query)
    assert torch.equal(actual_key, expected_key)


@pytest.mark.skipif(not _sm90_available(), reason="CUDA SM90 required")
@pytest.mark.parametrize(
    ("batch", "tokens", "gate_tokens", "with_table"),
    [(1, 256, 1, False), (2, 128, 128, True)],
)
def test_residual_gate_ops_are_bit_exact(
    batch: int,
    tokens: int,
    gate_tokens: int,
    with_table: bool,
) -> None:
    torch.manual_seed(20260826)
    shape = (batch, tokens, 4096)
    residual = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    update = torch.randn_like(residual)
    gate = torch.randn(batch, gate_tokens, shape[-1], device="cuda", dtype=torch.bfloat16)
    gate_table = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16) if with_table else None
    mask = torch.randn(batch, 1, 1, device="cuda", dtype=torch.bfloat16)
    materialized_gate = gate if gate_table is None else gate_table + gate
    expected = residual + update * materialized_gate
    expected_masked = residual + (update * mask) * materialized_gate

    with torch.inference_mode():
        actual = try_residual_gate_add_exact(residual, update, gate, gate_table)
        actual_masked = try_masked_residual_gate_add_exact(residual, update, gate, mask, gate_table)

    assert actual is not None
    assert actual_masked is not None
    assert torch.equal(actual, expected)
    assert torch.equal(actual_masked, expected_masked)


@pytest.mark.skipif(not _sm90_available(), reason="CUDA SM90 required")
@pytest.mark.parametrize(("batch", "tokens"), [(1, 256), (2, 128)])
def test_perturbation_blend_attention_gate_is_bit_exact(batch: int, tokens: int) -> None:
    torch.manual_seed(20260827)
    heads, head_dim = 32, 128
    shape = (batch, tokens, heads * head_dim)
    primary = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    fallback = torch.randn_like(primary)
    mask = torch.randn(batch, 1, 1, device="cuda", dtype=torch.bfloat16)
    gate_logits = torch.randn(batch, tokens, heads, device="cuda", dtype=torch.bfloat16)
    expected = primary * mask + fallback * (1 - mask)
    gates = 2.0 * torch.sigmoid(gate_logits)
    expected_composite = (expected.unflatten(2, (heads, head_dim)) * gates.unsqueeze(-1)).flatten(2, 3)

    with torch.inference_mode():
        actual_composite = try_perturbation_blend_attention_gate_exact(
            primary,
            fallback,
            mask,
            gate_logits,
            head_dim,
        )

    assert actual_composite is not None
    assert torch.equal(actual_composite, expected_composite)


@pytest.mark.skipif(not _sm90_available(), reason="CUDA SM90 required")
@pytest.mark.parametrize(
    ("hidden", "batch", "tokens", "modulation_tokens", "with_tables"),
    [
        (2048, 1, 64, 1, True),
        (2048, 2, 17, 17, False),
        (4096, 1, 64, 1, True),
        (4096, 2, 9, 9, False),
    ],
)
def test_rms_backends_are_bit_exact(
    hidden: int,
    batch: int,
    tokens: int,
    modulation_tokens: int,
    with_tables: bool,
) -> None:
    torch.manual_seed(20260828)
    states = torch.randn(batch, tokens, hidden, device="cuda", dtype=torch.bfloat16)
    scale_a = torch.randn(batch, modulation_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    shift_a = torch.randn_like(scale_a)
    scale_b = torch.randn_like(scale_a)
    shift_b = torch.randn_like(scale_a)
    scale_a_table = torch.randn(hidden, device="cuda", dtype=torch.bfloat16) if with_tables else None
    shift_a_table = torch.randn(hidden, device="cuda", dtype=torch.bfloat16) if with_tables else None
    scale_b_table = torch.randn(hidden, device="cuda", dtype=torch.bfloat16) if with_tables else None
    shift_b_table = torch.randn(hidden, device="cuda", dtype=torch.bfloat16) if with_tables else None

    def reference(
        scale: torch.Tensor,
        shift: torch.Tensor,
        scale_table: torch.Tensor | None,
        shift_table: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized = F.rms_norm(states, (hidden,), eps=1e-6)
        if scale_table is None or shift_table is None:
            return normalized * (1 + scale) + shift
        return normalized * (1 + (scale_table + scale)) + (shift_table + shift)

    with torch.inference_mode():
        actual_a = try_rms_norm_modulate_exact(
            states,
            scale_a,
            shift_a,
            1e-6,
            scale_a_table,
            shift_a_table,
        )
        actual_dual = try_rms_norm_dual_modulate_exact(
            states,
            scale_a,
            shift_a,
            scale_b,
            shift_b,
            1e-6,
            scale_a_table,
            shift_a_table,
            scale_b_table,
            shift_b_table,
        )
    expected_a = reference(scale_a, shift_a, scale_a_table, shift_a_table)
    expected_b = reference(scale_b, shift_b, scale_b_table, shift_b_table)
    assert actual_a is not None
    assert actual_dual is not None
    assert torch.equal(actual_a, expected_a)
    assert torch.equal(actual_dual[0], expected_a)
    assert torch.equal(actual_dual[1], expected_b)


@pytest.mark.skipif(not _sm90_available(), reason="CUDA SM90 required")
@pytest.mark.parametrize(("batch", "tokens", "heads", "head_dim"), [(1, 256, 64, 64), (2, 129, 32, 128)])
def test_attention_gate_is_bit_exact(batch: int, tokens: int, heads: int, head_dim: int) -> None:
    torch.manual_seed(20260829)
    states = torch.randn(
        batch,
        tokens,
        heads * head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    gate_logits = torch.randn(batch, tokens, heads, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        attention_gate = try_attention_gate_exact(states, gate_logits, head_dim)

    expected_attention_gate = (
        states.unflatten(2, (heads, head_dim)) * (2 * torch.sigmoid(gate_logits)).unsqueeze(-1)
    ).flatten(2, 3)
    assert attention_gate is not None
    assert torch.equal(attention_gate, expected_attention_gate)


def test_exact_ops_reject_unsupported_inputs() -> None:
    states = torch.randn(1, 2, 8)
    rope = torch.randn(1, 2, 2, 2)
    residual = torch.randn(1, 2, 8)
    gate = torch.randn(1, 1, 8)
    mask = torch.randn(1, 1, 1)
    gate_logits = torch.randn(1, 2, 2)

    with torch.inference_mode():
        assert (
            try_qknorm_split_rope_exact(
                states,
                rope,
                rope,
                None,
                states,
                rope,
                rope,
                None,
                1e-6,
                2,
                4,
            )
            is None
        )
        assert try_residual_gate_add_exact(residual, residual, gate) is None
        assert try_masked_residual_gate_add_exact(residual, residual, gate, mask) is None
        assert try_rms_norm_modulate_exact(states, gate, gate, 1e-6) is None
        assert (
            try_perturbation_blend_attention_gate_exact(
                residual,
                residual,
                mask,
                gate_logits,
                4,
            )
            is None
        )


@pytest.mark.skipif(not _sm90_available(), reason="CUDA SM90 required")
def test_exact_ops_fall_back_while_compiling(monkeypatch: pytest.MonkeyPatch) -> None:
    residual = torch.randn(1, 256, 4096, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(1, 1, 4096, device="cuda", dtype=torch.bfloat16)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    with torch.inference_mode():
        assert try_residual_gate_add_exact(residual, residual, gate) is None


@pytest.mark.skipif(not _sm90_available(), reason="CUDA SM90 required")
def test_exact_ops_fall_back_on_unvalidated_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.platforms.interface import DeviceCapability

    from vllm_omni.diffusion.models.ltx2.ops import platform as ltx2_platform

    residual = torch.randn(1, 256, 4096, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(1, 1, 4096, device="cuda", dtype=torch.bfloat16)
    ltx2_platform._is_verified_cuda_device.cache_clear()
    monkeypatch.setattr(
        ltx2_platform.current_omni_platform,
        "get_device_capability",
        lambda device_id=0: DeviceCapability(major=10, minor=0),
    )

    try:
        with torch.inference_mode():
            assert try_residual_gate_add_exact(residual, residual, gate) is None
    finally:
        ltx2_platform._is_verified_cuda_device.cache_clear()


@pytest.mark.skipif(not _sm90_available(), reason="CUDA SM90 required")
def test_first_triton_failure_disables_device(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion.models.ltx2.ops.denoise import residual_gate_add as residual_ops

    calls = 0

    def failing_launch(*_args, **_kwargs) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("synthetic launch failure")

    residual = torch.randn(1, 256, 4096, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(1, 1, 4096, device="cuda", dtype=torch.bfloat16)
    residual_ops._FAILED_DEVICES.clear()
    monkeypatch.setattr(residual_ops, "_run_residual_gate_add", failing_launch)

    try:
        with torch.inference_mode():
            assert residual_ops.try_residual_gate_add_exact(residual, residual, gate) is None
            assert residual_ops.try_residual_gate_add_exact(residual, residual, gate) is None
        assert calls == 1
        assert residual.device.index in residual_ops._FAILED_DEVICES
    finally:
        residual_ops._FAILED_DEVICES.clear()
