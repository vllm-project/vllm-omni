# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

_HEAD_DIM = 128
_ROTARY_DIM = 96
_EPS = 1e-5


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    half = _ROTARY_DIM // 2

    def apply(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x = F.rms_norm(x, (_HEAD_DIM,), weight, _EPS)
        cos = rope_table[..., :half].unsqueeze(1)
        sin = rope_table[..., half:].unsqueeze(1)
        return torch.cat(
            (
                x[..., :half] * cos - x[..., half:_ROTARY_DIM] * sin,
                x[..., half:_ROTARY_DIM] * cos + x[..., :half] * sin,
                x[..., _ROTARY_DIM:],
            ),
            dim=-1,
        )

    return apply(q, q_weight), apply(k, k_weight)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    tokens, heads = 11, 4
    q = torch.randn(tokens, heads, _HEAD_DIM, dtype=torch.bfloat16)
    k = torch.randn(tokens, heads, _HEAD_DIM, dtype=torch.bfloat16)
    q_weight = torch.randn(_HEAD_DIM, dtype=torch.bfloat16)
    k_weight = torch.randn(_HEAD_DIM, dtype=torch.bfloat16)
    freqs = torch.randn(tokens, _ROTARY_DIM // 2)
    rope_table = torch.cat((torch.cos(freqs), torch.sin(freqs)), dim=-1).to(torch.bfloat16)
    return q, k, q_weight, k_weight, rope_table


@pytest.mark.cpu
def test_npu_qk_norm_rope_uses_torch_npu_fused_primitives_without_mindiesd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_omni.diffusion.layers import fused_qk_norm_rope as fused

    calls: dict[str, int] = {"rms_norm": 0, "rotary_mul": 0}

    def npu_rms_norm(x: torch.Tensor, gamma: torch.Tensor, epsilon: float):
        calls["rms_norm"] += 1
        return (F.rms_norm(x, (_HEAD_DIM,), gamma, epsilon),)

    def npu_rotary_mul(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        calls["rotary_mul"] += 1
        assert x.shape[-1] == _ROTARY_DIM
        assert cos.shape == (1, x.shape[1], 1, _ROTARY_DIM)
        assert sin.shape == cos.shape
        half = _ROTARY_DIM // 2
        return torch.cat(
            (
                x[..., :half] * cos[..., :half] - x[..., half:] * sin[..., :half],
                x[..., half:] * cos[..., half:] + x[..., :half] * sin[..., half:],
            ),
            dim=-1,
        )

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        types.SimpleNamespace(npu_rms_norm=npu_rms_norm, npu_rotary_mul=npu_rotary_mul),
    )
    monkeypatch.setattr(fused, "find_spec", lambda _: None)
    q, k, q_weight, k_weight, rope_table = _inputs()

    actual_q, actual_k = fused._npu_qk_norm_rope(
        q,
        k,
        q_weight,
        k_weight,
        rope_table,
        _EPS,
        _ROTARY_DIM,
    )
    expected_q, expected_k = _reference(q, k, q_weight, k_weight, rope_table)

    assert calls == {"rms_norm": 2, "rotary_mul": 2}
    torch.testing.assert_close(actual_q, expected_q, atol=0, rtol=0)
    torch.testing.assert_close(actual_k, expected_k, atol=0, rtol=0)


@pytest.mark.cpu
def test_npu_qk_norm_rope_uses_mindiesd_and_normalizes_packed_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_omni.diffusion.layers import fused_qk_norm_rope as fused

    calls: dict[str, int] = {"rms_norm": 0, "mindiesd_rope": 0}

    def npu_rms_norm(x: torch.Tensor, gamma: torch.Tensor, epsilon: float):
        calls["rms_norm"] += 1
        return (F.rms_norm(x, (_HEAD_DIM,), gamma, epsilon),)

    def rotary_position_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, **kwargs):
        calls["mindiesd_rope"] += 1
        assert x.dim() == 4
        assert x.shape[0] == 1
        assert x.shape[-1] == _ROTARY_DIM
        assert cos.shape == (1, x.shape[1], 1, _ROTARY_DIM)
        assert sin.shape == cos.shape
        assert kwargs == {
            "rotated_mode": "rotated_half",
            "head_first": False,
            "fused": True,
        }
        half = _ROTARY_DIM // 2
        return torch.cat(
            (
                x[..., :half] * cos[..., :half] - x[..., half:] * sin[..., :half],
                x[..., half:] * cos[..., half:] + x[..., :half] * sin[..., half:],
            ),
            dim=-1,
        )

    monkeypatch.setitem(sys.modules, "torch_npu", types.SimpleNamespace(npu_rms_norm=npu_rms_norm))
    monkeypatch.setitem(
        sys.modules,
        "mindiesd",
        types.SimpleNamespace(rotary_position_embedding=rotary_position_embedding),
    )
    monkeypatch.setattr(fused, "find_spec", lambda _: object())
    q, k, q_weight, k_weight, rope_table = _inputs()

    actual_q, actual_k = fused._npu_qk_norm_rope(
        q,
        k,
        q_weight,
        k_weight,
        rope_table,
        _EPS,
        _ROTARY_DIM,
    )
    expected_q, expected_k = _reference(q, k, q_weight, k_weight, rope_table)

    assert calls == {"rms_norm": 2, "mindiesd_rope": 2}
    torch.testing.assert_close(actual_q, expected_q, atol=0, rtol=0)
    torch.testing.assert_close(actual_k, expected_k, atol=0, rtol=0)


@hardware_test(res={"npu": "A3"}, num_cards=1)
def test_fused_qk_norm_rope_npu_matches_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the public Ascend dispatch with real torch_npu/MindIE operators."""
    from vllm_omni.diffusion.layers import fused_qk_norm_rope as fused

    q, k, q_weight, k_weight, rope_table = (tensor.to("npu") for tensor in _inputs())
    calls = 0
    original_npu_impl = fused._npu_qk_norm_rope

    def capture_npu_impl(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_npu_impl(*args, **kwargs)

    monkeypatch.setattr(fused, "_npu_qk_norm_rope", capture_npu_impl)
    actual_q, actual_k = fused.fused_qk_norm_rope(
        q,
        k,
        q_weight,
        k_weight,
        rope_table,
        _EPS,
        head_dim=_HEAD_DIM,
        rotary_dim=_ROTARY_DIM,
    )
    expected_q, expected_k = _reference(q, k, q_weight, k_weight, rope_table)

    assert calls == 1
    torch.testing.assert_close(actual_q, expected_q, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual_k, expected_k, atol=2e-2, rtol=2e-2)
