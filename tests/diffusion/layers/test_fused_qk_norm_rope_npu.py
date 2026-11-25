# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

_HEAD_DIM = 128
_ROTARY_DIM = 96
_EPS = 1e-5
_HEAD_COUNTS = (4, 2, 2)


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


def _packed_inputs(device: str = "cpu"):
    q, k, q_weight, k_weight, rope_table = (tensor.to(device) for tensor in _inputs())
    k = k[:, : _HEAD_COUNTS[1]]
    v = -k
    qkv = torch.cat((q, k, v), dim=1).flatten(1)
    return qkv, q_weight, k_weight, rope_table


def _run_packed(qkv, q_weight, k_weight, rope_table, eps=_EPS, **options):
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_qkv_split_norm_rope

    return fused_qkv_split_norm_rope(
        qkv,
        q_weight,
        k_weight,
        rope_table,
        eps,
        num_heads=_HEAD_COUNTS[0],
        num_kv_heads=_HEAD_COUNTS[1],
        head_dim=q_weight.shape[0],
        **options,
    )


def _track_packed_splits(monkeypatch, packed):
    splits = []
    tensor_split = torch.Tensor.split

    def track(tensor, split_size, dim=0):
        parts = tensor_split(tensor, split_size, dim)
        if tensor.shape == packed.shape and tensor.data_ptr() == packed.data_ptr() and dim in (1, -2):
            assert tuple(split_size) == _HEAD_COUNTS
            splits.append(parts)
        return parts

    monkeypatch.setattr(torch.Tensor, "split", track)
    return splits


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
def test_npu_qk_norm_rope_ignores_installed_mindiesd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the #4042 contract: MindIE-SD stays unused even when importable.

    The NPU path always composes torch_npu's own fused primitives; a
    mindiesd install must never route ``_npu_apply_rope_table`` back to
    ``mindiesd.rotary_position_embedding``.
    """
    from vllm_omni.diffusion.layers import fused_qk_norm_rope as fused

    calls: dict[str, int] = {"rms_norm": 0, "rotary_mul": 0, "mindiesd_rope": 0}

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

    def rotary_position_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, **kwargs):
        calls["mindiesd_rope"] += 1
        return x

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        types.SimpleNamespace(npu_rms_norm=npu_rms_norm, npu_rotary_mul=npu_rotary_mul),
    )
    monkeypatch.setitem(
        sys.modules,
        "mindiesd",
        types.SimpleNamespace(rotary_position_embedding=rotary_position_embedding),
    )
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

    assert calls == {"rms_norm": 2, "rotary_mul": 2, "mindiesd_rope": 0}
    torch.testing.assert_close(actual_q, expected_q, atol=0, rtol=0)
    torch.testing.assert_close(actual_k, expected_k, atol=0, rtol=0)


@pytest.mark.cpu
@pytest.mark.parametrize("options", [{}, {"return_raw_qk": False}], ids=["default-raw", "no-raw"])
def test_npu_eagle_receives_original_packed_qkv(monkeypatch: pytest.MonkeyPatch, options: dict[str, bool]) -> None:
    from vllm_omni.diffusion.layers import fused_qk_norm_rope as fused

    qkv, expected_q_weight, expected_k_weight, rope_table = _packed_inputs()
    packed = qkv.unflatten(-1, (sum(_HEAD_COUNTS), _HEAD_DIM))
    q, k, v = packed.split(_HEAD_COUNTS, dim=1)
    outputs = (q.neg(), k.neg(), v.clone())
    eps = 1e-6
    splits = _track_packed_splits(monkeypatch, packed)

    def eagle(
        qkv, q_weight, k_weight, rope_cos, rope_sin, split_sizes, norm_type, rope_mode, rotary_dim, epsilon, output_bnsd
    ):
        assert not splits
        assert qkv.data_ptr() == packed.data_ptr()
        torch.testing.assert_close(qkv, packed, check_stride=True)
        assert q_weight is expected_q_weight and k_weight is expected_k_weight
        cos, sin = rope_table.chunk(2, dim=-1)
        torch.testing.assert_close(rope_cos, cos)
        torch.testing.assert_close(rope_sin, sin)
        assert rope_cos.is_contiguous() and rope_sin.is_contiguous()
        assert tuple(split_sizes) == _HEAD_COUNTS
        assert (norm_type, rope_mode, rotary_dim, epsilon, output_bnsd) == (3, 1, _ROTARY_DIM, eps, False)
        return outputs

    monkeypatch.setattr(fused.current_omni_platform, "is_npu", lambda: True)
    monkeypatch.setitem(sys.modules, "mindiesd.fused_ops", types.SimpleNamespace(eagle_split_norm_rope_concat=eagle))

    q_out, k_out, v_out, q_raw, k_raw = _run_packed(
        qkv, expected_q_weight, expected_k_weight, rope_table, eps, **options
    )

    assert q_out is outputs[0] and k_out is outputs[1] and v_out is outputs[2]
    if options.get("return_raw_qk", True):
        (parts,) = splits
        assert q_raw is parts[0] and k_raw is parts[1]
        torch.testing.assert_close(q_raw, q, check_stride=True)
        torch.testing.assert_close(k_raw, k, check_stride=True)
    else:
        assert not splits


@pytest.mark.cpu
@pytest.mark.parametrize("is_npu", [True, False], ids=["npu-without-eagle", "non-npu-with-eagle"])
@pytest.mark.parametrize("return_raw_qk", [True, False])
def test_packed_qkv_fallback(monkeypatch: pytest.MonkeyPatch, is_npu: bool, return_raw_qk: bool) -> None:
    from vllm_omni.diffusion.layers import fused_qk_norm_rope as fused

    qkv, q_weight, k_weight, rope_table = _packed_inputs()
    packed = qkv.unflatten(-1, (sum(_HEAD_COUNTS), _HEAD_DIM))
    q, k, v = packed.split(_HEAD_COUNTS, dim=1)
    splits = _track_packed_splits(monkeypatch, packed)
    monkeypatch.setattr(fused.current_omni_platform, "is_npu", lambda: is_npu)
    eagle = Mock(side_effect=AssertionError("unexpected Eagle invocation"))
    module = None if is_npu else types.SimpleNamespace(eagle_split_norm_rope_concat=eagle)
    monkeypatch.setitem(sys.modules, "mindiesd.fused_ops", module)

    q_out, k_out, v_out, q_raw, k_raw = _run_packed(qkv, q_weight, k_weight, rope_table, return_raw_qk=return_raw_qk)

    expected_q, expected_k = _reference(q, k, q_weight, k_weight, rope_table)
    torch.testing.assert_close(q_out, expected_q, atol=0, rtol=0)
    torch.testing.assert_close(k_out, expected_k, atol=0, rtol=0)
    (parts,) = splits
    assert v_out is parts[2]
    torch.testing.assert_close(v_out, v)
    if return_raw_qk:
        assert q_raw is parts[0] and k_raw is parts[1]


@pytest.mark.cpu
@pytest.mark.parametrize(("error_type", "tokens"), [(RuntimeError, 0), (ImportError, 2**23)])
def test_packed_qkv_propagates_eagle_failure(
    monkeypatch: pytest.MonkeyPatch, error_type: type[Exception], tokens: int
) -> None:
    from vllm_omni.diffusion.layers import fused_qk_norm_rope as fused

    eagle = Mock(side_effect=error_type("Eagle execution failed"))
    monkeypatch.setattr(fused.current_omni_platform, "is_npu", lambda: True)
    monkeypatch.setitem(sys.modules, "mindiesd.fused_ops", types.SimpleNamespace(eagle_split_norm_rope_concat=eagle))

    # Eagle owns dtype, head/rotary dimensions and packed index limits (0 or 2**31 elements).
    head_dim, rotary_dim = 32, 16
    width = sum(_HEAD_COUNTS) * head_dim
    qkv = torch.empty((tokens, width), dtype=torch.float32, device="meta")
    weight = qkv.new_empty((head_dim,))
    rope_table = qkv.new_empty((tokens, rotary_dim))
    with pytest.raises(error_type, match="Eagle execution failed"):
        _run_packed(qkv, weight, weight, rope_table)


@hardware_test(res={"npu": "A3"}, num_cards=1)
def test_fused_qk_norm_rope_npu_matches_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the split-Q/K entry point with real torch_npu primitives."""
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


@hardware_test(res={"npu": "A3"}, num_cards=1)
def test_packed_qkv_eagle_npu_matches_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    ops = pytest.importorskip("mindiesd.fused_ops")
    if not hasattr(ops, "eagle_split_norm_rope_concat"):
        pytest.skip("MindIE-SD Eagle is unavailable")
    eagle = Mock(wraps=ops.eagle_split_norm_rope_concat)
    monkeypatch.setattr(ops, "eagle_split_norm_rope_concat", eagle)
    qkv, q_weight, k_weight, rope_table = _packed_inputs("npu")
    q, k, v = qkv.unflatten(-1, (sum(_HEAD_COUNTS), _HEAD_DIM)).split(_HEAD_COUNTS, dim=1)

    q_out, k_out, v_out, q_raw, k_raw = _run_packed(qkv, q_weight, k_weight, rope_table)

    eagle.assert_called_once()
    expected_q, expected_k = _reference(q, k, q_weight, k_weight, rope_table)
    torch.testing.assert_close(q_out, expected_q, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k_out, expected_k, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(v_out, v, atol=0, rtol=0)
    torch.testing.assert_close(q_raw, q, atol=0, rtol=0)
    torch.testing.assert_close(k_raw, k, atol=0, rtol=0)
