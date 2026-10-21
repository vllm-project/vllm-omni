# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
import types

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_marks, hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

_HEAD_DIM = 128
_ROTARY_DIM = 96
_EPS = 1e-5


@pytest.mark.cpu
@pytest.mark.parametrize("api", ["absent", "legacy", "public"])
def test_single_stream_optional_public_api(monkeypatch, api):
    from vllm_omni.diffusion.layers import fused_qk_norm_rope as fused

    module = types.ModuleType("mindiesd")
    if api == "legacy":
        module.norm_rope_concat = lambda q, k, v: None
    elif api == "public":
        module.norm_rope_concat = lambda q, k, v, *, input_layout="BSND", output_layout="BSND": None
    monkeypatch.setitem(sys.modules, "mindiesd", module)
    monkeypatch.setattr(fused.current_omni_platform, "is_npu", lambda: True)
    monkeypatch.setattr(fused, "find_spec", lambda name: object())
    fused._load_norm_rope_concat.cache_clear()
    try:
        op = fused._load_norm_rope_concat()
        assert (op is not None) == (api == "public")
        # Resolution is cached; repeated calls do not inspect or import again.
        monkeypatch.setattr(fused, "find_spec", lambda name: pytest.fail("Repeated dependency lookup"))
        assert fused._load_norm_rope_concat() is op
        assert not fused.norm_rope_concat_available(torch.device("cpu"))
    finally:
        fused._load_norm_rope_concat.cache_clear()


@pytest.mark.cpu
@pytest.mark.parametrize("batch,heads,dim", [(1, 1, 64), (2, 4, 128)])
@pytest.mark.parametrize("interleaved", [True, False])
def test_single_stream_bsnd_contract(monkeypatch, batch, heads, dim, interleaved):
    from vllm_omni.diffusion.layers import fused_qk_norm_rope as fused

    packed = torch.randn(batch, 7, 3 * heads * dim)
    q, k, v = [x.unflatten(-1, (heads, dim)) for x in packed.chunk(3, dim=-1)]
    assert not q.is_contiguous()
    theta = torch.randn(7, dim // 2)
    tables = fused.prepare_norm_rope_concat_tables(theta.cos(), theta.sin(), interleaved=interleaved, dtype=q.dtype)
    weight = torch.ones(dim)
    calls = []

    def op(*args, **kwargs):
        calls.append(kwargs)
        assert all(a is b for a, b in zip(args, (q, k, v)))
        # Current public SD BSND outputs are BNSD-backed views.
        return tuple(t.transpose(1, 2).contiguous().transpose(1, 2) for t in args) + (None,) * 8

    monkeypatch.setattr(fused, "norm_rope_concat_available", lambda device: True)
    monkeypatch.setattr(fused, "_load_norm_rope_concat", lambda: op)
    with torch.inference_mode():
        result = fused.try_fused_qkv_norm_rope(
            q, k, v, weight, weight, tables, _EPS, head_dim=dim, rotary_dim=dim, interleaved=interleaved
        )
    assert len(calls) == 1
    assert calls[0]["input_layout"] == calls[0]["output_layout"] == "BSND"
    assert calls[0]["rope_type"] == (1 if interleaved else 2)
    assert calls[0]["norm_type"] == 4
    for actual, expected in zip(result, (q, k, v)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.transpose(1, 2).is_contiguous()
    expected_cos = theta.cos().repeat_interleave(2, -1) if interleaved else torch.cat([theta.cos()] * 2, -1)
    torch.testing.assert_close(tables[0], expected_cos)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "unsupported", ["gqa", "partial", "weight", "table", "dtype", "empty", "grad", "compile", "missing"]
)
def test_single_stream_fallback_does_not_launch(monkeypatch, unsupported):
    from vllm_omni.diffusion.layers import fused_qk_norm_rope as fused

    q = torch.randn(2, 7, 4, 64)
    k = v = q
    weight = torch.ones(64)
    tables = (torch.ones(7, 64), torch.zeros(7, 64))
    rotary_dim = 64
    if unsupported == "gqa":
        k = v = q[:, :, :2]
    elif unsupported == "partial":
        rotary_dim = 32
    elif unsupported == "weight":
        weight = weight.double()
    elif unsupported == "table":
        tables = tuple(t[:3] for t in tables)
    elif unsupported == "dtype":
        q = q.double()
    elif unsupported == "empty":
        q = k = v = q[:, :0]
    monkeypatch.setattr(fused, "norm_rope_concat_available", lambda device: unsupported != "missing")
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: unsupported == "compile")
    monkeypatch.setattr(fused, "_load_norm_rope_concat", lambda: pytest.fail("Unsupported input loaded SD"))
    with torch.set_grad_enabled(unsupported == "grad"):
        assert (
            fused.try_fused_qkv_norm_rope(
                q, k, v, weight, weight, tables, _EPS, head_dim=64, rotary_dim=rotary_dim, interleaved=True
            )
            is None
        )


@pytest.mark.cpu
def test_single_stream_execution_error_propagates(monkeypatch):
    from vllm_omni.diffusion.layers import fused_qk_norm_rope as fused

    def fail(*args, **kwargs):
        raise RuntimeError("kernel failure")

    monkeypatch.setattr(fused, "norm_rope_concat_available", lambda device: True)
    monkeypatch.setattr(fused, "_load_norm_rope_concat", lambda: fail)
    q = torch.ones(1, 3, 2, 64)
    weight = torch.ones(64)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="kernel failure"):
        fused.try_fused_qkv_norm_rope(
            q,
            q,
            q,
            weight,
            weight,
            (torch.ones(3, 64), torch.zeros(3, 64)),
            _EPS,
            head_dim=64,
            rotary_dim=64,
            interleaved=True,
        )


@pytest.mark.parametrize(
    "npu_family",
    [
        pytest.param("A2", marks=hardware_marks(res={"npu": "A2"}, num_cards=1)),
        pytest.param("A3", marks=hardware_marks(res={"npu": "A3"}, num_cards=1)),
        pytest.param("A5", marks=hardware_marks(res={"npu": "A5"}, num_cards=1)),
    ],
)
@pytest.mark.parametrize("batch,heads,dim", [(1, 2, 64), (2, 4, 128)])
@pytest.mark.parametrize("interleaved", [True, False])
def test_single_stream_npu_against_fp32_reference(npu_family, batch, heads, dim, interleaved):
    pytest.importorskip("torch_npu")
    from vllm_omni.diffusion.layers import fused_qk_norm_rope as fused

    if not fused.norm_rope_concat_available(torch.device("npu")):
        pytest.skip(f"{npu_family} requires the public SD layout API")
    with torch.inference_mode():
        torch.manual_seed(17)
        packed = torch.randn(batch, 17, 3 * heads * dim, device="npu", dtype=torch.bfloat16)
        q, k, v = [t.unflatten(-1, (heads, dim)) for t in packed.chunk(3, -1)]
        weight = torch.randn(dim, device="npu", dtype=q.dtype)
        theta = torch.randn(17, dim // 2, device="npu")
        cos, sin = fused.prepare_norm_rope_concat_tables(
            theta.cos(), theta.sin(), interleaved=interleaved, dtype=q.dtype
        )
        result = fused.try_fused_qkv_norm_rope(
            q, k, v, weight, weight, (cos, sin), _EPS, head_dim=dim, rotary_dim=dim, interleaved=interleaved
        )
        assert result is not None, "Supported QKV must execute the fused backend"
        for actual, source in zip(result[:2], (q, k)):
            normalized = source.float() * torch.rsqrt(source.float().square().mean(-1, keepdim=True) + _EPS)
            normalized = normalized * weight.float()
            if interleaved:
                pairs = normalized.unflatten(-1, (-1, 2))
                rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
            else:
                first, second = normalized.chunk(2, -1)
                rotated = torch.cat((-second, first), -1)
            expected = (normalized * cos.float()[None, :, None] + rotated * sin.float()[None, :, None]).to(q.dtype)
            torch.testing.assert_close(actual, expected, atol=0.0625, rtol=0.02)
        torch.testing.assert_close(result[2], v, atol=0, rtol=0)
        compiled = torch.compile(fused.try_fused_qkv_norm_rope, backend="eager", fullgraph=True)(
            q,
            k,
            v,
            weight,
            weight,
            (cos, sin),
            _EPS,
            head_dim=dim,
            rotary_dim=dim,
            interleaved=interleaved,
        )
        assert compiled is not None
        for actual, expected in zip(compiled, result):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


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
