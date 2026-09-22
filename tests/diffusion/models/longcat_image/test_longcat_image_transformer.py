# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from diffusers.models.embeddings import apply_rotary_emb

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def _clear_qk_rope_signature_caches():
    from vllm_omni.diffusion.models.longcat_image import longcat_image_transformer as longcat

    longcat._FAILED_QK_ROPE_SIGNATURES.clear()
    yield
    longcat._FAILED_QK_ROPE_SIGNATURES.clear()


def _inputs(sequence: int = 5, head_dim: int = 8):
    q = torch.randn(2, sequence, 3, head_dim, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    cos = torch.randn(sequence, head_dim)
    sin = torch.randn_like(cos)
    return q, k, (cos, sin)


def _reference(q, k, rotary_emb):
    return (
        apply_rotary_emb(q, rotary_emb, sequence_dim=1),
        apply_rotary_emb(k, rotary_emb, sequence_dim=1),
    )


def test_prepare_rotary_emb_resolves_threshold_once_only_when_enabled(monkeypatch):
    from vllm_omni.diffusion.models.longcat_image import longcat_image_transformer as longcat

    calls = 0

    def resolve(default):
        nonlocal calls
        calls += 1
        assert default == 512
        return 37

    monkeypatch.setattr(longcat, "fused_qk_norm_rope_min_tokens", resolve)
    txt_cos, txt_sin = torch.randn(2, 8), torch.randn(2, 8)
    img_cos, img_sin = torch.randn(3, 8), torch.randn(3, 8)
    native = longcat._prepare_rotary_emb(txt_cos, txt_sin, img_cos, img_sin, enable_fusion=False)
    fused = longcat._prepare_rotary_emb(txt_cos, txt_sin, img_cos, img_sin, enable_fusion=True)

    assert calls == 1
    assert len(native) == 2
    assert len(fused) == 3 and fused[2] == 37
    assert torch.equal(native[0], fused[0])
    assert torch.equal(native[1], fused[1])


@pytest.mark.parametrize(
    ("mode", "sp_size"),
    [
        ("unsupported", 1),
        ("compile", 1),
        ("grad", 1),
        ("capture", 1),
        ("sequence_parallel", 2),
        ("below_threshold", 1),
    ],
)
def test_qk_rope_non_eager_modes_use_exact_original_path(monkeypatch, mode, sp_size):
    from vllm_omni.diffusion.models.longcat_image import longcat_image_transformer as longcat

    torch.manual_seed(1)
    q, k, rotary_emb = _inputs()
    expected = _reference(q, k, rotary_emb)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: mode == "compile")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: mode == "capture")
    monkeypatch.setattr(longcat, "fused_qk_rope_supported", lambda *args: mode != "unsupported")
    monkeypatch.setattr(
        longcat,
        "fused_qk_rope",
        lambda *args: (_ for _ in ()).throw(AssertionError("fallback called fused op")),
    )

    context = nullcontext() if mode == "grad" else torch.no_grad()
    min_tokens = q.shape[0] * q.shape[1] + 1 if mode == "below_threshold" else 0
    with context:
        actual = longcat._apply_qk_rope(q, k, rotary_emb, sp_size, min_tokens)

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_qk_rope_eligible_path_skips_runtime_parity_check(monkeypatch):
    from vllm_omni.diffusion.models.longcat_image import longcat_image_transformer as longcat

    torch.manual_seed(2)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(longcat, "fused_qk_rope_supported", lambda *args: True)
    fused_calls = 0

    def fused(query, key, cos, sin):
        nonlocal fused_calls
        fused_calls += 1
        return query + 1, key + 2

    monkeypatch.setattr(
        longcat,
        "_apply_qk_rope_reference",
        lambda *args: (_ for _ in ()).throw(AssertionError("eligible path ran the native parity check")),
    )
    monkeypatch.setattr(longcat, "fused_qk_rope", fused)

    with torch.no_grad():
        q, k, rotary_emb = _inputs()
        actual = longcat._apply_qk_rope(q, k, rotary_emb, 1, 0)

    assert fused_calls == 1
    assert torch.equal(actual[0], q + 1)
    assert torch.equal(actual[1], k + 2)


def test_qk_rope_launch_failure_permanently_falls_back(monkeypatch):
    from vllm_omni.diffusion.models.longcat_image import longcat_image_transformer as longcat

    torch.manual_seed(3)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(longcat, "fused_qk_rope_supported", lambda *args: True)
    fused_calls = 0

    def broken_fused(query, key, cos, sin):
        nonlocal fused_calls
        fused_calls += 1
        raise RuntimeError("kernel failed")

    monkeypatch.setattr(longcat, "fused_qk_rope", broken_fused)
    q, k, rotary_emb = _inputs()
    expected = _reference(q, k, rotary_emb)
    with torch.no_grad():
        first = longcat._apply_qk_rope(q, k, rotary_emb, 1, 0)
        second = longcat._apply_qk_rope(q, k, rotary_emb, 1, 0)

    assert fused_calls == 1
    assert len(longcat._FAILED_QK_ROPE_SIGNATURES) == 1
    for actual in (first, second):
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])


def test_failed_qk_rope_signature_cache_is_bounded():
    from vllm_omni.diffusion.models.longcat_image import longcat_image_transformer as longcat

    for index in range(longcat._FAILED_QK_ROPE_SIGNATURES_MAX_SIZE + 1):
        longcat._record_failed_qk_rope_signature((index,))

    assert len(longcat._FAILED_QK_ROPE_SIGNATURES) == longcat._FAILED_QK_ROPE_SIGNATURES_MAX_SIZE
    assert (0,) not in longcat._FAILED_QK_ROPE_SIGNATURES
    assert (longcat._FAILED_QK_ROPE_SIGNATURES_MAX_SIZE,) in longcat._FAILED_QK_ROPE_SIGNATURES


class _FakeQKV(torch.nn.Module):
    def __init__(self, heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads = heads
        self.num_kv_heads = heads
        self.width = heads * head_dim

    def forward(self, x):
        return torch.cat((x, x + 1, x + 2), dim=-1), None


class _AddNorm(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, x):
        return x + self.value


class _CaptureAttention(torch.nn.Module):
    def forward(self, query, key, value, metadata=None):
        del key, value, metadata
        return query


class _PassthroughLinear(torch.nn.Module):
    def forward(self, x):
        return x, None


def test_dual_attention_keeps_native_norm_and_text_image_concat_order(monkeypatch):
    from vllm_omni.diffusion.models.longcat_image import longcat_image_transformer as longcat

    attention = longcat.LongCatImageAttention.__new__(longcat.LongCatImageAttention)
    torch.nn.Module.__init__(attention)
    attention.parallel_config = SimpleNamespace(sequence_parallel_size=1)
    attention.head_dim = 4
    attention.added_kv_proj_dim = 8
    attention.to_qkv = _FakeQKV(2, 4)
    attention.add_kv_proj = _FakeQKV(2, 4)
    attention.norm_q = _AddNorm(10)
    attention.norm_k = _AddNorm(20)
    attention.norm_added_q = _AddNorm(100)
    attention.norm_added_k = _AddNorm(200)
    attention.attn = _CaptureAttention()
    attention.to_out = _PassthroughLinear()
    attention.to_add_out = _PassthroughLinear()

    captured = []

    def paired(query, key, rotary_emb, sp_size, min_tokens):
        captured.append((query.clone(), key.clone(), rotary_emb, sp_size, min_tokens))
        return query, key

    monkeypatch.setattr(longcat, "_apply_qk_rope", paired)
    image = torch.randn(1, 3, 8)
    text = torch.randn(1, 2, 8)
    rotary_pair = (torch.randn(5, 4), torch.randn(5, 4))
    rotary_emb = (*rotary_pair, 0)
    image_out, text_out = attention(image, text, rotary_emb)

    image_q = image.unflatten(-1, (2, 4))
    text_q = text.unflatten(-1, (2, 4))
    expected_q = torch.cat((text_q + 100, image_q + 10), dim=1)
    expected_k = torch.cat((text_q + 1 + 200, image_q + 1 + 20), dim=1)
    assert len(captured) == 1
    assert torch.equal(captured[0][0], expected_q)
    assert torch.equal(captured[0][1], expected_k)
    assert captured[0][2][0] is rotary_pair[0]
    assert captured[0][2][1] is rotary_pair[1]
    assert captured[0][3:] == (1, 0)
    assert image_out.shape == image.shape
    assert text_out.shape == text.shape
