# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Attention dtype routing for GLM-TTS's float32 DiT stage."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm_omni.model_executor.models.glm_tts.glm_tts_dit import DiTAttention

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_FLOAT32_INCOMPATIBLE_BACKENDS = [
    "FLASH_ATTN",
    "FLASH_ATTN_HUB",
    "FLASH_ATTN_3_HUB",
    "CUDNN_ATTN",
    "FLASHINFER_ATTN",
]


class _BackendType:
    def __init__(self, name: str) -> None:
        self.name = name

    def get_name(self) -> str:
        return self.name


class _Spec:
    def __init__(self) -> None:
        self.name = "FLASH_ATTN"
        self.explicit = False


_SPEC = _Spec()


class _FakeDiffusionAttention(nn.Module):
    """Stands in for the project diffusion attention layer, recording calls."""

    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.attn_backend = _BackendType(_SPEC.name)
        self.backend_explicit = _SPEC.explicit
        self.calls = 0

    def forward(self, q, k, v, attn_metadata=None):
        del k, v, attn_metadata
        self.calls += 1
        return torch.full_like(q, 7)


@pytest.fixture
def fake_attention(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.model_executor.models.glm_tts.glm_tts_dit.DiffusionAttention", _FakeDiffusionAttention
    )


def _build_attention(*, backend_name: str, explicit: bool) -> DiTAttention:
    _SPEC.name = backend_name
    _SPEC.explicit = explicit
    attention = DiTAttention(dim=16, heads=2, dim_head=8, dropout=0.0).eval()
    assert isinstance(attention.attn, _FakeDiffusionAttention)
    return attention


def _reference_forward(attention: DiTAttention, x: torch.Tensor) -> torch.Tensor:
    """Independent SDPA reference through to_out, mirroring the forward path."""
    query = attention.to_q(x)
    key = attention.to_k(x)
    value = attention.to_v(x)
    batch_size, seq_len = x.shape[0], x.shape[1]
    query = query.view(batch_size, seq_len, attention.heads, attention.dim_head)
    key = key.view(batch_size, seq_len, attention.heads, attention.dim_head)
    value = value.view(batch_size, seq_len, attention.heads, attention.dim_head)
    out = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
        scale=attention.scale,
    ).transpose(1, 2)
    out = out.view(batch_size, seq_len, attention.inner_dim)
    return attention.to_out(out)


@pytest.mark.parametrize("backend_name", _FLOAT32_INCOMPATIBLE_BACKENDS)
def test_float32_uses_sdpa_for_automatic_incompatible_backend(backend_name, fake_attention):
    attention = _build_attention(backend_name=backend_name, explicit=False)
    x = torch.randn(2, 5, 16)

    actual = attention(x)

    torch.testing.assert_close(actual, _reference_forward(attention, x))
    assert attention.attn.calls == 0


@pytest.mark.parametrize("backend_name", _FLOAT32_INCOMPATIBLE_BACKENDS)
def test_explicit_backend_remains_authoritative(backend_name, fake_attention):
    attention = _build_attention(backend_name=backend_name, explicit=True)
    x = torch.zeros(1, 3, 16)

    actual = attention(x)

    batch_size, seq_len = x.shape[0], x.shape[1]
    expected_hidden = torch.full((batch_size, seq_len, attention.inner_dim), 7, dtype=x.dtype)
    torch.testing.assert_close(actual, attention.to_out(expected_hidden))
    assert attention.attn.calls == 1


def test_float32_keeps_compatible_automatic_backend(fake_attention):
    attention = _build_attention(backend_name="TORCH_SDPA", explicit=False)
    x = torch.zeros(1, 3, 16)

    actual = attention(x)

    batch_size, seq_len = x.shape[0], x.shape[1]
    expected_hidden = torch.full((batch_size, seq_len, attention.inner_dim), 7, dtype=x.dtype)
    torch.testing.assert_close(actual, attention.to_out(expected_hidden))
    assert attention.attn.calls == 1
