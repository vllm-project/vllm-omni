# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Attention dtype routing for MiniMax Music 3's float32 acoustic stage."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.minimax_music3 import dit

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


class _Backend(nn.Module):
    def __init__(self, *, name: str, explicit: bool) -> None:
        super().__init__()
        self.attn_backend = _BackendType(name)
        self.backend_explicit = explicit
        self.calls = 0

    def forward(self, q, k, v):
        del k, v
        self.calls += 1
        return torch.full_like(q, 7)


def _attention(monkeypatch, *, backend_name: str, explicit: bool):
    backend = _Backend(name=backend_name, explicit=explicit)
    monkeypatch.setattr(dit, "_build_native_attention", lambda **_kwargs: backend)
    return dit.Attention(8, head_dim=4), backend


@pytest.mark.parametrize(
    "backend_name",
    _FLOAT32_INCOMPATIBLE_BACKENDS,
)
def test_float32_uses_sdpa_for_automatic_incompatible_backend(monkeypatch, backend_name):
    attention, backend = _attention(monkeypatch, backend_name=backend_name, explicit=False)
    q = torch.randn(2, 5, 2, 4)

    actual = attention._attend(q, q, q)

    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        q.transpose(1, 2),
        q.transpose(1, 2),
        is_causal=False,
        scale=attention.softmax_scale,
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected)
    assert backend.calls == 0
    assert attention.backend_name == f"TORCH_SDPA(float32)/{backend_name}(lower precision)"


@pytest.mark.parametrize("backend_name", _FLOAT32_INCOMPATIBLE_BACKENDS)
@pytest.mark.parametrize(
    ("dtype", "explicit"),
    [
        (torch.float16, False),
        (torch.bfloat16, False),
        (torch.float32, True),
    ],
)
def test_lower_precision_or_explicit_backend_remains_authoritative(monkeypatch, backend_name, dtype, explicit):
    attention, backend = _attention(monkeypatch, backend_name=backend_name, explicit=explicit)
    q = torch.zeros(1, 3, 2, 4, dtype=dtype)

    actual = attention._attend(q, q, q)

    assert backend.calls == 1
    torch.testing.assert_close(actual, torch.full_like(q, 7))


def test_float32_uses_sdpa_when_native_attention_is_unavailable(monkeypatch):
    monkeypatch.setattr(dit, "_build_native_attention", lambda **_kwargs: None)
    attention = dit.Attention(8, head_dim=4)
    q = torch.randn(1, 3, 2, 4)

    actual = attention._attend(q, q, q)

    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        q.transpose(1, 2),
        q.transpose(1, 2),
        is_causal=False,
        scale=attention.softmax_scale,
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected)
    assert attention.backend_name == "TORCH_SDPA"


def test_float32_keeps_compatible_automatic_backend(monkeypatch):
    attention, backend = _attention(monkeypatch, backend_name="SDPA", explicit=False)
    q = torch.zeros(1, 3, 2, 4)

    actual = attention._attend(q, q, q)

    assert backend.calls == 1
    torch.testing.assert_close(actual, torch.full_like(q, 7))
