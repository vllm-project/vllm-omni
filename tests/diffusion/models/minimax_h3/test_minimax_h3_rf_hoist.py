# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Guards for MiniMax-H3's block-sparse row-reorder hoist.

The hoist rewrites the whole packed layout, so it is only safe when every
consumer agrees on the new row order. These tests pin the refusals: reference
KV addresses rows in pre-reorder coordinates, VDN-H3 has its own geometry, and
dense attention never reads the pre-rearranged flag.
"""

from types import SimpleNamespace
from unittest import mock

import pytest

from vllm_omni.diffusion.models.minimax_h3.denoise_loop import (
    _RF_HOIST_ENV,
    _rf_hoist_allowed,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _FakeAttention:
    def __init__(self, backend: str = "RAINFUSION_ATTN", sparsity: float = 0.8) -> None:
        self.attn_spec = SimpleNamespace(
            backend=backend,
            block_sparse=SimpleNamespace(sparsity=sparsity) if sparsity is not None else None,
        )


def _model(*, kv_offload_config=None, vdn_config=None, attention=None):
    modules = [attention] if attention is not None else []
    return SimpleNamespace(
        od_config=SimpleNamespace(kv_offload_config=kv_offload_config),
        vdn_config=vdn_config,
        modules=lambda: modules,
    )


def _allow(model, *, reference_kv=None):
    fake_attention = type(model.modules()[0]) if model.modules() else _FakeAttention
    with (
        mock.patch("vllm_omni.diffusion.attention.layer.Attention", fake_attention),
        mock.patch(
            "vllm_omni.diffusion.kv_offload_config.parse_reference_kv_config",
            return_value=reference_kv,
        ),
    ):
        return _rf_hoist_allowed(model)


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv(_RF_HOIST_ENV, raising=False)

    assert _allow(_model(attention=_FakeAttention())) is False


def test_allows_sparse_rainfusion_without_reference_kv(monkeypatch):
    monkeypatch.setenv(_RF_HOIST_ENV, "1")

    assert _allow(_model(kv_offload_config={}, attention=_FakeAttention())) is True


def test_refuses_reference_kv(monkeypatch):
    monkeypatch.setenv(_RF_HOIST_ENV, "1")
    model = _model(kv_offload_config={"tier2": True}, attention=_FakeAttention())

    assert _allow(model, reference_kv=object()) is False


def test_refuses_vdn(monkeypatch):
    monkeypatch.setenv(_RF_HOIST_ENV, "1")
    model = _model(vdn_config=object(), attention=_FakeAttention())

    assert _allow(model) is False


@pytest.mark.parametrize(
    "attention",
    [
        _FakeAttention(backend="FLASH_ATTN"),
        _FakeAttention(backend="RAINFUSION_ATTN", sparsity=0.0),
    ],
)
def test_refuses_dense_attention(monkeypatch, attention):
    monkeypatch.setenv(_RF_HOIST_ENV, "1")

    assert _allow(_model(attention=attention)) is False
