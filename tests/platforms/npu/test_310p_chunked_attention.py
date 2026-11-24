# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for the 310P chunked attention fallback.

The patch module is loaded from source so the tests exercise the fallback
numerics without importing vllm_omni or touching NPU kernels.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# More queries than one chunk, so rows past the first chunk are sliced.
_QUERY_LEN = 258
_KEY_LEN = 300
_HEAD_DIM = 16


def _repo_root() -> Path:
    marker = Path("vllm_omni") / "platforms" / "npu" / "_310p" / "patch"
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).is_dir():
            return parent
    raise FileNotFoundError(f"could not locate repo root containing {marker}")


def _load_source_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def chunked_attention():
    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "attention.py"
    return _load_source_module("_310p_patch_attention_under_test", path).chunked_attention


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    query = torch.randn(1, 2, _QUERY_LEN, _HEAD_DIM)
    key = torch.randn(1, 2, _KEY_LEN, _HEAD_DIM)
    value = torch.randn(1, 2, _KEY_LEN, _HEAD_DIM)
    return query, key, value


@pytest.mark.parametrize("dtype", [torch.bool, torch.float32])
def test_query_broadcast_mask_matches_sdpa(chunked_attention, dtype):
    """A key-padding mask shared by all queries must keep its query axis."""
    query, key, value = _inputs()
    if dtype is torch.bool:
        mask = torch.rand(1, 1, 1, _KEY_LEN) > 0.2
    else:
        mask = torch.zeros(1, 1, 1, _KEY_LEN)
        mask[..., -17:] = torch.finfo(torch.float32).min

    expected = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
    actual = chunked_attention(query, key, value, scale=_HEAD_DIM**-0.5, causal=False, attn_mask=mask)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_full_mask_matches_sdpa(chunked_attention):
    """A per-query mask is sliced per chunk."""
    query, key, value = _inputs()
    mask = torch.rand(1, 1, _QUERY_LEN, _KEY_LEN) > 0.2

    expected = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
    actual = chunked_attention(query, key, value, scale=_HEAD_DIM**-0.5, causal=False, attn_mask=mask)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
