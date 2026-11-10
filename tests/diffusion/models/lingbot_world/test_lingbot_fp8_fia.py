# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU ABI checks for MindIE dense FP8 EagleFIA, with no NPU execution."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def load():
    path = Path(__file__).resolve().parents[4] / "vllm_omni/platforms/npu/attention/eagle_fp8_fia.py"
    spec = importlib.util.spec_from_file_location("_fp8_fia", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("q_len,k_len", [(128, 128), (257, 2049), (4524, 27144)])
@pytest.mark.parametrize("kv_head_major", [False, True])
def test_dense_rectangular_tails_and_mode(monkeypatch, q_len, k_len, kv_head_major):
    provider = ModuleType("mindiesd.layers.flash_attn.quant_flash_attn")
    mode = ModuleType("mindiesd.quantization.mode")
    sentinel = object()
    mode.FP8FAMode = lambda name: sentinel if name == "C8V16_TILING512" else None
    calls = []

    def quant(q, k, v, **kw):
        calls.append(kw)
        assert q.shape == (1, 2, q_len, 128)
        assert k.shape == v.shape == (1, 2, k_len, 128)
        assert q.is_contiguous() and k.is_contiguous()
        assert kw == dict(precision="fp8", layout="BNSD", scale=0.125, fp8_fa_mode=sentinel)
        return q.clone()

    provider.quant_attention = quant
    monkeypatch.setitem(sys.modules, provider.__name__, provider)
    monkeypatch.setitem(sys.modules, mode.__name__, mode)
    q = torch.randn(q_len, 2, 128, dtype=torch.bfloat16)
    k = torch.randn(k_len, 2, 128, dtype=q.dtype)
    if kv_head_major:
        k = k.transpose(0, 1).contiguous()
    output = load().eagle_fp8_fia(q, k, k, scale=0.125, kv_head_major=kv_head_major)
    assert len(calls) == 1 and torch.equal(output, q) and output.is_contiguous()


def test_reject_dtype_before_provider_import():
    x = torch.zeros(4, 2, 128)
    with pytest.raises(ValueError, match="BF16"):
        load().eagle_fp8_fia(x, x, x, scale=0.125)
