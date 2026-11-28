"""Validate strict precision without requiring CUDA/NPU hardware."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

FILE = Path(__file__).resolve().parents[3] / "vllm_omni/diffusion/models/cosmos3/numerical_precision.py"
spec = importlib.util.spec_from_file_location("numerical_precision", FILE)
precision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(precision)


def test_default_preserves_gpu_policy(monkeypatch):
    monkeypatch.delenv("COSMOS3_STRICT_PRECISION", raising=False)
    assert not precision.configure_numerical_precision(torch.device("cuda"), torch.bfloat16)


def test_strict_rejects_reduced_precision(monkeypatch):
    monkeypatch.setenv("COSMOS3_STRICT_PRECISION", "1")
    with pytest.raises(ValueError, match="float32"):
        precision.configure_numerical_precision(torch.device("cuda"), torch.bfloat16)


def test_strict_cuda_disables_tf32(monkeypatch):
    monkeypatch.setenv("COSMOS3_STRICT_PRECISION", "1")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    assert precision.configure_numerical_precision(torch.device("cuda"), torch.float32)
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32


def test_strict_ascend_disables_hf32_in_worker(monkeypatch):
    options = []
    npu = SimpleNamespace(
        set_option=options.append, matmul=SimpleNamespace(allow_hf32=True), conv=SimpleNamespace(allow_hf32=True)
    )
    monkeypatch.setenv("COSMOS3_STRICT_PRECISION", "1")
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    assert precision.configure_numerical_precision(SimpleNamespace(type="npu"), torch.float32)
    assert options == [{"ACL_PRECISION_MODE": "must_keep_origin_dtype"}]
    assert not npu.matmul.allow_hf32
    assert not npu.conv.allow_hf32
