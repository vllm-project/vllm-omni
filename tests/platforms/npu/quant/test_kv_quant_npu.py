# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for NPU FP8 KV quantization helpers.

These tests load ``kv_quant_npu`` from its source file via ``importlib`` so
the test module itself does not ``import vllm_omni`` (which would pull
``patch`` → ``aenum``, vLLM, etc.).
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _repo_root() -> Path:
    """Resolve checkout root (parent of ``vllm_omni/``), not ``tests/``."""
    here = Path(__file__).resolve()
    marker = Path("vllm_omni") / "platforms" / "npu" / "quant" / "kv_quant_npu.py"
    for parent in here.parents:
        if (parent / marker).is_file():
            return parent
    msg = f"could not locate repo root (no {marker}) starting from {here}"
    raise FileNotFoundError(msg)


def _load_kv_quant_npu() -> ModuleType:
    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "quant" / "kv_quant_npu.py"
    if not path.is_file():
        msg = f"kv_quant_npu source not found: {path}"
        raise FileNotFoundError(msg)
    name = "vllm_omni_test_kv_quant_npu_standalone"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        msg = f"cannot load import spec for {path}"
        raise RuntimeError(msg)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kv_quant_npu = _load_kv_quant_npu()


def _npu_smoke_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return bool(hasattr(torch, "npu") and torch.npu.is_available())


npu_smoke = pytest.mark.skipif(not _npu_smoke_available(), reason="NPU device or torch_npu not available.")


def test_is_quantized_kv_cache() -> None:
    assert kv_quant_npu.is_quantized_kv_cache("fp8")
    assert not kv_quant_npu.is_quantized_kv_cache(None)
    assert not kv_quant_npu.is_quantized_kv_cache("int8")


class TestKVQuantNPUUnit:
    @pytest.fixture(autouse=True)
    def clear_rot_cache(self):
        kv_quant_npu._ROT_MATRIXS.clear()

    def test_get_rot_matrix_caches_by_device_dtype_and_head_dim(self) -> None:
        calls = {"count": 0}

        class FakeQuaRotMode:
            HADAMARD = "hadamard"

        def fake_create_rot(mode, head_dim, seed):
            calls["count"] += 1
            assert mode == FakeQuaRotMode.HADAMARD
            assert seed == 425500
            return torch.eye(head_dim, dtype=torch.float32)

        device = torch.device("cpu")
        rot_1 = kv_quant_npu._get_rot_matrix(device, torch.float16, 8, FakeQuaRotMode, fake_create_rot)
        rot_2 = kv_quant_npu._get_rot_matrix(device, torch.float16, 8, FakeQuaRotMode, fake_create_rot)
        rot_3 = kv_quant_npu._get_rot_matrix(device, torch.bfloat16, 8, FakeQuaRotMode, fake_create_rot)
        rot_4 = kv_quant_npu._get_rot_matrix(device, torch.float16, 16, FakeQuaRotMode, fake_create_rot)

        assert calls["count"] == 3
        assert rot_1 is rot_2
        assert rot_3.dtype == torch.bfloat16
        assert rot_4.shape == (16, 16)

    @pytest.fixture
    def fake_quant_ops(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        captured: dict[str, Any] = {
            "fa_calls": [],
            "npu_kwargs": None,
            "out_shape": None,
        }

        class FakeTorchNPU:
            float8_e4m3fn = "fp8_marker"

            @staticmethod
            def npu_fused_infer_attention_score_v2(q, k, v, **kwargs):
                del q, k, v
                captured["npu_kwargs"] = kwargs
                out_shape = captured["out_shape"]
                return (torch.ones(out_shape, dtype=torch.float32),)

        def fake_fa_block_quant_preprocess(x, block_size, dst_type, layout):
            captured["fa_calls"].append(
                {
                    "block_size": block_size,
                    "layout": layout,
                    "dst_type": dst_type,
                    "shape": tuple(x.shape),
                }
            )
            scale = torch.full((1,), float(block_size), dtype=torch.float32)
            return x, scale

        fake_qua_rot_mode = SimpleNamespace(HADAMARD="hadamard")

        def fake_create_rot(mode, head_dim, seed):
            assert mode == "hadamard"
            assert seed == 425500
            return torch.eye(head_dim, dtype=torch.float32)

        monkeypatch.setattr(
            kv_quant_npu,
            "_load_quant_ops",
            lambda: (FakeTorchNPU, fake_fa_block_quant_preprocess, fake_qua_rot_mode, fake_create_rot),
        )

        return captured

    @staticmethod
    def _make_qkv(shape: tuple[int, int, int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = torch.randn(*shape, dtype=torch.float32)
        key = torch.randn(*shape, dtype=torch.float32)
        value = torch.randn(*shape, dtype=torch.float32)
        return query, key, value

    @pytest.mark.parametrize(
        "layout,input_shape,out_shape,softmax_scale,expected_scale",
        [
            ("BNSD", (2, 3, 4, 8), (2, 3, 6, 8), None, 1.0 / math.sqrt(8)),
            ("BSND", (2, 4, 3, 8), (2, 6, 3, 8), 0.125, 0.125),
        ],
    )
    def test_fp8_rotate_quant_fa_layouts_scale_and_crop(
        self,
        fake_quant_ops: dict[str, Any],
        layout: str,
        input_shape: tuple[int, int, int, int],
        out_shape: tuple[int, int, int, int],
        softmax_scale: float | None,
        expected_scale: float,
    ) -> None:
        query, key, value = self._make_qkv(input_shape)
        fake_quant_ops["out_shape"] = out_shape

        out = kv_quant_npu.fp8_rotate_quant_fa(query, key, value, layout=layout, softmax_scale=softmax_scale)

        assert out.shape == query.shape
        assert out.dtype == query.dtype
        assert fake_quant_ops["npu_kwargs"]["input_layout"] == layout
        # BNSD: shape[1]==heads, BSND: shape[2]==heads.
        expected_heads = input_shape[1] if layout == "BNSD" else input_shape[2]
        assert fake_quant_ops["npu_kwargs"]["num_query_heads"] == expected_heads
        assert fake_quant_ops["npu_kwargs"]["softmax_scale"] == pytest.approx(expected_scale)
        assert [call["block_size"] for call in fake_quant_ops["fa_calls"]] == [128, 256, 256]

    def test_fp8_rotate_quant_fa_invalid_layout_raises(self, fake_quant_ops) -> None:
        query = torch.randn(1, 2, 3, 4, dtype=torch.float32)
        key = torch.randn(1, 2, 3, 4, dtype=torch.float32)
        value = torch.randn(1, 2, 3, 4, dtype=torch.float32)
        fake_quant_ops["out_shape"] = (1, 2, 3, 4)

        with pytest.raises(ValueError, match="unsupported layout"):
            kv_quant_npu.fp8_rotate_quant_fa(query, key, value, layout="INVALID")


@npu_smoke
class TestKVQuantNPUSmoke:
    """Smoke tests using real torch_npu/mindiesd stack, only on NPU."""

    def test_fp8_rotate_quant_fa_real_npu_shape_contract(self):
        try:
            kv_quant_npu._load_quant_ops.cache_clear()
            kv_quant_npu._load_quant_ops()
        except ImportError:
            pytest.skip("NPU quant dependencies are not fully installed.")

        query = torch.randn(1, 2, 4, 64, dtype=torch.float16, device="npu")
        key = torch.randn(1, 2, 4, 64, dtype=torch.float16, device="npu")
        value = torch.randn(1, 2, 4, 64, dtype=torch.float16, device="npu")

        out = kv_quant_npu.fp8_rotate_quant_fa(query, key, value, layout="BNSD")
        assert out.shape == query.shape
        assert out.dtype == query.dtype
