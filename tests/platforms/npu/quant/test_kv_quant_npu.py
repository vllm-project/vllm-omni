# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for NPU FP8 KV quantization helpers.

These tests load ``kv_quant_npu`` from its source file via ``importlib`` so
the test module itself does not ``import vllm_omni`` (which would pull
``patch`` → ``aenum``, vLLM, etc.).
"""

from __future__ import annotations

import importlib.util
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
            "op_args": None,
            "op_kwargs": None,
        }

        def fake_fp8_op(q, k, v, **kwargs):
            captured["op_args"] = (q, k, v)
            captured["op_kwargs"] = kwargs
            return torch.ones_like(q)

        fake_qua_rot_mode = SimpleNamespace(HADAMARD="hadamard")

        def fake_create_rot(mode, head_dim, seed):
            assert mode == "hadamard"
            assert seed == 425500
            return torch.eye(head_dim, dtype=torch.float32)

        monkeypatch.setattr(
            kv_quant_npu,
            "_load_sd_ops",
            lambda: (fake_fp8_op, fake_qua_rot_mode, fake_create_rot),
        )

        return captured

    @staticmethod
    def _make_qkv(shape: tuple[int, int, int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = torch.randn(*shape, dtype=torch.float32)
        key = torch.randn(*shape, dtype=torch.float32)
        value = torch.randn(*shape, dtype=torch.float32)
        return query, key, value

    @pytest.mark.parametrize(
        "layout,input_shape,softmax_scale",
        [
            ("BNSD", (2, 3, 4, 8), None),
            ("BSND", (2, 4, 3, 8), 0.125),
        ],
    )
    def test_fp8_rotate_quant_fa_forwards_layout_scale_and_rotation(
        self,
        fake_quant_ops: dict[str, Any],
        layout: str,
        input_shape: tuple[int, int, int, int],
        softmax_scale: float | None,
    ) -> None:
        query, key, value = self._make_qkv(input_shape)

        out = kv_quant_npu.fp8_rotate_quant_fa(query, key, value, layout=layout, softmax_scale=softmax_scale)

        assert out.shape == query.shape
        assert out.dtype == query.dtype
        op_query, op_key, op_value = fake_quant_ops["op_args"]
        assert op_query is query
        assert op_key is key
        assert op_value is value
        assert fake_quant_ops["op_kwargs"]["layout"] == layout
        assert fake_quant_ops["op_kwargs"]["softmax_scale"] == softmax_scale
        q_rot = fake_quant_ops["op_kwargs"]["q_rot"]
        k_rot = fake_quant_ops["op_kwargs"]["k_rot"]
        assert q_rot is k_rot
        assert q_rot.shape == (input_shape[-1], input_shape[-1])
        assert q_rot.dtype == query.dtype

    def test_fp8_rotate_quant_fa_invalid_layout_raises(self, fake_quant_ops) -> None:
        query = torch.randn(1, 2, 3, 4, dtype=torch.float32)
        key = torch.randn(1, 2, 3, 4, dtype=torch.float32)
        value = torch.randn(1, 2, 3, 4, dtype=torch.float32)

        with pytest.raises(ValueError, match="unsupported layout"):
            kv_quant_npu.fp8_rotate_quant_fa(query, key, value, layout="INVALID")


@npu_smoke
class TestKVQuantNPUSmoke:
    """Smoke tests using real torch_npu/mindiesd stack, only on NPU."""

    def test_fp8_rotate_quant_fa_real_npu_shape_contract(self):
        try:
            kv_quant_npu._load_sd_ops.cache_clear()
            kv_quant_npu._load_sd_ops()
        except ImportError:
            pytest.skip("NPU quant dependencies are not fully installed.")

        query = torch.randn(1, 2, 4, 64, dtype=torch.float16, device="npu")
        key = torch.randn(1, 2, 4, 64, dtype=torch.float16, device="npu")
        value = torch.randn(1, 2, 4, 64, dtype=torch.float16, device="npu")

        out = kv_quant_npu.fp8_rotate_quant_fa(query, key, value, layout="BNSD")
        assert out.shape == query.shape
        assert out.dtype == query.dtype
