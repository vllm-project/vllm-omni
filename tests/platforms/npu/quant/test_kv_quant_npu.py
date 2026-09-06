# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
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


def _load_chunking() -> ModuleType:
    """Load the framework-level chunking scheduler standalone (same pattern).

    The FIA adapter consumes its plan duck-typed (no runtime import), so the
    plan objects used here double as an interop check that a real ChunkCall
    sequence drives the chunked dispatch.
    """
    import sys

    path = _repo_root() / "vllm_omni" / "diffusion" / "attention" / "chunking.py"
    name = "vllm_omni_test_chunking_standalone"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        msg = f"cannot load import spec for {path}"
        raise RuntimeError(msg)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: dataclass KW_ONLY detection looks the module up
    # in sys.modules and crashes on a missing entry.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


chunking = _load_chunking()


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
            "fia_calls": [],
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

        def fake_fia_v2(q, k, v, **kwargs):
            captured["fia_calls"].append(
                {
                    "q_shape": tuple(q.shape),
                    "k_shape": tuple(k.shape),
                    "v_shape": tuple(v.shape),
                    "kwargs": kwargs,
                }
            )
            captured["npu_kwargs"] = kwargs
            out_dtype = kwargs.get("out_dtype") or torch.float32
            out_shape = captured["out_shape"]
            if out_shape is not None:
                return (torch.ones(out_shape, dtype=out_dtype),)
            # Default: echo q (FIA output has q's shape), so dispatch contract
            # tests can rely on the wrapper's own trim/transpose logic.
            return (q.to(out_dtype),)

        def fake_fa_block_quant_preprocess(x, block_size, dst_type, layout):
            captured["fa_calls"].append(
                {
                    "block_size": block_size,
                    "layout": layout,
                    "dst_type": dst_type,
                    "shape": tuple(x.shape),
                }
            )
            # Mirror the real kernel contract: BSND inputs are transposed
            # before quantization, so the returned tensor is ALWAYS
            # BNSD-logical [B, N, S, D] and the scale is per (head, row-block):
            # [B, N, ceil(S / block_size), ceil(D / 128)].
            if layout == "BSND":
                x = x.transpose(1, 2)
            x = x.contiguous()
            b, n, s, _d = x.shape
            blocks = -(-s // block_size)
            scale = torch.ones(b, n, blocks, 1, dtype=torch.float32)
            return x, scale

        fake_qua_rot_mode = SimpleNamespace(HADAMARD="hadamard")

        def fake_create_rot(mode, head_dim, seed):
            assert mode == "hadamard"
            assert seed == 425500
            return torch.eye(head_dim, dtype=torch.float32)

        monkeypatch.setattr(
            kv_quant_npu,
            "_load_quant_ops",
            lambda: (FakeTorchNPU, fake_fia_v2, fake_fa_block_quant_preprocess, fake_qua_rot_mode, fake_create_rot),
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

    @pytest.mark.parametrize(
        "layout,input_shape,fia_out_shape,expected_quant_shapes",
        [
            # Quant always returns BNSD-logical tensors, so the FIA output is
            # BNSD too; the wrapper transposes back to the caller's layout.
            ("BSND", (1, 8, 2, 4), (1, 2, 8, 4), [(1, 8, 2, 4), (1, 5, 2, 4), (1, 5, 2, 4)]),
            ("BNSD", (1, 2, 8, 4), (1, 2, 8, 4), [(1, 2, 8, 4), (1, 2, 5, 4), (1, 2, 5, 4)]),
        ],
    )
    def test_fp8_rotate_quant_kv_slice_dense_contract(
        self,
        fake_quant_ops: dict[str, Any],
        layout: str,
        input_shape: tuple[int, int, int, int],
        fia_out_shape: tuple[int, int, int, int],
        expected_quant_shapes: list[tuple[int, int, int, int]],
    ) -> None:
        query, key, value = self._make_qkv(input_shape)
        fake_quant_ops["out_shape"] = fia_out_shape

        out = kv_quant_npu.fp8_rotate_quant_kv_slice(query, key, value, 5, layout=layout, softmax_scale=0.125)

        assert out.shape == query.shape
        # Q is quantized at full length; K/V are sliced to kv_len BEFORE quant.
        assert [call["shape"] for call in fake_quant_ops["fa_calls"]] == expected_quant_shapes
        assert [call["layout"] for call in fake_quant_ops["fa_calls"]] == [layout] * 3
        assert [call["block_size"] for call in fake_quant_ops["fa_calls"]] == [128, 256, 256]
        # Single dense FIA call.
        assert len(fake_quant_ops["fia_calls"]) == 1
        kwargs = fake_quant_ops["npu_kwargs"]
        # Quant output is always BNSD-logical, so the FIA dispatch is BNSD
        # regardless of the caller-facing layout.
        assert kwargs["input_layout"] == "BNSD"
        # Dense dispatch: the FIA varlen feature stays off.
        assert "actual_seq_qlen" not in kwargs
        assert "actual_seq_kvlen" not in kwargs
        assert kwargs["query_quant_mode"] == 7
        assert kwargs["key_quant_mode"] == 7
        assert kwargs["value_quant_mode"] == 7
        assert kwargs["softmax_scale"] == 0.125
        expected_heads = input_shape[1] if layout == "BNSD" else input_shape[2]
        assert kwargs["num_query_heads"] == expected_heads
        assert kwargs["num_key_value_heads"] == expected_heads

    def test_fp8_rotate_quant_kv_slice_full_length_kv_is_noop_slice(self, fake_quant_ops) -> None:
        query, key, value = self._make_qkv((1, 8, 2, 4))
        fake_quant_ops["out_shape"] = (1, 2, 8, 4)  # FIA output is BNSD-logical

        out = kv_quant_npu.fp8_rotate_quant_kv_slice(query, key, value, 8, layout="BSND")

        assert out.shape == query.shape
        assert [call["shape"] for call in fake_quant_ops["fa_calls"]] == [(1, 8, 2, 4)] * 3

    @pytest.mark.parametrize("kv_len", [0, -1, 9, 5.0])
    def test_fp8_rotate_quant_kv_slice_invalid_kv_len_raises(self, fake_quant_ops, kv_len) -> None:
        query, key, value = self._make_qkv((1, 8, 2, 4))

        with pytest.raises(ValueError, match="kv_len"):
            kv_quant_npu.fp8_rotate_quant_kv_slice(query, key, value, kv_len, layout="BSND")

    def test_fp8_rotate_quant_kv_slice_invalid_layout_raises(self, fake_quant_ops) -> None:
        query, key, value = self._make_qkv((1, 8, 2, 4))

        with pytest.raises(ValueError, match="unsupported layout"):
            kv_quant_npu.fp8_rotate_quant_kv_slice(query, key, value, 5, layout="INVALID")

    # --- Chunked dispatch (plan from the generic chunking scheduler) -------

    @staticmethod
    def _chunk_plan(
        seq_len: int,
        num_heads: int,
        kv_len: int,
        *,
        q_chunk: int = 1,
        head_chunk: int = 0,
        num_kv_heads: int | None = None,
    ) -> list:
        options = chunking.AttnChunkingOptions(
            q_chunk=q_chunk,
            head_chunk=head_chunk,
            head_chunk_min_kv=0,  # gate off in tests
        )
        return chunking.build_chunk_plan(
            seq_len=seq_len,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads if num_kv_heads is not None else num_heads,
            kv_len=kv_len,
            options=options,
            row_align=kv_quant_npu._Q_BLOCK_SIZE,
        )

    def test_kv_slice_plan_splits_fia_calls_and_reassembles(self, fake_quant_ops) -> None:
        # BSND [1, T=256, N=2, D=4]; q_chunk=2 + head_chunk=1 → 2 q chunks x
        # 2 head slices = 4 FIA calls over one shared quantization.
        query, key, value = self._make_qkv((1, 256, 2, 4))
        plan = self._chunk_plan(256, 2, 200, q_chunk=2, head_chunk=1)

        out = kv_quant_npu.fp8_rotate_quant_kv_slice(
            query, key, value, 200, layout="BSND", softmax_scale=0.125, plan=plan
        )

        assert out.shape == query.shape
        # Quantization happens ONCE regardless of the plan.
        assert len(fake_quant_ops["fa_calls"]) == 3
        assert len(fake_quant_ops["fia_calls"]) == 4
        for call in fake_quant_ops["fia_calls"]:
            # Each dispatch: BNSD q slice (1, head_slice, 128, 4) against the
            # materialized K/V head part (1, head_slice, kv_len=200, 4).
            assert call["q_shape"] == (1, 1, 128, 4)
            assert call["k_shape"] == (1, 1, 200, 4)
            kwargs = call["kwargs"]
            assert kwargs["num_query_heads"] == 1
            assert kwargs["num_key_value_heads"] == 1
            assert kwargs["input_layout"] == "BNSD"
            # Exact block range of the full-length quantization: one 128-row
            # block per chunk.
            assert kwargs["dequant_scale_query"].shape == (1, 1, 1, 1)

    def test_kv_slice_plan_qchunk_only_keeps_full_heads(self, fake_quant_ops) -> None:
        query, key, value = self._make_qkv((1, 256, 2, 4))
        plan = self._chunk_plan(256, 2, 200, q_chunk=2)

        out = kv_quant_npu.fp8_rotate_quant_kv_slice(query, key, value, 200, layout="BSND", plan=plan)

        assert out.shape == query.shape
        assert len(fake_quant_ops["fia_calls"]) == 2
        for call in fake_quant_ops["fia_calls"]:
            assert call["q_shape"] == (1, 2, 128, 4)
            # No head chunking: the real GQA kv head count is passed through.
            assert call["kwargs"]["num_query_heads"] == 2
            assert call["kwargs"]["num_key_value_heads"] == 2

    def test_kv_slice_plan_ragged_tail_chunk(self, fake_quant_ops) -> None:
        # 200 rows with 128-block alignment → chunks (0,128) and (128,200).
        query, key, value = self._make_qkv((1, 200, 2, 4))
        plan = self._chunk_plan(200, 2, 200, q_chunk=2)

        out = kv_quant_npu.fp8_rotate_quant_kv_slice(query, key, value, 200, layout="BSND", plan=plan)

        assert out.shape == query.shape
        shapes = [call["q_shape"] for call in fake_quant_ops["fia_calls"]]
        assert shapes == [(1, 2, 128, 4), (1, 2, 72, 4)]

    def test_kv_slice_plan_single_call_matches_no_plan(self, fake_quant_ops) -> None:
        query, key, value = self._make_qkv((1, 256, 2, 4))
        single = [chunking.ChunkCall(0, 256, 0, 2)]

        out_plan = kv_quant_npu.fp8_rotate_quant_kv_slice(query, key, value, 200, layout="BSND", plan=single)
        assert len(fake_quant_ops["fia_calls"]) == 1
        out_plain = kv_quant_npu.fp8_rotate_quant_kv_slice(query, key, value, 200, layout="BSND")
        assert len(fake_quant_ops["fia_calls"]) == 2  # one more call recorded
        assert out_plan.shape == out_plain.shape == query.shape

    def test_kv_slice_plan_chunk_callback_consumes_chunks(self, fake_quant_ops) -> None:
        query, key, value = self._make_qkv((1, 256, 2, 4))
        plan = self._chunk_plan(256, 2, 200, q_chunk=2, head_chunk=1)
        chunks: list[tuple[torch.Tensor, tuple[int, int]]] = []

        def cb(out_chunk, call) -> None:
            chunks.append((out_chunk, (call.row0, call.row1)))

        result = kv_quant_npu.fp8_rotate_quant_kv_slice(
            query, key, value, 200, layout="BSND", plan=plan, chunk_callback=cb
        )

        assert result is None
        assert len(fake_quant_ops["fia_calls"]) == 4  # still fully executed
        # One head-merged chunk per q chunk, in caller (BSND) layout.
        assert [tuple(t.shape) for t, _ in chunks] == [(1, 128, 2, 4)] * 2
        assert [rows for _, rows in chunks] == [(0, 128), (128, 256)]

    def test_kv_slice_plan_misaligned_row_boundary_raises(self, fake_quant_ops) -> None:
        # Duck-typed plan (no chunking import needed at runtime); row0=100 is
        # off the 128-row block grid so scale slices would be inexact.
        query, key, value = self._make_qkv((1, 256, 2, 4))
        plan = [SimpleNamespace(row0=100, row1=200, h0=0, h1=2)]

        with pytest.raises(ValueError, match="row boundaries"):
            kv_quant_npu.fp8_rotate_quant_kv_slice(query, key, value, 200, layout="BSND", plan=plan)

    def test_kv_slice_plan_skips_calls_beyond_seq_len(self, fake_quant_ops) -> None:
        # Defensive: a plan covering rows past the real sequence contributes
        # nothing (its outputs would be dropped anyway).
        query, key, value = self._make_qkv((1, 256, 2, 4))
        plan = [
            SimpleNamespace(row0=0, row1=256, h0=0, h1=2),
            SimpleNamespace(row0=256, row1=384, h0=0, h1=2),
        ]

        out = kv_quant_npu.fp8_rotate_quant_kv_slice(query, key, value, 200, layout="BSND", plan=plan)

        assert out.shape == query.shape
        assert len(fake_quant_ops["fia_calls"]) == 1


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

    def test_fp8_rotate_quant_kv_slice_real_npu_shape_contract(self):
        try:
            kv_quant_npu._load_quant_ops.cache_clear()
            kv_quant_npu._load_quant_ops()
        except ImportError:
            pytest.skip("NPU quant dependencies are not fully installed.")

        query = torch.randn(1, 8, 2, 64, dtype=torch.float16, device="npu")
        key = torch.randn(1, 8, 2, 64, dtype=torch.float16, device="npu")
        value = torch.randn(1, 8, 2, 64, dtype=torch.float16, device="npu")

        out = kv_quant_npu.fp8_rotate_quant_kv_slice(query, key, value, 5, layout="BSND")
        assert out.shape == query.shape
        assert out.dtype == query.dtype

    def test_fp8_rotate_quant_kv_slice_chunked_matches_single_call(self):
        """Config-7 core contract on real hardware: chunked dispatch (q rows
        x heads) over one shared quantization matches the single wide call.

        Boundaries are 128-row aligned so per-chunk dequant scales are exact
        slices of the full-length scales — the outputs must agree bitwise.
        """
        try:
            kv_quant_npu._load_quant_ops.cache_clear()
            kv_quant_npu._load_quant_ops()
        except ImportError:
            pytest.skip("NPU quant dependencies are not fully installed.")

        torch.manual_seed(425500)
        seq_len, num_heads, head_dim, kv_len = 512, 4, 128, 400
        query = torch.randn(1, seq_len, num_heads, head_dim, dtype=torch.float16, device="npu")
        key = torch.randn(1, seq_len, num_heads, head_dim, dtype=torch.float16, device="npu")
        value = torch.randn(1, seq_len, num_heads, head_dim, dtype=torch.float16, device="npu")

        out_single = kv_quant_npu.fp8_rotate_quant_kv_slice(
            query, key, value, kv_len, layout="BSND", softmax_scale=0.088
        )
        plan = chunking.build_chunk_plan(
            seq_len=seq_len,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            kv_len=kv_len,
            options=chunking.AttnChunkingOptions(q_chunk=4, head_chunk=2, head_chunk_min_kv=0),
            row_align=kv_quant_npu._Q_BLOCK_SIZE,
        )
        assert len(plan) == 8  # 4 row chunks x 2 head slices
        out_chunked = kv_quant_npu.fp8_rotate_quant_kv_slice(
            query, key, value, kv_len, layout="BSND", softmax_scale=0.088, plan=plan
        )

        assert out_chunked.shape == out_single.shape
        assert torch.equal(out_single, out_chunked)
