# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the backend-neutral attention chunking scheduler.

These tests load ``chunking`` from its source file via ``importlib`` (same
pattern as ``tests/platforms/npu/quant/test_kv_quant_npu.py``) so the test
module does not ``import vllm_omni`` — the module must stay importable both
inside and outside the package, which is the whole point of the design.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _repo_root() -> Path:
    """Resolve checkout root (parent of ``vllm_omni/``), not ``tests/``."""
    here = Path(__file__).resolve()
    marker = Path("vllm_omni") / "diffusion" / "attention" / "chunking.py"
    for parent in here.parents:
        if (parent / marker).is_file():
            return parent
    msg = f"could not locate repo root (no {marker}) starting from {here}"
    raise FileNotFoundError(msg)


def _load_chunking() -> ModuleType:
    path = _repo_root() / "vllm_omni" / "diffusion" / "attention" / "chunking.py"
    if not path.is_file():
        msg = f"chunking source not found: {path}"
        raise FileNotFoundError(msg)
    name = "vllm_omni_test_chunking_standalone"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        msg = f"cannot load import spec for {path}"
        raise RuntimeError(msg)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: dataclass processing looks the module up in
    # sys.modules (KW_ONLY detection) and crashes on a missing entry.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


chunking = _load_chunking()


def _plan_calls(plan) -> list[tuple[int, int, int, int]]:
    return [(c.row0, c.row1, c.h0, c.h1) for c in plan]


class TestAttnChunkingOptions:
    def test_defaults_are_inactive(self) -> None:
        assert not chunking.AttnChunkingOptions().active

    def test_active_when_either_knob_set(self) -> None:
        assert chunking.AttnChunkingOptions(q_chunk=8).active
        assert chunking.AttnChunkingOptions(head_chunk=2).active
        # q_chunk=1 alone (the default value) is not a chunking request.
        assert not chunking.AttnChunkingOptions(q_chunk=1).active

    def test_frozen(self) -> None:
        options = chunking.AttnChunkingOptions()
        with pytest.raises(dataclasses.FrozenInstanceError):
            options.q_chunk = 4  # type: ignore[misc]


class TestBuildChunkPlan:
    def test_none_options_yields_single_call(self) -> None:
        plan = chunking.build_chunk_plan(seq_len=128, num_heads=4, options=None)
        assert _plan_calls(plan) == [(0, 128, 0, 4)]

    def test_default_options_yield_single_call(self) -> None:
        plan = chunking.build_chunk_plan(seq_len=128, num_heads=4, options=chunking.AttnChunkingOptions())
        assert _plan_calls(plan) == [(0, 128, 0, 4)]

    def test_q_chunk_even_split_aligned(self) -> None:
        plan = chunking.build_chunk_plan(
            seq_len=1024,
            num_heads=2,
            options=chunking.AttnChunkingOptions(q_chunk=8),
            row_align=128,
        )
        assert _plan_calls(plan) == [
            (0, 128, 0, 2),
            (128, 256, 0, 2),
            (256, 384, 0, 2),
            (384, 512, 0, 2),
            (512, 640, 0, 2),
            (640, 768, 0, 2),
            (768, 896, 0, 2),
            (896, 1024, 0, 2),
        ]

    def test_q_chunk_ragged_tail(self) -> None:
        # chunk = ceil(1000 / (8*128)) * 128 = 128 → last chunk keeps 104 rows.
        plan = chunking.build_chunk_plan(
            seq_len=1000,
            num_heads=1,
            options=chunking.AttnChunkingOptions(q_chunk=8),
            row_align=128,
        )
        calls = _plan_calls(plan)
        assert len(calls) == 8
        assert calls[-1] == (896, 1000, 0, 1)
        # Every boundary except the last lands on the alignment grid.
        assert all(row0 % 128 == 0 for row0, _, _, _ in calls)
        assert all(row1 % 128 == 0 for _, row1, _, _ in calls[:-1])

    def test_q_chunk_fewer_chunks_when_rows_run_out(self) -> None:
        # ceil(200 / (8*128)) * 128 = 128 → only 2 chunks fit.
        plan = chunking.build_chunk_plan(
            seq_len=200,
            num_heads=1,
            options=chunking.AttnChunkingOptions(q_chunk=8),
            row_align=128,
        )
        assert _plan_calls(plan) == [(0, 128, 0, 1), (128, 200, 0, 1)]

    def test_seq_at_or_below_align_is_single_chunk(self) -> None:
        plan = chunking.build_chunk_plan(
            seq_len=128,
            num_heads=1,
            options=chunking.AttnChunkingOptions(q_chunk=8),
            row_align=128,
        )
        assert _plan_calls(plan) == [(0, 128, 0, 1)]

    def test_align_1_allows_any_boundary(self) -> None:
        plan = chunking.build_chunk_plan(
            seq_len=10,
            num_heads=1,
            options=chunking.AttnChunkingOptions(q_chunk=3),
        )
        assert _plan_calls(plan) == [(0, 4, 0, 1), (4, 8, 0, 1), (8, 10, 0, 1)]

    def test_head_chunk_cartesian_q_major(self) -> None:
        plan = chunking.build_chunk_plan(
            seq_len=256,
            num_heads=4,
            options=chunking.AttnChunkingOptions(q_chunk=2, head_chunk=2),
            num_kv_heads=4,
            kv_len=60000,
            row_align=128,
        )
        assert _plan_calls(plan) == [
            (0, 128, 0, 2),
            (0, 128, 2, 4),
            (128, 256, 0, 2),
            (128, 256, 2, 4),
        ]

    def test_head_chunk_collapses_for_gqa(self) -> None:
        plan = chunking.build_chunk_plan(
            seq_len=256,
            num_heads=4,
            options=chunking.AttnChunkingOptions(head_chunk=2),
            num_kv_heads=2,  # GQA: KV heads cannot be split per query-head slice.
            kv_len=60000,
        )
        assert all(h1 - h0 == 4 for _, _, h0, h1 in _plan_calls(plan))

    def test_head_chunk_collapses_below_min_kv(self) -> None:
        options = chunking.AttnChunkingOptions(head_chunk=2, head_chunk_min_kv=50000)
        short = chunking.build_chunk_plan(seq_len=256, num_heads=4, options=options, num_kv_heads=4, kv_len=49999)
        assert all(h1 - h0 == 4 for _, _, h0, h1 in _plan_calls(short))
        long = chunking.build_chunk_plan(seq_len=256, num_heads=4, options=options, num_kv_heads=4, kv_len=50000)
        assert all(h1 - h0 == 2 for _, _, h0, h1 in _plan_calls(long))

    def test_min_kv_gate_leaves_q_chunking_alone(self) -> None:
        plan = chunking.build_chunk_plan(
            seq_len=256,
            num_heads=4,
            options=chunking.AttnChunkingOptions(q_chunk=2, head_chunk=2),
            num_kv_heads=4,
            kv_len=100,  # below the default gate: heads collapse, rows do not
            row_align=128,
        )
        assert _plan_calls(plan) == [(0, 128, 0, 4), (128, 256, 0, 4)]

    def test_head_chunk_larger_than_heads_is_single_head_slice(self) -> None:
        plan = chunking.build_chunk_plan(
            seq_len=64,
            num_heads=4,
            options=chunking.AttnChunkingOptions(head_chunk=8),
            num_kv_heads=4,
            kv_len=60000,
        )
        assert _plan_calls(plan) == [(0, 64, 0, 4)]

    def test_coverage_exactly_once(self) -> None:
        plan = chunking.build_chunk_plan(
            seq_len=1000,
            num_heads=6,
            options=chunking.AttnChunkingOptions(q_chunk=8, head_chunk=4),
            num_kv_heads=6,
            kv_len=60000,
            row_align=128,
        )
        by_row: dict[tuple[int, int], set[int]] = {}
        for c in plan:
            by_row.setdefault((c.row0, c.row1), set()).update(range(c.h0, c.h1))
        assert sorted(by_row) == [
            (0, 128),
            (128, 256),
            (256, 384),
            (384, 512),
            (512, 640),
            (640, 768),
            (768, 896),
            (896, 1000),
        ]
        for heads in by_row.values():
            assert heads == set(range(6))

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            (dict(seq_len=0, num_heads=1), "seq_len"),
            (dict(seq_len=8, num_heads=0), "num_heads"),
        ],
    )
    def test_invalid_shapes_raise(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            chunking.build_chunk_plan(**kwargs)

    def test_invalid_row_align_raises(self) -> None:
        with pytest.raises(ValueError, match="row_align"):
            chunking.build_chunk_plan(seq_len=8, num_heads=1, row_align=0)

    @pytest.mark.parametrize(
        "options,match",
        [
            (chunking.AttnChunkingOptions(q_chunk=0), "q_chunk"),
            (chunking.AttnChunkingOptions(head_chunk=-1), "head_chunk"),
            (chunking.AttnChunkingOptions(head_chunk=2, head_chunk_min_kv=-1), "head_chunk_min_kv"),
        ],
    )
    def test_invalid_options_raise(self, options, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            chunking.build_chunk_plan(seq_len=8, num_heads=2, options=options)


class TestRunChunked:
    """Executor contract via a stub make_call echoing the call's shape."""

    @staticmethod
    def _make_call(out_width: int = 4) -> Callable[[object], torch.Tensor]:
        def make_call(call) -> torch.Tensor:
            # Caller-layout output for one call: [1, rows, heads_slice, width].
            return torch.full((1, call.row1 - call.row0, call.h1 - call.h0, out_width), call.row0)

        return make_call

    def test_reassembles_seq_and_head_axes(self) -> None:
        plan = chunking.build_chunk_plan(
            seq_len=256,
            num_heads=4,
            options=chunking.AttnChunkingOptions(q_chunk=2, head_chunk=2),
            num_kv_heads=4,
            kv_len=60000,
            row_align=128,
        )
        out = chunking.run_chunked(plan, seq_dim=1, head_dim=2, make_call=self._make_call())
        assert out.shape == (1, 256, 4, 4)
        # Each row block is tagged with its row0: verifies both chunk order
        # (seq axis) and head merge (head axis grew to full width).
        assert torch.equal(out[0, :128, 0, 0], torch.full((128,), 0))
        assert torch.equal(out[0, 128:, 0, 0], torch.full((128,), 128))

    def test_single_call_plan_returns_make_call_result(self) -> None:
        plan = chunking.build_chunk_plan(seq_len=64, num_heads=2, options=None)
        out = chunking.run_chunked(plan, seq_dim=1, head_dim=2, make_call=self._make_call())
        assert out.shape == (1, 64, 2, 4)

    def test_q_chunk_only_reassembles_seq_axis(self) -> None:
        plan = chunking.build_chunk_plan(
            seq_len=10,
            num_heads=2,
            options=chunking.AttnChunkingOptions(q_chunk=3),
        )
        out = chunking.run_chunked(plan, seq_dim=1, head_dim=2, make_call=self._make_call())
        assert out.shape == (1, 10, 2, 4)

    def test_chunk_callback_consumes_per_q_chunk(self) -> None:
        plan = chunking.build_chunk_plan(
            seq_len=256,
            num_heads=4,
            options=chunking.AttnChunkingOptions(q_chunk=2, head_chunk=2),
            num_kv_heads=4,
            kv_len=60000,
            row_align=128,
        )
        chunks: list[tuple[torch.Tensor, tuple[int, int]]] = []

        def cb(out_chunk, call) -> None:
            chunks.append((out_chunk, (call.row0, call.row1)))

        result = chunking.run_chunked(plan, seq_dim=1, head_dim=2, make_call=self._make_call(), chunk_callback=cb)
        assert result is None
        # One head-merged chunk per q chunk, full head width each.
        assert [tuple(t.shape) for t, _ in chunks] == [(1, 128, 4, 4)] * 2
        assert [rows for _, rows in chunks] == [(0, 128), (128, 256)]

    def test_empty_plan_raises(self) -> None:
        with pytest.raises(ValueError, match="plan must not be empty"):
            chunking.run_chunked([], seq_dim=1, head_dim=2, make_call=self._make_call())
