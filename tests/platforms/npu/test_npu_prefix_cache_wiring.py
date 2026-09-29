# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""NPU prefix-cache wiring: save_outputs -> sid -> materialize on CPU.

The NPU runner consumes the prefix cache in eager mode (no CUDA streams) and
hands the step id across execute_model()/sample_tokens() as the LAST field of
its positionally-packed ``ExecuteModelState``. Neither path runs in CUDA CI,
so a hit-read change can silently break them (`vllm_ascend` imports keep the
runner modules out of reach of a plain unit test).

Two layers of protection, both pure CPU:

1. AST characterization of the ``ExecuteModelState`` field order in the NPU
   and GPU runners — the pack/unpack sites pass fields positionally, so a
   reorder or drop of ``prefix_cache_step_id`` corrupts the sid without any
   import-time error.
2. The eager-mode facade contract the NPU runner relies on: controller
   auto-selects eager when CUDA is unavailable, and save -> materialize
   returns byte-identical rows for both the miss and the hit-merge paths.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.core.prefix_cache.interface import PrefixCacheConfig
from vllm_omni.core.prefix_cache.manager import OmniPrefixCacheManager

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NPU_RUNNER = _REPO_ROOT / "vllm_omni/platforms/npu/worker/npu_ar_model_runner.py"
_GPU_RUNNER = _REPO_ROOT / "vllm_omni/worker/gpu_ar_model_runner.py"

NUM_BLOCKS = 16
BLOCK_SIZE = 4
HIDDEN = 8
DTYPE = torch.float32


def _named_tuple_fields(path: Path, class_name: str) -> list[str]:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return [stmt.target.id for stmt in node.body if isinstance(stmt, ast.AnnAssign)]
    raise AssertionError(f"{class_name} not found in {path}")


def test_execute_model_state_keeps_prefix_cache_sid_last():
    """Both runners pack ExecuteModelState positionally: the sid must stay
    the LAST field, and the NPU field order is pinned exactly — a reorder
    passes import and every CUDA test, then ships the wrong value as the
    sid on hardware."""
    npu_fields = _named_tuple_fields(_NPU_RUNNER, "ExecuteModelState")
    assert npu_fields == [
        "scheduler_output",
        "logits",
        "spec_decode_metadata",
        "spec_decode_common_attn_metadata",
        "hidden_states",
        "sample_hidden_states",
        "aux_hidden_states",
        "attn_metadata",
        "positions",
        "ec_connector_output",
        "cudagraph_stats",
        "batch_desc",
        "multimodal_outputs",
        "prefix_cache_step_id",
    ]
    gpu_fields = _named_tuple_fields(_GPU_RUNNER, "ExecuteModelState")
    assert gpu_fields[-1] == "prefix_cache_step_id"


class _FakeView:
    """Minimal group-view double (mirrors tests/core/test_prefix_cache)."""

    def __init__(self):
        self.block_size = BLOCK_SIZE
        self.num_blocks = NUM_BLOCKS
        self.req_blocks: dict[str, list[int]] = {}
        self.order: list[str] = []
        self.computed: dict[str, int] = {}

    def slots_for(self, req_id, token_start, token_end):
        blocks = self.req_blocks[req_id]
        return torch.tensor(
            [blocks[p // BLOCK_SIZE] * BLOCK_SIZE + p % BLOCK_SIZE for p in range(token_start, token_end)],
            dtype=torch.long,
        )

    def batch_req_ids(self) -> list[str]:
        return list(self.order)

    def step_slots_cpu(self, req_ids, num_scheduled) -> torch.Tensor:
        parts = []
        for r in req_ids:
            n = int(num_scheduled.get(r, 0))
            if n > 0:
                start = self.computed.get(r, 0)
                parts.append(self.slots_for(r, start, start + n))
        return torch.cat(parts) if parts else torch.empty((0,), dtype=torch.long)


def _run_step(mgr, view, req_id, blocks, start_pos, sched, *, hit=0, finished=()):
    """One NPU-shaped step: new_step_starts -> save_outputs -> sid."""
    view.order = [req_id]
    view.req_blocks[req_id] = blocks
    view.computed[req_id] = start_pos
    slots = view.slots_for(req_id, start_pos, start_pos + sched)
    hidden = slots.to(DTYPE).unsqueeze(1).expand(sched, HIDDEN).clone()
    sched_out = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id=req_id, num_computed_tokens=hit, block_ids=[list(blocks)]),
        ],
        finished_req_ids=set(finished),
        num_scheduled_tokens={req_id: sched},
    )
    mgr.new_step_starts(sched_out)
    return mgr.save_outputs(hidden, {}, num_tokens_unpadded=sched, num_tokens_padded=sched)


def _expected_rows(slots: torch.Tensor) -> torch.Tensor:
    return slots.to(DTYPE).unsqueeze(1).expand(slots.numel(), HIDDEN)


def _make_npu_mode_manager(monkeypatch) -> tuple[OmniPrefixCacheManager, _FakeView]:
    # The NPU condition: no CUDA -> the controller must auto-select eager
    # (submit() completes copy+scatter synchronously), exactly how
    # npu_model_runner builds the manager with the default eager=None.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    view = _FakeView()
    config = PrefixCacheConfig(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)
    return OmniPrefixCacheManager(config, view), view


def test_npu_mode_auto_selects_eager_and_roundtrips(monkeypatch):
    mgr, view = _make_npu_mode_manager(monkeypatch)
    assert mgr._controller._eager is True
    sid = _run_step(mgr, view, "a", [0, 1], 0, 8)
    outs = mgr.materialize(sid, ["a"])
    assert torch.equal(outs.hidden_states["a"], _expected_rows(view.slots_for("a", 0, 8)))


def test_npu_mode_hit_merges_cached_prefix(monkeypatch):
    """The hit-read path in eager mode: consumer's merged output must be
    [cached prefix rows + this step's rows], byte-identical."""
    mgr, view = _make_npu_mode_manager(monkeypatch)
    s1 = _run_step(mgr, view, "a", [0, 1], 0, 8)
    mgr.materialize(s1, ["a"])
    s2 = _run_step(mgr, view, "b", [0, 1, 2], 8, 4, hit=8, finished=["a"])
    merged = mgr.materialize(s2, ["b"]).hidden_states["b"]
    assert merged.shape == (12, HIDDEN)
    assert torch.equal(merged[:8], _expected_rows(view.slots_for("b", 0, 8)))
    assert torch.equal(merged[8:], _expected_rows(view.slots_for("b", 8, 12)))
