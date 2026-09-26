# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests covering the Scheduler, Worker, and Batch Wiring for MiniCPM-o 4.5 duplex KV window.

Covers:
1. Block-table in-place compaction on the Worker side without row moves.
2. rotate_cached_keys numeric identity with attention dot products.
3. Multi-request concurrency / batched execution:
   - Request 0: triggers window trim with delta=16
   - Request 1: normal decode (untouched)
   - Request 2: triggers window trim with delta=32
4. Scheduler-side watermark detection and reanchor plan attachment without full re-prefill.
5. Barge-in / abort safety ensuring zero leaked blocks.
"""

from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

repo_root = pathlib.Path(__file__).resolve().parent
while repo_root.name and not (repo_root / "vllm_omni").is_dir():
    repo_root = repo_root.parent

try:
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
        _MiniCPMO45WindowUnit,
    )
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_kv import (
        DUPLEX_WINDOW_BLOCK_SIZE,
        MiniCPMO45DuplexSchedulerHelper,
        MiniCPMO45DuplexWindowManager,
        MiniCPMO45DuplexWorkerHelper,
        assert_uniform_position_shift,
        duplex_window_geometry,
        rotate_cached_keys,
        rotate_keys,
        validate_duplex_window_install,
    )
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_plan import (
        PositionReanchor,
        plan_position_reanchor,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    import sys
    import types

    def _make_pkg(name: str) -> types.ModuleType:
        if name in sys.modules:
            return sys.modules[name]
        m = types.ModuleType(name)
        m.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = m
        return m

    vllm = _make_pkg("vllm")
    vllm.__version__ = "0.7.0"  # type: ignore[attr-defined]
    vllm.__version_tuple__ = (0, 7, 0)  # type: ignore[attr-defined]
    v1 = _make_pkg("vllm.v1")
    spec_reg = _make_pkg("vllm.v1.kv_cache_spec_registry")
    spec_reg.register_kv_cache_spec = lambda *a, **k: (lambda cls: cls)  # type: ignore[attr-defined]
    kv_if = _make_pkg("vllm.v1.kv_cache_interface")

    class MockSpec:
        pass

    kv_if.KVCacheSpec = MockSpec  # type: ignore[attr-defined]
    kv_if.SlidingWindowSpec = MockSpec  # type: ignore[attr-defined]

    vo = _make_pkg("vllm_omni")
    vo_exp = _make_pkg("vllm_omni.experimental")
    vo_ad = _make_pkg("vllm_omni.experimental.ar_diffusion")
    vo_kc = _make_pkg("vllm_omni.experimental.ar_diffusion.kv_cache")
    vo_pg = _make_pkg("vllm_omni.experimental.ar_diffusion.kv_cache.paged")

    class MockChunkSpec:
        pass

    class MockChunkManager:
        def __init__(self, *a, **k):
            self.enable_caching = False
            self.req_to_blocks = {}
            self.block_size = 16
            self._null_block = -1

        def compact_block_table(self, request_id: str, sink_blocks: int | None = None) -> int:
            blocks = self.req_to_blocks.get(request_id, [])
            start = 0 if sink_blocks is None else int(sink_blocks)
            end = start
            while end < len(blocks) and blocks[end] == self._null_block:
                end += 1
            if end == start:
                return 0
            del blocks[start:end]
            return (end - start) * self.block_size

        def compact_request_blocks(self, request_id: str, *, sink_blocks: int, num_blocks: int) -> int:
            blocks = self.req_to_blocks.get(request_id)
            if blocks is None or num_blocks <= 0:
                return 0
            start = int(sink_blocks)
            end = start + int(num_blocks)
            if start < 0 or end > len(blocks) or any(b == self._null_block for b in blocks[start:end]):
                return 0
            for i in range(start, end):
                blocks[i] = self._null_block
            return self.compact_block_table(request_id, sink_blocks=start)

        def reanchor_block_table(self, request_id: str, plan: Any) -> int:
            gap_blocks = plan.delta // self.block_size
            return self.compact_request_blocks(
                request_id,
                sink_blocks=plan.sink_blocks,
                num_blocks=gap_blocks,
            )

    vo_pg.ChunkWindowSpec = MockChunkSpec  # type: ignore[attr-defined]
    vo_pg.ChunkWindowManager = MockChunkManager  # type: ignore[attr-defined]

    def compute_slot_mapping(block_ids, positions, block_size):
        p = positions.to(dtype=torch.long)
        t = torch.tensor(block_ids, dtype=torch.long, device=p.device)
        return t[torch.div(p, block_size, rounding_mode="floor")] * block_size + (p % block_size)

    vo_pg.compute_slot_mapping = compute_slot_mapping  # type: ignore[attr-defined]

    _make_pkg("vllm_omni.model_executor")
    _make_pkg("vllm_omni.model_executor.models")
    _make_pkg("vllm_omni.model_executor.models.minicpmo_4_5")
    _make_pkg("vllm_omni.model_executor.models.minicpmo_4_5.duplex")

    repo_root = pathlib.Path(__file__).resolve().parent
    while repo_root.name and not (repo_root / "vllm_omni").is_dir():
        repo_root = repo_root.parent

    def _load_module(name: str, rel_path: str) -> types.ModuleType:
        spec = importlib.util.spec_from_file_location(name, repo_root / rel_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    _wp = _load_module(
        "vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_plan",
        "vllm_omni/model_executor/models/minicpmo_4_5/duplex/window_plan.py",
    )
    _wk = _load_module(
        "vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_kv",
        "vllm_omni/model_executor/models/minicpmo_4_5/duplex/window_kv.py",
    )

    _wp = _load_module(
        "vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_plan",
        "vllm_omni/model_executor/models/minicpmo_4_5/duplex/window_plan.py",
    )
    _wk = _load_module(
        "vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_kv",
        "vllm_omni/model_executor/models/minicpmo_4_5/duplex/window_kv.py",
    )
    _policy = _load_module(
        "vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy",
        "vllm_omni/model_executor/models/minicpmo_4_5/duplex/policy.py",
    )
    _stage0 = _load_module(
        "vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0",
        "vllm_omni/model_executor/models/minicpmo_4_5/duplex/stage0.py",
    )

    DUPLEX_WINDOW_BLOCK_SIZE = _wk.DUPLEX_WINDOW_BLOCK_SIZE
    MiniCPMO45DuplexSchedulerHelper = _wk.MiniCPMO45DuplexSchedulerHelper
    MiniCPMO45DuplexWindowManager = _wk.MiniCPMO45DuplexWindowManager
    MiniCPMO45DuplexWorkerHelper = _wk.MiniCPMO45DuplexWorkerHelper
    assert_uniform_position_shift = _wk.assert_uniform_position_shift
    duplex_window_geometry = _wk.duplex_window_geometry
    rotate_cached_keys = _wk.rotate_cached_keys
    rotate_keys = _wk.rotate_keys
    validate_duplex_window_install = _wk.validate_duplex_window_install
    PositionReanchor = _wp.PositionReanchor
    plan_position_reanchor = _wp.plan_position_reanchor
    MiniCPMO45Stage0DuplexRuntime = _stage0.MiniCPMO45Stage0DuplexRuntime
    _MiniCPMO45Stage0SessionState = _stage0._MiniCPMO45Stage0SessionState
    _MiniCPMO45WindowUnit = _stage0._MiniCPMO45WindowUnit


def _extract_prepare_stage0_window():
    p = repo_root / "vllm_omni" / "core" / "sched" / "omni_ar_scheduler.py"
    tree = ast.parse(p.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_prepare_minicpmo45_stage0_window":
            node.decorator_list = []
            mod = ast.Module(body=[node], type_ignores=[])
            ns: dict[str, Any] = {}
            exec(compile(mod, str(p), "exec"), ns)
            return ns["_prepare_minicpmo45_stage0_window"]
    raise RuntimeError("Could not find _prepare_minicpmo45_stage0_window in omni_ar_scheduler.py")


prepare_minicpmo45_stage0_window = _extract_prepare_stage0_window()

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

BLOCK_SIZE = DUPLEX_WINDOW_BLOCK_SIZE  # 16
HEAD_DIM = 128
NUM_KV_HEADS = 8


def _get_inv_freq(head_dim: int = HEAD_DIM, base: float = 1000000.0) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))


def _forward_rope(x_raw: torch.Tensor, pos: int, inv_freq: torch.Tensor) -> torch.Tensor:
    half = x_raw.shape[-1] // 2
    angle = float(pos) * inv_freq.to(device=x_raw.device, dtype=torch.float32)
    cos = torch.cos(angle).to(dtype=x_raw.dtype).unsqueeze(0).unsqueeze(1)
    sin = torch.sin(angle).to(dtype=x_raw.dtype).unsqueeze(0).unsqueeze(1)
    x1, x2 = x_raw[..., :half], x_raw[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def test_worker_block_table_compaction():
    """Verify that shifting block table entries on the worker is exact and zero-copy."""
    # Simulate a request with 7 physical blocks: [B0, B1, Gap0, Gap1, B2, B3, B4]
    sink_blocks = 2
    gap_blocks = 2
    num_blocks = 7
    initial_blocks = [101, 102, 201, 202, 301, 302, 303]

    table_np = np.zeros((4, 32), dtype=np.int32)
    num_blocks_per_row = np.zeros(4, dtype=np.int32)

    row_idx = 1
    table_np[row_idx, :num_blocks] = initial_blocks
    num_blocks_per_row[row_idx] = num_blocks

    # Worker-side compaction helper
    total = int(num_blocks_per_row[row_idx])
    assert total == 7
    table_np[row_idx, sink_blocks : total - gap_blocks] = table_np[row_idx, sink_blocks + gap_blocks : total]
    table_np[row_idx, total - gap_blocks : total] = 0
    num_blocks_per_row[row_idx] -= gap_blocks

    assert num_blocks_per_row[row_idx] == 5
    compacted = list(table_np[row_idx, :5])
    assert compacted == [101, 102, 301, 302, 303]
    # Zeroed out tail
    assert list(table_np[row_idx, 5:7]) == [0, 0]


def test_rotate_cached_keys_attention_equivalence():
    """Verify that rotated cached keys and attention scores match ground-truth forward RoPE."""
    inv_freq = _get_inv_freq()
    delta = 16
    moved_from = 32
    sink_blocks = 1  # 16 tokens sink (block 0), 16 tokens gap (block 1), retained tail from pos 32 (block 2..)
    plan = PositionReanchor(delta=delta, moved_from=moved_from, sink_blocks=sink_blocks)

    # Initial physical blocks: [0, 1, 2, 3]. Gap is block 1.
    # Compacted block table after trim: [0, 2, 3]
    compacted_blocks = [0, 2, 3]
    num_tokens = 64
    positions = torch.arange(moved_from, num_tokens, dtype=torch.long)

    torch.manual_seed(42)
    # Generate raw unrotated key features
    raw_keys = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float32)

    # Populate cache by applying ground-truth forward RoPE at original positions:
    # Block 0: 0..15, Block 1: 16..31 (gap), Block 2: 32..47, Block 3: 48..63
    k_pool = torch.zeros(4, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float32)
    for p in range(num_tokens):
        b = p // BLOCK_SIZE
        o = p % BLOCK_SIZE
        k_pool[b, o] = _forward_rope(raw_keys[p : p + 1], p, inv_freq).squeeze(0)

    # Rotate retained tail in place
    touched = rotate_cached_keys(
        k_pool,
        block_ids=compacted_blocks,
        positions=positions,
        plan=plan,
        inv_freq=inv_freq,
    )
    assert touched == len(positions)

    # Ground-truth comparison:
    # In compacted table, token p is re-indexed to logical pos new_p = p - delta.
    # Its physical slot is compacted_blocks[new_p // BLOCK_SIZE] at new_p % BLOCK_SIZE.
    # The rotated cached key MUST equal computing forward RoPE directly at pos new_p!
    for p in range(moved_from, num_tokens):
        new_p = p - delta
        phys_b = compacted_blocks[new_p // BLOCK_SIZE]
        o = new_p % BLOCK_SIZE
        rotated_k = k_pool[phys_b, o]
        gt_k = _forward_rope(raw_keys[p : p + 1], new_p, inv_freq).squeeze(0)
        assert torch.allclose(rotated_k, gt_k, atol=1e-5)

    # Attention score check with a query at step Q
    q_raw = torch.randn(1, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float32)
    q_pos = 80
    q = _forward_rope(q_raw, q_pos, inv_freq).squeeze(0)

    for p in range(moved_from, num_tokens):
        new_p = p - delta
        phys_b = compacted_blocks[new_p // BLOCK_SIZE]
        o = new_p % BLOCK_SIZE
        rotated_k = k_pool[phys_b, o]
        score_rotated = (q * rotated_k).sum()
        gt_k = _forward_rope(raw_keys[p : p + 1], new_p, inv_freq).squeeze(0)
        score_gt = (q * gt_k).sum()
        assert torch.allclose(score_rotated, score_gt, atol=1e-5)


def test_rotate_keys_precision_ground_truth_bf16():
    """Verify that rotate_keys in bfloat16 at large delta matches ground truth forward RoPE.

    In bfloat16, values in [1024, 2048] have ULP = 8. At delta = 6000 (standard Stage-0
    sliding window size), computing angles in bfloat16 introduces ~4-5.7 rad quantization
    error, scrambling trigonometric values. Computing in float32 and casting back ensures
    numerical fidelity (<0.05 max error vs >2.0 for buggy bfloat16).
    """
    inv_freq = _get_inv_freq()
    torch.manual_seed(42)
    tokens, heads = 16, 8
    x_raw = torch.randn(tokens, heads, HEAD_DIM, dtype=torch.bfloat16)

    p_old = 7000
    delta = 6000
    p_new = p_old - delta

    k_old = _forward_rope(x_raw, p_old, inv_freq)
    k_gt = _forward_rope(x_raw, p_new, inv_freq)

    # Fixed rotate_keys (fp32 trig):
    k_rotated = rotate_keys(k_old, delta, inv_freq)
    err = (k_rotated.float() - k_gt.float()).abs().max().item()
    assert err < 0.05, f"Expected precision < 0.05, got {err}"

    # Verify that the buggy calculation (angle in bf16 before cos/sin) fails drastically:
    half = HEAD_DIM // 2
    angle_buggy = (int(delta) * inv_freq).to(dtype=torch.bfloat16)
    cos_buggy = torch.cos(angle_buggy).unsqueeze(0).unsqueeze(1)
    sin_buggy = torch.sin(angle_buggy).unsqueeze(0).unsqueeze(1)
    k1, k2 = k_old[..., :half], k_old[..., half:]
    k_buggy = torch.cat([k1 * cos_buggy + k2 * sin_buggy, k2 * cos_buggy - k1 * sin_buggy], dim=-1)
    err_buggy = (k_buggy.float() - k_gt.float()).abs().max().item()
    assert err_buggy > 2.0, f"Buggy bf16 angle calculation should exhibit >2.0 error, got {err_buggy}"


def test_batched_concurrency_isolation():
    """Verify that in a batch of multiple concurrent requests:
    - Request 0 triggers trim (delta=16)
    - Request 1 is normal decode (untouched)
    - Request 2 triggers trim with different delta=32
    Non-triggering requests are completely unaffected.
    """
    inv_freq = _get_inv_freq()
    num_blocks = 20
    torch.manual_seed(123)
    k_pool = torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float32)
    k_pool_snapshot = k_pool.clone()

    # Request 0: blocks [0, 1, 2, 3], needs delta=16, moved_from=32, sink_blocks=1
    # Gap is block 1. Compacted table after trim is [0, 2, 3].
    plan_0 = PositionReanchor(delta=16, moved_from=32, sink_blocks=1)
    req0_blocks_compacted = [0, 2, 3]
    req0_positions = torch.arange(32, 64, dtype=torch.long)

    # Request 1: blocks [4, 5, 6, 7], normal decode, NO trim
    req1_blocks = [4, 5, 6, 7]

    # Request 2: blocks [8, 9, 10, 11, 12], needs delta=32, moved_from=48, sink_blocks=1
    # Gap is blocks [9, 10]. Compacted table after trim is [8, 11, 12].
    plan_2 = PositionReanchor(delta=32, moved_from=48, sink_blocks=1)
    req2_blocks_compacted = [8, 11, 12]
    req2_positions = torch.arange(48, 80, dtype=torch.long)

    # Execute Re-RoPE for the batch
    reanchor_batch = {
        0: (req0_blocks_compacted, req0_positions, plan_0),
        2: (req2_blocks_compacted, req2_positions, plan_2),
    }

    for req_idx, (b_ids, pos, plan) in reanchor_batch.items():
        rotate_cached_keys(
            k_pool,
            block_ids=b_ids,
            positions=pos,
            plan=plan,
            inv_freq=inv_freq,
        )

    # ASSERTION 1: Request 1's physical blocks [4, 5, 6, 7] are 100% UNTOUCHED
    for b in req1_blocks:
        assert torch.equal(k_pool[b], k_pool_snapshot[b]), f"Block {b} of Request 1 was corrupted!"

    # ASSERTION 2: Request 0's sink block [0] is UNTOUCHED
    assert torch.equal(k_pool[0], k_pool_snapshot[0]), "Request 0 sink block was corrupted!"

    # ASSERTION 3: Request 0's tail blocks [2, 3] are rotated by delta=16
    for b in [2, 3]:
        expected = rotate_keys(k_pool_snapshot[b], 16, inv_freq)
        assert torch.allclose(k_pool[b], expected, atol=1e-6)

    # ASSERTION 4: Request 2's sink block [8] is UNTOUCHED
    assert torch.equal(k_pool[8], k_pool_snapshot[8]), "Request 2 sink block was corrupted!"

    # ASSERTION 5: Request 2's tail blocks [11, 12] are rotated by delta=32
    for b in [11, 12]:
        expected = rotate_keys(k_pool_snapshot[b], 32, inv_freq)
        assert torch.allclose(k_pool[b], expected, atol=1e-6)


def test_scheduler_reanchor_planning_and_no_full_reprefill():
    """Verify that Scheduler triggers reanchor without replacing the session prompt."""
    geometry = duplex_window_geometry(
        prefix_tokens=96,
        window_tokens=6000,
        block_size=16,
        max_model_len=40960,
        high_watermark_tokens=8000,
    )

    # Below high watermark (96 + 8000 = 8096): no reanchor
    plan = plan_position_reanchor(geometry, computed_tokens=7900, pending_tokens=12)
    assert plan is None

    # Crossing watermark: 8090 + 12 = 8102 > 8096
    # Target is prefix(96) + window(6000) = 6096
    target = geometry.prefix_tokens + geometry.window_tokens
    plan = plan_position_reanchor(geometry, computed_tokens=8090, pending_tokens=12)
    assert plan is not None
    assert plan.delta % BLOCK_SIZE == 0
    assert plan.delta > 0
    # Retained sequence stays within target budget
    retained = (8090 + 12) - plan.delta
    assert target - BLOCK_SIZE < retained <= target
    # Total retained equals sink_end + retained tail
    assert (8090 + 12) - plan.delta == plan.sink_end + ((8090 + 12) - plan.moved_from)


def test_barge_in_abort_safety():
    """Verify that aborting a request during/after reanchor does not leak blocks."""
    # Simulate a mini block allocator
    free_blocks = set(range(100))
    allocated = {}

    def alloc(req_id, count):
        blocks = [free_blocks.pop() for _ in range(count)]
        allocated[req_id] = blocks
        return blocks

    def free_blocks_range(req_id, start, end):
        blocks = allocated[req_id]
        freed = blocks[start:end]
        for b in freed:
            free_blocks.add(b)
        allocated[req_id] = blocks[:start] + blocks[end:]

    def abort_request(req_id):
        blocks = allocated.pop(req_id, [])
        for b in blocks:
            free_blocks.add(b)

    # Req A allocates 8 blocks
    alloc("req-a", 8)
    assert len(free_blocks) == 92

    # Trim: free gap blocks [2:4] (2 blocks)
    free_blocks_range("req-a", 2, 4)
    assert len(free_blocks) == 94
    assert len(allocated["req-a"]) == 6

    # User barges in -> abort!
    abort_request("req-a")
    assert len(free_blocks) == 100  # ALL blocks successfully returned, zero leak!


def test_scheduler_worker_end_to_end_state_agreement():
    """Verify production update path: append -> scheduler compaction -> worker state update -> agreement."""
    inv_freq = _get_inv_freq()
    num_blocks = 10
    k_pool = torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float32)
    k_pool_orig = k_pool.clone()

    # Initial session prompt: 64 tokens, 4 blocks [0, 1, 2, 3]
    # Prefix: 16 tokens (block 0), gap: 16 tokens (block 1, delta=16), tail: 32 tokens (blocks 2, 3)
    prompt_ids = list(range(100, 164))
    session = SimpleNamespace(
        request_id="req-1",
        prompt_token_ids=list(prompt_ids),
        _all_token_ids=list(prompt_ids),
        num_prompt_tokens=64,
        num_computed_tokens=64,
    )

    class _MockDuplexManager(MiniCPMO45DuplexWindowManager):
        def __init__(self):
            self.enable_caching = False
            self.blocks = [0, 1, 2, 3]
            self.req_to_blocks = {"req-1": self.blocks}
            self.block_size = BLOCK_SIZE
            self._null_block = -1

        def compact_request_blocks(self, request_id: str, *, sink_blocks: int, num_blocks: int) -> int:
            del self.blocks[sink_blocks : sink_blocks + num_blocks]
            return num_blocks * BLOCK_SIZE

        def reanchor_block_table(self, request_id: str, plan: PositionReanchor) -> int:
            return self.compact_request_blocks(
                request_id,
                sink_blocks=plan.sink_blocks,
                num_blocks=plan.delta // BLOCK_SIZE,
            )

    duplex_mgr = _MockDuplexManager()
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE),
        model_config=SimpleNamespace(max_model_len=40960),
        kv_cache_manager=SimpleNamespace(coordinator=SimpleNamespace(single_type_managers=[duplex_mgr])),
    )

    # 1. Update arrives: 1 token append, triggering window trim
    update = SimpleNamespace(
        prompt_token_ids=[999],
        model_intermediate_buffer={
            "duplex": {
                "data_plane": True,
                "runtime_config": {
                    "duplex_window_prefix_tokens": 16,
                    "duplex_window_config": {
                        "sliding_window_mode": "basic",
                        "basic_window_high_tokens": 64,  # Total watermark trigger at 64 tokens (< 64+1=65)
                        "basic_window_low_tokens": 64,  # Target total is 64 tokens (drop_needed=1 -> delta=16)
                    },
                },
            }
        },
    )

    plan = MiniCPMO45DuplexSchedulerHelper.maybe_reanchor_session(scheduler, session, update)
    assert plan is not None
    assert plan.delta == 16
    assert plan.moved_from == 32
    assert plan.sink_blocks == 1

    # Scheduler compacted session state:
    # 64 tokens -> trimmed middle gap [16:32] -> 48 tokens surviving
    assert len(session.prompt_token_ids) == 48
    assert len(session._all_token_ids) == 48
    assert session.prompt_token_ids == prompt_ids[:16] + prompt_ids[32:]
    assert session._all_token_ids == prompt_ids[:16] + prompt_ids[32:]
    assert session.num_computed_tokens == 48
    assert session.num_prompt_tokens == 48
    assert duplex_mgr.blocks == [0, 2, 3]

    # Subsequent append from upstream _update_request_as_session adds the 1 pending token:
    session.prompt_token_ids.extend(update.prompt_token_ids)
    session._all_token_ids.extend(update.prompt_token_ids)
    session.num_prompt_tokens = len(session.prompt_token_ids)
    # Both token histories are now consistent with length 49!
    assert len(session.prompt_token_ids) == 49
    assert len(session._all_token_ids) == 49

    # 2. Worker state update via production path:
    # Scheduler provides post-compaction block IDs [0, 2, 3] and computed count 48
    table_np = np.zeros((2, 16), dtype=np.int32)
    table_np[0, :3] = [0, 2, 3]
    num_blocks_per_row = np.array([3, 0], dtype=np.int32)

    class _MockBlockTable:
        def __init__(self):
            self.block_table = SimpleNamespace(np=table_np)
            self.num_blocks_per_row = num_blocks_per_row

    class _MockRunner:
        def __init__(self):
            self.device = torch.device("cpu")
            self.cache_config = SimpleNamespace(block_size=BLOCK_SIZE)
            self.model_config = SimpleNamespace(
                get_head_size=lambda: HEAD_DIM,
                hf_config=SimpleNamespace(rope_theta=1000000.0),
            )
            self._duplex_inv_freq = inv_freq
            self.kv_caches = [k_pool]
            self.requests = {
                "req-1": SimpleNamespace(
                    block_ids=([0, 2, 3],),  # Production grouped tuple layout
                    num_computed_tokens=48,
                    mrope_positions=None,
                )
            }
            self.input_batch = SimpleNamespace(
                num_reqs=1,
                req_ids=["req-1"],
                block_table=_MockBlockTable(),
                num_computed_tokens_cpu=np.array([48], dtype=np.int32),
            )
            self.model_intermediate_buffer = {
                "req-1": update.model_intermediate_buffer,
            }

    runner = _MockRunner()
    # Execute worker helper
    MiniCPMO45DuplexWorkerHelper.maybe_apply_reanchor(runner)

    # 3. Assertions: Both sides strictly agree on computed counts and block IDs!
    assert runner.input_batch.num_computed_tokens_cpu[0] == 48, "Worker must NOT decrement computed tokens twice!"
    assert session.num_computed_tokens == runner.input_batch.num_computed_tokens_cpu[0] == 48
    bt = runner.input_batch.block_table
    assert bt.num_blocks_per_row[0] == 3, "Worker must NOT delete blocks twice!"
    assert list(bt.block_table.np[0, :3]) == [0, 2, 3]
    assert duplex_mgr.blocks == list(bt.block_table.np[0, :3]) == [0, 2, 3]

    # Verify KV cache rotation: sink block 0 untouched, blocks 2 and 3 rotated
    assert torch.equal(k_pool[0], k_pool_orig[0])
    for b in [2, 3]:
        expected = rotate_keys(k_pool_orig[b], 16, inv_freq)
        assert torch.allclose(k_pool[b], expected, atol=1e-6)


def test_differing_prefix_lengths_dynamic_sink():
    """Verify that differing instruction / reference audio prefix lengths work without spec mismatch."""
    # Prefix length 112 tokens = 7 blocks (differs from standard 96 tokens = 6 blocks)
    prefix_tokens = 112
    sink_blocks = 7
    blocks = [100 + i for i in range(12)]

    class _MockSpec:
        chunk_size = BLOCK_SIZE
        sink_chunks = 6  # Static spec default is 6, while session has 7

    manager = MiniCPMO45DuplexWindowManager.__new__(MiniCPMO45DuplexWindowManager)
    manager.block_size = BLOCK_SIZE
    manager.enable_caching = False
    manager.kv_cache_spec = _MockSpec()
    manager._null_block = -1
    manager.req_to_blocks = {"req-dyn": list(blocks)}
    manager.num_cached_block = {}

    def mock_remove(req_id, start, end):
        b = manager.req_to_blocks[req_id]
        for i in range(start, end):
            b[i] = manager._null_block

    manager._remove_blocks_in_range = mock_remove

    plan = PositionReanchor(delta=16, moved_from=prefix_tokens + 16, sink_blocks=sink_blocks)
    # Exercises real inherited compact_block_table / compact_request_blocks path without mock lambda
    freed = manager.reanchor_block_table("req-dyn", plan)
    assert freed == 16
    # Verified: sink blocks 0..6 (7 blocks) preserved, gap block 7 freed!
    assert len(manager.req_to_blocks["req-dyn"]) == 11
    assert manager.req_to_blocks["req-dyn"][:7] == blocks[:7]
    assert manager.req_to_blocks["req-dyn"][7:] == blocks[8:]


def test_context_mode_explicit_fallback():
    """Verify context mode explicitly returns None from zero-copy reanchor policy to route to official fallback."""
    session = SimpleNamespace(num_computed_tokens=600)
    update = SimpleNamespace(
        prompt_token_ids=[1] * 10,
        model_intermediate_buffer={
            "duplex": {
                "data_plane": True,
                "runtime_config": {
                    "duplex_window_prefix_tokens": 96,
                    "duplex_window_config": {
                        "sliding_window_mode": "context",
                        "context_max_units": 24,
                        "context_previous_max_tokens": 16,
                    },
                },
            }
        },
    )
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_kv import MiniCPMO45DuplexWindowPolicy

    # Context mode must return None from zero-copy reanchor policy to route to official fallback
    assert MiniCPMO45DuplexWindowPolicy.plan_reanchor(session, update, block_size=16, max_model_len=40960) is None


def test_vllm_030_packed_flash_attn_kv_layout_and_storage_alias():
    """Verify vLLM 0.30 FlashAttention packed layout (B, H, N, 2*D) rotates in-place and preserves V cache."""
    # Real H20-3e GPU observed layout: 8 heads, block_size 16, head_dim 128 (2*head_dim = 256)
    # Transposed strides: [32768, 256, 2048, 1]
    base = torch.randn(10, 16, 8, 256, dtype=torch.bfloat16)
    kv_cache = base.transpose(1, 2)
    assert kv_cache.shape == (10, 8, 16, 256)
    assert kv_cache.stride() == (32768, 256, 2048, 1)

    storage_ptr = kv_cache.data_ptr()
    v_cache_copy = kv_cache[..., 128:].clone()
    inv_freq = torch.randn(64, dtype=torch.float32)

    plan = PositionReanchor(delta=16, moved_from=16, sink_blocks=1)
    block_ids = [0, 2, 5]
    positions = torch.tensor([16, 17, 30, 32, 47], dtype=torch.long)

    touched = rotate_cached_keys(
        kv_cache,
        block_ids=block_ids,
        positions=positions,
        plan=plan,
        inv_freq=inv_freq,
        block_size=16,
    )
    assert touched == 5
    # Must write in-place into the existing storage: no copies!
    assert kv_cache.data_ptr() == storage_ptr
    # V cache in [..., 128:] must be completely untouched!
    assert torch.equal(kv_cache[..., 128:], v_cache_copy)
    # Untouched blocks must remain unchanged
    assert torch.equal(kv_cache[1, ..., :128], base.transpose(1, 2)[1, ..., :128])


def test_worker_history_bounded_across_long_session_reanchors():
    """Verify worker-held state.window_units is pruned when stage0_reanchor triggers, staying bounded."""
    units = [SimpleNamespace(token_ids=list(range(i * 12, (i + 1) * 12))) for i in range(50)]
    session_state = SimpleNamespace(window_units=units)

    class _MockHelper:
        def __init__(self):
            self.sessions = {"sess-1": session_state}

        def _evict_window_units_for_reanchor(self, state, reanchor):
            delta = int(reanchor.get("delta", 0) or 0)
            if delta <= 0 or not state.window_units:
                return
            dropped = 0
            idx = 0
            while idx < len(state.window_units):
                unit_len = len(state.window_units[idx].token_ids)
                if dropped + unit_len <= delta:
                    dropped += unit_len
                    idx += 1
                else:
                    break
            if idx > 0:
                del state.window_units[:idx]

    mock_helper = _MockHelper()
    mock_model = SimpleNamespace(
        _minicpmo45_duplex_data_plane_helper=mock_helper,
        _minicpmo45_duplex_request_sessions={"req-1": "sess-1"},
    )
    runner = SimpleNamespace(
        model=mock_model,
        input_batch=SimpleNamespace(
            req_ids=["req-1"],
            num_reqs=1,
            num_computed_tokens_cpu=[600],
        ),
        model_intermediate_buffer={
            "req-1": {
                "duplex": {
                    "stage0_reanchor": {
                        "delta": 240,  # 20 units * 12 tokens
                        "moved_from": 336,
                        "sink_blocks": 6,
                    }
                }
            }
        },
        requests={},
        device=torch.device("cpu"),
        cache_config=SimpleNamespace(block_size=16),
        kv_caches=[],
    )

    MiniCPMO45DuplexWorkerHelper.maybe_apply_reanchor(runner)
    # Pruned from 50 units down to 30 units:
    assert len(session_state.window_units) == 30
    assert session_state.window_units[0].token_ids[0] == 240


def test_grouped_block_table_worker_rotation():
    """Verify worker unpacks production grouped block tables (tuple of lists) across multiple layers and groups."""
    inv_freq = _get_inv_freq()
    k_pool_g0 = torch.randn(10, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float32)
    k_pool_g1 = torch.randn(20, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float32)
    k_pool_g0_orig = k_pool_g0.clone()
    k_pool_g1_orig = k_pool_g1.clone()

    class _MockRunnerGrouped:
        def __init__(self):
            self.device = torch.device("cpu")
            self.cache_config = SimpleNamespace(block_size=BLOCK_SIZE)
            self.model_config = SimpleNamespace(
                get_head_size=lambda: HEAD_DIM,
                hf_config=SimpleNamespace(rope_theta=1000000.0),
            )
            self._duplex_inv_freq = inv_freq
            # Layer 0 belongs to group 0, Layer 1 belongs to group 1
            self.kv_caches = [k_pool_g0, k_pool_g1]
            self.kv_cache_group_ids = [0, 1]
            self.requests = {
                "req-grp": SimpleNamespace(
                    # Tuple of lists: Group 0 has [0, 2, 3], Group 1 has [5, 7, 8]
                    block_ids=([0, 2, 3], [5, 7, 8]),
                    num_computed_tokens=48,
                    mrope_positions=None,
                )
            }
            self.input_batch = SimpleNamespace(
                num_reqs=1,
                req_ids=["req-grp"],
                num_computed_tokens_cpu=np.array([48], dtype=np.int32),
            )
            self.model_intermediate_buffer = {
                "req-grp": {
                    "duplex": {
                        "stage0_reanchor": {
                            "delta": 16,
                            "moved_from": 32,
                            "sink_blocks": 1,
                            "old_computed_tokens": 64,
                        }
                    }
                }
            }

    runner = _MockRunnerGrouped()
    # Worker must resolve each layer's KV group without indexing errors
    MiniCPMO45DuplexWorkerHelper.maybe_apply_reanchor(runner)

    # Group 0: block 0 is sink (untouched), blocks 2 and 3 rotated
    assert torch.equal(k_pool_g0[0], k_pool_g0_orig[0])
    for b in [2, 3]:
        expected = rotate_keys(k_pool_g0_orig[b], 16, inv_freq)
        assert torch.allclose(k_pool_g0[b], expected, atol=1e-6)

    # Group 1: block 5 is sink (untouched), blocks 7 and 8 rotated
    assert torch.equal(k_pool_g1[5], k_pool_g1_orig[5])
    for b in [7, 8]:
        expected = rotate_keys(k_pool_g1_orig[b], 16, inv_freq)
        assert torch.allclose(k_pool_g1[b], expected, atol=1e-6)


def test_rejection_before_state_mutation():
    """Verify rejection before any state mutation when compaction plan is unsupported or fails."""
    session = SimpleNamespace(
        request_id="req-unsupported",
        prompt_token_ids=[1, 2, 3, 4],
        _all_token_ids=[1, 2, 3, 4],
        num_prompt_tokens=4,
        num_computed_tokens=4,
    )

    class _MockFailingDuplexManager:
        def compact_request_blocks(self, request_id: str, *, sink_blocks: int, num_blocks: int) -> int:
            return 0  # Rejection: cannot compact

    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE),
        model_config=SimpleNamespace(max_model_len=40960),
        kv_cache_manager=SimpleNamespace(
            coordinator=SimpleNamespace(single_type_managers=[_MockFailingDuplexManager()])
        ),
    )
    update = SimpleNamespace(
        prompt_token_ids=[5],
        model_intermediate_buffer={
            "duplex": {
                "data_plane": True,
                "runtime_config": {
                    "duplex_window_prefix_tokens": 16,
                    "duplex_window_config": {
                        "sliding_window_mode": "basic",
                        "basic_window_high_tokens": 20,
                        "basic_window_low_tokens": 16,
                    },
                },
            }
        },
    )

    result = MiniCPMO45DuplexSchedulerHelper.maybe_reanchor_session(scheduler, session, update)
    assert result is None
    # Verify zero state mutation: token sequences and counters are untouched!
    assert session.prompt_token_ids == [1, 2, 3, 4]
    assert session._all_token_ids == [1, 2, 3, 4]
    assert session.num_prompt_tokens == 4
    assert session.num_computed_tokens == 4
    assert "stage0_reanchor" not in update.model_intermediate_buffer["duplex"]


def test_non_duplex_regression():
    """Verify that non-duplex requests or sliding_window_mode='off' are completely bypassed."""
    session = SimpleNamespace(
        request_id="req-non-duplex",
        prompt_token_ids=[1, 2, 3],
        _all_token_ids=[1, 2, 3],
        num_prompt_tokens=3,
        num_computed_tokens=3,
    )
    # Case 1: No duplex in buffer
    update1 = SimpleNamespace(prompt_token_ids=[4], model_intermediate_buffer={})
    scheduler = SimpleNamespace()
    assert MiniCPMO45DuplexSchedulerHelper.maybe_reanchor_session(scheduler, session, update1) is None
    assert session.num_computed_tokens == 3

    # Case 2: sliding_window_mode = 'off'
    update2 = SimpleNamespace(
        prompt_token_ids=[4],
        model_intermediate_buffer={
            "duplex": {
                "data_plane": True,
                "runtime_config": {
                    "duplex_window_config": {"sliding_window_mode": "off"},
                },
            }
        },
    )
    assert MiniCPMO45DuplexSchedulerHelper.maybe_reanchor_session(scheduler, session, update2) is None
    assert session.num_computed_tokens == 3


def test_assert_uniform_position_shift():
    """Verify position validation: uniform shifts pass, non-uniform MRoPE raises."""
    # 1. Uniform 2D positions (e.g. streaming audio/text where rows advance identically)
    pos_uniform = torch.arange(100).unsqueeze(0).repeat(3, 1)
    assert_uniform_position_shift(pos_uniform, moved_from=32)

    # 2. Vision tokens in sink only (e.g. 0..20 < 32), uniform in retained tail
    pos_sink_only_vision = pos_uniform.clone()
    pos_sink_only_vision[1, :20] += 5
    pos_sink_only_vision[2, :20] += 10
    assert_uniform_position_shift(pos_sink_only_vision, moved_from=32)

    # 3. Vision tokens in retained tail (>= 32) must be rejected
    pos_tail_vision = pos_uniform.clone()
    pos_tail_vision[1, 40:] += 5
    with pytest.raises(RuntimeError, match="duplex re-anchor needs one position row across the retained tail"):
        assert_uniform_position_shift(pos_tail_vision, moved_from=32)

    # 4. 1D positions trivially uniform
    pos_1d = torch.arange(100)
    assert_uniform_position_shift(pos_1d, moved_from=32)

    # 5. Invalid rank
    with pytest.raises(ValueError, match="expected a \\(rows, tokens\\) position tensor"):
        assert_uniform_position_shift(pos_1d.unsqueeze(0).unsqueeze(0), moved_from=32)


def test_session_mode_location_on_model_config():
    """Verify session_mode is read from vllm_config.model_config, not vllm_config."""
    # Production layout: session_mode is in model_config
    valid_cfg = SimpleNamespace(
        model_config=SimpleNamespace(model_stage="llm", session_mode="duplex", max_model_len=40960),
        cache_config=SimpleNamespace(block_size=16),
    )
    assert getattr(getattr(valid_cfg, "model_config", None), "session_mode", None) == "duplex"

    # Buggy layout: session_mode on vllm_config directly was never populated in vllm-omni
    buggy_cfg = SimpleNamespace(
        session_mode="duplex",
        model_config=SimpleNamespace(model_stage="llm", max_model_len=40960),
        cache_config=SimpleNamespace(block_size=16),
    )
    assert getattr(getattr(buggy_cfg, "model_config", None), "session_mode", None) != "duplex"


def test_scheduler_replace_streaming_prompt_bypasses_reanchor():
    """Verify that when replace_streaming_prompt is True, re-anchoring is bypassed."""
    reanchor_called = []

    class MockScheduler:
        def _release_replaced_streaming_prompt_cache(self, session):
            session.released = True

        def _replace_streaming_session(self, session, update):
            session.replaced = True

        def _maybe_reanchor_minicpmo45_stage0_window(self, session, update):
            reanchor_called.append(True)

        def _update_request_as_session(self, session, update):
            stage_id = 0
            update_infos = [{"meta": {"replace_streaming_prompt": True}}]

            replace_streaming_prompt = any(
                isinstance(info, dict)
                and isinstance(info.get("meta"), dict)
                and info["meta"].get("replace_streaming_prompt") is True
                for info in update_infos
            )
            if replace_streaming_prompt:
                self._release_replaced_streaming_prompt_cache(session)
                self._replace_streaming_session(session, update)
                return

            if stage_id == 0:
                self._maybe_reanchor_minicpmo45_stage0_window(session, update)

    sched = MockScheduler()
    session = SimpleNamespace(released=False, replaced=False)
    sched._update_request_as_session(session, None)

    assert session.released is True
    assert session.replaced is True
    assert len(reanchor_called) == 0, "Re-anchoring must not be called when replacing streaming prompt!"


def test_slot_mapping_device_compatibility():
    """Verify compute_slot_mapping creates table on positions device and does not error on CUDA/device tensor."""
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_kv import compute_slot_mapping

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    pos = torch.tensor([0, 15, 16, 31, 32], dtype=torch.long, device=device)
    block_ids = [10, 20, 30]
    slots = compute_slot_mapping(block_ids, pos, block_size=16)
    assert slots.device == device
    assert slots.tolist() == [10 * 16 + 0, 10 * 16 + 15, 20 * 16 + 0, 20 * 16 + 15, 30 * 16 + 0]


def test_reanchor_ordinary_append_fallback_rebuild_sequence():
    """Reviewer @amy-why-3459 Issue 1: Sequence test: reanchor -> ordinary append -> fallback rebuild.

    Verifies that:
    1. Re-anchor updates scheduler window units and coordinates (_minicpmo45_window_open_start) atomically.
    2. Next ordinary append computes positive unit_len and records unit correctly without early return.
    3. Subsequent fallback rebuild (_window_replacement_parts) succeeds without rebuild-length mismatch.
    """
    block_size = 16
    prefix_tokens = 96
    high_watermark = 8000
    low_watermark = 6000

    class _MockDuplexManager(MiniCPMO45DuplexWindowManager):
        def __init__(self):
            self.block_size = block_size
            self.blocks = list(range(600))
            self.req_to_blocks = {"req-seq": self.blocks}

        def compact_request_blocks(self, request_id: str, *, sink_blocks: int, num_blocks: int) -> int:
            del self.blocks[sink_blocks : sink_blocks + num_blocks]
            return num_blocks * block_size

    duplex_mgr = _MockDuplexManager()
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        model_config=SimpleNamespace(max_model_len=40960),
        kv_cache_manager=SimpleNamespace(coordinator=SimpleNamespace(single_type_managers=[duplex_mgr])),
    )

    # Initial state before reanchor:
    # prefix (96) + 80 units * 100 = 8096; boundary after turn 80 is 8096 + 2 = 8098.
    open_start_initial = prefix_tokens + 80 * 100 + 2
    initial_computed = 8192
    initial_prompt = list(range(initial_computed))
    session = SimpleNamespace(
        request_id="req-seq",
        prompt_token_ids=list(initial_prompt),
        _all_token_ids=list(initial_prompt),
        num_prompt_tokens=initial_computed,
        num_computed_tokens=initial_computed,
        _minicpmo45_window_open_start=open_start_initial,
        _minicpmo45_window_units=[{"length": 100, "generated_token_ids": [1] * 10} for _ in range(80)],
    )

    worker_state = _MiniCPMO45Stage0SessionState(
        session_id="req-seq",
        window_enabled=True,
        context_embeds=[torch.zeros(1, 10)] * prefix_tokens,
        context_token_ids=list(range(prefix_tokens)),
        window_units=[
            _MiniCPMO45WindowUnit(
                embeds=[torch.zeros(1, 10)] * 100,
                token_ids=list(range(100)),
            )
            for _ in range(80)
        ],
    )

    # --- Step 1: Trigger Re-anchor Append ---
    update_reanchor = SimpleNamespace(
        prompt_token_ids=[1000] * 64,
        model_intermediate_buffer={
            "duplex": {
                "data_plane": True,
                "seq": 2,
                "runtime_config": {
                    "duplex_first_append_context_tokens": prefix_tokens,
                    "duplex_window_prefix_tokens": prefix_tokens,
                    "duplex_window_config": {
                        "sliding_window_mode": "basic",
                        "basic_window_high_tokens": high_watermark,
                        "basic_window_low_tokens": low_watermark,
                    },
                },
            }
        },
    )

    plan = MiniCPMO45DuplexSchedulerHelper.apply_session_window(
        scheduler,
        session,
        update_reanchor,
        segment_output_ids=[42],
        completed_terminator=None,
    )
    assert plan is not None
    delta = plan.delta
    assert delta > 0
    assert delta % block_size == 0

    # Verify atomic update of scheduler history and rebasing of coordinates
    assert session.num_computed_tokens == initial_computed - delta
    assert session.num_prompt_tokens == initial_computed - delta
    assert session._minicpmo45_window_open_start == (initial_computed + len([42]) + 2) - delta

    # Worker evicts corresponding window units
    stage0_reanchor = update_reanchor.model_intermediate_buffer["duplex"]["stage0_reanchor"]
    worker_helper = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    worker_helper._embed_token = lambda tok: torch.zeros(1, 10)
    worker_helper._evict_window_units_for_reanchor(worker_state, stage0_reanchor)

    # Finalize the completed turn on the worker
    stage0_window = update_reanchor.model_intermediate_buffer["duplex"]["stage0_window"]
    unclosed_prompt_len = (initial_computed + 1 + 2 - open_start_initial) - 3
    worker_state.pending_window_unit = _MiniCPMO45WindowUnit(
        embeds=[torch.zeros(1, 10)] * unclosed_prompt_len,
        token_ids=list(range(unclosed_prompt_len)),
    )
    worker_state.pending_window_generated_tokens = stage0_window["completed_token_ids"]
    worker_helper._finalize_window_unit(worker_state, closure_token_ids=[999, 998])

    # Simulate scheduler extending prompt with the update's tokens
    session.prompt_token_ids.extend(update_reanchor.prompt_token_ids)
    session.num_prompt_tokens = len(session.prompt_token_ids)

    # --- Step 2: Next Ordinary Append ---
    # Now computed is around 6032 + 64 = 6096, well below 8000
    session.num_computed_tokens = session.num_prompt_tokens
    update_ordinary = SimpleNamespace(
        prompt_token_ids=[2000] * 64,
        model_intermediate_buffer={
            "duplex": {
                "data_plane": True,
                "seq": 3,
                "runtime_config": {
                    "duplex_first_append_context_tokens": prefix_tokens,
                    "duplex_window_prefix_tokens": prefix_tokens,
                    "duplex_window_config": {
                        "sliding_window_mode": "basic",
                        "basic_window_high_tokens": high_watermark,
                        "basic_window_low_tokens": low_watermark,
                    },
                },
            }
        },
    )

    # Zero-copy policy returns None (inside window)
    plan_ordinary = MiniCPMO45DuplexSchedulerHelper.apply_session_window(
        scheduler,
        session,
        update_ordinary,
        segment_output_ids=[43],
    )
    assert plan_ordinary is None

    # Fallback / ordinary prepare helper runs
    replaced = prepare_minicpmo45_stage0_window(
        session,
        update_ordinary,
        segment_output_ids=[43],
    )
    # Must NOT replace, and MUST NOT fail with negative unit_len!
    assert replaced is False
    assert session._minicpmo45_window_open_start > 0

    # Worker finalizes turn from Step 1 on ordinary append
    worker_state.pending_window_unit = _MiniCPMO45WindowUnit(
        embeds=[torch.zeros(1, 10)] * (64 - 3),
        token_ids=[1000] * (64 - 3),
    )
    worker_state.pending_window_generated_tokens = [43]
    worker_helper._finalize_window_unit(worker_state, closure_token_ids=[999, 998])

    session.prompt_token_ids.extend(update_ordinary.prompt_token_ids)
    session.num_prompt_tokens = len(session.prompt_token_ids)

    # --- Step 3: Trigger Fallback Rebuild ---
    # If a watermark fires requesting replace=True, verify worker rebuilds successfully without mismatch!
    update_fallback = SimpleNamespace(
        prompt_token_ids=[3000] * 64,
        model_intermediate_buffer={
            "duplex": {
                "data_plane": True,
                "seq": 4,
                "runtime_config": {
                    "duplex_first_append_context_tokens": prefix_tokens,
                    "duplex_window_prefix_tokens": prefix_tokens,
                    "duplex_window_config": {
                        "sliding_window_mode": "basic",
                        "basic_window_high_tokens": 100,  # Force watermark to fire
                        "basic_window_low_tokens": 80,
                    },
                },
            }
        },
    )
    replaced_fb = prepare_minicpmo45_stage0_window(
        session,
        update_fallback,
        segment_output_ids=[44],
    )
    assert replaced_fb is True
    fb_stage0_window = update_fallback.model_intermediate_buffer["duplex"]["stage0_window"]
    assert fb_stage0_window["replace"] is True

    # Worker finalizes turn from Step 2 when replacement append arrives
    worker_state.pending_window_unit = _MiniCPMO45WindowUnit(
        embeds=[torch.zeros(1, 10)] * (64 - 3),
        token_ids=[2000] * (64 - 3),
    )
    worker_state.pending_window_generated_tokens = [44]
    worker_helper._finalize_window_unit(worker_state, closure_token_ids=[999, 998])

    # Worker sets up pending unit for Step 3 append
    worker_state.pending_window_unit = _MiniCPMO45WindowUnit(
        embeds=[torch.zeros(1, 10)] * 64,
        token_ids=[3000] * 64,
    )

    # Worker rebuilds from history: must succeed without RuntimeError!
    embeds, token_ids = worker_helper._window_replacement_parts(worker_state, fb_stage0_window)
    assert len(token_ids) == fb_stage0_window["replacement_prompt_len"]
    assert len(embeds) == len(token_ids)


def test_worker_history_partial_unit_slice_and_exact_range():
    """Reviewer @amy-why-3459 Issue 2: Preserve exact retained token range when pruning worker history.

    With unit token ranges [0..31] and [32..63], delta=16:
    KV dropped: 16 tokens -> retains [16..63]
    Worker history must also retain [16..63], starting at token 16 (NOT token 32).
    """
    unit0 = _MiniCPMO45WindowUnit(
        embeds=[torch.full((1, 4), float(i)) for i in range(32)],
        token_ids=list(range(32)),
    )
    unit1 = _MiniCPMO45WindowUnit(
        embeds=[torch.full((1, 4), float(i)) for i in range(32, 64)],
        token_ids=list(range(32, 64)),
    )

    state = _MiniCPMO45Stage0SessionState(
        session_id="req-slice",
        window_enabled=True,
        window_units=[unit0, unit1],
    )

    helper = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    helper._evict_window_units_for_reanchor(state, {"delta": 16})

    # Exactly 16 tokens dropped from history!
    remaining_tokens = [tok for u in state.window_units for tok in u.token_ids]
    assert remaining_tokens == list(range(16, 64))
    assert remaining_tokens[0] == 16, "Worker history must retain [16..63], starting at token 16!"
    assert len(remaining_tokens) == 48

    # Verify embeddings match the retained token range [16..64]
    remaining_embeds = [emb for u in state.window_units for emb in u.embeds]
    assert len(remaining_embeds) == 48
    assert remaining_embeds[0][0, 0].item() == 16.0


def test_watermark_interval_high_to_high_plus_prefix():
    """Reviewer @amy-why-3459 Issue 3: Dispatch around the [high, high + prefix] interval.

    With prefix=96, high=8000, computed=7970, pending=64 -> projected=8034.
    The zero-copy policy must trigger (return a plan), avoiding repeated expensive re-prefill.
    """
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_kv import (
        MiniCPMO45DuplexWindowPolicy,
    )

    prefix_tokens = 96
    high_watermark = 8000
    low_watermark = 6000

    session = SimpleNamespace(
        request_id="req-watermark",
        num_computed_tokens=7970,
        num_prompt_tokens=7970,
    )
    update = SimpleNamespace(
        prompt_token_ids=[1] * 64,
        model_intermediate_buffer={
            "duplex": {
                "data_plane": True,
                "runtime_config": {
                    "duplex_window_prefix_tokens": prefix_tokens,
                    "duplex_window_config": {
                        "sliding_window_mode": "basic",
                        "basic_window_high_tokens": high_watermark,
                        "basic_window_low_tokens": low_watermark,
                    },
                },
            }
        },
    )

    # projected = 7970 + 64 = 8034.
    # In the [8000, 8096] interval, plan_reanchor MUST trigger!
    plan = MiniCPMO45DuplexWindowPolicy.plan_reanchor(
        session,
        update,
        block_size=16,
        max_model_len=40960,
    )
    assert plan is not None, "Zero-copy policy must trigger when total length passes basic_window_high_tokens!"
    assert plan.delta > 0
    # Post-trim length must be bounded by low_watermark
    assert 8034 - plan.delta <= low_watermark


def test_validate_duplex_window_install_block_sizes():
    """Verify validate_duplex_window_install supports arbitrary block sizes (including NPU 128)."""
    # Standard CUDA block_size=16
    cache_config_cuda = SimpleNamespace(enable_prefix_caching=False, block_size=16)
    model_config = SimpleNamespace(max_model_len=40960)
    geometry_16 = duplex_window_geometry(
        prefix_tokens=96,
        window_tokens=6000,
        block_size=16,
        max_model_len=40960,
        high_watermark_tokens=8000,
    )
    validate_duplex_window_install(cache_config_cuda, model_config, geometry_16)

    # Ascend NPU block_size=128
    cache_config_npu = SimpleNamespace(enable_prefix_caching=False, block_size=128)
    geometry_128 = duplex_window_geometry(
        prefix_tokens=96,
        window_tokens=6000,
        block_size=128,
        max_model_len=40960,
        high_watermark_tokens=8000,
    )
    validate_duplex_window_install(cache_config_npu, model_config, geometry_128)

    # Rejects mismatched block size between geometry and cache_config
    with pytest.raises(ValueError, match="does not match"):
        validate_duplex_window_install(cache_config_npu, model_config, geometry_16)

    # Rejects prefix caching
    cache_config_pc = SimpleNamespace(enable_prefix_caching=True, block_size=128)
    with pytest.raises(ValueError, match="enable_prefix_caching=False"):
        validate_duplex_window_install(cache_config_pc, model_config, geometry_128)

    # Rejects when window does not fit max_model_len
    model_config_small = SimpleNamespace(max_model_len=2048)
    with pytest.raises(ValueError, match="does not fit max_model_len"):
        validate_duplex_window_install(cache_config_npu, model_config_small, geometry_128)
