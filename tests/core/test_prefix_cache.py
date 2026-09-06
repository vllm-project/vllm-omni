# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for vllm_omni/core/prefix_cache (manager + controller + pool).

CPU-only: the controller runs in eager mode. Uses a fake group view, so no
vLLM runtime is required (runnable without a vllm install via
``pytest --confcutdir=tests/core``).
"""

import ast
import logging
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

try:  # pragma: no cover - shim only matters on vllm-less dev machines
    import vllm  # noqa: F401
except ModuleNotFoundError:
    # Bypass vllm_omni/__init__ (which imports vllm): register namespace
    # parents so the pure-torch prefix_cache subpackage imports directly.
    _root = Path(__file__).resolve().parents[2]
    for _pkg in ("vllm_omni", "vllm_omni.core"):
        if _pkg not in sys.modules:
            _m = __import__("types").ModuleType(_pkg)
            _m.__path__ = [str(_root / _pkg.replace(".", "/"))]
            sys.modules[_pkg] = _m

from vllm_omni.core.prefix_cache.controller import StagingBufferHolder
from vllm_omni.core.prefix_cache.group_view import FullAttentionGroupView, check_prefix_cache_kv_groups
from vllm_omni.core.prefix_cache.interface import (
    HIDDEN_KEY,
    ModelCachePolicy,
    OmniPrefixCacheUnmatchError,
    PrefixCacheConfig,
    WriteSchedule,
)
from vllm_omni.core.prefix_cache.manager import (
    MmValueKind,
    OmniPrefixCacheManager,
    _snapshot_leftover_mm_cpu,
    classify_mm_value,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

NUM_BLOCKS = 16
BLOCK_SIZE = 4
HIDDEN = 8
DTYPE = torch.float32


class FakeView:
    """Duck-typed group view backed by plain dicts."""

    def __init__(self):
        self.block_size = BLOCK_SIZE
        self.req_blocks: dict[str, list[int]] = {}
        self.order: list[str] = []
        self.computed: dict[str, int] = {}
        self.step_slot_mapping: torch.Tensor | None = None

    def slots_for(self, req_id, token_start, token_end):
        blocks = self.req_blocks[req_id]
        slots = []
        for pos in range(token_start, token_end):
            slots.append(blocks[pos // BLOCK_SIZE] * BLOCK_SIZE + pos % BLOCK_SIZE)
        return torch.tensor(slots, dtype=torch.long)

    def batch_req_ids(self) -> list[str]:
        return list(self.order)

    def step_slots_cpu(self, req_ids, num_scheduled) -> torch.Tensor:
        parts = []
        for r in req_ids:
            n = int(num_scheduled.get(r, 0))
            if n <= 0:
                continue
            start = self.computed.get(r, 0)
            parts.append(self.slots_for(r, start, start + n))
        return torch.cat(parts) if parts else torch.empty((0,), dtype=torch.long)


class FakeNewReq:
    def __init__(self, req_id, num_computed_tokens=0, block_ids=None):
        self.req_id = req_id
        self.num_computed_tokens = num_computed_tokens
        self.block_ids = block_ids


class FakeSchedOut:
    def __init__(self, new_reqs=(), finished=(), num_scheduled=None):
        self.scheduled_new_reqs = list(new_reqs)
        self.finished_req_ids = set(finished)
        self.num_scheduled_tokens = dict(num_scheduled or {})


def make_manager(view=None, policy=None, **cfg_kwargs) -> tuple[OmniPrefixCacheManager, FakeView]:
    view = view or FakeView()
    config = PrefixCacheConfig(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE, **cfg_kwargs)
    mgr = OmniPrefixCacheManager(config, view, eager=True)
    if policy is not None:
        mgr.register_policy(policy)
    return mgr, view


def run_step(
    mgr,
    view,
    reqs: dict[str, tuple[list[int], int, int]],
    new_hits=None,
    finished=(),
    mm=None,
    num_tokens_padded=None,
) -> int:
    """One step: reqs = req_id -> (blocks, sched_start_pos, sched_tokens)."""
    view.order = list(reqs.keys())
    new_reqs = []
    num_sched = {}
    slot_parts = []
    hidden_parts = []
    for req_id, (blocks, start_pos, sched) in reqs.items():
        view.req_blocks[req_id] = blocks
        view.computed[req_id] = start_pos
        num_sched[req_id] = sched
        slots = view.slots_for(req_id, start_pos, start_pos + sched)
        slot_parts.append(slots)
        hidden_parts.append(slots.to(DTYPE).unsqueeze(1).expand(sched, HIDDEN).clone())
        hit = (new_hits or {}).get(req_id, 0)
        new_reqs.append(FakeNewReq(req_id, num_computed_tokens=hit, block_ids=[list(blocks)]))
    view.step_slot_mapping = torch.cat(slot_parts)
    hidden = torch.cat(hidden_parts)
    sched_out = FakeSchedOut(new_reqs=new_reqs, finished=finished, num_scheduled=num_sched)
    mgr.new_step_starts(sched_out)
    n = int(view.step_slot_mapping.numel())
    padded = n if num_tokens_padded is None else int(num_tokens_padded)
    return mgr.save_outputs(hidden, mm or {}, num_tokens_unpadded=n, num_tokens_padded=padded)


def expected_rows(slots: torch.Tensor) -> torch.Tensor:
    return slots.to(DTYPE).unsqueeze(1).expand(slots.numel(), HIDDEN)


def plan_fetch(mgr, slots, key, *, strict, req_id):
    with mgr._state_lock:
        src = mgr._plan_rows(slots, key, strict, req_id)
    with torch.inference_mode():
        return mgr._fetch_source(src)


def _assert_leftover_shapes(inp, got, n: int) -> None:
    """Every leftover tensor keeps its shape unless it is exactly token-major."""
    if isinstance(inp, torch.Tensor):
        if inp.ndim >= 1 and n > 0 and inp.shape[0] == n:
            assert got.shape == (n, *inp.shape[1:])
        else:
            assert got.shape == inp.shape
        return
    if isinstance(inp, dict):
        assert set(got) == set(inp)
        for k in inp:
            _assert_leftover_shapes(inp[k], got[k], n)
        return
    if isinstance(inp, (list, tuple)):
        assert len(got) == len(inp)
        for a, b in zip(inp, got):
            _assert_leftover_shapes(a, b, n)


def _module_scope_imported_modules(path: Path) -> list[str]:
    """Top-level import module names, skipping ``if TYPE_CHECKING`` blocks."""
    tree = ast.parse(path.read_text())
    names: list[str] = []

    def _from_stmt(node: ast.AST) -> None:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)

    for node in tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            continue
        _from_stmt(node)
    return names


def test_prefix_cache_package_has_no_module_scope_vllm_import():
    """This package must import without vllm. See the package docstring."""
    pkg = Path(__file__).resolve().parents[2] / "vllm_omni" / "core" / "prefix_cache"
    leaked: list[str] = []
    for path in sorted(pkg.glob("*.py")):
        for mod in _module_scope_imported_modules(path):
            if mod == "vllm" or mod.startswith("vllm."):
                leaked.append(f"{path.name}: {mod}")
    assert not leaked, leaked


def test_no_hit_passthrough():
    mgr, view = make_manager()
    sid = run_step(mgr, view, {"a": ([0, 1], 0, 8)})
    outs = mgr.materialize(sid, ["a"])
    assert torch.equal(outs.hidden_states["a"], expected_rows(view.slots_for("a", 0, 8)))


def test_hit_merge_from_mirror():
    mgr, view = make_manager()
    s1 = run_step(mgr, view, {"a": ([0, 1], 0, 8)})
    mgr.materialize(s1, ["a"])
    s2 = run_step(mgr, view, {"b": ([0, 1, 2], 8, 4)}, new_hits={"b": 8}, finished=["a"])
    outs = mgr.materialize(s2, ["b"])
    merged = outs.hidden_states["b"]
    assert merged.shape == (12, HIDDEN)
    assert torch.equal(merged[:8], expected_rows(view.slots_for("b", 0, 8)))
    assert torch.equal(merged[8:], expected_rows(view.slots_for("b", 8, 12)))


def test_join_next_step_hit_waits_done_then_reads_pool():
    """CPU stand-in for the non-eager path: submit registers but does not
    scatter. A same-step hit must join(done), drain, then read the pool."""
    mgr, view = make_manager()
    held: list = []
    real_run = mgr._controller._run_eager
    real_join = mgr._controller.join

    def hold_submit(task):
        mgr._controller._tasks[task.tid] = task
        held.append(task)

    def join_then_scatter(tids):
        for task in list(held):
            if task.tid in tids and not task.done.is_set():
                real_run(task)
                held.remove(task)
        real_join(tids)

    mgr._controller.submit = hold_submit
    mgr._controller.join = join_then_scatter

    view.req_blocks["a"] = [0, 1]
    sid = run_step(
        mgr,
        view,
        {"a": ([0, 1], 0, 8), "b": ([0, 1, 2], 8, 4)},
        new_hits={"b": 8},
    )
    outs = mgr.materialize(sid, ["a", "b"])
    assert torch.equal(outs.hidden_states["b"][:8], expected_rows(view.slots_for("b", 0, 8)))
    slots = view.slots_for("a", 0, 8)
    assert torch.equal(mgr._pool.rows(HIDDEN_KEY, slots), expected_rows(slots))


def test_same_step_hit_reads_in_transit():
    mgr, view = make_manager()
    view.req_blocks["a"] = [0, 1]
    sid = run_step(
        mgr,
        view,
        {"a": ([0, 1], 0, 8), "b": ([0, 1, 2], 8, 4)},
        new_hits={"b": 8},
    )
    outs = mgr.materialize(sid, ["a", "b"])
    assert torch.equal(outs.hidden_states["b"][:8], expected_rows(view.slots_for("b", 0, 8)))


def test_absent_hit_fails_fast():
    mgr, view = make_manager()
    view.req_blocks["c"] = [5, 6]
    sid = run_step(mgr, view, {"c": ([5, 6, 7], 8, 4)}, new_hits={"c": 8})
    d2h = mgr._step_ctxs[sid].d2h
    with pytest.raises(OmniPrefixCacheUnmatchError):
        mgr.materialize(sid, ["c"])
    if d2h is not None:
        assert not mgr._controller._staging_pool._busy[d2h.slot]


def test_hit_not_block_aligned_asserts():
    mgr, view = make_manager()
    s1 = run_step(mgr, view, {"a": ([0, 1], 0, 8)})
    mgr.materialize(s1, ["a"])
    s2 = run_step(mgr, view, {"b": ([0, 1, 2], 8, 4)}, new_hits={"b": 6})
    with pytest.raises(AssertionError):
        mgr.materialize(s2, ["b"])


def test_mm_cached_key_merge():
    mgr, view = make_manager()
    feat = 3
    mm1 = {"talker.h": torch.arange(8 * feat, dtype=DTYPE).reshape(8, feat)}
    s1 = run_step(mgr, view, {"a": ([0, 1], 0, 8)}, mm=mm1)
    mgr.materialize(s1, ["a"])
    mm2 = {"talker.h": torch.full((4, feat), 7.0)}
    s2 = run_step(mgr, view, {"b": ([0, 1, 2], 8, 4)}, new_hits={"b": 8}, finished=["a"], mm=mm2)
    outs = mgr.materialize(s2, ["b"])
    merged = outs.mm_outputs["talker.h"]["b"]
    assert merged.shape == (12, feat)
    assert torch.equal(merged[:8], mm1["talker.h"])
    assert torch.equal(merged[8:], mm2["talker.h"])


def test_codes_audio_matches_ordinary_mm_key():
    """Without a deferred policy, codes.audio is an ordinary token-major mm key."""
    mgr, view = make_manager()
    step1 = torch.arange(8 * 2, dtype=DTYPE).reshape(8, 2)
    s1 = run_step(mgr, view, {"a": ([0, 1], 0, 8)}, mm={"codes.audio": step1})
    assert torch.equal(mgr.materialize(s1, ["a"]).mm_outputs["codes.audio"]["a"], step1)
    step2 = torch.full((4, 2), 7.0)
    s2 = run_step(mgr, view, {"b": ([0, 1, 2], 8, 4)}, new_hits={"b": 8}, finished=["a"], mm={"codes.audio": step2})
    merged = mgr.materialize(s2, ["b"]).mm_outputs["codes.audio"]["b"]
    assert torch.equal(merged, torch.cat([step1, step2], dim=0))


def test_unpadded_mm_registers_on_padded_step():
    """Qwen3-TTS codes.audio is scheduled-length; cudagraph makes padded > n.

    Immediate-only `shape[0] == padded` would skip ensure_key, drop the
    rows into leftover, and silently serve hits without a cached prefix.
    """
    mgr, view = make_manager()
    audio = torch.arange(8 * 2, dtype=DTYPE).reshape(8, 2)
    s1 = run_step(mgr, view, {"a": ([0, 1], 0, 8)}, mm={"codes.audio": audio}, num_tokens_padded=16)
    assert mgr._pool.has_key("codes.audio")
    assert "codes.audio" not in mgr._step_ctxs[s1].mm_cpu_snapshot
    assert torch.equal(mgr.materialize(s1, ["a"]).mm_outputs["codes.audio"]["a"], audio)
    s2 = run_step(
        mgr,
        view,
        {"b": ([0, 1, 2], 8, 4)},
        new_hits={"b": 8},
        finished=["a"],
        mm={"codes.audio": torch.full((4, 2), 3.0)},
        num_tokens_padded=16,
    )
    merged = mgr.materialize(s2, ["b"]).mm_outputs["codes.audio"]["b"]
    assert torch.equal(merged[:8], audio)
    assert torch.equal(merged[8:], torch.full((4, 2), 3.0))


def test_leftover_snapshot_preserves_non_token_major_shapes():
    """P1: leftover copy must not use `shape[0] >= n` as a slice predicate.

    Token-major tensors (`shape[0] == n`) may be sliced; every other tensor
    — including `codes.ref` with ref_len >> n, and list-held tensors —
    must keep the input shape.
    """
    n = 4
    mm = {
        "codes.ref": torch.arange(30, dtype=DTYPE).reshape(15, 2),
        "tags": [torch.ones(3, 5), torch.zeros(7)],
        "tokenish": torch.arange(n * 2, dtype=DTYPE).reshape(n, 2),
        "scalar": torch.tensor(3.0),
    }
    out = _snapshot_leftover_mm_cpu(mm, set(), n)
    _assert_leftover_shapes(mm, out, n)
    assert torch.equal(out["codes.ref"], mm["codes.ref"])
    assert out["codes.ref"].shape == (15, 2)


def test_codes_ref_matches_old_build_mm_cpu_path():
    """Leftover `codes.ref` equals the old runner path: build_mm_cpu + to_payload_element."""
    pytest.importorskip("vllm")
    from vllm_omni.utils.mm_outputs import build_mm_cpu, to_payload_element

    refs = [torch.arange(8, dtype=torch.long), torch.arange(12, dtype=torch.long)]
    mgr, view = make_manager()
    sid = run_step(mgr, view, {"a": ([0], 0, 2), "b": ([1], 0, 2)}, mm={"codes.ref": refs})
    ctx = mgr._step_ctxs[sid]
    assert "codes.ref" in ctx.mm_cpu_snapshot
    assert [t.shape for t in ctx.mm_cpu_snapshot["codes.ref"]] == [t.shape for t in refs]
    outs = mgr.materialize(sid, ["a", "b"])
    old = build_mm_cpu({"codes.ref": refs})
    total = 4
    for req_id, idx, start, end in (("a", 0, 0, 2), ("b", 1, 2, 4)):
        want = to_payload_element(old["codes.ref"], idx, start=start, end=end, pass_lists_through=True, seq_len=total)
        got = outs.mm_outputs["codes.ref"][req_id]
        assert len(got) == len(want)
        for g, w in zip(got, want):
            assert torch.equal(g, w)


def test_classify_mm_value_three_buckets():
    n, padded = 4, 8
    assert classify_mm_value(torch.zeros(n, 2), n, padded) is MmValueKind.TOKEN_MAJOR
    assert classify_mm_value(torch.zeros(padded, 2), n, padded) is MmValueKind.TOKEN_MAJOR
    assert classify_mm_value([torch.zeros(1, 8)], n, padded) is MmValueKind.REQ_LIST
    assert classify_mm_value(torch.zeros(15, 2), n, padded) is MmValueKind.PASSTHROUGH
    assert classify_mm_value(torch.tensor([1, 2, 3]), n, padded) is MmValueKind.PASSTHROUGH


def test_policy_from_model_shim():
    class M:
        requires_full_prefix_cached_hidden_states = False
        deferred_prefix_cache_mm_keys = {"codes.audio"}

    p = ModelCachePolicy.from_model(M())
    assert p.needs_full_hidden_states is False
    assert p.deferred_keys == frozenset({"codes.audio"})
    d = ModelCachePolicy.from_model(object())
    assert d.needs_full_hidden_states is True and not d.deferred_keys


def test_deferred_key_accumulates_and_flushes_on_finish():
    policy = ModelCachePolicy(needs_full_hidden_states=False, deferred_keys=frozenset({"codes.audio"}))
    mgr, view = make_manager(policy=policy)
    feat = 2
    for pos in range(2):
        sid = run_step(
            mgr,
            view,
            {"a": ([3], pos, 1)},
            mm={"codes.audio": torch.full((1, feat), float(pos + 1))},
        )
        mgr.materialize(sid, ["a"])
    slots = view.slots_for("a", 0, 2)
    rows = plan_fetch(mgr, slots, "codes.audio", strict=False, req_id="a")
    assert torch.equal(rows[:, 0], torch.tensor([1.0, 2.0]))
    sid = run_step(mgr, view, {"z": ([9], 0, 1)}, finished=["a"])
    mgr.materialize(sid, ["z"])
    mirror = mgr._pool.rows("codes.audio", slots)
    assert torch.equal(mirror[:, 0], torch.tensor([1.0, 2.0]))


def test_deferred_gpu_bytes_held_until_last_view_drops():
    """C→1 clone is one allocation; finishing the first co-scheduled
    request must not drop ``_staged_bytes`` while the other still pins it."""
    policy = ModelCachePolicy(needs_full_hidden_states=False, deferred_keys=frozenset({"k"}))
    mgr, view = make_manager(policy=policy)
    feat = 2
    n = 4
    s1 = run_step(
        mgr,
        view,
        {"a": ([0], 0, 2), "b": ([1], 0, 2)},
        mm={"k": torch.zeros(n, feat)},
    )
    mgr.materialize(s1, ["a", "b"])
    clone_bytes = n * feat * 4  # float32
    assert mgr._controller._staged_bytes == clone_bytes
    s2 = run_step(mgr, view, {"z": ([9], 0, 1)}, finished=["a"])
    mgr.materialize(s2, ["z"])
    assert mgr._controller._staged_bytes == clone_bytes
    s3 = run_step(mgr, view, {"z2": ([9], 1, 1)}, finished=["b"])
    mgr.materialize(s3, ["z2"])
    assert mgr._controller._staged_bytes == 0


def test_cap_forces_flush_of_deferred():
    policy = ModelCachePolicy(needs_full_hidden_states=False, deferred_keys=frozenset({"k"}))
    mgr, view = make_manager(policy=policy, gpu_staging_bytes=48)
    for pos in range(4):
        sid = run_step(mgr, view, {"a": ([2, 3], pos, 1)}, mm={"k": torch.full((1, 4), float(pos))})
        mgr.materialize(sid, ["a"])
    early = view.slots_for("a", 0, 1)
    assert float(mgr._pool.rows("k", early)[0, 0]) == 0.0


def test_deferred_tenant_succession_no_stale_wins():
    policy = ModelCachePolicy(needs_full_hidden_states=False, deferred_keys=frozenset({"k"}))
    mgr, view = make_manager(policy=policy)
    s1 = run_step(mgr, view, {"a": ([2], 0, 2)}, mm={"k": torch.full((2, 2), 1.0)})
    mgr.materialize(s1, ["a"])
    s2 = run_step(mgr, view, {"b": ([2], 0, 2)}, mm={"k": torch.full((2, 2), 2.0)})
    mgr.materialize(s2, ["b"])
    slots = view.slots_for("b", 0, 2)
    s3 = run_step(mgr, view, {"z1": ([9], 0, 1)}, finished=["b"])
    mgr.materialize(s3, ["z1"])
    s4 = run_step(mgr, view, {"z2": ([9], 1, 1)}, finished=["a"])
    mgr.materialize(s4, ["z2"])
    assert torch.equal(mgr._pool.rows("k", slots), torch.full((2, 2), 2.0))


def test_deferred_key_hit_reads_staged_rows_not_mirror():
    policy = ModelCachePolicy(needs_full_hidden_states=True, deferred_keys=frozenset({"k"}))
    mgr, view = make_manager(policy=policy)
    s1 = run_step(mgr, view, {"a": ([0], 0, 4)}, mm={"k": torch.full((4, 2), 1.0)})
    mgr.materialize(s1, ["a"])
    s2 = run_step(mgr, view, {"b": ([0, 1], 4, 2)}, new_hits={"b": 4}, mm={"k": torch.full((2, 2), 9.0)})
    rows = mgr.materialize(s2, ["b"]).mm_outputs["k"]["b"]
    assert torch.equal(rows[:4], torch.full((4, 2), 1.0)), rows
    assert torch.equal(rows[4:], torch.full((2, 2), 9.0))


def test_append_to_closed_deferred_entry_opens_new_one():
    policy = ModelCachePolicy(needs_full_hidden_states=False, deferred_keys=frozenset({"k"}))
    mgr, view = make_manager(policy=policy)
    s1 = run_step(mgr, view, {"a": ([2], 0, 1)}, mm={"k": torch.full((1, 2), 1.0)})
    mgr.materialize(s1, ["a"])
    first = mgr._deferred_tasks["a"]
    mgr._controller.escalate([first.tid])
    s2 = run_step(mgr, view, {"a": ([2], 1, 1)}, mm={"k": torch.full((1, 2), 2.0)})
    mgr.materialize(s2, ["a"])
    assert mgr._deferred_tasks["a"].tid != first.tid
    slots = view.slots_for("a", 0, 2)
    rows = plan_fetch(mgr, slots, "k", strict=False, req_id="a")
    assert torch.equal(rows[:, 0], torch.tensor([1.0, 2.0]))


def test_deferred_unpadded_registers_on_padded_step():
    """Qwen3-TTS: deferred codes.audio is unpadded; cudagraph pads hidden."""
    policy = ModelCachePolicy(needs_full_hidden_states=False, deferred_keys=frozenset({"codes.audio"}))
    mgr, view = make_manager(policy=policy)
    audio = torch.arange(8 * 2, dtype=DTYPE).reshape(8, 2)
    s1 = run_step(mgr, view, {"a": ([0, 1], 0, 8)}, mm={"codes.audio": audio}, num_tokens_padded=16)
    assert mgr._pool.has_key("codes.audio")
    assert "codes.audio" in mgr._step_ctxs[s1].mm_cpu_snapshot
    # Hidden opt-out + no hit: materialize returns empty; the freeze is
    # the durable copy. Fetch it before consume.
    rows = plan_fetch(mgr, view.slots_for("a", 0, 8), "codes.audio", strict=False, req_id="a")
    assert torch.equal(rows, audio)
    mgr.materialize(s1, ["a"])


def test_check_kv_groups_rejects_empty_or_multi():
    with pytest.raises(OmniPrefixCacheUnmatchError, match="single full-attention"):
        check_prefix_cache_kv_groups([])
    with pytest.raises(OmniPrefixCacheUnmatchError, match="single full-attention"):
        check_prefix_cache_kv_groups([object(), object()])


def _table_slots(table, req_idx, token_start, token_end, block_size=BLOCK_SIZE):
    if token_end <= token_start:
        return torch.empty((0,), dtype=torch.long)
    token_positions = torch.arange(token_start, token_end, dtype=torch.long)
    block_offsets = token_positions // block_size
    max_blocks = int(table.shape[1])
    valid = block_offsets < max_blocks
    if not bool(valid.all()):
        token_positions = token_positions[valid]
        block_offsets = block_offsets[valid]
    if token_positions.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    block_ids = table[req_idx, block_offsets].to(torch.long)
    return block_ids * block_size + (token_positions % block_size)


def test_group_view_step_slots_cpu():
    class TensorWrap:
        def __init__(self, t):
            self.cpu = t

    class Group:
        def __init__(self, t):
            self.block_table = TensorWrap(t)

    class BT:
        def __init__(self, t):
            self._g = Group(t)
            self.block_tables = [self._g.block_table]

        def __getitem__(self, idx):
            assert idx == 0
            return self._g

    class IB:
        def __init__(self, table):
            self.req_ids = ["r1", "r2"]
            self.req_id_to_index = {"r1": 0, "r2": 1}
            self.num_computed_tokens_cpu = torch.tensor([8, 0])
            self.block_table = BT(table)

    table = torch.tensor([[2, 5, 7], [1, 0, 0]])
    view = FullAttentionGroupView(IB(table), block_size=BLOCK_SIZE)
    assert view.step_slots_cpu(["r1"], {"r1": 2}).tolist() == [7 * 4 + 0, 7 * 4 + 1]
    assert view.step_slots_cpu(["r1"], {"r1": 8}).tolist() == [7 * 4 + i for i in range(4)]
    assert view.step_slots_cpu(["r1"], {"r1": 0}).numel() == 0


def test_join_next_step_previous_save():
    mgr, view = make_manager()
    s1 = run_step(mgr, view, {"a": ([0], 0, 2)})
    assert len(mgr._join_next_step_tids) == 1
    task_id = mgr._join_next_step_tids[0]
    mgr.materialize(s1, ["a"])
    s2 = run_step(mgr, view, {"a": ([0], 2, 1)})
    assert mgr._controller.get_task(task_id) is None
    assert int(mgr._key_state[HIDDEN_KEY][view.slots_for("a", 0, 2)].min()) == 2
    mgr.materialize(s2, ["a"])


def test_tenant_succession_hit_reads_newest():
    mgr, view = make_manager()
    s1 = run_step(mgr, view, {"a": ([0], 0, 4)})
    mgr.materialize(s1, ["a"])
    s2 = run_step(mgr, view, {"b": ([0], 0, 4)}, finished=["a"])
    mgr.materialize(s2, ["b"])
    b_hidden = expected_rows(view.slots_for("b", 0, 4))
    s3 = run_step(mgr, view, {"c": ([0, 1], 4, 2)}, new_hits={"c": 4}, finished=["b"])
    outs = mgr.materialize(s3, ["c"])
    assert torch.equal(outs.hidden_states["c"][:4], b_hidden)


def test_tenant_succession_mm_key():
    mgr, view = make_manager()
    s1 = run_step(mgr, view, {"a": ([2], 0, 4)}, mm={"k": torch.full((4, 2), 1.0)})
    mgr.materialize(s1, ["a"])
    s2 = run_step(mgr, view, {"b": ([2], 0, 4)}, finished=["a"], mm={"k": torch.full((4, 2), 2.0)})
    mgr.materialize(s2, ["b"])
    s3 = run_step(
        mgr, view, {"c": ([2, 3], 4, 2)}, new_hits={"c": 4}, finished=["b"], mm={"k": torch.full((2, 2), 3.0)}
    )
    rows = mgr.materialize(s3, ["c"]).mm_outputs["k"]["c"]
    assert torch.equal(rows[:4], torch.full((4, 2), 2.0))
    assert torch.equal(rows[4:], torch.full((2, 2), 3.0))


def test_slot_reuse_pushes_skip_to_old_task():
    """Reassignment = task swap. Keep the old write in-transit so remount records it."""
    mgr, view = make_manager()

    def register_only(task):
        with mgr._controller._lock:
            mgr._controller._tasks[task.tid] = task
        task.host_ready.set()

    mgr._controller.submit = register_only
    policy = ModelCachePolicy(needs_full_hidden_states=False)
    mgr.register_policy(policy)
    s1 = run_step(mgr, view, {"a": ([2], 0, 1)}, mm={"k": torch.ones(1, 2)})
    mgr.materialize(s1, ["a"])
    old_tid = next(iter(mgr._req_tasks["a"]))
    old_task = mgr._controller.get_task(old_tid)
    s2 = run_step(mgr, view, {"b": ([2], 0, 2)}, mm={"k": torch.full((2, 2), 2.0)})
    assert old_task is not None and "k" in old_task.reassigned
    reused = view.slots_for("b", 0, 1)
    assert bool(torch.isin(reused, old_task.reassigned["k"]).all())
    mgr.materialize(s2, ["b"])


def test_step_slots_cpu_matches_block_table_math():
    class TensorWrap:
        def __init__(self, t):
            self.cpu = t

    class Group:
        def __init__(self, t):
            self.block_table = TensorWrap(t)

    class BT:
        def __init__(self, t):
            self._g = Group(t)
            self.block_tables = [self._g.block_table]

        def __getitem__(self, idx):
            return self._g

    class IB:
        def __init__(self, table, computed):
            self.req_ids = ["r1", "r2"]
            self.req_id_to_index = {"r1": 0, "r2": 1}
            self.num_computed_tokens_cpu = torch.tensor(computed)
            self.block_table = BT(table)

    table = torch.tensor([[2, 5, 7], [1, 3, 4]])
    view = FullAttentionGroupView(IB(table, [4, 0]), block_size=BLOCK_SIZE)
    num_sched = {"r1": 3, "r2": 5}
    got = view.step_slots_cpu(["r1", "r2"], num_sched)
    want = torch.cat([_table_slots(table, 0, 4, 7), _table_slots(table, 1, 0, 5)])
    assert torch.equal(got, want), (got, want)
    assert torch.equal(view.step_slots_cpu(["r1", "r2"], {"r1": 3, "r2": 0}), _table_slots(table, 0, 4, 7))


def test_step_context_consumed_by_id_not_order():
    mgr, view = make_manager()
    s1 = run_step(mgr, view, {"a": ([0], 0, 4)})
    s2 = run_step(mgr, view, {"b": ([1], 0, 4)})
    mgr.discard_step(s2)
    outs = mgr.materialize(s1, ["a"])
    assert torch.equal(outs.hidden_states["a"], expected_rows(view.slots_for("a", 0, 4)))
    assert len(mgr._step_ctxs) == 0


def test_step_context_exactly_once():
    mgr, view = make_manager()
    sid = run_step(mgr, view, {"a": ([0], 0, 4)})
    mgr.materialize(sid, ["a"])
    with pytest.raises(OmniPrefixCacheUnmatchError):
        mgr.materialize(sid, ["a"])
    with pytest.raises(OmniPrefixCacheUnmatchError):
        mgr.discard_step(sid)


def test_unconsumed_contexts_overflow_fails_fast():
    mgr, view = make_manager()
    with pytest.raises(OmniPrefixCacheUnmatchError, match="unconsumed step contexts"):
        for pos in range(8):
            run_step(mgr, view, {"a": ([0, 1, 2, 3], pos, 1)})


def test_save_slot_mismatch_fails_fast():
    mgr, view = make_manager()
    view.order = ["a"]
    view.req_blocks["a"] = [0]
    view.computed["a"] = 0
    mgr.new_step_starts(FakeSchedOut(new_reqs=[FakeNewReq("a")], num_scheduled={"a": 2}))
    hidden = torch.zeros(4, HIDDEN, dtype=DTYPE)
    with pytest.raises(OmniPrefixCacheUnmatchError):
        mgr.save_outputs(hidden, {}, num_tokens_unpadded=4, num_tokens_padded=4)


def test_materialize_rejects_out_of_snapshot_ids():
    mgr, view = make_manager()
    s1 = run_step(mgr, view, {"a": ([0, 1], 0, 8)})
    mgr.materialize(s1, ["a"])
    s2 = run_step(mgr, view, {"b": ([0, 1, 2], 8, 4)}, new_hits={"b": 8}, finished=["a"])
    with pytest.raises(AssertionError, match="outside the save snapshot"):
        mgr.materialize(s2, ["b", "late_joiner"])


def test_materialize_unknown_step_id_fails_fast():
    mgr, view = make_manager()
    sid = run_step(mgr, view, {"a": ([0], 0, 4)})
    with pytest.raises(OmniPrefixCacheUnmatchError):
        mgr.materialize(sid + 999, ["ghost"])
    mgr.discard_step(sid)


def test_mm_hit_span_never_registered_serves_mirror_baseline():
    """A hit span slot on which a sparse mm key was never registered is
    legitimate absence — served from the mirror baseline, not a crash."""
    policy = ModelCachePolicy(needs_full_hidden_states=True, deferred_keys=frozenset({"k"}))
    mgr, view = make_manager(policy=policy)
    s1 = run_step(mgr, view, {"a": ([0], 0, 4)}, mm={"k": torch.full((4, 2), 5.0)})
    mgr.materialize(s1, ["a"])
    s2 = run_step(mgr, view, {"c": ([1], 0, 4)})
    mgr.materialize(s2, ["c"])
    s3 = run_step(mgr, view, {"b": ([0, 1, 2], 8, 2)}, new_hits={"b": 8}, mm={"k": torch.full((2, 2), 9.0)})
    rows = mgr.materialize(s3, ["b"]).mm_outputs["k"]["b"]
    assert torch.equal(rows[:4], torch.full((4, 2), 5.0))
    assert torch.equal(rows[4:8], torch.zeros(4, 2))
    assert torch.equal(rows[8:], torch.full((2, 2), 9.0))


def test_mm_in_transit_unresolvable_fails_fast():
    """Rows registered in-transit whose entry cannot serve them must raise."""
    policy = ModelCachePolicy(needs_full_hidden_states=True, deferred_keys=frozenset({"k"}))
    mgr, view = make_manager(policy=policy)
    s1 = run_step(mgr, view, {"a": ([0], 0, 4)}, mm={"k": torch.full((4, 2), 1.0)})
    mgr.materialize(s1, ["a"])
    tid = mgr._deferred_tasks["a"].tid
    mgr._controller._tasks.pop(tid)
    s2 = run_step(mgr, view, {"b": ([0, 1], 4, 2)}, new_hits={"b": 4}, mm={"k": torch.full((2, 2), 9.0)})
    with pytest.raises(OmniPrefixCacheUnmatchError):
        mgr.materialize(s2, ["b"])


def test_lock_never_covers_fetch_or_join():
    """State lock must not be held across join_host_ready or fetch_host."""
    policy = ModelCachePolicy(needs_full_hidden_states=True, deferred_keys=frozenset({"k"}))
    mgr, view = make_manager(policy=policy)
    calls = []

    def probe(kind):
        on_facade = not threading.current_thread().name.startswith("omni-prefix-cache-prefetch")
        calls.append((kind, on_facade and mgr._state_lock.locked()))

    real_fetch = mgr._controller.fetch_host
    real_join = mgr._controller.join_host_ready
    mgr._controller.fetch_host = lambda *a, **kw: (probe("fetch"), real_fetch(*a, **kw))[1]
    mgr._controller.join_host_ready = lambda ids: (probe("join"), real_join(ids))[1]

    s1 = run_step(mgr, view, {"a": ([0], 0, 4)}, mm={"k": torch.full((4, 2), 1.0)})
    mgr.materialize(s1, ["a"])
    s2 = run_step(mgr, view, {"b": ([0, 1], 4, 2)}, new_hits={"b": 4}, mm={"k": torch.full((2, 2), 9.0)})
    mgr.materialize(s2, ["b"])
    assert any(kind == "fetch" for kind, _ in calls)
    assert any(kind == "join" for kind, _ in calls)
    assert all(not locked for _, locked in calls), calls


def test_failed_write_fails_fast_at_next_facade_entry():
    mgr, view = make_manager()
    sid = run_step(mgr, view, {"a": ([0], 0, 4)})
    mgr.materialize(sid, ["a"])
    from vllm_omni.core.prefix_cache.controller import WriteTask, _Segment

    task = WriteTask(
        tid=999,
        req_id="x",
        write_n=1,
        schedule=WriteSchedule.JOIN_NEXT_STEP,
        segments=[_Segment(slots_cpu=torch.tensor([0]), tensors={})],
    )
    mgr._controller._tasks[999] = task
    mgr._controller._fail_task(999)
    with pytest.raises(OmniPrefixCacheUnmatchError, match="write failed"):
        run_step(mgr, view, {"a": ([0, 1], 4, 1)})


def test_per_request_staging_writes():
    mgr, view = make_manager()
    sid = run_step(mgr, view, {"p": ([0, 1], 0, 8), "d": ([2], 0, 1)})
    tp = mgr._controller.get_task(next(iter(mgr._req_tasks["p"])))
    td = mgr._controller.get_task(next(iter(mgr._req_tasks["d"])))
    assert tp is not None and td is not None
    assert (tp.req_id, td.req_id) == ("p", "d")
    assert mgr._join_next_step_tids == [tp.tid, td.tid]
    outs = mgr.materialize(sid, ["p", "d"])
    assert torch.equal(outs.hidden_states["p"], expected_rows(view.slots_for("p", 0, 8)))
    assert torch.equal(outs.hidden_states["d"], expected_rows(view.slots_for("d", 0, 1)))


def test_staging_step_prefills_task_host_and_recycles():
    mgr, view = make_manager()
    for i in range(6):
        sid = run_step(mgr, view, {"a": ([i % 8, (i % 8) + 8], i, 1)})
        ctx = mgr._step_ctxs[sid]
        assert ctx.d2h is not None
        expect = expected_rows(view.slots_for("a", i, i + 1))
        outs = mgr.materialize(sid, ["a"])
        assert torch.equal(outs.hidden_states["a"], expect), i


def test_oversized_step_fails_fast():
    mgr, view = make_manager(staging_capacity_tokens=4)
    with pytest.raises(OmniPrefixCacheUnmatchError, match="staging capacity is 4"):
        run_step(mgr, view, {"a": ([0, 1], 0, 8)})


def test_staging_task_slot_held_until_drain():
    mgr, view = make_manager()
    sid = run_step(mgr, view, {"p": ([0, 1], 0, 8)})
    ctx = mgr._step_ctxs[sid]
    tid = next(iter(mgr._req_tasks["p"]))
    assert ctx.d2h is not None
    busy = mgr._controller._staging_pool._busy[ctx.d2h.slot]
    assert StagingBufferHolder.for_task(tid) in busy and StagingBufferHolder.for_step(sid) in busy
    mgr.materialize(sid, ["p"])
    assert not busy


def test_hit_prefetch_prebuilds_merged_buffer():
    mgr, view = make_manager()
    s1 = run_step(mgr, view, {"a": ([0, 1], 0, 8)})
    mgr.materialize(s1, ["a"])
    s2 = run_step(mgr, view, {"b": ([0, 1, 2], 8, 4)}, new_hits={"b": 8}, finished=["a"])
    ctx = mgr._step_ctxs[s2]
    fut = ctx.hit_prefetch["b"][HIDDEN_KEY]
    buf = fut.result()
    assert buf.shape == (12, HIDDEN)
    assert torch.equal(buf[:8], expected_rows(view.slots_for("b", 0, 8)))
    merged = mgr.materialize(s2, ["b"]).hidden_states["b"]
    assert merged.data_ptr() == buf.data_ptr()
    assert torch.equal(merged[8:], expected_rows(view.slots_for("b", 8, 12)))


def test_same_step_hit_skips_prefetch(caplog):
    mgr, view = make_manager()
    view.req_blocks["a"] = [0, 1]
    with caplog.at_level(logging.CRITICAL, logger="vllm_omni.core.prefix_cache.manager"):
        sid = run_step(mgr, view, {"a": ([0, 1], 0, 8), "b": ([0, 1, 2], 8, 4)}, new_hits={"b": 8})
    assert not any("omni prefix cache unmatch" in r.message for r in caplog.records)
    ctx = mgr._step_ctxs[sid]
    assert HIDDEN_KEY not in ctx.hit_prefetch.get("b", {})
    outs = mgr.materialize(sid, ["a", "b"])
    assert torch.equal(outs.hidden_states["b"][:8], expected_rows(view.slots_for("b", 0, 8)))


def test_join_next_step_hit_survives_task_already_drained():
    mgr, view = make_manager()

    def register_only(task):
        with mgr._controller._lock:
            mgr._controller._tasks[task.tid] = task

    mgr._controller.submit = register_only
    view.req_blocks["a"] = [0, 1]
    sid = run_step(mgr, view, {"a": ([0, 1], 0, 8), "b": ([0, 1, 2], 8, 4)}, new_hits={"b": 8})
    with mgr._state_lock:
        src = mgr._plan_hit_rows("b", 8, [0, 1], HIDDEN_KEY, strict=True)
    for tid in src.join_tids:
        mgr._controller._run_eager(mgr._controller.get_task(tid))
    with torch.inference_mode():
        with mgr._state_lock:
            mgr._commit_drained_writes()
        assert mgr._controller.get_task(src.join_tids[0]) is None
        assert torch.equal(mgr._fetch_source(src), expected_rows(view.slots_for("b", 0, 8)))
    mgr.materialize(sid, ["a", "b"])


def test_leftover_mm_snapshot_survives_live_overwrite():
    """Uncached passthrough is copied at save; mutating the live buffer
    afterwards must not change the snapshot."""
    mgr, view = make_manager()
    live = [torch.arange(10, dtype=DTYPE)]
    sid = run_step(mgr, view, {"a": ([0], 0, 2)}, mm={"codes.ref": live})
    assert "codes.ref" in mgr._step_ctxs[sid].mm_cpu_snapshot
    snap = mgr._step_ctxs[sid].mm_cpu_snapshot["codes.ref"]
    live[0].fill_(99.0)
    assert torch.equal(snap[0], torch.arange(10, dtype=DTYPE))
    mgr.discard_step(sid)


def test_deferred_leftover_snapshot_survives_live_overwrite():
    """Deferred tails are leftover-copied at save for materialize; the GPU
    freeze is a different product (pool write on finish)."""
    policy = ModelCachePolicy(needs_full_hidden_states=True, deferred_keys=frozenset({"codes.audio"}))
    mgr, view = make_manager(policy=policy)
    live = torch.full((2, 2), 1.0)
    sid = run_step(mgr, view, {"a": ([0], 0, 2)}, mm={"codes.audio": live})
    assert "codes.audio" in mgr._step_ctxs[sid].mm_cpu_snapshot
    live.fill_(99.0)
    outs = mgr.materialize(sid, ["a"])
    assert torch.equal(outs.mm_outputs["codes.audio"]["a"], torch.full((2, 2), 1.0))


def test_frozen_mm_clone_survives_live_overwrite():
    mgr, view = make_manager()
    live = torch.full((2, 2), 1.0)
    sid = run_step(mgr, view, {"a": ([0], 0, 2)}, mm={"codes.audio": live})
    assert "codes.audio" not in mgr._step_ctxs[sid].mm_cpu_snapshot
    live.fill_(99.0)
    outs = mgr.materialize(sid, ["a"])
    assert torch.equal(outs.mm_outputs["codes.audio"]["a"], torch.full((2, 2), 1.0))


def test_staging_slot_released_on_no_consumer_early_return():
    policy = ModelCachePolicy(needs_full_hidden_states=False)
    mgr, view = make_manager(policy=policy)
    for i in range(6):
        sid = run_step(mgr, view, {"a": ([i % 8, (i % 8) + 8], i, 1)}, mm={"k": torch.full((1, 2), float(i))})
        ctx = mgr._step_ctxs[sid]
        assert ctx.d2h is not None, i
        mgr.materialize(sid, ["a"])
        assert not mgr._controller._staging_pool._busy[ctx.d2h.slot], i


def test_save_releases_staging_if_commit_drained_writes_fails():
    mgr, view = make_manager(staging_depth=2)
    calls = {"n": 0}
    real = mgr._commit_drained_writes

    def wrapped():
        calls["n"] += 1
        if calls["n"] == 2:
            raise OmniPrefixCacheUnmatchError("injected fail")
        return real()

    mgr._commit_drained_writes = wrapped
    with pytest.raises(OmniPrefixCacheUnmatchError, match="injected fail"):
        run_step(mgr, view, {"a": ([0], 0, 4)})
    assert all(not busy for busy in mgr._controller._staging_pool._busy)
    mgr._commit_drained_writes = real
    sid = run_step(mgr, view, {"a": ([0], 0, 4)})
    assert mgr._step_ctxs[sid].d2h is not None
    mgr.materialize(sid, ["a"])


def test_fetch_host_maps_slots_across_layouts():
    """Identity, prefix, gapped, and a two-segment gather."""
    from vllm_omni.core.prefix_cache.block_pool import PrefixBlockPool
    from vllm_omni.core.prefix_cache.controller import OmniPrefixCacheController, WriteTask, _Segment

    cfg = PrefixCacheConfig(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)
    pool = PrefixBlockPool(cfg)
    pool.ensure_key(HIDDEN_KEY, DTYPE, HIDDEN)
    ctrl = OmniPrefixCacheController(pool, cfg, eager=True)

    slots = torch.tensor([40, 41, 42, 43, 44], dtype=torch.int64)
    rows = torch.arange(slots.numel() * HIDDEN, dtype=DTYPE).reshape(slots.numel(), HIDDEN)

    def _expect(want):
        return torch.stack([rows[(slots == s).nonzero()[0, 0]] for s in want.tolist()])

    one = WriteTask(
        tid=1,
        req_id="r",
        write_n=1,
        schedule=WriteSchedule.JOIN_ON_FINISH,
        segments=[_Segment(slots_cpu=slots, tensors={HIDDEN_KEY: rows})],
    )
    for want in (slots, slots[:3], slots[[0, 1, 3, 4]]):
        assert torch.equal(ctrl.fetch_host(one, want, HIDDEN_KEY), _expect(want))

    two = WriteTask(
        tid=2,
        req_id="r",
        write_n=2,
        schedule=WriteSchedule.JOIN_ON_FINISH,
        segments=[
            _Segment(slots_cpu=slots[:3], tensors={HIDDEN_KEY: rows[:3]}),
            _Segment(slots_cpu=slots[3:], tensors={HIDDEN_KEY: rows[3:]}),
        ],
    )
    want = slots[[0, 3, 1, 4]]
    assert torch.equal(ctrl.fetch_host(two, want, HIDDEN_KEY), _expect(want))


def test_fetch_host_waits_staging_step_d2h_event():
    """JOIN_NEXT_STEP hangs seg.host as a staging view; fetch_host waits
    step_d2h_event before slicing. Production JOIN_NEXT_STEP hits join scatter
    instead; this is the unit-level wait contract for that branch."""
    from vllm_omni.core.prefix_cache.block_pool import PrefixBlockPool
    from vllm_omni.core.prefix_cache.controller import OmniPrefixCacheController, WriteTask, _Segment

    cfg = PrefixCacheConfig(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)
    pool = PrefixBlockPool(cfg)
    ctrl = OmniPrefixCacheController(pool, cfg, eager=True)

    slots = torch.tensor([0, 1, 2], dtype=torch.int64)
    src = torch.arange(3 * HIDDEN, dtype=DTYPE).reshape(3, HIDDEN)
    landing = torch.zeros_like(src)

    class _HostEvent:
        n = 0

        def synchronize(self):
            self.n += 1
            landing.copy_(src)

    event = _HostEvent()
    seg = _Segment(slots_cpu=slots, tensors={HIDDEN_KEY: src})
    seg.host = {HIDDEN_KEY: landing}
    task = WriteTask(
        tid=1,
        req_id="r",
        write_n=1,
        schedule=WriteSchedule.JOIN_NEXT_STEP,
        segments=[seg],
        staging_slot=0,
        step_d2h_event=event,
    )
    rows = ctrl.fetch_host(task, slots, HIDDEN_KEY)
    assert event.n == 1
    assert torch.equal(rows, src)


def test_from_vllm_config_uses_batched_tokens():
    cfg = PrefixCacheConfig.from_vllm_config(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192, max_model_len=32768, max_num_seqs=64),
    )
    assert cfg.staging_capacity_tokens == 8192
    assert cfg.staging_depth == 4
    cfg_fallback = PrefixCacheConfig.from_vllm_config(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=None, max_model_len=4096),
    )
    assert cfg_fallback.staging_capacity_tokens == 4096
