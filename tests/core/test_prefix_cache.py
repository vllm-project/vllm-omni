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
import time
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
    for _pkg in ("vllm_omni", "vllm_omni.core", "vllm_omni.utils"):
        if _pkg not in sys.modules:
            _m = __import__("types").ModuleType(_pkg)
            _m.__path__ = [str(_root / _pkg.replace(".", "/"))]
            sys.modules[_pkg] = _m
    _vllm = __import__("types").ModuleType("vllm")
    _vllm_logger = __import__("types").ModuleType("vllm.logger")
    _vllm_logger.init_logger = logging.getLogger
    _vllm.logger = _vllm_logger
    sys.modules["vllm"] = _vllm
    sys.modules["vllm.logger"] = _vllm_logger

from vllm_omni.core.prefix_cache.controller import StagingBufferHolder
from vllm_omni.core.prefix_cache.group_view import (
    FullAttentionGroupView,
    check_prefix_cache_kv_groups,
    stage_prefix_cache_config,
)
from vllm_omni.core.prefix_cache.interface import (
    HIDDEN_KEY,
    ModelCachePolicy,
    OmniPrefixCacheStagingTimeoutError,
    OmniPrefixCacheUnmatchError,
    PrefixCacheConfig,
    WriteSchedule,
)
from vllm_omni.core.prefix_cache.manager import (
    OmniPrefixCacheManager,
    _is_step_token_tensor,
    _snapshot_leftover_mm_cpu,
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


def plan_fetch(mgr, slots, key, *, req_id):
    with mgr._state_lock:
        src = mgr._slot_ref(slots, key, req_id)
    with torch.inference_mode():
        return mgr._fetch_source(src)


def _assert_leftover_shapes(inp, got, n: int) -> None:
    """Every leftover tensor keeps its shape unless dim0 is this step's token count."""
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


@pytest.mark.parametrize("a_finished_before_b", [False, True])
def test_partial_match_hits_shared_prefix_and_writes_divergent_block(a_finished_before_b):
    """`a` occupies [0, 1, 2]; `b` shares the first two blocks and diverges on
    the third ([0, 1, 3]). `b` must merge 8 hit rows from the shared blocks
    plus its own 4 rows, land block 3 in the pool, and leave `a`'s block 2
    untouched — whether `a` already finished or is still running."""
    mgr, view = make_manager()
    s1 = run_step(mgr, view, {"a": ([0, 1, 2], 0, 12)})
    mgr.materialize(s1, ["a"])
    a_tail = view.slots_for("a", 8, 12)
    assert torch.equal(plan_fetch(mgr, a_tail, HIDDEN_KEY, req_id="a"), expected_rows(a_tail))

    finished = ["a"] if a_finished_before_b else ()
    s2 = run_step(mgr, view, {"b": ([0, 1, 3], 8, 4)}, new_hits={"b": 8}, finished=finished)
    outs = mgr.materialize(s2, ["b"])
    merged = outs.hidden_states["b"]
    assert merged.shape == (12, HIDDEN)
    assert torch.equal(merged[:8], expected_rows(view.slots_for("b", 0, 8)))
    assert torch.equal(merged[8:], expected_rows(view.slots_for("b", 8, 12)))
    # Divergent third block landed; the shared prefix and a's own tail did not move.
    b_tail = view.slots_for("b", 8, 12)
    assert torch.equal(plan_fetch(mgr, b_tail, HIDDEN_KEY, req_id="b"), expected_rows(b_tail))
    assert torch.equal(plan_fetch(mgr, a_tail, HIDDEN_KEY, req_id="a"), expected_rows(a_tail))
    shared = view.slots_for("a", 0, 8)
    assert torch.equal(plan_fetch(mgr, shared, HIDDEN_KEY, req_id="a"), expected_rows(shared))

    if not a_finished_before_b:
        # `a` finishes later; releasing its task must not disturb b's rows.
        s3 = run_step(mgr, view, {"b": ([0, 1, 3, 4], 12, 1)}, finished=["a"])
        mgr.materialize(s3, ["b"])
        assert torch.equal(plan_fetch(mgr, b_tail, HIDDEN_KEY, req_id="b"), expected_rows(b_tail))


def test_live_req_reentering_new_reqs_is_not_a_hit():
    """async_chunk continuation: the same id re-enters scheduled_new_reqs with
    num_computed_tokens equal to what it computed itself. No hit span is
    opened and materialize returns only this step's rows (parity with the
    pre-refactor cache; a delivered_upto span is Phase 2). Once the id has
    finished, the same id is a new request again and its hit is merged."""
    mgr, view = make_manager()
    s1 = run_step(mgr, view, {"a": ([0, 1], 0, 8)})
    mgr.materialize(s1, ["a"])

    # Next chunk arrives: `a` is scheduled again as a "new" request at 8.
    s2 = run_step(mgr, view, {"a": ([0, 1, 2], 8, 4)}, new_hits={"a": 8})
    assert not mgr._step_ctxs[s2].hits
    outs = mgr.materialize(s2, ["a"])
    assert torch.equal(outs.hidden_states["a"], expected_rows(view.slots_for("a", 8, 12)))

    # `a` finishes; a later request reusing the id is new and does hit.
    s3 = run_step(mgr, view, {"b": ([3], 0, 2)}, finished=["a"])
    mgr.materialize(s3, ["b"])
    s4 = run_step(mgr, view, {"a": ([0, 1, 4], 8, 4)}, new_hits={"a": 8}, finished=["b"])
    assert mgr._step_ctxs[s4].hits == {"a": (8, [0, 1])}
    outs = mgr.materialize(s4, ["a"])
    assert outs.hidden_states["a"].shape == (12, HIDDEN)
    assert torch.equal(outs.hidden_states["a"][:8], expected_rows(view.slots_for("a", 0, 8)))


def test_join_next_step_hit_waits_done_then_reads_pool():
    """CPU stand-in for the non-eager path: dispatch holds the task instead of
    writing the pool. A same-step hit must join(done), drain, then read the pool."""
    mgr, view = make_manager()
    real_run = mgr._controller._run_eager
    real_join = mgr._controller.join

    def join_then_scatter(tids):
        # Stand-in committer: the write lands only when someone joins it.
        for tid in tids:
            task = mgr._controller.get_task(tid)
            if task is not None and not task.done.is_set():
                real_run(task)
        real_join(tids)

    mgr._controller.dispatch = lambda tasks: None
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
        assert not mgr._controller._staging_pool._busy[d2h.staging_slot]


def test_hit_not_block_aligned_fails_at_register():
    mgr, view = make_manager()
    s1 = run_step(mgr, view, {"a": ([0, 1], 0, 8)})
    mgr.materialize(s1, ["a"])
    with pytest.raises(OmniPrefixCacheUnmatchError, match="prefix hit not block aligned"):
        run_step(mgr, view, {"b": ([0, 1, 2], 8, 4)}, new_hits={"b": 6})


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
    """Without a deferred policy, codes.audio is an ordinary mm key (dim0 = tokens)."""
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


def test_split_step_outputs_routes_immediate_deferred_leftover():
    """One split: immediate device→host, deferred write + leftover read, leftover only."""
    policy = ModelCachePolicy(needs_full_hidden_states=True, deferred_keys=frozenset({"codes.audio"}))
    mgr, view = make_manager(policy=policy)
    view.order = ["a"]
    view.req_blocks["a"] = [0]
    view.computed["a"] = 0
    n = 4
    mgr.new_step_starts(FakeSchedOut(new_reqs=[FakeNewReq("a")], num_scheduled={"a": n}))
    hidden = torch.ones(n, HIDDEN)
    mm = {
        "codes.audio": torch.full((n, 2), 5.0),
        "codes.ref": torch.zeros(15, 2),
        "tok": torch.full((n, 2), 7.0),
    }
    step_outputs = mgr._split_step_outputs(
        hidden,
        mm,
        n,
        n,
        slots_cpu=view.step_slots_cpu(["a"], {"a": n}),
        req_order=["a"],
        num_sched={"a": n},
        query_start={"a": 0},
    )
    assert HIDDEN_KEY in step_outputs.immediate and "tok" in step_outputs.immediate
    assert "codes.audio" not in step_outputs.immediate and "codes.ref" not in step_outputs.immediate
    assert "codes.audio" in step_outputs.leftover and "codes.ref" in step_outputs.leftover
    assert "tok" not in step_outputs.leftover
    assert len(step_outputs.deferred_chunks) == 1
    assert step_outputs.deferred_chunks[0][0] == "a"
    assert "codes.audio" in step_outputs.deferred_chunks[0][1].tensors


def test_leftover_snapshot_preserves_non_token_major_shapes():
    """P1: leftover copy must not use `shape[0] >= n` as a slice predicate.

    Tensors whose first dim is this step's token count (`shape[0] == n`)
    may be sliced; every other tensor — including `codes.ref` with
    ref_len >> n, and list-held tensors — must keep the input shape.
    """
    n = 4
    mm = {
        "codes.ref": torch.arange(30, dtype=DTYPE).reshape(15, 2),
        "tags": [torch.ones(3, 5), torch.zeros(7)],
        "tokenish": torch.arange(n * 2, dtype=DTYPE).reshape(n, 2),
        "scalar": torch.tensor(3.0),
    }
    out, event = _snapshot_leftover_mm_cpu(mm, set(), n)
    assert event is None
    _assert_leftover_shapes(mm, out, n)
    assert torch.equal(out["codes.ref"], mm["codes.ref"])
    assert out["codes.ref"].shape == (15, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA leftover copy")
def test_leftover_snapshot_cuda_is_pinned_and_event_ordered():
    """CUDA leftover lands pinned via a non-blocking copy; values are exact once the event is waited."""
    n = 4
    mm = {
        "codes.audio": torch.arange(n * 2, dtype=DTYPE, device="cuda").reshape(n, 2),
        "codes.ref": torch.arange(30, dtype=DTYPE, device="cuda").reshape(15, 2),
        "tags": [torch.ones(3, 5, device="cuda")],
    }
    out, event = _snapshot_leftover_mm_cpu(mm, set(), n)
    assert event is not None
    event.synchronize()
    _assert_leftover_shapes(mm, out, n)
    for t in (out["codes.audio"], out["codes.ref"], out["tags"][0]):
        assert t.device.type == "cpu" and t.is_pinned()
    assert torch.equal(out["codes.audio"], mm["codes.audio"].cpu())
    assert torch.equal(out["codes.ref"], mm["codes.ref"].cpu())


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


def test_is_step_token_tensor():
    n, padded = 4, 8
    assert _is_step_token_tensor(torch.zeros(n, 2), n, padded)
    assert _is_step_token_tensor(torch.zeros(padded, 2), n, padded)
    assert not _is_step_token_tensor([torch.zeros(1, 8)], n, padded)
    assert not _is_step_token_tensor(torch.zeros(15, 2), n, padded)
    assert not _is_step_token_tensor(torch.tensor([1, 2, 3]), n, padded)


def test_policy_from_model_shim():
    class M:
        requires_full_prefix_cached_hidden_states = False
        deferred_prefix_cache_mm_keys = {"codes.audio"}

    p = ModelCachePolicy.from_model(M())
    assert p.needs_full_hidden_states is False
    assert p.hidden_key is None
    assert p.deferred_keys == frozenset({"codes.audio"})
    assert p.get_hit_keys([HIDDEN_KEY, "codes.audio"]) == ["codes.audio"]
    assert p.skip_immediate_mm("codes.audio")
    d = ModelCachePolicy.from_model(object())
    assert d.needs_full_hidden_states is True and not d.deferred_keys
    assert d.hidden_key == HIDDEN_KEY
    assert d.get_hit_keys(["talker.h", HIDDEN_KEY]) == [HIDDEN_KEY, "talker.h"]


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
    rows = plan_fetch(mgr, slots, "codes.audio", req_id="a")
    assert torch.equal(rows[:, 0], torch.tensor([1.0, 2.0]))
    sid = run_step(mgr, view, {"z": ([9], 0, 1)}, finished=["a"])
    mgr.materialize(sid, ["z"])
    mirror = mgr._pool.rows("codes.audio", slots)
    assert torch.equal(mirror[:, 0], torch.tensor([1.0, 2.0]))


def test_deferred_gpu_bytes_held_until_last_view_drops():
    """Shared deferred clone is one allocation; finishing the first
    co-scheduled request must not drop ``_staged_bytes`` while the other
    still pins it."""
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
    """One long request's deferred clone must not pin the whole budget:
    the flush closes its open task and the next step opens a new one."""
    policy = ModelCachePolicy(needs_full_hidden_states=False, deferred_keys=frozenset({"k"}))
    mgr, view = make_manager(policy=policy, gpu_staging_bytes=48)  # three (1, 4) fp32 chunks
    for pos in range(4):
        sid = run_step(mgr, view, {"a": ([2, 3], pos, 1)}, mm={"k": torch.full((1, 4), float(pos + 1))})
        mgr.materialize(sid, ["a"])
    # Fourth chunk flushed the first three; only it is still charged.
    assert mgr._controller._staged_bytes == 16
    rows = mgr._pool.rows("k", view.slots_for("a", 0, 4))[:, 0].tolist()
    assert rows == [1.0, 2.0, 3.0, 0.0]
    assert mgr._request_tasks.deferred["a"].write_n == 2
    assert len(mgr._request_tasks.tasks["a"]) == 1  # the flushed write was drained


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


def test_reacquired_slot_deferred_rewrite_wins_over_interim_tenant():
    """Preempt + resume onto the same block: the rewrite must open a new
    write, or the old task's reassigned tombstone drops the fresh rows and
    the interim tenant's stale value is served as a's."""
    policy = ModelCachePolicy(needs_full_hidden_states=False, deferred_keys=frozenset({"k"}))
    mgr, view = make_manager(policy=policy)
    s1 = run_step(mgr, view, {"a": ([2], 0, 1)}, mm={"k": torch.full((1, 2), 11.0)})
    mgr.materialize(s1, ["a"])
    slots = view.slots_for("a", 0, 1)
    s2 = run_step(mgr, view, {"b": ([2], 0, 1)}, mm={"k": torch.full((1, 2), 22.0)})
    mgr.materialize(s2, ["b"])
    s3 = run_step(mgr, view, {"z": ([9], 0, 1)}, finished=["b"])  # 22 committed
    mgr.materialize(s3, ["z"])
    s4 = run_step(mgr, view, {"a": ([2], 0, 1)}, mm={"k": torch.full((1, 2), 33.0)})  # a re-claims
    mgr.materialize(s4, ["a"])
    s5 = run_step(mgr, view, {"z2": ([9], 1, 1)}, finished=["a"])  # flush a's write
    mgr.materialize(s5, ["z2"])
    assert torch.equal(mgr._pool.rows("k", slots), torch.full((1, 2), 33.0))


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
    """A deferred task closed by the budget flush (or an early escalate)
    cannot take more chunks; the next save opens a new write and a hit
    reads both."""
    policy = ModelCachePolicy(needs_full_hidden_states=False, deferred_keys=frozenset({"k"}))
    mgr, view = make_manager(policy=policy)
    s1 = run_step(mgr, view, {"a": ([2], 0, 1)}, mm={"k": torch.full((1, 2), 1.0)})
    mgr.materialize(s1, ["a"])
    first = mgr._request_tasks.deferred["a"]
    mgr._controller.escalate([first.tid])
    s2 = run_step(mgr, view, {"a": ([2], 1, 1)}, mm={"k": torch.full((1, 2), 2.0)})
    mgr.materialize(s2, ["a"])
    assert mgr._request_tasks.deferred["a"].tid != first.tid
    slots = view.slots_for("a", 0, 2)
    rows = plan_fetch(mgr, slots, "k", req_id="a")
    assert torch.equal(rows[:, 0], torch.tensor([1.0, 2.0]))


def test_deferred_unpadded_registers_on_padded_step():
    """Qwen3-TTS: deferred codes.audio is unpadded; cudagraph pads hidden."""
    policy = ModelCachePolicy(needs_full_hidden_states=False, deferred_keys=frozenset({"codes.audio"}))
    mgr, view = make_manager(policy=policy)
    audio = torch.arange(8 * 2, dtype=DTYPE).reshape(8, 2)
    s1 = run_step(mgr, view, {"a": ([0, 1], 0, 8)}, mm={"codes.audio": audio}, num_tokens_padded=16)
    assert mgr._pool.has_key("codes.audio")
    assert "codes.audio" in mgr._step_ctxs[s1].mm_cpu_snapshot
    rows = plan_fetch(mgr, view.slots_for("a", 0, 8), "codes.audio", req_id="a")
    assert torch.equal(rows, audio)
    outs = mgr.materialize(s1, ["a"])
    assert torch.equal(outs.mm_outputs["codes.audio"]["a"], audio)


def test_check_kv_groups_rejects_empty_or_multi():
    with pytest.raises(OmniPrefixCacheUnmatchError, match="single full-attention"):
        check_prefix_cache_kv_groups([])
    with pytest.raises(OmniPrefixCacheUnmatchError, match="single full-attention"):
        check_prefix_cache_kv_groups([object(), object()])


def _stage_cfg(*, enable=True, pooling=False, kv_transfer=None, groups=(object(),), spec=None, match_unit=None):
    return stage_prefix_cache_config(
        kv_cache_config=SimpleNamespace(num_blocks=NUM_BLOCKS, kv_cache_groups=list(groups)),
        cache_config=SimpleNamespace(enable_prefix_caching=enable, block_size=BLOCK_SIZE, prefix_match_unit=match_unit),
        kv_transfer_config=kv_transfer,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64, max_model_len=128),
        model_config=None,
        is_pooling_model=pooling,
        speculative_config=spec,
    )


def test_stage_prefix_cache_config_gate():
    """The runner-side gate the GPU and NPU runners share: off / pooling
    stages get no cache; kv_consumer, spec decode, sub-block matching and
    hybrid groups refuse loudly."""
    assert _stage_cfg(enable=False) is None
    assert _stage_cfg(pooling=True) is None
    # kv_consumer / kv_both refuse before the group check, on both platforms.
    with pytest.raises(OmniPrefixCacheUnmatchError, match="KV connector"):
        _stage_cfg(kv_transfer=SimpleNamespace(is_kv_consumer=True), groups=())
    assert _stage_cfg(kv_transfer=SimpleNamespace(is_kv_consumer=True), enable=False) is None
    with pytest.raises(OmniPrefixCacheUnmatchError, match="speculative decoding"):
        _stage_cfg(spec=SimpleNamespace(method="ngram"), groups=())
    with pytest.raises(OmniPrefixCacheUnmatchError, match="block-aligned"):
        _stage_cfg(match_unit=BLOCK_SIZE // 2, groups=())
    # A unit equal to the block is a whole-block match; only the group check is left.
    with pytest.raises(OmniPrefixCacheUnmatchError, match="single full-attention"):
        _stage_cfg(match_unit=BLOCK_SIZE, groups=())
    with pytest.raises(OmniPrefixCacheUnmatchError, match="single full-attention"):
        _stage_cfg(groups=(object(), object()))


def test_npu_runner_uses_shared_prefix_cache_gate():
    """npu_model_runner cannot be imported without vllm_ascend; pin at the
    source level that it stages through the shared gate instead of a
    hand-rolled subset of the GPU checks."""
    path = Path(__file__).resolve().parents[2] / "vllm_omni/platforms/npu/worker/npu_model_runner.py"
    tree = ast.parse(path.read_text())
    calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "stage_prefix_cache_config" in calls
    assert "check_prefix_cache_kv_groups" not in calls


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
    assert int(mgr._slot_status.state[HIDDEN_KEY][view.slots_for("a", 0, 2)].min()) == 2
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


def test_scatter_rows_coalesces_per_key_last_chunk_wins_and_skips_reassigned():
    from vllm_omni.core.prefix_cache.controller import WriteSchedule, WriteTask, _BudgetTicket, _WriteChunk

    t1, t2 = _BudgetTicket(nbytes=8), _BudgetTicket(nbytes=8)

    def chunk(slots, base, ticket):
        s = torch.tensor(slots, dtype=torch.int64)
        host = {"k": torch.full((len(slots), 2), float(base)) + s[:, None]}
        return _WriteChunk(slots_cpu=s, tensors={}, host=host, budget=ticket)

    task = WriteTask(
        tid=1,
        req_id="a",
        write_n=1,
        schedule=WriteSchedule.JOIN_ON_FINISH,
        chunks=[chunk([0, 1], 10, t1), chunk([2, 3], 20, t1), chunk([1, 4], 30, t2)],
    )
    assert task.budget_tickets() == [t1, t2]
    task.add_reassigned("k", torch.tensor([3], dtype=torch.int64))
    rows = task.scatter_rows()
    assert len(rows) == 1
    key, slots, host = rows[0]
    assert key == "k" and slots.tolist() == [0, 2, 1, 4]
    # slot 1: the third chunk's row (base 30) replaces the first chunk's; slot 3 is reassigned.
    assert host[:, 0].tolist() == [10.0, 22.0, 31.0, 34.0]


def test_slot_reuse_pushes_skip_to_old_task():
    """Reassignment = task swap. Keep the old write in progress so the
    new write records those rows as no longer owned by it."""
    mgr, view = make_manager()

    def hold(tasks):
        for task in tasks:
            task.host_ready.set()
            task.done.set()  # satisfy the next save's done barrier; state stays QUEUED

    mgr._controller.dispatch = hold
    policy = ModelCachePolicy(needs_full_hidden_states=False)
    mgr.register_policy(policy)
    s1 = run_step(mgr, view, {"a": ([2], 0, 1)}, mm={"k": torch.ones(1, 2)})
    mgr.materialize(s1, ["a"])
    old_tid = next(iter(mgr._request_tasks.tasks["a"]))
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


def _ib_with_layout(**layout):
    """InputBatch fake whose group-0 block table carries vLLM's layout attrs."""
    attrs = dict(kv_cache_block_size=BLOCK_SIZE, blocks_per_kv_block=1, use_hybrid_blocks=False, dcp_world_size=1)
    attrs.update(layout)
    group = SimpleNamespace(block_table=SimpleNamespace(cpu=torch.zeros((1, 2), dtype=torch.int32)), **attrs)

    class BT:
        block_tables = [group.block_table]

        def __getitem__(self, idx):
            return group

    return SimpleNamespace(req_ids=[], req_id_to_index={}, num_computed_tokens_cpu=torch.zeros(1), block_table=BT())


def test_group_view_refuses_hybrid_kernel_blocks_and_dcp():
    """step_slots_cpu assumes one allocator block per table column and no
    token striping; both are silent wrong-slot writes if not refused."""
    FullAttentionGroupView(_ib_with_layout(), block_size=BLOCK_SIZE)
    with pytest.raises(OmniPrefixCacheUnmatchError, match="kernel_block_size"):
        FullAttentionGroupView(_ib_with_layout(use_hybrid_blocks=True, blocks_per_kv_block=2), block_size=BLOCK_SIZE)
    with pytest.raises(OmniPrefixCacheUnmatchError, match="decode context parallel"):
        FullAttentionGroupView(_ib_with_layout(dcp_world_size=2), block_size=BLOCK_SIZE)
    with pytest.raises(OmniPrefixCacheUnmatchError, match="does not match"):
        FullAttentionGroupView(_ib_with_layout(kv_cache_block_size=BLOCK_SIZE * 2), block_size=BLOCK_SIZE)


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


def test_unconsumed_contexts_timeout():
    mgr, view = make_manager(staging_claim_timeout_s=0.0)
    with pytest.raises(OmniPrefixCacheStagingTimeoutError, match="unconsumed step contexts"):
        for pos in range(8):
            run_step(mgr, view, {"a": ([0, 1, 2, 3], pos, 1)})


def test_save_waits_until_a_slot_is_consumed():
    mgr, view = make_manager(staging_depth=2, staging_claim_timeout_s=2.0)
    s0 = run_step(mgr, view, {"a": ([0], 0, 1)})
    s1 = run_step(mgr, view, {"a": ([0], 1, 1)})
    err: list[BaseException] = []
    sid_holder: list[int] = []

    def _blocked_save() -> None:
        try:
            sid_holder.append(run_step(mgr, view, {"a": ([0], 2, 1)}))
        except BaseException as e:
            err.append(e)

    t = threading.Thread(target=_blocked_save, daemon=True)
    t.start()
    time.sleep(0.1)
    assert t.is_alive()
    mgr.discard_step(s0)
    t.join(timeout=1.0)
    assert not t.is_alive()
    assert not err
    mgr.discard_step(s1)
    mgr.discard_step(sid_holder[0])


def test_leftover_only_save_claims_staging_slot():
    policy = ModelCachePolicy(needs_full_hidden_states=False)
    mgr, view = make_manager(policy=policy, staging_depth=2, staging_claim_timeout_s=0.0)
    leftover = {"codes.ref": torch.zeros(3, 2)}
    s0 = run_step(mgr, view, {"a": ([0], 0, 1)}, mm=leftover)
    s1 = run_step(mgr, view, {"a": ([0], 1, 1)}, mm=leftover)
    assert mgr._step_ctxs[s0].d2h is not None and mgr._step_ctxs[s0].d2h.views == {}
    assert mgr._step_ctxs[s1].d2h is not None and mgr._step_ctxs[s1].d2h.views == {}
    with pytest.raises(OmniPrefixCacheStagingTimeoutError, match="unconsumed step contexts"):
        run_step(mgr, view, {"a": ([0], 2, 1)}, mm=leftover)


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
    with pytest.raises(OmniPrefixCacheUnmatchError, match="outside the save snapshot"):
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
    tid = mgr._request_tasks.deferred["a"].tid
    mgr._controller._tasks.pop(tid)
    s2 = run_step(mgr, view, {"b": ([0, 1], 4, 2)}, new_hits={"b": 4}, mm={"k": torch.full((2, 2), 9.0)})
    with pytest.raises(OmniPrefixCacheUnmatchError):
        mgr.materialize(s2, ["b"])


def test_lock_never_covers_fetch_or_join():
    """State lock must not be held across join / join_host_ready / fetch_host."""
    policy = ModelCachePolicy(needs_full_hidden_states=True, deferred_keys=frozenset({"k"}))
    mgr, view = make_manager(policy=policy)
    calls = []

    def probe(kind):
        on_facade = not threading.current_thread().name.startswith("omni-prefix-cache-prefetch")
        calls.append((kind, on_facade and mgr._state_lock.locked()))

    real_fetch = mgr._controller.fetch_host
    real_join_hr = mgr._controller.join_host_ready
    real_join = mgr._controller.join
    mgr._controller.fetch_host = lambda *a, **kw: (probe("fetch"), real_fetch(*a, **kw))[1]
    mgr._controller.join_host_ready = lambda ids: (probe("join"), real_join_hr(ids))[1]
    mgr._controller.join = lambda ids: (probe("join"), real_join(ids))[1]

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
    from vllm_omni.core.prefix_cache.controller import WriteTask, _WriteChunk

    task = WriteTask(
        tid=999,
        req_id="x",
        write_n=1,
        schedule=WriteSchedule.JOIN_NEXT_STEP,
        chunks=[_WriteChunk(slots_cpu=torch.tensor([0]), tensors={})],
    )
    mgr._controller._tasks[999] = task
    mgr._controller._fail_task(999)
    with pytest.raises(OmniPrefixCacheUnmatchError, match="write failed"):
        run_step(mgr, view, {"a": ([0, 1], 4, 1)})
    # Sticky: the record is one-shot and the first raise may land in a
    # swallowed path (prefetch Future dropped by discard_step); a later
    # entry must keep raising, never serve zeros for the lost rows.
    with pytest.raises(OmniPrefixCacheUnmatchError, match="write failed"):
        run_step(mgr, view, {"c": ([0, 1], 0, 4)})


def test_join_raises_on_failed_task():
    """A joiner woken by FAILED must not treat the wake as a completed
    write; the one-shot failure record may already be drained elsewhere."""
    mgr, view = make_manager()
    _hold_writes(mgr)
    run_step(mgr, view, {"a": ([0], 0, 4)})
    tid = next(iter(mgr._request_tasks.tasks["a"]))
    mgr._controller._fail_task(tid)
    with pytest.raises(OmniPrefixCacheUnmatchError, match="failed before done"):
        mgr._controller.join([tid])


@pytest.mark.parametrize("fail_at", ["copy", "scatter"])
def test_eager_dispatch_failure_releases_step_and_all_task_owners(monkeypatch, fail_at):
    from vllm_omni.core.prefix_cache.controller import TaskState, WriteTask

    mgr, view = make_manager(staging_claim_timeout_s=0)
    ctrl = mgr._controller
    preserved_sid = run_step(mgr, view, {"previous": ([3], 0, 4)})
    if fail_at == "copy":
        original = WriteTask.mark_host_ready

        def fail_copy(task):
            if task.req_id == "b":
                raise RuntimeError("injected copy failure")
            return original(task)

        monkeypatch.setattr(WriteTask, "mark_host_ready", fail_copy)
    else:
        original = ctrl._scatter

        def fail_scatter(task):
            if task.req_id == "b":
                raise RuntimeError("injected scatter failure")
            return original(task)

        monkeypatch.setattr(ctrl, "_scatter", fail_scatter)

    try:
        with pytest.raises(RuntimeError, match=f"injected {fail_at} failure"):
            run_step(mgr, view, {"a": ([0], 0, 4), "b": ([1], 0, 4), "c": ([2], 0, 4)})
        assert set(mgr._step_ctxs) == {preserved_sid}
        assert ctrl._staged_bytes == 0
        tasks = [task for task in ctrl._tasks.values() if task.req_id in {"a", "b", "c"}]
        assert len(tasks) == 3
        for task in tasks:
            expected = TaskState.WRITTEN if task.req_id == "a" else TaskState.FAILED
            assert task.state is expected
            assert task.done.is_set() and task.host_ready.is_set()
        assert all(
            holder == StagingBufferHolder.for_step(preserved_sid)
            for holders in ctrl._staging_pool._busy
            for holder in holders
        )
        mgr.discard_step(preserved_sid)
        assert all(not holders for holders in ctrl._staging_pool._busy)
        with pytest.raises(OmniPrefixCacheUnmatchError, match="write failed"):
            mgr.new_step_starts(FakeSchedOut())
    finally:
        mgr.shutdown()


def test_eager_escalation_failure_releases_deferred_tasks(monkeypatch):
    from vllm_omni.core.prefix_cache.controller import TaskState

    mgr, view = make_manager(policy=ModelCachePolicy(needs_full_hidden_states=False, deferred_keys=frozenset({"k"})))
    ctrl = mgr._controller
    sid = run_step(mgr, view, {"a": ([0], 0, 4), "b": ([1], 0, 4)}, mm={"k": torch.ones(8, 2)})
    mgr.discard_step(sid)
    assert ctrl._staged_bytes > 0
    tasks = list(ctrl._tasks.values())

    def fail_scatter(task):
        raise RuntimeError("injected deferred scatter failure")

    monkeypatch.setattr(ctrl, "_scatter", fail_scatter)
    try:
        with pytest.raises(RuntimeError, match="injected deferred scatter failure"):
            ctrl.escalate([task.tid for task in tasks])
        assert ctrl._staged_bytes == 0
        assert all(task.state is TaskState.FAILED for task in tasks)
        assert all(task.done.is_set() and task.host_ready.is_set() for task in tasks)
        assert all(not holders for holders in ctrl._staging_pool._busy)
        with pytest.raises(OmniPrefixCacheUnmatchError, match="write failed"):
            mgr.new_step_starts(FakeSchedOut())
    finally:
        mgr.shutdown()


def test_per_request_staging_writes():
    mgr, view = make_manager()
    sid = run_step(mgr, view, {"p": ([0, 1], 0, 8), "d": ([2], 0, 1)})
    tp = mgr._controller.get_task(next(iter(mgr._request_tasks.tasks["p"])))
    td = mgr._controller.get_task(next(iter(mgr._request_tasks.tasks["d"])))
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


def test_staging_slot_eager_save_leaves_only_step_holder():
    """Eager submit writes the pool inline, so the task holder is bound and
    released inside save; materialize then empties the slot."""
    mgr, view = make_manager()
    sid = run_step(mgr, view, {"p": ([0, 1], 0, 8)})
    ctx = mgr._step_ctxs[sid]
    assert ctx.d2h is not None
    busy = mgr._controller._staging_pool._busy[ctx.d2h.staging_slot]
    assert busy == {StagingBufferHolder.for_step(sid)}
    mgr.materialize(sid, ["p"])
    assert not busy


def _hold_writes(mgr) -> None:
    """Non-eager stand-in: dispatch takes the task to device→host done, but
    leaves the pool write to the caller (`mgr._controller._scatter`)."""

    def dispatch(tasks):
        for task in tasks:
            assert task.claim_copy()
            task.mark_host_ready()

    mgr._controller.dispatch = dispatch


def test_eager_copy_runs_outside_state_lock_and_key_install_inside():
    """Eager mode (NPU/CPU): the copy + pool write must not run under
    _state_lock, or materialize on the builder thread waits a whole D2H;
    publishing a new pool key must happen under it."""
    policy = ModelCachePolicy(needs_full_hidden_states=True, deferred_keys=frozenset({"k"}))
    mgr, view = make_manager(policy=policy)
    seen: dict[str, bool] = {}

    def probe(name):
        free = mgr._state_lock.acquire(blocking=False)
        if free:
            mgr._state_lock.release()
        seen[name] = free

    real_run, real_install = mgr._controller._run_eager, mgr._pool.install_key

    def run_eager(task):
        probe("run_eager")
        real_run(task)

    def install_key(key, storage):
        probe("install_key")
        real_install(key, storage)

    mgr._controller._run_eager = run_eager
    mgr._pool.install_key = install_key
    sid = run_step(mgr, view, {"a": ([0], 0, 2)}, mm={"k": torch.ones(2, 2)})
    assert seen == {"run_eager": True, "install_key": False}
    mgr.materialize(sid, ["a"])
    assert mgr._pool.has_key("k") and "k" in mgr._slot_status.state


def test_eager_finish_escalate_runs_outside_state_lock():
    """Finish/abort escalate must not _run_eager under _state_lock."""
    policy = ModelCachePolicy(needs_full_hidden_states=False, deferred_keys=frozenset({"k"}))
    mgr, view = make_manager(policy=policy)
    s1 = run_step(mgr, view, {"a": ([2], 0, 1)}, mm={"k": torch.ones(1, 2)})
    mgr.materialize(s1, ["a"])
    seen: list[bool] = []
    real_run = mgr._controller._run_eager

    def run_eager(task):
        free = mgr._state_lock.acquire(blocking=False)
        if free:
            mgr._state_lock.release()
        seen.append(free)
        real_run(task)

    mgr._controller._run_eager = run_eager
    s2 = run_step(mgr, view, {"z": ([9], 0, 1)}, finished=["a"])
    assert seen == [True]
    mgr.materialize(s2, ["z"])


def test_staging_task_slot_released_at_pool_write_not_drain():
    mgr, view = make_manager()
    _hold_writes(mgr)
    sid = run_step(mgr, view, {"p": ([0, 1], 0, 8)})
    ctx = mgr._step_ctxs[sid]
    tid = next(iter(mgr._request_tasks.tasks["p"]))
    busy = mgr._controller._staging_pool._busy[ctx.d2h.staging_slot]
    assert StagingBufferHolder.for_task(tid) in busy and StagingBufferHolder.for_step(sid) in busy
    task = mgr._controller.get_task(tid)
    mgr._controller._scatter(task)
    # Released by the committer's pool write; the manager has not drained yet.
    assert busy == {StagingBufferHolder.for_step(sid)}
    assert mgr._controller.get_task(tid) is not None
    mgr.materialize(sid, ["p"])
    assert not busy
    assert mgr._controller.get_task(tid) is None


def test_blocked_save_is_released_by_pool_write_alone():
    """All step contexts consumed, the only slot held by a pending write:
    the committer finishing that write must unblock the waiting save."""
    mgr, view = make_manager(staging_depth=1, staging_claim_timeout_s=2.0)
    _hold_writes(mgr)
    s0 = run_step(mgr, view, {"a": ([0], 0, 1)})
    tid = next(iter(mgr._request_tasks.tasks["a"]))
    mgr.materialize(s0, ["a"])
    slot = mgr._step_ctxs.get(s0)  # consumed
    assert slot is None
    assert mgr._controller._staging_pool._busy[0] == {StagingBufferHolder.for_task(tid)}
    err: list[BaseException] = []
    sid_holder: list[int] = []

    def _blocked_save() -> None:
        try:
            sid_holder.append(run_step(mgr, view, {"a": ([0], 1, 1)}))
        except BaseException as e:
            err.append(e)

    t = threading.Thread(target=_blocked_save, daemon=True)
    t.start()
    time.sleep(0.1)
    assert t.is_alive()
    mgr._controller._scatter(mgr._controller.get_task(tid))
    t.join(timeout=1.0)
    assert not t.is_alive()
    assert not err
    mgr.discard_step(sid_holder[0])


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


def test_same_step_hit_prefetch_starts_at_save(caplog):
    """b hits blocks a computes in the same step: nothing to plan at
    new_step_starts (rows ABSENT), planned at publish once a's write is
    registered, so the gather runs before the next step can reuse them."""
    mgr, view = make_manager()
    view.req_blocks["a"] = [0, 1]
    with caplog.at_level(logging.CRITICAL, logger="vllm_omni.core.prefix_cache.manager"):
        sid = run_step(mgr, view, {"a": ([0, 1], 0, 8), "b": ([0, 1, 2], 8, 4)}, new_hits={"b": 8})
    assert not any("omni prefix cache unmatch" in r.message for r in caplog.records)
    ctx = mgr._step_ctxs[sid]
    fut = ctx.hit_prefetch["b"][HIDDEN_KEY]
    assert torch.equal(fut.result()[:8], expected_rows(view.slots_for("b", 0, 8)))
    outs = mgr.materialize(sid, ["a", "b"])
    assert torch.equal(outs.hidden_states["b"][:8], expected_rows(view.slots_for("b", 0, 8)))


@pytest.mark.parametrize("reuse_blocks", [False, True], ids=["fresh", "reused"])
@pytest.mark.parametrize("deferred_mm", [False, True], ids=["immediate", "deferred"])
def test_same_step_hit_refreshes_prefetch_after_write(reuse_blocks, deferred_mm):
    policy = ModelCachePolicy(deferred_keys=frozenset({"mm"}) if deferred_mm else frozenset())
    mgr, view = make_manager(policy=policy)
    try:
        blocks = [0, 1] if reuse_blocks else [8, 9]
        sid = run_step(mgr, view, {"old": (blocks, 0, 8)}, mm={"mm": torch.full((8, 2), 100.0)})
        mgr.materialize(sid, ["old"])
        mgr.new_step_starts(FakeSchedOut(finished=["old"]))

        mgr.new_step_starts(
            FakeSchedOut(
                new_reqs=[FakeNewReq("a", 0, [[0, 1]]), FakeNewReq("b", 8, [[0, 1, 2]])],
                num_scheduled={"a": 8, "b": 4},
            )
        )
        old_prefetch = dict(mgr._hit_prefetch["b"])
        # Complete the early reads before A publishes the prefix B actually hits.
        for future in old_prefetch.values():
            future.result(timeout=5)

        view.order = ["a", "b"]
        view.req_blocks.update(a=[0, 1], b=[0, 1, 2])
        view.computed.update(a=0, b=8)
        hidden = torch.cat([torch.full((8, HIDDEN), 20.0), torch.full((4, HIDDEN), 30.0)])
        mm = torch.cat([torch.full((8, 2), 200.0), torch.full((4, 2), 300.0)])
        sid = mgr.save_outputs(hidden, {"mm": mm}, num_tokens_unpadded=12, num_tokens_padded=12)
        outs = mgr.materialize(sid, ["a", "b"])

        assert torch.equal(outs.hidden_states["b"][:8], hidden[:8])
        assert torch.equal(outs.hidden_states["b"][8:], hidden[8:])
        assert torch.equal(outs.mm_outputs["mm"]["b"], mm)
    finally:
        mgr.shutdown()


def test_delayed_read_of_reassigned_hit_raises():
    """A version check with no COW copy still raises for live and finished.
    The production path registers the ref first so remap preserves the rows
    (see test_delayed_read_after_committed_reuse_serves_preserved)."""
    mgr, view = make_manager()
    s1 = run_step(mgr, view, {"a": ([0, 1], 0, 8)})
    mgr.materialize(s1, ["a"])
    slots = view.slots_for("a", 0, 8)
    with mgr._state_lock:
        planned = mgr._slot_status.get_slot_status(HIDDEN_KEY).slot_version[slots].clone()
    mgr._controller.dispatch = lambda tasks: None  # new tenant's write stays IN_TRANSIT
    s2 = run_step(mgr, view, {"c": ([0, 1], 0, 8)}, finished=["a"])
    for req in ("c", "a"):  # live then finished — both fatal without a preserved copy
        with pytest.raises(OmniPrefixCacheUnmatchError, match="reassigned"):
            mgr._ensure_not_reassigned(slots, HIDDEN_KEY, req_id=req, planned_version=planned)
    mgr.discard_step(s2)


def test_delayed_read_after_committed_reuse_serves_preserved():
    """Amy's block-0 reuse: plan A's read, then B commits over the same
    slots. The reusing write copy-on-writes A's committed rows into the
    planned ref; the delayed fetch must return those, not B's."""
    mgr, view = make_manager()
    sa = run_step(mgr, view, {"a": ([0], 0, 4)})
    mgr.materialize(sa, ["a"])  # A committed into block 0
    slots = view.slots_for("a", 0, 4)
    with mgr._state_lock:
        src = mgr._slot_ref(slots, HIDDEN_KEY, "a")  # plan A's read
    assert src.already_staged and not src.staged_list and not src.join_tids
    sb = run_step(mgr, view, {"b": ([0], 0, 4)}, finished=["a"])
    mgr.materialize(sb, ["b"])  # A freed, block 0 reused by B and COMMITTED
    # run_step derives rows from slot ids, so A's and B's values coincide.
    # Scribble the pool so only the preserved copy can produce A's rows.
    assert set(src.preserved) == {int(s) for s in slots.tolist()}
    with torch.inference_mode():
        mgr._pool.write(HIDDEN_KEY, slots, torch.full((4, HIDDEN), 99.0, dtype=DTYPE))
        got = mgr._fetch_source(src)
    assert torch.equal(got, expected_rows(slots))
    assert all(ref is not src for ref in mgr._pending_reads)


def test_done_fires_only_after_completion_record_published():
    """A join(done) waiter (the save barrier) must find the completion record
    on its next drain — done-before-append leaves occupancy IN_TRANSIT and the
    COW preserve skips the reused rows."""
    mgr, view = make_manager()
    checks = []
    orig_scatter = mgr._controller._scatter

    def scatter(task):
        orig_set = task.done.set

        def checked_set():
            checks.append(task.tid in mgr._controller._completed)
            orig_set()

        task.done.set = checked_set
        orig_scatter(task)

    mgr._controller._scatter = scatter
    sid = run_step(mgr, view, {"a": ([0], 0, 4)})
    mgr.materialize(sid, ["a"])
    assert checks and all(checks)


def test_failed_wakes_only_after_failure_record_published():
    """Same window as the done/_completed one, failure side: a joiner woken
    by FAILED must find the record on drain, not read never-written rows."""
    mgr, view = make_manager()
    _hold_writes(mgr)  # keep the write non-terminal
    run_step(mgr, view, {"a": ([0], 0, 4)})
    tid = next(iter(mgr._request_tasks.tasks["a"]))
    task = mgr._controller.get_task(tid)
    checks = []
    orig_set = task.done.set

    def checked_set():
        checks.append(tid in mgr._controller._failed)
        orig_set()

    task.done.set = checked_set
    mgr._controller._fail_task(tid)
    assert checks and all(checks)


def test_save_waits_pool_write_of_previous_step():
    """A save must not reuse slots whose JOIN_NEXT_STEP write has not reached
    the pool — no delayed read could recover such a row."""
    mgr, view = make_manager(staging_claim_timeout_s=0.2)
    _hold_writes(mgr)  # host_ready only; the pool write never happens
    run_step(mgr, view, {"a": ([0], 0, 4)})
    with pytest.raises(OmniPrefixCacheUnmatchError, match="did not reach done"):
        run_step(mgr, view, {"b": ([1], 0, 4)})


def test_join_next_step_hit_survives_task_already_drained():
    mgr, view = make_manager()
    mgr._controller.dispatch = lambda tasks: None  # registered, never run
    view.req_blocks["a"] = [0, 1]
    sid = run_step(mgr, view, {"a": ([0, 1], 0, 8), "b": ([0, 1, 2], 8, 4)}, new_hits={"b": 8})
    with mgr._state_lock:
        src = mgr._slot_ref(view.slots_for("b", 0, 8), HIDDEN_KEY, "b")
    for tid in src.join_tids:
        mgr._controller._run_eager(mgr._controller.get_task(tid))
    with torch.inference_mode():
        with mgr._state_lock:
            mgr._commit_drained_writes()
        assert mgr._controller.get_task(src.join_tids[0]) is None
        assert torch.equal(mgr._fetch_source(src), expected_rows(view.slots_for("b", 0, 8)))
    mgr.materialize(sid, ["a", "b"])


def test_leftover_mm_snapshot_survives_live_overwrite():
    """Leftover mm is copied at save; mutating the live buffer
    afterwards must not change the snapshot."""
    mgr, view = make_manager()
    live = [torch.arange(10, dtype=DTYPE)]
    sid = run_step(mgr, view, {"a": ([0], 0, 2)}, mm={"codes.ref": live})
    assert "codes.ref" in mgr._step_ctxs[sid].mm_cpu_snapshot
    snap = mgr._step_ctxs[sid].mm_cpu_snapshot["codes.ref"]
    live[0].fill_(99.0)
    assert torch.equal(snap[0], torch.arange(10, dtype=DTYPE))
    mgr.discard_step(sid)


@pytest.mark.parametrize("needs_full_hidden_states", [True, False])
def test_deferred_leftover_snapshot_survives_live_overwrite(needs_full_hidden_states):
    """Deferred tails are leftover-copied at save for materialize; the GPU
    freeze is a different product (pool write on finish). ``False`` is the
    hidden opt-out + miss path (Qwen3-TTS / Higgs talker)."""
    policy = ModelCachePolicy(
        needs_full_hidden_states=needs_full_hidden_states, deferred_keys=frozenset({"codes.audio"})
    )
    mgr, view = make_manager(policy=policy)
    live = torch.full((2, 2), 1.0)
    ref = [torch.arange(4, dtype=DTYPE)]
    sid = run_step(mgr, view, {"a": ([0], 0, 2)}, mm={"codes.audio": live, "codes.ref": ref})
    assert "codes.audio" in mgr._step_ctxs[sid].mm_cpu_snapshot
    live.fill_(99.0)
    ref[0].fill_(99.0)
    outs = mgr.materialize(sid, ["a"])
    assert torch.equal(outs.mm_outputs["codes.audio"]["a"], torch.full((2, 2), 1.0))
    assert torch.equal(outs.mm_outputs["codes.ref"]["a"][0], torch.arange(4, dtype=DTYPE))
    assert (outs.hidden_states is None) == (not needs_full_hidden_states)


def test_frozen_mm_clone_survives_live_overwrite():
    mgr, view = make_manager()
    live = torch.full((2, 2), 1.0)
    sid = run_step(mgr, view, {"a": ([0], 0, 2)}, mm={"codes.audio": live})
    assert "codes.audio" not in mgr._step_ctxs[sid].mm_cpu_snapshot
    live.fill_(99.0)
    outs = mgr.materialize(sid, ["a"])
    assert torch.equal(outs.mm_outputs["codes.audio"]["a"], torch.full((2, 2), 1.0))


def test_hidden_opt_out_miss_returns_mm_and_releases_slot():
    policy = ModelCachePolicy(needs_full_hidden_states=False)
    mgr, view = make_manager(policy=policy)
    for i in range(6):
        sid = run_step(mgr, view, {"a": ([i % 8, (i % 8) + 8], i, 1)}, mm={"k": torch.full((1, 2), float(i))})
        ctx = mgr._step_ctxs[sid]
        assert ctx.d2h is not None, i
        outs = mgr.materialize(sid, ["a"])
        assert outs.hidden_states is None
        assert torch.equal(outs.mm_outputs["k"]["a"], torch.full((1, 2), float(i)))
        assert not mgr._controller._staging_pool._busy[ctx.d2h.staging_slot], i


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
    """Identity, prefix, gapped, and a two-`_WriteChunk` gather."""
    from vllm_omni.core.prefix_cache.block_pool import PrefixBlockPool
    from vllm_omni.core.prefix_cache.controller import OmniPrefixCacheController, WriteTask, _WriteChunk

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
        chunks=[_WriteChunk(slots_cpu=slots, tensors={HIDDEN_KEY: rows})],
    )
    for want in (slots, slots[:3], slots[[0, 1, 3, 4]]):
        assert torch.equal(ctrl.fetch_host(one, want, HIDDEN_KEY), _expect(want))

    two = WriteTask(
        tid=2,
        req_id="r",
        write_n=2,
        schedule=WriteSchedule.JOIN_ON_FINISH,
        chunks=[
            _WriteChunk(slots_cpu=slots[:3], tensors={HIDDEN_KEY: rows[:3]}),
            _WriteChunk(slots_cpu=slots[3:], tensors={HIDDEN_KEY: rows[3:]}),
        ],
    )
    want = slots[[0, 3, 1, 4]]
    assert torch.equal(ctrl.fetch_host(two, want, HIDDEN_KEY), _expect(want))


def test_fetch_host_waits_staging_step_d2h_event():
    """JOIN_NEXT_STEP stores chunk.host as a staging view; fetch_host waits
    step_d2h_event before slicing. Production JOIN_NEXT_STEP hits wait for
    the pool write instead; this is the unit-level wait contract for that branch."""
    from vllm_omni.core.prefix_cache.block_pool import PrefixBlockPool
    from vllm_omni.core.prefix_cache.controller import OmniPrefixCacheController, WriteTask, _WriteChunk

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
    chunk = _WriteChunk(slots_cpu=slots, tensors={HIDDEN_KEY: src})
    chunk.host = {HIDDEN_KEY: landing}
    task = WriteTask(
        tid=1,
        req_id="r",
        write_n=1,
        schedule=WriteSchedule.JOIN_NEXT_STEP,
        chunks=[chunk],
        staging_slot=0,
        step_d2h_event=event,
    )
    rows = ctrl.fetch_host(task, slots, HIDDEN_KEY)
    assert event.n == 1
    assert torch.equal(rows, src)


def _bare_controller(eager=True):
    from vllm_omni.core.prefix_cache.block_pool import PrefixBlockPool
    from vllm_omni.core.prefix_cache.controller import OmniPrefixCacheController

    cfg = PrefixCacheConfig(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)
    pool = PrefixBlockPool(cfg)
    pool.ensure_key(HIDDEN_KEY, DTYPE, HIDDEN)
    ctrl = OmniPrefixCacheController(pool, cfg, eager=eager)
    return ctrl, pool


def _bare_task(tid=7, schedule=WriteSchedule.JOIN_NEXT_STEP, host=None, tensors=None, budget=None, queued=False):
    from vllm_omni.core.prefix_cache.controller import TaskState, WriteTask, _WriteChunk

    slots = torch.tensor([0, 1], dtype=torch.int64)
    chunk = _WriteChunk(slots_cpu=slots, tensors=tensors or {}, budget=budget)
    if host is not None:
        chunk.host = {HIDDEN_KEY: host}
    task = WriteTask(tid=tid, req_id="r", write_n=1, schedule=schedule, chunks=[chunk])
    if queued:
        task.transition(TaskState.QUEUED)
    return task, slots


def test_task_state_is_a_strict_chain():
    """PENDING -> QUEUED -> COPYING -> HOST_READY -> WRITTEN, one step at a time."""
    from vllm_omni.core.prefix_cache.controller import TaskState

    task, _ = _bare_task()
    assert task.state is TaskState.PENDING
    assert not task.claim_copy()  # PENDING may not skip QUEUED
    with pytest.raises(OmniPrefixCacheUnmatchError, match="illegal transition PENDING -> HOST_READY"):
        task.transition(TaskState.HOST_READY)
    task.transition(TaskState.QUEUED)
    assert task.claim_copy()
    assert task.state is TaskState.COPYING and not task.host_ready.is_set()
    assert not task.claim_copy()  # second claimant loses
    task.mark_host_ready()
    assert task.state is TaskState.HOST_READY and task.host_ready.is_set() and not task.done.is_set()
    with pytest.raises(OmniPrefixCacheUnmatchError, match="illegal transition HOST_READY -> COPYING"):
        task.transition(TaskState.COPYING)  # no going back
    task.mark_done()
    assert task.state is TaskState.WRITTEN and task.done.is_set()
    assert not task.mark_failed()  # terminal stays terminal
    assert task.state is TaskState.WRITTEN


def test_task_failed_sets_both_events_and_closes_append():
    from vllm_omni.core.prefix_cache.controller import TaskState, _WriteChunk

    task, slots = _bare_task()
    assert task.mark_failed()
    assert task.state is TaskState.FAILED and task.host_ready.is_set() and task.done.is_set()
    assert task.append_chunk(_WriteChunk(slots_cpu=slots, tensors={})) is TaskState.FAILED


def test_escalate_leaves_claimed_task_alone():
    """Once the worker owns the copy stage (COPYING), escalate must not
    re-queue the tid — that re-queue was the double-write race."""
    from vllm_omni.core.prefix_cache.controller import TaskState

    ctrl, _ = _bare_controller(eager=True)
    ctrl._eager = False  # exercise the queue path without a worker thread
    later, _ = _bare_task(tid=1, schedule=WriteSchedule.JOIN_ON_FINISH)
    ctrl.submit(later, queued=False)
    assert later.state is TaskState.PENDING
    ctrl.escalate([1])
    assert later.state is TaskState.QUEUED and list(ctrl._queue_hi) == [1]
    ctrl.escalate([1])  # already high-priority: no duplicate
    assert list(ctrl._queue_hi) == [1]

    ctrl._queue_hi.clear()
    later.transition(TaskState.COPYING)
    ctrl.escalate([1])
    assert list(ctrl._queue_hi) == [] and list(ctrl._queue_lo) == []


def test_join_on_stuck_committer_raises_instead_of_hanging():
    from vllm_omni.core.prefix_cache.block_pool import PrefixBlockPool
    from vllm_omni.core.prefix_cache.controller import OmniPrefixCacheController, TaskState

    cfg = PrefixCacheConfig(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE, staging_claim_timeout_s=0.05)
    ctrl = OmniPrefixCacheController(PrefixBlockPool(cfg), cfg, eager=True)
    ctrl._eager = False  # queue path without a worker thread: nothing will finish the task
    task, _ = _bare_task(tid=1, schedule=WriteSchedule.JOIN_ON_FINISH)
    ctrl.submit(task, queued=True)
    task.transition(TaskState.COPYING)

    with pytest.raises(
        OmniPrefixCacheUnmatchError, match=r"task 1 \(join_on_finish\) did not reach host_ready .*state=COPYING"
    ):
        ctrl.join_host_ready([1])
    with pytest.raises(OmniPrefixCacheUnmatchError, match="did not reach done"):
        ctrl.join([1])


def test_scatter_host_ready_writes_only_staged_tasks():
    from vllm_omni.core.prefix_cache.controller import TaskState

    ctrl, pool = _bare_controller()
    task, slots = _bare_task(host=torch.ones(2, HIDDEN), queued=True)
    ctrl._tasks[task.tid] = task
    assert task.claim_copy()
    ctrl._blocked.append(task.tid)
    ctrl._scatter_host_ready()  # COPYING: not ready, stays blocked
    assert ctrl._blocked == [task.tid] and list(ctrl._completed) == []
    task.mark_host_ready()
    ctrl._scatter_host_ready()
    assert ctrl._blocked == [] and list(ctrl._completed) == [task.tid]
    assert task.state is TaskState.WRITTEN
    assert torch.equal(pool.rows(HIDDEN_KEY, slots), torch.ones(2, HIDDEN))


def test_budget_ticket_uncharges_once_per_clone():
    from vllm_omni.core.prefix_cache.controller import _BudgetTicket

    ctrl, _ = _bare_controller()
    ticket = _BudgetTicket(nbytes=100)
    ctrl.reserve(100)
    a, _ = _bare_task(tid=1, budget=ticket)
    b, _ = _bare_task(tid=2, budget=ticket)
    ctrl.pin_budget(ticket, 1)
    ctrl.pin_budget(ticket, 2)
    ctrl._release_staged_bytes(a)
    assert ctrl._staged_bytes == 100  # b still holds the clone
    ctrl._release_staged_bytes(a)  # idempotent per tid
    assert ctrl._staged_bytes == 100
    ctrl._release_staged_bytes(b)
    assert ctrl._staged_bytes == 0
    ctrl._release_staged_bytes(b)  # freed once, never twice
    assert ctrl._staged_bytes == 0


def test_fail_task_after_copy_does_not_uncharge_twice():
    from vllm_omni.core.prefix_cache.controller import TaskState, _BudgetTicket

    ctrl, _ = _bare_controller()
    ticket = _BudgetTicket(nbytes=64)
    ctrl.reserve(64)
    task, _ = _bare_task(tid=3, budget=ticket, queued=True)
    ctrl._tasks[task.tid] = task
    ctrl.pin_budget(ticket, 3)
    assert task.claim_copy()
    task.mark_host_ready()
    ctrl._release_staged_bytes(task)  # what _copy_task does on HOST_READY
    assert ctrl._staged_bytes == 0
    ctrl._fail_task(3)
    assert task.state is TaskState.FAILED and ctrl._staged_bytes == 0
    assert list(ctrl._failed) == [3]
    ctrl._fail_task(3)  # terminal: no second publish
    assert list(ctrl._failed) == [3]


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
