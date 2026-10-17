# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for prefix caching in vLLM Omni.

CPU-only and needs no vLLM runtime (runs without a vllm install via
pytest --confcutdir=tests/core).
"""

import ast
import logging
import sys
import weakref
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

from vllm_omni.core.prefix_cache.adapter import (
    PrefixCacheEventKind,
    PrefixCacheRequestEvent,
    PrefixCacheSchedulerAdapter,
    PrefixCacheStep,
)
from vllm_omni.core.prefix_cache.group_view import (
    FullAttentionGroupView,
    check_prefix_cache_kv_groups,
    stage_prefix_cache_config,
)
from vllm_omni.core.prefix_cache.interface import (
    HIDDEN_KEY,
    ModelCachePolicy,
    OmniPrefixCacheUnmatchError,
    PrefixCacheConfig,
)
from vllm_omni.core.prefix_cache.manager import OmniPrefixCacheManager, _is_per_token_tensor
from vllm_omni.core.prefix_cache.slot_index import PrefixHit, SavedOutputs, SlotIndex

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
    mgr = OmniPrefixCacheManager(config)
    if policy is not None:
        mgr.register_policy(policy)
    return mgr, view


def run_engine_step_prefix_update(
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
    adapter = getattr(mgr, "_test_adapter", None)
    if adapter is None:
        adapter = mgr._test_adapter = PrefixCacheSchedulerAdapter()
    step = adapter.translate_step(sched_out)
    mgr.new_step_starts(step)
    layout = adapter.build_write_layout(view, num_scheduled_tokens=dict(step.scheduled_tokens))
    n = int(view.step_slot_mapping.numel())
    padded = n if num_tokens_padded is None else int(num_tokens_padded)
    return mgr.save_outputs(hidden, mm or {}, num_tokens_unpadded=n, num_tokens_padded=padded, write_layout=layout)


def cached_rows(mgr, slots: torch.Tensor, key=HIDDEN_KEY, width=HIDDEN) -> torch.Tensor:
    """The rows the cache holds for these KV slots, read through a lookup of their blocks."""
    blocks = list(dict.fromkeys(int(slot) // BLOCK_SIZE for slot in slots.tolist()))
    rows = mgr._index.lookup(blocks).cat(key, torch.empty(0, width, dtype=DTYPE))
    position = {
        block * BLOCK_SIZE + offset: i * BLOCK_SIZE + offset
        for i, block in enumerate(blocks)
        for offset in range(BLOCK_SIZE)
    }
    return rows[[position[int(slot)] for slot in slots.tolist()]]


def expected_rows(slots: torch.Tensor) -> torch.Tensor:
    return slots.to(DTYPE).unsqueeze(1).expand(slots.numel(), HIDDEN)


@pytest.mark.parametrize("step_count,layout_count", [(6, 4), (2, 4), (4, 0)])
def test_save_rejects_layout_that_disagrees_with_step(step_count, layout_count):
    mgr, view = make_manager()
    mgr.merge_batch(run_engine_step_prefix_update(mgr, view, {"a": ([0, 1], 0, 8)}), ["a"])
    adapter = mgr._test_adapter
    mgr.new_step_starts(adapter.translate_step(FakeSchedOut(finished=["a"])))
    view.order = ["b"] if layout_count else []
    view.req_blocks["b"], view.computed["b"] = [0, 1, 2], 8
    hit = FakeNewReq("b", 8, [[0, 1, 2]])
    mgr.new_step_starts(adapter.translate_step(FakeSchedOut(new_reqs=[hit], num_scheduled={"b": step_count})))
    layout = adapter.build_write_layout(view, num_scheduled_tokens={"b": layout_count})
    rows = expected_rows(view.slots_for("b", 8, 8 + layout_count)).clone()
    with pytest.raises(OmniPrefixCacheUnmatchError, match="scheduled token count"):
        mgr.save_outputs(
            rows, {}, num_tokens_unpadded=layout_count, num_tokens_padded=layout_count, write_layout=layout
        )
    assert not mgr._batches  # Rejection must precede saving the batch.


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
    batch_id = run_engine_step_prefix_update(mgr, view, {"a": ([0, 1], 0, 8)})
    outs = mgr.merge_batch(batch_id, ["a"])
    assert torch.equal(outs.hidden_states["a"], expected_rows(view.slots_for("a", 0, 8)))


def test_hit_merge_from_mirror():
    mgr, view = make_manager()
    s1 = run_engine_step_prefix_update(mgr, view, {"a": ([0, 1], 0, 8)})
    mgr.merge_batch(s1, ["a"])
    s2 = run_engine_step_prefix_update(mgr, view, {"b": ([0, 1, 2], 8, 4)}, new_hits={"b": 8}, finished=["a"])
    outs = mgr.merge_batch(s2, ["b"])
    merged = outs.hidden_states["b"]
    assert merged.shape == (12, HIDDEN)
    assert torch.equal(merged[:8], expected_rows(view.slots_for("b", 0, 8)))
    assert torch.equal(merged[8:], expected_rows(view.slots_for("b", 8, 12)))


@pytest.mark.parametrize("a_finished_before_b", [False, True])
def test_partial_match_hits_shared_prefix_and_writes_divergent_block(a_finished_before_b):
    """`a` occupies [0, 1, 2]; `b` shares the first two blocks and diverges on
    the third ([0, 1, 3]). `b` must merge 8 hit rows from the shared blocks
    plus its own 4 rows, save block 3, and leave `a`'s block 2
    untouched — whether `a` already finished or is still running."""
    mgr, view = make_manager()
    s1 = run_engine_step_prefix_update(mgr, view, {"a": ([0, 1, 2], 0, 12)})
    mgr.merge_batch(s1, ["a"])
    a_tail = view.slots_for("a", 8, 12)
    assert torch.equal(cached_rows(mgr, a_tail), expected_rows(a_tail))

    finished = ["a"] if a_finished_before_b else ()
    s2 = run_engine_step_prefix_update(mgr, view, {"b": ([0, 1, 3], 8, 4)}, new_hits={"b": 8}, finished=finished)
    outs = mgr.merge_batch(s2, ["b"])
    merged = outs.hidden_states["b"]
    assert merged.shape == (12, HIDDEN)
    assert torch.equal(merged[:8], expected_rows(view.slots_for("b", 0, 8)))
    assert torch.equal(merged[8:], expected_rows(view.slots_for("b", 8, 12)))
    # Divergent third block landed; the shared prefix and a's own tail did not move.
    b_tail = view.slots_for("b", 8, 12)
    assert torch.equal(cached_rows(mgr, b_tail), expected_rows(b_tail))
    assert torch.equal(cached_rows(mgr, a_tail), expected_rows(a_tail))
    shared = view.slots_for("a", 0, 8)
    assert torch.equal(cached_rows(mgr, shared), expected_rows(shared))

    if not a_finished_before_b:
        # `a` finishes later; that must not disturb b's rows.
        s3 = run_engine_step_prefix_update(mgr, view, {"b": ([0, 1, 3, 4], 12, 1)}, finished=["a"])
        mgr.merge_batch(s3, ["b"])
        assert torch.equal(cached_rows(mgr, b_tail), expected_rows(b_tail))


def test_live_req_reentering_new_reqs_is_not_a_hit():
    """async_chunk continuation: the same id re-enters scheduled_new_reqs with
    num_computed_tokens equal to what it computed itself. No hit is
    registered and merge_batch returns only this step's rows. Once the id has
    finished, the same id is a new request again and its hit is merged."""
    mgr, view = make_manager()
    s1 = run_engine_step_prefix_update(mgr, view, {"a": ([0, 1], 0, 8)})
    mgr.merge_batch(s1, ["a"])

    # Next chunk arrives: `a` is scheduled again as a "new" request at 8.
    s2 = run_engine_step_prefix_update(mgr, view, {"a": ([0, 1, 2], 8, 4)}, new_hits={"a": 8})
    assert not mgr._batches[s2].hits
    outs = mgr.merge_batch(s2, ["a"])
    assert torch.equal(outs.hidden_states["a"], expected_rows(view.slots_for("a", 8, 12)))

    # `a` finishes; a later request reusing the id is new and does hit.
    s3 = run_engine_step_prefix_update(mgr, view, {"b": ([3], 0, 2)}, finished=["a"])
    mgr.merge_batch(s3, ["b"])
    s4 = run_engine_step_prefix_update(mgr, view, {"a": ([0, 1, 4], 8, 4)}, new_hits={"a": 8}, finished=["b"])
    assert set(mgr._batches[s4].hits) == {"a"}
    outs = mgr.merge_batch(s4, ["a"])
    assert outs.hidden_states["a"].shape == (12, HIDDEN)
    assert torch.equal(outs.hidden_states["a"][:8], expected_rows(view.slots_for("a", 0, 8)))


@pytest.mark.parametrize("terminal", ["finished", "aborted", "discarded"])
def test_same_step_terminal_id_reuse_merges_only_the_new_start(terminal):
    mgr, view = make_manager()
    first = run_engine_step_prefix_update(mgr, view, {"a": ([0], 0, 4)})
    mgr.merge_batch(first, ["a"])
    if terminal == "finished":
        second = run_engine_step_prefix_update(mgr, view, {"a": ([0, 1], 4, 1)}, new_hits={"a": 4}, finished=["a"])
    else:
        # The fake scheduler helper has no abort field, so construct the
        # explicit side channel as a real producer would supply it.
        adapter = mgr._test_adapter
        sched = FakeSchedOut(new_reqs=[FakeNewReq("a", 4, [[0, 1]])], num_scheduled={"a": 1})
        setattr(sched, f"{terminal}_req_ids", {"a"})
        mgr.new_step_starts(adapter.translate_step(sched))
        view.order = ["a"]
        view.req_blocks["a"] = [0, 1]
        view.computed["a"] = 4
        second = mgr.save_outputs(
            expected_rows(view.slots_for("a", 4, 5)),
            {},
            num_tokens_unpadded=1,
            num_tokens_padded=1,
            write_layout=adapter.build_write_layout(view, num_scheduled_tokens={"a": 1}),
        )
    assert set(mgr._batches[second].hits) == {"a"}
    merged = mgr.merge_batch(second, ["a"]).hidden_states["a"]
    assert torch.equal(merged, expected_rows(view.slots_for("a", 0, 5)))
    third = run_engine_step_prefix_update(mgr, view, {"a": ([0, 1], 5, 1)}, new_hits={"a": 4})
    assert not mgr._batches[third].hits
    assert mgr.merge_batch(third, ["a"]).hidden_states["a"].shape == (1, HIDDEN)


def test_same_step_hit_reads_in_transit():
    mgr, view = make_manager()
    view.req_blocks["a"] = [0, 1]
    batch_id = run_engine_step_prefix_update(
        mgr,
        view,
        {"a": ([0, 1], 0, 8), "b": ([0, 1, 2], 8, 4)},
        new_hits={"b": 8},
    )
    outs = mgr.merge_batch(batch_id, ["a", "b"])
    assert torch.equal(outs.hidden_states["b"][:8], expected_rows(view.slots_for("b", 0, 8)))


def test_absent_hit_fails_fast():
    mgr, view = make_manager()
    view.req_blocks["c"] = [5, 6]
    with pytest.raises(OmniPrefixCacheUnmatchError, match="not readable"):
        run_engine_step_prefix_update(mgr, view, {"c": ([5, 6, 7], 8, 4)}, new_hits={"c": 8})


def test_hit_not_block_aligned_fails_at_register():
    mgr, view = make_manager()
    s1 = run_engine_step_prefix_update(mgr, view, {"a": ([0, 1], 0, 8)})
    mgr.merge_batch(s1, ["a"])
    with pytest.raises(OmniPrefixCacheUnmatchError, match="prefix hit not block aligned"):
        run_engine_step_prefix_update(mgr, view, {"b": ([0, 1, 2], 8, 4)}, new_hits={"b": 6})


def test_mm_cached_key_merge():
    mgr, view = make_manager()
    feat = 3
    mm1 = {"talker.h": torch.arange(8 * feat, dtype=DTYPE).reshape(8, feat)}
    s1 = run_engine_step_prefix_update(mgr, view, {"a": ([0, 1], 0, 8)}, mm=mm1)
    mgr.merge_batch(s1, ["a"])
    mm2 = {"talker.h": torch.full((4, feat), 7.0)}
    s2 = run_engine_step_prefix_update(mgr, view, {"b": ([0, 1, 2], 8, 4)}, new_hits={"b": 8}, finished=["a"], mm=mm2)
    outs = mgr.merge_batch(s2, ["b"])
    merged = outs.mm_outputs["talker.h"]["b"]
    assert merged.shape == (12, feat)
    assert torch.equal(merged[:8], mm1["talker.h"])
    assert torch.equal(merged[8:], mm2["talker.h"])


def test_unpadded_mm_registers_on_padded_step():
    """Qwen3-TTS codes.audio is scheduled-length; cudagraph makes padded > n.

    Matching only `shape[0] == padded` would pass the rows through uncached
    and silently serve hits without a cached prefix.
    """
    mgr, view = make_manager()
    audio = torch.arange(8 * 2, dtype=DTYPE).reshape(8, 2)
    s1 = run_engine_step_prefix_update(
        mgr, view, {"a": ([0, 1], 0, 8)}, mm={"codes.audio": audio}, num_tokens_padded=16
    )
    assert "codes.audio" in mgr._batches[s1].outputs
    assert torch.equal(mgr.merge_batch(s1, ["a"]).mm_outputs["codes.audio"]["a"], audio)
    s2 = run_engine_step_prefix_update(
        mgr,
        view,
        {"b": ([0, 1, 2], 8, 4)},
        new_hits={"b": 8},
        finished=["a"],
        mm={"codes.audio": torch.full((4, 2), 3.0)},
        num_tokens_padded=16,
    )
    merged = mgr.merge_batch(s2, ["b"]).mm_outputs["codes.audio"]["b"]
    assert torch.equal(merged[:8], audio)
    assert torch.equal(merged[8:], torch.full((4, 2), 3.0))


def test_codes_ref_matches_old_build_mm_cpu_path():
    """Leftover `codes.ref` equals the old runner path: build_mm_cpu + to_payload_element."""
    pytest.importorskip("vllm")
    from vllm_omni.utils.mm_outputs import build_mm_cpu, to_payload_element

    refs = [torch.arange(8, dtype=torch.long), torch.arange(12, dtype=torch.long)]
    mgr, view = make_manager()
    batch_id = run_engine_step_prefix_update(mgr, view, {"a": ([0], 0, 2), "b": ([1], 0, 2)}, mm={"codes.ref": refs})
    step = mgr._batches[batch_id]
    assert [t.shape for t in step.other_mm["codes.ref"]] == [t.shape for t in refs]
    outs = mgr.merge_batch(batch_id, ["a", "b"])
    old = build_mm_cpu({"codes.ref": refs})
    total = 4
    for req_id, idx, start, end in (("a", 0, 0, 2), ("b", 1, 2, 4)):
        want = to_payload_element(old["codes.ref"], idx, start=start, end=end, pass_lists_through=True, seq_len=total)
        got = outs.mm_outputs["codes.ref"][req_id]
        assert len(got) == len(want)
        for g, w in zip(got, want):
            assert torch.equal(g, w)


def test_is_per_token_tensor():
    n, padded = 4, 8
    assert _is_per_token_tensor(torch.zeros(n, 2), n, padded)
    assert _is_per_token_tensor(torch.zeros(padded, 2), n, padded)
    assert not _is_per_token_tensor([torch.zeros(1, 8)], n, padded)
    assert not _is_per_token_tensor(torch.zeros(15, 2), n, padded)
    assert not _is_per_token_tensor(torch.tensor([1, 2, 3]), n, padded)


def test_policy_from_model_shim():
    class M:
        requires_full_prefix_cached_hidden_states = False
        mm_outputs_written_in_sample = True

    p = ModelCachePolicy.from_model(M())
    assert p.needs_full_hidden_states is False
    assert p.mm_outputs_written_in_sample is True
    d = ModelCachePolicy.from_model(object())
    assert d.needs_full_hidden_states is True
    assert d.mm_outputs_written_in_sample is False
    with pytest.raises(OmniPrefixCacheUnmatchError, match="mm_outputs_written_in_sample"):
        ModelCachePolicy(mm_outputs_written_in_sample=True)


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


def test_tenant_succession_hit_reads_newest():
    mgr, view = make_manager()
    s1 = run_engine_step_prefix_update(mgr, view, {"a": ([0], 0, 4)})
    mgr.merge_batch(s1, ["a"])
    s2 = run_engine_step_prefix_update(mgr, view, {"b": ([0], 0, 4)}, finished=["a"])
    mgr.merge_batch(s2, ["b"])
    b_hidden = expected_rows(view.slots_for("b", 0, 4))
    s3 = run_engine_step_prefix_update(mgr, view, {"c": ([0, 1], 4, 2)}, new_hits={"c": 4}, finished=["b"])
    outs = mgr.merge_batch(s3, ["c"])
    assert torch.equal(outs.hidden_states["c"][:4], b_hidden)


def test_tenant_succession_mm_key():
    mgr, view = make_manager()
    s1 = run_engine_step_prefix_update(mgr, view, {"a": ([2], 0, 4)}, mm={"k": torch.full((4, 2), 1.0)})
    mgr.merge_batch(s1, ["a"])
    s2 = run_engine_step_prefix_update(mgr, view, {"b": ([2], 0, 4)}, finished=["a"], mm={"k": torch.full((4, 2), 2.0)})
    mgr.merge_batch(s2, ["b"])
    s3 = run_engine_step_prefix_update(
        mgr, view, {"c": ([2, 3], 4, 2)}, new_hits={"c": 4}, finished=["b"], mm={"k": torch.full((2, 2), 3.0)}
    )
    rows = mgr.merge_batch(s3, ["c"]).mm_outputs["k"]["c"]
    assert torch.equal(rows[:4], torch.full((4, 2), 2.0))
    assert torch.equal(rows[4:], torch.full((2, 2), 3.0))


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
    s1 = run_engine_step_prefix_update(mgr, view, {"a": ([0], 0, 4)})
    s2 = run_engine_step_prefix_update(mgr, view, {"b": ([1], 0, 4)})
    mgr.skip_merge(s2)
    outs = mgr.merge_batch(s1, ["a"])
    assert torch.equal(outs.hidden_states["a"], expected_rows(view.slots_for("a", 0, 4)))
    assert len(mgr._batches) == 0


def test_step_context_exactly_once():
    mgr, view = make_manager()
    batch_id = run_engine_step_prefix_update(mgr, view, {"a": ([0], 0, 4)})
    mgr.merge_batch(batch_id, ["a"])
    with pytest.raises(OmniPrefixCacheUnmatchError):
        mgr.merge_batch(batch_id, ["a"])
    with pytest.raises(OmniPrefixCacheUnmatchError):
        mgr.skip_merge(batch_id)


def test_save_slot_mismatch_fails_fast():
    mgr, view = make_manager()
    view.order = ["a"]
    view.req_blocks["a"] = [0]
    view.computed["a"] = 0
    sched = FakeSchedOut(new_reqs=[FakeNewReq("a")], num_scheduled={"a": 2})
    adapter = PrefixCacheSchedulerAdapter()
    mgr.new_step_starts(adapter.translate_step(sched))
    hidden = torch.zeros(4, HIDDEN, dtype=DTYPE)
    with pytest.raises(OmniPrefixCacheUnmatchError):
        mgr.save_outputs(
            hidden,
            {},
            num_tokens_unpadded=4,
            num_tokens_padded=4,
            write_layout=adapter.build_write_layout(view, num_scheduled_tokens=sched.num_scheduled_tokens),
        )


@pytest.mark.parametrize(
    ("hidden", "mm"),
    [
        (torch.zeros(2, HIDDEN, dtype=DTYPE, device="meta"), {}),
        (torch.zeros(2, HIDDEN, dtype=DTYPE), {"codes": {"ref": [torch.zeros(2, 1, device="meta")]}}),
    ],
    ids=["hidden", "nested_mm"],
)
def test_save_rejects_device_tensors(hidden, mm):
    """Ensure save_outputs raises on device tensors instead of copying them."""
    # We use meta tensors here in place of cuda tensors.
    mgr, view = make_manager()
    view.order = ["a"]
    view.req_blocks["a"] = [0]
    view.computed["a"] = 0
    sched = FakeSchedOut(new_reqs=[FakeNewReq("a")], num_scheduled={"a": 2})
    adapter = PrefixCacheSchedulerAdapter()
    mgr.new_step_starts(adapter.translate_step(sched))
    n = sched.num_scheduled_tokens["a"]
    with pytest.raises(OmniPrefixCacheUnmatchError, match="CPU tensors"):
        mgr.save_outputs(
            hidden,
            mm,
            num_tokens_unpadded=n,
            num_tokens_padded=n,
            write_layout=adapter.build_write_layout(view, num_scheduled_tokens=sched.num_scheduled_tokens),
        )
    assert not mgr._batches


def test_merge_batch_rejects_out_of_snapshot_ids():
    mgr, view = make_manager()
    s1 = run_engine_step_prefix_update(mgr, view, {"a": ([0, 1], 0, 8)})
    mgr.merge_batch(s1, ["a"])
    s2 = run_engine_step_prefix_update(mgr, view, {"b": ([0, 1, 2], 8, 4)}, new_hits={"b": 8}, finished=["a"])
    with pytest.raises(OmniPrefixCacheUnmatchError, match="outside the save snapshot"):
        mgr.merge_batch(s2, ["b", "late_joiner"])


def test_mm_key_never_saved_for_part_of_a_hit_fails_fast():
    """Ensure save fails when part of a hit has no saved rows for a key the batch saves."""
    mgr, view = make_manager()
    s1 = run_engine_step_prefix_update(mgr, view, {"a": ([0], 0, 4)}, mm={"k": torch.full((4, 2), 5.0)})
    mgr.merge_batch(s1, ["a"])
    s2 = run_engine_step_prefix_update(mgr, view, {"c": ([1], 0, 4)})
    mgr.merge_batch(s2, ["c"])
    with pytest.raises(OmniPrefixCacheUnmatchError, match="key=k"):
        run_engine_step_prefix_update(
            mgr, view, {"b": ([0, 1, 2], 8, 2)}, new_hits={"b": 8}, mm={"k": torch.full((2, 2), 9.0)}
        )
    assert not mgr._batches


def test_delayed_read_after_reuse_serves_the_original_rows():
    """Ensure a hit read late (as under async output) returns its rows even after its block is reused."""
    mgr, view = make_manager()
    mgr.merge_batch(run_engine_step_prefix_update(mgr, view, {"a": ([0], 0, 4)}), ["a"])
    late = run_engine_step_prefix_update(mgr, view, {"c": ([0, 1], 4, 1)}, new_hits={"c": 4}, finished=["a"])
    # Before `late` is read, block 0 goes to b, which writes other rows into it.
    adapter = mgr._test_adapter
    sched = FakeSchedOut(new_reqs=[FakeNewReq("b", 0, [[0]])], finished=["c"], num_scheduled={"b": 4})
    mgr.new_step_starts(adapter.translate_step(sched))
    view.order, view.req_blocks["b"], view.computed["b"] = ["b"], [0], 0
    layout = adapter.build_write_layout(view, num_scheduled_tokens={"b": 4})
    other = torch.full((4, HIDDEN), 99.0, dtype=DTYPE)
    mgr.skip_merge(mgr.save_outputs(other, {}, num_tokens_unpadded=4, num_tokens_padded=4, write_layout=layout))

    merged = mgr.merge_batch(late, ["c"]).hidden_states["c"]
    assert torch.equal(merged[:4], expected_rows(view.slots_for("a", 0, 4)))


def test_hidden_opt_out_returns_mm_only():
    """Ensure a model that opts out of hidden states gets only its mm outputs back."""
    policy = ModelCachePolicy(needs_full_hidden_states=False)
    mgr, view = make_manager(policy=policy)
    mm = {"k": torch.full((4, 2), 1.0)}
    batch_id = run_engine_step_prefix_update(mgr, view, {"a": ([0], 0, 4)}, mm=mm)
    outs = mgr.merge_batch(batch_id, ["a"])
    assert outs.hidden_states is None
    assert torch.equal(outs.mm_outputs["k"]["a"], mm["k"])


def test_new_step_requires_step_and_never_falls_back_to_event_counts():
    mgr, _ = make_manager()
    event = PrefixCacheRequestEvent("a", PrefixCacheEventKind.STARTED, scheduled_tokens=6)
    with pytest.raises(TypeError, match="PrefixCacheStep"):
        mgr.new_step_starts((event,))
    mgr.new_step_starts(PrefixCacheStep((event,), ()))
    assert mgr._cur_num_scheduled == {}


@pytest.mark.parametrize("blocks", [[0, 1], [0, 1, 2]])
def test_hit_block_group_accepts_complete_prefix_and_ignores_tail(blocks):
    mgr, view = make_manager()
    mgr.merge_batch(run_engine_step_prefix_update(mgr, view, {"a": ([0, 1], 0, 8)}), ["a"])
    batch_id = run_engine_step_prefix_update(mgr, view, {"b": (blocks + [3], 8, 1)}, new_hits={"b": 8}, finished=["a"])
    result = mgr.merge_batch(batch_id, ["b"])
    expected = expected_rows(view.slots_for("b", 0, 9))
    assert torch.equal(result.hidden_states["b"], expected)


### Direct tests against SlotIndex correctness

# NOTE: block b holds slots [b * SLOT_BLOCK_SIZE, (b + 1) * SLOT_BLOCK_SIZE)
SLOT_BLOCK_SIZE = 2


def saved_rows(first_value: int, num_rows: int) -> SavedOutputs:
    """Rows whose values are first_value, first_value + 1, ... so tests can see where a row came from."""
    return {"hidden": torch.arange(first_value, first_value + num_rows, dtype=torch.float32)}


def hit_values(hit: PrefixHit, key: str = "hidden") -> list[float]:
    """The values of a hit's rows, in slot order."""
    return hit.cat(key, new_rows=torch.empty(0)).tolist()  # the hit's rows alone, with nothing after them


@pytest.fixture
def index() -> SlotIndex:
    return SlotIndex(num_blocks=8, block_size=SLOT_BLOCK_SIZE)


@pytest.mark.parametrize(
    ("blocks", "expected"),
    [([4, 5, 0], [100, 101, 102, 103, 200, 201]), ([5, 4], [102, 103, 100, 101])],
)
def test_lookup_returns_rows_in_the_requests_block_order(index, blocks, expected):
    """Ensure a hit's rows come back in the request's block order, across saves and within one."""
    # A slot's block is slot // SLOT_BLOCK_SIZE. Each block has 2 slots in these tests,
    # so slots 8-11 are blocks 4 and 5, and slots 0-1 are block 0.
    index.record(saved_rows(100, 4), slots=torch.tensor([8, 9, 10, 11]))  # blocks 4 & 5
    index.record(saved_rows(200, 2), slots=torch.tensor([0, 1]))  # block 0

    # Each record above saved one batch. lookup builds one request's hit, which is
    # the saved rows of the blocks its cached prefix covers in the request's block order.
    hit = index.lookup(blocks)
    assert hit_values(hit) == expected
    assert hit.all_slots_saved("hidden")


def test_cat_puts_new_rows_after_the_hit(index):
    """Ensure cat puts the request's new rows after the hit's rows."""
    index.record(saved_rows(200, 2), slots=torch.tensor([0, 1]))  # block 0
    computed_row = torch.tensor([7.0])  # the one row the request computed this step
    rows = index.lookup([0]).cat("hidden", new_rows=computed_row)
    assert rows.tolist() == [200, 201, 7]


def test_saved_outputs_are_freed_once_all_their_slots_are_taken_over(index):
    """Ensure a save is freed once no slot points at it."""
    # The first save writes slots 0 and 1. The test keeps only a weak reference
    # to its rows, so it can see whether the index still holds them.
    first_save = saved_rows(100, 2)
    first_save_rows = weakref.ref(first_save["hidden"])
    index.record(first_save, slots=torch.tensor([0, 1]))
    del first_save

    # A second save takes over slot 0. Slot 1 still points at the first save, so it's kept.
    index.record(saved_rows(200, 1), slots=torch.tensor([0]))
    assert first_save_rows() is not None

    # A third save takes over slot 1. No slot points at the first save now, so it's freed.
    index.record(saved_rows(300, 1), slots=torch.tensor([1]))
    assert first_save_rows() is None


def test_lookup_is_unaffected_by_later_saves(index):
    """Ensure a hit keeps the rows it looked up after later saves take over its slots."""
    first = saved_rows(100, 2)
    first_freed = weakref.ref(first["hidden"])
    index.record(first, slots=torch.tensor([0, 1]))
    del first

    # Lookup block 0, then write to block 0. Record creates a new save and updates
    # reference counts to reflect the new write. Since the hit has a SavedSlice in it,
    # the old values need to be retrievable until we delete it.
    hit = index.lookup([0])
    index.record(saved_rows(200, 2), slots=torch.tensor([0, 1]))

    # Ensure that our look up still returns the correct original values on the hit object
    assert hit_values(hit) == [100, 101]
    del hit
    assert first_freed() is None  # only the hit was keeping it


def test_all_slots_saved_needs_a_row_for_every_slot(index):
    """Ensure all_slots_saved is False when any slot of the hit has no row for the key."""
    index.record(saved_rows(100, 1), slots=torch.tensor([1]))
    hit = index.lookup([0])

    assert not hit.all_slots_saved("hidden")  # slot 0 is missing
    assert not hit.all_slots_saved("codes")  # both are missing


@pytest.mark.parametrize("slots", [[0, 0], [-1, 1]])
def test_record_rejects_repeated_or_negative_slots(index, slots):
    """Ensure record rejects repeated or negative slots."""
    with pytest.raises(OmniPrefixCacheUnmatchError):
        index.record(saved_rows(100, 2), slots=torch.tensor(slots))


def test_record_rejects_tensors_without_one_row_per_slot(index):
    """Ensure record rejects tensors whose row count doesn't match the slot count."""
    with pytest.raises(OmniPrefixCacheUnmatchError):
        index.record(saved_rows(100, 2), slots=torch.tensor([8, 9, 10, 11]))
