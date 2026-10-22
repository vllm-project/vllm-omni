# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for PrefixCacheRunnerMixin: one-shot construction, policy
snapshot identity/isolation, and the shared save/materialize gates.
No vLLM and no GPU required (same shim as test_prefix_cache.py)."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

try:  # pragma: no cover - shim only matters on vllm-less dev machines
    import vllm  # noqa: F401
except ModuleNotFoundError:
    _root = Path(__file__).resolve().parents[2]
    for _pkg in ("vllm_omni", "vllm_omni.core"):
        if _pkg not in sys.modules:
            _m = ModuleType(_pkg)
            _m.__path__ = [str(_root / _pkg.replace(".", "/"))]
            sys.modules[_pkg] = _m
    import logging

    _vllm = ModuleType("vllm")
    _vllm_logger = ModuleType("vllm.logger")
    setattr(_vllm_logger, "init_logger", logging.getLogger)
    setattr(_vllm, "logger", _vllm_logger)
    sys.modules["vllm"] = _vllm
    sys.modules["vllm.logger"] = _vllm_logger

from vllm_omni.core.prefix_cache.adapter import PrefixCacheStep, PrefixCacheWriteLayout
from vllm_omni.core.prefix_cache.interface import ModelCachePolicy, PrefixCacheConfig, StageCacheOutputs
from vllm_omni.core.prefix_cache.runner_mixin import PrefixCacheRunnerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

BLOCK_SIZE = 4


def _patch_pp(monkeypatch, is_last: bool) -> None:
    """Route the mixin's call-time get_pp_group import to a fake rank."""
    mod = ModuleType("vllm.distributed.parallel_state")
    setattr(mod, "get_pp_group", lambda: SimpleNamespace(is_last_rank=is_last))
    if "vllm.distributed" not in sys.modules:
        monkeypatch.setitem(sys.modules, "vllm.distributed", ModuleType("vllm.distributed"))
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", mod)


def _full_attention_groups():
    """A single-group config that passes check_prefix_cache_kv_groups.

    Real FullAttentionSpec when vllm is installed; otherwise a stand-in
    registered under the import path the check resolves at call time.
    """
    try:
        from vllm.v1.kv_cache_interface import FullAttentionSpec
    except ModuleNotFoundError:
        iface = sys.modules.get("vllm.v1.kv_cache_interface")
        if iface is None:
            iface = ModuleType("vllm.v1.kv_cache_interface")

            class _FullAttentionSpecStub:
                pass

            setattr(iface, "FullAttentionSpec", _FullAttentionSpecStub)
            sys.modules["vllm.v1"] = ModuleType("vllm.v1")
            sys.modules["vllm.v1.kv_cache_interface"] = iface
        FullAttentionSpec = getattr(iface, "FullAttentionSpec")
    return [SimpleNamespace(kv_cache_spec=FullAttentionSpec.__new__(FullAttentionSpec))]


class _FakeBlockTables:
    """Group-0 table whose layout attrs pass check_prefix_cache_block_layout."""

    def __init__(self):
        self.block_tables = [object()]

    def __getitem__(self, idx):
        return SimpleNamespace()


class _Runner(PrefixCacheRunnerMixin):
    def __init__(self, cfg: PrefixCacheConfig | None = None):
        # Any: tests swap in _CacheStub for the manager.
        self.omni_prefix_cache: Any = None
        self._omni_prefix_cache_cfg = cfg
        self.input_batch = SimpleNamespace(block_table=_FakeBlockTables(), req_ids=[])
        self.kv_cache_config = SimpleNamespace(kv_cache_groups=_full_attention_groups())
        self.is_pooling_model = False


def _cfg() -> PrefixCacheConfig:
    return PrefixCacheConfig(num_blocks=8, block_size=BLOCK_SIZE)


def _sched():
    return SimpleNamespace(scheduled_new_reqs=(), finished_req_ids=set(), num_scheduled_tokens={})


class _CacheStub:
    def __init__(self):
        self.save_calls = []
        self.materialize_calls = []
        self.outs = StageCacheOutputs(hidden_states=None, mm_outputs={})

    def save_outputs(self, hidden, mm, *, num_tokens_unpadded, num_tokens_padded, write_layout):
        self.save_calls.append((hidden, mm, num_tokens_unpadded, num_tokens_padded, write_layout))
        return 7

    def materialize(self, step_id, req_ids):
        self.materialize_calls.append((step_id, req_ids))
        return self.outs


def test_step_begin_builds_once_and_registers_snapshot_policy(monkeypatch):
    _patch_pp(monkeypatch, is_last=True)
    r = _Runner(_cfg())
    r._snapshot_prefix_cache_model_policy(
        SimpleNamespace(requires_full_prefix_cached_hidden_states=True, deferred_prefix_cache_mm_keys=("k",))
    )
    r._prefix_cache_step_begin(_sched())
    mgr = r.omni_prefix_cache
    assert mgr is not None and r._omni_prefix_cache_cfg is None
    assert mgr._policy is r._omni_cache_policy  # the load-time snapshot, same object
    r._prefix_cache_step_begin(_sched())
    assert r.omni_prefix_cache is mgr  # one-shot


def test_default_policy_isolated_per_instance():
    a, b = _Runner(), _Runner()
    a._snapshot_prefix_cache_model_policy(SimpleNamespace(requires_full_prefix_cached_hidden_states=False))
    assert a._omni_cache_policy.needs_full_hidden_states is False
    assert a._model_needs_full_prefix_hidden_states() is False
    assert b._omni_cache_policy.needs_full_hidden_states is True  # class default untouched
    assert PrefixCacheRunnerMixin._omni_cache_policy.needs_full_hidden_states is True


def test_non_last_rank_consumes_cfg_without_building(monkeypatch):
    _patch_pp(monkeypatch, is_last=False)
    r = _Runner(_cfg())
    r._prefix_cache_step_begin(_sched())
    assert r.omni_prefix_cache is None and r._omni_prefix_cache_cfg is None


def test_disabled_cache_step_begin_is_a_noop():
    r = _Runner(cfg=None)  # enable_prefix_caching off: nothing staged
    r._prefix_cache_step_begin(_sched())
    assert r.omni_prefix_cache is None


def test_save_step_gates_and_passthrough(monkeypatch):
    r = _Runner()
    stub = _CacheStub()
    hidden = torch.zeros(2, 2)

    def save():
        return r._prefix_cache_save_step(hidden, None, num_tokens_unpadded=2, num_tokens_padded=2)

    _patch_pp(monkeypatch, is_last=True)
    assert save() is None  # cache off
    r.omni_prefix_cache = stub
    r._prefix_cache_adapter = SimpleNamespace(
        build_write_layout=lambda view, *, num_scheduled_tokens: PrefixCacheWriteLayout((), 0)
    )
    r._prefix_cache_group_view = SimpleNamespace()
    r._prefix_cache_step = PrefixCacheStep((), ())
    r.is_pooling_model = True
    assert save() is None  # pooling stage never writes
    r.is_pooling_model = False
    _patch_pp(monkeypatch, is_last=False)
    assert save() is None  # not the last PP rank
    _patch_pp(monkeypatch, is_last=True)
    assert save() == 7
    assert stub.save_calls == [(hidden, {}, 2, 2, PrefixCacheWriteLayout((), 0))]  # empty mm stays {}


def test_materialize_requires_explicit_step_and_snapshot_req_ids():
    r = _Runner()
    stub = _CacheStub()
    r.omni_prefix_cache = stub
    assert r._prefix_cache_materialize(None, ["a"]) == (None, None)
    assert stub.materialize_calls == []  # step_id None: nothing to consume
    hidden = {"a": torch.zeros(1, 2)}
    stub.outs = StageCacheOutputs(hidden_states=hidden, mm_outputs={})
    assert r._prefix_cache_materialize(7, ["a"]) == (hidden, None)  # empty mm -> None
    assert stub.materialize_calls == [(7, ["a"])]


def test_step_begin_and_save_keep_the_translated_snapshot(monkeypatch):
    _patch_pp(monkeypatch, is_last=True)
    runner = _Runner()
    stub = _CacheStub()
    steps = []
    stub.new_step_starts = steps.append
    runner.omni_prefix_cache = stub
    from vllm_omni.core.prefix_cache.adapter import PrefixCacheSchedulerAdapter

    runner._prefix_cache_adapter = PrefixCacheSchedulerAdapter()
    runner._prefix_cache_group_view = SimpleNamespace(
        batch_req_ids=lambda: ["r"],
        step_slots_cpu=lambda req_ids, counts: torch.arange(counts["r"], dtype=torch.long),
    )
    arrival = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="r", num_computed_tokens=0)],
        finished_req_ids=set(),
        num_scheduled_tokens={"r": 2},
    )
    runner._prefix_cache_step_begin(arrival)
    arrival.num_scheduled_tokens["r"] = 99
    assert runner._prefix_cache_step is steps[0]
    assert steps[0].scheduled_tokens == (("r", 2),)
    runner._prefix_cache_save_step(torch.zeros(2, 2), None, num_tokens_unpadded=2, num_tokens_padded=2)
    layout = stub.save_calls[-1][-1]
    assert layout.total_rows == 2 and layout.slots_cpu.tolist() == [0, 1]

    continuation = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=["r"], resumed_req_ids=set()),
        finished_req_ids=set(),
        num_scheduled_tokens={"r": 1},
    )
    runner._prefix_cache_step_begin(continuation)
    assert steps[-1].events == ()
    assert steps[-1].extended_req_ids == ("r",)


@pytest.mark.parametrize("missing", ["adapter", "view"])
def test_step_begin_rejects_incomplete_cache_initialization(missing):
    from vllm_omni.core.prefix_cache.adapter import PrefixCacheSchedulerAdapter

    runner = _Runner()
    steps = []
    runner.omni_prefix_cache = SimpleNamespace(new_step_starts=steps.append)
    runner._prefix_cache_adapter = None if missing == "adapter" else PrefixCacheSchedulerAdapter()
    runner._prefix_cache_group_view = None if missing == "view" else SimpleNamespace()
    with pytest.raises(RuntimeError, match="adapter.*view.*initialized"):
        runner._prefix_cache_step_begin(_sched())
    assert steps == []
    assert runner._prefix_cache_step is None


@pytest.mark.parametrize("fail", [False, True])
def test_shutdown_clears_cache_unit_even_on_error(fail):
    import weakref

    from vllm_omni.core.prefix_cache.adapter import PrefixCacheSchedulerAdapter

    runner = _Runner(_cfg())
    calls = []

    def shutdown():
        calls.append(True)
        if fail:
            raise RuntimeError("shutdown failed")

    class View:
        pass

    view = View()
    ref = weakref.ref(view)
    runner.omni_prefix_cache = SimpleNamespace(shutdown=shutdown)
    runner._prefix_cache_adapter = PrefixCacheSchedulerAdapter()
    runner._prefix_cache_group_view = view
    runner._prefix_cache_step = PrefixCacheStep((), ())
    del view
    if fail:
        with pytest.raises(RuntimeError, match="shutdown failed"):
            runner._shutdown_omni_prefix_cache()
    else:
        runner._shutdown_omni_prefix_cache()
    assert runner.omni_prefix_cache is None
    assert runner._prefix_cache_adapter is None
    assert runner._prefix_cache_group_view is None
    assert runner._prefix_cache_step is None
    assert runner._omni_prefix_cache_cfg is None
    assert ref() is None
    runner._shutdown_omni_prefix_cache()
    runner._prefix_cache_step_begin(_sched())
    assert calls == [True]


def test_negative_non_hit_step_is_a_noop_through_manager(monkeypatch):
    _patch_pp(monkeypatch, is_last=True)
    runner = _Runner(_cfg())
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="r", num_computed_tokens=0)],
        finished_req_ids=set(),
        num_scheduled_tokens={"r": -512},
    )
    try:
        runner._prefix_cache_step_begin(scheduler_output)
        assert runner._prefix_cache_step.scheduled_tokens == (("r", 0),)
        assert runner.omni_prefix_cache._cur_num_scheduled == {"r": 0}
        assert runner.omni_prefix_cache._hit_spans == {}
        assert not runner.omni_prefix_cache._prefetch_queue
    finally:
        runner._shutdown_omni_prefix_cache()


def test_shared_runner_shutdown_releases_cache_before_upstream():
    """Execute the production method without importing CUDA/NPU modules."""
    import ast

    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / "vllm_omni/worker/gpu_model_runner.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OmniGPUModelRunner")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "shutdown")
    order = []

    class Base:
        def shutdown(self):
            assert self.omni_prefix_cache is None
            assert self._prefix_cache_group_view is None
            order.append("upstream")

    # Keep zero-argument super() semantics by compiling it inside a class.
    cls.bases = [ast.Name(id="PrefixCacheRunnerMixin", ctx=ast.Load()), ast.Name(id="Base", ctx=ast.Load())]
    cls.decorator_list = []
    cls.body = [method]
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = {"PrefixCacheRunnerMixin": PrefixCacheRunnerMixin, "Base": Base}
    exec(compile(module, "<shared runner shutdown>", "exec"), namespace)
    runner = namespace["OmniGPUModelRunner"]()
    runner.omni_prefix_cache = SimpleNamespace(shutdown=lambda: order.append("cache"))
    runner._prefix_cache_group_view = SimpleNamespace()
    runner.shutdown()
    assert order == ["cache", "upstream"]

    ar_tree = ast.parse((root / "vllm_omni/worker/gpu_ar_model_runner.py").read_text(encoding="utf-8"))
    ar_cls = next(node for node in ar_tree.body if isinstance(node, ast.ClassDef) and node.name == "GPUARModelRunner")
    ar_shutdown = next(node for node in ar_cls.body if isinstance(node, ast.FunctionDef) and node.name == "shutdown")
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_shutdown_omni_prefix_cache"
        for node in ast.walk(ar_shutdown)
    )


def test_native_only_snapshot_is_explicit_and_immutable():
    a, b = _Runner(), _Runner()
    model = SimpleNamespace(requires_full_prefix_cached_hidden_states=False, uses_native_kv_cache_only=True)
    a._snapshot_prefix_cache_model_policy(model)
    model.uses_native_kv_cache_only = False
    assert a._omni_cache_policy.native_kv_cache_only is True
    assert b._omni_cache_policy.native_kv_cache_only is False
    assert a._omni_cache_policy.needs_full_hidden_states is False


@pytest.mark.parametrize("mode", ["none", "all"])
@pytest.mark.parametrize("native_only", [False, True])
def test_mamba_prefix_cache_non_align_is_noop(mode, native_only):
    from vllm_omni.worker.mamba_prefix_cache import prepare_mamba_prefix_cache

    def callback():
        pytest.fail("must remain deferred")

    runner = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode=mode),
        _omni_cache_policy=ModelCachePolicy(needs_full_hidden_states=False, native_kv_cache_only=native_only),
    )
    assert prepare_mamba_prefix_cache(runner, object(), 2, callback) is callback


@pytest.mark.parametrize("policy", [None, ModelCachePolicy()], ids=["absent", "default"])
@pytest.mark.parametrize("deferred", [False, True])
def test_mamba_prefix_cache_requires_explicit_native_only_model(policy, deferred):
    from vllm_omni.worker.mamba_prefix_cache import prepare_mamba_prefix_cache

    def callback():
        pytest.fail("another model's correction must remain deferred")

    runner = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode="align"))
    if policy is not None:
        runner._omni_cache_policy = policy
    # Unselected models do not need Ming's state buffers or migration hooks.
    correction = callback if deferred else None
    assert prepare_mamba_prefix_cache(runner, object(), 2, correction) is correction


@pytest.mark.parametrize("postprocess", [False, True])
@pytest.mark.parametrize("deferred", [False, True])
def test_mamba_prefix_cache_align_order_and_native_arguments(monkeypatch, postprocess, deferred):
    from vllm_omni.worker.mamba_prefix_cache import prepare_mamba_prefix_cache

    events: list[Any] = []
    scheduler = object()
    pre, post, copies = object(), object() if postprocess else None, object()
    cache = SimpleNamespace(mamba_cache_mode="align")
    inputs = SimpleNamespace(num_accepted_tokens_cpu=torch.tensor([9, 8, 7]), req_ids=["a", "b"])
    accepted = torch.tensor([0, 0, 99])
    runner = SimpleNamespace(
        _omni_cache_policy=ModelCachePolicy(needs_full_hidden_states=False, native_kv_cache_only=True),
        cache_config=cache,
        kv_cache_config=object(),
        mamba_state_idx=object(),
        input_batch=inputs,
        requests=object(),
        compilation_config=SimpleNamespace(static_forward_context=object()),
        num_accepted_tokens=SimpleNamespace(
            np=accepted, copy_to_gpu=lambda n: events.append(("sync", n, accepted.clone()))
        ),
    )

    def get_bufs():
        events.append("buffers")
        return SimpleNamespace(preprocess=pre, postprocess_align=post)

    runner._get_mamba_bufs = get_bufs
    runner._get_mamba_state_copy_funcs = lambda: copies

    def preprocess(*args, **kwargs):
        assert args == (
            scheduler,
            runner.kv_cache_config,
            cache,
            runner.mamba_state_idx,
            inputs,
            runner.requests,
            runner.compilation_config.static_forward_context,
            copies,
            pre,
        )
        assert kwargs == {"align_ctx": post}
        assert events == (["deferred", "buffers"] if deferred else ["buffers"])
        events.append("preprocess")
        inputs.num_accepted_tokens_cpu[:2] = 1

    def stage(*args):
        assert args == (post, scheduler, inputs.req_ids, 2, runner.requests, runner.mamba_state_idx)
        assert events[-1][0] == "sync"
        events.append("stage")

    native = ModuleType("vllm.v1.worker.mamba_utils")
    setattr(native, "preprocess_mamba", preprocess)
    setattr(native, "stage_postprocess_inputs_to_gpu", stage)
    worker = ModuleType("vllm.v1.worker")
    setattr(worker, "mamba_utils", native)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker", worker)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.mamba_utils", native)
    callback = (lambda: events.append("deferred")) if deferred else None
    assert prepare_mamba_prefix_cache(runner, scheduler, 2, callback) is None
    sync = events[-2] if postprocess else events[-1]
    assert sync[:2] == ("sync", 2)
    assert sync[2].tolist() == [1, 1, 99]
    assert events[-1] == "stage" if postprocess else events[-2] == "preprocess"
