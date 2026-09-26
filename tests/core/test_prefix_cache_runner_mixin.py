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
from vllm_omni.core.prefix_cache.interface import PrefixCacheConfig, StageCacheOutputs
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
            setattr(iface, "RSWASpec", type("RSWASpec", (_FullAttentionSpecStub,), {}))
            setattr(iface, "SlidingWindowSpec", type("SlidingWindowSpec", (), {}))
            sys.modules["vllm.v1"] = ModuleType("vllm.v1")
            sys.modules["vllm.v1.kv_cache_interface"] = iface
        FullAttentionSpec = getattr(iface, "FullAttentionSpec")
    return [SimpleNamespace(kv_cache_spec=FullAttentionSpec.__new__(FullAttentionSpec))]


class _FakeBlockTables:
    """Group-0 table whose layout attrs pass check_prefix_cache_block_layout."""

    def __init__(self):
        self.block_tables = [object()]

    def __getitem__(self, idx):
        return SimpleNamespace(
            block_size=BLOCK_SIZE, kv_cache_block_size=BLOCK_SIZE, blocks_per_kv_block=1, dcp_world_size=1
        )


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
