# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for capability-driven AR-Diffusion sessions."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.experimental.ar_diffusion.capability import (
    ARDiffusionCrossAttentionKVSpec,
    ARDiffusionKVBranchSpec,
    ARDiffusionKVCacheSpec,
)
from vllm_omni.experimental.ar_diffusion.kv_cache import ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.runner import ARDiffusionModelRunner

BLOCK = 16
POS = "positive"
NEG = "negative"

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def lingbot_like_spec(*, capacity: int = 2) -> ARDiffusionKVCacheSpec:
    """Single-kv_branch causal DMD: 3 latent frames/block + sink/window + text KV."""
    return ARDiffusionKVCacheSpec(
        num_layers=2,
        num_kv_heads=4,  # already TP-local
        head_size=64,
        tokens_per_frame=BLOCK,
        frames_per_block=3,
        window_frames=5,
        sink_frames=1,
        kv_branches=(ARDiffusionKVBranchSpec("main", 0),),
        session_capacity=capacity,
        cross_attention=(ARDiffusionCrossAttentionKVSpec("text", 8),),
    )


def dreamzero_like_spec(*, capacity: int = 2) -> ARDiffusionKVCacheSpec:
    return ARDiffusionKVCacheSpec(
        num_layers=2,
        num_kv_heads=4,
        head_size=64,
        tokens_per_frame=BLOCK,
        frames_per_block=4,
        window_frames=6,
        kv_branches=(ARDiffusionKVBranchSpec(POS, 0), ARDiffusionKVBranchSpec(NEG, 1)),
        session_capacity=capacity,
        cross_attention=(ARDiffusionCrossAttentionKVSpec("text", 8),),
    )


class CapablePipeline:
    def __init__(self, spec: ARDiffusionKVCacheSpec) -> None:
        self.spec = spec
        self.bound_state = None
        self.binds: list[str] = []
        self.resets: list[str] = []
        self.closes: list[str] = []

    def ar_diffusion_kv_cache_spec(self) -> ARDiffusionKVCacheSpec:
        return self.spec

    @contextmanager
    def bind_ar_diffusion_state(self, session_id: str, state):
        assert self.bound_state is None
        self.bound_state = state
        self.binds.append(session_id)
        try:
            yield
        finally:
            self.bound_state = None

    def reset_ar_diffusion_session(self, session_id: str) -> None:
        self.resets.append(session_id)

    def close_ar_diffusion_session(self, session_id: str) -> None:
        self.closes.append(session_id)


class WarmupPipeline(CapablePipeline):
    def __init__(self, spec: ARDiffusionKVCacheSpec, requests: list[object]) -> None:
        super().__init__(spec)
        self.requests = requests

    def ar_diffusion_warmup_requests(self, session_id: str):
        assert session_id == ARDiffusionModelRunner._WARMUP_SID
        return iter(self.requests)


class BatchCapablePipeline(CapablePipeline):
    supports_request_batch = True


def make_runner(
    pipeline: object,
    *,
    available_bytes: int = 1 << 28,
    step_execution: bool = False,
) -> ARDiffusionModelRunner:
    runner = object.__new__(ARDiffusionModelRunner)
    runner.od_config = SimpleNamespace(
        max_num_seqs=1,
        dtype=torch.float32,
        enforce_eager=True,
        step_execution=step_execution,
    )
    runner.device = torch.device("cpu")
    runner.pipeline = pipeline
    runner.ar_diffusion_kv_config = ARDiffusionKVConfig(enable=True)
    runner.kv_cache = None
    runner._ar_diffusion_capability = None
    runner._ar_diffusion_kv_cache_spec = None
    runner._sessions = OrderedDict()
    runner._session_capacity = 0
    runner._perf_e2e_times = []
    runner._preallocate_kv_cache(available_bytes=available_bytes)
    return runner


def commit_one_frame(runner: ARDiffusionModelRunner, session_id: str, kv_branch: str):
    state = runner._get_or_create_session(session_id)
    ctx = state.get_kv_caches(kv_branch, seq_len=BLOCK, commit_current=True)[0].forward_ctx
    ctx.ensure_video_slots(torch.device("cpu"))
    state.commit_paged_context(kv_branch)
    return state


def test_ar_runner_rejects_pipeline_without_capability():
    runner = object.__new__(ARDiffusionModelRunner)
    runner.od_config = SimpleNamespace(max_num_seqs=1)
    runner.pipeline = object()
    with pytest.raises(TypeError, match="SupportsARDiffusionPipeline"):
        runner._preallocate_kv_cache(available_bytes=1 << 20)


def test_lingbot_like_single_branch_session_reuse_reset_and_close():
    pipeline = CapablePipeline(lingbot_like_spec())
    runner = make_runner(pipeline)
    kv = runner.kv_cache
    assert kv is not None
    assert kv.num_local_kv_branches == 1
    assert kv.frames_per_block == 3
    assert kv.spec.window_chunks == 5
    assert kv.spec.sink_chunks == 1
    assert kv.cross_attention_lengths == {"text": 8}

    first = commit_one_frame(runner, "s1", "main")
    assert runner._get_or_create_session("s1") is first
    k = torch.randn(1, 8, 4, 64)
    v = torch.randn(1, 8, 4, 64)
    first.populate_cross_attention("main", "text", [(k, v)] * first.num_layers)
    assert first.get_cross_attention_kv("main", "text")[0]["k"].shape == k.shape

    runner.reset_session("s1")
    assert "s1" not in runner._sessions
    assert "s1" not in kv._cross_sessions
    assert pipeline.resets == ["s1"]
    second = runner._get_or_create_session("s1")
    assert second is not first

    runner.close_session("s1")
    assert "s1" not in runner._sessions
    assert pipeline.closes == ["s1"]


def test_lingbot_like_sink_survives_sliding_window_eviction():
    runner = make_runner(CapablePipeline(lingbot_like_spec()))
    kv = runner.kv_cache
    assert kv is not None

    for _ in range(8):
        state = commit_one_frame(runner, "s1", "main")

    table = kv.block_table(state.adapter("main"))
    assert table[0] != kv.null_block_id
    assert table[1] == kv.null_block_id

    ctx = state.get_kv_caches("main", seq_len=BLOCK, commit_current=False)[0].forward_ctx
    visible, _ = ctx.video_block_table(torch.device("cpu"))
    assert visible[0] == table[0]
    assert len(visible) == 6  # sink + recent window, including current scratch


def test_dreamzero_like_two_branches_are_independent():
    runner = make_runner(CapablePipeline(dreamzero_like_spec()))
    kv = runner.kv_cache
    assert kv is not None and kv.num_local_kv_branches == 2
    state = commit_one_frame(runner, "s1", POS)
    assert len(kv.window_block_ids(state.adapter(POS))) == 1
    assert kv.window_block_ids(state.adapter(NEG)) == []


def test_lru_eviction_releases_blocks_and_notifies_pipeline():
    pipeline = CapablePipeline(lingbot_like_spec(capacity=8))
    runner = make_runner(pipeline)
    kv = runner.kv_cache
    assert kv is not None
    assert runner._session_capacity == 1
    assert kv.session_capacity == 1
    free_total = kv.manager.block_pool.get_num_free_blocks()
    old = commit_one_frame(runner, "old", "main")
    k = torch.randn(1, 8, 4, 64)
    old.populate_cross_attention("main", "text", [(k, k)] * old.num_layers)
    assert kv.manager.block_pool.get_num_free_blocks() < free_total

    runner._get_or_create_session("new")

    assert tuple(runner._sessions) == ("new",)
    assert pipeline.closes == ["old"]
    assert "old" not in kv._cross_sessions
    assert kv.manager.block_pool.get_num_free_blocks() == free_total


def test_forward_exception_releases_pending_allocation_and_model_state(monkeypatch):
    pipeline = CapablePipeline(lingbot_like_spec())
    runner = make_runner(pipeline)
    kv = runner.kv_cache
    assert kv is not None
    free_total = kv.manager.block_pool.get_num_free_blocks()

    def boom(self, req, kv_prefetch_job=None):
        state = pipeline.bound_state
        ctx = state.get_kv_caches("main", seq_len=BLOCK, commit_current=True)[0].forward_ctx
        ctx.ensure_video_slots(torch.device("cpu"))
        raise RuntimeError("layer exploded")

    monkeypatch.setattr(DiffusionModelRunner, "execute_model", boom)
    request = SimpleNamespace(sampling_params=SimpleNamespace(extra_args={"session_id": "broken"}))

    with pytest.raises(RuntimeError, match="layer exploded"):
        runner.execute_model(request)

    assert pipeline.bound_state is None
    assert pipeline.closes == ["broken"]
    assert not runner._sessions
    assert not kv._adapters
    assert kv.manager.block_pool.get_num_free_blocks() == free_total


def test_synchronize_exception_uses_forward_cleanup_path(monkeypatch):
    pipeline = CapablePipeline(lingbot_like_spec())
    runner = make_runner(pipeline)
    kv = runner.kv_cache
    assert kv is not None
    free_total = kv.manager.block_pool.get_num_free_blocks()

    def return_after_allocation(self, req, kv_prefetch_job=None):
        state = pipeline.bound_state
        ctx = state.get_kv_caches("main", seq_len=BLOCK, commit_current=True)[0].forward_ctx
        ctx.ensure_video_slots(torch.device("cpu"))
        return object()

    def synchronize_boom(device):
        raise RuntimeError("asynchronous kernel failed")

    monkeypatch.setattr(DiffusionModelRunner, "execute_model", return_after_allocation)
    monkeypatch.setattr(torch.accelerator, "synchronize", synchronize_boom)
    runner.device = torch.device("cuda")
    request = SimpleNamespace(sampling_params=SimpleNamespace(extra_args={"session_id": "broken"}))

    with pytest.raises(RuntimeError, match="asynchronous kernel failed"):
        runner.execute_model(request)

    assert pipeline.bound_state is None
    assert pipeline.closes == ["broken"]
    assert not runner._sessions
    assert not kv._adapters
    assert not runner._perf_e2e_times
    assert kv.manager.block_pool.get_num_free_blocks() == free_total


def test_ar_runner_rejects_step_and_request_batch_modes():
    with pytest.raises(ValueError, match="step_execution=True"):
        make_runner(CapablePipeline(lingbot_like_spec()), step_execution=True)
    with pytest.raises(ValueError, match="request-batch execution"):
        make_runner(BatchCapablePipeline(lingbot_like_spec()))


def test_ar_runner_defensively_rejects_inherited_batch_and_step_entrypoints():
    runner = object.__new__(ARDiffusionModelRunner)
    with pytest.raises(RuntimeError, match="request-batch execution"):
        runner.execute_model_batch(None, None)
    with pytest.raises(RuntimeError, match="step execution"):
        runner.execute_stepwise(None)


def test_model_specific_warmup_provider_is_consumed(monkeypatch):
    requests = [object(), object()]
    pipeline = WarmupPipeline(lingbot_like_spec(), requests)
    runner = make_runner(pipeline)
    seen: list[object] = []
    monkeypatch.setattr(runner, "execute_model", seen.append)

    runner._warmup_ar_rollout()

    assert seen == requests
    assert pipeline.closes == [runner._WARMUP_SID]


def test_pipeline_without_warmup_provider_is_safely_skipped(monkeypatch):
    pipeline = CapablePipeline(lingbot_like_spec())
    runner = make_runner(pipeline)
    execute = SimpleNamespace(called=False)

    def fail_if_called(request):
        execute.called = True

    monkeypatch.setattr(runner, "execute_model", fail_if_called)
    runner._warmup_ar_rollout()
    assert execute.called is False
