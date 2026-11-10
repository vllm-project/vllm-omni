# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for capability-driven AR-Diffusion sessions."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.sched.interface import CachedRequestData, DiffusionSchedulerOutput
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.experimental.ar_diffusion.capability import (
    ARDiffusionCrossAttentionKVSpec,
    ARDiffusionKVBranchSpec,
    ARDiffusionKVCacheSpec,
)
from vllm_omni.experimental.ar_diffusion.kv_cache import ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState
from vllm_omni.experimental.ar_diffusion.runner import ARDiffusionModelRunner
from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionTickRequest

BLOCK = 16
POS = "positive"
NEG = "negative"

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def platform_synchronize_calls(monkeypatch) -> list[None]:
    """Stub the platform barrier the runner takes at every chunk boundary.

    These tests run on a CPU host, where the resolved platform is
    ``UnspecifiedOmniPlatform`` and ``synchronize()`` raises. Tests that need
    a failing barrier override this with their own fake.
    """
    calls: list[None] = []
    monkeypatch.setattr(
        "vllm_omni.experimental.ar_diffusion.runner.current_omni_platform",
        SimpleNamespace(synchronize=lambda: calls.append(None)),
    )
    return calls


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


def tiny_spec(*, capacity: int = 2) -> ARDiffusionKVCacheSpec:
    return ARDiffusionKVCacheSpec(
        num_layers=1,
        num_kv_heads=1,
        head_size=1,
        tokens_per_frame=1,
        frames_per_block=3,
        window_frames=3,
        sink_frames=3,
        kv_branches=(ARDiffusionKVBranchSpec("main", 0),),
        session_capacity=capacity,
        cross_attention=(ARDiffusionCrossAttentionKVSpec("text", 2),),
    )


class CapablePipeline:
    def __init__(self, spec: ARDiffusionKVCacheSpec) -> None:
        self.spec = spec
        self.bound_state: ARDiffusionKVState | None = None
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


class StepCapablePipeline(CapablePipeline):
    supports_step_execution = True

    def prepare_encode(self, state, **kwargs):
        del kwargs
        return state

    def denoise_step(self, input_batch, *, states=None, **kwargs):
        del input_batch, states, kwargs
        return None

    def step_scheduler(self, state, noise_pred, **kwargs):
        del state, noise_pred, kwargs

    def post_decode(self, state, **kwargs):
        del state, kwargs
        from vllm_omni.diffusion.data import DiffusionOutput

        return DiffusionOutput()


def scheduler_output(
    request_ids: tuple[str, ...] = ("req-1",),
    *,
    finished: tuple[str, ...] = (),
) -> DiffusionSchedulerOutput:
    """Build the real scheduler payload; stepwise requests arrive as cached rows."""
    return DiffusionSchedulerOutput(
        step_id=0,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(request_ids=list(request_ids)),
        finished_req_ids=set(finished),
        num_running_reqs=len(request_ids),
        num_waiting_reqs=0,
    )


def make_runner(
    pipeline: object,
    *,
    available_bytes: int = 1 << 28,
    step_execution: bool = False,
    gpu_memory_fraction: float = 0.1,
    preallocate: bool = True,
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
    runner.ar_diffusion_kv_config = ARDiffusionKVConfig(
        enable=True,
        gpu_memory_fraction=gpu_memory_fraction,
    )
    runner.kv_cache = None
    runner.noisy_kv_cache = None
    runner._ar_diffusion_capability = None
    runner._ar_diffusion_cell_capability = None
    runner._ar_diffusion_kv_cache_spec = None
    runner._sessions = OrderedDict()
    runner._session_capacity = 0
    runner._perf_e2e_times = []
    runner._stepwise_chunk_started = {}
    runner._active_stepwise_request_id = None
    runner.state_cache = {}
    runner.input_batch = None
    if preallocate:
        runner._preallocate_kv_cache(available_bytes=available_bytes)
    return runner


def commit_one_frame(runner: ARDiffusionModelRunner, session_id: str, kv_branch: str):
    state = runner._get_or_create_session(session_id)
    ctx = state.get_kv_caches(kv_branch, seq_len=BLOCK, commit_current=True)[0].forward_ctx
    ctx.ensure_video_slots(torch.device("cpu"))
    state.commit_paged_context(kv_branch)
    return state


def commit_one_block(runner: ARDiffusionModelRunner, session_id: str, kv_branch: str):
    state = runner._get_or_create_session(session_id)
    assert runner.kv_cache is not None
    seq_len = BLOCK * runner.kv_cache.frames_per_block
    ctx = state.get_kv_caches(kv_branch, seq_len=seq_len, commit_current=True)[0].forward_ctx
    ctx.ensure_video_slots(torch.device("cpu"))
    state.commit_paged_context(kv_branch)
    return state


def test_ar_runner_rejects_pipeline_without_capability():
    runner = object.__new__(ARDiffusionModelRunner)
    runner.od_config = SimpleNamespace(max_num_seqs=1)
    runner.pipeline = object()
    with pytest.raises(TypeError, match="SupportsARDiffusionPipeline"):
        runner._preallocate_kv_cache(available_bytes=1 << 20)


def test_runner_uses_typed_tick_as_authoritative_session_contract():
    tick = ARDiffusionTickRequest(
        session_id="world-7",
        request_id="request-3",
        chunk_index=3,
        reset=True,
    )
    req = SimpleNamespace(
        request_id="request-3",
        sampling_params=SimpleNamespace(extra_args=tick.to_extra_args()),
    )

    session_id, extra_args, parsed = ARDiffusionModelRunner._request_session(req)

    assert session_id == "world-7"
    assert extra_args == tick.to_extra_args()
    assert parsed == tick


def test_runner_keeps_engine_request_id_separate_from_tick_correlation_id():
    tick = ARDiffusionTickRequest(
        session_id="world-7",
        request_id="client-request-3",
        chunk_index=3,
    )
    req = SimpleNamespace(
        request_id="engine-request-uuid",
        sampling_params=SimpleNamespace(extra_args=tick.to_extra_args()),
    )

    session_id, _, parsed = ARDiffusionModelRunner._request_session(req)

    assert session_id == "world-7"
    assert req.request_id == "engine-request-uuid"
    assert parsed.request_id == "client-request-3"


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


def test_lingbot_like_interleaved_sessions_keep_independent_kv_partitions():
    runner = make_runner(CapablePipeline(lingbot_like_spec(capacity=2)))
    kv = runner.kv_cache
    assert kv is not None

    session_a = commit_one_block(runner, "world-a", "main")
    a0_blocks = kv.window_block_ids(session_a.adapter("main"))
    session_b = commit_one_block(runner, "world-b", "main")
    b0_blocks = kv.window_block_ids(session_b.adapter("main"))
    session_a_again = commit_one_block(runner, "world-a", "main")
    a1_blocks = kv.window_block_ids(session_a_again.adapter("main"))

    assert session_a_again is session_a
    assert session_a.adapter("main").request_id == "ar::world-a::main"
    assert session_b.adapter("main").request_id == "ar::world-b::main"
    assert session_a.adapter("main").completed_chunks == 6
    assert session_b.adapter("main").completed_chunks == 3
    assert len(a0_blocks) == 3
    assert len(b0_blocks) == 3
    assert len(a1_blocks) == 6
    assert set(a0_blocks) <= set(a1_blocks)
    assert set(a1_blocks).isdisjoint(b0_blocks)
    assert tuple(runner._sessions) == ("world-b", "world-a")


def test_lingbot_like_sink_survives_sliding_window_eviction():
    runner = make_runner(CapablePipeline(lingbot_like_spec()))
    kv = runner.kv_cache
    assert kv is not None

    state = commit_one_frame(runner, "s1", "main")
    sink_block = kv.block_table(state.adapter("main"))[0]
    for _ in range(7):
        state = commit_one_frame(runner, "s1", "main")

    adapter = state.adapter("main")
    table = kv.block_table(adapter)
    assert len(table) == 6  # one sink + five recent frames
    assert table[0] == sink_block
    assert kv.null_block_id not in table
    assert adapter.absolute_num_computed_tokens == 8 * BLOCK
    assert adapter.num_computed_tokens == 6 * BLOCK

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
    pipeline = CapablePipeline(lingbot_like_spec(capacity=2))
    runner = make_runner(pipeline)
    kv = runner.kv_cache
    assert kv is not None
    assert runner._session_capacity == 2
    assert kv.session_capacity == 2
    free_total = kv.manager.block_pool.get_num_free_blocks()
    old = commit_one_frame(runner, "old", "main")
    k = torch.randn(1, 8, 4, 64)
    old.populate_cross_attention("main", "text", [(k, k)] * old.num_layers)
    assert kv.manager.block_pool.get_num_free_blocks() < free_total

    runner._get_or_create_session("new")
    assert tuple(runner._sessions) == ("old", "new")
    runner._get_or_create_session("newest")

    assert tuple(runner._sessions) == ("new", "newest")
    assert pipeline.closes == ["old"]
    assert "old" not in kv._cross_sessions
    assert kv.manager.block_pool.get_num_free_blocks() == free_total


def test_budget_reduced_capacity_drives_runner_lru():
    pipeline = CapablePipeline(tiny_spec(capacity=2))
    # One tiny LingBot-like session requires 1,808 bytes; two require 2,592.
    #
    # These read 128 and 192 while the paging unit was the frame. tiny_spec
    # declares one token per frame, so it used to page a single token at a
    # time -- a block size no attention kernel accepts. Paging at 16 makes
    # each page sixteen times larger. The block *counts* are unchanged, and
    # so is what this test is about: a budget that fits fewer sessions than
    # the pipeline asked for must drive the runner's LRU.
    runner = make_runner(
        pipeline,
        available_bytes=2048,
        gpu_memory_fraction=1.0,
    )
    kv = runner.kv_cache
    assert kv is not None
    assert kv.requested_session_capacity == 2
    assert kv.session_capacity == 1
    assert runner._session_capacity == 1

    runner._get_or_create_session("old")
    runner._get_or_create_session("new")

    assert tuple(runner._sessions) == ("new",)
    assert pipeline.closes == ["old"]


def test_dreamzero_like_requested_capacity_is_capped_by_budget():
    spec = dreamzero_like_spec(capacity=64)
    # Per all-layer self-KV page: 65,536 bytes. Two resident sessions need:
    # managed=(2 * (2 * 6 + 4) + 2)=34 pages, scratch=8 pages,
    # cross-attention=1 page/session, for 44 pages total.
    page_bytes = 65_536
    pipeline = CapablePipeline(spec)
    runner = make_runner(
        pipeline,
        available_bytes=44 * page_bytes,
        gpu_memory_fraction=1.0,
    )
    kv = runner.kv_cache
    assert kv is not None
    assert kv.requested_session_capacity == 64
    assert kv.session_capacity == 2
    assert runner._session_capacity == 2
    assert kv.cross_attention_reserved_bytes == 2 * page_bytes


def test_forward_exception_releases_pending_allocation_and_model_state(monkeypatch):
    pipeline = CapablePipeline(lingbot_like_spec())
    runner = make_runner(pipeline)
    kv = runner.kv_cache
    assert kv is not None
    free_total = kv.manager.block_pool.get_num_free_blocks()

    def boom(self, req, kv_prefetch_job=None, diffusion_kv_metadata=None):
        state = pipeline.bound_state
        assert state is not None
        ctx = state.get_kv_caches("main", seq_len=BLOCK, commit_current=True)[0].forward_ctx
        ctx.ensure_video_slots(torch.device("cpu"))
        raise RuntimeError("layer exploded")

    monkeypatch.setattr(DiffusionModelRunner, "execute_model", boom)
    request = SimpleNamespace(
        request_id="broken-request",
        sampling_params=SimpleNamespace(extra_args={"session_id": "broken"}),
    )

    with pytest.raises(RuntimeError, match="layer exploded"):
        runner.execute_model(request)

    assert pipeline.bound_state is None
    assert pipeline.closes == ["broken"]
    assert not runner._sessions
    assert not kv._adapters
    assert kv.manager.block_pool.get_num_free_blocks() == free_total


def test_decode_failure_after_kv_commit_releases_session(monkeypatch):
    pipeline = CapablePipeline(lingbot_like_spec())
    runner = make_runner(pipeline)
    kv = runner.kv_cache
    assert kv is not None
    free_total = kv.manager.block_pool.get_num_free_blocks()

    def decode_failure(self, req, kv_prefetch_job=None, diffusion_kv_metadata=None):
        state = commit_one_block(runner, "broken", "main")
        assert state is pipeline.bound_state
        assert state._committed["main"] > 0
        raise RuntimeError("VAE decode failed")

    monkeypatch.setattr(DiffusionModelRunner, "execute_model", decode_failure)
    request = SimpleNamespace(
        request_id="broken-request",
        sampling_params=SimpleNamespace(extra_args={"session_id": "broken"}),
    )

    with pytest.raises(RuntimeError, match="VAE decode failed"):
        runner.execute_model(request)

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

    def return_after_allocation(self, req, kv_prefetch_job=None, diffusion_kv_metadata=None):
        state = pipeline.bound_state
        assert state is not None
        ctx = state.get_kv_caches("main", seq_len=BLOCK, commit_current=True)[0].forward_ctx
        ctx.ensure_video_slots(torch.device("cpu"))
        return object()

    def synchronize_boom():
        raise RuntimeError("asynchronous kernel failed")

    monkeypatch.setattr(DiffusionModelRunner, "execute_model", return_after_allocation)
    # The runner defers the device barrier to the platform, so the failure is
    # injected there rather than on a device-specific torch entry point.
    monkeypatch.setattr(
        "vllm_omni.experimental.ar_diffusion.runner.current_omni_platform",
        SimpleNamespace(synchronize=synchronize_boom),
    )
    request = SimpleNamespace(
        request_id="broken-request",
        sampling_params=SimpleNamespace(extra_args={"session_id": "broken"}),
    )

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


def test_ar_runner_allows_step_execution_when_pipeline_implements_the_contract():
    runner = make_runner(StepCapablePipeline(lingbot_like_spec()), step_execution=True)
    assert runner.kv_cache is not None


def test_ar_runner_defensively_rejects_inherited_batch_and_step_entrypoints():
    runner = make_runner(CapablePipeline(lingbot_like_spec()))
    with pytest.raises(RuntimeError, match="request-batch execution"):
        runner.execute_model_batch(None, None)
    with pytest.raises(RuntimeError, match="step execution"):
        runner.execute_stepwise(None)


def test_execute_stepwise_binds_request_id_session_and_unbinds_after_call(monkeypatch):
    from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput

    pipeline = StepCapablePipeline(lingbot_like_spec())
    runner = make_runner(pipeline, step_execution=True)
    bound_during = []

    def fake_execute(self, scheduler_output):
        del self, scheduler_output
        bound_during.append(pipeline.bound_state is not None)
        return BatchRunnerOutput.from_list([RunnerOutput(request_id="req-1", finished=False)])

    monkeypatch.setattr(DiffusionModelRunner, "execute_stepwise", fake_execute)
    output = runner.execute_stepwise(scheduler_output())

    assert bound_during == [True]
    assert pipeline.bound_state is None
    assert pipeline.binds == ["req-1"]
    assert "req-1" in runner._sessions
    assert output.get_request_output("req-1") is not None
    assert not pipeline.closes


def test_execute_stepwise_closes_session_when_request_finishes(monkeypatch):
    from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput

    pipeline = StepCapablePipeline(lingbot_like_spec())
    runner = make_runner(pipeline, step_execution=True)

    def fake_execute(self, scheduler_output):
        del self, scheduler_output
        return BatchRunnerOutput.from_list([RunnerOutput(request_id="req-1", finished=True)])

    monkeypatch.setattr(DiffusionModelRunner, "execute_stepwise", fake_execute)
    runner.execute_stepwise(scheduler_output())

    assert pipeline.bound_state is None
    assert pipeline.closes == ["req-1"]
    assert "req-1" not in runner._sessions


@pytest.mark.parametrize("reason", ["interaction", "timeout"])
def test_execute_stepwise_resumes_inside_bound_session_before_batch(monkeypatch, reason):
    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput, StepWait

    pipeline = StepCapablePipeline(lingbot_like_spec())
    runner = make_runner(pipeline, step_execution=True)
    state = SimpleNamespace(ready=False)
    runner.state_cache = {"req-1": state}

    def resume(request_state, received_reason):
        assert request_state is state
        assert received_reason == reason
        assert pipeline.bound_state is not None
        request_state.ready = True

    def wait(request_state):
        assert request_state is state
        assert pipeline.bound_state is not None
        return StepWait()

    monkeypatch.setattr(pipeline, "ar_diffusion_resume_step", resume, raising=False)
    monkeypatch.setattr(pipeline, "ar_diffusion_step_wait", wait, raising=False)

    def execute(self, output):
        assert state.ready, "Resume must update prompts before input batching"
        return BatchRunnerOutput.from_list([RunnerOutput("req-1", result=DiffusionOutput())])

    monkeypatch.setattr(DiffusionModelRunner, "execute_stepwise", execute)
    scheduled = scheduler_output()
    scheduled.step_resumes["req-1"] = reason
    output = runner.execute_stepwise(scheduled)
    assert output.get_request_output("req-1").wait_for_interaction == StepWait()
    assert "req-1" in runner._sessions
    assert pipeline.bound_state is None


def test_step_resume_failure_releases_session(monkeypatch):
    pipeline = StepCapablePipeline(lingbot_like_spec())
    runner = make_runner(pipeline, step_execution=True)
    runner.state_cache = {"req-1": object()}

    def resume(state, reason):
        raise ValueError("fallback encoding failed")

    monkeypatch.setattr(pipeline, "ar_diffusion_resume_step", resume, raising=False)
    scheduled = scheduler_output()
    scheduled.step_resumes["req-1"] = "timeout"
    with pytest.raises(ValueError, match="fallback encoding failed"):
        runner.execute_stepwise(scheduled)
    assert "req-1" not in runner._sessions
    assert "req-1" not in runner.state_cache


def test_execute_stepwise_exception_releases_kv_and_does_not_resume_pages(monkeypatch):
    pipeline = StepCapablePipeline(lingbot_like_spec())
    runner = make_runner(pipeline, step_execution=True)
    first = runner._get_or_create_session("req-1")

    def boom(self, scheduler_output):
        del self, scheduler_output
        raise RuntimeError("step failed")

    monkeypatch.setattr(DiffusionModelRunner, "execute_stepwise", boom)
    with pytest.raises(RuntimeError, match="step failed"):
        runner.execute_stepwise(scheduler_output())

    assert pipeline.bound_state is None
    assert pipeline.closes == ["req-1"]
    assert "req-1" not in runner._sessions
    second = runner._get_or_create_session("req-1")
    assert second is not first


def test_execute_stepwise_decode_error_after_kv_commit_releases_session(monkeypatch):
    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput

    pipeline = StepCapablePipeline(lingbot_like_spec())
    runner = make_runner(pipeline, step_execution=True)
    kv = runner.kv_cache
    assert kv is not None
    free_total = kv.manager.block_pool.get_num_free_blocks()

    def decode_error(self, scheduler_output):
        state = commit_one_block(runner, "req-1", "main")
        assert state is pipeline.bound_state
        assert state._committed["main"] > 0
        return BatchRunnerOutput.from_list(
            [
                RunnerOutput(
                    request_id="req-1",
                    finished=True,
                    result=DiffusionOutput(error="VAE decode failed"),
                )
            ]
        )

    monkeypatch.setattr(DiffusionModelRunner, "execute_stepwise", decode_error)
    output = runner.execute_stepwise(scheduler_output())

    assert output.get_request_output("req-1").result.error == "VAE decode failed"
    assert pipeline.closes == ["req-1"]
    assert not runner._sessions
    assert not kv._adapters
    assert kv.manager.block_pool.get_num_free_blocks() == free_total


def test_execute_stepwise_closes_session_for_scheduler_aborted_request(monkeypatch):
    from vllm_omni.diffusion.worker.utils import BatchRunnerOutput

    pipeline = StepCapablePipeline(lingbot_like_spec())
    runner = make_runner(pipeline, step_execution=True)
    runner._get_or_create_session("req-1")

    def fake_execute(self, scheduler_output):
        del self, scheduler_output
        return BatchRunnerOutput.from_list([])

    monkeypatch.setattr(DiffusionModelRunner, "execute_stepwise", fake_execute)
    # An aborted request is never scheduled again and never reports finished,
    # so the retire path is the only thing that can free its KV.
    runner.execute_stepwise(scheduler_output(request_ids=(), finished=("req-1",)))

    assert "req-1" not in runner._sessions
    assert pipeline.closes == ["req-1"]


def test_worker_release_held_request_closes_segmented_kv_without_another_wave():
    from tests.diffusion.ar_diffusion.test_segmented_kv import append, make_cache, make_state
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

    pipeline = StepCapablePipeline(lingbot_like_spec())
    runner = make_runner(pipeline, step_execution=True, preallocate=False)
    runner._ar_diffusion_capability = pipeline
    runner.kv_cache = cache = make_cache()
    free = cache.manager.block_pool.get_num_free_blocks()
    owner = make_state(cache, "held")
    runner._sessions["held"] = owner
    segments = owner.segmented("main")
    append(segments, (3, 5))
    segments.prepare_layer_views(6, 0)
    segments.begin_commit((2, 4))
    assert cache.manager.block_pool.get_num_free_blocks() < free
    runner.state_cache["held"] = SimpleNamespace(extra={"taomate_held": True})
    runner._stepwise_chunk_started["held"] = 0.0
    runner.input_batch = SimpleNamespace(request_ids=["held"])
    worker = object.__new__(DiffusionWorker)
    worker.model_runner = runner
    worker._step_lora_state = {"held": object()}

    # This is the cancellation RPC boundary. A held last request cannot rely
    # on another execute_stepwise call to retire either committed or pending KV.
    worker.release_step_requests(["held"])
    worker.release_step_requests(["held"])

    assert owner._closed
    assert not runner._sessions
    assert not runner.state_cache
    assert not runner._stepwise_chunk_started
    assert runner.input_batch is None
    assert not worker._step_lora_state
    assert pipeline.closes == ["held"]
    assert not cache._adapters
    assert cache.manager.block_pool.get_num_free_blocks() == free


@pytest.mark.parametrize("outcome", ["finished", "error", "exception"])
def test_step_release_defers_model_close_until_unbound(monkeypatch, platform_synchronize_calls, outcome):
    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput

    pipeline = StepCapablePipeline(lingbot_like_spec())
    runner = make_runner(pipeline, step_execution=True)
    runner.state_cache["req-1"] = object()

    def close(session_id):
        assert pipeline.bound_state is None
        if outcome != "exception":
            assert len(platform_synchronize_calls) == 1
        pipeline.closes.append(session_id)

    monkeypatch.setattr(pipeline, "close_ar_diffusion_session", close)

    def execute(self, scheduled):
        owner = pipeline.bound_state
        assert owner is not None
        self.release_step_requests(scheduled.scheduled_request_ids)
        assert not owner._closed
        assert self._sessions["req-1"] is owner
        assert not pipeline.closes
        if outcome == "exception":
            raise RuntimeError("injected forward failure")
        return BatchRunnerOutput.from_list(
            [
                RunnerOutput(
                    "req-1",
                    finished=True,
                    result=DiffusionOutput(error="decode failed" if outcome == "error" else None),
                )
            ]
        )

    monkeypatch.setattr(DiffusionModelRunner, "execute_stepwise", execute)
    if outcome == "exception":
        with pytest.raises(RuntimeError, match="injected forward failure"):
            runner.execute_stepwise(scheduler_output())
    else:
        runner.execute_stepwise(scheduler_output())
    assert pipeline.closes == ["req-1"]
    assert not runner._sessions
    assert not runner.state_cache
    assert not runner._stepwise_chunk_started
    assert runner._active_stepwise_request_id is None


def test_execute_stepwise_times_once_per_chunk_not_once_per_step(monkeypatch, platform_synchronize_calls):
    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput

    pipeline = StepCapablePipeline(lingbot_like_spec())
    runner = make_runner(pipeline, step_execution=True)
    results = [None, None, None, DiffusionOutput()]

    def fake_execute(self, scheduler_output):
        del self, scheduler_output
        return BatchRunnerOutput.from_list([RunnerOutput(request_id="req-1", finished=False, result=results.pop(0))])

    monkeypatch.setattr(DiffusionModelRunner, "execute_stepwise", fake_execute)
    for _ in range(4):
        runner.execute_stepwise(scheduler_output())

    # Four denoise steps, one emitted AR block: one timing sample, matching the
    # one-sample-per-block meaning request mode already has.
    assert len(runner._perf_e2e_times) == 1
    assert len(platform_synchronize_calls) == 1
    assert "req-1" not in runner._stepwise_chunk_started


def test_execute_stepwise_synchronize_exception_fails_closed(monkeypatch):
    """The chunk barrier is where an asynchronous device error surfaces. It must
    clean up like an in-step failure instead of leaving the session, the cached
    step state and the timing entry to the scheduler's next cycle."""
    from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput

    pipeline = StepCapablePipeline(lingbot_like_spec())
    runner = make_runner(pipeline, step_execution=True)
    runner.state_cache = {"req-1": object()}

    def fake_execute(self, scheduler_output):
        del self, scheduler_output
        return BatchRunnerOutput.from_list([RunnerOutput(request_id="req-1", finished=False, result=object())])

    def synchronize_boom():
        raise RuntimeError("asynchronous kernel failed")

    monkeypatch.setattr(DiffusionModelRunner, "execute_stepwise", fake_execute)
    monkeypatch.setattr(
        "vllm_omni.experimental.ar_diffusion.runner.current_omni_platform",
        SimpleNamespace(synchronize=synchronize_boom),
    )

    with pytest.raises(RuntimeError, match="asynchronous kernel failed"):
        runner.execute_stepwise(scheduler_output())

    assert pipeline.bound_state is None
    assert pipeline.closes == ["req-1"]
    assert "req-1" not in runner._sessions
    assert "req-1" not in runner.state_cache
    assert "req-1" not in runner._stepwise_chunk_started
    assert not runner._perf_e2e_times


@pytest.mark.parametrize(
    ("enforce_eager", "warmup_eager", "warmup_enabled", "expect_warmup"),
    [
        pytest.param(True, None, True, False, id="eager-default"),
        pytest.param(True, False, True, False, id="eager-opt-out"),
        pytest.param(True, True, True, True, id="eager-opt-in"),
        pytest.param(True, True, False, False, id="eager-warmup-disabled"),
        pytest.param(False, None, True, True, id="compiled-default"),
        pytest.param(False, True, True, True, id="compiled-opt-in"),
        pytest.param(False, None, False, False, id="compiled-warmup-disabled"),
        pytest.param(False, True, False, False, id="compiled-opt-in-warmup-disabled"),
    ],
)
def test_load_model_warms_only_when_enabled_after_kv_allocation(
    monkeypatch, enforce_eager, warmup_eager, warmup_enabled, expect_warmup
):
    sid = ARDiffusionModelRunner._WARMUP_SID
    request = SimpleNamespace(
        request_id="warmup-request",
        sampling_params=SimpleNamespace(extra_args={"session_id": sid}),
    )
    pipeline = WarmupPipeline(lingbot_like_spec(), [request])
    if warmup_eager is not None:
        monkeypatch.setattr(pipeline, "ar_diffusion_warmup_eager", warmup_eager, raising=False)
    runner = make_runner(pipeline, preallocate=False)
    runner.pipeline = None
    runner.od_config.enforce_eager = enforce_eager
    runner.ar_diffusion_kv_config.warmup_cudagraph = warmup_enabled
    calls: list[str] = []

    def load_pipeline(self, **kwargs):
        assert self.kv_cache is None
        calls.append("load")
        self.pipeline = pipeline

    def forward(self, req, kv_prefetch_job=None, diffusion_kv_metadata=None):
        assert self.kv_cache is not None
        assert self._ar_diffusion_capability is pipeline
        assert pipeline.bound_state is not None
        assert pipeline.bound_state.session_id == sid
        assert req is request
        calls.append("forward")
        return object()

    monkeypatch.setattr(DiffusionModelRunner, "load_model", load_pipeline)
    monkeypatch.setattr(DiffusionModelRunner, "execute_model", forward)
    monkeypatch.setattr(runner, "_available_memory_bytes", lambda: 1 << 28)

    runner.load_model()

    assert runner.kv_cache is not None
    assert calls == (["load", "forward"] if expect_warmup else ["load"])
    assert pipeline.closes == ([sid] if expect_warmup else [])
    assert pipeline.bound_state is None
    assert not runner._sessions
    assert not runner._perf_e2e_times


def test_eager_load_warmup_failure_releases_kv_and_keeps_model_loaded(monkeypatch):
    sid = ARDiffusionModelRunner._WARMUP_SID
    requests: list[object] = [
        SimpleNamespace(
            request_id=f"warmup-{index}",
            sampling_params=SimpleNamespace(extra_args={"session_id": sid}),
        )
        for index in range(3)
    ]
    pipeline = WarmupPipeline(lingbot_like_spec(), requests)
    monkeypatch.setattr(pipeline, "ar_diffusion_warmup_eager", True, raising=False)
    runner = make_runner(pipeline, preallocate=False)
    runner.pipeline = None
    seen: list[object] = []
    initial_free: list[int] = []
    injected_failures: list[object] = []

    def load_pipeline(self, **kwargs):
        self.pipeline = pipeline

    def forward(self, req, kv_prefetch_job=None, diffusion_kv_metadata=None):
        kv = self.kv_cache
        assert kv is not None
        if not seen:
            initial_free.append(kv.manager.block_pool.get_num_free_blocks())
        seen.append(req)
        state = pipeline.bound_state
        assert state is not None
        ctx = state.get_kv_caches("main", seq_len=BLOCK, commit_current=True)[0].forward_ctx
        ctx.ensure_video_slots(torch.device("cpu"))
        assert kv.manager.block_pool.get_num_free_blocks() < initial_free[0]
        if len(seen) == 2:
            assert len(self._perf_e2e_times) == 1
            injected_failures.append(req)
            raise RuntimeError("warmup forward failed with pending KV")
        state.commit_paged_context("main")
        return object()

    monkeypatch.setattr(DiffusionModelRunner, "load_model", load_pipeline)
    monkeypatch.setattr(DiffusionModelRunner, "execute_model", forward)
    monkeypatch.setattr(runner, "_available_memory_bytes", lambda: 1 << 28)

    runner.load_model()

    assert runner.pipeline is pipeline
    assert seen == requests[:2]
    assert injected_failures == [requests[1]]
    assert pipeline.bound_state is None
    assert pipeline.closes and set(pipeline.closes) == {sid}
    assert not runner._sessions
    assert not runner._perf_e2e_times
    kv = runner.kv_cache
    assert kv is not None
    assert not kv._adapters
    assert kv.manager.block_pool.get_num_free_blocks() == initial_free[0]


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


# ── paging unit vs eviction unit ────────────────────────────────────────────


def test_a_frame_that_is_already_a_legal_block_keeps_its_paging():
    """Every resolution that runs today must page exactly as it does today."""
    from vllm_omni.experimental.ar_diffusion.runner import paging_block_size

    for tokens_per_frame in (16, 320, 1440, 4096):
        assert tokens_per_frame % 16 == 0
        assert paging_block_size(tokens_per_frame) == tokens_per_frame


def test_a_frame_the_kernel_would_reject_pages_at_the_finest_legal_unit():
    """832x480 is the checkpoint's own default and the kernel rejects it.

    (480/16) x (832/16) = 1560 tokens per frame, and 1560 % 16 == 8, so a
    frame-sized block is not something FlashAttention's paged kernel accepts.
    """
    from vllm_omni.experimental.ar_diffusion.runner import paging_block_size

    assert 1560 % 16 == 8
    assert paging_block_size(1560) == 16
    # And whatever it returns is always something the kernel takes.
    for tokens_per_frame in range(1, 200):
        assert paging_block_size(tokens_per_frame) % 16 == 0


# ── the kernel's constraint is data, not a constant ─────────────────────────


def _multiple_of(base):
    from vllm.v1.attention.backend import MultipleOf

    return MultipleOf(base)


def test_the_dispatching_module_answers_in_vllms_own_vocabulary():
    """AR-Diffusion calls flash_attn itself, so no backend can be asked.

    The answer has to come from the module that picks the kernel, and it has to
    have the shape vLLM uses, so the policy that consumes it does not care
    where it came from.
    """
    from vllm.v1.attention.backend import MultipleOf

    from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import supported_kernel_block_sizes

    advertised = supported_kernel_block_sizes()
    assert advertised, "a kernel that accepts nothing cannot be paged for"
    for entry in advertised:
        assert isinstance(entry, int | MultipleOf)


def test_a_kernel_that_takes_one_fixed_size_gets_that_size():
    """hpc_attn advertises [64] -- not a multiple, a single legal value.

    A frame of 1560 tokens is not 64, and 1560 % 64 == 24, so the frame cannot
    be the block and the only thing left to page at is 64. Assuming multiples
    of 16 here would hand the kernel a block size it rejects.
    """
    from vllm_omni.experimental.ar_diffusion.runner import paging_block_size

    assert 1560 % 64 == 24
    assert paging_block_size(1560, [64]) == 64
    assert paging_block_size(64, [64]) == 64
    assert paging_block_size(128, [64]) == 64  # a legal multiple is still not the advertised size


def test_a_kernel_with_large_pages_still_keeps_a_whole_frame_when_it_fits():
    """FlashInfer offers 128-and-up pages on Blackwell and not on Hopper.

    The same checkpoint on the two cards must therefore page differently, which
    is exactly what a constant cannot express.
    """
    from vllm_omni.experimental.ar_diffusion.runner import paging_block_size

    hopper = [_multiple_of(16)]
    blackwell = [_multiple_of(16), 128, 256]
    assert paging_block_size(1440, hopper) == 1440  # a legal multiple of 16
    assert paging_block_size(1440, blackwell) == 1440
    assert paging_block_size(1560, hopper) == 16
    assert paging_block_size(1560, blackwell) == 16  # finest legal unit, not the largest


def test_whatever_comes_back_is_always_something_the_kernel_accepts():
    from vllm_omni.experimental.ar_diffusion.runner import paging_block_size

    for advertised, check in (
        ([_multiple_of(16)], lambda n: n % 16 == 0),
        ([64], lambda n: n == 64),
        ([_multiple_of(32), 48], lambda n: n % 32 == 0 or n == 48),
    ):
        for tokens_per_frame in range(1, 300):
            assert check(paging_block_size(tokens_per_frame, advertised))


def test_a_kernel_that_advertises_nothing_is_an_error_not_a_guess():
    from vllm_omni.experimental.ar_diffusion.runner import paging_block_size

    with pytest.raises(ValueError, match="no legal block size"):
        paging_block_size(1560, [])
