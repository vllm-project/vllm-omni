# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
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
from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionTickRequest

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
    gpu_memory_fraction: float = 0.1,
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

    def boom(self, req, kv_prefetch_job=None):
        state = pipeline.bound_state
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
