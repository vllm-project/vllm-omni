# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for Noisy PP session_id / chunk_base continuity."""

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_executor import (
    ARDiffusionCellContext,
    CellAdapter,
    CellRunSpec,
    CellTopology,
    run_cell_pipeline,
)
from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_plan import (
    PRIOR_HANDOFF_SLOT,
    CellSchedule,
    Ordering,
    build_cell_plan,
    expected_prior_cells,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.experimental.ar_diffusion.kv_cache.config import ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import (
    ARDiffusionNoisyKVSpec,
    NoisyKVCache,
    NoisyKVState,
    clean_history_cells,
    prior_history_in_pool,
)
from vllm_omni.experimental.ar_diffusion.runner import ARDiffusionModelRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _cache(*, history: int = 2) -> NoisyKVCache:
    return NoisyKVCache(
        ARDiffusionNoisyKVSpec(
            num_layers=1,
            num_kv_heads=1,
            head_size=4,
            block_size=4,
            max_chunk_tokens=4,
            max_history_chunks=history,
        ),
        dtype=torch.float32,
        device=torch.device("cpu"),
        layer_groups=1,
        max_batch_size=1,
        step_replicas=1,
    )


def _schedule(**kwargs) -> CellSchedule:
    base = dict(
        chunks=1,
        num_denoise_steps=1,
        step_replicas=1,
        layer_groups=1,
        ordering=Ordering.SELF_FORCING,
        kv_history_chunks=1,
    )
    base.update(kwargs)
    return CellSchedule(**base)


class _CountingAdapter(CellAdapter):
    def forward(self, tasks, kv_contexts, *, hidden):
        return torch.zeros(1, 4, 8)

    def pack_activation(self, output):
        return {"hidden_states": output}

    def unpack_activation(self, payload):
        return payload["hidden_states"]


class _NullCM:
    def __init__(self, ctx):
        self.ctx = ctx

    def __enter__(self):
        return self.ctx

    def __exit__(self, *args):
        return False


def _bare_noisy_runner() -> ARDiffusionModelRunner:
    runner = object.__new__(ARDiffusionModelRunner)
    runner.od_config = SimpleNamespace(max_num_seqs=1, dtype=torch.float32, ar_diffusion_stage_config={})
    runner.device = torch.device("cpu")
    runner.pipeline = object()
    runner.ar_diffusion_kv_config = ARDiffusionKVConfig(enable=True)
    runner.kv_cache = None
    runner.noisy_kv_cache = _cache()
    runner._ar_diffusion_capability = None
    runner._ar_diffusion_noisy_kv_capability = SimpleNamespace(bind_ar_diffusion_noisy_kv=lambda state: _NullCM(state))
    runner._ar_diffusion_kv_cache_spec = None
    runner._sessions = OrderedDict()
    runner._session_capacity = 0
    runner._perf_e2e_times = []
    runner._stepwise_chunk_started = {}
    runner._active_stepwise_request_id = None
    return runner


def test_prior_plan_sources_and_cross_rank_handoff():
    local = _schedule(chunk_base=1)
    prior = expected_prior_cells(local)
    assert prior == frozenset({(0, 1, 0)})
    plan = build_cell_plan(local, prior_history=prior)
    task = next(plan.task(t, 0) for t in range(plan.num_slots) if plan.task(t, 0) and plan.task(t, 0)[0] == 1)
    assert plan.sources(task, 0)[0].version == (0, 1, 0)
    assert plan.transfers(0) == ()

    vertical = _schedule(num_denoise_steps=1, step_replicas=2, ordering=Ordering.LATEST_KV, chunk_base=1)
    prior = expected_prior_cells(vertical)
    plan = build_cell_plan(vertical, prior_history=prior)
    task = next(plan.task(t, 0) for t in range(plan.num_slots) if plan.task(t, 0) == (1, 0, 0))
    src = plan.sources(task, 0)[0]
    assert src.version == (0, 1, 0) and src.owner == 1
    assert any(x.version == (0, 1, 0) and x.src == 1 and x.dst == 0 for x in plan.transfers(PRIOR_HANDOFF_SLOT))
    assert plan.transfers(0) == ()
    assert (0, 1, 0) not in plan.wait_ready(0, 0)


def test_prior_handoff_runs_before_first_prepare():
    schedule = _schedule(step_replicas=2, ordering=Ordering.LATEST_KV, chunk_base=1)
    plan = build_cell_plan(schedule, prior_history=expected_prior_cells(schedule))
    cache = _cache(history=1)
    kv = NoisyKVState(cache)
    order: list[str] = []
    orig_prior, orig_prepare = kv.exchange_and_await_prior_handoffs, kv.prepare

    def _prior(slot):
        order.append("prior")
        return orig_prior(slot)

    def _prepare(tasks):
        order.append("prepare")
        for req, task in tasks:
            for src in plan.sources(task, 0):
                key = (req, *src.version)
                if not cache.pool.has(key):
                    cache.pool.alloc(key)
        return orig_prepare(tasks)

    kv.exchange_and_await_prior_handoffs = _prior  # type: ignore[method-assign]
    kv.prepare = _prepare  # type: ignore[method-assign]
    ctx = ARDiffusionCellContext(
        spec=CellRunSpec(topology=CellTopology(step_replicas=2, layer_groups=1), rank=0),
        kv=kv,
    )
    ctx.enqueue("world", plan, chunk_tokens=4, retain=clean_history_cells(schedule, num_blocks=1))
    run_cell_pipeline(ctx=ctx, adapter=_CountingAdapter())
    assert order.index("prior") < order.index("prepare")


def test_retain_window_structural_key_and_same_step_sink():
    cache = _cache(history=1)
    state = NoisyKVState(cache)
    same = _schedule(num_denoise_steps=2, step_replicas=2, ordering=Ordering.SAME_STEP, chunk_base=3)
    assert clean_history_cells(same, num_blocks=1) == {(0, 2, 0), (3, 2, 0)}
    cache.pool.alloc(("world", 0, 2, 0))
    cache.pool.alloc(("world", 3, 2, 0))
    cache.pool.alloc(("world", 2, 2, 0))
    state.end_request("world", retain=clean_history_cells(same, num_blocks=1))
    assert cache.pool.has(("world", 0, 2, 0)) and cache.pool.has(("world", 3, 2, 0))
    assert not cache.pool.has(("world", 2, 2, 0))

    first = _schedule()
    cache.bind_session_structure("world", first.structural_key())
    with pytest.raises(ValueError, match="structural_key mismatch"):
        cache.bind_session_structure("world", _schedule(num_denoise_steps=2).structural_key())
    state.release_session("world")


def test_session_continuation_and_nonlocking_retain():
    cache = _cache(history=1)
    kv = NoisyKVState(cache)
    ctx = ARDiffusionCellContext(
        spec=CellRunSpec(topology=CellTopology(step_replicas=1, layer_groups=1), rank=0),
        kv=kv,
    )
    first = _schedule(chunks=2)
    cache.bind_session_structure("world", first.structural_key())
    retain = clean_history_cells(first, num_blocks=1)
    ctx.enqueue("world", build_cell_plan(first), chunk_tokens=4, retain=retain)
    run_cell_pipeline(ctx=ctx, adapter=_CountingAdapter())
    kv.end_request("world", retain=retain)
    assert cache.pool.has(("world", 1, 1, 0)) and not cache.pool.has(("world", 0, 1, 0))

    second = _schedule(chunk_base=2)
    prior = expected_prior_cells(second)
    assert prior == frozenset({(1, 1, 0)}) <= prior_history_in_pool(cache, "world", chunk_base=2)
    retain2 = clean_history_cells(second, num_blocks=1)
    assert prior.isdisjoint(retain2)
    ctx.enqueue("world", build_cell_plan(second, prior_history=prior), chunk_tokens=4, retain=retain2)
    run_cell_pipeline(ctx=ctx, adapter=_CountingAdapter())
    kv.end_request("world", retain=retain2)
    assert cache.pool.has(("world", 2, 1, 0)) and not cache.pool.has(("world", 1, 1, 0))

    # Separate pool: world still holds one retained version after the half above.
    wide_cache = _cache(history=6)
    wide_kv = NoisyKVState(wide_cache)
    wide = _schedule(chunks=6, kv_history_chunks=6, chunk_base=6)
    for chunk in range(6):
        wide_cache.pool.alloc(("wide", chunk, 1, 0))
    plan = build_cell_plan(wide, prior_history=expected_prior_cells(wide))
    keep = clean_history_cells(wide, num_blocks=1)
    wide_kv.begin_request("wide", plan, chunk_tokens=4, retain=keep)
    assert expected_prior_cells(wide).isdisjoint(keep)
    assert all(not wide_kv._is_retained("wide", cell) for cell in expected_prior_cells(wide))
    assert all(wide_kv._is_retained("wide", cell) for cell in keep)


def test_runner_lifecycle_and_batch_isolation(monkeypatch):
    from vllm_omni.diffusion.models.causvid_wan.pipeline_causvid_wan import CausVidWanPipeline
    from vllm_omni.diffusion.worker import diffusion_model_runner as dmr

    def _req(request_id="r0", **extra):
        return OmniDiffusionRequest(
            prompt="x",
            request_id=request_id,
            sampling_params=OmniDiffusionSamplingParams(extra_args=extra),
        )

    assert CausVidWanPipeline._noisy_request_lifecycle(_req(session_id="world", close_session=True)) == (
        "world",
        False,
        False,
        True,
    )

    runner = _bare_noisy_runner()
    released: list[str] = []
    monkeypatch.setattr(runner, "release_noisy_kv", lambda kv_id: released.append(kv_id))
    monkeypatch.setattr(
        dmr.DiffusionModelRunner,
        "execute_model",
        lambda self, req, kv_prefetch_job=None, diffusion_kv_metadata=None: SimpleNamespace(ok=True),
    )
    assert ARDiffusionModelRunner.execute_model(runner, _req("ephemeral-1")).ok
    assert released == ["ephemeral-1"]
    released.clear()
    assert ARDiffusionModelRunner.execute_model(runner, _req("req-1", session_id="world")).ok
    assert released == []

    runner = _bare_noisy_runner()
    key = _schedule().structural_key()
    runner.noisy_kv_cache.pool.alloc(("world", 1, 1, 0))
    runner.noisy_kv_cache.bind_session_structure("world", key)
    runner.reset_session("world")
    assert not runner.noisy_kv_cache.pool.has(("world", 1, 1, 0))
    runner.noisy_kv_cache.pool.alloc(("world", 2, 1, 0))
    runner.close_session("world")
    assert not runner.noisy_kv_cache.pool.has(("world", 2, 1, 0))

    # Batch failure for B/C must not wipe retained session A.
    runner = _bare_noisy_runner()
    cache = runner.noisy_kv_cache
    cache.pool.alloc(("session-a", 0, 1, 0))
    cache.bind_session_structure("session-a", key)

    def _boom(self, scheduler_output, od_config):
        self.noisy_kv_cache.pool.alloc(("session-b", 0, 1, 0))
        raise ValueError("chunk_base must be non-negative, got -1")

    monkeypatch.setattr(dmr.DiffusionModelRunner, "execute_model_batch", _boom)
    with pytest.raises(ValueError, match="chunk_base"):
        runner.execute_model_batch(
            SimpleNamespace(
                scheduled_new_reqs=[
                    SimpleNamespace(req=_req("rb", session_id="session-b", chunk_base=-1)),
                    SimpleNamespace(req=_req("rc", session_id="session-c")),
                ]
            ),
            od_config=SimpleNamespace(),
        )
    assert cache.pool.has(("session-a", 0, 1, 0))
    assert "session-a" in cache._session_structural_keys
    assert not cache.pool.has(("session-b", 0, 1, 0))
    assert frozenset({(0, 1, 0)}) <= prior_history_in_pool(cache, "session-a", chunk_base=1)

    # Batch shares single-request lifecycle: reset clears before bind; close releases after.
    runner = _bare_noisy_runner()
    cache = runner.noisy_kv_cache
    cache.pool.alloc(("world", 1, 1, 0))
    cache.bind_session_structure("world", key)
    monkeypatch.setattr(
        dmr.DiffusionModelRunner,
        "execute_model_batch",
        lambda self, scheduler_output, od_config: SimpleNamespace(ok=True),
    )
    assert runner.execute_model_batch(
        SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(req=_req("r-reset", session_id="world", reset=True)),
            ]
        ),
        od_config=SimpleNamespace(),
    ).ok
    assert not cache.pool.has(("world", 1, 1, 0))

    cache.pool.alloc(("world", 2, 1, 0))
    cache.bind_session_structure("world", key)
    assert runner.execute_model_batch(
        SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(req=_req("r-keep", session_id="world")),
            ]
        ),
        od_config=SimpleNamespace(),
    ).ok
    assert cache.pool.has(("world", 2, 1, 0))
    assert runner.execute_model_batch(
        SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(req=_req("r-close", session_id="world", close_session=True)),
            ]
        ),
        od_config=SimpleNamespace(),
    ).ok
    assert not cache.pool.has(("world", 2, 1, 0))

    # Same-session co-batch must fail before any reset wipes history for R1.
    # No stub on ``_prepare_noisy_kv_lifecycles``: if the guard ran after reset,
    # the pool key and structural binding below would already be gone.
    runner = _bare_noisy_runner()
    cache = runner.noisy_kv_cache
    cache.pool.alloc(("world", 10, 1, 0))
    cache.bind_session_structure("world", key)
    with pytest.raises(ValueError, match="same session"):
        runner.execute_model_batch(
            SimpleNamespace(
                scheduled_new_reqs=[
                    SimpleNamespace(req=_req("r1", session_id="world", chunk_base=10)),
                    SimpleNamespace(req=_req("r2", session_id="world", reset=True, chunk_base=0)),
                ]
            ),
            od_config=SimpleNamespace(),
        )
    assert cache.pool.has(("world", 10, 1, 0))
    assert "world" in cache._session_structural_keys
