# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for Noisy KV session_id pooling (no runner-owned session object)."""

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import OmniDiffusionSamplingParams
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.experimental.ar_diffusion.cell_executor import (
    ARDiffusionCellContext,
    CellAdapter,
    CellRunSpec,
    CellTopology,
    run_cell_pipeline,
)
from vllm_omni.experimental.ar_diffusion.cell_schedule import CellSchedule, Ordering, build_cell_plan
from vllm_omni.experimental.ar_diffusion.kv_cache.config import ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import (
    ARDiffusionNoisyKVSpec,
    NoisyKVCache,
    NoisyKVState,
    clean_history_cells,
    prior_history_in_pool,
)
from vllm_omni.experimental.ar_diffusion.runner import ARDiffusionModelRunner

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


class _CountingAdapter(CellAdapter):
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def forward(self, tasks, kv_contexts, *, hidden):
        self.calls.append(tasks)
        return torch.zeros(1, 4, 8)

    def pack_activation(self, output):
        return {"hidden_states": output}

    def unpack_activation(self, payload):
        return payload["hidden_states"]


def test_chunk_base_plan_sources_prior_history():
    prior = frozenset({(0, 1, 0)})
    schedule = CellSchedule(
        chunks=1,
        num_denoise_steps=1,
        step_replicas=1,
        layer_groups=1,
        ordering=Ordering.SELF_FORCING,
        kv_history_chunks=1,
        chunk_base=1,
    )
    plan = build_cell_plan(schedule, prior_history=prior)
    task = None
    for slot in range(plan.num_slots):
        task = plan.task(slot, 0)
        if task is not None and task[0] == 1:
            break
    assert task is not None
    sources = plan.sources(task, 0)
    assert sources
    assert sources[0].version == (0, 1, 0)
    assert plan.transfers(0) == ()


def test_end_request_can_retain_history_for_kv_id():
    cache = _cache()
    state = NoisyKVState(cache)
    plan = build_cell_plan(
        CellSchedule(
            chunks=2,
            num_denoise_steps=1,
            step_replicas=1,
            layer_groups=1,
            ordering=Ordering.SELF_FORCING,
            kv_history_chunks=1,
        )
    )
    state.begin_request("world", plan, chunk_tokens=4, retain={(1, 1, 0)})
    cache.pool.alloc(("world", 0, 1, 0))
    cache.pool.alloc(("world", 1, 1, 0))
    state.end_request("world")
    assert not cache.pool.has(("world", 0, 1, 0))
    assert cache.pool.has(("world", 1, 1, 0))
    state.release_session("world")
    assert not cache.pool.has(("world", 1, 1, 0))


def test_prior_history_and_clean_window_helpers():
    cache = _cache(history=1)
    cache.pool.alloc(("world", 0, 1, 0))
    cache.pool.alloc(("world", 1, 1, 0))
    assert prior_history_in_pool(cache, "world", chunk_base=2) == frozenset({(0, 1, 0), (1, 1, 0)})
    schedule = CellSchedule(
        chunks=1,
        num_denoise_steps=1,
        step_replicas=1,
        layer_groups=1,
        ordering=Ordering.SELF_FORCING,
        kv_history_chunks=1,
        chunk_base=2,
    )
    assert clean_history_cells(schedule, num_blocks=1) == {(2, 1, 0)}
    same = CellSchedule(
        chunks=1,
        num_denoise_steps=2,
        step_replicas=2,
        layer_groups=1,
        ordering=Ordering.SAME_STEP,
        kv_history_chunks=1,
        chunk_base=3,
    )
    # Trailing window keeps chunk 3; Same-Step also pins absolute chunk-0 clean.
    assert clean_history_cells(same, num_blocks=1) == {(0, 2, 0), (3, 2, 0)}


def test_session_structural_key_fail_closed_on_mismatch():
    cache = _cache(history=1)
    first = CellSchedule(
        chunks=1,
        num_denoise_steps=1,
        step_replicas=1,
        layer_groups=1,
        ordering=Ordering.SELF_FORCING,
        kv_history_chunks=1,
    )
    cache.bind_session_structure("world", first.structural_key())
    cache.bind_session_structure("world", first.structural_key())  # idempotent
    changed = CellSchedule(
        chunks=1,
        num_denoise_steps=2,
        step_replicas=1,
        layer_groups=1,
        ordering=Ordering.SELF_FORCING,
        kv_history_chunks=1,
    )
    with pytest.raises(ValueError, match="structural_key mismatch"):
        cache.bind_session_structure("world", changed.structural_key())
    NoisyKVState(cache).release_session("world")
    cache.bind_session_structure("world", changed.structural_key())


def test_request_chunk_base_continues_from_pool_history():
    cache = _cache(history=1)
    kv = NoisyKVState(cache)
    ctx = ARDiffusionCellContext(
        spec=CellRunSpec(topology=CellTopology(step_replicas=1, layer_groups=1), rank=0),
        kv=kv,
    )

    first = CellSchedule(
        chunks=2,
        num_denoise_steps=1,
        step_replicas=1,
        layer_groups=1,
        ordering=Ordering.SELF_FORCING,
        kv_history_chunks=1,
    )
    plan = build_cell_plan(first)
    cache.bind_session_structure("world", first.structural_key())
    retain = clean_history_cells(first, num_blocks=1)
    ctx.enqueue("world", plan, chunk_tokens=4, retain=retain)
    run_cell_pipeline(ctx=ctx, adapter=_CountingAdapter())
    kv.end_request("world", retain=retain)
    assert cache.pool.has(("world", 1, 1, 0))
    assert not cache.pool.has(("world", 0, 1, 0))

    second = CellSchedule(
        chunks=1,
        num_denoise_steps=1,
        step_replicas=1,
        layer_groups=1,
        ordering=Ordering.SELF_FORCING,
        kv_history_chunks=1,
        chunk_base=2,
    )
    prior = prior_history_in_pool(cache, "world", chunk_base=2)
    assert prior
    cache.bind_session_structure("world", second.structural_key())
    plan2 = build_cell_plan(second, prior_history=prior)
    retain2 = set(prior) | clean_history_cells(second, num_blocks=1)
    ctx.enqueue("world", plan2, chunk_tokens=4, retain=retain2)
    run_cell_pipeline(ctx=ctx, adapter=_CountingAdapter())
    kept = clean_history_cells(second, num_blocks=1)
    kv.end_request("world", retain=kept)
    assert cache.pool.has(("world", 2, 1, 0))
    assert not cache.pool.has(("world", 1, 1, 0))


def _bare_noisy_runner() -> ARDiffusionModelRunner:
    runner = object.__new__(ARDiffusionModelRunner)
    runner.od_config = SimpleNamespace(max_num_seqs=1, dtype=torch.float32, ar_diffusion_stage_config={})
    runner.device = torch.device("cpu")
    runner.pipeline = object()
    runner.ar_diffusion_kv_config = ARDiffusionKVConfig(enable=True)
    runner.kv_cache = None
    runner.noisy_kv_cache = _cache()
    runner._ar_diffusion_capability = None
    runner._ar_diffusion_cell_capability = SimpleNamespace(bind_ar_diffusion_cell_context=lambda ctx: _NullCM(ctx))
    runner._ar_diffusion_kv_cache_spec = None
    runner._sessions = OrderedDict()
    runner._session_capacity = 0
    runner._perf_e2e_times = []
    runner._stepwise_chunk_started = {}
    runner._active_stepwise_request_id = None
    return runner


def test_same_step_end_request_keeps_chunk0_clean_sink():
    cache = _cache(history=1)
    state = NoisyKVState(cache)
    schedule = CellSchedule(
        chunks=2,
        num_denoise_steps=2,
        step_replicas=2,
        layer_groups=1,
        ordering=Ordering.SAME_STEP,
        kv_history_chunks=1,
        chunk_base=3,
    )
    # Simulate resident sink + recent clean after a continuation request.
    cache.pool.alloc(("world", 0, 2, 0))
    cache.pool.alloc(("world", 3, 2, 0))
    cache.pool.alloc(("world", 2, 2, 0))  # outside trailing window; should drop
    kept = clean_history_cells(schedule, num_blocks=1)
    state.end_request("world", retain=kept)
    assert cache.pool.has(("world", 0, 2, 0))
    assert cache.pool.has(("world", 3, 2, 0))
    assert not cache.pool.has(("world", 2, 2, 0))


def test_runner_ephemeral_request_releases_pool_key(monkeypatch):
    from vllm_omni.diffusion.worker import diffusion_model_runner as dmr

    runner = _bare_noisy_runner()
    released: list[str] = []
    monkeypatch.setattr(runner, "release_noisy_kv", lambda kv_id: released.append(kv_id))
    monkeypatch.setattr(
        dmr.DiffusionModelRunner,
        "execute_model",
        lambda self, req, kv_prefetch_job=None, diffusion_kv_metadata=None: SimpleNamespace(ok=True),
    )

    req = OmniDiffusionRequest(
        prompt="x",
        request_id="ephemeral-1",
        sampling_params=OmniDiffusionSamplingParams(),
    )
    out = ARDiffusionModelRunner.execute_model(runner, req)
    assert out.ok is True
    assert released == ["ephemeral-1"]


def test_runner_persistent_session_id_leaves_pool_to_outer_owner(monkeypatch):
    from vllm_omni.diffusion.worker import diffusion_model_runner as dmr

    runner = _bare_noisy_runner()
    released: list[str] = []
    monkeypatch.setattr(runner, "release_noisy_kv", lambda kv_id: released.append(kv_id))
    monkeypatch.setattr(
        dmr.DiffusionModelRunner,
        "execute_model",
        lambda self, req, kv_prefetch_job=None, diffusion_kv_metadata=None: SimpleNamespace(ok=True),
    )

    req = OmniDiffusionRequest(
        prompt="x",
        request_id="req-1",
        sampling_params=OmniDiffusionSamplingParams(extra_args={"session_id": "world", "chunk_base": 0}),
    )
    out = ARDiffusionModelRunner.execute_model(runner, req)
    assert out.ok is True
    assert released == []
    assert runner._sessions == OrderedDict()


def test_runner_release_noisy_kv_is_outer_lifecycle_hook():
    runner = _bare_noisy_runner()
    runner.noisy_kv_cache.pool.alloc(("world", 1, 1, 0))
    runner.release_noisy_kv("world")
    assert not runner.noisy_kv_cache.pool.has(("world", 1, 1, 0))


class _NullCM:
    def __init__(self, ctx):
        self.ctx = ctx

    def __enter__(self):
        return self.ctx

    def __exit__(self, *args):
        return False
