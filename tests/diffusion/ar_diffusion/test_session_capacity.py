# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Session capacity: the scheduler's grant and the worker's report are one loop."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import ARDiffusionNoisyKVSpec, NoisyKVCache, NoisyKVState
from vllm_omni.experimental.ar_diffusion.reservation import (
    CapacityDemand,
    CapacityUsage,
    ReservationDeniedError,
    ReservationOvercommitError,
    SessionCapacityLedger,
)
from vllm_omni.experimental.ar_diffusion.runner import ARDiffusionModelRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_ledger_reconcile_accepts_usage_within_the_grant() -> None:
    ledger = SessionCapacityLedger(budget_bytes=1000)
    ledger.reserve("s1", CapacityDemand(versions=6, bytes=400))

    reported = ledger.reconcile("s1", CapacityUsage(versions=4, bytes=300))

    assert reported == CapacityUsage(versions=4, bytes=300)
    assert ledger.usage("s1") == CapacityUsage(versions=4, bytes=300)
    assert ledger.overcommit("s1") is None
    assert ledger.held_bytes("s1") == 400


def test_ledger_reconcile_fails_closed_on_overcommit() -> None:
    ledger = SessionCapacityLedger(budget_bytes=1000)
    ledger.reserve("s1", CapacityDemand(versions=6, bytes=400))

    with pytest.raises(ReservationOvercommitError):
        ledger.reconcile("s1", CapacityUsage(versions=7, bytes=500))

    assert ledger.overcommit("s1") == CapacityUsage(versions=7, bytes=500)
    assert ledger.usage("s1") is None


def test_release_forgets_usage_and_overcommit() -> None:
    ledger = SessionCapacityLedger(budget_bytes=1000)
    ledger.reserve("s1", CapacityDemand(versions=6, bytes=400))
    with pytest.raises(ReservationOvercommitError):
        ledger.reconcile("s1", CapacityUsage(versions=7, bytes=500))

    ledger.release("s1")

    assert ledger.usage("s1") is None
    assert ledger.overcommit("s1") is None
    assert ledger.free_bytes() == 1000


def test_a_demand_beyond_the_budget_is_denied() -> None:
    ledger = SessionCapacityLedger(budget_bytes=100)

    with pytest.raises(ReservationDeniedError):
        ledger.reserve("s1", CapacityDemand(versions=6, bytes=400))


def _cache(*, num_layers: int = 2) -> NoisyKVCache:
    spec = ARDiffusionNoisyKVSpec(
        num_layers=num_layers,
        num_kv_heads=1,
        head_size=4,
        block_size=4,
        max_chunk_tokens=4,
        max_history_chunks=1,
    )
    return NoisyKVCache(spec, dtype=torch.float32, device=torch.device("cpu"), layer_groups=1, max_batch_size=1)


def _runner(ledger: SessionCapacityLedger | None, usage: tuple[int, int] | None = None) -> ARDiffusionModelRunner:
    runner = object.__new__(ARDiffusionModelRunner)
    runner.noisy_kv_cache = _cache() if usage is None else SimpleNamespace(session_usage=lambda _kv_id: usage)
    runner._capacity_ledger = ledger
    return runner


def test_session_usage_counts_each_pool_key_once() -> None:
    """A key holds one block's layer; a full pool reports exactly its reservation."""
    cache = _cache(num_layers=2)
    for layer in range(cache.spec.num_layers):
        for slot in range(cache.capacity):
            cache.pool.alloc(("s1", slot, 1, layer))

    versions, nbytes = cache.session_usage("s1")

    assert versions == cache.capacity * cache.spec.num_layers
    assert nbytes == cache.reserved_bytes
    assert cache.session_usage("other") == (0, 0)


def test_the_runner_reports_worker_usage_after_a_batch() -> None:
    ledger = SessionCapacityLedger(budget_bytes=1000)
    ledger.reserve("s1", CapacityDemand(versions=6, bytes=400))
    runner = _runner(ledger, (3, 200))

    runner._report_session_capacity("s1")

    assert ledger.usage("s1") == CapacityUsage(versions=3, bytes=200)
    assert ledger.overcommit("s1") is None


def test_a_worker_overcommit_fails_the_request_and_drops_its_history() -> None:
    runner = _runner(SessionCapacityLedger(budget_bytes=10**9))
    cache = runner.noisy_kv_cache
    cache.pool.alloc(("s1", 0, 1, 0))
    cache.pool.alloc(("s1", 1, 1, 0))
    cache.pool.alloc(("s2", 0, 1, 0))
    runner._capacity_ledger.reserve("s1", CapacityDemand(versions=1, bytes=cache.bytes_per_version))

    with pytest.raises(ReservationOvercommitError):
        runner._report_session_capacity("s1")

    assert runner._capacity_ledger.overcommit("s1") == CapacityUsage(versions=2, bytes=2 * cache.bytes_per_version)
    assert cache.session_usage("s1") == (0, 0), "an overcommitted session kept its history"
    assert cache.pool.has(("s2", 0, 1, 0)), "another session's history was dropped"


def test_teardown_reporting_never_raises() -> None:
    """Releasing a key must not turn a drifted grant into a second failure."""
    ledger = SessionCapacityLedger(budget_bytes=1000)
    ledger.reserve("s1", CapacityDemand(versions=6, bytes=400))
    runner = _runner(ledger, (9, 900))

    runner._report_session_capacity("s1", fail_closed=False)

    assert ledger.overcommit("s1") == CapacityUsage(versions=9, bytes=900)


def test_releasing_an_unreserved_key_leaves_no_ledger_entry() -> None:
    """Ephemeral request ids are released after every forward; none may accumulate."""
    ledger = SessionCapacityLedger(budget_bytes=1000)
    runner = _runner(ledger)

    runner.release_noisy_kv("ephemeral-1")

    assert ledger.usage("ephemeral-1") is None
    assert ledger._usage == {}


def test_releasing_a_reserved_key_reports_the_empty_holding() -> None:
    ledger = SessionCapacityLedger(budget_bytes=10**9)
    runner = _runner(ledger)
    runner.noisy_kv_cache.pool.alloc(("s1", 0, 1, 0))
    ledger.reserve("s1", CapacityDemand(versions=1, bytes=runner.noisy_kv_cache.bytes_per_version))

    runner.release_noisy_kv("s1")

    assert ledger.usage("s1") == CapacityUsage(versions=0, bytes=0)


def test_a_batch_wave_reports_each_retained_session(monkeypatch) -> None:
    from vllm_omni.diffusion.worker import diffusion_model_runner as dmr

    ledger = SessionCapacityLedger(budget_bytes=10**9)
    runner = _runner(ledger)
    cache = runner.noisy_kv_cache
    runner._ar_diffusion_cell_capability = SimpleNamespace(bind_ar_diffusion_cell_context=lambda _ctx: nullcontext())
    monkeypatch.setattr(runner, "_make_chunk_context", lambda *, batch=None: SimpleNamespace(kv=NoisyKVState(cache)))
    ledger.reserve("s1", CapacityDemand(versions=1, bytes=cache.bytes_per_version))
    ledger.reserve("s2", CapacityDemand(versions=1, bytes=cache.bytes_per_version))

    def _forward(self, scheduler_output, od_config):
        cache.pool.alloc(("s1", 0, 1, 0))
        cache.pool.alloc(("s2", 0, 1, 0))
        cache.pool.alloc(("s2", 1, 1, 0))
        return "out"

    monkeypatch.setattr(dmr.DiffusionModelRunner, "execute_model_batch", _forward)

    def _req(request_id: str, **extra) -> SimpleNamespace:
        return SimpleNamespace(
            req=OmniDiffusionRequest(
                prompt="x", request_id=request_id, sampling_params=OmniDiffusionSamplingParams(extra_args=extra)
            )
        )

    wave = SimpleNamespace(scheduled_new_reqs=[_req("r1", session_id="s1"), _req("r2", session_id="s2"), _req("r3")])
    with pytest.raises(ReservationOvercommitError):
        runner.execute_model_batch(wave, od_config=SimpleNamespace())

    # s1 fit its grant and was reconciled; s2 held two versions on a one-version grant.
    assert ledger.usage("s1") == CapacityUsage(versions=1, bytes=cache.bytes_per_version)
    assert ledger.overcommit("s2") is not None
    assert cache.pool.has(("s1", 0, 1, 0))
    assert cache.session_usage("s2") == (0, 0)


def test_without_a_ledger_nothing_is_reported() -> None:
    runner = _runner(None, (9, 900))

    runner._report_session_capacity("s1")
