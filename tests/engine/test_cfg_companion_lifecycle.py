# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic tests for the CFG-companion output lifecycle.

Regression suite for the companion-output race behind the nightly Bagel
failures: ``kv_ready`` used to mark a companion done before its processed
output was stashed, letting the parent dispatch to the diffusion stage with
0/N companion outputs (degraded CFG conditioning, and after an abort a TP
rank desync). The invariant under test: **dispatch to the diffusion stage
never happens with an incomplete companion bundle** — the parent is
re-deferred while outputs are pending and failed when the bundle can no
longer complete.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from vllm_omni.engine.cfg_companion_tracker import CfgCompanionTracker
from vllm_omni.engine.messages import AbortRequestMessage, ErrorMessage
from vllm_omni.engine.orchestrator import Orchestrator

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ───────── tracker-level ordering semantics ─────────


def test_companion_first_then_parent_is_not_deferred():
    tracker = CfgCompanionTracker()
    tracker.register_companion("p", "neg", "p__neg")
    tracker.set_companion_output("p__neg", {"o": 1})
    assert tracker.on_companion_completed("p__neg") is None  # parent not deferred yet
    # Parent arrives afterwards: bundle already complete, no defer needed.
    assert tracker.all_companions_done("p")
    assert tracker.get_companion_outputs("p") == [{"o": 1}]


def test_parent_first_deferred_then_companion_flushes():
    tracker = CfgCompanionTracker()
    tracker.register_companion("p", "neg", "p__neg")
    tracker.defer_parent("p", {"parent": True}, stage_id=0)
    tracker.set_companion_output("p__neg", {"o": 1})
    assert tracker.on_companion_completed("p__neg") == "p"
    deferred = tracker.pop_pending_parent("p")
    assert deferred is not None and deferred["engine_outputs"] == {"parent": True}


def test_duplicate_companion_completion_is_idempotent():
    tracker = CfgCompanionTracker()
    tracker.register_companion("p", "neg", "p__neg")
    tracker.defer_parent("p", {}, stage_id=0)
    tracker.set_companion_output("p__neg", {"o": 1})
    assert tracker.on_companion_completed("p__neg") == "p"
    assert tracker.on_companion_completed("p__neg") is None  # duplicate: no re-flush
    assert tracker.get_companion_outputs("p") == [{"o": 1}]


def test_bundle_survives_resubmission_until_cleanup():
    """get_companion_outputs is non-destructive: a streaming re-submission of
    the parent bundles the complete set again (never a partial/empty one).
    Only cleanup_parent releases the outputs."""
    tracker = CfgCompanionTracker()
    tracker.register_companion("p", "neg", "p__neg")
    tracker.set_companion_output("p__neg", {"o": 1})
    assert tracker.get_companion_outputs("p") == [{"o": 1}]
    assert tracker.get_companion_outputs("p") == [{"o": 1}]  # re-submit: full bundle again
    tracker.cleanup_parent("p")
    assert tracker.get_companion_outputs("p") == []


# ───────── orchestrator-method-level lifecycle ─────────


class _FakePool:
    """Minimal stage pool satisfying _forward_to_next_stage's preamble and cleanup."""

    def __init__(self, stage_type: str = "diffusion") -> None:
        self.stage_type = stage_type
        self.stage_client = SimpleNamespace(requires_multimodal_data=False, custom_process_input_func=None)
        self.aborted: list[list[str]] = []

    def live_replica_ids(self) -> list[int]:
        # Single always-live replica: these tests cover CFG bundling, not
        # replica death, and the forward path fast-fails on an empty pool.
        return [0]

    async def abort_requests(self, request_ids):
        self.aborted.append(list(request_ids))

    def release_bindings(self, request_ids):
        pass


def _make_orchestrator(num_stages: int = 2) -> Orchestrator:
    orch = Orchestrator.__new__(Orchestrator)
    orch._cfg_tracker = CfgCompanionTracker()
    orch.request_states = {}
    orch.stage_pools = [_FakePool("llm"), _FakePool("diffusion")][:num_stages]
    orch.output_async_queue = asyncio.Queue()
    orch.async_chunk = False
    orch.duplex_control_plane = None
    orch._pd_kv_params = {}
    orch._pd_pair = None
    orch._running_counter = None
    orch._transfer_emitter = None
    return orch


def _req_state(final_stage_id: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        final_stage_id=final_stage_id,
        stage_submit_ts={},
        sampling_params_list=[SimpleNamespace(output_kind=None)] * (final_stage_id + 1),
        streaming=SimpleNamespace(enabled=False, segment_finished=False),
        running_counter_registered=False,
        prompt={"prompt": "x"},
        pipeline_timings={},
    )


@pytest.mark.asyncio
async def test_kv_ready_does_not_mark_companion_done():
    """kv_ready alone must not count a companion as done — only the processed
    output (set_companion_output) may. Guards the original race."""
    orch = _make_orchestrator()
    orch._cfg_tracker.register_companion("p", "neg", "p__neg")
    orch.request_states["p__neg"] = _req_state()

    raw = SimpleNamespace(request_id="p__neg", kv_transfer_params={"kv_ready": True})
    await orch._handle_kv_ready_raw_outputs(0, SimpleNamespace(outputs=[raw]))

    assert not orch._cfg_tracker.all_companions_done("p")
    assert orch._cfg_tracker.pop_pending_parent("p") is None


@pytest.mark.asyncio
async def test_parent_kv_ready_defers_until_companion_outputs_stashed():
    """Parent kv_ready with an incomplete bundle defers; the companion's
    processed output later flushes it (companion-first vs parent-first order
    both funnel through the same defer/flush path)."""
    orch = _make_orchestrator()
    orch._cfg_tracker.register_companion("p", "neg", "p__neg")
    orch.request_states["p"] = _req_state()

    raw = SimpleNamespace(request_id="p", kv_transfer_params={"kv_ready": True})
    await orch._handle_kv_ready_raw_outputs(0, SimpleNamespace(outputs=[raw]))

    deferred = orch._cfg_tracker.pop_pending_parent("p")
    assert deferred is not None and deferred["stage_id"] == 0


@pytest.mark.asyncio
async def test_forward_re_defers_when_bundle_incomplete():
    """The forward gate re-defers (non-destructively) instead of dispatching
    with 0/N companion outputs."""
    orch = _make_orchestrator()
    orch._cfg_tracker.register_companion("p", "neg", "p__neg")
    orch._cfg_tracker.set_companion_output("p__neg", {"o": 1})  # stashed but NOT done-marked
    state = _req_state()
    orch.request_states["p"] = state

    await orch._forward_to_next_stage("p", 0, {"parent": True}, state)

    # Not dispatched: re-deferred with the stashed output left intact.
    deferred = orch._cfg_tracker.pop_pending_parent("p")
    assert deferred is not None
    assert orch._cfg_tracker.get_companion_outputs("p") == [{"o": 1}]
    assert orch.output_async_queue.empty()


@pytest.mark.asyncio
async def test_forward_fails_request_when_bundle_lost():
    """Companions done but outputs gone without a prior bundle: the request is
    failed (terminal ErrorMessage + abort) instead of dispatching degraded CFG."""
    orch = _make_orchestrator()
    orch._cfg_tracker.register_companion("p", "neg", "p__neg")
    orch._cfg_tracker.set_companion_output("p__neg", {"o": 1})
    orch._cfg_tracker.on_companion_completed("p__neg")
    orch._cfg_tracker._companion_outputs.pop("p__neg")  # simulate lost output (unreachable normally)
    state = _req_state()
    orch.request_states["p"] = state

    await orch._forward_to_next_stage("p", 0, {"parent": True}, state)

    msg = orch.output_async_queue.get_nowait()
    assert isinstance(msg, ErrorMessage)
    assert msg.request_id == "p"
    assert "incomplete" in msg.error
    assert "p" not in orch.request_states
    assert any("p" in batch for pool in orch.stage_pools for batch in pool.aborted)


@pytest.mark.asyncio
async def test_companion_abort_fails_deferred_parent():
    """A companion aborted directly (parent not in the batch) can never
    complete the bundle — the deferred parent is failed immediately."""
    orch = _make_orchestrator()
    orch._cfg_tracker.register_companion("p", "neg", "p__neg")
    orch._cfg_tracker.defer_parent("p", {"parent": True}, stage_id=0)
    orch.request_states["p"] = _req_state()

    await orch._handle_abort(AbortRequestMessage(request_ids=["p__neg"]))

    msg = orch.output_async_queue.get_nowait()
    assert isinstance(msg, ErrorMessage)
    assert msg.request_id == "p"
    assert "p__neg" in msg.error
    assert "p" not in orch.request_states
    assert not orch._cfg_tracker.is_companion("p__neg")
    # Idempotent: a repeated abort of the same companion is a no-op.
    await orch._handle_abort(AbortRequestMessage(request_ids=["p__neg"]))
    assert orch.output_async_queue.empty()


@pytest.mark.asyncio
async def test_companion_abort_after_completion_does_not_fail_parent():
    """A late/duplicate abort of a companion whose output is already stashed
    must not cancel the (possibly already dispatched) parent."""
    orch = _make_orchestrator()
    orch._cfg_tracker.register_companion("p", "neg", "p__neg")
    orch._cfg_tracker.set_companion_output("p__neg", {"o": 1})
    orch._cfg_tracker.on_companion_completed("p__neg")
    orch.request_states["p"] = _req_state()

    await orch._handle_abort(AbortRequestMessage(request_ids=["p__neg"]))

    assert orch.output_async_queue.empty()  # no parent failure emitted
    assert "p" in orch.request_states  # parent untouched
    assert orch._cfg_tracker.get_companion_outputs("p") == [{"o": 1}]  # bundle intact


class _CapturedDispatchError(Exception):
    """Sentinel raised by the capture hook to stop the forward after bundling."""


@pytest.mark.asyncio
async def test_repeated_forward_bundles_full_set_again():
    """A second forward of the same parent (streaming terminal update) must
    dispatch the complete [parent, *companions] bundle again — never [parent]
    alone. Captured at the diffusion input hook, i.e. actual dispatch content."""
    orch = _make_orchestrator()
    orch._cfg_tracker.register_companion("p", "neg", "p__neg")
    orch._cfg_tracker.set_companion_output("p__neg", {"o": 1})
    orch._cfg_tracker.on_companion_completed("p__neg")
    state = _req_state()
    orch.request_states["p"] = state

    captured: list[list] = []

    def _capture(source_outputs, prompt, requires_multimodal_data):
        captured.append(list(source_outputs))
        raise _CapturedDispatchError

    orch.stage_pools[1].stage_client.custom_process_input_func = _capture

    for _ in range(2):
        with pytest.raises(_CapturedDispatchError):
            await orch._forward_to_next_stage("p", 0, {"parent": True}, state)

    assert captured == [[{"parent": True}, {"o": 1}], [{"parent": True}, {"o": 1}]]


@pytest.mark.asyncio
async def test_replica_loss_cleanup_releases_parent_tracker_state():
    """Replica-loss/membership teardown calls _cleanup_request_ids directly
    (no cleanup_parent at the call site): the parent's tracker state must be
    released and its companions folded into the same cleanup batch."""
    orch = _make_orchestrator()
    orch._cfg_tracker.register_companion("p", "neg", "p__neg")
    orch._cfg_tracker.set_companion_output("p__neg", {"o": 1})
    orch._cfg_tracker.defer_parent("p", {"parent": True}, stage_id=0)
    orch.request_states["p"] = _req_state()

    await orch._cleanup_request_ids(["p"], abort=True)

    assert not orch._cfg_tracker.has_companions("p")
    assert not orch._cfg_tracker.is_companion("p__neg")
    assert orch._cfg_tracker.get_companion_outputs("p") == []
    assert orch._cfg_tracker.pop_pending_parent("p") is None
    aborted = {rid for pool in orch.stage_pools for b in pool.aborted for rid in b}
    assert {"p", "p__neg"} <= aborted


@pytest.mark.asyncio
async def test_replica_loss_of_companion_fails_deferred_parent():
    """A companion torn down by replica loss before its output arrived means
    the parent can never complete its bundle — the cleanup fails the parent."""
    orch = _make_orchestrator()
    orch._cfg_tracker.register_companion("p", "neg", "p__neg")
    orch._cfg_tracker.defer_parent("p", {"parent": True}, stage_id=0)
    orch.request_states["p"] = _req_state()

    await orch._cleanup_request_ids(["p__neg"])

    msg = orch.output_async_queue.get_nowait()
    assert isinstance(msg, ErrorMessage)
    assert msg.request_id == "p"
    assert "p__neg" in msg.error
    assert "p" not in orch.request_states
    assert not orch._cfg_tracker.has_companions("p")


@pytest.mark.asyncio
async def test_finished_companion_cleanup_preserves_parent_bundle():
    """The normal companion-finished cleanup (_route_output cleans just the
    companion id) must not disturb the parent's stashed bundle."""
    orch = _make_orchestrator()
    orch._cfg_tracker.register_companion("p", "neg", "p__neg")
    orch._cfg_tracker.set_companion_output("p__neg", {"o": 1})
    orch._cfg_tracker.on_companion_completed("p__neg")
    orch.request_states["p"] = _req_state()

    await orch._cleanup_request_ids(["p__neg"])

    assert orch.output_async_queue.empty()
    assert "p" in orch.request_states
    assert orch._cfg_tracker.get_companion_outputs("p") == [{"o": 1}]


@pytest.mark.asyncio
async def test_parent_abort_still_expands_to_companions():
    """Regression-guard the existing behavior: aborting the parent aborts its
    companions and clears tracker state, without emitting an error."""
    orch = _make_orchestrator()
    orch._cfg_tracker.register_companion("p", "neg", "p__neg")
    orch._cfg_tracker.defer_parent("p", {"parent": True}, stage_id=0)
    orch.request_states["p"] = _req_state()

    await orch._handle_abort(AbortRequestMessage(request_ids=["p"]))

    assert orch.output_async_queue.empty()
    assert "p" not in orch.request_states
    assert not orch._cfg_tracker.is_companion("p__neg")
    aborted = {rid for pool in orch.stage_pools for batch in pool.aborted for rid in batch}
    assert {"p", "p__neg"} <= aborted
