# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A forked parent must outlive its children, and an abort must reach them.

Duo's H50 handoff runs every Intent block as its own request forked from the
session's root request, and the root owns the Intent-side KV that the *next*
block reads. So the root has to survive each block retiring, while an abort or
session teardown has to take the whole family with it.

The previous implementation derived "parents to protect" from
``request_states`` *after* ``super()._cleanup_request_ids`` had already popped
every id in the batch, so a parent in the batch was gone before it could be
protected and the next ``fork_stage_request`` hit a KeyError. The abort path
skipped the computation entirely. These tests pin the behaviour instead of the
presence of a field.
"""

import ast
import dataclasses
import inspect

import pytest

from vllm_omni.engine.duplex_orchestrator import DuplexOrchestrator, DuplexOrchestratorRequestState
from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PARENT = "sess-root"


class _StubSessionManager:
    """Records unregister calls; the real manager needs a live session."""

    def __init__(self) -> None:
        self.unregistered: list[str] = []

    def unregister_request(self, request_id: str) -> None:
        self.unregistered.append(request_id)


def _build_orchestrator() -> DuplexOrchestrator:
    """A DuplexOrchestrator with only the state cleanup touches."""
    orch = object.__new__(DuplexOrchestrator)
    orch.request_states = {}
    orch.session_manager = _StubSessionManager()
    orch._fork_parents = set()
    orch._fork_children = {}
    return orch


def _add_request(orch: DuplexOrchestrator, request_id: str, parent_id: str | None = None) -> None:
    state = DuplexOrchestratorRequestState(request_id=request_id)
    state.parent_request_id = parent_id
    orch.request_states[request_id] = state


def _fork(orch: DuplexOrchestrator, parent_id: str, child_id: str) -> None:
    """Record a fork without needing a live session/runner."""
    _add_request(orch, child_id, parent_id=parent_id)
    orch._fork_parents.add(parent_id)
    orch._fork_children.setdefault(parent_id, set()).add(child_id)


@pytest.fixture
def base_cleanup_calls(monkeypatch):
    """Replace the base cleanup with a request_states-popping stand-in."""
    calls: list[list[str]] = []

    async def _fake_cleanup(self, request_ids, *, abort=False, release_owners=False, trace_error=None):
        calls.append(list(request_ids))
        for request_id in list(request_ids):
            self.request_states.pop(request_id, None)
        return []

    monkeypatch.setattr(Orchestrator, "_cleanup_request_ids", _fake_cleanup)
    return calls


@pytest.mark.asyncio
async def test_retiring_child_keeps_the_parent_alive(base_cleanup_calls):
    # The block that just finished is cleaned; the root must survive so the next
    # block can fork from it again.
    orch = _build_orchestrator()
    _add_request(orch, PARENT)
    child = f"{PARENT}-block0"
    _fork(orch, PARENT, child)

    await orch._cleanup_request_ids([child])

    assert PARENT in orch.request_states, "a retiring child must not retire the parent it was forked from"
    assert PARENT in orch._fork_parents
    assert child not in orch.request_states
    assert base_cleanup_calls == [[child]], "the parent must be filtered before the base cleanup runs"


@pytest.mark.asyncio
async def test_parent_survives_and_can_be_forked_again(base_cleanup_calls):
    # The regression the reviewers flagged: after a block retires, forking the
    # next block from the same root has to work.
    orch = _build_orchestrator()
    _add_request(orch, PARENT)
    _fork(orch, PARENT, f"{PARENT}-block0")

    await orch._cleanup_request_ids([f"{PARENT}-block0"])

    assert orch.request_states.get(PARENT) is not None
    # Emulate the next block's fork bookkeeping against the surviving parent.
    _fork(orch, PARENT, f"{PARENT}-block1")
    assert orch.request_states.get(PARENT) is not None


@pytest.mark.asyncio
async def test_cleanup_batch_with_only_the_parent_is_a_noop(base_cleanup_calls):
    # A streaming session can submit the root for cleanup on its own. Dropping
    # it silently would break the next fork, so the whole batch is held back.
    orch = _build_orchestrator()
    _add_request(orch, PARENT)
    _fork(orch, PARENT, f"{PARENT}-block0")

    await orch._cleanup_request_ids([PARENT])

    assert PARENT in orch.request_states
    assert base_cleanup_calls == [], "nothing should reach the base cleanup when the whole batch is protected"


@pytest.mark.asyncio
async def test_parent_and_child_in_the_same_batch_keeps_the_parent(base_cleanup_calls):
    # The realistic shape of the bug: the session submits the root and the block
    # that just finished together. The old code popped the parent inside
    # ``super()`` and only then looked for parents to protect, so it protected
    # nothing and the next fork raised KeyError.
    orch = _build_orchestrator()
    _add_request(orch, PARENT)
    child = f"{PARENT}-block0"
    _fork(orch, PARENT, child)

    await orch._cleanup_request_ids([PARENT, child])

    assert PARENT in orch.request_states, "a parent batched with its own child must still survive"
    assert child not in orch.request_states, "the retiring child is still cleaned"
    assert base_cleanup_calls == [[child]]


@pytest.mark.asyncio
async def test_abort_cascades_to_in_flight_children(base_cleanup_calls):
    # Aborting the root must abort the children that are still streaming, or
    # they stay registered against a session that is being torn down.
    orch = _build_orchestrator()
    _add_request(orch, PARENT)
    for block in range(3):
        _fork(orch, PARENT, f"{PARENT}-block{block}")

    await orch._cleanup_request_ids([PARENT], abort=True)

    cleaned = set(base_cleanup_calls[0])
    assert {PARENT, *(f"{PARENT}-block{b}" for b in range(3))} <= cleaned, (
        f"abort must cascade to forked children, cleaned={sorted(cleaned)}"
    )
    assert not orch.request_states, "abort tears down the whole family"
    assert not orch._fork_children
    assert not orch._fork_parents


@pytest.mark.asyncio
async def test_unrelated_requests_are_untouched(base_cleanup_calls):
    # The fork bookkeeping must not widen or shrink ordinary cleanup.
    orch = _build_orchestrator()
    _add_request(orch, PARENT)
    _fork(orch, PARENT, f"{PARENT}-block0")
    _add_request(orch, "plain")

    await orch._cleanup_request_ids(["plain"])

    assert "plain" not in orch.request_states
    assert PARENT in orch.request_states
    assert base_cleanup_calls == [["plain"]]


def test_fork_stage_request_reports_a_missing_parent():
    # Reaching a fork with no live parent means the session was torn down. Say
    # so, instead of surfacing a bare KeyError from the dict lookup.
    orch = _build_orchestrator()
    with pytest.raises(RuntimeError, match="no longer registered"):
        orch.fork_stage_request(PARENT, f"{PARENT}-block7", stage_id=2)


def test_abort_requests_routes_through_cleanup():
    # Duo's abort used to call the stage-level abort directly, which bypasses the
    # fork cascade entirely. Check the call graph, not the prose: a comment
    # naming the old helper must not make this pass or fail.
    tree = ast.parse(inspect.getsource(DuplexOrchestrator.abort_requests).lstrip())
    called = {
        node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "_cleanup_request_ids" in called
    assert "_abort_request_ids" not in called


def test_parent_request_id_is_declared_on_the_request_state():
    # The plugin sets this on a forked child; it is the documented edge between
    # the two requests, so it has to stay a real dataclass field.
    state = DuplexOrchestratorRequestState(request_id="x")
    assert state.parent_request_id is None
    assert "parent_request_id" in {f.name for f in dataclasses.fields(DuplexOrchestratorRequestState)}
    assert "parent_request_id" not in {f.name for f in dataclasses.fields(OrchestratorRequestState)}


def test_fork_stage_request_marks_the_child_as_forked_on_its_session(monkeypatch):
    # Regression for the duplex output boundary: the mailbox may only accept a
    # child that the session itself forked, so `fork_stage_request` has to hand
    # the session that mark. Driving the real method (with a session stub in
    # place of a live runner) is what makes this catch a missing `forked=True`;
    # calling `reserve_stage_request` directly would pass either way.
    from vllm_omni.engine.duplex_orchestrator import DuplexStageRequestContext

    orch = _build_orchestrator()
    _add_request(orch, PARENT)
    parent = orch.request_states[PARENT]
    parent.session_id = "sess"
    parent.fence = "fence"
    parent.final_stage_id = 3
    parent.config_generation = 0
    parent.sampling_params_list = ()
    parent.streaming = type("S", (), {"bridge_states": {"duplex": {"session_config": {}, "runtime_config": {}}}})()

    reservations: list[tuple[int, str, bool]] = []

    class _Session:
        def reserve_stage_request(self, stage_id, request_id, *, fence, forked=False):
            reservations.append((stage_id, request_id, forked))

    runner = type("R", (), {"session": _Session()})()
    orch.session_manager = type("M", (), {"runner_for_request_id": lambda self, rid: runner})()

    def _ensure_request(self, context: DuplexStageRequestContext) -> None:
        orch.request_states[context.request_id] = DuplexOrchestratorRequestState(request_id=context.request_id)

    monkeypatch.setattr(DuplexOrchestrator, "ensure_request", _ensure_request, raising=False)

    child = orch.fork_stage_request(PARENT, "sess-block0", 1)

    assert child.parent_request_id == PARENT
    assert reservations == [(1, "sess-block0", True)], (
        "fork_stage_request must reserve the child as forked, otherwise the "
        f"output mailbox drops its stage output (got {reservations})"
    )
