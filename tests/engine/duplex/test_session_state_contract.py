# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Every duplex model session state carries the runner's full flag contract.

``DuplexModelSessionState`` only annotates its fields; a plugin state that
subclasses the ABC directly (Qwen3-Omni, AURA) must declare each one itself.
The session runner's real-append acceptance callback bumps
``native_input_generation`` on every accepted non-silence append, so a state
without that field fails the append (AttributeError) and closes the session.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
from types import SimpleNamespace

import pytest

from vllm_omni.engine.duplex.plugin import DuplexModelSessionState
from vllm_omni.engine.duplex.session import helpers
from vllm_omni.engine.duplex.session import runner as runner_mod
from vllm_omni.engine.duplex.session.runner import DuplexSessionRunner
from vllm_omni.model_executor.models.aura_omni.duplex.session import AuraServingSessionState
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.session import MiniCPMO45ServingSessionState
from vllm_omni.model_executor.models.qwen3_omni.duplex.session import QwenDuplexSessionState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

STATE_CLASSES = [QwenDuplexSessionState, AuraServingSessionState, MiniCPMO45ServingSessionState]
CONTRACT_FIELDS = sorted(inspect.get_annotations(DuplexModelSessionState))


@pytest.mark.parametrize("state_cls", STATE_CLASSES, ids=lambda cls: cls.__name__)
def test_state_declares_every_contract_field(state_cls: type[DuplexModelSessionState]) -> None:
    state = state_cls()
    declared = {f.name for f in dataclasses.fields(state_cls)}
    assert [name for name in CONTRACT_FIELDS if name not in declared] == []
    for name in CONTRACT_FIELDS:
        getattr(state, name)
    # Mutable defaults are per instance.
    assert state.pending_silence_tasks is not state_cls().pending_silence_tasks


@pytest.mark.parametrize("state_cls", STATE_CLASSES, ids=lambda cls: cls.__name__)
def test_clear_continuation_resets_silence_bookkeeping(state_cls: type[DuplexModelSessionState]) -> None:
    state = state_cls()
    state.last_native_submit_monotonic = 1.0
    state.silence_deadline_monotonic = 2.0
    state.pending_silence_tasks.append(object())  # type: ignore[arg-type]
    state.native_input_generation = 3
    state.silence_append_seq = 4
    state.clear_continuation()
    assert state.last_native_submit_monotonic is None
    assert state.silence_deadline_monotonic is None
    assert state.pending_silence_tasks == []
    assert state.native_input_generation == 0
    assert state.silence_append_seq == 0


class _AcceptingAttempt:
    """Stands in for ``AppendAttempt``: the stage accepts the append at t=123.0.

    Mirrors ``AppendAttempt._submit``: an exception from the acceptance
    callback fails the append and the session.
    """

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    async def run_in_wire_order(self, predecessor: object) -> bool:
        del predecessor
        try:
            self.kwargs["on_append_accepted"](123.0)  # type: ignore[operator]
        except Exception as exc:
            self.kwargs["fail_session"](f"runtime_append_task_failed: {exc!r}")  # type: ignore[operator]
            return False
        return True

    def release_on_failure(self, task: object) -> None:
        del task

    def clear_pending_silence(self, task: object) -> None:
        del task


def _fake_runner(model_state: DuplexModelSessionState, failures: list[str]) -> SimpleNamespace:
    session = SimpleNamespace(
        epoch=0,
        active_response_id=None,
        capabilities=SimpleNamespace(supports_core_resumable_request=True),
    )
    return SimpleNamespace(
        session=session,
        model_state=model_state,
        ctx=SimpleNamespace(manager=SimpleNamespace(stage_request_id=lambda fence, **kw: "sess.r.stage0")),
        out=None,
        model=None,
        tasks=SimpleNamespace(append_tail=None, track_append_task=lambda task, **kw: None),
        _mark_pending_silence_superseded=lambda: None,
        _fail_session_from_append=failures.append,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("state_cls", STATE_CLASSES, ids=lambda cls: cls.__name__)
async def test_accepted_real_append_reanchors_chain(
    state_cls: type[DuplexModelSessionState], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_mod, "AppendAttempt", _AcceptingAttempt)
    monkeypatch.setattr(helpers, "append_fence", lambda session, payload, *, epoch=None: SimpleNamespace(turn_id=0))
    state = state_cls()
    failures: list[str] = []
    fake = _fake_runner(state, failures)

    task = await DuplexSessionRunner._start_append(fake, {"type": "audio"}, final=False)  # type: ignore[arg-type]
    ok = await asyncio.wait_for(task, timeout=2.0)

    assert failures == []
    assert ok is True
    assert state.native_input_generation == 1
    assert state.last_native_submit_monotonic == 123.0
    assert state.silence_deadline_monotonic is None
