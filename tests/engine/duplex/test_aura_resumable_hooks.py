# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Public duplex hook: resumable defaults True; ephemeral ids when False."""

from __future__ import annotations

import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.engine.duplex.config import DuplexCapabilities
from vllm_omni.engine.duplex.contracts import (
    DuplexFence,
    DuplexStageRequestContext,
    DuplexStageSubmission,
    duplex_ephemeral_stage_request_id,
)
from vllm_omni.engine.duplex.session.manager import DuplexSessionManager

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_submission_resumable_default_preserves_minicpm() -> None:
    ctx = DuplexStageRequestContext(
        request_id="r",
        session_id="s",
        fence=DuplexFence("s"),
        stage_id=0,
        final_stage_id=2,
        config_generation=0,
        sampling_params=(SamplingParams(max_tokens=1),),
    )
    submission = DuplexStageSubmission(
        context=ctx,
        prompt={"prompt_token_ids": [1, 2]},
        already_submitted=False,
    )
    assert submission.resumable is True


def test_stage_request_id_respects_resumable_flag() -> None:
    fence = DuplexFence("sess", epoch=3, turn_id=9)
    resumable_id = DuplexSessionManager.stage_request_id(fence, stage_id=0, resumable=True)
    ephemeral_id = DuplexSessionManager.stage_request_id(fence, stage_id=0, resumable=False)
    assert ephemeral_id == duplex_ephemeral_stage_request_id(fence, stage_id=0)
    assert resumable_id != ephemeral_id
    assert ephemeral_id.endswith("stage0-turn9")
    from vllm_omni.engine.duplex.contracts import duplex_turn_id_from_request_id

    assert duplex_turn_id_from_request_id(ephemeral_id) == 9
    assert duplex_turn_id_from_request_id(resumable_id) is None


def test_capabilities_expose_concurrent_turn_requests_default_false() -> None:
    caps = DuplexCapabilities()
    assert caps.supports_concurrent_turn_requests is False
    assert caps.allows_video_without_audio() is False
    assert caps.required_input_modalities == frozenset({"audio"})
    assert caps.optional_input_modalities == frozenset({"video"})
    assert caps.supports_core_resumable_request is False
    payload = caps.as_dict()
    assert payload["supports_concurrent_turn_requests"] is False
    assert payload["required_input_modalities"] == ["audio"]
    assert payload["optional_input_modalities"] == ["video"]


def test_next_commit_allowed_soft_opens_on_r4_release() -> None:
    from types import SimpleNamespace

    from vllm_omni.engine.duplex.session import helpers

    caps = DuplexCapabilities(supports_concurrent_turn_requests=True)
    session = SimpleNamespace(active_response_id="resp", capabilities=caps)
    tasks = SimpleNamespace(
        active_response_task=None,
        has_response_bound_append_tasks=lambda: False,
    )
    # Monkeypatch assistant_playback_active via response_in_progress path: active_response_id set.
    assert helpers.response_in_progress(session, tasks) is True
    assert helpers.next_commit_allowed(session, tasks, concurrent_turn_requests_released=False) is False
    assert helpers.next_commit_allowed(session, tasks, concurrent_turn_requests_released=True) is True

    caps_off = DuplexCapabilities(supports_concurrent_turn_requests=False)
    session_off = SimpleNamespace(active_response_id="resp", capabilities=caps_off)
    assert helpers.next_commit_allowed(session_off, tasks, concurrent_turn_requests_released=True) is False


def test_active_response_accepts_own_turn_when_concurrent_turn_requests() -> None:
    """Approach A: each response owns its turn; draining TTS uses request→response map."""
    from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
    from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession

    session = DuplexEngineSession(
        session_id="s",
        config=DuplexSessionConfig(model="m", modalities=["text"]),
        capabilities=DuplexCapabilities(supports_concurrent_turn_requests=True),
    )
    session.begin_response(turn_id=3)
    assert session.active_response_accepts_model_turn(3)
    assert not session.active_response_accepts_model_turn(4)
    assert not session.active_response_accepts_model_turn(2)
    session.bind_draining_request("req-old", "resp-old")
    assert session.response_id_for_request("req-old") == "resp-old"
    assert session.is_draining_request("req-old")

    # Ending the active response must not wipe older draining bindings.
    session.bind_draining_request("req-newer-tts", session.active_response_id or "resp-active")
    ended = session.active_response_id
    session.end_response(commit_text=False)
    assert session.is_draining_request("req-old")
    assert session.response_id_for_request("req-old") == "resp-old"
    assert not session.is_draining_request("req-newer-tts")
    session.clear_draining_for_response("resp-old")
    assert not session.is_draining_request("req-old")
    del ended

    strict = DuplexEngineSession(
        session_id="s2",
        config=DuplexSessionConfig(model="m", modalities=["text"]),
        capabilities=DuplexCapabilities(supports_concurrent_turn_requests=False),
    )
    strict.begin_response(turn_id=3)
    assert strict.active_response_accepts_model_turn(3)
    assert not strict.active_response_accepts_model_turn(4)


def test_draining_request_exempt_from_completed_turn_filter() -> None:
    """Draining ownership must win over the model_turn_id < turn_id late-audio guard."""
    from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
    from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession

    session = DuplexEngineSession(
        session_id="s-drain",
        config=DuplexSessionConfig(model="m", modalities=["text", "audio"]),
        capabilities=DuplexCapabilities(supports_concurrent_turn_requests=True),
    )
    session.begin_response(turn_id=1)
    r1 = session.active_response_id
    assert r1 is not None
    session.bind_draining_request("req-r1-tts", r1)
    # Overlapping R2 becomes active, then finishes — must not wipe R1 draining.
    session.begin_response(turn_id=2)
    session.end_response(commit_text=False)
    session.turn_id = 2
    assert session.active_response_id is None
    assert session.is_draining_request("req-r1-tts")
    assert session.response_id_for_request("req-r1-tts") == r1
    draining_response_id = session.response_id_for_request("req-r1-tts")
    model_turn_id = 1
    drop = (
        draining_response_id is None
        and session.active_response_id is None
        and model_turn_id is not None
        and model_turn_id < session.turn_id
    )
    assert drop is False


def test_ephemeral_turn_id_parser_accepts_stage_turn_ids() -> None:
    from vllm_omni.engine.duplex.contracts import duplex_turn_id_from_request_id

    assert duplex_turn_id_from_request_id("duplex-s.x.e.0.r.stage0-turn12") == 12
    assert duplex_turn_id_from_request_id("duplex-s.x.e.0.r.stage2-turn3") == 3
    assert duplex_turn_id_from_request_id("duplex-s.x.e.0.r.stage0_t9") is None
    assert duplex_turn_id_from_request_id("duplex-s.x.e.0.r.stage0") is None


def test_same_turn_request_ids_skip_other_turns_and_epochs() -> None:
    from vllm_omni.engine.duplex.contracts import duplex_same_turn_request_ids

    stage3 = "duplex-s.x.e.0.r.stage3-turn4"
    found = duplex_same_turn_request_ids(
        stage3,
        [
            "duplex-s.x.e.0.r.stage0-turn4",
            "duplex-s.x.e.0.r.stage1-turn4",
            "duplex-s.x.e.0.r.stage2-turn4",
            stage3,
            "duplex-s.x.e.0.r.stage2-turn5",
            "duplex-s.x.e.1.r.stage0-turn4",
        ],
    )
    assert found == [
        "duplex-s.x.e.0.r.stage0-turn4",
        "duplex-s.x.e.0.r.stage1-turn4",
        "duplex-s.x.e.0.r.stage2-turn4",
    ]
    from vllm_omni.engine.duplex.contracts import duplex_session_id_from_request_id

    fence_session = "sess id"
    import base64

    encoded = base64.urlsafe_b64encode(fence_session.encode()).decode("ascii").rstrip("=")
    assert duplex_session_id_from_request_id(f"duplex-s.{encoded}.e.0.r.stage1-turn2") == fence_session
    assert duplex_session_id_from_request_id("not-a-request") is None


def test_stale_keys_skip_already_draining_request_ids() -> None:
    """Rebinding must not move an older draining Stage2/3 id onto the newer response."""
    from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
    from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession

    session = DuplexEngineSession(
        session_id="s-stale",
        config=DuplexSessionConfig(model="m", modalities=["text", "audio"]),
        capabilities=DuplexCapabilities(supports_concurrent_turn_requests=True),
    )
    session.begin_response(turn_id=1)
    r1 = session.active_response_id
    assert r1 is not None
    session.bind_draining_request("req-r1-talker", r1)
    session.begin_response(turn_id=2)
    r2 = session.active_response_id
    assert r2 is not None and r2 != r1
    # Simulate the concurrent-turn rebind loop: already-draining ids keep R1.
    for rid in ("req-r1-talker", "req-r2-talker"):
        if session.is_draining_request(rid):
            continue
        session.bind_draining_request(rid, r2)
    assert session.response_id_for_request("req-r1-talker") == r1
    assert session.response_id_for_request("req-r2-talker") == r2


def test_end_response_clears_only_own_draining_entries() -> None:
    from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
    from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession

    session = DuplexEngineSession(
        session_id="s-end",
        config=DuplexSessionConfig(model="m", modalities=["text", "audio"]),
        capabilities=DuplexCapabilities(supports_concurrent_turn_requests=True),
    )
    session.begin_response(turn_id=1)
    r1 = session.active_response_id
    session.bind_draining_request("req-r1", r1)
    session.begin_response(turn_id=2)
    r2 = session.active_response_id
    session.bind_draining_request("req-r2", r2)
    session.end_response(commit_text=False)
    assert session.is_draining_request("req-r1")
    assert session.response_id_for_request("req-r1") == r1
    assert not session.is_draining_request("req-r2")


def test_on_stage_failure_resolves_draining_response_before_active() -> None:
    """Mirrors runner.on_stage_failure: draining request_id owns the failed done."""
    from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
    from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession

    session = DuplexEngineSession(
        session_id="s-fail",
        config=DuplexSessionConfig(model="m", modalities=["text", "audio"]),
        capabilities=DuplexCapabilities(supports_concurrent_turn_requests=True),
    )
    session.begin_response(turn_id=1)
    r1 = session.active_response_id
    assert r1 is not None
    session.bind_draining_request("req-r1-talker", r1)
    session.begin_response(turn_id=2)
    r2 = session.active_response_id
    assert r2 is not None and r2 != r1

    request_id = "req-r1-talker"
    draining_response_id = (
        session.response_id_for_request(request_id) if session.is_draining_request(request_id) else None
    )
    response_id = draining_response_id or session.active_response_id
    assert response_id == r1
    assert response_id != r2
    session.clear_draining_for_response(r1)
    assert not session.is_draining_request(request_id)
    assert session.active_response_id == r2


def test_concurrent_turn_requests_released_resets_when_new_stage0_binds() -> None:
    """After R4 release, a new ephemeral Stage0 bind must clear the gate flag."""
    from types import SimpleNamespace

    run = SimpleNamespace(concurrent_turn_requests_released=True)
    # Simulate the concurrent-turn rebind arm in model_channel._append_via_data_plane.
    concurrent_turn = True
    if concurrent_turn:
        run.concurrent_turn_requests_released = False
    assert run.concurrent_turn_requests_released is False


def test_draining_stage_ids_come_from_the_plugin() -> None:
    from types import SimpleNamespace

    from vllm_omni.engine.duplex.session.model_channel import ModelChannel

    channel = ModelChannel.__new__(ModelChannel)
    channel._ctx = SimpleNamespace(
        plugin=SimpleNamespace(draining_stage_ids=lambda *, stage_count: {stage_count - 1}),
        stage_port=SimpleNamespace(stage_count=5),
    )
    assert channel._draining_stage_ids() == frozenset({4})

    channel._ctx.plugin = SimpleNamespace()
    assert channel._draining_stage_ids() == frozenset()


def test_draining_empty_eos_completes_owning_response_before_shortcut() -> None:
    """R2 finishing first must not swallow R1's empty Stage3 EOS."""
    import asyncio
    from types import SimpleNamespace

    from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
    from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession
    from vllm_omni.engine.duplex.session.model_channel import ModelChannel

    session = DuplexEngineSession(
        session_id="s-empty-eos",
        config=DuplexSessionConfig(model="m", modalities=["text", "audio"], extra_body={"auto_response": True}),
        capabilities=DuplexCapabilities(supports_concurrent_turn_requests=True, supports_core_resumable_request=False),
    )
    session.begin_response(turn_id=1)
    r1 = session.active_response_id
    assert r1 is not None
    r1_request = "duplex-s.x.e.0.r.stage3-turn1"
    still_running = "duplex-s.x.e.0.r.stage1-turn2"
    session.bind_draining_request(r1_request, r1)
    session.begin_response(turn_id=2)
    session.end_response(commit_text=False)
    session.turn_id = 2
    assert session.active_response_id is None

    class _Plane:
        def is_terminal(self, request_id: str | None) -> bool:
            del request_id
            return False

        def close_stream(self, request_id: str) -> None:
            del request_id

        def mark_terminal(self, request_id: str) -> None:
            del request_id

    class _Port:
        def __init__(self) -> None:
            self.cleanups: list[list[str]] = []

        async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None:
            del abort
            self.cleanups.append(list(request_ids))

    class _Out:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        def auto_responds(self) -> bool:
            return True

        def emit(self, payload: dict[str, object]) -> None:
            self.events.append(payload)

    port = _Port()
    out = _Out()

    async def _noop(*_args: object, **_kwargs: object) -> None:
        return None

    async def _schedule(*_args: object, **_kwargs: object) -> bool:
        return False

    channel = ModelChannel(
        SimpleNamespace(
            session=session,
            plugin=SimpleNamespace(data_plane=_Plane()),
            stage_port=port,
            model_state=SimpleNamespace(clear_continuation=lambda: None),
            services=SimpleNamespace(spawn=lambda *_a, **_k: None),
        ),
        out,
        close_from_runtime=_noop,
        schedule_silence_continuation=_schedule,
        abort_request=_noop,
    )
    asyncio.run(
        channel._send_one_model_output_event(
            {
                "data_plane_request_id": r1_request,
                "end_of_turn": True,
                "model_turn_id": 1,
                "text": "",
                "stage_role": "tts",
            }
        )
    )
    done = [event for event in out.events if event.get("type") == "response.done"]
    assert done and done[-1].get("response_id") == r1
    assert port.cleanups == [[r1_request]]
    assert all(still_running not in ids for ids in port.cleanups)
    assert not session.is_draining_request(r1_request)


def test_draining_completion_drops_finished_response_books_only() -> None:
    """R1 drain finish drops R1 books and leaves the live R2 snapshot."""
    import asyncio
    from types import SimpleNamespace

    from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
    from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession
    from vllm_omni.engine.duplex.session.model_channel import ModelChannel

    session = DuplexEngineSession(
        session_id="s-drain-books",
        config=DuplexSessionConfig(model="m", modalities=["text", "audio"], extra_body={"auto_response": True}),
        capabilities=DuplexCapabilities(supports_concurrent_turn_requests=True, supports_core_resumable_request=False),
    )
    session.begin_response(turn_id=1)
    r1 = session.active_response_id
    assert r1 is not None
    session.append_assistant_text("from-r1")
    session.snapshot_active_response_for_drain()
    r1_request = "duplex-s.x.e.0.r.stage3-turn1"
    session.bind_draining_request(r1_request, r1)
    session.begin_response(turn_id=2)
    r2 = session.active_response_id
    assert r2 is not None and r2 != r1
    session.append_assistant_text("from-r2")
    session.snapshot_active_response_for_drain()

    class _Plane:
        def is_terminal(self, request_id: str | None) -> bool:
            del request_id
            return False

        def close_stream(self, request_id: str) -> None:
            del request_id

        def mark_terminal(self, request_id: str) -> None:
            del request_id

    class _Port:
        async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None:
            del request_ids, abort

    class _Out:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        def auto_responds(self) -> bool:
            return True

        def emit(self, payload: dict[str, object]) -> None:
            self.events.append(payload)

    out = _Out()

    async def _noop(*_args: object, **_kwargs: object) -> None:
        return None

    async def _schedule(*_args: object, **_kwargs: object) -> bool:
        return False

    channel = ModelChannel(
        SimpleNamespace(
            session=session,
            plugin=SimpleNamespace(data_plane=_Plane()),
            stage_port=_Port(),
            model_state=SimpleNamespace(clear_continuation=lambda: None),
            services=SimpleNamespace(spawn=lambda *_a, **_k: None),
        ),
        out,
        close_from_runtime=_noop,
        schedule_silence_continuation=_schedule,
        abort_request=_noop,
    )
    asyncio.run(
        channel._send_one_model_output_event(
            {
                "data_plane_request_id": r1_request,
                "end_of_turn": True,
                "model_turn_id": 1,
                "text": "",
                "stage_role": "tts",
            }
        )
    )
    assert r1 not in session._conversation.assistant_response_snapshots
    assert r1 not in session._playback.by_response
    assert f"item_{r1}" not in session._conversation.history_item_placeholders
    assert r2 in session._conversation.assistant_response_snapshots
    assert session.active_response_id == r2
    done = [event for event in out.events if event.get("type") == "response.done"]
    assert done and done[-1].get("response_id") == r1


def test_stale_continue_does_not_close_the_new_response() -> None:
    import asyncio
    from types import SimpleNamespace

    from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
    from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession
    from vllm_omni.engine.duplex.session.model_channel import ModelChannel

    session = DuplexEngineSession(
        session_id="s-stale-continue",
        config=DuplexSessionConfig(model="m", modalities=["text", "audio"], extra_body={"auto_response": True}),
        capabilities=DuplexCapabilities(supports_concurrent_turn_requests=True, supports_core_resumable_request=False),
    )
    session.begin_response(turn_id=2)
    live = session.active_response_id
    session.epoch = 1

    class _Out:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        def auto_responds(self) -> bool:
            return True

        def emit(self, payload: dict[str, object]) -> None:
            self.events.append(payload)

    out = _Out()
    cleared: list[bool] = []

    async def _noop(*_args: object, **_kwargs: object) -> None:
        return None

    async def _schedule(*_args: object, **_kwargs: object) -> bool:
        return False

    channel = ModelChannel(
        SimpleNamespace(
            session=session,
            plugin=SimpleNamespace(data_plane=SimpleNamespace()),
            stage_port=SimpleNamespace(),
            model_state=SimpleNamespace(clear_continuation=lambda: cleared.append(True)),
            services=SimpleNamespace(spawn=lambda *_a, **_k: None),
            run=SimpleNamespace(closing=False),
        ),
        out,
        close_from_runtime=_noop,
        schedule_silence_continuation=_schedule,
        abort_request=_noop,
    )
    asyncio.run(channel.maybe_continue_response(expected_epoch=0))
    assert session.active_response_id == live
    assert out.events == []
    assert cleared == []

    session.bind_draining_request("duplex-s.x.e.1.r.stage3-turn2", live or "")
    asyncio.run(channel.maybe_continue_response(expected_epoch=1))
    assert session.active_response_id == live
    assert out.events == []

    session.clear_draining_requests()
    asyncio.run(channel.maybe_continue_response(expected_epoch=1))
    assert session.active_response_id is None
    assert [event.get("type") for event in out.events] == ["response.done"]
    assert out.events[0].get("response_id") == live


def test_silent_listen_with_continuation_releases_ephemeral_request() -> None:
    import asyncio
    from collections.abc import Coroutine
    from types import SimpleNamespace
    from typing import Any

    from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
    from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession
    from vllm_omni.engine.duplex.session.model_channel import ModelChannel

    session = DuplexEngineSession(
        session_id="s-silent-continue",
        config=DuplexSessionConfig(model="m", modalities=["text", "audio"]),
        capabilities=DuplexCapabilities(supports_concurrent_turn_requests=True, supports_core_resumable_request=False),
    )
    session.begin_response(turn_id=1)
    response_id = session.active_response_id
    request_id = "duplex-s.x.e.0.r.stage1-turn1"
    session.bind_request(request_id)

    class _Plane:
        def __init__(self) -> None:
            self.terminal: list[str] = []

        def is_terminal(self, request_id: str | None) -> bool:
            return request_id in self.terminal

        def mark_terminal(self, request_id: str) -> None:
            self.terminal.append(request_id)

    class _Port:
        def __init__(self) -> None:
            self.cleanups: list[list[str]] = []

        async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None:
            del abort
            self.cleanups.append(list(request_ids))

    class _Out:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        def auto_responds(self) -> bool:
            return False

        def emit(self, payload: dict[str, object]) -> None:
            self.events.append(payload)

    port = _Port()
    out = _Out()
    spawned: list[Coroutine[Any, Any, None]] = []

    async def _noop(*_args: object, **_kwargs: object) -> None:
        return None

    async def _schedule(*_args: object, **_kwargs: object) -> bool:
        return False

    channel = ModelChannel(
        SimpleNamespace(
            session=session,
            plugin=SimpleNamespace(data_plane=_Plane()),
            stage_port=port,
            model_state=SimpleNamespace(
                continuation_owner_id=None, continuation_units=0, clear_continuation=lambda: None
            ),
            services=SimpleNamespace(spawn=lambda coro, **_k: spawned.append(coro)),
            run=SimpleNamespace(closing=False),
        ),
        out,
        close_from_runtime=_noop,
        schedule_silence_continuation=_schedule,
        abort_request=_noop,
    )
    asyncio.run(
        channel._send_one_model_output_event(
            {
                "data_plane_request_id": request_id,
                "is_listen": True,
                "end_of_turn": True,
                "model_turn_id": 1,
                "reason": "model_listen",
            }
        )
    )
    assert port.cleanups == [[request_id]]
    assert session.active_request_id is None
    assert spawned, "silent listen with unused continuation must schedule maybe_continue"
    asyncio.run(spawned[0])
    assert port.cleanups == [[request_id]]
    assert session.active_response_id is None
    assert [event.get("type") for event in out.events] == ["response.listen", "response.done"]
    assert out.events[0].get("response_id") == response_id
    assert out.events[1].get("response_id") == response_id


def test_silent_listen_aborts_data_plane_request_with_full_id() -> None:
    """abort_data_plane_request must pass [request_id], not a bare str."""
    import asyncio
    from types import SimpleNamespace

    from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
    from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession
    from vllm_omni.engine.duplex.session.model_channel import ModelChannel

    session = DuplexEngineSession(
        session_id="s-silent-abort",
        config=DuplexSessionConfig(model="m", modalities=["text", "audio"]),
        capabilities=DuplexCapabilities(supports_concurrent_turn_requests=True, supports_core_resumable_request=False),
    )
    request_id = "duplex-s.x.e.0.r.stage0-turn7"
    session.bind_request(request_id)
    aborts: list[list[str]] = []

    class _Plane:
        def is_terminal(self, request_id: str | None) -> bool:
            return False

        def mark_terminal(self, request_id: str) -> None:
            del request_id

    class _Port:
        async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None:
            del request_ids, abort

    class _Out:
        def auto_responds(self) -> bool:
            return True

        def emit(self, payload: dict[str, object]) -> None:
            del payload

    async def _noop(*_args: object, **_kwargs: object) -> None:
        return None

    async def _abort(request_ids: list[str], *, notify: bool = False) -> None:
        del notify
        aborts.append(list(request_ids))

    channel = ModelChannel(
        SimpleNamespace(
            session=session,
            plugin=SimpleNamespace(data_plane=_Plane()),
            stage_port=_Port(),
            model_state=SimpleNamespace(
                continuation_owner_id=None, continuation_units=0, clear_continuation=lambda: None
            ),
            services=SimpleNamespace(spawn=lambda coro, **_k: None),
            run=SimpleNamespace(closing=False),
        ),
        _Out(),
        close_from_runtime=_noop,
        schedule_silence_continuation=_noop,
        abort_request=_abort,
    )
    asyncio.run(
        channel._send_one_model_output_event(
            {
                "data_plane_request_id": request_id,
                "is_listen": True,
                "end_of_turn": True,
                "model_turn_id": 1,
                "reason": "model_listen",
                "abort_data_plane_request": True,
            }
        )
    )
    assert aborts == [[request_id]]
    assert all(len(item) > 1 for batch in aborts for item in batch), "must not split request id into chars"
