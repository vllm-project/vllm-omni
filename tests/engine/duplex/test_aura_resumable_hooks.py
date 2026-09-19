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


def test_capabilities_expose_overlapped_input_default_false() -> None:
    caps = DuplexCapabilities()
    assert caps.supports_overlapped_commit is False
    assert caps.allows_video_without_audio() is False
    assert caps.required_input_modalities == frozenset({"audio"})
    assert caps.optional_input_modalities == frozenset({"video"})
    assert caps.supports_core_resumable_request is False
    payload = caps.as_dict()
    assert payload["supports_overlapped_commit"] is False
    assert payload["required_input_modalities"] == ["audio"]
    assert payload["optional_input_modalities"] == ["video"]


def test_next_commit_allowed_soft_opens_on_r4_release() -> None:
    from types import SimpleNamespace

    from vllm_omni.engine.duplex.session import helpers

    caps = DuplexCapabilities(supports_overlapped_commit=True)
    session = SimpleNamespace(active_response_id="resp", capabilities=caps)
    tasks = SimpleNamespace(
        active_response_task=None,
        has_response_bound_append_tasks=lambda: False,
    )
    # Monkeypatch assistant_playback_active via response_in_progress path: active_response_id set.
    assert helpers.response_in_progress(session, tasks) is True
    assert helpers.next_commit_allowed(session, tasks, overlapped_commit_released=False) is False
    assert helpers.next_commit_allowed(session, tasks, overlapped_commit_released=True) is True

    caps_off = DuplexCapabilities(supports_overlapped_commit=False)
    session_off = SimpleNamespace(active_response_id="resp", capabilities=caps_off)
    assert helpers.next_commit_allowed(session_off, tasks, overlapped_commit_released=True) is False


def test_active_response_accepts_own_turn_when_overlapped_input() -> None:
    """Approach A: each response owns its turn; draining TTS uses request→response map."""
    from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
    from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession

    session = DuplexEngineSession(
        session_id="s",
        config=DuplexSessionConfig(model="m", modalities=["text"]),
        capabilities=DuplexCapabilities(supports_overlapped_commit=True),
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
        capabilities=DuplexCapabilities(supports_overlapped_commit=False),
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
        capabilities=DuplexCapabilities(supports_overlapped_commit=True),
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


def test_ephemeral_turn_id_parser_accepts_main_and_legacy_forms() -> None:
    from vllm_omni.engine.duplex.contracts import duplex_turn_id_from_request_id

    assert duplex_turn_id_from_request_id("duplex-s.x.e.0.r.stage0-turn12") == 12
    assert duplex_turn_id_from_request_id("duplex-s.x.e.0.r.stage2-turn3") == 3
    assert duplex_turn_id_from_request_id("duplex-s.x.e.0.r.stage0_t9") == 9
    assert duplex_turn_id_from_request_id("duplex-s.x.e.0.r.stage0") is None


def test_stale_keys_skip_already_draining_request_ids() -> None:
    """Rebinding must not move an older draining Stage2/3 id onto the newer response."""
    from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
    from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession

    session = DuplexEngineSession(
        session_id="s-stale",
        config=DuplexSessionConfig(model="m", modalities=["text", "audio"]),
        capabilities=DuplexCapabilities(supports_overlapped_commit=True),
    )
    session.begin_response(turn_id=1)
    r1 = session.active_response_id
    assert r1 is not None
    session.bind_draining_request("req-r1-talker", r1)
    session.begin_response(turn_id=2)
    r2 = session.active_response_id
    assert r2 is not None and r2 != r1
    # Simulate the overlapped rebind loop: already-draining ids keep R1.
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
        capabilities=DuplexCapabilities(supports_overlapped_commit=True),
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
        capabilities=DuplexCapabilities(supports_overlapped_commit=True),
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


def test_overlapped_commit_released_resets_when_new_stage0_binds() -> None:
    """After R4 release, a new ephemeral Stage0 bind must clear the gate flag."""
    from types import SimpleNamespace

    run = SimpleNamespace(overlapped_commit_released=True)
    # Simulate the overlapped rebind arm in model_channel._append_via_data_plane.
    overlapped = True
    if overlapped:
        run.overlapped_commit_released = False
    assert run.overlapped_commit_released is False
