# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import base64
import importlib.util
import sys
import wave
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

DEMO_PATH = Path(__file__).resolve().parents[2] / "examples/online_serving/minicpmo/realtime_duplex_demo.py"


def _load_demo_module():
    spec = importlib.util.spec_from_file_location(
        "minicpmo_realtime_duplex_simple_demo_test",
        DEMO_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_post_commit_decision_ignores_streaming_listens_before_commit_ack():
    demo = _load_demo_module()
    events = [
        {"type": "session.created"},
        {"type": "response.listen"},
        {"type": "response.listen"},
        {"type": "input_audio_buffer.committed"},
    ]

    commit_index = demo._input_committed_index(events, after_index=0)

    assert commit_index == 3
    assert demo._post_commit_model_decision(events, commit_index) is None


def test_post_commit_decision_accepts_listen_after_commit_ack():
    demo = _load_demo_module()
    events = [
        {"type": "response.listen"},
        {"type": "input_audio_buffer.committed"},
        {"type": "response.listen"},
    ]

    commit_index = demo._input_committed_index(events, after_index=0)

    assert demo._post_commit_model_decision(events, commit_index) == "listen"


def test_post_commit_decision_accepts_drained_speak_after_commit_ack():
    demo = _load_demo_module()
    events = [
        {"type": "input_audio_buffer.committed"},
        {"type": "response.created", "response": {"id": "resp-a"}},
        {"type": "response.output_audio.delta", "response_id": "resp-a", "delta": "AAAA"},
        {"type": "response.done", "response_id": "resp-a"},
    ]

    commit_index = demo._input_committed_index(events, after_index=0)

    assert demo._post_commit_model_decision(events, commit_index) == "speak"


def test_post_commit_decision_ignores_cancelled_response():
    demo = _load_demo_module()
    events = [
        {"type": "input_audio_buffer.committed"},
        {"type": "response.done", "response": {"status": "cancelled"}},
    ]

    commit_index = demo._input_committed_index(events, after_index=0)

    assert demo._post_commit_model_decision(events, commit_index) is None


def test_commit_ack_search_starts_after_stream_cursor():
    demo = _load_demo_module()
    events = [
        {"type": "input_audio_buffer.committed"},
        {"type": "response.listen"},
        {"type": "input_audio_buffer.committed"},
    ]

    assert demo._input_committed_index(events, after_index=1) == 2


def test_exact_model_unit_boundary_does_not_require_an_extra_decision():
    demo = _load_demo_module()
    unit_bytes = demo.PCM16_SAMPLE_RATE * demo.PCM16_BYTES_PER_SAMPLE

    assert not demo._has_residual_model_unit(b"\0" * (unit_bytes * 6), chunk_period_ms=1000)


def test_partial_model_unit_requires_a_post_commit_decision():
    demo = _load_demo_module()
    unit_bytes = demo.PCM16_SAMPLE_RATE * demo.PCM16_BYTES_PER_SAMPLE

    assert demo._has_residual_model_unit(b"\0" * (unit_bytes * 6 + 512), chunk_period_ms=1000)


def test_latest_streaming_decision_is_used_at_exact_boundary():
    demo = _load_demo_module()
    events = [
        {"type": "response.listen"},
        {"type": "response.listen"},
        {"type": "input_audio_buffer.committed"},
    ]

    assert demo._latest_model_decision(events, after_index=0) == "listen"


def test_ref_audio_data_url_encodes_explicit_wav(tmp_path):
    demo = _load_demo_module()
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFFfake")

    assert demo._ref_audio_data_url(str(ref)) == "data:audio/wav;base64,UklGRmZha2U="
    assert demo._ref_audio_data_url(None) is None


def test_realtime_duplex_demo_allows_the_server_default_ref_audio(monkeypatch):
    demo = _load_demo_module()
    monkeypatch.setattr(
        demo.sys,
        "argv",
        [
            "realtime_duplex_demo.py",
            "--input-wav",
            "input.wav",
        ],
    )

    args = demo.parse_args()

    assert args.ref_audio is None


def test_realtime_duplex_demo_accepts_explicit_ref_audio(monkeypatch):
    demo = _load_demo_module()
    monkeypatch.setattr(
        demo.sys,
        "argv",
        [
            "realtime_duplex_demo.py",
            "--input-wav",
            "input.wav",
            "--ref-audio",
            "ref.wav",
        ],
    )

    args = demo.parse_args()

    assert args.ref_audio == "ref.wav"


def test_open_streaming_response_requires_post_commit_drain():
    demo = _load_demo_module()

    assert demo._response_in_progress([{"type": "response.created"}])
    assert not demo._response_in_progress(
        [
            {"type": "response.created"},
            {"type": "response.done"},
        ]
    )


def test_final_input_commit_checkpoints_playback_first():
    demo = _load_demo_module()
    collector = demo.EventCollector()
    collector.add(
        {
            "type": "response.created",
            "response": {"id": "resp-zero-audio"},
        }
    )
    calls: list[dict[str, object]] = []

    class _Client:
        async def ack_playback(self, played_ms, *, response_id=None, item_id=None, committed_ms=None):
            calls.append({"type": "playback.ack", "response_id": response_id, "played_ms": int(played_ms)})

        async def commit(self, *, create_response=None):
            calls.append({"type": "input_audio_buffer.commit", "create_response": create_response})

    commit_sent_at_s = asyncio.run(demo._commit_input_after_playback_checkpoint(_Client(), collector))

    # The zero-audio response is still acked (0 ms) before the commit: the
    # ack checkpoints its history position on the server.
    assert [event["type"] for event in calls] == ["playback.ack", "input_audio_buffer.commit"]
    assert calls[0]["response_id"] == "resp-zero-audio"
    assert calls[0]["played_ms"] == 0
    # Legacy callers keep the previous wire shape: no response_create flag.
    assert calls[1]["create_response"] is None
    assert isinstance(commit_sent_at_s, float)


def test_final_commit_forwards_the_mandatory_response_flag():
    demo = _load_demo_module()
    collector = demo.EventCollector()
    calls: list[object] = []

    class _Client:
        async def ack_playback(self, played_ms, *, response_id=None, item_id=None, committed_ms=None):
            pass

        async def commit(self, *, create_response=None):
            calls.append(create_response)

    asyncio.run(demo._commit_input_after_playback_checkpoint(_Client(), collector, create_response=True))

    assert calls == [True]


def test_final_create_response_flag_is_opt_in_for_legacy_namespaces(monkeypatch):
    demo = _load_demo_module()
    monkeypatch.setattr(
        demo.sys,
        "argv",
        ["realtime_duplex_demo.py", "--input-wav", "input.wav"],
    )

    assert demo.parse_args().final_create_response is False
    # Tests build bare Namespaces without the new field; run_demo must still
    # treat that as "not requested" instead of raising AttributeError.
    assert getattr(demo.argparse.Namespace(input_wav="input.wav"), "final_create_response", False) is False


def test_final_create_response_flag_parses_from_the_cli(monkeypatch):
    demo = _load_demo_module()
    monkeypatch.setattr(
        demo.sys,
        "argv",
        ["realtime_duplex_demo.py", "--input-wav", "input.wav", "--final-create-response"],
    )

    assert demo.parse_args().final_create_response is True


def test_a_forced_response_must_drain_before_the_session_closes():
    demo = _load_demo_module()
    events = [
        {"type": "input_audio_buffer.committed"},
        {"type": "response.created", "response": {"id": "resp-forced"}},
        {"type": "response.done", "response": {"id": "resp-forced"}},
    ]

    assert demo._post_commit_response_created(events, 0)
    assert demo._post_commit_forced_response_settled(events, 0)
    # A listener decision or a cancelled generation is not a drained answer.
    assert not demo._post_commit_forced_response_settled(
        [{"type": "input_audio_buffer.committed"}, {"type": "response.listen"}],
        0,
    )
    assert not demo._post_commit_forced_response_settled(
        [
            {"type": "input_audio_buffer.committed"},
            {"type": "response.done", "response": {"status": "cancelled"}},
        ],
        0,
    )


def test_forced_response_drains_an_opened_response_without_reasking():
    demo = _load_demo_module()
    collector = demo.EventCollector()
    collector.add({"type": "input_audio_buffer.committed"})
    collector.add({"type": "response.created", "response": {"id": "resp-forced"}})
    collector.add({"type": "response.done", "response": {"id": "resp-forced"}})
    sent: list[dict[str, object]] = []

    class _Client:
        async def send(self, event):
            sent.append(event)

    fallback = asyncio.run(
        demo._drain_forced_post_commit_response(
            _Client(), collector, committed_index=0, timeout_s=0.05, create_timeout_s=0.05, settle_quiet_s=0.0
        )
    )

    assert fallback is False
    assert sent == []


def test_forced_response_falls_back_to_an_explicit_create_when_idle():
    demo = _load_demo_module()
    collector = demo.EventCollector()
    collector.add({"type": "input_audio_buffer.committed"})
    sent: list[dict[str, object]] = []

    class _Client:
        async def send(self, event):
            sent.append(event)

    with pytest.raises(TimeoutError):
        asyncio.run(
            demo._drain_forced_post_commit_response(
                _Client(), collector, committed_index=0, timeout_s=0.05, create_timeout_s=0.05, settle_quiet_s=0.0
            )
        )

    assert sent == [{"type": "response.create"}]


def test_forced_response_retries_the_fallback_send_through_connection_errors():
    demo = _load_demo_module()
    collector = demo.EventCollector()
    collector.add({"type": "input_audio_buffer.committed"})
    sent: list[dict[str, object]] = []

    class _Client:
        async def send(self, event):
            sent.append(event)
            if len(sent) < 3:
                # A send inside a transport resume window fails even though
                # the session itself is still alive.
                raise demo.DuplexConnectionError("send failed: transport is not connected")
            collector.add({"type": "response.created", "response": {"id": "resp-forced"}})
            collector.add({"type": "response.done", "response": {"id": "resp-forced"}})

    fallback = asyncio.run(
        demo._drain_forced_post_commit_response(
            _Client(),
            collector,
            committed_index=0,
            timeout_s=0.5,
            create_timeout_s=0.05,
            settle_quiet_s=0.0,
            fallback_retry_window_s=5.0,
            fallback_retry_interval_s=0.01,
        )
    )

    assert fallback is True
    assert sent == [{"type": "response.create"}] * 3


def test_forced_response_does_not_retry_the_fallback_after_the_session_closes():
    demo = _load_demo_module()
    collector = demo.EventCollector()
    collector.add({"type": "input_audio_buffer.committed"})
    sent: list[dict[str, object]] = []

    class _Client:
        async def send(self, event):
            sent.append(event)
            raise demo.DuplexSessionClosedError("closed")

    with pytest.raises(TimeoutError):
        asyncio.run(
            demo._drain_forced_post_commit_response(
                _Client(),
                collector,
                committed_index=0,
                timeout_s=0.05,
                create_timeout_s=0.05,
                settle_quiet_s=0.0,
                fallback_retry_window_s=5.0,
                fallback_retry_interval_s=0.01,
            )
        )

    assert sent == [{"type": "response.create"}]


def test_forced_response_falls_back_exactly_once_when_the_retry_window_is_zero():
    demo = _load_demo_module()
    collector = demo.EventCollector()
    collector.add({"type": "input_audio_buffer.committed"})
    sent: list[dict[str, object]] = []

    class _Client:
        async def send(self, event):
            sent.append(event)
            raise demo.DuplexConnectionError("send failed: transport is not connected")

    with pytest.raises(TimeoutError):
        asyncio.run(
            demo._drain_forced_post_commit_response(
                _Client(),
                collector,
                committed_index=0,
                timeout_s=0.05,
                create_timeout_s=0.05,
                settle_quiet_s=0.0,
                fallback_retry_window_s=0.0,
                fallback_retry_interval_s=0.01,
            )
        )

    assert sent == [{"type": "response.create"}]


@pytest.mark.parametrize(
    ("code", "wire"),
    [
        ("response_create_without_input", True),
        ("text_only_turn_unsupported", True),
        ("response_create_without_input", False),
    ],
)
def test_forced_response_settles_when_the_server_definitively_refuses(code, wire):
    demo = _load_demo_module()
    collector = demo.EventCollector()
    collector.add({"type": "input_audio_buffer.committed"})
    sent: list[dict[str, object]] = []

    if wire:
        error_event = {
            "type": "error",
            "error": {"type": "invalid_request_error", "code": code, "message": "refused"},
        }
    else:
        error_event = {"type": "error", "code": code, "message": "refused"}

    class _Client:
        async def send(self, event):
            sent.append(event)
            # The commit retained nothing answerable, so the server answers
            # the create with a definitive refusal instead of a response.
            collector.add(error_event)

    fallback = asyncio.run(
        demo._drain_forced_post_commit_response(
            _Client(),
            collector,
            committed_index=0,
            timeout_s=0.5,
            create_timeout_s=0.05,
            settle_quiet_s=0.0,
        )
    )

    assert fallback is True
    assert sent == [{"type": "response.create"}]


def test_forced_response_keeps_waiting_after_other_server_errors():
    demo = _load_demo_module()
    collector = demo.EventCollector()
    collector.add({"type": "input_audio_buffer.committed"})
    sent: list[dict[str, object]] = []

    class _Client:
        async def send(self, event):
            sent.append(event)
            # Any other error (a promotion race, a transport complaint) is
            # not a definitive refusal: the drain keeps waiting for the
            # response instead of settling.
            collector.add(
                {
                    "type": "error",
                    "code": "response_already_active",
                    "message": "response is already active",
                }
            )

    with pytest.raises(TimeoutError):
        asyncio.run(
            demo._drain_forced_post_commit_response(
                _Client(),
                collector,
                committed_index=0,
                timeout_s=0.05,
                create_timeout_s=0.05,
                settle_quiet_s=0.0,
            )
        )

    assert sent == [{"type": "response.create"}]


def test_forced_response_reports_a_sent_fallback_even_when_the_drain_times_out():
    demo = _load_demo_module()
    collector = demo.EventCollector()
    collector.add({"type": "input_audio_buffer.committed"})
    sent: list[dict[str, object]] = []
    marked: list[bool] = []

    class _Client:
        async def send(self, event):
            sent.append(event)

    with pytest.raises(TimeoutError):
        asyncio.run(
            demo._drain_forced_post_commit_response(
                _Client(),
                collector,
                committed_index=0,
                timeout_s=0.05,
                create_timeout_s=0.05,
                settle_quiet_s=0.0,
                on_fallback_sent=lambda: marked.append(True),
            )
        )

    # A timed-out drain must not read as "the re-ask was never sent".
    assert sent == [{"type": "response.create"}]
    assert marked == [True]


def test_refusal_forgiveness_is_scoped_to_the_post_commit_window():
    demo = _load_demo_module()
    pre_commit_refusal = {"type": "error", "error": {"code": "response_create_without_input"}}
    post_commit_refusal = {"type": "error", "error": {"code": "text_only_turn_unsupported"}}
    other_error = {"type": "error", "error": {"code": "bad_event"}}
    events = [
        pre_commit_refusal,
        {"type": "input_audio_buffer.committed"},
        post_commit_refusal,
        other_error,
    ]

    # Only the refusal that answered the forced create is forgiven; the
    # earlier one came from some other ask and other codes stay errors.
    assert demo._post_commit_refusal_indexes(events, 1) == frozenset({2})
    assert demo._post_commit_refusal_indexes(events, None) == frozenset()


def test_forced_response_does_not_reask_while_a_response_is_active():
    demo = _load_demo_module()
    collector = demo.EventCollector()
    collector.add({"type": "response.created", "response": {"id": "resp-inflight"}})
    collector.add({"type": "input_audio_buffer.committed"})
    sent: list[dict[str, object]] = []

    class _Client:
        async def send(self, event):
            sent.append(event)

    # The deferred commit is promoted with its forced create once the active
    # response settles; re-asking now only yields response_already_active.
    with pytest.raises(TimeoutError):
        asyncio.run(
            demo._drain_forced_post_commit_response(
                _Client(), collector, committed_index=1, timeout_s=0.05, create_timeout_s=0.05, settle_quiet_s=0.0
            )
        )

    assert sent == []


def test_forced_response_does_not_reask_after_a_commit_error():
    demo = _load_demo_module()
    collector = demo.EventCollector()
    collector.add({"type": "input_audio_buffer.committed"})
    collector.add({"type": "error", "error": {"message": "input_audio_buffer_empty"}})
    sent: list[dict[str, object]] = []

    class _Client:
        async def send(self, event):
            sent.append(event)

    with pytest.raises(TimeoutError):
        asyncio.run(
            demo._drain_forced_post_commit_response(
                _Client(), collector, committed_index=0, timeout_s=0.05, create_timeout_s=0.05, settle_quiet_s=0.0
            )
        )

    assert sent == []


def test_post_commit_settle_waits_for_the_promoted_response_after_a_pre_commit_one_settles():
    demo = _load_demo_module()
    # Deferred-create path: the commit is held behind an active response.
    events = [
        {"type": "response.created", "response": {"id": "resp-inflight"}},
        {"type": "input_audio_buffer.committed"},
        {"type": "response.done", "response": {"id": "resp-inflight"}},
    ]

    # The pre-commit response settling after the commit must not pose as the
    # forced answer: no response has been created after the commit yet.
    assert not demo._post_commit_forced_response_settled(events, 1)

    events.append({"type": "response.created", "response": {"id": "resp-promoted"}})
    assert not demo._post_commit_forced_response_settled(events, 1)

    events.append({"type": "response.done", "response": {"id": "resp-promoted"}})
    assert demo._post_commit_forced_response_settled(events, 1)


def test_post_commit_settle_ignores_terminals_of_responses_created_before_the_commit():
    demo = _load_demo_module()
    events = [
        {"type": "response.created", "response": {"id": "resp-pre"}},
        {"type": "input_audio_buffer.committed"},
        # A spontaneous response pairs its own creation, but cannot speak for
        # the forced one: settled only once a post-commit creation exists and
        # every post-commit creation is done.
        {"type": "response.created", "response": {"id": "resp-spont"}},
        {"type": "response.done", "response": {"id": "resp-pre"}},
        {"type": "response.done", "response": {"id": "resp-spont"}},
    ]

    assert demo._post_commit_forced_response_settled(events, 1)


def test_drain_holds_through_the_window_between_a_spontaneous_settle_and_the_promoted_create():
    demo = _load_demo_module()
    collector = demo.EventCollector()
    collector.add({"type": "input_audio_buffer.committed"})
    collector.add({"type": "response.created", "response": {"id": "resp-spont"}})
    collector.add({"type": "response.done", "response": {"id": "resp-spont"}})
    sent: list[dict[str, object]] = []

    class _Client:
        async def send(self, event):
            sent.append(event)

    # Id-pairing alone settles here (the spontaneous response pairs with its
    # own done); only the quiescence guard keeps the session open while the
    # promoted forced create is still queued behind the settle.
    with pytest.raises(TimeoutError):
        asyncio.run(
            demo._drain_forced_post_commit_response(
                _Client(),
                collector,
                committed_index=0,
                timeout_s=0.05,
                create_timeout_s=0.05,
                settle_quiet_s=1.0,
            )
        )

    assert sent == []


def test_drain_returns_once_the_promoted_response_settles_and_quiesces():
    demo = _load_demo_module()
    collector = demo.EventCollector()
    collector.add({"type": "input_audio_buffer.committed"})
    collector.add({"type": "response.created", "response": {"id": "resp-spont"}})
    collector.add({"type": "response.done", "response": {"id": "resp-spont"}})
    quiet_ago_s = demo.time.monotonic() - 1.0
    collector.add({"type": "response.created", "response": {"id": "resp-promoted"}}, received_at_s=quiet_ago_s)
    collector.add({"type": "response.done", "response": {"id": "resp-promoted"}}, received_at_s=quiet_ago_s)
    sent: list[dict[str, object]] = []

    class _Client:
        async def send(self, event):
            sent.append(event)

    fallback = asyncio.run(
        demo._drain_forced_post_commit_response(
            _Client(),
            collector,
            committed_index=0,
            timeout_s=0.05,
            create_timeout_s=0.05,
            settle_quiet_s=1.0,
        )
    )

    assert fallback is False
    assert sent == []


def test_drain_does_not_reask_when_a_post_commit_response_already_settled():
    demo = _load_demo_module()
    collector = demo.EventCollector()
    collector.add({"type": "input_audio_buffer.committed"})
    collector.add({"type": "response.done", "response": {"id": "resp-unknown"}})
    sent: list[dict[str, object]] = []

    class _Client:
        async def send(self, event):
            sent.append(event)

    # The server already worked through a response after the commit, so the
    # deferred promotion may still be in flight; re-asking could collide with
    # it. The drain keeps waiting instead of sending response.create.
    with pytest.raises(TimeoutError):
        asyncio.run(
            demo._drain_forced_post_commit_response(
                _Client(), collector, committed_index=0, timeout_s=0.05, create_timeout_s=0.05, settle_quiet_s=0.0
            )
        )

    assert sent == []


def test_native_drain_reasks_after_a_post_commit_settle_when_nothing_was_deferred():
    demo = _load_demo_module()
    collector = demo.EventCollector()
    collector.add({"type": "response.created", "response": {"id": "resp-precommit"}})
    collector.add({"type": "input_audio_buffer.committed"})
    # The response that opened before the commit settles after it, and the
    # unit-aligned input left nothing retained: the server acknowledged the
    # commit without scheduling another response.
    collector.add({"type": "response.done", "response": {"id": "resp-precommit"}})
    sent: list[dict[str, object]] = []

    class _Client:
        async def send(self, event):
            sent.append(event)
            collector.add(
                {
                    "type": "error",
                    "error": {"code": "response_create_without_input"},
                }
            )

    fallback = asyncio.run(
        demo._drain_forced_post_commit_response(
            _Client(),
            collector,
            committed_index=1,
            timeout_s=0.05,
            create_timeout_s=0.05,
            settle_quiet_s=0.0,
            commit_may_defer_create=False,
        )
    )

    # A native auto-response session never parks its forced create behind the
    # response that settled: that settle is final, the server is idle, and the
    # re-ask settles the session on the definitive refusal instead of
    # stranding an already-answered session in a client timeout.
    assert sent == [{"type": "response.create"}]
    assert fallback is True


def test_post_commit_settle_lets_a_cancelled_response_close_without_blocking_the_drain():
    demo = _load_demo_module()
    # A cancelled spontaneous response must close its own creation (so its id
    # cannot hang the drain) without counting as the drained answer.
    events = [
        {"type": "input_audio_buffer.committed"},
        {"type": "response.created", "response": {"id": "resp-spont"}},
        {"type": "response.done", "response": {"id": "resp-spont", "status": "cancelled"}},
        {"type": "response.created", "response": {"id": "resp-forced"}},
        {"type": "response.done", "response": {"id": "resp-forced"}},
    ]

    assert demo._post_commit_forced_response_settled(events, 0)


def test_post_commit_settle_does_not_count_a_cancelled_response_as_the_answer():
    demo = _load_demo_module()
    events = [
        {"type": "input_audio_buffer.committed"},
        {"type": "response.created", "response": {"id": "resp-spont"}},
        {"type": "response.done", "response": {"id": "resp-spont", "status": "cancelled"}},
    ]

    # Everything pairs, but the only terminal is cancelled: not drained.
    assert not demo._post_commit_forced_response_settled(events, 0)


def test_post_commit_anonymous_events_settle_best_effort():
    demo = _load_demo_module()
    # Accepted limitation (pinned here so nobody relies on it): without ids
    # nothing pairs, so an id-less pre-commit settle can pass for the answer
    # and only the quiescence guard holds the window open. The server always
    # sends ids on the wire; this only matters for anonymous event shapes.
    events = [
        {"type": "response.created"},
        {"type": "input_audio_buffer.committed"},
        {"type": "response.done"},
        {"type": "response.created"},
    ]

    assert demo._post_commit_forced_response_settled(events, 1)


def test_streaming_collector_acks_each_finished_reply_like_the_web_client():
    demo = _load_demo_module()
    audio = base64.b64encode(bytes(2 * 2400)).decode("ascii")
    raw_events = [
        {"type": "response.created", "response": {"id": "resp-audio"}},
        {"type": "response.output_audio.delta", "response_id": "resp-audio", "delta": audio},
        {"type": "response.done", "response": {"id": "resp-audio"}},
        {"type": "response.created", "response": {"id": "resp-silent"}},
        {"type": "response.done", "response": {"id": "resp-silent"}},
    ]
    acks: list[dict[str, object]] = []

    class _Event:
        def __init__(self, raw):
            self.raw = raw

    class _Client:
        async def events(self):
            for raw in raw_events:
                yield _Event(raw)

        async def ack_playback(self, played_ms, *, response_id=None, item_id=None, committed_ms=None):
            acks.append({"response_id": response_id, "item_id": item_id, "played_ms": int(played_ms)})

    class _Writer:
        def handle(self, event):
            pass

    asyncio.run(demo._StreamingEventCollector(_Writer()).consume(_Client()))

    assert acks == [{"response_id": "resp-audio", "item_id": "item_resp-audio", "played_ms": 100}]


def _asymmetric_image():
    from PIL import Image

    image = Image.new("RGB", (4, 2), (0, 0, 0))
    image.putpixel((0, 0), (255, 0, 0))
    return image


@pytest.mark.parametrize(
    "rotation,expected_corner",
    [
        (0, (0, 0)),
        (90, (0, 3)),
        (180, (3, 1)),
        (-90, (1, 0)),
        (270, (1, 0)),
        (-270, (0, 3)),
    ],
)
def test_upright_applies_the_display_matrix_quarter_turn(rotation, expected_corner):
    demo = _load_demo_module()

    rotated = demo._upright(_asymmetric_image(), rotation)

    assert rotated.getpixel(expected_corner) == (255, 0, 0)


def test_decoded_frames_apply_the_container_display_matrix(monkeypatch, tmp_path):
    """A -90 phone clip must reach the model upright, not on its side."""
    demo = _load_demo_module()
    source = _asymmetric_image()

    class _Frame:
        rotation = -90

        def to_image(self):
            return source

    class _Container:
        streams = type("_Streams", (), {"video": [type("_Stream", (), {})()]})()

        def decode(self, stream):
            yield _Frame()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setitem(sys.modules, "av", type("_Av", (), {"open": staticmethod(lambda path: _Container())}))

    (image,) = demo._decode_video_frames_rgb(tmp_path / "clip.mp4", [0])

    assert image.size == (2, 4)
    assert image.getpixel((1, 0)) == (255, 0, 0)


def test_streaming_output_writer_persists_audio_deltas_as_they_arrive(tmp_path, capsys):
    demo = _load_demo_module()
    writer = demo._StreamingOutputWriter(tmp_path)
    first_pcm = b"\x01\x00\x02\x00"
    second_pcm = b"\x03\x00"

    writer.handle(
        {
            "type": "response.output_audio.delta",
            "response_id": "resp-a",
            "delta": base64.b64encode(first_pcm).decode("ascii"),
            "sample_rate_hz": 24_000,
        }
    )
    writer.handle(
        {
            "type": "response.output_audio_transcript.delta",
            "response_id": "resp-a",
            "delta": "你好",
        }
    )
    writer.handle(
        {
            "type": "response.output_audio.delta",
            "response_id": "resp-a",
            "delta": base64.b64encode(second_pcm).decode("ascii"),
            "sample_rate_hz": 24_000,
        }
    )

    assert (tmp_path / "output.pcm").read_bytes() == first_pcm + second_pcm
    chunk_paths = sorted((tmp_path / "audio_chunks").glob("*.wav"))
    assert [path.name for path in chunk_paths] == ["chunk_0001.wav", "chunk_0002.wav"]
    with wave.open(str(chunk_paths[0]), "rb") as chunk_wav:
        assert chunk_wav.getframerate() == 24_000
        assert chunk_wav.readframes(chunk_wav.getnframes()) == first_pcm
    assert writer.audio_chunk_paths == chunk_paths
    stderr = capsys.readouterr().err
    assert "audio chunk 1" in stderr
    assert "你好" in stderr


@pytest.mark.parametrize("audio_samples", [3 * 16000 + 512, 2 * 16000])
def test_video_soundtrack_aligns_to_video_units(monkeypatch, tmp_path, audio_samples):
    """AAC padding must not create an extra frame-less model unit at EOF."""
    demo = _load_demo_module()
    audio_path = tmp_path / "input.wav"
    pcm = b"\x01\x00" * audio_samples
    demo.write_pcm16_wav(audio_path, pcm, sample_rate_hz=16000)
    monkeypatch.setattr(demo, "_probe_video_frames_and_fps", lambda path: (90, 30.0))
    monkeypatch.setattr(demo, "_extract_video_soundtrack", lambda *args: audio_path)

    def extract(path, work_dir, **kwargs):
        assert kwargs["duration_s"] == 3.0
        return ["f0", "f1", "f2"], [None] * 3

    monkeypatch.setattr(demo, "_extract_video_frames", extract)
    wav, frames, _ = demo._extract_video_input(tmp_path / "clip.mp4", tmp_path, fps=1.0, max_side=0, stack_frames=1)
    aligned = demo.read_pcm16_wav(wav)
    assert aligned == pcm[:96000].ljust(96000, b"\0")
    assert len(frames) == 3
    assert not demo._has_residual_model_unit(aligned, chunk_period_ms=1000)


def test_video_without_soundtrack_uses_video_duration(monkeypatch, tmp_path):
    demo = _load_demo_module()
    monkeypatch.setattr(demo, "_video_has_audio", lambda path: False)
    monkeypatch.setattr(demo, "_probe_video_frames_and_fps", lambda path: (90, 30.0))

    def extract(path, work_dir, **kwargs):
        assert kwargs["duration_s"] == 3.0
        return ["f0", "f1", "f2"], [None] * 3

    monkeypatch.setattr(demo, "_extract_video_frames", extract)
    wav, frames, _ = demo._resolve_duplex_av_inputs(
        input_wav=None,
        input_video="silent.mp4",
        work_dir=tmp_path,
        fps=1.0,
        max_side=0,
    )
    assert wav is None
    assert len(frames) == 3
