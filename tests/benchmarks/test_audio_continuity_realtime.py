# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Reply-local PCM continuity: no cross-turn silence or fake passing samples."""

import base64

import pytest

from vllm_omni.benchmarks.audio_continuity import (
    audio_continuity_threshold_s,
    collect_duplex_audio_continuity,
    continuity_sample,
)

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


def audio(rid, rate=24000, duration=0.1):
    return {
        "type": "response.output_audio.delta",
        "response_id": rid,
        "sample_rate_hz": rate,
        "delta": base64.b64encode(bytes(int(rate * duration) * 2)).decode(),
    }


def done(rid, status="completed"):
    return {"type": "response.done", "response": {"id": rid, "status": status}}


def test_replies_are_isolated_and_cancelled_listen_incomplete_are_excluded():
    events = [
        audio("a"),
        audio("a"),
        done("a"),
        audio("b"),
        audio("b"),
        done("b"),
        audio("cancel"),
        audio("cancel"),
        done("cancel", "cancelled"),
        {"type": "response.listen", "response_id": "listen"},
        done("listen"),
        audio("unfinished"),
    ]
    rows = collect_duplex_audio_continuity(events, [1, 1.05, 2, 100, 100.5, 101, 102, 105, 106, 107, 108, 109])
    assert set(rows) == {"a", "b"}
    assert rows["a"]["audio_underrun_s"] == 0
    assert rows["a"]["audio_continuity_ok"] is True
    assert rows["b"]["audio_underrun_s"] == pytest.approx(0.4)
    assert rows["b"]["audio_continuity_ok"] is False


def test_late_audio_and_done_wait_do_not_add_stalls():
    rows = collect_duplex_audio_continuity([audio("a"), done("a"), audio("a")], [1, 100, 200])
    assert rows["a"]["audio_chunk_count"] == 1
    assert rows["a"]["audio_underrun_s"] == 0


def test_each_packet_uses_its_actual_sample_rate():
    rows = collect_duplex_audio_continuity([audio("a", 16000), audio("a", 48000), done("a")], [1, 1.3, 2])
    assert rows["a"]["audio_underrun_s"] == pytest.approx(0.2)


@pytest.mark.parametrize("bad", ["!bad-base64!", base64.b64encode(b"x").decode()])
def test_malformed_audio_invalidates_reply(bad):
    event = audio("a")
    event["delta"] = bad
    assert collect_duplex_audio_continuity([audio("a"), event, done("a")], [1, 2, 3]) == {}


def test_unsupported_format_and_pre_window_audio_are_unmeasured():
    session = {"type": "session.created", "session": {"output_audio_format": "g711_ulaw"}}
    assert collect_duplex_audio_continuity([session, audio("a"), done("a")], [0, 1, 2]) == {}
    assert collect_duplex_audio_continuity([audio("a"), done("a")], [1, 2], after_s=3) == {}


@pytest.mark.parametrize(
    "times,durations", [([], []), ([0], []), ([0], [0]), ([1, 0], [0.1, 0.1]), ([float("nan")], [0.1])]
)
def test_missing_or_invalid_stream_is_not_a_passing_sample(times, durations):
    assert continuity_sample(times, durations, threshold_s=0.1) is None


def test_threshold_boundary_and_first_packet_delay():
    sample = continuity_sample([100, 100.2], [0.1, 0.1], threshold_s=0.1)
    assert sample["audio_continuity_ok"] is True
    assert sample["audio_underrun_s"] == pytest.approx(0.1)


@pytest.mark.parametrize("value,expected", [("0.25", 0.25), ("nan", 0.1), ("inf", 0.1), ("bad", 0.1)])
def test_threshold_is_shared_and_finite(monkeypatch, value, expected):
    monkeypatch.setenv("VLLM_OMNI_BENCH_AUDIO_CONTINUITY_THRESHOLD_S", value)
    assert audio_continuity_threshold_s() == expected


def test_interleaved_replies_do_not_share_buffers():
    rows = collect_duplex_audio_continuity(
        [audio("a"), audio("b"), audio("a"), done("a"), audio("b"), done("b")],
        [0, 0.1, 0.5, 0.6, 0.8, 0.9],
    )
    assert rows["a"]["audio_underrun_s"] == pytest.approx(0.4)
    assert rows["b"]["audio_underrun_s"] == pytest.approx(0.6)


def test_session_sample_rate_is_used_when_packet_rate_is_absent():
    packet = audio("a", 16000)
    del packet["sample_rate_hz"]
    session = {"type": "session.updated", "session": {"audio": {"output": {"sample_rate_hz": 16000}}}}
    rows = collect_duplex_audio_continuity([session, packet, packet, done("a")], [0, 1, 1.3, 2])
    assert rows["a"]["audio_underrun_s"] == pytest.approx(0.2)


def test_continuity_reaches_session_rows_without_server_timing_metrics():
    from vllm_omni.benchmarks.duplex_session_metrics import collect_duplex_session_metrics
    from vllm_omni.clients.duplex import EventCollector

    collector = EventCollector()
    events = [
        (0.0, {"type": "response.created", "response": {"id": "a"}}),
        (0.3, audio("a", duration=0.1)),
        (0.45, audio("a", duration=0.2)),
        (0.85, audio("a", duration=0.2)),
        (2.0, done("a")),
        (100.0, {"type": "response.created", "response": {"id": "b"}}),
        (101.0, audio("b")),
        (101.05, audio("b")),
        (102.0, done("b")),
    ]
    for arrival, event in events:
        collector.add(event, received_at_s=arrival)
    bundle = collect_duplex_session_metrics(collector, stream_start=0.0, session_id="session")
    samples = {row["response_id"]: row["audio_continuity"] for row in bundle.request_metrics}
    assert samples["a"]["audio_max_single_underrun_s"] == pytest.approx(0.2)
    assert samples["a"]["audio_underrun_s"] == pytest.approx(0.25)
    assert samples["a"]["total_underrun_s"] == pytest.approx(0.25)
    assert samples["a"]["audio_underrun_event_count"] == 2
    assert samples["a"]["audio_underrun_ratio"] == pytest.approx(1 / 3)
    assert samples["a"]["audio_underrun_events_per_minute"] == pytest.approx(240)
    assert samples["b"]["audio_underrun_s"] == 0
