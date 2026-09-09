# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CI coverage for the MiniCPM-o 4.5 native-duplex Realtime API."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from urllib.request import urlopen

import pytest
import websockets

from tests.e2e.online_serving.helpers.minicpmo_4_5_duplex import (
    FOUR_SESSION_SERVER_PARAMS,
    SERVER_PARAMS,
    demo_args,
    deploy_max_sessions,
    duplex_camera_frames,
    multi_session_args,
    realtime_url,
    resolve_ref_audio,
    validated_input_wav,
)
from tests.e2e.online_serving.helpers.minicpmo_realtime_duplex_scenarios import (
    _ref_audio_data_url,
    run_demo,
)
from tests.e2e.online_serving.run_minicpmo_realtime_duplex_multi_session import (
    run_multi_session,
)
from tests.helpers.mark import hardware_test
from vllm_omni.clients.duplex import build_realtime_url
from vllm_omni.experimental.fullduplex.video_stacking import concat_frames_b64

pytestmark = pytest.mark.omni

_FOUR_SESSION_ASYNC_COVERAGE = [
    *FOUR_SESSION_SERVER_PARAMS,
    pytest.param(
        FOUR_SESSION_SERVER_PARAMS[0]
        .values[0]
        ._replace(
            server_args=["--trust-remote-code", "--stage-overrides", '{"0":{"async_scheduling":true}}'],
            # Omni workers use V1. Resolve the scheduler's mode in the same
            # server process, including CI paths not using the project wrapper.
            env_dict={"VLLM_USE_V2_MODEL_RUNNER": "0"},
        ),
        id="four-session-stage0-async-on",
    ),
]


def _assert_positive_int(value: object) -> None:
    assert isinstance(value, int)
    assert value > 0


def _assert_request_metrics(metrics: object, *, expected_count: int) -> None:
    assert isinstance(metrics, list)
    assert len(metrics) == expected_count
    for request_index, request in enumerate(metrics):
        assert isinstance(request["session_id"], str)
        assert request["request_index"] == request_index
        assert isinstance(request["response_id"], str)
        assert request["ttft_ms"] is not None and request["ttft_ms"] >= 0
        assert request["ttfp_ms"] >= 0
        assert request["rtf"] is not None and request["rtf"] >= 0
        assert request["audio_generation_ms"] >= 0
        assert request["audio_duration_ms"] > 0


def _assert_session_metrics(metrics: object, *, expected_count: int) -> None:
    assert isinstance(metrics, dict)
    assert isinstance(metrics["session_id"], str)
    assert metrics["audio_turn_count"] == expected_count
    assert metrics["mean_ttft_ms"] is not None and metrics["mean_ttft_ms"] >= 0
    assert metrics["mean_ttfp_ms"] is not None and metrics["mean_ttfp_ms"] >= 0
    assert metrics["mean_rtf"] is not None and metrics["mean_rtf"] >= 0


def _prometheus_metric_values(metrics: str, name: str, *required_labels: str) -> list[float]:
    values: list[float] = []
    for line in metrics.splitlines():
        if not line.startswith(name) or any(label not in line for label in required_labels):
            continue
        try:
            values.append(float(line.rsplit(" ", 1)[-1]))
        except ValueError:
            continue
    return values


async def _receive_protocol_events(ws, required_types: set[str], *, timeout_s: float) -> list[dict[str, object]]:
    async def receive() -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        seen: set[str] = set()
        while not required_types.issubset(seen):
            raw = await ws.recv()
            if not isinstance(raw, str):
                continue
            event = json.loads(raw)
            if not isinstance(event, dict):
                continue
            events.append(event)
            event_type = event.get("type")
            if event_type == "error":
                raise AssertionError(f"WebSocket protocol smoke received an error: {event}")
            if isinstance(event_type, str):
                seen.add(event_type)
        return events

    return await asyncio.wait_for(receive(), timeout=timeout_s)


async def _run_protocol_smoke(*, url: str, model: str, ref_audio: Path) -> list[dict[str, object]]:
    session_id = f"duplex-ci-protocol-{uuid.uuid4().hex}"
    websocket_url = build_realtime_url(
        url,
        model,
        autostart=False,
        session_id=session_id,
        extra_query={"native_duplex": "1"},
    )
    async with websockets.connect(websocket_url, max_size=64 * 1024 * 1024) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "session_id": session_id,
                        "model": model,
                        "modalities": ["audio", "text"],
                        "ref_audio": _ref_audio_data_url(str(ref_audio)),
                        "extra_body": {"native_duplex": True},
                    },
                }
            )
        )
        events = await _receive_protocol_events(
            ws,
            {"session.created", "session.updated"},
            timeout_s=60,
        )
        await ws.send(json.dumps({"type": "session.close"}))
        events.extend(await _receive_protocol_events(ws, {"session.closed"}, timeout_s=60))
    return events


@pytest.mark.core_model
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_websocket_protocol_smoke(omni_server) -> None:
    ref_audio = resolve_ref_audio()
    events = asyncio.run(
        _run_protocol_smoke(
            url=realtime_url(omni_server),
            model=omni_server.model,
            ref_audio=ref_audio,
        )
    )
    event_types = [event.get("type") for event in events]
    assert "session.created" in event_types
    assert "session.updated" in event_types
    assert event_types[-1] == "session.closed"


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_single_session_response_required(omni_server, tmp_path: Path) -> None:
    args = demo_args(
        omni_server=omni_server,
        input_wav=validated_input_wav(),
        ref_audio=resolve_ref_audio(),
        output_dir=tmp_path / "single_session",
    )
    args.turns = 2
    args.timeout_s = 180.0
    # Every turn replays the same active-speech window as the first one. The default
    # shorter follow-up window is a different mid-utterance slice, which the native
    # duplex model may legitimately answer with "listen" instead of a response.
    args.turn_duration_ms = [args.first_turn_ms] * args.turns
    args.emit_duplex_control_results = True
    result = asyncio.run(run_demo(args))
    assert result["ok"] is True
    _assert_positive_int(result["audio_delta_count"])
    assert result["done_count"] == 2
    assert result["native_model_turn_end_ok"] is True
    assert result["model_turn_end_count"] == 2
    assert result["model_turn_end_response_ids"] == result["completed_response_ids"]
    assert result["error_count"] == 0
    assert result["all_audio_responses_have_transcript"] is True
    assert result["transcript_delta_done_ok"] is True
    _assert_request_metrics(result["request_metrics"], expected_count=2)
    _assert_session_metrics(result["session_metrics"], expected_count=2)
    append_observations = result["native_append_observations"]
    assert isinstance(append_observations, list)
    assert append_observations
    assert len({item["request_id"] for item in append_observations}) == 1
    assert len({item["replica_id"] for item in append_observations}) == 1
    context_tokens = [item["omni_context_tokens"] for item in append_observations]
    assert context_tokens == sorted(context_tokens)
    assert all(item["omni_context_limit"] == 40960 for item in append_observations)


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_single_session_video_input(omni_server, tmp_path: Path) -> None:
    """Audio plus a 1 fps camera track, the omni-duplex video contract.

    Frames ride the audio appends, so the response contract is unchanged: what
    this covers is that interleaved vision input keeps the turn intact instead
    of stalling or erroring out mid-segment.
    """
    args = demo_args(
        omni_server=omni_server,
        input_wav=validated_input_wav(),
        ref_audio=resolve_ref_audio(),
        output_dir=tmp_path / "video_input",
    )
    args.turns = 2
    args.turn_duration_ms = [args.first_turn_ms] * args.turns
    frames = duplex_camera_frames(seconds=4, cache_dir=tmp_path / "camera")
    args.video_frames_b64 = frames
    # Each unit also carries a composite of its interior sub-frames, so the
    # append exercises the official two-image frame_list that carries motion.
    args.video_stacked_frames_b64 = [concat_frames_b64([frames[index]] * 2) for index in range(len(frames))]

    result = asyncio.run(run_demo(args))

    assert result["ok"] is True
    assert result["video_frame_count"] == 4
    assert result["video_stacked_frame_count"] == 4
    _assert_positive_int(result["audio_delta_count"])
    assert result["done_count"] == 2
    assert result["error_count"] == 0
    assert result["all_audio_responses_have_transcript"] is True
    assert result["transcript_delta_done_ok"] is True
    _assert_request_metrics(result["request_metrics"], expected_count=2)
    _assert_session_metrics(result["session_metrics"], expected_count=2)


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_two_sessions_resume_and_takeover(omni_server, tmp_path: Path) -> None:
    result = asyncio.run(
        run_multi_session(
            multi_session_args(
                omni_server=omni_server,
                input_wav=validated_input_wav(),
                ref_audio=resolve_ref_audio(),
                output_dir=tmp_path / "multi_session",
                response_required=True,
            )
        )
    )
    assert result["ok"] is True
    assert result["session_count"] == 2
    assert isinstance(result["resume"], dict)
    assert isinstance(result["takeover"], dict)
    assert isinstance(result["sessions"], list)
    assert result["resume"]["ok"] is True
    assert result["takeover"]["ok"] is True
    assert not result["failures"]
    assert all(session["audio_delta_count"] > 0 for session in result["sessions"])
    assert all(session["done_count"] == 1 for session in result["sessions"])
    assert result["native_model_turn_end_ok"] is True
    assert result["model_turn_end_count"] == 2
    assert all(session["model_turn_end_count"] == 1 for session in result["sessions"])
    assert all(session["error_count"] == 0 for session in result["sessions"])
    for session in result["sessions"]:
        _assert_request_metrics(session["request_metrics"], expected_count=1)
        _assert_session_metrics(session["session_metrics"], expected_count=1)


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_configured_capacity_synchronized_isolation(omni_server, tmp_path: Path) -> None:
    args = multi_session_args(
        omni_server=omni_server,
        input_wav=validated_input_wav(),
        ref_audio=resolve_ref_audio(),
        output_dir=tmp_path / "configured_capacity_isolation",
        response_required=True,
    )
    args.sessions = deploy_max_sessions()
    assert args.sessions == 4
    args.disconnect_session_index = None
    args.takeover_session_index = None
    args.synchronized_start = True
    args.emit_duplex_control_results = True

    result = asyncio.run(run_multi_session(args))

    assert result["ok"] is True, json.dumps(result, ensure_ascii=False, indent=2)
    assert result["session_count"] == args.sessions
    assert result["identity_isolation_ok"] is True
    assert result["semantic_isolation_ok"] is True
    assert not result["failures"]
    assert isinstance(result["sessions"], list)
    assert all(session["done_count"] == 1 for session in result["sessions"])
    assert all(session["error_count"] == 0 for session in result["sessions"])
    observations_by_session = [session["native_append_observations"] for session in result["sessions"]]
    assert all(observations for observations in observations_by_session)
    assert all(item["streaming_prompt"] is True for observations in observations_by_session for item in observations)
    assert all(any(item["appended_tokens"] > 0 for item in observations) for observations in observations_by_session)
    request_ids_by_session = [{item["request_id"] for item in observations} for observations in observations_by_session]
    assert all(len(request_ids) == 1 for request_ids in request_ids_by_session)
    assert len(set().union(*request_ids_by_session)) == args.sessions
    assert all(len({item["replica_id"] for item in observations}) == 1 for observations in observations_by_session)
    assert all(
        [item["omni_context_tokens"] for item in observations]
        == sorted(item["omni_context_tokens"] for item in observations)
        for observations in observations_by_session
    )
    assert all(item["omni_context_tokens"] > 0 for observations in observations_by_session for item in observations)
    assert all(item["omni_context_limit"] == 40960 for observations in observations_by_session for item in observations)
    receipt_counts_by_session = [
        [item["omni_append_receipt_count"] for item in observations] for observations in observations_by_session
    ]
    assert all(receipt_counts == sorted(receipt_counts) for receipt_counts in receipt_counts_by_session)
    assert all(receipt_counts[-1] >= 1 for receipt_counts in receipt_counts_by_session)


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _FOUR_SESSION_ASYNC_COVERAGE, indirect=True)
def test_duplex_four_sessions_rotate_without_starvation_across_multiple_turns(omni_server, tmp_path: Path) -> None:
    trial_summaries: list[dict[str, object]] = []
    for trial in range(2):
        args = multi_session_args(
            omni_server=omni_server,
            input_wav=validated_input_wav(),
            ref_audio=resolve_ref_audio(),
            output_dir=tmp_path / f"four_session_trial_{trial}",
            response_required=True,
        )
        args.sessions = 4
        args.turns = 2
        args.turn_duration_ms = [args.first_turn_ms] * args.turns
        args.disconnect_session_index = None
        args.takeover_session_index = None
        args.synchronized_start = True
        args.emit_duplex_control_results = True
        args.verify_admission_limit = 4 if trial == 0 else None
        args.timeout_s = 180.0

        started_at = time.monotonic()
        result = asyncio.run(run_multi_session(args))
        elapsed_s = time.monotonic() - started_at

        assert result["ok"] is True, json.dumps(result, ensure_ascii=False, indent=2)
        assert result["session_count"] == 4
        assert result["identity_isolation_ok"] is True
        assert not result["failures"]
        sessions = result["sessions"]
        assert isinstance(sessions, list)
        assert len(sessions) == 4
        assert all(session["done_count"] == 2 for session in sessions)
        assert result["native_model_turn_end_ok"] is True
        assert result["model_turn_end_count"] == 8
        assert all(session["native_model_turn_end_ok"] is True for session in sessions)
        assert all(session["model_turn_end_count"] == 2 for session in sessions)
        assert all(session["error_count"] == 0 for session in sessions)
        assert all(session["audio_delta_count"] > 0 for session in sessions)
        assert all(len(session["request_metrics"]) == 2 for session in sessions)
        assert all(session["native_append_observations"] for session in sessions)
        assert all(
            len({item["request_id"] for item in session["native_append_observations"]}) == 1 for session in sessions
        )
        ttft_ms = [
            metric["ttft_ms"]
            for session in sessions
            for metric in session["request_metrics"]
            if metric["ttft_ms"] is not None
        ]
        assert len(ttft_ms) == 8
        assert max(ttft_ms) < args.timeout_s * 1000
        trial_summaries.append(
            {
                "trial": trial,
                "elapsed_s": elapsed_s,
                "responses": len(ttft_ms),
                "responses_per_s": len(ttft_ms) / elapsed_s,
                "mean_ttft_ms": sum(ttft_ms) / len(ttft_ms),
                "max_ttft_ms": max(ttft_ms),
            }
        )

    with urlopen(f"http://{omni_server.host}:{omni_server.port}/metrics", timeout=15) as response:  # noqa: S310
        metrics = response.read().decode("utf-8")
    assert _prometheus_metric_values(metrics, "vllm_omni:duplex_append_requests_total", 'outcome="success"')
    assert _prometheus_metric_values(metrics, "vllm_omni:duplex_append_latency_s_count", 'outcome="success"')
    assert _prometheus_metric_values(metrics, "vllm_omni:duplex_control_queue_wait_s_count", 'operation="append"')
    print("FOUR_SESSION_ENVELOPE " + json.dumps(trial_summaries, sort_keys=True), flush=True)
