"""CI coverage for the MiniCPM-o 4.5 native-duplex Realtime API."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import pytest
import websockets

from tests.e2e.online_serving.minicpmo_realtime_duplex_scenarios import (
    _ref_audio_data_url,
    run_demo,
)
from tests.e2e.online_serving.run_minicpmo_realtime_duplex_multi_session import (
    run_multi_session,
)
from tests.helpers.mark import hardware_test
from tests.helpers.minicpmo_4_5_duplex import (
    CORE_SERVER_PARAMS,
    SERVER_PARAMS,
    demo_args,
    multi_session_args,
    realtime_url,
    resolve_ref_audio,
    validated_input_wav,
)
from vllm_omni.experimental.fullduplex.client import build_realtime_url

pytestmark = pytest.mark.omni


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
    websocket_url = build_realtime_url(url, model, autostart=False, session_id=session_id)
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
                        "extra_body": {"minicpmo45_native_duplex": True},
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
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", CORE_SERVER_PARAMS, indirect=True)
def test_duplex_websocket_protocol_smoke(omni_server, model_prefix: str) -> None:
    ref_audio = resolve_ref_audio(model_prefix)
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
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_single_session_response_required(omni_server, model_prefix: str, tmp_path: Path) -> None:
    result = asyncio.run(
        run_demo(
            demo_args(
                omni_server=omni_server,
                input_wav=validated_input_wav(),
                ref_audio=resolve_ref_audio(model_prefix),
                output_dir=tmp_path / "single_session",
            )
        )
    )
    assert result["ok"] is True
    assert result["audio_delta_count"] > 0
    assert result["done_count"] == 1
    assert result["error_count"] == 0
    assert result["all_audio_responses_have_transcript"] is True
    assert result["transcript_delta_done_ok"] is True


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_two_sessions_resume_and_takeover(omni_server, model_prefix: str, tmp_path: Path) -> None:
    result = asyncio.run(
        run_multi_session(
            multi_session_args(
                omni_server=omni_server,
                input_wav=validated_input_wav(),
                ref_audio=resolve_ref_audio(model_prefix),
                output_dir=tmp_path / "multi_session",
                response_required=True,
            )
        )
    )
    assert result["ok"] is True
    assert result["session_count"] == 2
    assert result["resume"]["ok"] is True
    assert result["takeover"]["ok"] is True
    assert not result["failures"]
    assert all(session["audio_delta_count"] > 0 for session in result["sessions"])
    assert all(session["done_count"] == 1 for session in result["sessions"])
    assert all(session["error_count"] == 0 for session in result["sessions"])
