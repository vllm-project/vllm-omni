# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Live E2E coverage for the public duplex client (``vllm_omni.clients.duplex``).

Drives a real MiniCPM-o 4.5 duplex session end to end through
:class:`DuplexClient`: session handshake with reference audio, paced PCM
streaming, a speak response consumed via :class:`ResponseHandle` with
incremental playback acks, a forced transport drop recovered by automatic
``session.resume``, and a clean close.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest

from tests.e2e.online_serving.helpers.minicpmo_4_5_duplex import (
    CORE_SERVER_PARAMS,
    realtime_url,
    resolve_ref_audio,
    validated_input_wav,
)
from tests.helpers.mark import hardware_test
from vllm_omni.clients.duplex import (
    DuplexClient,
    EventCollector,
    ReconnectPolicy,
    audio_data_url,
    read_pcm16_wav,
)
from vllm_omni.clients.minicpmo_4_5 import create_duplex_session_config

pytestmark = pytest.mark.omni

_SESSION_TIMEOUT_S = 900.0
_MAX_LISTEN_ROUNDS = 30


async def _run_live_session(*, url: str, model: str, ref_audio: Path, input_wav: Path) -> dict[str, object]:
    # temperature 0.0 keeps the response-required reply short and
    # deterministic (same as the validated demo driver).
    config = create_duplex_session_config(ref_audio=audio_data_url(ref_audio), temperature=0.0)
    client = DuplexClient(
        url,
        model=model,
        config=config,
        session_id=f"duplex-client-live-{uuid.uuid4().hex}",
        reconnect=ReconnectPolicy(max_attempts=5, backoff_s=(0.25, 1.0)),
        # Short interval so the periodic heartbeat path runs during the
        # session — including across the forced transport drop below.
        heartbeat_interval_s=5.0,
    )
    collector = EventCollector()
    pcm = read_pcm16_wav(input_wav)
    summary: dict[str, object] = {}
    async with client:
        collector_task = asyncio.create_task(collector.consume(client))
        summary["session_id"] = client.session_id
        summary["resume_token_issued"] = client.resume_token is not None

        # Phase 1: stream the validated utterance and consume one speak response.
        await client.stream_pcm(pcm, chunk_ms=200)
        await client.commit()
        listens = 0
        async for response in client.responses():
            if response.decision == "listen":
                listens += 1
                if listens > _MAX_LISTEN_ROUNDS:
                    raise AssertionError(f"no speak response after {listens} listen decisions")
                continue
            chunk_count = 0
            audio_byte_count = 0
            async for chunk in response.audio():
                chunk_count += 1
                audio_byte_count += len(chunk)
                await client.ack_playback(response.played_ms)
            summary.update(
                {
                    "decision": response.decision,
                    "listen_rounds": listens,
                    "audio_chunks": chunk_count,
                    "audio_bytes": audio_byte_count,
                    "played_ms": response.played_ms,
                    "transcript": response.transcript,
                }
            )
            break

        # Phase 2: force-drop the transport underneath the client and require
        # the automatic session.resume path to bring the session back.
        resumed_waiter = asyncio.create_task(client.wait_for("session.resumed", timeout_s=90.0))
        await asyncio.sleep(0.1)  # let the waiter subscribe before the drop
        assert client._ws is not None
        await client._ws.close()
        resumed = await resumed_waiter
        summary["resumed_session_id"] = resumed.session_id or client.session_id

        # The resumed transport must be live: a heartbeat gets acknowledged.
        ack_waiter = asyncio.create_task(client.wait_for("session.heartbeat_ack", timeout_s=30.0))
        await asyncio.sleep(0.1)
        await client.send({"type": "session.heartbeat"})
        await ack_waiter
        summary["post_resume_heartbeat_ok"] = True

        await client.close()
        await asyncio.wait_for(collector_task, timeout=10.0)

    summary["error_events"] = collector.errors()
    summary["closed_cleanly"] = collector.count("session.closed") > 0
    summary["timing"] = collector.timing_summary(after_s=0.0)
    return summary


@pytest.mark.core_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", CORE_SERVER_PARAMS, indirect=True)
def test_duplex_client_live_session(omni_server) -> None:
    summary = asyncio.run(
        asyncio.wait_for(
            _run_live_session(
                url=realtime_url(omni_server),
                model=omni_server.model,
                ref_audio=resolve_ref_audio(),
                input_wav=validated_input_wav(),
            ),
            timeout=_SESSION_TIMEOUT_S,
        )
    )
    assert summary["decision"] == "speak", summary
    assert summary["audio_chunks"] > 0, summary
    assert summary["played_ms"] > 0, summary
    assert summary["transcript"], summary
    assert summary["resume_token_issued"] is True, summary
    assert summary["post_resume_heartbeat_ok"] is True, summary
    assert summary["closed_cleanly"] is True, summary
    assert summary["error_events"] == [], summary
