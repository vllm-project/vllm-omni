# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""End-to-end action capture over the MiniCPM-o 4.5 duplex route.

Runs on GPU/NPU CI: opens a capture session over the Realtime websocket
(``action_capture``/``seed`` ride the session object, ``extra_body.auto_response``
is the explicit opt-in that lets a websocket commit open a response), drives one
spoken response plus a barge-in episode, and pins the model-implementation exit
criteria at engine scale: complete ordered records with finite sampled logprobs
and internally consistent two-stage quantities, budget-cap / force-listen and
reemit rows carrying no draw (the mid-turn listen rewrite carries the policy's
own), engine-injected silence units carrying ``origin=engine``, one
``effective_seed`` per scope, same-seed replay emitting identical token ids, and
the interrupted scope sealing as ``truncated_interruption``.

Collect semantics follow the retention contract: the episode itself makes NO
in-flight collects (a collect racing a live scope cannot win the one-sealed-set
retention window); once the episode's response has closed, the retained scopes
are drained through ``session.action_records.get`` — one sealed set per request,
most-recent-first — and re-sorted chronologically by
``(epoch, turn_id, emission_ns)``. A barge-in landing right at speak onset seals
an EMPTY ``truncated_interruption`` scope; that is the contract, not a failure.
On the wire barge-in is observed through its projection (``response.done`` with
status ``"cancelled"``); the internal ``audio.cancelled`` event never reaches the
socket.

The exact-distribution math against a trainer-style recomputation is proven at
CPU scale in ``tests/model_executor/models/minicpmo_4_5/test_duplex_action_capture.py``;
here the payload's own per-stage quantities are cross-checked against the
recorded joint (per-stage self-consistency) over real model draws.
"""

from __future__ import annotations

import asyncio
import json
import math
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.online_serving.helpers.minicpmo_4_5_duplex import (
    SERVER_PARAMS,
    realtime_url,
    resolve_ref_audio,
    validated_input_wav,
)
from tests.e2e.online_serving.helpers.minicpmo_realtime_duplex_scenarios import (
    _send_pcm16,
)
from tests.e2e.online_serving.test_minicpmo_4_5_duplex import (
    _collect_socket_events,
    _wait_for_events,
)
from tests.helpers.mark import hardware_test
from vllm_omni.clients.duplex import (
    build_realtime_url,
    read_pcm16_wav,
    reference_audio_data_url,
)

pytestmark = [pytest.mark.omni, pytest.mark.core_model]

#: One episode seed; the per-scope ``effective_seed`` derives from it.
_EPISODE_SEED = 20261118
#: Drain bound: the retention window holds one sealed scope, and the post-close
#: terminal cache answers ``action_capture_not_retained`` once empty.
_DRAIN_REQUEST_CAP = 8


def _capture_session_payload(model: str, ref_audio: Path) -> dict[str, object]:
    """The seeded capture session object.

    The reference audio rides the session object as a base64 data URL — never a
    JSON-dumped path. ``extra_body.auto_response`` is the engine's explicit
    opt-in: without it a websocket ``input_audio_buffer.commit`` never opens a
    response and the episode stalls after ``input_audio_buffer.committed``.
    """
    return {
        "model": model,
        "modalities": ["audio", "text"],
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "turn_detection": None,
        "ref_audio": reference_audio_data_url(str(ref_audio)),
        "action_capture": True,
        "seed": _EPISODE_SEED,
        "extra_body": {"auto_response": True},
    }


async def _recv_reply(ws: Any, types: frozenset[str], *, timeout_s: float) -> dict[str, Any]:
    """Next wire event whose type is in ``types``; other traffic is skipped."""
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
        if not isinstance(raw, str):
            continue
        event = json.loads(raw)
        if isinstance(event, dict) and event.get("type") in types:
            return event


async def _drain_record_sets(ws: Any) -> list[dict[str, Any]]:
    """Drain the ledger's retained window once the episode's responses have closed.

    No collect happens while a scope is live: the drain starts only after the
    response terminal, so nothing can seal between two requests. Each
    ``session.action_records.get`` returns one sealed set, most-recent-first;
    the typed ``action_capture_not_retained`` error answers that the window is
    empty and ends the drain. The caller re-sorts chronologically. The wire
    reaches the one-set window only: scopes the engine spilled earlier in the
    session are retrievable post-close through the programmatic
    ``DuplexOmni.collect_action_records``.
    """
    record_sets: list[dict[str, Any]] = []
    for request_index in range(_DRAIN_REQUEST_CAP):
        await ws.send(json.dumps({"type": "session.action_records.get", "event_id": f"evt_drain_{request_index}"}))
        reply = await _recv_reply(ws, frozenset({"session.action.records", "error"}), timeout_s=60)
        if reply.get("type") == "error":
            error = reply.get("error")
            code = str(error.get("code")) if isinstance(error, dict) else ""
            assert code == "action_capture_not_retained", f"drain collect failed with {code!r}: {reply}"
            break
        records = reply.get("records")
        assert isinstance(records, dict), reply
        status = str(records.get("status"))
        assert status != "in_progress", (
            "drained an in_progress snapshot (a scope was open mid-flight); the episode "
            f"must collect only after its responses closed: {records}"
        )
        record_sets.append(records)
    else:
        raise AssertionError(f"drain never emptied after {_DRAIN_REQUEST_CAP} requests; sets so far: {record_sets}")
    return record_sets


def _chronological_key(record_set: dict[str, Any]) -> tuple[int, int, int]:
    """The trainer's record-set re-sort key: ``(epoch, turn_id, emission_ns)``."""
    emission_ns = next((ns for ns in record_set.get("emission_ns") or () if ns is not None), 0)
    return (int(record_set["epoch"]), int(record_set["turn_id"]), int(emission_ns))


async def _run_capture_episode(url: str, model: str, ref_audio: Path, *, barge_in: bool) -> dict[str, Any]:
    """One websocket episode with capture on; returns the drained record sets.

    Drive order: session.update (capture + seed + ``auto_response`` opt-in),
    paced speech appends, a commit riding the last speech-carrying chunk, the
    response, an optional barge-in at speak onset, then — and only then — the
    post-response drain of the retained window over the wire, then
    ``session.close``.
    """
    import websockets

    events: list[dict[str, object]] = []
    stop = asyncio.Event()
    websocket_url = build_realtime_url(url, model, autostart=False)
    async with websockets.connect(websocket_url, max_size=64 * 1024 * 1024) as ws:
        reader = asyncio.create_task(_collect_socket_events(ws, events, stop))
        try:
            await ws.send(json.dumps({"type": "session.update", "session": _capture_session_payload(model, ref_audio)}))
            await _wait_for_events(
                events,
                lambda: any(event.get("type") == "session.created" for event in events),
                timeout_s=60,
                label="session.created",
            )
            await _wait_for_events(
                events,
                lambda: any(event.get("type") == "session.updated" for event in events),
                timeout_s=60,
                label="session.updated",
            )
            await _send_pcm16(ws, read_pcm16_wav(validated_input_wav()), chunk_ms=200, realtime_delay=True)
            # The commit rides the last speech-carrying chunk while its audio is
            # still pending: a commit sent after the whole feed lands in the
            # engine's silent-input path and is dropped without a response.
            await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
            await _wait_for_events(
                events,
                lambda: any(event.get("type") == "response.speak" for event in events),
                timeout_s=180,
                label="response.speak",
            )
            expected_done_status = "cancelled" if barge_in else "completed"
            if barge_in:
                # Interrupt mid-response once the model starts speaking.
                await ws.send(json.dumps({"type": "barge_in"}))

            def _episode_done() -> bool:
                for event in events:
                    if event.get("type") != "response.done":
                        continue
                    response = event.get("response")
                    if isinstance(response, dict) and response.get("status") == expected_done_status:
                        return True
                return False

            await _wait_for_events(
                events,
                _episode_done,
                timeout_s=180,
                label=f"response.done status={expected_done_status}",
            )
            # The episode made no in-flight collects; the response has closed
            # and sealed. Own the socket (stop the reader), drain, then close.
            stop.set()
            reader.cancel()
            with suppress(asyncio.CancelledError):
                await reader
            record_sets = await _drain_record_sets(ws)
            await ws.send(json.dumps({"type": "session.close"}))
            closed = await _recv_reply(ws, frozenset({"session.closed", "error"}), timeout_s=60)
            assert closed.get("type") == "session.closed", closed
        finally:
            stop.set()
            reader.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await reader
    record_sets.sort(key=_chronological_key)
    event_types = sorted({str(event.get("type")) for event in events})
    return {
        "record_sets": record_sets,
        "event_types": event_types,
        "done_status": _observed_done_status(events),
    }


def _observed_done_status(events: list[dict[str, object]]) -> str | None:
    """The wire's own ``response.done`` status, last one first; ``None`` if none arrived."""
    for event in reversed(events):
        if event.get("type") != "response.done":
            continue
        response = event.get("response")
        return str(response.get("status")) if isinstance(response, dict) else None
    return None


def _check_record_set(record_set: dict[str, Any], *, expect_truncated: bool) -> None:
    from vllm_omni.protocol.duplex.action_records import ACTION_KINDS, RECORD_STATUSES

    status = str(record_set["status"])
    # A drained set is sealed: one of the four terminal statuses, never
    # ``in_progress``.
    assert status in RECORD_STATUSES, status
    if expect_truncated:
        assert status.startswith("truncated_"), status
    else:
        assert status == "completed", status
    kinds = record_set["action_kinds"]
    assert all(kind in ACTION_KINDS for kind in kinds), kinds
    token_ids = record_set["token_ids"]
    logprobs = record_set["logprobs"]
    if expect_truncated and not token_ids:
        # A barge-in landing right at speak onset seals an EMPTY truncated
        # scope; that is the contract, not a failure.
        assert not kinds and not logprobs, record_set
        return
    if not expect_truncated:
        assert token_ids, "a completed spoken response must carry sampled units"
    assert len(token_ids) == len(kinds) == len(logprobs)
    reason_codes = record_set["reason_codes"]
    # Every policy draw carries a finite logprob of the exact distribution —
    # sampled rows and the mid-turn listen rewrite (a forced emission of the
    # policy's own draw). Budget-cap and force-listen rows and terminator
    # re-emissions record no draw.
    for token_id, logprob, kind, reason in zip(token_ids, logprobs, kinds, reason_codes, strict=True):
        assert isinstance(token_id, int) and token_id >= 0
        if kind == "sampled" or reason == "mid_turn_listen_rewrite":
            assert logprob is not None and math.isfinite(logprob), (kind, reason, logprob)
        else:
            assert logprob is None, (kind, reason, logprob)
    # Self-consistency on the terminal row only: the set-level model extension
    # is a last-writer-wins merge over the units, so it describes the joint
    # exactly when the last unit is the sampled one.
    model_extension = (record_set.get("model") or {}).get("minicpmo_duplex", {})
    p0 = model_extension.get("stage0_prob_chunk_eos")
    stage2 = model_extension.get("stage2_logprob")
    sampled = [i for i, kind in enumerate(kinds) if kind == "sampled"]
    if p0 is not None and stage2 is not None and sampled and sampled[-1] == len(kinds) - 1:
        joint = logprobs[sampled[-1]]
        if joint is not None:
            expected = math.log(max(1.0 - p0, 1e-38)) + stage2
            assert math.isclose(joint, expected, rel_tol=1e-3, abs_tol=1e-4), (joint, expected)
    # Input origins: only client-fed or engine-injected silence units appear.
    assert set(record_set["input_origins"]) <= {"client", "engine"}


@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_action_capture_records_a_response(omni_server) -> None:
    result = asyncio.run(
        _run_capture_episode(
            url=realtime_url(omni_server),
            model=omni_server.model,
            ref_audio=resolve_ref_audio(),
            barge_in=False,
        )
    )
    assert result["record_sets"], f"no records collected; events: {result['event_types']}"
    for record_set in result["record_sets"]:
        _check_record_set(record_set, expect_truncated=False)
        # One seed per scope: every drained set carries exactly one effective
        # seed. Different effective seeds across an episode's scopes are
        # legitimate, so cross-set equality is never asserted here.
        assert isinstance(record_set.get("effective_seed"), int), record_set.get("effective_seed")
    keys = [_chronological_key(record_set) for record_set in result["record_sets"]]
    assert keys == sorted(keys), keys  # drained most-recent-first, re-sorted chronologically


@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_action_capture_seals_barge_in_as_truncated(omni_server) -> None:
    result = asyncio.run(
        _run_capture_episode(
            url=realtime_url(omni_server),
            model=omni_server.model,
            ref_audio=resolve_ref_audio(),
            barge_in=True,
        )
    )
    assert result["record_sets"], f"no records collected; events: {result['event_types']}"
    # Barge-in reaches the wire only through its projection: ``response.done``
    # with status "cancelled". The internal ``audio.cancelled`` event never
    # crosses the socket.
    assert result["done_status"] == "cancelled", result["done_status"]
    assert "audio.cancelled" not in result["event_types"], result["event_types"]
    assert any(str(record_set["status"]).startswith("truncated_") for record_set in result["record_sets"]), [
        record_set["status"] for record_set in result["record_sets"]
    ]
    for record_set in result["record_sets"]:
        _check_record_set(record_set, expect_truncated=str(record_set["status"]).startswith("truncated_"))


@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_action_capture_seed_replays_identical_tokens(omni_server) -> None:
    first = asyncio.run(
        _run_capture_episode(
            url=realtime_url(omni_server), model=omni_server.model, ref_audio=resolve_ref_audio(), barge_in=False
        )
    )
    second = asyncio.run(
        _run_capture_episode(
            url=realtime_url(omni_server), model=omni_server.model, ref_audio=resolve_ref_audio(), barge_in=False
        )
    )
    assert first["record_sets"] and second["record_sets"]
    assert len(first["record_sets"]) == len(second["record_sets"]), (
        [item["status"] for item in first["record_sets"]],
        [item["status"] for item in second["record_sets"]],
    )
    for replayed, rerun in zip(first["record_sets"], second["record_sets"], strict=True):
        assert replayed["token_ids"] == rerun["token_ids"]
        assert replayed["logprobs"] == rerun["logprobs"]
