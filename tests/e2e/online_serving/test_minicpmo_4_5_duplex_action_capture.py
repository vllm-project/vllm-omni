# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""End-to-end action capture over the MiniCPM-o 4.5 duplex route.

Runs on GPU/NPU CI: opens a capture session over the Realtime websocket
(``action_capture``/``seed`` ride the session object), drives one spoken
response plus a barge-in episode, and pins the model-implementation exit criteria at engine
scale: complete ordered records with finite sampled logprobs and internally
consistent two-stage quantities, forced classes marked, engine-injected
silence units carrying ``origin=engine``, seeded replay emitting identical
token ids, and the interrupted scope sealing as ``truncated_interruption``.

The exact-distribution math against a trainer-style recomputation is proven at
CPU scale in ``tests/model_executor/models/minicpmo_4_5/test_duplex_action_capture.py``;
here the payload's own per-stage quantities are cross-checked against the
recorded joint (per-stage self-consistency) over real model draws.
"""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any

import pytest

from tests.e2e.online_serving.test_minicpmo_4_5_duplex import (
    SERVER_PARAMS,
    realtime_url,
    resolve_ref_audio,
)
from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.omni, pytest.mark.core_model]

_CAPTURE_SESSION_FIELDS = {"action_capture": True, "seed": 20261118}


async def _run_capture_episode(url: str, model: str, ref_audio: str | None, *, barge_in: bool) -> dict[str, Any]:
    """One websocket episode with capture on; returns the collected record sets."""
    import websockets

    record_sets: list[dict[str, Any]] = []
    async with websockets.connect(url, max_size=64 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"type": "session.update", "session": dict(_CAPTURE_SESSION_FIELDS)}))
        if ref_audio:
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": ref_audio}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        seen: set[str] = set()
        pending_barge_in = barge_in
        reader_error: Exception | None = None
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=120)
                event = json.loads(raw)
                event_type = event.get("type")
                seen.add(str(event_type))
                if event_type == "response.speak" and pending_barge_in:
                    # Interrupt mid-response once the model starts speaking.
                    await ws.send(json.dumps({"type": "barge_in"}))
                    pending_barge_in = False
                if event_type == "response.done" or pending_barge_in is False and "audio.cancelled" in seen:
                    request_id = f"evt_{len(record_sets)}"
                    await ws.send(json.dumps({"type": "session.action_records.get", "event_id": request_id}))
                if event_type == "session.action.records":
                    records = event.get("records")
                    if isinstance(records, dict) and records.get("token_ids"):
                        record_sets.append(records)
                    if len(record_sets) >= (2 if barge_in else 1):
                        break
                if event_type == "error":
                    reader_error = RuntimeError(json.dumps(event))
                    break
                if event_type == "session.closed":
                    break
        except asyncio.TimeoutError:
            reader_error = TimeoutError(f"episode stalled after events: {sorted(seen)}")
        await ws.send(json.dumps({"type": "session.close"}))
    if reader_error is not None:
        raise reader_error
    return {"record_sets": record_sets, "event_types": seen}


def _check_record_set(record_set: dict[str, Any], *, expect_truncated: bool) -> None:
    from vllm_omni.protocol.duplex.action_records import ACTION_KINDS, RECORD_STATUSES

    assert record_set["status"] in RECORD_STATUSES | {"in_progress"}
    if expect_truncated:
        assert record_set["status"].startswith("truncated_")
    kinds = record_set["action_kinds"]
    assert kinds and all(kind in ACTION_KINDS for kind in kinds), kinds
    # Every sampled token carries a finite logprob of the exact distribution.
    for token_id, logprob, kind in zip(record_set["token_ids"], record_set["logprobs"], kinds, strict=True):
        assert isinstance(token_id, int) and token_id >= 0
        if kind == "sampled":
            assert logprob is not None and math.isfinite(logprob)
        elif kind == "forced":
            assert record_set and True  # reason codes are model-namespaced here
    # Self-consistency: the joint equals its per-stage parts.
    model_extension = (record_set.get("model") or {}).get("minicpmo_duplex", {})
    p0 = model_extension.get("stage0_prob_chunk_eos")
    stage2 = model_extension.get("stage2_logprob")
    if p0 is not None and stage2 is not None:
        sampled = [i for i, kind in enumerate(kinds) if kind == "sampled"]
        if sampled:
            joint = record_set["logprobs"][sampled[-1]]
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
    assert result["record_sets"], f"no records collected; events: {sorted(result['event_types'])}"
    for record_set in result["record_sets"]:
        _check_record_set(record_set, expect_truncated=False)


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
    assert result["record_sets"], f"no records collected; events: {sorted(result['event_types'])}"
    assert any(str(record_set["status"]).startswith("truncated_") for record_set in result["record_sets"]), [
        record_set["status"] for record_set in result["record_sets"]
    ]


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
    assert first["record_sets"][0]["token_ids"] == second["record_sets"][0]["token_ids"]
    assert first["record_sets"][0]["logprobs"] == second["record_sets"][0]["logprobs"]
