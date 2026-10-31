# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for the real-model regression's fixtures and failure oracle."""

import json
from pathlib import Path

import pytest

from tests.e2e.online_serving.helpers.minicpmo_duplex_recognition import (
    ASSETS,
    covered_scenes,
    evaluate_session,
    load_manifest,
    unpack_inputs,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _completed_events() -> list[dict]:
    return [
        {"type": "response.created", "response": {"id": "one"}},
        {"type": "response.output_audio.delta", "delta": "AAAA"},
        {"type": "input_audio_buffer.committed"},
        {"type": "response.done", "response": {"id": "one", "status": "completed"}},
        {"type": "session.closed"},
    ]


def _recordings() -> dict:
    return json.loads((ASSETS / "recorded_transcripts.json").read_text(encoding="utf-8"))


def test_pinned_fixture_is_exactly_the_validated_180_second_input(tmp_path: Path) -> None:
    wav, frames = unpack_inputs(tmp_path)
    assert wav.is_file()
    assert len(frames) == 180
    assert len(set(frames)) > 12  # Preserve the moving video, not just one still per scene.


@pytest.mark.parametrize(
    "name",
    [
        "omni_sdpa",
        "omni_ref_tail_fixed_sdpa",
        "omni_ref_tail_fixed_eager",
        "omni_final_clean",
        "transformers_4_51_3_eager",
    ],
)
def test_recorded_before_after_and_transformers_coverage(name: str) -> None:
    recording = _recordings()[name]
    scenes = load_manifest()["scenes"]
    assert covered_scenes(recording["transcript"], scenes) == recording["covered_scenes"]
    # Synthetic lifecycle events test the oracle, not a fake server inference.
    report = evaluate_session(recording["transcript"], _completed_events(), scenes)
    assert report["passed"] is (name in {"omni_final_clean", "transformers_4_51_3_eager"})


@pytest.mark.parametrize("text", ["红色圆形，数字十一。", "红色圆形，数字12。", "红色。圆形。数字一。"])
def test_number_prefix_and_cross_sentence_matches_do_not_pass(text: str) -> None:
    assert covered_scenes(text, load_manifest()["scenes"][:1]) == []


@pytest.mark.parametrize("failure", ["missing_done", "wrong_id", "duplicate_done", "failed", "timeout", "no_commit"])
def test_correct_transcript_does_not_hide_protocol_failure(failure: str) -> None:
    events = _completed_events()
    if failure == "missing_done":
        events.pop(3)
    elif failure == "wrong_id":
        events[3]["response"]["id"] = "other"
    elif failure == "duplicate_done":
        events.append(events[3])
    elif failure == "failed":
        events[3]["response"]["status"] = "failed"
    elif failure == "timeout":
        events.append({"type": "client.timeout", "message": "drain timeout"})
    else:
        events.pop(2)
    report = evaluate_session(_recordings()["omni_final_clean"]["transcript"], events, load_manifest()["scenes"])
    assert report["missing_scenes"] == []
    assert report["protocol_errors"]
    assert not report["passed"]
