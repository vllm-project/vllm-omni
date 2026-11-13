# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage0 text projection for a Thinker-only MiniCPM-o deployment."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.model_executor.models.minicpmo_4_5.duplex.data_plane import (
    MiniCPMO45DataPlaneContext,
    MiniCPMO45DataPlaneSession,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

REQUEST_ID = "duplex-s.c2Vzc2lvbg==.e.0.r.stage0"
LISTEN, SPEAK, CHUNK_EOS, CHUNK_TTS_EOS, TURN_EOS = 7, 8, 5, 6, 9
SPECIAL_TOKEN_IDS = {
    "listen_token_id": LISTEN,
    "speak_token_id": SPEAK,
    "chunk_eos_token_id": CHUNK_EOS,
    "chunk_tts_eos_token_id": CHUNK_TTS_EOS,
    "turn_eos_token_id": TURN_EOS,
}
A, B = 100, 101


def _context(**overrides: object) -> MiniCPMO45DataPlaneContext:
    values: dict[str, object] = {"auto_responds": True, "modalities": ("text",), "text_output": True}
    values.update(overrides)
    return MiniCPMO45DataPlaneContext(**values)


def _snapshot(
    token_ids: list[int],
    text: str,
    *,
    stop: bool = False,
    special: dict[str, int] | None = SPECIAL_TOKEN_IDS,
) -> SimpleNamespace:
    """A Stage0 output as the resumable request reports it: every token so far."""
    multimodal_output: dict[str, object] = {"latent": np.ones((1, 16), dtype=np.float32)}
    if special is not None:
        multimodal_output["special_token_ids"] = dict(special)
    completion = SimpleNamespace(
        token_ids=list(token_ids),
        text=text,
        finish_reason="stop" if stop else None,
        stop_reason=token_ids[-1] if stop and token_ids else None,
        multimodal_output={},
    )
    return SimpleNamespace(
        request_id=REQUEST_ID,
        outputs=[completion],
        finished=False,
        multimodal_output=multimodal_output,
    )


def _project(
    session: MiniCPMO45DataPlaneSession, output: SimpleNamespace, **context: object
) -> list[dict[str, object]]:
    return list(session.project_output(output, context=_context(**context)))


def _session() -> tuple[MiniCPMO45DataPlaneSession, list[object]]:
    encoded: list[object] = []

    def encode(audio: object, *args: object) -> str:
        encoded.append(audio)
        return "pcm"

    return MiniCPMO45DataPlaneSession(encode), encoded


def _texts(results: list[dict[str, object]]) -> str:
    return "".join(str(result.get("text") or "") for result in results)


def test_latent_is_never_encoded_as_audio() -> None:
    session, encoded = _session()
    results = _project(session, _snapshot([SPEAK, A], "嗯"))
    assert encoded == []
    assert not any(result.get("audio_data") for result in results)
    assert all(result["stage_role"] == "llm" for result in results)
    assert _texts(results) == "嗯"


def test_listen_only_units_are_listen_decisions_without_text() -> None:
    session, _ = _session()
    first = _project(session, _snapshot([LISTEN], "", stop=True))
    second = _project(session, _snapshot([LISTEN, LISTEN], "", stop=True))
    for results in (first, second):
        assert len(results) == 1
        assert results[0]["is_listen"] is True
        assert results[0]["end_of_turn"] is False
        assert not results[0].get("text")


def test_reply_units_emit_text_once_and_close_each_segment() -> None:
    session, _ = _session()
    steps = [
        ([LISTEN], "", True),
        ([LISTEN, SPEAK], "", False),
        ([LISTEN, SPEAK, A], "嗯，", False),
        ([LISTEN, SPEAK, A, B], "嗯，这", False),
        ([LISTEN, SPEAK, A, B, CHUNK_EOS], "嗯，这", True),
        # A repeated snapshot of the same unit end emits nothing.
        ([LISTEN, SPEAK, A, B, CHUNK_EOS], "嗯，这", True),
        ([LISTEN, SPEAK, A, B, CHUNK_EOS, SPEAK, A], "嗯，这种", False),
    ]
    collected: list[list[dict[str, object]]] = []
    for token_ids, text, stop in steps:
        collected.append(_project(session, _snapshot(token_ids, text, stop=stop), active_response_id="resp"))

    assert _texts([r for results in collected for r in results]) == "嗯，这种"
    segment_ends = [r for results in collected for r in results if r.get("segment_complete")]
    assert len(segment_ends) == 1
    assert segment_ends[0]["end_of_turn"] is False
    assert collected[5] == []
    assert not any(r.get("end_of_turn") for results in collected for r in results)


def test_equal_text_in_two_units_is_two_deltas() -> None:
    session, _ = _session()
    first = _project(session, _snapshot([SPEAK, A, CHUNK_EOS], "好", stop=True))
    second = _project(session, _snapshot([SPEAK, A, CHUNK_EOS, SPEAK, A, CHUNK_EOS], "好好", stop=True))
    assert _texts(first) == "好"
    assert _texts(second) == "好"


@pytest.mark.parametrize(
    "unit",
    [
        [SPEAK, A, TURN_EOS, LISTEN],  # text, turn_eos and LISTEN in one unit
        [SPEAK, A, TURN_EOS, CHUNK_EOS],
    ],
)
def test_final_reply_unit_emits_trailing_text_before_ending_the_turn(unit: list[int]) -> None:
    session, _ = _session()
    results = _project(session, _snapshot(unit, "再见", stop=True), active_response_id="resp", turn_id=3)
    assert [bool(r.get("text")) for r in results] == [True, False]
    assert results[0]["text"] == "再见"
    terminal = results[-1]
    assert terminal["end_of_turn"] is True
    assert terminal["is_listen"] is False
    assert terminal["model_turn_id"] == 3


def test_turn_eos_from_an_earlier_unit_does_not_end_a_later_listen() -> None:
    session, _ = _session()
    _project(session, _snapshot([SPEAK, A, TURN_EOS, CHUNK_EOS], "嗯", stop=True))
    results = _project(session, _snapshot([SPEAK, A, TURN_EOS, CHUNK_EOS, LISTEN], "嗯", stop=True))
    assert len(results) == 1
    assert results[0]["is_listen"] is True
    assert results[0]["end_of_turn"] is False


def test_turns_in_one_snapshot_get_consecutive_ids() -> None:
    session, _ = _session()
    tokens = [SPEAK, A, TURN_EOS, LISTEN, SPEAK, B, TURN_EOS, LISTEN]
    results = _project(session, _snapshot(tokens, "一二", stop=True), turn_id=4)
    ends = [r["model_turn_id"] for r in results if r.get("end_of_turn")]
    assert ends == [4, 5]


def test_missing_control_tokens_fail_before_text_and_end_the_request() -> None:
    session, _ = _session()
    first = _project(session, _snapshot([SPEAK, A], "嗯", special=None))
    assert len(first) == 1
    assert first[0]["error_code"] == "runtime_data_plane_missing_control_tokens"
    assert first[0]["close_session"] is True
    assert not first[0].get("text"), "text would open a response nothing can end"
    assert session.is_terminal(REQUEST_ID)
    # Later snapshots of the failed request, even with the ids, are dropped.
    assert _project(session, _snapshot([SPEAK, A, B], "嗯这", special=None)) == []
    assert _project(session, _snapshot([SPEAK, A, B, CHUNK_EOS], "嗯这", stop=True)) == []


def test_control_tokens_from_an_earlier_snapshot_are_reused() -> None:
    session, _ = _session()
    _project(session, _snapshot([LISTEN], "", stop=True))
    results = _project(session, _snapshot([LISTEN, SPEAK, A, CHUNK_EOS], "嗯", stop=True, special=None))
    assert not any(r.get("error_code") for r in results)
    assert any(r.get("segment_complete") for r in results)


@pytest.mark.parametrize("tail", [[SPEAK, A, TURN_EOS], [SPEAK, A, B]])
def test_length_stop_without_a_terminator_fails_instead_of_completing(tail: list[int]) -> None:
    # vLLM checks stop tokens before the length cap, so a capped tail is unterminated.
    session, _ = _session()
    output = _snapshot(tail, "嗯这", stop=True)
    output.outputs[0].finish_reason = "length"
    output.outputs[0].stop_reason = None
    results = _project(session, output, active_response_id="resp")
    assert _texts(results) == "嗯这"
    assert not any(r.get("end_of_turn") or r.get("segment_complete") or r.get("is_listen") for r in results)
    assert [r.get("error_code") for r in results if r.get("error_code")] == ["runtime_data_plane_unterminated_unit"]
    # The failed tail is consumed: a repeated snapshot does not fail again.
    assert _project(session, output, active_response_id="resp") == []


def test_speech_deployment_keeps_the_audio_path() -> None:
    session, encoded = _session()
    _project(session, _snapshot([SPEAK, A], "嗯"), text_output=False)
    assert len(encoded) == 1
