# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A unit that closes its speech with ``<|turn_eos|>`` and then LISTEN is speech, not a LISTEN decision.

On silent input the policy forces the unit terminator after ``<|turn_eos|>`` to
be LISTEN. Treating that unit as a model LISTEN keeps its final speech and
``<|turn_eos|>`` from the Talker, so the response never ends.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.minicpmo_4_5.duplex.data_plane import MiniCPMO45DataPlaneSession
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import MiniCPMO45DuplexPlugin
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import MiniCPMO45DuplexPolicy

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CHUNK_EOS, CHUNK_TTS_EOS, LISTEN, TURN_EOS = 5, 6, 7, 9
SPECIAL_TOKEN_IDS = {
    "chunk_eos_token_id": CHUNK_EOS,
    "chunk_tts_eos_token_id": CHUNK_TTS_EOS,
    "listen_token_id": LISTEN,
    "turn_eos_token_id": TURN_EOS,
}
TEXT = 100


@pytest.mark.parametrize(
    ("token_ids", "closes_speech"),
    [
        ([TEXT, TEXT, TURN_EOS, LISTEN], True),
        ([TURN_EOS, LISTEN], True),
        ([LISTEN, LISTEN, TEXT, TURN_EOS, LISTEN], True),
        ([LISTEN], False),
        ([TEXT, TEXT, LISTEN], False),
        # A <|turn_eos|> from an earlier unit does not count.
        ([TEXT, TURN_EOS, CHUNK_EOS, LISTEN], False),
        ([TEXT, TURN_EOS, LISTEN, LISTEN], False),
        ([TEXT, TURN_EOS, CHUNK_EOS], False),
        ([], False),
    ],
)
def test_speech_unit_closed_by_listen(token_ids: list[int], closes_speech: bool) -> None:
    assert MiniCPMO45DuplexPolicy.speech_unit_closed_by_listen(token_ids, SPECIAL_TOKEN_IDS) is closes_speech


def _stage0_output(token_ids: list[int], *, finished: bool = True) -> SimpleNamespace:
    completion = SimpleNamespace(token_ids=token_ids, stop_reason=token_ids[-1], text="", multimodal_output={})
    return SimpleNamespace(
        request_id="duplex-s.c2Vzc2lvbg==.e.0.r.stage0",
        outputs=[completion],
        finished=finished,
        multimodal_output={"special_token_ids": dict(SPECIAL_TOKEN_IDS)},
    )


def _decide(token_ids: list[int]):
    return MiniCPMO45DuplexPlugin(lambda *args: None).decide_output(
        stage_id=0,
        final_stage_id=2,
        segment_finished=True,
        segment_token_ids=tuple(token_ids),
        segment_output_metadata={},
        output=_stage0_output(token_ids),
    )


def test_plugin_forwards_speech_closed_by_listen_to_the_talker() -> None:
    assert _decide([TEXT, TEXT, TURN_EOS, LISTEN]) is None


def test_plugin_keeps_a_plain_listen_unit_a_listen_decision() -> None:
    decision = _decide([LISTEN])

    assert decision is not None
    assert decision.metadata["duplex_native_decision"] == "listen"


def _projected_model_listens(token_ids: list[int]) -> list[dict[str, object]]:
    session = MiniCPMO45DataPlaneSession(lambda *args, **kwargs: "")
    return [result for result in session.project_output(_stage0_output(token_ids)) if result.get("model_listen")]


def test_data_plane_does_not_project_speech_closed_by_listen_as_model_listen() -> None:
    assert _projected_model_listens([TEXT, TEXT, TURN_EOS, LISTEN]) == []


def test_data_plane_projects_a_plain_listen_unit_as_model_listen() -> None:
    assert len(_projected_model_listens([LISTEN])) == 1
