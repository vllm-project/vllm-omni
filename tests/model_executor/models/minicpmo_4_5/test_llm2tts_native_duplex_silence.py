# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Guard MiniCPM-o native duplex listen / no-speech handoffs in llm2tts.

A control-only Thinker unit (listen, empty, terminal, or a listen segment after
speech) with no latent or hidden_states must skip Talker conditioning instead
of raising. A speak unit or an ordinary (non-duplex) request missing hidden
states must still fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import llm2tts

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


_PROMPT_IDS = [10, 11]
_LISTEN_ID = 9303
_SPEAK_ID = 9304
_CHUNK_EOS_ID = 9308
_TURN_EOS_ID = 9310
_NATIVE_DUPLEX_META = {
    "tts_bos_token_id": 9301,
    "tts_eos_token_id": 9302,
    "listen_token_id": _LISTEN_ID,
    "speak_token_id": _SPEAK_ID,
    "chunk_eos_token_id": _CHUNK_EOS_ID,
    "chunk_tts_eos_token_id": 9309,
    "turn_eos_token_id": _TURN_EOS_ID,
}


@dataclass
class _Completion:
    token_ids: list[int]
    multimodal_output: dict[str, object]
    text: str = ""


@dataclass
class _ThinkerOutput:
    request_id: str
    prompt_token_ids: list[int]
    outputs: list[_Completion]


@dataclass
class _StreamingContext:
    bridge_states: dict[str, object] = field(default_factory=dict)


def _thinker_output(
    output_token_ids: list[int],
    *,
    request_id: str = "req-silence",
    native_duplex: bool = True,
) -> _ThinkerOutput:
    multimodal_output: dict[str, object] = {}
    if native_duplex:
        multimodal_output = {
            "duplex_prompt_token_ids": list(_PROMPT_IDS),
            "meta": dict(_NATIVE_DUPLEX_META),
        }
    return _ThinkerOutput(
        request_id=request_id,
        prompt_token_ids=list(_PROMPT_IDS),
        outputs=[
            _Completion(
                token_ids=output_token_ids,
                multimodal_output=multimodal_output,
            )
        ],
    )


def _native_duplex_thinker_output(
    output_token_ids: list[int],
    *,
    request_id: str = "req-silence",
) -> _ThinkerOutput:
    return _thinker_output(output_token_ids, request_id=request_id, native_duplex=True)


@pytest.mark.parametrize(
    "output_token_ids",
    [
        pytest.param([_LISTEN_ID], id="listen"),
        pytest.param([_CHUNK_EOS_ID], id="chunk_eos"),
        pytest.param([_TURN_EOS_ID], id="turn_eos"),
        pytest.param([_LISTEN_ID, _CHUNK_EOS_ID], id="listen_then_chunk_eos"),
    ],
)
def test_native_duplex_control_only_without_hidden_states_skips_talker(
    output_token_ids: list[int],
) -> None:
    """Non-empty control/terminal units with no latent/hidden_states must skip.

    Empty output hits the helper's ``if not output_ids`` branch. These cases
    require each token to be recognized as control-only (listen, chunk_eos,
    turn_eos, or listen followed by an end marker).
    """
    assert llm2tts([_native_duplex_thinker_output(output_token_ids)], prompt=None) == []


def test_native_duplex_empty_unit_without_hidden_states_skips_talker() -> None:
    """An empty native-duplex output with no speech payload must also skip.

    Control/terminal units can arrive with no generated ids and no hidden
    states. Same skip contract as listen-only: do not wake the Talker.
    """
    assert llm2tts([_native_duplex_thinker_output([])], prompt=None) == []


def test_native_duplex_listen_after_speak_without_hidden_states_skips_talker() -> None:
    """After a spoken segment, a listen-only delta without hidden states skips.

    Cumulative ids still include the earlier speak tokens, but the streaming
    cursor has already handed those off. Only the new listen token is in this
    segment, so the adapter must skip rather than require hidden states for
    the whole cumulative sequence.
    """
    spoken = [_SPEAK_ID, 21, 22, _CHUNK_EOS_ID]
    context = _StreamingContext(
        bridge_states={
            "duplex": {"epoch": 1, "model_turn_id": 1},
            "minicpmo45_tts_handoff": {
                "request_id": "req-silence",
                "sent_output_len": len(spoken),
                "sent_output_ids": list(spoken),
                "turn_id": 1,
                "condition_seq": 0,
            },
        }
    )

    assert (
        llm2tts(
            [_native_duplex_thinker_output([*spoken, _LISTEN_ID])],
            prompt=None,
            _streaming_context=context,
        )
        == []
    )


def test_native_duplex_speak_without_hidden_states_still_raises() -> None:
    """A speak unit still fails closed when speech conditioning is missing.

    Skip applies only to confirmed control-only/no-speech handoffs. If the
    Thinker emitted speak/content tokens, missing latent/hidden_states is a
    malformed speech handoff and must keep raising.
    """
    with pytest.raises(ValueError, match="No latent or hidden_states"):
        llm2tts([_native_duplex_thinker_output([_SPEAK_ID, 21])], prompt=None)


def test_ordinary_empty_output_without_hidden_states_still_raises() -> None:
    """Skip stays native-duplex-only: ordinary empty output still raises.

    An ordinary request has no native-duplex marker. Empty token ids and no
    latent/hidden_states must not be swallowed as a listen/control handoff.
    The adjacent missing-hidden test uses a content token, so it does not
    cover this boundary.
    """
    with pytest.raises(ValueError, match="No latent or hidden_states"):
        llm2tts(
            [_thinker_output([], request_id="req-ordinary", native_duplex=False)],
            prompt=None,
        )
