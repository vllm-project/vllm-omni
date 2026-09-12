# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import (
    _extract_first_audio_ref,
    llm2tts,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _output(
    *,
    prompt_ids: list[int],
    output_ids: list[int],
    latent: torch.Tensor,
    multimodal_output: dict | None = None,
    token_list: list[int] | None = None,
    request_id: str = "req-1",
):
    mm_output = dict(multimodal_output or {})
    mm_output["latent"] = latent
    completion = SimpleNamespace(
        token_ids=output_ids if token_list is None else token_list,
        text="hello",
        multimodal_output=mm_output,
    )
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=prompt_ids,
        outputs=[completion],
    )


def test_extract_first_audio_ref_accepts_dict_stereo_audio() -> None:
    ref = _extract_first_audio_ref(
        {
            "audio": {
                "array": [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]],
                "sampling_rate": 16000,
            }
        }
    )

    assert ref is not None
    waveform, sample_rate = ref
    assert sample_rate == 16000
    assert torch.allclose(waveform, torch.tensor([1.5, 3.5, 5.5]))


def test_plain_chat_handoff_owns_talker_prompt_contract() -> None:
    prompt_ids = [101, 102]
    output_ids = [11, 12]
    latent = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    converted = llm2tts(
        [_output(prompt_ids=prompt_ids, output_ids=output_ids, latent=latent)],
        prompt=[{}],
    )[0]

    info = converted["model_intermediate_buffer"]
    assert info["ids"]["tts"] == output_ids
    assert torch.equal(torch.tensor(info["hidden_states"]["tts"]), latent[2:4])
    assert converted["prompt_token_ids"] == [0, 0, 0, 0]
    assert info["meta"]["replace_streaming_prompt"] is True
    assert info["meta"]["next_stage_prompt_len"] == 4


def test_llm2tts_carries_request_ref_audio() -> None:
    latent = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    source = _output(
        prompt_ids=[101, 9001],
        output_ids=[11, 12, 9002],
        latent=latent,
        multimodal_output={
            "meta": {
                "tts_bos_token_id": 9001,
                "tts_eos_token_id": 9002,
            }
        },
    )
    ref_waveform = torch.tensor([0.1, 0.2, 0.3])

    converted = llm2tts(
        [source],
        prompt=[{"multi_modal_data": {"audio": (ref_waveform, 22050)}}],
    )[0]

    info = converted["model_intermediate_buffer"]
    assert info["codes"]["ref"] == ref_waveform.tolist()
    assert info["meta"]["ref_audio_sr"] == 22050
    assert info["ids"]["tts"] == [11, 12]


def test_native_duplex_speak_segment_reaches_split_talker() -> None:
    prompt_ids = [101, 102]
    output_ids = [9304, 21, 22, 9308]
    latent = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    source = _output(
        prompt_ids=prompt_ids,
        output_ids=output_ids,
        latent=latent,
        multimodal_output={
            "duplex_prompt_token_ids": prompt_ids,
            "meta": {
                "tts_bos_token_id": 9301,
                "tts_eos_token_id": 9302,
                "listen_token_id": 9303,
                "speak_token_id": 9304,
                "chunk_eos_token_id": 9308,
                "chunk_tts_eos_token_id": 9309,
                "turn_eos_token_id": 9310,
            },
        },
    )
    context = SimpleNamespace(
        bridge_states={
            "duplex": {
                "epoch": 3,
                "model_turn_id": 7,
            }
        }
    )

    converted = llm2tts([source], prompt=[{}], _streaming_context=context)[0]

    info = converted["model_intermediate_buffer"]
    assert info["native_duplex"] is True
    assert info["ids"]["tts"] == [21, 22]
    assert converted["prompt_token_ids"] == [0, 0, 0]
    assert info["meta"]["replace_streaming_prompt"] is True
    assert info["meta"]["next_stage_prompt_len"] == 3
    assert info["meta"]["next_stage_generation_tokens"] == 26
    assert info["meta"]["turn_start"] is True
    assert info["meta"]["segment_end"] is True
    assert info["duplex"]["epoch"] == 3
    assert info["duplex"]["turn_id"] == 7


def test_request_level_transport_metadata_does_not_shadow_completion_handoff() -> None:
    prompt_ids = [101, 102]
    source = _output(
        prompt_ids=prompt_ids,
        output_ids=[9304, 21, 22, 9308],
        latent=torch.arange(24, dtype=torch.float32).reshape(6, 4),
        multimodal_output={
            "duplex_prompt_token_ids": prompt_ids,
            "meta": {
                "tts_bos_token_id": 9301,
                "tts_eos_token_id": 9302,
                "listen_token_id": 9303,
                "speak_token_id": 9304,
                "chunk_eos_token_id": 9308,
                "chunk_tts_eos_token_id": 9309,
                "turn_eos_token_id": 9310,
            },
        },
    )
    source.multimodal_output = {
        "duplex_prompt_len": len(prompt_ids),
        "duplex_token_offset": 0,
    }
    context = SimpleNamespace(
        bridge_states={"duplex": {"epoch": 3, "model_turn_id": 7}},
    )

    converted = llm2tts([source], prompt=[{}], _streaming_context=context)

    assert converted[0]["model_intermediate_buffer"]["ids"]["tts"] == [21, 22]
    assert converted[0]["model_intermediate_buffer"]["duplex"]["epoch"] == 3


def test_native_duplex_text_only_unit_does_not_create_talker_request() -> None:
    source = _output(
        prompt_ids=[101, 102],
        output_ids=[],
        latent=torch.zeros((2, 4)),
    )
    source.multimodal_output = {
        "duplex_prompt_len": 2,
        "duplex_token_offset": 0,
    }
    context = SimpleNamespace(
        bridge_states={"duplex": {"epoch": 3, "model_turn_id": 7}},
    )

    assert llm2tts([source], prompt=[{}], _streaming_context=context) == []


def test_native_duplex_raw_turn_end_becomes_control_talker_request() -> None:
    prompt_ids = [101, 102]
    turn_eos_id = 9310
    source = _output(
        prompt_ids=prompt_ids,
        output_ids=[],
        latent=torch.zeros((2, 4)),
    )
    raw_metadata = {
        "duplex_prompt_token_ids": prompt_ids,
        "meta": {
            "tts_bos_token_id": 9301,
            "tts_eos_token_id": 9302,
            "listen_token_id": 9303,
            "speak_token_id": 9304,
            "chunk_eos_token_id": 9308,
            "chunk_tts_eos_token_id": 9309,
            "turn_eos_token_id": turn_eos_id,
        },
    }
    context = SimpleNamespace(
        bridge_states={
            "duplex": {
                "session_id": "session-1",
                "incarnation": 2,
                "epoch": 3,
                "model_turn_id": 7,
            }
        },
        segment=lambda stage_id: SimpleNamespace(
            token_ids=[turn_eos_id] if stage_id == 0 else [],
            output_metadata=raw_metadata if stage_id == 0 else {},
        ),
    )

    converted = llm2tts([source], prompt=[{}], _streaming_context=context)

    assert len(converted) == 1
    assert converted[0]["prompt_token_ids"] == [0]
    info = converted[0]["model_intermediate_buffer"]
    assert info["native_duplex"] is True
    assert info["ids"]["tts"] == []
    assert info["hidden_states"]["tts"] == []
    assert info["duplex"]["epoch"] == 3
    assert info["duplex"]["turn_id"] == 7
    assert info["meta"]["turn_end"] is True
    assert info["meta"]["turn_eos_token_id"] == turn_eos_id
    assert info["meta"]["replace_streaming_prompt"] is True
    assert info["meta"]["next_stage_prompt_len"] == 1
    assert info["meta"]["next_stage_generation_tokens"] == 26
    assert context.bridge_states["duplex"]["model_turn_id"] == 8


def test_native_duplex_raw_non_terminal_control_unit_is_not_forwarded() -> None:
    prompt_ids = [101, 102]
    listen_id = 9303
    source = _output(
        prompt_ids=prompt_ids,
        output_ids=[],
        latent=torch.zeros((2, 4)),
    )
    context = SimpleNamespace(
        bridge_states={"duplex": {"epoch": 3, "model_turn_id": 7}},
        segment=lambda _stage_id: SimpleNamespace(
            token_ids=[listen_id],
            output_metadata={
                "duplex_prompt_token_ids": prompt_ids,
                "meta": {
                    "tts_bos_token_id": 9301,
                    "tts_eos_token_id": 9302,
                    "listen_token_id": listen_id,
                    "speak_token_id": 9304,
                    "chunk_eos_token_id": 9308,
                    "chunk_tts_eos_token_id": 9309,
                    "turn_eos_token_id": 9310,
                },
            },
        ),
    )

    assert llm2tts([source], prompt=[{}], _streaming_context=context) == []
    assert context.bridge_states["duplex"]["model_turn_id"] == 7


def test_native_duplex_final_append_empty_chunk_becomes_turn_end_control() -> None:
    prompt_ids = [101, 102]
    chunk_eos_id = 9308
    turn_eos_id = 9310
    source = _output(
        prompt_ids=prompt_ids,
        output_ids=[],
        latent=torch.zeros((2, 4)),
    )
    raw_metadata = {
        "duplex_prompt_token_ids": prompt_ids,
        "meta": {
            "tts_bos_token_id": 9301,
            "tts_eos_token_id": 9302,
            "listen_token_id": 9303,
            "speak_token_id": 9304,
            "chunk_eos_token_id": chunk_eos_id,
            "chunk_tts_eos_token_id": 9309,
            "turn_eos_token_id": turn_eos_id,
        },
    }
    context = SimpleNamespace(
        bridge_states={
            "duplex": {
                "session_id": "session-1",
                "incarnation": 2,
                "epoch": 3,
                "model_turn_id": 7,
            }
        },
        segment=lambda stage_id: SimpleNamespace(
            token_ids=[chunk_eos_id] if stage_id == 0 else [],
            output_metadata=raw_metadata if stage_id == 0 else {},
            input_metadata=(
                {
                    "duplex": {
                        "data_plane": True,
                        "session_id": "session-1",
                        "incarnation": 2,
                        "epoch": 3,
                        "seq": 3,
                        "turn_id": 7,
                        "final": True,
                    }
                }
                if stage_id == 0
                else {}
            ),
        ),
    )

    converted = llm2tts([source], prompt=[{}], _streaming_context=context)

    assert len(converted) == 1
    assert converted[0]["prompt_token_ids"] == [0]
    info = converted[0]["model_intermediate_buffer"]
    assert info["ids"]["tts"] == []
    assert info["hidden_states"]["tts"] == []
    assert info["meta"]["turn_end"] is True
    assert info["meta"]["replace_streaming_prompt"] is True
    assert info["meta"]["next_stage_prompt_len"] == 1
    assert info["duplex"]["turn_id"] == 7
    assert context.bridge_states["duplex"]["model_turn_id"] == 8


def test_native_duplex_final_speech_then_silence_boundary_consumes_turn_fence() -> None:
    prompt_ids = [101, 102]
    speak_id = 9304
    chunk_eos_id = 9308
    turn_eos_id = 9310
    token_metadata = {
        "tts_bos_token_id": 9301,
        "tts_eos_token_id": 9302,
        "listen_token_id": 9303,
        "speak_token_id": speak_id,
        "chunk_eos_token_id": chunk_eos_id,
        "chunk_tts_eos_token_id": 9309,
        "turn_eos_token_id": turn_eos_id,
    }
    segment = SimpleNamespace(
        token_ids=[chunk_eos_id],
        output_metadata={
            "duplex_prompt_token_ids": prompt_ids,
            "meta": token_metadata,
        },
        input_metadata={
            "duplex": {
                "data_plane": True,
                "session_id": "session-1",
                "incarnation": 2,
                "epoch": 3,
                "seq": 2,
                "turn_id": 7,
                "final": True,
            }
        },
    )
    context = SimpleNamespace(
        bridge_states={
            "duplex": {
                "session_id": "session-1",
                "incarnation": 2,
                "epoch": 3,
                "model_turn_id": 7,
            }
        },
        segment=lambda _stage_id: segment,
    )
    speech = _output(
        prompt_ids=prompt_ids,
        output_ids=[speak_id, 21, 22, chunk_eos_id],
        latent=torch.arange(24, dtype=torch.float32).reshape(6, 4),
        multimodal_output={
            "duplex_prompt_token_ids": prompt_ids,
            "meta": token_metadata,
        },
    )

    speech_input = llm2tts([speech], prompt=[{}], _streaming_context=context)

    assert speech_input[0]["model_intermediate_buffer"]["ids"]["tts"] == [21, 22]
    assert context.bridge_states["minicpmo45_turn_end_fence"]["seq"] == 2
    assert context.bridge_states["duplex"]["model_turn_id"] == 7

    segment.input_metadata = {
        "duplex": {
            "data_plane": True,
            "session_id": "session-1",
            "incarnation": 2,
            "epoch": 3,
            "seq": 3,
            "turn_id": 7,
            "final": False,
        }
    }
    control = _output(
        prompt_ids=prompt_ids,
        output_ids=[],
        latent=torch.zeros((2, 4)),
        multimodal_output={
            "duplex_prompt_token_ids": prompt_ids,
            "meta": token_metadata,
        },
    )

    terminal_input = llm2tts([control], prompt=[{}], _streaming_context=context)

    assert len(terminal_input) == 1
    terminal_info = terminal_input[0]["model_intermediate_buffer"]
    assert terminal_info["ids"]["tts"] == []
    assert terminal_info["meta"]["turn_end"] is True
    assert context.bridge_states["minicpmo45_turn_end_fence"] == {}
    assert context.bridge_states["duplex"]["model_turn_id"] == 8


def test_native_duplex_old_turn_fence_does_not_end_new_epoch() -> None:
    prompt_ids = [101, 102]
    chunk_eos_id = 9308
    context = SimpleNamespace(
        bridge_states={
            "duplex": {
                "session_id": "session-1",
                "incarnation": 2,
                "epoch": 4,
                "model_turn_id": 7,
            },
            "minicpmo45_turn_end_fence": {
                "identity": ("session-1", 2, 3, 7),
                "seq": 2,
            },
        },
        segment=lambda _stage_id: SimpleNamespace(
            token_ids=[chunk_eos_id],
            output_metadata={
                "duplex_prompt_token_ids": prompt_ids,
                "meta": {
                    "tts_bos_token_id": 9301,
                    "tts_eos_token_id": 9302,
                    "listen_token_id": 9303,
                    "speak_token_id": 9304,
                    "chunk_eos_token_id": chunk_eos_id,
                    "chunk_tts_eos_token_id": 9309,
                    "turn_eos_token_id": 9310,
                },
            },
            input_metadata={
                "duplex": {
                    "data_plane": True,
                    "session_id": "session-1",
                    "incarnation": 2,
                    "epoch": 4,
                    "seq": 1,
                    "turn_id": 7,
                    "final": False,
                }
            },
        ),
    )
    source = _output(
        prompt_ids=prompt_ids,
        output_ids=[],
        latent=torch.zeros((2, 4)),
    )

    assert llm2tts([source], prompt=[{}], _streaming_context=context) == []
    assert context.bridge_states["minicpmo45_turn_end_fence"] == {}
    assert context.bridge_states["duplex"]["model_turn_id"] == 7


def test_native_duplex_nonfinal_empty_chunk_is_not_turn_end_control() -> None:
    prompt_ids = [101, 102]
    chunk_eos_id = 9308
    source = _output(
        prompt_ids=prompt_ids,
        output_ids=[],
        latent=torch.zeros((2, 4)),
    )
    context = SimpleNamespace(
        bridge_states={
            "duplex": {
                "session_id": "session-1",
                "incarnation": 2,
                "epoch": 3,
                "model_turn_id": 7,
            }
        },
        segment=lambda _stage_id: SimpleNamespace(
            token_ids=[chunk_eos_id],
            output_metadata={
                "duplex_prompt_token_ids": prompt_ids,
                "meta": {
                    "tts_bos_token_id": 9301,
                    "tts_eos_token_id": 9302,
                    "listen_token_id": 9303,
                    "speak_token_id": 9304,
                    "chunk_eos_token_id": chunk_eos_id,
                    "chunk_tts_eos_token_id": 9309,
                    "turn_eos_token_id": 9310,
                },
            },
            input_metadata={
                "duplex": {
                    "data_plane": True,
                    "session_id": "session-1",
                    "incarnation": 2,
                    "epoch": 3,
                    "seq": 2,
                    "turn_id": 7,
                    "final": False,
                }
            },
        ),
    )

    assert llm2tts([source], prompt=[{}], _streaming_context=context) == []
    assert context.bridge_states["duplex"]["model_turn_id"] == 7


def test_native_duplex_continuation_appends_only_new_talker_condition() -> None:
    prompt_ids = [101, 102]
    token_ids = {
        "tts_bos_token_id": 9301,
        "tts_eos_token_id": 9302,
        "listen_token_id": 9303,
        "speak_token_id": 9304,
        "chunk_eos_token_id": 9308,
        "chunk_tts_eos_token_id": 9309,
        "turn_eos_token_id": 9310,
    }
    context = SimpleNamespace(
        bridge_states={
            "duplex": {
                "epoch": 3,
                "model_turn_id": 7,
            }
        }
    )

    first_ids = [9304, 21, 22, 9308]
    first = _output(
        prompt_ids=prompt_ids,
        output_ids=first_ids,
        latent=torch.arange(24, dtype=torch.float32).reshape(6, 4),
        multimodal_output={
            "duplex_prompt_token_ids": prompt_ids,
            "meta": token_ids,
        },
    )
    second_ids = [*first_ids, 9304, 23, 24, 9308]
    second = _output(
        prompt_ids=prompt_ids,
        output_ids=second_ids,
        latent=torch.arange(40, dtype=torch.float32).reshape(10, 4),
        multimodal_output={
            "duplex_prompt_token_ids": prompt_ids,
            "meta": token_ids,
        },
    )
    third_ids = [*second_ids, 9304, 25, 26, 9308]
    third = _output(
        prompt_ids=prompt_ids,
        output_ids=third_ids,
        latent=torch.arange(56, dtype=torch.float32).reshape(14, 4),
        multimodal_output={
            "duplex_prompt_token_ids": prompt_ids,
            "meta": token_ids,
        },
    )

    first_input = llm2tts([first], prompt=[{}], _streaming_context=context)[0]
    replayed_inputs = llm2tts([first], prompt=[{}], _streaming_context=context)
    second_input = llm2tts([second], prompt=[{}], _streaming_context=context)[0]
    context.bridge_states["duplex"]["model_turn_id"] = 8
    third_input = llm2tts([third], prompt=[{}], _streaming_context=context)[0]
    restarted_input = llm2tts(
        [
            _output(
                prompt_ids=prompt_ids,
                output_ids=first_ids,
                latent=torch.arange(24, dtype=torch.float32).reshape(6, 4),
                multimodal_output={
                    "duplex_prompt_token_ids": prompt_ids,
                    "meta": token_ids,
                },
                request_id="req-2",
            )
        ],
        prompt=[{}],
        _streaming_context=context,
    )[0]

    assert first_input["model_intermediate_buffer"]["ids"]["tts"] == [21, 22]
    assert replayed_inputs == []
    assert second_input["model_intermediate_buffer"]["ids"]["tts"] == [23, 24]
    assert third_input["model_intermediate_buffer"]["ids"]["tts"] == [25, 26]
    assert first_input["model_intermediate_buffer"]["meta"]["turn_start"] is True
    assert second_input["model_intermediate_buffer"]["meta"]["turn_start"] is False
    assert third_input["model_intermediate_buffer"]["meta"]["turn_start"] is True
    assert first_input["model_intermediate_buffer"]["meta"]["streaming_condition_seq"] == 0
    assert second_input["model_intermediate_buffer"]["meta"]["streaming_condition_seq"] == 1
    # A new turn keeps the same stage request, so its condition sequence stays
    # monotonic; a new request/incarnation starts again from zero.
    assert third_input["model_intermediate_buffer"]["meta"]["streaming_condition_seq"] == 2
    assert restarted_input["model_intermediate_buffer"]["meta"]["streaming_condition_seq"] == 0
    assert first_input["model_intermediate_buffer"]["meta"]["replace_streaming_prompt"] is True
    assert second_input["model_intermediate_buffer"]["meta"]["replace_streaming_prompt"] is False
    assert third_input["model_intermediate_buffer"]["meta"]["replace_streaming_prompt"] is True
    assert second_input["model_intermediate_buffer"]["meta"]["next_stage_prompt_len"] == 3
    assert first_input["model_intermediate_buffer"]["meta"]["next_stage_generation_tokens"] == 26
    assert second_input["model_intermediate_buffer"]["meta"]["next_stage_generation_tokens"] == 26
    assert third_input["model_intermediate_buffer"]["meta"]["next_stage_generation_tokens"] == 26
    assert second_input["prompt_token_ids"] == [0, 0, 0]


def test_native_duplex_transcript_decodes_the_talker_condition_slice() -> None:
    prompt_ids = [101, 102]
    metadata = {
        "tts_bos_token_id": 9301,
        "tts_eos_token_id": 9302,
        "listen_token_id": 9303,
        "speak_token_id": 9304,
        "chunk_eos_token_id": 9308,
        "chunk_tts_eos_token_id": 9309,
        "turn_eos_token_id": 9310,
    }
    token_text = {21: "杭州", 22: "和", 23: "州和", 24: "上海之间", 25: "大"}

    def output(token_ids: list[int]):
        return _output(
            prompt_ids=prompt_ids,
            output_ids=token_ids,
            latent=torch.zeros((len(prompt_ids) + len(token_ids), 1)),
            multimodal_output={
                "duplex_prompt_token_ids": prompt_ids,
                "meta": metadata,
            },
        )

    context = SimpleNamespace(
        bridge_states={"duplex": {"epoch": 3, "model_turn_id": 7}},
        source_token_decoder=lambda ids, **_: "".join(token_text.get(int(token_id), "") for token_id in ids),
    )
    first_ids = [9304, 21, 22, 9308]
    first_info = llm2tts([output(first_ids)], prompt=[{}], _streaming_context=context)[0]["model_intermediate_buffer"]
    second_info = llm2tts(
        [output([*first_ids, 23, 24, 25, 9308])],
        prompt=[{}],
        _streaming_context=context,
    )[0]["model_intermediate_buffer"]

    assert first_info["meta"]["native_duplex_segment_text"] == "杭州和"
    assert second_info["ids"]["tts"] == [24, 25]
    assert second_info["meta"]["native_duplex_segment_text"] == "上海之间大"


def test_native_duplex_requires_tokenizer_boundary_metadata() -> None:
    latent = torch.zeros((3, 4))
    source = _output(
        prompt_ids=[101],
        output_ids=[21, 22],
        latent=latent,
        multimodal_output={"duplex_prompt_token_ids": [101]},
    )

    with pytest.raises(ValueError, match="tokenizer-derived.*metadata"):
        llm2tts([source], prompt=[{}], _streaming_context=SimpleNamespace(bridge_states={}))


def test_llm2tts_does_not_alias_live_thinker_token_list() -> None:
    live_tokens = [11, 12]
    latent = torch.zeros((3, 4))
    source = _output(
        prompt_ids=[101],
        output_ids=list(live_tokens),
        latent=latent,
        token_list=live_tokens,
    )

    converted = llm2tts([source], prompt=[{}])[0]
    live_tokens.append(13)

    assert converted["model_intermediate_buffer"]["ids"]["output"] == [11, 12]


@pytest.mark.parametrize("segment_end", [False, True])
def test_native_handoff_clears_previous_turn_end_on_resumable_talker(segment_end: bool) -> None:
    from tests.model_executor.models.minicpmo_4_5.test_talker_batching import _make_talker
    from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

    output_ids = [9304, 21, 22] + ([9308] if segment_end else [])
    source = _output(
        prompt_ids=[101, 102],
        output_ids=output_ids,
        latent=torch.arange((2 + len(output_ids)) * 4, dtype=torch.float32).reshape(-1, 4),
        multimodal_output={
            "duplex_prompt_token_ids": [101, 102],
            "meta": {
                "tts_bos_token_id": 9301,
                "tts_eos_token_id": 9302,
                "listen_token_id": 9303,
                "speak_token_id": 9304,
                "chunk_eos_token_id": 9308,
                "chunk_tts_eos_token_id": 9309,
                "turn_eos_token_id": 9310,
            },
        },
    )
    context = SimpleNamespace(bridge_states={"duplex": {"epoch": 0, "model_turn_id": 1}})
    current = llm2tts([source], prompt=[{}], _streaming_context=context)[0]["model_intermediate_buffer"]
    runner = SimpleNamespace(
        model=SimpleNamespace(),
        model_intermediate_buffer={
            "req-1": {
                "request_id": "req-1",
                "codes": {"audio": torch.empty(0)},
                "meta": {"turn_end": True, "segment_end": True},
            }
        },
        requests={"req-1": SimpleNamespace()},
    )
    OmniGPUModelRunner._update_streaming_input_additional_info(
        runner, SimpleNamespace(model_intermediate_buffer=current), "req-1"
    )
    merged = runner.model_intermediate_buffer["req-1"]
    assert merged["meta"]["segment_end"] is segment_end
    talker = _make_talker()
    output = talker.make_omni_output(torch.ones(1, 2), model_intermediate_buffer=[merged], request_token_spans=[(0, 1)])
    assert output.multimodal_outputs["meta"]["turn_end"][0].item() is False
    assert output.multimodal_outputs["meta"]["duplex_turn_id"][0].item() == 1


def test_native_duplex_output_cursor_reset_keeps_talker_turn():
    from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import (
        _native_duplex_segment_output_ids,
    )

    context = SimpleNamespace(bridge_states={"duplex": {"model_turn_id": 7}})
    first = _native_duplex_segment_output_ids([10, 11, 12], "first", context, request_id="r")
    assert first[2] is True
    # The first actual handoff has reached the Talker. The next native unit
    # folds these output tokens into the prompt, restarting the output cursor.
    context.bridge_states["minicpmo45_tts_handoff"]["condition_seq"] = 0
    second = _native_duplex_segment_output_ids([20, 21, 22], "second", context, request_id="r")
    assert second[0] == [20, 21, 22]
    assert second[2] is False
    context.bridge_states["duplex"]["model_turn_id"] = 8
    assert _native_duplex_segment_output_ids([30, 31], "next turn", context, request_id="r")[2] is True
    assert _native_duplex_segment_output_ids([40], "new request", context, request_id="r2")[2] is True


def test_gander_terminal_condition_preserves_talker_context():
    special = {
        "tts_bos_token_id": 9301,
        "tts_eos_token_id": 9302,
        "listen_token_id": 9303,
        "speak_token_id": 9304,
        "chunk_eos_token_id": 9308,
        "chunk_tts_eos_token_id": 9309,
        "turn_eos_token_id": 9310,
        "gander_speech_tokens": 50,
    }
    context = SimpleNamespace(bridge_states={"duplex": {"epoch": 0, "model_turn_id": 1}})

    def handoff(ids):
        return llm2tts(
            [
                _output(
                    prompt_ids=[101, 102],
                    output_ids=ids,
                    latent=torch.arange((2 + len(ids)) * 4, dtype=torch.float32).reshape(-1, 4),
                    multimodal_output={"duplex_prompt_token_ids": [101, 102], "meta": special},
                )
            ],
            prompt=[{}],
            _streaming_context=context,
        )[0]["model_intermediate_buffer"]["meta"]

    assert handoff([9304, 21, 22, 9308])["replace_streaming_prompt"] is True
    terminal = handoff([9304, 23, 9310, 9309])
    assert terminal["turn_end"] is True
    assert terminal["turn_start"] is False
    assert terminal["replace_streaming_prompt"] is False


def test_identical_tokens_in_distinct_native_units_are_not_replay():
    from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import _native_duplex_segment_output_ids

    segment = SimpleNamespace(input_metadata={"duplex": {"epoch": 0, "seq": 1}})
    context = SimpleNamespace(bridge_states={"duplex": {"model_turn_id": 7}}, segment=lambda _: segment)
    assert _native_duplex_segment_output_ids([10, 11], "same", context, request_id="r")[0] == [10, 11]
    context.bridge_states["minicpmo45_tts_handoff"]["condition_seq"] = 0
    assert _native_duplex_segment_output_ids([10, 11], "same", context, request_id="r")[0] == []
    segment.input_metadata["duplex"]["seq"] = 2
    result = _native_duplex_segment_output_ids([10, 11], "same", context, request_id="r")
    assert result[0] == [10, 11]
    assert result[2] is False
