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
):
    mm_output = dict(multimodal_output or {})
    mm_output["latent"] = latent
    completion = SimpleNamespace(
        token_ids=output_ids if token_list is None else token_list,
        text="hello",
        multimodal_output=mm_output,
    )
    return SimpleNamespace(
        request_id="req-1",
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


def test_plain_chat_handoff_owns_talker_prompt_contract(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_MINICPMO45_TENSOR_HANDOFF", raising=False)
    prompt_ids = [101, 102]
    output_ids = [11, 12]
    latent = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    converted = llm2tts(
        [_output(prompt_ids=prompt_ids, output_ids=output_ids, latent=latent)],
        prompt=[{}],
    )[0]

    info = converted["model_intermediate_buffer"]
    assert info["ids"]["tts"] == output_ids
    assert isinstance(info["hidden_states"]["tts"], torch.Tensor)
    assert torch.equal(info["hidden_states"]["tts"], latent[2:4])
    envelope = info["meta"]["tensor_envelopes"]["hidden_states.tts"]
    assert envelope["version"] == 1
    assert envelope["request_id"] == "req-1"
    assert envelope["shape"] == [2, 4]
    assert envelope["handle"] == {
        "kind": "inline_tensor",
        "payload_path": "hidden_states.tts",
    }
    assert converted["prompt_token_ids"] == [0, 0, 0, 0]
    assert info["meta"]["replace_streaming_prompt"] is True
    assert info["meta"]["next_stage_prompt_len"] == 4


def test_llm2tts_carries_request_ref_audio(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_MINICPMO45_TENSOR_HANDOFF", raising=False)
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
    torch.testing.assert_close(info["codes"]["ref"], ref_waveform)
    assert info["meta"]["ref_audio_sr"] == 22050
    assert info["ids"]["tts"] == [11, 12]


def test_llm2tts_legacy_list_fallback(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_TENSOR_HANDOFF", "0")
    latent = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    ref_waveform = torch.tensor([0.1, 0.2, 0.3])

    converted = llm2tts(
        [_output(prompt_ids=[101, 102], output_ids=[11, 12], latent=latent)],
        prompt=[{"multi_modal_data": {"audio": (ref_waveform, 22050)}}],
    )[0]
    info = converted["model_intermediate_buffer"]

    assert info["hidden_states"]["tts"] == latent[2:4].tolist()
    assert info["codes"]["ref"] == ref_waveform.tolist()
    assert "tensor_envelopes" not in info["meta"]


def test_llm2tts_rejects_invalid_tensor_handoff_switch(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_TENSOR_HANDOFF", "sometimes")

    with pytest.raises(ValueError, match="VLLM_OMNI_MINICPMO45_TENSOR_HANDOFF"):
        llm2tts(
            [
                _output(
                    prompt_ids=[101, 102],
                    output_ids=[11, 12],
                    latent=torch.arange(16, dtype=torch.float32).reshape(4, 4),
                )
            ],
            prompt=[{}],
        )


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
    assert info["meta"]["turn_start"] is True
    assert info["meta"]["segment_end"] is True
    assert info["duplex"]["epoch"] == 3
    assert info["duplex"]["turn_id"] == 7


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
    second_input = llm2tts([second], prompt=[{}], _streaming_context=context)[0]
    context.bridge_states["duplex"]["model_turn_id"] = 8
    third_input = llm2tts([third], prompt=[{}], _streaming_context=context)[0]

    assert first_input["model_intermediate_buffer"]["ids"]["tts"] == [21, 22]
    assert second_input["model_intermediate_buffer"]["ids"]["tts"] == [23, 24]
    assert third_input["model_intermediate_buffer"]["ids"]["tts"] == [25, 26]
    assert first_input["model_intermediate_buffer"]["meta"]["turn_start"] is True
    assert second_input["model_intermediate_buffer"]["meta"]["turn_start"] is False
    assert third_input["model_intermediate_buffer"]["meta"]["turn_start"] is True
    assert first_input["model_intermediate_buffer"]["meta"]["replace_streaming_prompt"] is True
    assert "replace_streaming_prompt" not in second_input["model_intermediate_buffer"]["meta"]
    assert third_input["model_intermediate_buffer"]["meta"]["replace_streaming_prompt"] is True
    assert second_input["model_intermediate_buffer"]["meta"]["next_stage_prompt_len"] == 3
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
