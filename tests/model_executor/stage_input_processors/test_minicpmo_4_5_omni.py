# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.engine.orchestrator import StreamingInputState
from vllm_omni.model_executor.models.minicpmo_4_5.handoff import unpack_handoff_tensor
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
    if "duplex_prompt_token_ids" in mm_output and "latent_input_ids" not in mm_output:
        forwarded_ids = [*prompt_ids, *output_ids]
        assert len(forwarded_ids) == latent.shape[0]
        mm_output["latent_input_ids"] = torch.tensor(forwarded_ids).reshape(-1, 1)
        mm_output["latent_positions"] = torch.arange(len(forwarded_ids)).reshape(-1, 1)
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
    assert torch.equal(torch.as_tensor(unpack_handoff_tensor(info["hidden_states"]["tts"])), latent[2:4])
    assert converted["prompt_token_ids"] == [0, 0, 0, 0]
    assert info["meta"]["replace_streaming_prompt"] is True
    assert info["meta"]["next_stage_prompt_len"] == 4


def _ledger_output(prompt_ids, output_ids, forwarded):
    """Thinker output whose latent rows follow ``forwarded``: (token, position) per row."""
    latent = torch.stack([torch.full((4,), float(position * 100 + token)) for token, position in forwarded])
    multimodal_output = {
        "latent_input_ids": torch.tensor([token for token, _ in forwarded]).reshape(-1, 1),
        "latent_positions": torch.tensor([position for _, position in forwarded]).reshape(-1, 1),
    }
    return _output(prompt_ids=prompt_ids, output_ids=output_ids, latent=latent, multimodal_output=multimodal_output)


def _row(token: int, position: int) -> torch.Tensor:
    return torch.full((4,), float(position * 100 + token))


def test_plain_chat_handoff_reads_rows_from_ledger_when_steps_repeat() -> None:
    """MRv2 full-payload outputs report every step's rows twice."""
    prompt_ids, output_ids = [101, 102], [11, 12, 13]
    steps = [[(101, 0), (102, 1)], [(11, 2)], [(12, 3)]]
    forwarded = [row for step in steps for row in step * 2]

    info = llm2tts([_ledger_output(prompt_ids, output_ids, forwarded)], prompt=[{}])[0]["model_intermediate_buffer"]

    # The last sampled token is never forwarded, so it has no row.
    assert info["ids"]["tts"] == [11, 12]
    hidden = torch.as_tensor(unpack_handoff_tensor(info["hidden_states"]["tts"]))
    torch.testing.assert_close(hidden, torch.stack([_row(11, 2), _row(12, 3)]))


@pytest.mark.parametrize("repeat", [1, 2], ids=["v1", "mrv2_both_channels"])
def test_plain_chat_handoff_after_a_prefix_hit_has_no_rows_for_the_cached_prompt(repeat) -> None:
    """Native KV reuse forwards only the uncached tail, fewer rows than the reply's position."""
    prompt_ids, output_ids = list(range(100, 116)), [11, 12, 13]
    # Tokens 0-14 hit; the last prompt token and the reply are forwarded.
    steps = [[(115, 15)], [(11, 16)], [(12, 17)]]
    forwarded = [row for step in steps for row in step * repeat]

    info = llm2tts([_ledger_output(prompt_ids, output_ids, forwarded)], prompt=[{}])[0]["model_intermediate_buffer"]

    assert info["ids"]["tts"] == [11, 12]
    hidden = torch.as_tensor(unpack_handoff_tensor(info["hidden_states"]["tts"]))
    torch.testing.assert_close(hidden, torch.stack([_row(11, 16), _row(12, 17)]))


def test_plain_chat_handoff_without_reply_rows_is_an_empty_condition() -> None:
    """A Thinker that stopped at its first token has no reply rows; the Talker finishes such a request."""
    prompt_ids, output_ids = [101, 102], [11]

    info = llm2tts([_ledger_output(prompt_ids, output_ids, [(101, 0), (102, 1)])], prompt=[{}])[0][
        "model_intermediate_buffer"
    ]

    assert info["ids"]["tts"] == []
    assert torch.as_tensor(unpack_handoff_tensor(info["hidden_states"]["tts"])).numel() == 0


def test_plain_chat_handoff_after_a_preemption_keeps_reply_rows_from_before_it() -> None:
    """A resumed request hits on its earlier reply tokens, so their rows come from before the preemption."""
    prompt_ids, output_ids = list(range(100, 116)), [11, 12, 13, 14]
    # Forwarded up to token 13, preempted, then resumed from a hit that ends after token 11.
    forwarded = [(115, 15), (11, 16), (12, 17), (13, 18), (12, 17), (13, 18)]

    info = llm2tts([_ledger_output(prompt_ids, output_ids, forwarded)], prompt=[{}])[0]["model_intermediate_buffer"]

    assert info["ids"]["tts"] == [11, 12, 13]
    hidden = torch.as_tensor(unpack_handoff_tensor(info["hidden_states"]["tts"]))
    torch.testing.assert_close(hidden, torch.stack([_row(11, 16), _row(12, 17), _row(13, 18)]))


def test_plain_chat_handoff_skips_rejected_draft_rows() -> None:
    """A rejected draft's row is replaced by the row forwarded at its position later."""
    prompt_ids, output_ids = [101, 102], [11, 12, 13, 14, 15]
    forwarded = [(101, 0), (102, 1), (11, 2), (12, 3), (99, 4), (13, 4), (14, 5)]

    info = llm2tts([_ledger_output(prompt_ids, output_ids, forwarded)], prompt=[{}])[0]["model_intermediate_buffer"]

    assert info["ids"]["tts"] == [11, 12, 13, 14]
    hidden = torch.as_tensor(unpack_handoff_tensor(info["hidden_states"]["tts"]))
    torch.testing.assert_close(hidden, torch.stack([_row(11, 2), _row(12, 3), _row(13, 4), _row(14, 5)]))


def test_plain_chat_handoff_warns_when_rows_stop_before_the_last_token(mocker) -> None:
    from vllm_omni.model_executor.stage_input_processors import minicpmo_4_5_omni as module

    warning = mocker.patch.object(module.logger, "warning")
    prompt_ids, output_ids = [101, 102], [11, 12, 13, 14]
    # Position 3 never got a row, so the reply stops after token 11.
    forwarded = [(101, 0), (102, 1), (11, 2), (13, 4)]

    info = llm2tts([_ledger_output(prompt_ids, output_ids, forwarded)], prompt=[{}])[0]["model_intermediate_buffer"]

    assert info["ids"]["tts"] == [11]
    warning.assert_called_once()
    assert "stop at token" in warning.call_args.args[0]
    assert warning.call_args.args[1] == "req-1"


def test_plain_chat_handoff_logs_a_ledger_of_the_wrong_length_once(mocker) -> None:
    from vllm_omni.model_executor.stage_input_processors import minicpmo_4_5_omni as module

    warning_once = mocker.patch.object(module.logger, "warning_once")
    debug = mocker.patch.object(module.logger, "debug")
    prompt_ids, output_ids = [101, 102], [11, 12]
    forwarded = [(101, 0), (102, 1), (11, 2)]
    source = _ledger_output(prompt_ids, output_ids, forwarded)
    source.outputs[0].multimodal_output["latent_positions"] = torch.tensor([0, 1]).reshape(-1, 1)

    info = llm2tts([source], prompt=[{}])[0]["model_intermediate_buffer"]

    # Without a usable ledger the rows are taken by index.
    assert info["ids"]["tts"] == [11]
    warning_once.assert_called_once()
    assert "row ledger" in warning_once.call_args.args[0]
    # The counts go to the debug log, one line per mismatching output.
    debug.assert_called_once()
    assert debug.call_args.args[1:] == ("req-1", 3, 2, 3)


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
    assert torch.equal(unpack_handoff_tensor(info["codes"]["ref"]), ref_waveform)
    assert info["meta"]["ref_audio_sr"] == 22050
    assert info["ids"]["tts"] == [11, 12]


@pytest.mark.parametrize("metadata_layout", ["scalar", "list", "tensor", "flat_tensor"])
def test_native_duplex_speak_segment_reaches_split_talker(metadata_layout: str) -> None:
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
    # Per-step boundary snapshots are accumulated as lists or concatenated
    # tensors by the output processor before the native duplex handoff.
    meta = source.outputs[0].multimodal_output["meta"]
    if metadata_layout == "list":
        meta.update({key: [value, value] for key, value in meta.items()})
    elif metadata_layout in ("tensor", "flat_tensor"):
        meta.update({key: torch.tensor([[value], [value]]) for key, value in meta.items()})
    if metadata_layout == "flat_tensor":
        mm_output = source.outputs[0].multimodal_output
        mm_output.pop("meta")
        mm_output.update({f"meta.{key}": value for key, value in meta.items()})
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


def test_native_duplex_uses_forwarded_row_ledger_after_later_audio_prefill() -> None:
    prompt_ids = [101, 102]
    output_ids = [9304, 21, 22, 9308]
    row_ids = [101, 102, 9304, 21, 22, 999, 999]
    latent = torch.arange(len(row_ids) * 4, dtype=torch.float32).reshape(-1, 4)
    metadata = {
        "tts_bos_token_id": 9301,
        "tts_eos_token_id": 9302,
        "listen_token_id": 9303,
        "speak_token_id": 9304,
        "chunk_eos_token_id": 9308,
        "chunk_tts_eos_token_id": 9309,
        "turn_eos_token_id": 9310,
    }
    source = _output(
        prompt_ids=prompt_ids,
        output_ids=output_ids,
        latent=latent,
        multimodal_output={
            "duplex_prompt_token_ids": prompt_ids,
            "latent_input_ids": torch.tensor(row_ids).reshape(-1, 1),
            "latent_positions": torch.arange(len(row_ids)).reshape(-1, 1),
            "meta": metadata,
        },
    )
    context = SimpleNamespace(bridge_states={"duplex": {"epoch": 3, "model_turn_id": 7}})

    converted = llm2tts([source], prompt=[{}], _streaming_context=context)[0]

    info = converted["model_intermediate_buffer"]
    assert info["ids"]["tts"] == [21, 22]
    assert torch.equal(torch.as_tensor(unpack_handoff_tensor(info["hidden_states"]["tts"])), latent[3:5])


def test_native_duplex_selects_latest_contiguous_forwarded_span() -> None:
    prompt_ids = [101, 102]
    output_ids = [9304, 21, 22, 9308]
    row_ids = [101, 21, 22, 102, 9304, 21, 22, 999]
    positions = [0, 1, 3, 4, 5, 6, 7, 8]
    latent = torch.arange(len(row_ids) * 4, dtype=torch.float32).reshape(-1, 4)
    source = _output(
        prompt_ids=prompt_ids,
        output_ids=output_ids,
        latent=latent,
        multimodal_output={
            "duplex_prompt_token_ids": prompt_ids,
            "latent_input_ids": torch.tensor(row_ids).reshape(-1, 1),
            "latent_positions": torch.tensor(positions).reshape(-1, 1),
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
    context = SimpleNamespace(bridge_states={"duplex": {"epoch": 3, "model_turn_id": 7}})

    converted = llm2tts([source], prompt=[{}], _streaming_context=context)[0]

    info = converted["model_intermediate_buffer"]
    assert torch.equal(torch.as_tensor(unpack_handoff_tensor(info["hidden_states"]["tts"])), latent[5:7])


def test_native_duplex_rejects_missing_forwarded_terminal_token() -> None:
    prompt_ids = [101, 102]
    output_ids = [9304, 21, 9310, 9308]
    row_ids = [101, 102, 9304, 21, 999, 999]
    latent = torch.arange(len(row_ids) * 4, dtype=torch.float32).reshape(-1, 4)
    source = _output(
        prompt_ids=prompt_ids,
        output_ids=output_ids,
        latent=latent,
        multimodal_output={
            "duplex_prompt_token_ids": prompt_ids,
            "latent_input_ids": torch.tensor(row_ids).reshape(-1, 1),
            "latent_positions": torch.arange(len(row_ids)).reshape(-1, 1),
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
    context = SimpleNamespace(bridge_states={"duplex": {"epoch": 3, "model_turn_id": 7}})

    with pytest.raises(ValueError, match="missing own-token hidden states"):
        llm2tts([source], prompt=[{}], _streaming_context=context)


_NATIVE_META = {
    "tts_bos_token_id": 9301,
    "tts_eos_token_id": 9302,
    "listen_token_id": 9303,
    "speak_token_id": 9304,
    "chunk_eos_token_id": 9308,
    "chunk_tts_eos_token_id": 9309,
    "turn_eos_token_id": 9310,
}


def _native_source(prompt_ids, output_ids, row_ids, positions=None):
    latent = torch.arange(len(row_ids) * 4, dtype=torch.float32).reshape(-1, 4)
    positions = list(range(len(row_ids))) if positions is None else positions
    source = _output(
        prompt_ids=prompt_ids,
        output_ids=output_ids,
        latent=latent,
        multimodal_output={
            "duplex_prompt_token_ids": prompt_ids,
            "latent_input_ids": torch.tensor(row_ids).reshape(-1, 1),
            "latent_positions": torch.tensor(positions).reshape(-1, 1),
            "meta": dict(_NATIVE_META),
        },
    )
    return source, latent


def _native_handoff(source):
    context = SimpleNamespace(bridge_states={"duplex": {"epoch": 3, "model_turn_id": 7}})
    return llm2tts([source], prompt=[{}], _streaming_context=context)[0]["model_intermediate_buffer"]


def test_native_duplex_ledger_tolerates_async_lookahead_row_after_terminator() -> None:
    """An async scheduler forwards the sampled chunk terminator in one extra
    frame before the segment stop lands; its row trails the unit's rows and
    must not shift the Talker slice."""
    source, latent = _native_source(
        prompt_ids=[101, 102],
        output_ids=[9304, 21, 22, 9308],
        row_ids=[101, 102, 9304, 21, 22, 9308],
    )

    info = _native_handoff(source)

    assert info["ids"]["tts"] == [21, 22]
    assert torch.equal(torch.as_tensor(unpack_handoff_tensor(info["hidden_states"]["tts"])), latent[3:5])


def test_native_duplex_ledger_without_forwarded_terminator() -> None:
    """Synchronous scheduling never forwards the chunk terminator."""
    source, latent = _native_source(
        prompt_ids=[101, 102],
        output_ids=[9304, 21, 22, 9308],
        row_ids=[101, 102, 9304, 21, 22],
    )

    info = _native_handoff(source)

    assert info["ids"]["tts"] == [21, 22]
    assert torch.equal(torch.as_tensor(unpack_handoff_tensor(info["hidden_states"]["tts"])), latent[3:5])


@pytest.mark.parametrize("boundary", [9308, 9309])
def test_native_duplex_empty_speech_chunk_preserves_segment_boundary(boundary: int) -> None:
    # A speech unit can contain only the speak decision and chunk terminator.
    # Dropping this handoff loses the TTS boundary that drives silence continuation.
    source, _ = _native_source(
        prompt_ids=[101, 102],
        output_ids=[9304, boundary],
        row_ids=[101, 102, 9304],
    )
    info = _native_handoff(source)
    assert info["ids"]["tts"] == []
    assert torch.as_tensor(unpack_handoff_tensor(info["hidden_states"]["tts"])).numel() == 0
    assert info["meta"]["segment_end"] is True
    assert info["meta"]["turn_end"] is False
    assert info["meta"]["next_stage_prompt_len"] == 2
    assert info["meta"]["streaming_condition_seq"] == 0


@pytest.mark.parametrize("boundary", [9308, 9309])
@pytest.mark.parametrize("leading_listens", [0, 2])
def test_native_duplex_terminator_first_unit_is_an_empty_condition_and_the_next_unit_keeps_its_own(
    boundary: int, leading_listens: int
) -> None:
    """Stage 0 draws the chunk terminator on every uncapped step, so a unit
    can open with it: no <|speak|>, no folded <|tts_bos|>, no speech. The
    unit still ends a segment, so it is handed over as an empty condition
    (closed by the Talker like any segment) and the following speaking unit
    gets its own hand-off and condition sequence rather than standing in for
    the terminator-only unit's missing segment end."""
    prompt_ids = [101, 102]
    context = SimpleNamespace(bridge_states={"duplex": {"epoch": 3, "model_turn_id": 7}})
    first_ids = [*([9303] * leading_listens), boundary]
    first = _output(
        prompt_ids=prompt_ids,
        output_ids=first_ids,
        latent=torch.arange((len(prompt_ids) + len(first_ids)) * 4, dtype=torch.float32).reshape(-1, 4),
        multimodal_output={"duplex_prompt_token_ids": prompt_ids, "meta": dict(_NATIVE_META)},
    )
    second_ids = [*first_ids, 9304, 21, 22, 9308]
    second_latent = torch.arange((len(prompt_ids) + len(second_ids)) * 4, dtype=torch.float32).reshape(-1, 4)
    second = _output(
        prompt_ids=prompt_ids,
        output_ids=second_ids,
        latent=second_latent,
        multimodal_output={"duplex_prompt_token_ids": prompt_ids, "meta": dict(_NATIVE_META)},
    )

    first_info = llm2tts([first], prompt=[{}], _streaming_context=context)[0]["model_intermediate_buffer"]
    second_info = llm2tts([second], prompt=[{}], _streaming_context=context)[0]["model_intermediate_buffer"]

    assert first_info["ids"]["tts"] == []
    assert torch.as_tensor(unpack_handoff_tensor(first_info["hidden_states"]["tts"])).numel() == 0
    assert first_info["meta"]["segment_end"] is True
    assert first_info["meta"]["turn_end"] is False
    assert first_info["meta"]["next_stage_prompt_len"] == 2
    assert first_info["meta"]["streaming_condition_seq"] == 0
    text_rows = len(prompt_ids) + len(first_ids) + 1
    assert second_info["ids"]["tts"] == [21, 22]
    assert torch.equal(
        torch.as_tensor(unpack_handoff_tensor(second_info["hidden_states"]["tts"])),
        second_latent[text_rows : text_rows + 2],
    )
    assert second_info["meta"]["segment_end"] is True
    assert second_info["meta"]["streaming_condition_seq"] == 1


def test_native_duplex_listen_only_unit_hands_the_talker_nothing() -> None:
    """A unit that ends in <|listen|> (not a chunk terminator) has no Talker
    segment: it is the listen decision, not an empty speech segment."""
    source, _ = _native_source(prompt_ids=[101, 102], output_ids=[9303, 9303], row_ids=[101, 102, 9303, 9303])
    context = SimpleNamespace(bridge_states={"duplex": {"epoch": 3, "model_turn_id": 7}})

    assert llm2tts([source], prompt=[{}], _streaming_context=context) == []


def test_native_duplex_turn_eos_row_is_handed_to_talker() -> None:
    """<|turn_eos|> is forwarded (it is not a stop token) and the model keeps
    sampling until a chunk terminator; the Talker gets every forwarded row
    before that boundary, including text that reopens the turn."""
    source, latent = _native_source(
        prompt_ids=[101, 102],
        output_ids=[9304, 21, 9310, 77, 9308],
        row_ids=[101, 102, 9304, 21, 9310, 77],
    )

    info = _native_handoff(source)

    assert info["ids"]["tts"] == [21, 9310, 77]
    assert torch.equal(torch.as_tensor(unpack_handoff_tensor(info["hidden_states"]["tts"])), latent[3:6])
    assert info["meta"]["turn_end"] is False


@pytest.mark.parametrize("previous_units", [[9303, 9303], [9303, 9308], [9308, 9309]])
def test_native_duplex_ledger_anchors_on_unit_after_reinjected_controls(previous_units: list[int]) -> None:
    """Listens and empty chunks sit at the head of the segment delta but
    were re-injected as prompt tokens, so they are not contiguous with this
    unit's decode rows in the ledger. Anchor on the unit, not the segment."""
    unit_end, unit, audio = 2, 1, 11
    source, latent = _native_source(
        prompt_ids=[101, 102],
        output_ids=[*previous_units, 9304, 21, 22, 9308],
        row_ids=[previous_units[0], unit_end, unit, audio, previous_units[1], unit_end, unit, audio, 9304, 21, 22],
    )

    info = _native_handoff(source)

    assert info["ids"]["tts"] == [21, 22]
    assert torch.equal(torch.as_tensor(unpack_handoff_tensor(info["hidden_states"]["tts"])), latent[9:11])


@pytest.mark.parametrize("previous_terminator_forwarded", [False, True])
@pytest.mark.parametrize("current_terminator_forwarded", [False, True])
def test_native_duplex_ledger_prefers_latest_repeat_of_the_unit(
    previous_terminator_forwarded: bool, current_terminator_forwarded: bool
) -> None:
    previous_ids = [9304, 21, 22] + ([9308] if previous_terminator_forwarded else [])
    current_ids = [9304, 21, 22] + ([9308] if current_terminator_forwarded else [])
    current_start = len(previous_ids) + 1
    source, latent = _native_source(
        prompt_ids=[101, 102],
        output_ids=[9304, 21, 22, 9308],
        row_ids=[*previous_ids, 555, *current_ids],
        positions=[*range(current_start), *range(10, 10 + len(current_ids))],
    )

    info = _native_handoff(source)

    assert info["ids"]["tts"] == [21, 22]
    assert torch.equal(
        torch.as_tensor(unpack_handoff_tensor(info["hidden_states"]["tts"])),
        latent[current_start + 1 : current_start + 3],
    )


def test_native_duplex_ledger_rejects_non_contiguous_positions() -> None:
    source, _ = _native_source(
        prompt_ids=[101, 102],
        output_ids=[9304, 21, 22, 9308],
        row_ids=[101, 102, 9304, 21, 22],
        positions=[0, 1, 2, 3, 9],
    )

    with pytest.raises(ValueError, match="missing own-token hidden states"):
        _native_handoff(source)


def test_native_duplex_mid_turn_tts_bos_slices_after_boundary() -> None:
    source, latent = _native_source(
        prompt_ids=[101, 102],
        output_ids=[9301, 21, 22, 9308],
        row_ids=[101, 102, 9301, 21, 22],
    )

    info = _native_handoff(source)

    assert info["ids"]["tts"] == [21, 22]
    assert torch.equal(torch.as_tensor(unpack_handoff_tensor(info["hidden_states"]["tts"])), latent[3:5])


@pytest.mark.parametrize("opening", [9304, 9301])
@pytest.mark.parametrize("turn_end", [True, False])
def test_native_duplex_mid_unit_tts_bos_keeps_earlier_text(opening, turn_end) -> None:
    # The policy rewrites a mid-turn <|listen|> into <|tts_bos|> anywhere in a
    # unit. It is conditioning like text; it must not restart the slice and
    # drop the words sampled before it.
    tail = [9310, 9308] if turn_end else [9308]
    source, latent = _native_source(
        prompt_ids=[101, 102],
        output_ids=[opening, 21, 22, 9301, *tail],
        row_ids=[101, 102, opening, 21, 22, 9301, *tail[:1]],
    )

    info = _native_handoff(source)

    expected = [21, 22, 9301, 9310] if turn_end else [21, 22, 9301]
    assert info["ids"]["tts"] == expected
    torch.testing.assert_close(
        torch.as_tensor(unpack_handoff_tensor(info["hidden_states"]["tts"])), latent[3 : 3 + len(expected)]
    )
    assert info.get("meta", {}).get("turn_end", False) is turn_end


@pytest.mark.parametrize("terminator", [[9308], [9310, 9308]])
def test_native_duplex_folded_boundary_with_mid_unit_tts_bos(terminator) -> None:
    # A boundary folded as the last prompt token opens the unit even when the
    # policy also rewrites a later listen into <|tts_bos|> inside it.
    source, latent = _native_source(
        prompt_ids=[101, 9301],
        output_ids=[21, 22, 9301, *terminator],
        row_ids=[101, 9301, 21, 22, 9301, *terminator[:-1]],
    )

    info = _native_handoff(source)

    expected = [21, 22, 9301, *terminator[:-1]]
    assert info["ids"]["tts"] == expected
    torch.testing.assert_close(
        torch.as_tensor(unpack_handoff_tensor(info["hidden_states"]["tts"])), latent[2 : 2 + len(expected)]
    )


@pytest.mark.parametrize("folded_decisions", [0, 1, 2])
@pytest.mark.parametrize("bos_in_prompt", [False, True])
def test_native_duplex_tts_bos_aligns_after_window_rebuild(folded_decisions, bos_in_prompt) -> None:
    prompt_ids = [101] * 10
    output_ids = [9303] * folded_decisions + [9301, 21, 22, 9308]
    if bos_in_prompt:
        prompt_ids[-1] = 9301
        output_ids = [21, 22, 9308]
    rows = len(prompt_ids) + len(output_ids) - folded_decisions
    latent = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4)
    forwarded_ids = (
        prompt_ids[folded_decisions:] + output_ids if bos_in_prompt else prompt_ids + output_ids[folded_decisions:]
    )
    source = _output(
        prompt_ids=prompt_ids,
        output_ids=output_ids,
        latent=latent,
        multimodal_output={
            "duplex_prompt_token_ids": prompt_ids,
            "latent_input_ids": torch.tensor(forwarded_ids).reshape(-1, 1),
            "latent_positions": torch.arange(len(forwarded_ids)).reshape(-1, 1),
            "meta": {
                "tts_bos_token_id": 9301,
                "tts_eos_token_id": 9302,
                "listen_token_id": 9303,
                "speak_token_id": 9304,
                "chunk_eos_token_id": 9308,
            },
        },
    )
    converted = llm2tts([source], prompt=[{}], _streaming_context=SimpleNamespace(bridge_states={}))[0]
    info = converted["model_intermediate_buffer"]
    assert info["ids"]["tts"] == [21, 22]
    torch.testing.assert_close(torch.as_tensor(unpack_handoff_tensor(info["hidden_states"]["tts"])), latent[-3:-1])


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
    assert "replace_streaming_prompt" not in second_input["model_intermediate_buffer"]["meta"]
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


@pytest.mark.parametrize(
    "text,complete",
    [
        ("See you then.", True),
        ("可以去看看哦。", True),
        ('He said "go!"', True),
        ("像之前电影当中的", False),
        ("Wait, ", False),
        ("", False),
    ],
)
def test_native_duplex_turn_end_closes_text_only_after_a_sentence_end(text, complete) -> None:
    source, _ = _native_source(
        prompt_ids=[101, 102],
        output_ids=[9304, 21, 9310, 9308],
        row_ids=[101, 102, 9304, 21, 9310],
    )
    context = StreamingInputState(
        bridge_states={"duplex": {"epoch": 3, "model_turn_id": 7}},
        source_token_decoder=lambda ids, **_: text if 21 in ids else "",
    )

    info = llm2tts([source], prompt=[{}], _streaming_context=context)[0]["model_intermediate_buffer"]

    assert info["meta"]["turn_end"] is True
    assert info["meta"]["turn_text_ends_sentence"] is complete
    # The condition is the forwarded rows plus <audio_bos>, with <text_eos> before it once closed.
    assert info["meta"]["next_stage_prompt_len"] == len(info["ids"]["tts"]) + (2 if complete else 1)


@pytest.mark.parametrize(
    "unit_text,after_turn_eos,terminal_eos,complete",
    [
        ("So the", ".", False, False),
        ("Sure, here it is.", "And", False, False),
        ("So the", ".", True, True),
        ("Sure, here it is.", "And", True, False),
    ],
)
def test_native_duplex_turn_end_follows_last_forwarded_token(unit_text, after_turn_eos, terminal_eos, complete) -> None:
    # An intermediate turn_eos does not end the unit. Later tokens reopen the
    # turn, and must keep their text and own-token hidden rows for the Talker.
    content = [21, 9310, 22, *([9310] if terminal_eos else [])]
    source, latent = _native_source(
        prompt_ids=[101, 102],
        output_ids=[9304, *content, 9308],
        row_ids=[101, 102, 9304, *content],
    )
    token_text = {21: unit_text, 22: after_turn_eos}
    context = StreamingInputState(
        bridge_states={"duplex": {"epoch": 3, "model_turn_id": 7}},
        source_token_decoder=lambda ids, **_: "".join(token_text.get(int(i), "") for i in ids),
    )

    info = llm2tts([source], prompt=[{}], _streaming_context=context)[0]["model_intermediate_buffer"]

    assert info["ids"]["tts"] == content
    torch.testing.assert_close(torch.as_tensor(unpack_handoff_tensor(info["hidden_states"]["tts"])), latent[3:])
    assert info["meta"]["native_duplex_segment_text"] == unit_text + after_turn_eos
    assert info["meta"]["turn_end"] is terminal_eos
    assert info["meta"]["turn_text_ends_sentence"] is complete
    assert info["meta"]["next_stage_prompt_len"] == len(info["ids"]["tts"]) + (2 if complete else 1)
    assert context.bridge_states["duplex"]["model_turn_id"] == 7 + terminal_eos


def test_native_duplex_preserves_ci19446_mid_unit_turn_eos() -> None:
    # CI 19446 unit 159: the model recognized the color, but the first EOS
    # used to discard "是一个绿色的正" and close the response before unit 160.
    speak, turn_eos, chunk_eos = 151706, 151717, 151718
    content = [turn_eos, speak, 101909, 100706, 9370, 36556]
    output_ids = [speak, *content, chunk_eos]
    source, latent = _native_source(
        prompt_ids=[101, 102],
        output_ids=output_ids,
        # The extra forwarded row is async lookahead after the real boundary.
        row_ids=[101, 102, *output_ids, 999],
    )
    source.outputs[0].multimodal_output["meta"].update(
        speak_token_id=speak, turn_eos_token_id=turn_eos, chunk_eos_token_id=chunk_eos
    )
    token_text = {101909: "是一个", 100706: "绿色", 9370: "的", 36556: "正"}
    context = StreamingInputState(
        bridge_states={"duplex": {"epoch": 0, "model_turn_id": 10}},
        source_token_decoder=lambda ids, **_: "".join(token_text.get(int(i), "") for i in ids),
    )

    info = llm2tts([source], prompt=[{}], _streaming_context=context)[0]["model_intermediate_buffer"]

    assert info["ids"]["tts"] == content
    torch.testing.assert_close(
        torch.as_tensor(unpack_handoff_tensor(info["hidden_states"]["tts"])), latent[3 : 3 + len(content)]
    )
    assert info["meta"]["native_duplex_segment_text"] == "是一个绿色的正"
    assert info["meta"]["segment_end"] is True
    assert info["meta"]["turn_end"] is False
    assert context.bridge_states["duplex"]["model_turn_id"] == 10


def test_native_duplex_turn_end_uses_the_last_text_of_the_turn() -> None:
    prompt_ids = [101, 102]
    token_text = {21: "Sure, here it is.", 22: "And", 23: "more."}

    def output(token_ids: list[int]):
        return _output(
            prompt_ids=prompt_ids,
            output_ids=token_ids,
            latent=torch.zeros((len(prompt_ids) + len(token_ids), 1)),
            multimodal_output={"duplex_prompt_token_ids": prompt_ids, "meta": dict(_NATIVE_META)},
        )

    context = StreamingInputState(
        bridge_states={"duplex": {"epoch": 3, "model_turn_id": 7}},
        source_token_decoder=lambda ids, **_: " ".join(token_text[int(i)] for i in ids if int(i) in token_text),
    )
    first_ids = [9304, 21, 9308]
    first = llm2tts([output(first_ids)], prompt=[{}], _streaming_context=context)[0]["model_intermediate_buffer"]
    # The turn ends in a unit that carries only <|turn_eos|>: the sentence ended in the unit before.
    second = llm2tts([output([*first_ids, 9304, 9310, 9308])], prompt=[{}], _streaming_context=context)[0][
        "model_intermediate_buffer"
    ]
    assert first["meta"]["turn_text_ends_sentence"] is False
    assert second["meta"]["turn_end"] is True
    assert second["meta"]["turn_text_ends_sentence"] is True

    # The next turn starts over: a turn-end unit without text of its own does not
    # inherit the previous turn's sentence end.
    context.bridge_states["duplex"]["model_turn_id"] = 8
    third_ids = [9304, 9310, 9308]
    third = llm2tts([output(third_ids)], prompt=[{}], _streaming_context=context)[0]["model_intermediate_buffer"]
    assert third["meta"]["turn_end"] is True
    assert third["meta"]["turn_text_ends_sentence"] is False


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


def test_venus_delegation_masks_aligned_speech_rows_and_preserves_text_tokens():
    ids = [9304, 21, 9400, 31, 32, 9401, 22, 9310, 9308]
    source, latent = _native_source([101], ids, [101, *ids[:-1]])
    source.outputs[0].multimodal_output["meta"].update(delegate_open_token_id=9400, delegate_close_token_id=9401)
    info = _native_handoff(source)
    assert info["ids"]["tts"] == [21, 22, 9310]
    assert torch.equal(torch.as_tensor(unpack_handoff_tensor(info["hidden_states"]["tts"])), latent[[2, 7, 8]])
    assert info["meta"]["turn_end"] is True
    assert source.outputs[0].token_ids == ids


@pytest.mark.parametrize("decision_prefix", [[], [9304], [9303]], ids=["no_speak", "speak", "leading_listen"])
def test_venus_handoff_preserves_delegate_tags_when_tokenizer_skips_specials(decision_prefix):
    from vllm_omni.model_executor.models.realtime_venus.delegation import DelegateParser

    def decode(ids, *, skip_special_tokens=False):
        vocabulary = {21: "add ", 22: "12 and 30", 9400: "<delegate>", 9401: "</delegate>", 9308: "<|chunk_eos|>"}
        return "".join(vocabulary.get(token, "") for token in ids if not (skip_special_tokens and token >= 9000))

    context = StreamingInputState(
        bridge_states={"duplex": {"epoch": 0, "model_turn_id": 0}}, source_token_decoder=decode
    )
    parser = DelegateParser()
    cumulative = []
    texts = []
    calls = []
    for content in ([9400, 21], [22, 9401]):
        unit = [*decision_prefix, *content, 9308]
        cumulative.extend(unit)
        source, _ = _native_source([101], list(cumulative), [101, *unit])
        source.outputs[0].multimodal_output["meta"].update(delegate_open_token_id=9400, delegate_close_token_id=9401)
        handoff = llm2tts([source], prompt=[{}], _streaming_context=context)[0]["model_intermediate_buffer"]
        text = handoff["meta"]["native_duplex_segment_text"]
        texts.append(text)
        calls.append(parser.feed(text))
        assert handoff["ids"]["tts"] == []  # Delimiters/payload still never condition speech.
    assert texts == ["<delegate>add ", "12 and 30</delegate>"]
    assert calls == [None, "add 12 and 30"]


def test_venus_closing_delegate_decision_restores_aligned_speech():
    from vllm_omni.model_executor.models.realtime_venus.delegation import DelegateParser

    context = StreamingInputState(
        bridge_states={"duplex": {"epoch": 0, "model_turn_id": 0}},
        source_token_decoder=lambda ids, **_: "".join({21: "lookup", 22: "Done."}.get(i, "") for i in ids),
    )
    parser = DelegateParser()
    cumulative = []
    handoffs = []
    calls = []
    for unit in ([9304, 9400, 21, 9308], [9401, 22, 9310, 9308]):
        cumulative.extend(unit)
        source, latent = _native_source([101], list(cumulative), [101, *unit[:-1]])
        source.outputs[0].multimodal_output["meta"].update(delegate_open_token_id=9400, delegate_close_token_id=9401)
        info = llm2tts([source], prompt=[{}], _streaming_context=context)[0]["model_intermediate_buffer"]
        handoffs.append(info)
        calls.append(parser.feed(info["meta"]["native_duplex_segment_text"]))

    assert calls == [None, "lookup"]
    assert handoffs[0]["ids"]["tts"] == []
    resumed = handoffs[1]
    assert resumed["ids"]["tts"] == [22, 9310]
    assert torch.equal(torch.as_tensor(unpack_handoff_tensor(resumed["hidden_states"]["tts"])), latent[[2, 3]])
    assert resumed["meta"]["turn_start"] is True
    assert resumed["meta"]["turn_end"] is True
    assert resumed["meta"]["text_without_audio"] is True


@pytest.mark.parametrize("split_delegate", [False, True])
@pytest.mark.parametrize("turn_end", [False, True])
def test_venus_mixed_delegate_zero_frames_reaches_tool_parser(split_delegate, turn_end):
    from vllm.sampling_params import SamplingParams

    from tests.model_executor.models.minicpmo_4_5.test_code2wav_batching import _forward, _model
    from tests.model_executor.models.minicpmo_4_5.test_talker_mrv2 import _EOS, _batch, _req_states, _talker
    from tests.model_executor.stage_input_processors.test_minicpmo_4_5_async_chunk import (
        _duplex_delta,
        _manager,
    )
    from vllm_omni.data_entry_keys import to_dict
    from vllm_omni.engine.duplex.plugin import DuplexDataPlaneContext
    from vllm_omni.model_executor.models.realtime_venus.duplex import VenusDataPlane
    from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import tts2code2wav_async_chunk
    from vllm_omni.outputs import OmniRequestOutput
    from vllm_omni.worker_v2.omni_data_plane import _NativeRequestState

    context = StreamingInputState(
        bridge_states={"duplex": {"epoch": 0, "model_turn_id": 0}},
        source_token_decoder=lambda ids, **_: "".join({21: "lookup", 22: "Done"}.get(i, "") for i in ids),
    )
    units = [[9304, 9400, 21, 9308], [9401, 22]] if split_delegate else [[9304, 9400, 21, 9401, 22]]
    units[-1].extend([9310, 9308] if turn_end else [9308])
    talker = _talker()
    vocoder, _ = _model()
    manager = _manager()
    request_state = _NativeRequestState(
        request_id="delegate",
        external_req_id="delegate",
        prompt_token_ids=[0],
        resumable=True,
        sampling_params=SamplingParams(stop_token_ids=[_EOS]),
    )
    plane = VenusDataPlane(lambda *args: "")
    data_context = DuplexDataPlaneContext(auto_responds=True, active_response_id="response", turn_id=0)
    cumulative, events, emitted_texts = [], [], []
    for unit in units:
        cumulative.extend(unit)
        source, _ = _native_source([101], list(cumulative), [101, *unit[:-1]])
        source.outputs[0].multimodal_output["meta"].update(delegate_open_token_id=9400, delegate_close_token_id=9401)
        info = llm2tts([source], prompt=[{}], _streaming_context=context)[0]["model_intermediate_buffer"]
        # Without sentence-final punctuation, turn-end draining permits EOS
        # immediately. Verify the real sampler mask before injecting that sample.
        prefill, _ = _batch([dict(slot=0, prompt_len=1, computed=0, span=[0], prefill=True)])
        talker.make_omni_output_mrv2(
            torch.zeros((1, 4)),
            input_batch=prefill,
            req_states=_req_states({0: 1}),
            model_intermediate_buffer=[info],
        )
        assert not talker._mrv2_masked_eos[0].item()
        batch, _ = _batch([dict(slot=0, prompt_len=1, computed=1, span=[_EOS], prefill=False)])
        output = talker.make_omni_output_mrv2(
            torch.zeros((1, 4)),
            input_batch=batch,
            req_states=_req_states({0: 1}),
            model_intermediate_buffer=[info],
        ).multimodal_outputs
        output["meta"] = {key: value[0] if isinstance(value, list) else value for key, value in output["meta"].items()}
        request_state.accept_tokens([_EOS])
        request = request_state.snapshot(include_token_history=True, sampled_token_ids=[_EOS])
        payload = tts2code2wav_async_chunk(manager, output, request)
        assert payload.meta.code_flat_numel == 0
        emitted_texts.append(bytes(payload.meta.llm_output_text_utf8.tolist()).decode())
        waveform = _forward(vocoder, [to_dict(payload)]).multimodal_outputs
        projected = OmniRequestOutput(
            request_id="delegate",
            finished=False,
            outputs=[],
            _multimodal_output={key: value[0] for key, value in waveform.items()},
        )
        events.extend(plane.project_output(projected, context=data_context))
        repeated = list(plane.project_output(projected, context=data_context))
        assert not any(event.get("text") or event.get("function_call") or event.get("error") for event in repeated)

    assert not [event for event in events if event.get("error")]
    calls = [event for event in events if event.get("function_call")]
    assert len(calls) == 1
    assert json.loads(calls[0]["arguments"]) == {"question": "lookup"}
    assert "".join(event.get("text", "") for event in events) == "<delegate>lookup</delegate>Done"
    assert "".join(emitted_texts) == "<delegate>lookup</delegate>Done"
    assert sum(event.get("end_of_turn") is True for event in events) == int(turn_end)
    # A later audible unit must not replay text already delivered at zero frames.
    request = request_state.snapshot(include_token_history=True, sampled_token_ids=[])
    following = tts2code2wav_async_chunk(
        manager, _duplex_delta(*range(25), epoch=0, turn_id=int(turn_end), text="Next."), request
    )
    assert bytes(following.meta.llm_output_text_utf8.tolist()) == b"Next."


def test_venus_delimiter_only_decisions_flush_once_and_resume_speech():
    from vllm_omni.engine.duplex.plugin import DuplexDataPlaneContext
    from vllm_omni.model_executor.models.realtime_venus.delegation import DelegateParser
    from vllm_omni.model_executor.models.realtime_venus.duplex import VenusDataPlane

    context = StreamingInputState(
        bridge_states={"duplex": {"epoch": 0, "model_turn_id": 0}},
        source_token_decoder=lambda ids, **_: "".join(
            {21: "lookup", 22: "Done.", 23: "Checking."}.get(i, "") for i in ids
        ),
    )
    parser = DelegateParser()
    plane = VenusDataPlane(lambda *args: "")
    data_context = DuplexDataPlaneContext(auto_responds=True, active_response_id="response", turn_id=0)
    tool_calls: list[dict[str, object]] = []
    cumulative = []
    handoffs = []
    calls = []
    for unit in ([9304, 23, 9308], [9400, 9308], [9304, 21, 9308], [9401, 9308], [9304, 22, 9310, 9308]):
        cumulative.extend(unit)
        source, latent = _native_source([101], list(cumulative), [101, *unit[:-1]])
        source.outputs[0].multimodal_output["meta"].update(delegate_open_token_id=9400, delegate_close_token_id=9401)
        info = llm2tts([source], prompt=[{}], _streaming_context=context)[0]["model_intermediate_buffer"]
        handoffs.append(info)
        calls.append(parser.feed(info["meta"]["native_duplex_segment_text"]))
        # Project the handoff text with the codec bridge's per-unit identity.
        output = SimpleNamespace(
            request_id="req-1",
            finished=False,
            outputs=[],
            multimodal_output={
                "meta": {
                    **info["meta"],
                    "llm_output_text": info["meta"]["native_duplex_segment_text"],
                    "tts_is_last_chunk": True,
                    "duplex_audio_cache_epoch": 0,
                    "duplex_audio_chunk_seq": len(handoffs),
                    "duplex_turn_id": 0,
                }
            },
        )
        tool_calls.extend(
            event for event in plane.project_output(output, context=data_context) if event.get("function_call")
        )

    assert calls == [None, None, None, "lookup", None]
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "delegate"
    arguments = tool_calls[0]["arguments"]
    assert isinstance(arguments, str)
    assert json.loads(arguments) == {"question": "lookup"}
    assert [info["ids"]["tts"] for info in handoffs] == [[23], [], [], [], [22, 9310]]
    assert [info["meta"]["speech_flush"] for info in handoffs] == [False, True, False, False, False]
    assert [info["meta"]["text_without_audio"] for info in handoffs] == [True] * 5
    assert [info["meta"]["native_duplex_segment_text"] for info in handoffs] == [
        "Checking.",
        "<delegate>",
        "lookup",
        "</delegate>",
        "Done.",
    ]
    resumed = handoffs[-1]
    assert resumed["meta"]["turn_start"] is True
    assert resumed["meta"]["turn_end"] is True
    assert torch.equal(torch.as_tensor(unpack_handoff_tensor(resumed["hidden_states"]["tts"])), latent[[2, 3]])


def test_venus_delegation_state_crosses_units_and_resets_at_turn_end():
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.delegation import filter_delegate_condition

    special = {"delegate_open_token_id": 9400, "delegate_close_token_id": 9401, "turn_eos_token_id": 9310}
    state: dict[str, bool] = {}
    hidden = torch.arange(16).reshape(4, 4)
    ids, rows = filter_delegate_condition([21, 9400, 31, 32], hidden, special, state)
    assert ids == [21]
    assert torch.equal(rows, hidden[:1])
    ids, rows = filter_delegate_condition([33, 9401, 22, 23], hidden, special, state)
    assert ids == [22, 23]
    assert torch.equal(rows, hidden[2:])
    ids, rows = filter_delegate_condition([9400, 34, 35, 9310], hidden, special, state)
    assert ids == []
    assert rows.shape == (0, 4)
    assert not state["delegate_muted"]
    ids, rows = filter_delegate_condition([21, 22, 23, 24], hidden, special, state)
    assert ids == [21, 22, 23, 24]
    assert rows is hidden


@pytest.mark.parametrize("resume_speech", [False, True])
def test_venus_muted_units_flush_once_and_keep_dialogue_open(resume_speech):
    context = StreamingInputState(
        bridge_states={"duplex": {"epoch": 0, "model_turn_id": 0}},
        source_token_decoder=lambda ids, **_: "".join(
            {21: "Hello.", 31: "secret", 22: "Again."}.get(i, "") for i in ids
        ),
    )
    cumulative = []

    def handoff(content):
        unit = [9304, *content, 9308]
        cumulative.extend(unit)
        source, _ = _native_source([101], list(cumulative), [101, *unit])
        source.outputs[0].multimodal_output["meta"].update(delegate_open_token_id=9400, delegate_close_token_id=9401)
        return llm2tts([source], prompt=[{}], _streaming_context=context)[0]["model_intermediate_buffer"]

    first = handoff([21])
    assert first["ids"]["tts"] == [21]
    muted = handoff([9400, 31])
    assert muted["ids"]["tts"] == []
    assert muted["meta"]["speech_flush"] is True
    assert muted["meta"]["turn_end"] is False
    assert muted["meta"]["next_stage_prompt_len"] == 1
    assert muted["meta"]["native_duplex_segment_text"] == "<delegate>secret"
    again = handoff([31])
    assert again["ids"]["tts"] == []
    assert again["meta"]["speech_flush"] is False
    assert context.bridge_states["duplex"]["model_turn_id"] == 0

    if resume_speech:
        resumed = handoff([9401, 22])
        assert resumed["ids"]["tts"] == [22]
        assert resumed["meta"]["turn_start"] is True
        assert resumed["meta"]["replace_streaming_prompt"] is True
    terminal = handoff([9310])
    assert terminal["meta"]["turn_end"] is True
    assert terminal["meta"]["speech_flush"] is False
    assert terminal["ids"]["tts"] == ([9310] if resume_speech else [])
    assert context.bridge_states["duplex"]["model_turn_id"] == 1
    following = handoff([22])
    assert following["ids"]["tts"] == [22]
    assert following["meta"]["turn_start"] is True


def test_venus_delegation_only_turn_is_silent_and_new_request_resets_mask():
    context = StreamingInputState(bridge_states={"duplex": {"epoch": 0, "model_turn_id": 0}})
    source, _ = _native_source([101], [9304, 9400, 31, 9308], [101, 9304, 9400, 31, 9308])
    source.outputs[0].multimodal_output["meta"].update(delegate_open_token_id=9400, delegate_close_token_id=9401)
    info = llm2tts([source], prompt=[{}], _streaming_context=context)[0]["model_intermediate_buffer"]
    assert info["ids"]["tts"] == []
    assert info["meta"]["speech_flush"] is False
    # An aborted/replaced request must not inherit the old request's mute state.
    source, _ = _native_source([101], [9304, 21, 9308], [101, 9304, 21, 9308])
    source.request_id = "replacement"
    source.outputs[0].multimodal_output["meta"].update(delegate_open_token_id=9400, delegate_close_token_id=9401)
    info = llm2tts([source], prompt=[{}], _streaming_context=context)[0]["model_intermediate_buffer"]
    assert info["ids"]["tts"] == [21]


@pytest.mark.parametrize("muted_units", [1, 2])
def test_venus_first_speech_after_initial_delegation_starts_fresh(muted_units: int) -> None:
    """Initial silent units must not turn the first speech into a continuation."""
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        _native_duplex_chunk_budget,
    )

    context = StreamingInputState(bridge_states={"duplex": {"epoch": 0, "model_turn_id": 0}})
    cumulative: list[int] = []

    def handoff(content: list[int]):
        unit = [9304, *content, 9308]
        cumulative.extend(unit)
        source, _ = _native_source([101], list(cumulative), [101, *unit])
        source.outputs[0].multimodal_output["meta"].update(delegate_open_token_id=9400, delegate_close_token_id=9401)
        return llm2tts([source], prompt=[{}], _streaming_context=context)[0]["model_intermediate_buffer"]

    for index in range(muted_units):
        muted = handoff([9400, 31] if index == 0 else [31])
        assert muted["ids"]["tts"] == []
        assert muted["meta"]["text_without_audio"] is True
        assert muted["meta"]["speech_flush"] is False

    first = handoff([9401, 21])
    assert first["ids"]["tts"] == [21]
    assert first["meta"]["turn_start"] is True
    assert first["meta"]["replace_streaming_prompt"] is True
    assert first["meta"]["speech_flush"] is False
    assert first["meta"]["text_without_audio"] is True
    assert _native_duplex_chunk_budget(first["meta"]) == (26, 0)

    following = handoff([22])
    assert following["ids"]["tts"] == [22]
    assert following["meta"]["turn_start"] is False
    assert not following["meta"].get("replace_streaming_prompt", False)
    assert _native_duplex_chunk_budget(following["meta"]) == (26, 26)
    assert context.bridge_states["duplex"]["model_turn_id"] == 0
