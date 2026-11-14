# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3-TTS streamed text: the stage policy, the text lead layout, the Talker's text queue and EOS gate."""

from unittest.mock import create_autospec

import pytest
import torch
from vllm.config import ModelConfig
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.core.sched.output import TextStreamDelivery
from vllm_omni.core.sched.text_stream import TextStreamSpec, end_read
from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
    Qwen3TTSConfig,
    Qwen3TTSTalkerConfig,
)
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    PRECOMPUTED_TEXT_IDS_KEY,
    Qwen3TTSPromptEmbedsBuilder,
)
from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import Qwen3TTSTalkerForConditionalGeneration
from vllm_omni.model_executor.models.qwen3_tts.streamed_text import StreamedTexts
from vllm_omni.model_executor.models.qwen3_tts.text_stream import text_lead, text_stream_spec

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CPU = torch.device("cpu")
HIDDEN = 48
CODEC_PAD = 14
CODEC_BOS = 15
CODEC_EOS = 9
TEXT = [20, 21, 22, 23, 24, 25]
# Assistant template: 3 role ids, the text, 5 closing ids.
ASSISTANT_IDS = [1, 2, 3, *TEXT, 4, 5, 6, 7, 8]


def _delivery(start, ids, ended=False, prompt_ids=1):
    return TextStreamDelivery(start=start, token_ids=list(ids), ended=ended, prompt_ids=prompt_ids)


# -------------------- the stage policy --------------------


def _info(**overrides):
    info = {"task_type": ["CustomVoice"], "non_streaming_mode": [False]}
    info.update(overrides)
    return info


def test_the_policy_reads_the_text_lead_of_a_streaming_custom_voice_request():
    assert text_stream_spec(_info()) == TextStreamSpec(prompt_ids=1, min_ids=0)
    assert text_stream_spec(_info(text_lead=[3])) == TextStreamSpec(prompt_ids=4, min_ids=2)
    # Appended ids must index the Talker's text embedding.
    model_config = create_autospec(ModelConfig, instance=True)
    model_config.hf_config = Qwen3TTSConfig(talker_config={"text_vocab_size": 151936})
    assert text_stream_spec(_info(), model_config).id_limit == 151936


@pytest.mark.parametrize(
    ("info", "message"),
    [
        (_info(non_streaming_mode=[True]), "streaming mode"),
        ({"task_type": ["CustomVoice"]}, "streaming mode"),  # CustomVoice defaults to non-streaming
        (_info(task_type=["Base"]), "only for CustomVoice"),
        (_info(task_type=["VoiceDesign"]), "only for CustomVoice"),
        (_info(text_lead=[-1]), "text_lead"),
    ],
)
def test_the_policy_refuses_requests_whose_prompt_cannot_stream(info, message):
    with pytest.raises(ValueError, match=message):
        text_stream_spec(info)


@pytest.mark.parametrize("value", [-1, 1.5, True, "2", [3], float("inf"), float("nan")])
def test_a_text_lead_must_be_a_non_negative_integer(value):
    with pytest.raises(ValueError, match="text_lead"):
        text_lead({"text_lead": [value]})


def test_a_missing_text_lead_is_the_stock_layout():
    assert text_lead({}) == 0 and text_lead(None) == 0 and text_lead({"text_lead": [2.0]}) == 2


# -------------------- the Talker's streamed text --------------------


def _rows(ids):
    """Stand-in projection: ``[1, k]`` ids to ``[1, k, 2]`` rows that differ by id and position."""
    ids = torch.as_tensor(ids).reshape(-1)
    count = ids.numel()
    return torch.arange(count * 2, dtype=torch.float32).reshape(1, count, 2) + ids.reshape(1, -1, 1).float() * 100


def _end_row(device):
    return torch.full((1, 1, 2), -9.0, device=device)


def test_deliveries_are_taken_once_and_a_gap_is_refused():
    texts = StreamedTexts()
    texts.receive({"r": _delivery(0, [5, 6])})
    texts.receive({"r": _delivery(1, [6, 7])})  # overlaps: only 7 is new
    ids, ended = texts.prompt_text("r", torch.device("cpu"))
    assert ids.tolist() == [[5, 6, 7]] and not ended
    with pytest.raises(RuntimeError, match="skips ids"):
        texts.receive({"r": _delivery(9, [1])})


def test_the_queue_grows_by_new_rows_once_and_ends_once():
    texts = StreamedTexts()
    texts.receive({"r": _delivery(0, [5])})
    texts.prompt_text("r", torch.device("cpu"))  # the prompt build takes id 5
    tail = torch.zeros((0, 2))
    same, offset, changed = texts.extend_queue("r", tail, 0, device=CPU, project=_rows, end_row=_end_row)
    assert (same is tail, offset, changed) == (True, 0, False)

    texts.receive({"r": _delivery(1, [6, 7])})
    tail, offset, changed = texts.extend_queue("r", tail, 0, device=CPU, project=_rows, end_row=_end_row)
    assert changed and offset == 0 and torch.equal(tail, _rows([6, 7])[0])
    # Reading one row then receiving the END keeps the unread row first.
    texts.receive({"r": _delivery(3, [], ended=True)})
    unread = tail[1:]
    tail, offset, changed = texts.extend_queue("r", tail, 1, device=CPU, project=_rows, end_row=_end_row)
    assert changed and torch.equal(tail, torch.cat([unread, _end_row(None)[0]]))
    assert not texts.extend_queue("r", tail, 0, device=CPU, project=_rows, end_row=_end_row)[2]
    # A request without a stream passes through.
    assert texts.extend_queue("other", tail, 1, device=CPU, project=_rows, end_row=_end_row) == (tail, 1, False)


@pytest.mark.parametrize("prompt_ids", [1, 3])
def test_the_eos_gate_is_the_same_on_the_host_and_on_the_device(prompt_ids):
    texts = StreamedTexts()
    states = {"open": ([5, 6, 7, 8], False), "ended": ([5, 6, 7, 8], True), "short": ([5], True)}
    for req_id, (ids, ended) in states.items():
        texts.receive({req_id: _delivery(0, ids, ended, prompt_ids)})
    req_ids = [*states, "plain"]
    for frame in range(8):
        frames = [frame] * len(req_ids)
        host = texts.forbid_early_eos(torch.zeros(len(req_ids), 16), req_ids, frames, CODEC_EOS)
        device = texts.forbid_early_eos(torch.zeros(len(req_ids), 16), req_ids, torch.tensor(frames), CODEC_EOS)
        assert torch.equal(host, device)
        for row, req_id in enumerate(req_ids):
            banned = bool(torch.isinf(host[row, CODEC_EOS]))
            if req_id == "plain":
                assert not banned
            else:
                ids, ended = states[req_id]
                assert banned == (not end_read(prompt_ids, frame, len(ids), ended))
            # The gate touches codec EOS only.
            assert not torch.isinf(torch.cat([host[row, :CODEC_EOS], host[row, CODEC_EOS + 1 :]])).any()


# -------------------- the prompt layout --------------------


def _builder() -> Qwen3TTSPromptEmbedsBuilder:
    torch.manual_seed(0)
    config = Qwen3TTSConfig(tts_bos_token_id=50, tts_eos_token_id=51, tts_pad_token_id=52)
    talker_config = Qwen3TTSTalkerConfig(
        codec_nothink_id=10,
        codec_think_id=11,
        codec_think_bos_id=12,
        codec_think_eos_id=13,
        codec_pad_id=CODEC_PAD,
        codec_bos_id=CODEC_BOS,
        codec_language_id={"english": 16},
        spk_id={"vivian": 17},
        num_code_groups=4,
    )
    projection = torch.nn.Sequential(torch.nn.Linear(32, 40), torch.nn.SiLU(), torch.nn.Linear(40, HIDDEN))
    builder = Qwen3TTSPromptEmbedsBuilder(
        config=config,
        talker_config=talker_config,
        model_path="",
        text_embedding=torch.nn.Embedding(200, 32),
        text_projection=projection,
        codec_embed=torch.nn.Embedding(64, HIDDEN),
        residual_code_embeddings=lambda: [],
        speaker_encoder=torch.nn.Identity(),
        tts_pad_embed=torch.randn(1, HIDDEN),
        encode_ref_audio_batch=lambda *args, **kwargs: [],
    )
    builder._text_tokenizer = lambda *args, **kwargs: {"input_ids": torch.tensor([ASSISTANT_IDS])}
    return builder


def _custom_voice(**overrides):
    info = {
        "text": ["hello"],
        "task_type": ["CustomVoice"],
        "language": ["English"],
        "speaker": ["vivian"],
        "instruct": [""],
        "non_streaming_mode": [False],
        PRECOMPUTED_TEXT_IDS_KEY: [ASSISTANT_IDS],
    }
    info.update(overrides)
    return info


def _build(builder, info, streamed=None):
    with torch.inference_mode():
        prompt, trailing, _, _ = builder.build_prompt_embeds(
            task_type="CustomVoice", info_dict=dict(info), streamed_text=streamed
        )
    return prompt, trailing


def _project(builder, ids):
    with torch.inference_mode():
        return builder._text_projection(builder._text_embedding(torch.tensor([ids], dtype=torch.long)))[0]


def test_a_lead_of_zero_is_the_stock_streaming_layout_row_for_row():
    builder = _builder()
    stock = _build(builder, _custom_voice())
    streamed = _build(builder, _custom_voice(text=[""]), (torch.tensor([TEXT]), True))
    assert torch.equal(stock[0], streamed[0]) and torch.equal(stock[1], streamed[1])


@pytest.mark.parametrize("lead", [1, 2, 5])
def test_a_text_lead_places_the_first_ids_before_codec_bos(lead):
    builder = _builder()
    stock_prompt, _ = _build(builder, _custom_voice())
    prompt, trailing = _build(builder, _custom_voice(text_lead=[lead]))
    prefix = stock_prompt.shape[0] - 1
    assert prompt.shape[0] == stock_prompt.shape[0] + lead
    assert torch.equal(prompt[:prefix], stock_prompt[:prefix])
    codec = builder._codec_embed(torch.tensor([CODEC_PAD, CODEC_BOS])).detach()
    torch.testing.assert_close(prompt[prefix : prefix + lead], _project(builder, TEXT[:lead]) + codec[0])
    torch.testing.assert_close(prompt[prefix + lead], _project(builder, TEXT[lead : lead + 1])[0] + codec[1])
    end_row = builder.text_end_row(torch.device("cpu"))[0]
    torch.testing.assert_close(trailing, torch.cat([_project(builder, TEXT[lead + 1 :]), end_row]))


def test_a_prompt_length_estimate_counts_the_lead():
    def tokenize(_text):
        return ASSISTANT_IDS

    for lead in (0, 3):
        info = _custom_voice(text_lead=[lead])
        estimate = Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
            additional_information=info,
            task_type="CustomVoice",
            tokenize_prompt=tokenize,
            codec_language_id={"english": 16},
            spk_is_dialect=None,
        )
        assert estimate == _build(_builder(), info)[0].shape[0]


def test_an_open_stream_trails_no_end_and_a_short_ended_text_fills_the_lead():
    builder = _builder()
    # Lead 1 with two ids so far: id 0 leads, id 1 sits on BOS, nothing trails yet.
    prompt, trailing = _build(builder, _custom_voice(text=[""], text_lead=[1]), (torch.tensor([[20, 21]]), False))
    assert trailing.shape[0] == 0
    # Lead 3 on a text that ended after two ids: the lead rows hold both ids and the END, BOS sits on PAD.
    prompt, trailing = _build(builder, _custom_voice(text=[""], text_lead=[3]), (torch.tensor([[20, 21]]), True))
    prefix = prompt.shape[0] - 4
    codec = builder._codec_embed(torch.tensor([CODEC_PAD, CODEC_BOS])).detach()
    end_row = builder.text_end_row(CPU)[0, 0]
    torch.testing.assert_close(prompt[prefix + 2], end_row + codec[0])
    torch.testing.assert_close(prompt[prefix + 3], builder._tts_pad_embed_buffer[0] + codec[1])
    with pytest.raises(ValueError, match="needs"):
        _build(builder, _custom_voice(text=[""], text_lead=[4]), (torch.tensor([[20, 21]]), True))


# -------------------- the Talker --------------------


def _talker(builder=None):
    model = Qwen3TTSTalkerForConditionalGeneration.__new__(Qwen3TTSTalkerForConditionalGeneration)
    model.talker_config = Qwen3TTSTalkerConfig(codec_pad_id=CODEC_PAD, num_code_groups=4)
    model._codebook_vocab_size = 2048
    model._embedding_dtype = torch.float32
    model._tts_pad_embed = torch.full((1, HIDDEN), -1.0)
    model._prompt_builder = builder if builder is not None else _builder()
    model._codec_eos_token_id = CODEC_EOS
    model._silence_ban_frames = 0
    model._req_x_vector_only = {}
    model._batch_req_ids = []
    return model


def _decode(model, info):
    with torch.inference_mode():
        _, _, _, text_step, [update] = model.preprocess_decode_batch(
            input_ids=torch.tensor([1]), req_infos=[info], input_embeds=torch.zeros(1, HIDDEN)
        )
    info["meta"] = {**info["meta"], **update["meta"]}
    if "hidden_states" in update:
        info["hidden_states"] = {**info["hidden_states"], **update["hidden_states"]}
    return text_step[0]


def test_batched_decode_takes_a_streamed_request_without_trailing_text_as_the_single_path_does():
    # preprocess() lets a streamed request through before it has any text conditioning; the batched decode path
    # mirrors its decode branch, so it must too, and read the request's next rows from its stream.
    builder = _builder()
    model = _talker(builder)
    model.append_streamed_text({"r": _delivery(0, [20, 21])})
    model._streamed_texts.prompt_text("r", torch.device("cpu"))  # the prefill takes ids 0 and 1
    model.append_streamed_text({"r": _delivery(2, [22])})
    info = {
        "req_id": "r",
        "text": [""],
        "task_type": ["CustomVoice"],
        "hidden_states": {"last": torch.zeros(HIDDEN)},
        "meta": {"talker_text_offset": 0},
    }
    torch.testing.assert_close(_decode(model, info), _project(builder, [22])[0])


def test_the_talker_reads_streamed_text_in_order_across_appends_then_its_end():
    builder = _builder()
    model = _talker(builder)
    model.append_streamed_text({"r": _delivery(0, [20, 21])})
    ids, ended = model._streamed_texts.prompt_text("r", torch.device("cpu"))  # the prefill takes ids 0 and 1
    tail = _project(builder, [21])  # lead 0: id 1 trails the BOS row
    info = {
        "req_id": "r",
        "text": [""],
        "task_type": ["CustomVoice"],
        "hidden_states": {"trailing_text": tail, "last": torch.zeros(HIDDEN)},
        "meta": {"talker_text_offset": 0},
    }
    torch.testing.assert_close(_decode(model, info), _project(builder, [21])[0])
    model.append_streamed_text({"r": _delivery(2, [22, 23])})
    torch.testing.assert_close(_decode(model, info), _project(builder, [22, 23])[0])
    torch.testing.assert_close(_decode(model, info), _project(builder, [22, 23])[1])
    model.append_streamed_text({"r": _delivery(4, [], ended=True)})
    torch.testing.assert_close(_decode(model, info), builder.text_end_row(torch.device("cpu"))[0, 0])
    torch.testing.assert_close(_decode(model, info), model._tts_pad_embed[0])

    model.on_requests_finished({"r"})
    assert "r" not in model._streamed_texts


def test_a_streamed_request_builds_its_prompt_from_the_ids_it_has_received():
    seen = {}

    def build_prompt_embeds(*, task_type, info_dict, streamed_text):
        seen["streamed"] = streamed_text
        return torch.zeros(6, HIDDEN), torch.zeros(0, HIDDEN), None, None

    model = _talker()
    model._prompt_builder = create_autospec(Qwen3TTSPromptEmbedsBuilder, instance=True)
    model._prompt_builder.build_prompt_embeds.side_effect = build_prompt_embeds
    model.append_streamed_text({"r": _delivery(0, [20, 21], prompt_ids=2)})
    with torch.inference_mode():
        model.preprocess(
            input_ids=torch.zeros(6, dtype=torch.long),
            input_embeds=None,
            req_id="r",
            text=[""],
            task_type=["CustomVoice"],
            _omni_is_prefill=True,
            _omni_num_computed_tokens=0,
            _omni_prompt_len=6,
        )
    ids, ended = seen["streamed"]
    assert ids.tolist() == [[20, 21]] and ended is False


def test_an_empty_queue_of_an_open_stream_is_not_settled():
    """MRV2 skips the decode preprocess of settled rows; a streamed request settles only after its END."""
    model = _talker()
    spent = {"req_id": "r", "hidden_states": {"trailing_text": torch.zeros(0, HIDDEN)}, "meta": {}}
    assert model.eager_decode_settled({**spent, "req_id": "plain"})
    model.append_streamed_text({"r": _delivery(0, [20])})
    model._streamed_texts.prompt_text("r", torch.device("cpu"))
    assert not model.eager_decode_settled(spent)
    model.append_streamed_text({"r": _delivery(1, [], ended=True)})
    assert not model.eager_decode_settled(spent)  # the END has not reached the queue yet
    model._streamed_texts.extend_queue("r", torch.zeros(0, 2), 0, device=CPU, project=_rows, end_row=_end_row)
    assert model.eager_decode_settled(spent)


def _sampling_metadata(output_token_ids: list[list[int]]) -> SamplingMetadata:
    metadata = create_autospec(SamplingMetadata, instance=True)
    metadata.output_token_ids = output_token_ids
    return metadata


def test_the_eos_gate_reads_the_frame_counts_of_both_runners():
    model = _talker()
    model.append_streamed_text({"r": _delivery(0, [20, 21], ended=True)})
    model.set_batch_req_ids(["plain", "r"])
    # v1: sampling metadata carries each row's sampled tokens.
    early = model._forbid_early_eos(torch.zeros(2, 16), _sampling_metadata([[1], [1]]))
    late = model._forbid_early_eos(torch.zeros(2, 16), _sampling_metadata([[1], [1, 1]]))
    assert torch.isinf(early[1, CODEC_EOS]) and not torch.isinf(early[0, CODEC_EOS])
    assert not torch.isinf(late[1, CODEC_EOS])
    # MRV2: the per-row counts stay on the device inside the sampling context.
    with model.mrv2_sampling_context(req_ids=["plain", "r"], num_output_tokens=torch.tensor([1, 1])):
        mrv2 = model._forbid_early_eos(torch.zeros(2, 16), None)
    assert torch.equal(mrv2, early)
    assert model._mrv2_num_output_tokens is None
