# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections import OrderedDict, UserDict
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.qwen3_tts.input_stream import Qwen3TTSTextInput
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    NORMALIZED_REF_AUDIO_KEY,
    PRECOMPUTED_REF_CODE_KEY,
    PRECOMPUTED_REF_IDS_KEY,
    PRECOMPUTED_TEXT_IDS_KEY,
    REF_AUDIO_CACHE_KEY,
    Qwen3TTSPromptEmbedsBuilder,
)
from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
    Qwen3TTSTalkerForConditionalGeneration,
)
from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("capture_sizes,expected", [(None, [1, 8]), ([2, 4, 16], [1, 2, 4, 8])])
def test_talker_initializes_predictor_buckets_without_cuda_graphs(mocker, capture_sizes, expected):
    from vllm_omni.model_executor.models.qwen3_tts import qwen3_tts_talker as module

    # Stop at the predictor boundary, before the constructor loads tokenizers.
    class BucketsConfiguredError(Exception):
        pass

    predictor = mocker.Mock()
    predictor.configure_mtp_execution_buckets.side_effect = BucketsConfiguredError
    predictor_cls = mocker.patch.object(
        module, "Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM", return_value=predictor
    )
    predictor_cls._stage_connector_extra_config.return_value = {}
    mocker.patch.object(module, "Qwen3Model")
    mocker.patch.object(module, "get_pp_group", return_value=SimpleNamespace(is_last_rank=False))
    mocker.patch.object(module, "ECAPATimeDelayNet")
    mocker.patch("vllm.config.vllm.set_current_vllm_config", return_value=nullcontext())
    talker_config = SimpleNamespace(
        code_predictor_config=SimpleNamespace(vocab_size=16),
        hidden_size=8,
        text_hidden_size=8,
        text_vocab_size=16,
        vocab_size=16,
        num_code_groups=4,
        hidden_act="silu",
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="unused",
            hf_config=SimpleNamespace(talker_config=talker_config, speaker_encoder_config=None),
            use_v2_model_runner=True,
        ),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=capture_sizes),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        parallel_config=SimpleNamespace(),
    )
    with pytest.raises(BucketsConfiguredError):
        module.Qwen3TTSTalkerForConditionalGeneration(vllm_config=config)
    predictor.configure_mtp_execution_buckets.assert_called_once_with(expected)


def _make_minimal_talker(
    tts_pad_embed: torch.Tensor | None = None,
    *,
    build_prompt_embeds=None,
):
    """Construct a bare ``Qwen3TTSTalkerForConditionalGeneration`` for preprocess tests.

    Skips ``__init__`` and ``load_weights`` (so we don't need real config /
    checkpoint files) and pre-populates the minimum set of attributes the
    preprocess / decode paths read: the persistent ``tts_pad`` buffer and a
    stub :class:`Qwen3TTSPromptEmbedsBuilder` with a configurable
    ``build_prompt_embeds`` (defaults to raising, since most preprocess
    tests don't exercise the builder).
    """
    model = Qwen3TTSTalkerForConditionalGeneration.__new__(Qwen3TTSTalkerForConditionalGeneration)
    model.talker_config = SimpleNamespace(codec_pad_id=7, num_code_groups=16)
    model._codebook_vocab_size = 2048
    model._embedding_dtype = torch.bfloat16
    if tts_pad_embed is None:
        tts_pad_embed = torch.zeros((1, 4), dtype=torch.bfloat16)
    model._tts_pad_embed = tts_pad_embed

    def _default_raise(**_kwargs):
        raise AssertionError("build_prompt_embeds was not stubbed in this test")

    model._embedding_dtype = torch.bfloat16
    model._prompt_builder = SimpleNamespace(
        build_prompt_embeds=build_prompt_embeds if build_prompt_embeds is not None else _default_raise,
    )
    return model


def test_postprocess_batch_gathers_each_request_tail():
    from vllm_omni.model_executor.models.output_templates import OwnedBatchTensor

    model = _make_minimal_talker()
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    key, values = model.postprocess_batch_mrv2(
        hidden_states=hidden,
        last_token_indices=torch.tensor([1, 3]),
    )

    assert key == ("hidden_states", "last")
    assert isinstance(values, OwnedBatchTensor)  # freshly allocated rows transfer ownership
    assert torch.equal(values.tensor, torch.tensor([[3.0, 4.0], [7.0, 8.0]]))


def _make_minimal_builder(
    *,
    config: SimpleNamespace | None = None,
    talker_config: SimpleNamespace | None = None,
    tts_pad_embed: torch.Tensor | None = None,
):
    """Construct a bare :class:`Qwen3TTSPromptEmbedsBuilder` instance.

    Bypasses ``__init__`` so we don't need a real model checkpoint; tests
    populate the attributes they actually exercise.
    """
    builder = Qwen3TTSPromptEmbedsBuilder.__new__(Qwen3TTSPromptEmbedsBuilder)
    builder._config = config or SimpleNamespace(
        tts_bos_token_id=100,
        tts_eos_token_id=101,
        tts_pad_token_id=102,
    )
    builder._talker_config = talker_config or SimpleNamespace(
        codec_nothink_id=10,
        codec_think_bos_id=11,
        codec_think_eos_id=12,
        codec_think_id=13,
        codec_language_id={},
        codec_pad_id=7,
        codec_bos_id=8,
        num_code_groups=2,
        spk_is_dialect={},
        hidden_size=4,
    )
    builder._model_path = ""
    builder._text_embedding = None
    builder._text_projection = lambda x: x
    builder._codec_embed = lambda ids: torch.zeros((*ids.shape, 4), device=ids.device)
    builder._residual_code_embeddings = lambda: []
    builder._speaker_encoder = None
    builder._embedding_dtype = torch.bfloat16
    builder._tts_pad_embed_buffer = (
        tts_pad_embed if tts_pad_embed is not None else torch.zeros((1, 4), dtype=torch.bfloat16)
    )
    builder._encode_ref_audio_batch_fn = lambda wavs, sr, *, device: (_ for _ in ()).throw(
        AssertionError("encode_ref_audio_batch_fn not stubbed")
    )
    builder._speaker_cache = None
    builder._text_tokenizer = None
    builder._projected_token_cache = {}
    builder._embedding_dtype = torch.bfloat16
    builder._ref_audio_artifact_cache_max_entries = 256
    builder._ref_audio_artifact_cache = OrderedDict()
    builder._resampler_cache = OrderedDict()
    builder._resampler_cache_max = 16
    return builder


def _make_custom_voice_builder():
    builder = _make_minimal_builder()
    builder._device = lambda: torch.device("cpu")
    builder._text_embedding = lambda ids: ids.to(torch.float32).unsqueeze(-1).expand(*ids.shape, 4)
    builder._codec_embed = builder._text_embedding
    builder._text_tokenizer = lambda *_args, **_kwargs: {"input_ids": torch.tensor([[1, 2, 3, 21, 22, 4, 5, 6, 7, 8]])}
    builder._talker_config.spk_id = {"vivian": 20}
    builder._speaker_embed_cache = {}
    builder._custom_voice_prompt_cache = {}
    builder._batched_text_embeds = {}
    builder._projected_text_table = None
    return builder


@pytest.mark.parametrize("input_complete", [False, True])
@pytest.mark.parametrize("body_ids", [[21], [21, 22, 23]])
def test_custom_voice_open_text_prefill_only_ends_text_after_eof(input_complete: bool, body_ids: list[int]) -> None:
    """Keep packet boundaries out of the model's text EOS sequence."""
    builder = _make_custom_voice_builder()
    wrapped_ids = torch.tensor([[1, 2, 3, *body_ids, 4, 5, 6, 7, 8]])
    prompt, trailing, ref_length, ref_codes = builder.build_prompt_embeds(
        task_type="CustomVoice",
        info_dict={
            "text": ["hello"],
            "speaker": ["vivian"],
            "input_stream": {"token_ids": body_ids, "final": input_complete},
            PRECOMPUTED_TEXT_IDS_KEY: wrapped_ids,
        },
    )

    expected_tail = body_ids[1:] + ([101] if input_complete else [])
    assert torch.equal(prompt[-1], torch.full((4,), body_ids[0] + 8.0))
    assert torch.equal(trailing, torch.tensor(expected_tail).reshape(-1, 1).expand(-1, 4))
    assert ref_length is None and ref_codes is None


@pytest.mark.parametrize("input_complete", [False, True])
def test_streaming_text_update_uses_body_ids_without_bos(input_complete: bool) -> None:
    builder = _make_custom_voice_builder()
    builder._codec_embed = lambda _ids: pytest.fail("Continuation must not embed codec BOS.")
    rows = builder.build_streaming_text_update([31, 32], input_complete=input_complete)
    expected_ids = [31, 32] + ([101] if input_complete else [])
    assert torch.equal(rows, torch.tensor(expected_ids).reshape(-1, 1).expand(-1, 4))


@pytest.mark.parametrize("projected_table", [False, True])
def test_streaming_text_projection_keeps_token_major_shape(projected_table: bool) -> None:
    builder = _make_custom_voice_builder()
    builder._text_embedding = torch.nn.Embedding(128, 5)
    builder._text_projection = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.SiLU(), torch.nn.Linear(7, 4))
    ids = torch.tensor([31, 32, 101])
    expected = builder._text_projection(builder._text_embedding(ids))
    if projected_table:
        builder._projected_text_table = builder._text_projection(builder._text_embedding.weight)
    result = builder.build_streaming_text_update([31, 32], input_complete=True)
    assert result.shape == (3, 4)
    torch.testing.assert_close(result, expected)


def test_native_initial_and_appended_text_share_the_projected_table():
    builder = _make_custom_voice_builder()
    builder._projected_text_table = (torch.arange(128, dtype=torch.float32) + 0.25).reshape(-1, 1).expand(-1, 4)
    prompt, trailing, _, _ = builder.build_prompt_embeds(
        task_type="CustomVoice",
        info_dict={"speaker": ["vivian"], "input_stream": {"token_ids": [31, 32], "final": True}},
    )
    torch.testing.assert_close(trailing, builder.build_streaming_text_update([32], input_complete=True))
    torch.testing.assert_close(prompt[-1], builder._projected_text_table[31] + 8.0)


@pytest.mark.parametrize("batched", [False, True])
def test_active_text_queue_consumes_body_then_eos_once_and_keeps_the_same_prompt(batched):
    model = _make_minimal_talker()
    model.eager_frames_active = True
    model._prompt_builder = _make_custom_voice_builder()
    model.embed_input_ids = lambda ids: torch.zeros((*ids.shape, 4), dtype=torch.bfloat16)
    buffer = OmniIntermediateBuffer(1)
    runtime: dict[str, Any] = {"req_id": "request", "speaker": ["vivian"]}
    buffer.buffers[0] = runtime
    model.apply_text_input_update(
        runtime, {"input_stream_update": {"token_ids": [31, 32], "final": False}}, replace=True
    )
    expected_prompt, _, _, _ = model._prompt_builder.build_prompt_embeds(
        task_type="CustomVoice", info_dict=dict(runtime)
    )
    _, _, updates = model.preprocess(
        torch.zeros(expected_prompt.shape[0], dtype=torch.long), None, **runtime, _omni_is_prefill=True
    )
    buffer.update(0, updates)
    original_prompt = runtime["embed"]["prefill"]
    assert original_prompt.shape == expected_prompt.shape
    assert torch.equal(original_prompt[-1], torch.full((4,), 39.0))
    assert runtime["meta"]["qwen3_tts_text_projected"] == 2
    model.apply_text_input_update(runtime, {"input_stream_update": {"token_ids": [33], "final": False}})

    def decode(expected):
        if batched:
            _, _, _, text_step, current_updates = model.preprocess_decode_batch(
                input_ids=torch.tensor([1]), req_infos=[runtime]
            )
            current_updates = current_updates[0]
        else:
            _, _, current_updates = model.preprocess(torch.tensor([1]), None, **runtime, _omni_is_prefill=False)
            text_step = current_updates.pop("mtp_inputs")[1]
        buffer.update(0, current_updates)
        assert text_step.tolist() == [[expected] * 4]
        assert runtime["embed"]["prefill"] is original_prompt

    decode(32)
    decode(33)
    assert not model.eager_decode_settled(runtime)
    model.apply_text_input_update(runtime, {"input_stream_update": {"token_ids": [], "final": True}})
    decode(101)
    decode(0)
    assert model.eager_decode_settled(runtime)


@pytest.mark.parametrize("final", [False, True])
def test_reinstalled_text_snapshot_preserves_projection_and_consumption(final):
    from vllm_omni.worker_v2.model_states.mtp_model_state import _snapshot_runtime

    model = _make_minimal_talker()
    model.eager_frames_active = True
    model._prompt_builder = _make_custom_voice_builder()
    model.embed_input_ids = lambda ids: torch.zeros((*ids.shape, 4), dtype=torch.bfloat16)
    stream = Qwen3TTSTextInput([31, 32, 33], 8, 128, final=final)
    buffer = OmniIntermediateBuffer(1)
    buffer.buffers[0] = {"req_id": "request", "speaker": ["vivian"]}
    model.apply_text_input_update(
        buffer.buffers[0], {"input_stream_update": stream.take_update(reset=True)}, replace=True
    )
    prompt, _, _, _ = model._prompt_builder.build_prompt_embeds(
        task_type="CustomVoice", info_dict=dict(buffer.buffers[0])
    )
    _, _, updates = model.preprocess(
        torch.zeros(len(prompt), dtype=torch.long), None, **buffer.buffers[0], _omni_is_prefill=True
    )
    buffer.update(0, updates)

    def decode(expected):
        _, _, updates = model.preprocess(torch.tensor([1]), None, **buffer.buffers[0], _omni_is_prefill=False)
        assert updates.pop("mtp_inputs")[1].tolist() == [[expected] * 4]
        buffer.update(0, updates)

    decode(32)
    restored = _snapshot_runtime(buffer.buffers[0])
    buffer.buffers[0] = restored
    meta = restored["meta"]
    trailing = restored["hidden_states"]["trailing_text"]
    model.apply_text_input_update(restored, {"input_stream_update": stream.take_update(reset=True)}, replace=True)
    assert restored["meta"] is meta
    assert restored["hidden_states"]["trailing_text"] is trailing
    assert meta["qwen3_tts_text_projected"] == 3
    assert meta["talker_text_offset"] == 1
    assert meta["qwen3_tts_text_eos_projected"] is final
    decode(33)
    if not final:
        stream.append([34], final=True)
        model.apply_text_input_update(restored, {"input_stream_update": stream.take_update(reset=False)})
        decode(34)
    decode(101)
    model.apply_text_input_update(restored, {"input_stream_update": stream.take_update(reset=True)}, replace=True)
    decode(0)


def test_text_appended_during_chunked_prefill_is_consumed_only_after_prefill():
    model = _make_minimal_talker()
    model.eager_frames_active = True
    model._prompt_builder = _make_custom_voice_builder()
    model.embed_input_ids = lambda ids: torch.zeros((*ids.shape, 4), dtype=torch.bfloat16)
    buffer = OmniIntermediateBuffer(1)
    runtime: dict[str, Any] = {"req_id": "request", "speaker": ["vivian"]}
    buffer.buffers[0] = runtime
    model.apply_text_input_update(
        runtime, {"input_stream_update": {"token_ids": [31, 32], "final": False}}, replace=True
    )
    _, first_chunk, updates = model.preprocess(
        torch.zeros(2, dtype=torch.long), None, **runtime, _omni_is_prefill=True, _omni_num_computed_tokens=0
    )
    buffer.update(0, updates)
    prompt = runtime["embed"]["prefill"]
    model.apply_text_input_update(runtime, {"input_stream_update": {"token_ids": [33], "final": True}})
    _, last_chunk, updates = model.preprocess(
        torch.zeros(len(prompt) - 2, dtype=torch.long), None, **runtime, _omni_is_prefill=True
    )
    buffer.update(0, updates)
    torch.testing.assert_close(first_chunk, prompt[:2].to(dtype=model._embedding_dtype))
    torch.testing.assert_close(last_chunk, prompt[2:].to(dtype=model._embedding_dtype))
    assert runtime["meta"]["qwen3_tts_text_projected"] == 2
    assert not runtime["meta"]["qwen3_tts_text_eos_projected"]
    for expected in (32, 33, 101):
        _, _, updates = model.preprocess(torch.tensor([1]), None, **runtime, _omni_is_prefill=False)
        assert updates.pop("mtp_inputs")[1].tolist() == [[expected] * 4]
        buffer.update(0, updates)
    assert runtime["embed"]["prefill"] is prompt
    assert runtime["meta"]["qwen3_tts_text_eos_projected"]


@pytest.mark.parametrize("channel", ["additional", "native"])
def test_text_updates_do_not_mutate_the_initial_request_payload(channel):
    model = _make_minimal_talker()
    initial = {"token_ids": [31], "final": False}
    request = SimpleNamespace(
        req_id="request",
        mm_features=[],
        additional_information={"input_stream": initial} if channel == "additional" else None,
        model_intermediate_buffer={"input_stream": initial} if channel == "native" else None,
    )
    buffer = OmniIntermediateBuffer(1)
    buffer.add_request(0, request)
    runtime = buffer.buffers[0]
    snapshot = {"token_ids": [31], "final": False}
    model.apply_text_input_update(runtime, {"input_stream_update": snapshot}, replace=True)
    model.apply_text_input_update(runtime, {"input_stream_update": {"token_ids": [32], "final": True}})
    assert initial == {"token_ids": [31], "final": False}
    assert snapshot == {"token_ids": [31], "final": False}
    assert runtime["input_stream"] == {"token_ids": [31, 32], "final": True}
    assert "input_stream_update" not in runtime


def test_single_token_prefill_uses_prefill_path():
    full_prompt_embeds = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    trailing_text = torch.ones((2, 4), dtype=torch.float32)
    ref_code = torch.arange(32, dtype=torch.long).reshape(2, 16)

    def fake_build_prompt_embeds(*, task_type, info_dict):
        return full_prompt_embeds, trailing_text, 2, ref_code

    model = _make_minimal_talker(build_prompt_embeds=fake_build_prompt_embeds)

    input_ids = torch.tensor([123], dtype=torch.long)
    out_ids, out_embeds, update = model.preprocess(
        input_ids=input_ids,
        input_embeds=None,
        text=["hello"],
        task_type=["CustomVoice"],
        _omni_is_prefill=True,
        _omni_num_computed_tokens=0,
        _omni_prompt_len=3,
    )

    assert out_ids.tolist() == [7]
    assert torch.equal(out_embeds.cpu(), full_prompt_embeds[:1].to(torch.bfloat16))
    assert update["meta"]["talker_prefill_offset"] == 1
    assert update["meta"]["talker_text_offset"] == 0
    assert update["meta"]["ref_code_len"] == 2
    assert torch.equal(update["embed"]["prefill"], full_prompt_embeds)
    # ``tts_pad`` is no longer round-tripped through ``info_dict`` — it is
    # precomputed once into ``model._tts_pad_embed`` at load time.
    assert "tts_pad" not in update["embed"]
    assert torch.equal(update["hidden_states"]["trailing_text"], trailing_text)
    assert torch.equal(update["codes"]["ref"], ref_code)
    assert update["codes"]["audio"].shape == (1, 16)


def test_single_token_prefill_can_be_inferred_from_token_progress():
    full_prompt_embeds = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    trailing_text = torch.ones((1, 4), dtype=torch.float32)

    def fake_build_prompt_embeds(*, task_type, info_dict):
        return full_prompt_embeds, trailing_text, None, None

    model = _make_minimal_talker(build_prompt_embeds=fake_build_prompt_embeds)

    out_ids, out_embeds, update = model.preprocess(
        input_ids=torch.tensor([123], dtype=torch.long),
        input_embeds=None,
        text=["hello"],
        task_type=["CustomVoice"],
        _omni_num_computed_tokens=0,
        _omni_prompt_len=2,
    )

    assert out_ids.tolist() == [7]
    assert torch.equal(out_embeds.cpu(), full_prompt_embeds[:1].to(torch.bfloat16))
    assert update["meta"]["talker_prefill_offset"] == 1


def test_decode_advances_trailing_text_by_offset_without_rewriting_tail():
    tts_pad = torch.full((1, 4), -1.0, dtype=torch.bfloat16)
    model = _make_minimal_talker(tts_pad_embed=tts_pad)

    def fake_embed_input_ids(input_ids):
        return input_ids.to(torch.float32).reshape(1, 1, 1).expand(1, 1, 4)

    model.embed_input_ids = fake_embed_input_ids
    trailing_text = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    last_hidden = torch.full((4,), 2.0, dtype=torch.float32)

    out_ids, out_embeds, update = model.preprocess(
        input_ids=torch.tensor([123], dtype=torch.long),
        input_embeds=None,
        text=["hello"],
        task_type=["CustomVoice"],
        hidden_states={"trailing_text": trailing_text, "last": last_hidden},
        meta={"talker_text_offset": 1},
        _omni_is_prefill=False,
        _omni_num_computed_tokens=2,
        _omni_prompt_len=2,
    )

    assert out_ids.tolist() == [123]
    assert torch.equal(out_embeds.cpu(), torch.full((1, 4), 123.0, dtype=torch.bfloat16))
    assert "hidden_states" not in update
    assert update["meta"]["talker_text_offset"] == 2
    past_hidden, text_step = update["mtp_inputs"]
    assert torch.equal(past_hidden.cpu(), last_hidden.reshape(1, -1).to(torch.bfloat16))
    assert torch.equal(text_step.cpu(), trailing_text[1:2].to(torch.bfloat16))


def test_decode_advances_trailing_text_offset_across_multiple_steps():
    tts_pad = torch.full((1, 4), -1.0, dtype=torch.bfloat16)
    model = _make_minimal_talker(tts_pad_embed=tts_pad)

    def fake_embed_input_ids(input_ids):
        return input_ids.to(torch.float32).reshape(1, 1, 1).expand(1, 1, 4)

    model.embed_input_ids = fake_embed_input_ids
    trailing_text = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    state_tail = trailing_text
    last_hidden = torch.full((4,), 2.0, dtype=torch.float32)
    meta = {"talker_text_offset": 0}
    seen_steps = []

    for _ in range(3):
        _, _, update = model.preprocess(
            input_ids=torch.tensor([123], dtype=torch.long),
            input_embeds=None,
            text=["hello"],
            task_type=["CustomVoice"],
            hidden_states={"trailing_text": state_tail, "last": last_hidden},
            meta=meta,
            _omni_is_prefill=False,
            _omni_num_computed_tokens=2,
            _omni_prompt_len=2,
        )
        seen_steps.append(update["mtp_inputs"][1].cpu())
        if "hidden_states" in update and "trailing_text" in update["hidden_states"]:
            state_tail = update["hidden_states"]["trailing_text"]
        meta = update["meta"]

    assert torch.equal(seen_steps[0], trailing_text[0:1].to(torch.bfloat16))
    assert torch.equal(seen_steps[1], trailing_text[1:2].to(torch.bfloat16))
    assert torch.equal(seen_steps[2], tts_pad.to(torch.bfloat16))
    assert meta["talker_text_offset"] == 0
    assert state_tail.numel() == 0


def test_decode_compacts_long_trailing_text_after_large_offset():
    tts_pad = torch.full((1, 4), -1.0, dtype=torch.bfloat16)
    model = _make_minimal_talker(tts_pad_embed=tts_pad)

    def fake_embed_input_ids(input_ids):
        return input_ids.to(torch.float32).reshape(1, 1, 1).expand(1, 1, 4)

    model.embed_input_ids = fake_embed_input_ids
    trailing_text = torch.arange(130 * 4, dtype=torch.float32).reshape(130, 4)
    last_hidden = torch.full((4,), 2.0, dtype=torch.float32)

    _, _, update = model.preprocess(
        input_ids=torch.tensor([123], dtype=torch.long),
        input_embeds=None,
        text=["hello"],
        task_type=["CustomVoice"],
        hidden_states={"trailing_text": trailing_text, "last": last_hidden},
        meta={"talker_text_offset": 64},
        _omni_is_prefill=False,
        _omni_num_computed_tokens=2,
        _omni_prompt_len=2,
    )

    assert torch.equal(update["mtp_inputs"][1].cpu(), trailing_text[64:65].to(torch.bfloat16))
    assert update["meta"]["talker_text_offset"] == 0
    assert torch.equal(update["hidden_states"]["trailing_text"], trailing_text[65:])


def test_decode_marks_codec_frame_validity_from_processed_input_token():
    model = _make_minimal_talker()
    model.embed_input_ids = lambda input_ids: input_ids.to(torch.float32).reshape(1, 1, 1).expand(1, 1, 4)
    common = {
        "input_embeds": None,
        "text": ["hello"],
        "task_type": ["CustomVoice"],
        "hidden_states": {
            "trailing_text": torch.ones((2, 4)),
            "last": torch.ones(4),
        },
        "meta": {"talker_text_offset": 0},
        "_omni_is_prefill": False,
        "_omni_num_computed_tokens": 2,
        "_omni_prompt_len": 2,
    }

    _, _, valid_update = model.preprocess(input_ids=torch.tensor([123]), **common)
    _, _, eos_update = model.preprocess(input_ids=torch.tensor([4198]), **common)

    assert valid_update["meta"]["codec_frame_valid"].item() is True
    assert eos_update["meta"]["codec_frame_valid"].item() is False


def test_decode_replay_span_embeds_all_tokens_without_mutating_decode_state():
    model = _make_minimal_talker()

    def fake_embed_input_ids(input_ids):
        return input_ids.to(torch.float32).reshape(-1, 1, 1).expand(-1, 1, 4)

    model.embed_input_ids = fake_embed_input_ids
    trailing_text = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    last_hidden = torch.full((4,), 2.0, dtype=torch.float32)

    out_ids, out_embeds, update = model.preprocess(
        input_ids=torch.tensor([101, 202, 303], dtype=torch.long),
        input_embeds=None,
        text=["hello"],
        task_type=["Base"],
        hidden_states={"trailing_text": trailing_text, "last": last_hidden},
        meta={"talker_text_offset": 1, "codec_streaming": True},
        _omni_is_prefill=False,
        _omni_num_computed_tokens=2400,
        _omni_prompt_len=213,
    )

    assert out_ids.tolist() == [101, 202, 303]
    assert torch.equal(
        out_embeds.cpu(),
        torch.tensor([[101.0] * 4, [202.0] * 4, [303.0] * 4], dtype=torch.bfloat16),
    )
    assert update["meta"]["codec_streaming"] is True
    assert update["meta"]["codec_frame_valid"].item() is False


@pytest.mark.parametrize("mrv2", [False, True])
def test_decode_batch_preprocess_matches_decode_state_updates(mrv2):
    tts_pad = torch.full((1, 4), -1.0, dtype=torch.bfloat16)
    model = _make_minimal_talker(tts_pad_embed=tts_pad)

    def fake_embed_input_ids(input_ids):
        assert not mrv2, "MRV2 must reuse the prepared token embeddings"
        return input_ids.to(torch.float32).reshape(-1, 1, 1).expand(-1, 1, 4)

    model.embed_input_ids = fake_embed_input_ids
    trailing_a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    trailing_b = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 100
    last_a = torch.full((4,), 2.0, dtype=torch.float32)
    last_b = torch.full((4,), 3.0, dtype=torch.float32)

    preprocess = model.preprocess_decode_batch_mrv2 if mrv2 else model.preprocess_decode_batch
    prepared = torch.tensor([[101.0] * 4, [202.0] * 4], dtype=torch.bfloat16)
    out_ids, out_embeds, past_hidden, text_step, updates = preprocess(
        **({"input_embeds": prepared} if mrv2 else {}),
        input_ids=torch.tensor([101, 202], dtype=torch.long),
        req_infos=[
            {
                "text": ["hello"],
                "task_type": ["Base"],
                "hidden_states": {"trailing_text": trailing_a, "last": last_a},
                "meta": {"talker_text_offset": 1},
            },
            {
                "text": ["world"],
                "task_type": ["CustomVoice"],
                "hidden_states": {"trailing_text": trailing_b, "last": last_b},
                "meta": {"talker_text_offset": 2},
            },
        ],
    )

    assert out_ids.tolist() == [101, 202]
    if mrv2:
        assert out_embeds.data_ptr() == prepared.data_ptr()
    assert torch.equal(out_embeds.cpu(), torch.tensor([[101.0] * 4, [202.0] * 4], dtype=torch.bfloat16))
    assert torch.equal(past_hidden.cpu(), torch.stack([last_a, last_b]).to(torch.bfloat16))
    assert torch.equal(text_step[0].cpu(), trailing_a[1].to(torch.bfloat16))
    assert torch.equal(text_step[1].cpu(), tts_pad.reshape(-1).to(torch.bfloat16))
    assert updates[0]["meta"]["talker_text_offset"] == 2
    assert updates[0]["meta"]["codec_streaming"] is True
    assert updates[0]["meta"]["codec_frame_valid"].item() is True
    assert "hidden_states" not in updates[0]
    assert updates[1]["meta"]["talker_text_offset"] == 0
    assert updates[1]["meta"]["codec_streaming"] is False
    assert updates[1]["meta"]["codec_frame_valid"].item() is True
    assert updates[1]["hidden_states"]["trailing_text"].numel() == 0


def test_decode_batch_zero_fills_missing_last_hidden(monkeypatch):
    tts_pad = torch.full((1, 4), -1.0, dtype=torch.bfloat16)
    model = _make_minimal_talker(tts_pad_embed=tts_pad)

    def fake_embed_input_ids(input_ids):
        return input_ids.to(torch.float32).reshape(-1, 1, 1).expand(-1, 1, 4)

    model.embed_input_ids = fake_embed_input_ids
    warnings: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker.logger.warning_once",
        lambda *args, **kwargs: warnings.append(args),
    )
    trailing = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    _, _, past_hidden, _, _ = model.preprocess_decode_batch(
        input_ids=torch.tensor([101], dtype=torch.long),
        req_infos=[
            {
                "text": ["hello"],
                "task_type": ["CustomVoice"],
                "hidden_states": {"trailing_text": trailing},
                "meta": {"talker_text_offset": 0},
            }
        ],
    )

    assert torch.equal(past_hidden.cpu(), torch.zeros((1, 4), dtype=torch.bfloat16))
    assert len(warnings) == 1
    assert "zero-filling" in str(warnings[0][0])


def _stub_text_embedding(device_param: torch.nn.Parameter):
    """Build a lambda that emulates ``nn.Embedding`` for the ``_device()`` helper.

    ``Qwen3TTSPromptEmbedsBuilder._device()`` calls
    ``next(self._text_embedding.parameters()).device``, so the stub needs a
    ``parameters()`` attribute *and* must be callable as an embedding.
    """
    fn = lambda ids: torch.ones((*ids.shape, 4), device=ids.device)  # noqa: E731
    fn.parameters = lambda: iter([device_param])  # type: ignore[attr-defined]
    return fn


def test_base_voice_clone_normalizes_ref_audio_once_for_ref_code_and_speaker():
    builder = _make_minimal_builder()
    device_param = torch.nn.Parameter(torch.empty(0))
    builder._text_embedding = _stub_text_embedding(device_param)
    builder._text_projection = lambda embeds: embeds
    builder._codec_embed = lambda ids: torch.zeros((*ids.shape, 4), device=ids.device)

    class FakeTokenizer:
        def __call__(self, *_args, **_kwargs):
            return {"input_ids": torch.arange(8, dtype=torch.long).reshape(1, -1)}

    builder._text_tokenizer = FakeTokenizer()

    builder._generate_icl_prompt = lambda **kwargs: (
        torch.ones((1, 2, 4), device=kwargs["ref_code"].device),
        torch.ones((1, 4), device=kwargs["ref_code"].device),
    )

    normalize_calls = []
    ref_audio = np.arange(1024, dtype=np.float32)
    builder.normalize_ref_audio = lambda raw: normalize_calls.append(raw) or (ref_audio, 16000)

    ref_audio_ids = []
    builder._encode_ref_audio_batch_fn = lambda wavs, sr, *, device: (
        [ref_audio_ids.append(id(w)) or torch.ones((2, 2), dtype=torch.long) for w in wavs]
    )
    builder.extract_speaker_embedding = lambda wav, _sr: (
        ref_audio_ids.append(id(wav)) or torch.ones(4, dtype=torch.bfloat16)
    )

    _prompt, _trailing, ref_code_len, ref_code = builder.build_prompt_embeds(
        task_type="Base",
        info_dict={
            "text": ["hello"],
            "ref_audio": ["ref.wav"],
            "ref_ids": torch.arange(8, dtype=torch.long).reshape(1, -1),
            "non_streaming_mode": [False],
        },
    )

    assert normalize_calls == ["ref.wav"]
    assert ref_audio_ids == [id(ref_audio), id(ref_audio)]
    assert ref_code_len == 2
    assert torch.equal(ref_code, torch.ones((2, 2), dtype=torch.long))


def test_base_voice_clone_rejects_speaker_embedding_dimension_before_cat():
    """Keep a worker-side guard for callers that bypass the Speech API."""
    builder = _make_minimal_builder()
    device_param = torch.nn.Parameter(torch.empty(0))
    builder._text_embedding = _stub_text_embedding(device_param)
    builder._text_projection = lambda embeds: embeds
    builder._codec_embed = lambda ids: torch.zeros((*ids.shape, 4), device=ids.device)

    class FakeTokenizer:
        def __call__(self, *_args, **_kwargs):
            return {"input_ids": torch.arange(8, dtype=torch.long).reshape(1, -1)}

    builder._text_tokenizer = FakeTokenizer()
    builder.normalize_ref_audio = lambda _raw: (np.arange(1024, dtype=np.float32), 16000)
    builder.extract_speaker_embedding = lambda _wav, _sr: torch.ones(3, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="got 3, but the talker requires 4"):
        builder.build_prompt_embeds(
            task_type="Base",
            info_dict={
                "text": ["hello"],
                "ref_audio": ["ref.wav"],
                "x_vector_only_mode": [True],
                "non_streaming_mode": [False],
            },
        )


def test_base_voice_clone_batch_preprocess_encodes_ref_code_by_sample_rate():
    builder = _make_minimal_builder()
    wav1 = np.arange(2048, dtype=np.float32)
    wav2 = np.arange(3072, dtype=np.float32)
    normalize_calls = []
    builder.normalize_ref_audio = lambda raw: (
        normalize_calls.append(raw)
        or (
            wav1 if raw == "a.wav" else wav2,
            16000,
        )
    )

    encode_calls = []

    def _fake_encode_batch(wavs, sr, *, device):
        encode_calls.append((wavs, sr))
        return [
            torch.full((2, 2), 11, dtype=torch.long),
            torch.full((3, 2), 22, dtype=torch.long),
        ]

    builder._encode_ref_audio_batch_fn = _fake_encode_batch

    class FakeTextTokenizer:
        def __init__(self):
            self.calls = []

        def __call__(self, texts, *, padding=False):
            self.calls.append((texts, padding))
            return UserDict({"input_ids": [[1, 2, 3], [2, 3, 4, 5]]})

    text_tok = FakeTextTokenizer()
    builder._text_tokenizer = text_tok
    buf: dict[str, dict[str, Any]] = {
        "r1": {
            "task_type": ["Base"],
            "text": ["one"],
            "ref_audio": ["a.wav"],
            "ref_text": ["hello"],
            "x_vector_only_mode": [False],
        },
        "r2": {
            "task_type": ["Base"],
            "text": ["two"],
            "ref_audio": ["b.wav"],
            "ref_text": ["world"],
            "x_vector_only_mode": [False],
        },
    }

    builder.preprocess_batch(
        req_ids=["r1", "r2"],
        model_intermediate_buffer=buf,
        device=torch.device("cpu"),
    )

    assert normalize_calls == ["a.wav", "b.wav"]
    assert len(encode_calls) == 1
    wavs, sr = encode_calls[0]
    assert wavs[0] is wav1
    assert wavs[1] is wav2
    assert sr == 16000
    assert torch.equal(buf["r1"]["codes"][PRECOMPUTED_REF_CODE_KEY], torch.full((2, 2), 11))
    assert torch.equal(buf["r2"]["codes"][PRECOMPUTED_REF_CODE_KEY], torch.full((3, 2), 22))
    assert buf["r1"][NORMALIZED_REF_AUDIO_KEY][0] is wav1
    assert buf["r2"][NORMALIZED_REF_AUDIO_KEY][0] is wav2
    assert len(text_tok.calls) == 2
    assert torch.equal(buf["r1"][PRECOMPUTED_TEXT_IDS_KEY], torch.tensor([1, 2, 3]))
    assert torch.equal(buf["r2"][PRECOMPUTED_TEXT_IDS_KEY], torch.tensor([2, 3, 4, 5]))
    assert torch.equal(buf["r1"][PRECOMPUTED_REF_IDS_KEY], torch.tensor([1, 2, 3]))
    assert torch.equal(buf["r2"][PRECOMPUTED_REF_IDS_KEY], torch.tensor([2, 3, 4, 5]))
    assert (
        buf["r1"][PRECOMPUTED_TEXT_IDS_KEY].untyped_storage().data_ptr()
        == buf["r2"][PRECOMPUTED_TEXT_IDS_KEY].untyped_storage().data_ptr()
    )
    assert (
        buf["r1"][PRECOMPUTED_REF_IDS_KEY].untyped_storage().data_ptr()
        == buf["r2"][PRECOMPUTED_REF_IDS_KEY].untyped_storage().data_ptr()
    )


def test_base_voice_clone_batch_preprocess_reuses_singleton_normalized_audio_without_speech_tokenizer():
    builder = _make_minimal_builder()
    wav = np.arange(2048, dtype=np.float32)
    builder.normalize_ref_audio = lambda raw: (wav, 16000)
    builder._encode_ref_audio_batch_fn = lambda wavs, sr, *, device: (_ for _ in ()).throw(
        AssertionError("singleton should not call encode_ref_audio_batch")
    )

    class FakeTextTokenizer:
        def __call__(self, texts, *, padding=False):
            return {"input_ids": [[7, 8, 9] for _ in texts]}

    builder._text_tokenizer = FakeTextTokenizer()
    buf: dict[str, dict[str, Any]] = {
        "r1": {
            "task_type": ["Base"],
            "text": ["one"],
            "ref_audio": ["a.wav"],
            "ref_text": ["hello"],
            "x_vector_only_mode": [False],
        }
    }

    builder.preprocess_batch(
        req_ids=["r1"],
        model_intermediate_buffer=buf,
        device=torch.device("cpu"),
    )

    assert buf["r1"][NORMALIZED_REF_AUDIO_KEY][0] is wav
    assert torch.equal(buf["r1"][PRECOMPUTED_TEXT_IDS_KEY], torch.tensor([7, 8, 9]))
    assert torch.equal(buf["r1"][PRECOMPUTED_REF_IDS_KEY], torch.tensor([7, 8, 9]))


def test_base_voice_clone_batch_preprocess_skips_after_initial_prefill_state_exists():
    builder = _make_minimal_builder()
    builder.normalize_ref_audio = lambda _raw: (_ for _ in ()).throw(AssertionError("normalize not expected"))

    class _RaisingTokenizer:
        def __call__(self, *_args, **_kwargs):
            raise AssertionError("tokenizer not expected")

    builder._text_tokenizer = _RaisingTokenizer()
    builder._encode_ref_audio_batch_fn = lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("speech tokenizer not expected")
    )
    buf: dict[str, dict[str, Any]] = {
        "r1": {
            "task_type": ["Base"],
            "text": ["one"],
            "ref_audio": ["a.wav"],
            "ref_text": ["hello"],
            "x_vector_only_mode": [False],
            "embed": {"prefill": torch.ones((1, 4))},
        }
    }

    builder.preprocess_batch(
        req_ids=["r1"],
        model_intermediate_buffer=buf,
        device=torch.device("cpu"),
    )

    assert PRECOMPUTED_TEXT_IDS_KEY not in buf["r1"]
    assert NORMALIZED_REF_AUDIO_KEY not in buf["r1"]


def test_base_voice_clone_batch_preprocess_uses_serving_artifact_cache_key_without_normalize():
    builder = _make_minimal_builder()
    ref_code = torch.full((2, 2), 5, dtype=torch.long)
    builder.put_ref_audio_artifacts("same-ref", ref_code=ref_code)
    builder.normalize_ref_audio = lambda _raw: (_ for _ in ()).throw(AssertionError("normalize not expected"))
    builder._encode_ref_audio_batch_fn = lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("speech tokenizer not expected")
    )

    class FakeTextTokenizer:
        def __call__(self, texts, *, padding=False):
            return {"input_ids": [[7, 8, 9] for _ in texts]}

    builder._text_tokenizer = FakeTextTokenizer()
    buf: dict[str, dict[str, Any]] = {
        "r1": {
            "task_type": ["Base"],
            "text": ["one"],
            "ref_audio": ["a.wav"],
            "ref_text": ["hello"],
            "x_vector_only_mode": [False],
            REF_AUDIO_CACHE_KEY: ["same-ref"],
        }
    }

    builder.preprocess_batch(
        req_ids=["r1"],
        model_intermediate_buffer=buf,
        device=torch.device("cpu"),
    )

    assert torch.equal(buf["r1"]["codes"][PRECOMPUTED_REF_CODE_KEY], ref_code)
    assert NORMALIZED_REF_AUDIO_KEY not in buf["r1"]


def test_base_voice_clone_uses_batched_ref_code_without_serial_encode():
    builder = _make_minimal_builder()
    device_param = torch.nn.Parameter(torch.empty(0))
    builder._text_embedding = _stub_text_embedding(device_param)
    builder._text_projection = lambda embeds: embeds
    builder._codec_embed = lambda ids: torch.zeros((*ids.shape, 4), device=ids.device)

    class FakeTokenizer:
        def __call__(self, *_args, **_kwargs):
            return {"input_ids": torch.arange(8, dtype=torch.long).reshape(1, -1)}

    builder._text_tokenizer = FakeTokenizer()
    builder._generate_icl_prompt = lambda **kwargs: (
        torch.ones((1, 2, 4), device=kwargs["ref_code"].device),
        torch.ones((1, 4), device=kwargs["ref_code"].device),
    )

    ref_audio = np.arange(2048, dtype=np.float32)
    ref_code = torch.arange(4, dtype=torch.long).reshape(2, 2)
    builder.normalize_ref_audio = lambda _raw: (_ for _ in ()).throw(AssertionError("serial normalize not expected"))
    builder.encode_ref_audio_to_code = lambda _wav, _sr: (_ for _ in ()).throw(
        AssertionError("serial encode not expected")
    )
    speaker_wav_ids = []
    builder.extract_speaker_embedding = lambda wav, _sr: (
        speaker_wav_ids.append(id(wav)) or torch.ones(4, dtype=torch.bfloat16)
    )

    _prompt, _trailing, ref_code_len, out_ref_code = builder.build_prompt_embeds(
        task_type="Base",
        info_dict={
            "text": ["hello"],
            "ref_audio": ["ref.wav"],
            "ref_ids": torch.arange(8, dtype=torch.long).reshape(1, -1),
            "non_streaming_mode": [False],
            "codes": {PRECOMPUTED_REF_CODE_KEY: ref_code},
            NORMALIZED_REF_AUDIO_KEY: (ref_audio, 16000),
        },
    )

    assert speaker_wav_ids == [id(ref_audio)]
    assert ref_code_len == 2
    assert torch.equal(out_ref_code, ref_code)


def test_voice_clone_prompt_artifacts_do_not_resolve_ref_audio_for_cache_fill():
    builder = _make_minimal_builder()
    device_param = torch.nn.Parameter(torch.empty(0))
    builder._text_embedding = _stub_text_embedding(device_param)
    builder._text_projection = lambda embeds: embeds
    builder._codec_embed = lambda ids: torch.zeros((*ids.shape, 4), device=ids.device)

    class FakeTokenizer:
        def __call__(self, *_args, **_kwargs):
            return {"input_ids": torch.arange(8, dtype=torch.long).reshape(1, -1)}

    builder._text_tokenizer = FakeTokenizer()
    builder._generate_icl_prompt = lambda **kwargs: (
        torch.ones((1, 2, 4), device=kwargs["ref_code"].device),
        torch.ones((1, 4), device=kwargs["ref_code"].device),
    )
    builder.normalize_ref_audio = lambda _raw: (_ for _ in ()).throw(AssertionError("normalize not expected"))
    builder.encode_ref_audio_to_code = lambda _wav, _sr: (_ for _ in ()).throw(AssertionError("encode not expected"))
    builder.extract_speaker_embedding = lambda _wav, _sr: (_ for _ in ()).throw(
        AssertionError("speaker embedding not expected")
    )
    ref_code = torch.arange(4, dtype=torch.long).reshape(2, 2)

    _prompt, _trailing, ref_code_len, out_ref_code = builder.build_prompt_embeds(
        task_type="Base",
        info_dict={
            "text": ["hello"],
            "ref_audio": ["ref.wav"],
            "ref_ids": torch.arange(8, dtype=torch.long).reshape(1, -1),
            "non_streaming_mode": [False],
            "voice_clone_prompt": {
                "ref_code": ref_code,
                "ref_spk_embedding": torch.ones(4, dtype=torch.bfloat16),
                "icl_mode": True,
            },
        },
    )

    assert ref_code_len == 2
    assert torch.equal(out_ref_code, ref_code)


def test_ref_audio_artifact_only_uses_cache_without_ref_audio_payload():
    builder = _make_minimal_builder()
    device_param = torch.nn.Parameter(torch.empty(0))
    builder._text_embedding = _stub_text_embedding(device_param)
    builder._text_projection = lambda embeds: embeds
    builder._codec_embed = lambda ids: torch.zeros((*ids.shape, 4), device=ids.device)

    class FakeTokenizer:
        def __call__(self, *_args, **_kwargs):
            return {"input_ids": torch.arange(8, dtype=torch.long).reshape(1, -1)}

    builder._text_tokenizer = FakeTokenizer()
    builder._generate_icl_prompt = lambda **kwargs: (
        torch.ones((1, 2, 4), device=kwargs["ref_code"].device),
        torch.ones((1, 4), device=kwargs["ref_code"].device),
    )
    builder.normalize_ref_audio = lambda _raw: (_ for _ in ()).throw(AssertionError("normalize not expected"))
    builder.encode_ref_audio_to_code = lambda _wav, _sr: (_ for _ in ()).throw(AssertionError("encode not expected"))
    builder.extract_speaker_embedding = lambda _wav, _sr: (_ for _ in ()).throw(
        AssertionError("speaker embedding not expected")
    )
    ref_code = torch.arange(4, dtype=torch.long).reshape(2, 2)
    builder.put_ref_audio_artifacts(
        "same-ref",
        ref_code=ref_code,
        ref_spk_embedding=torch.ones(4, dtype=torch.bfloat16),
    )

    _prompt, _trailing, ref_code_len, out_ref_code = builder.build_prompt_embeds(
        task_type="Base",
        info_dict={
            "text": ["hello"],
            "ref_ids": torch.arange(8, dtype=torch.long).reshape(1, -1),
            "non_streaming_mode": [False],
            REF_AUDIO_CACHE_KEY: ["same-ref"],
        },
    )

    assert ref_code_len == 2
    assert torch.equal(out_ref_code, ref_code)


def test_ref_audio_artifact_only_cache_miss_fails_fast():
    builder = _make_minimal_builder()
    device_param = torch.nn.Parameter(torch.empty(0))
    builder._text_embedding = _stub_text_embedding(device_param)
    builder._text_projection = lambda embeds: embeds
    builder._codec_embed = lambda ids: torch.zeros((*ids.shape, 4), device=ids.device)

    class FakeTokenizer:
        def __call__(self, *_args, **_kwargs):
            return {"input_ids": torch.arange(8, dtype=torch.long).reshape(1, -1)}

    builder._text_tokenizer = FakeTokenizer()

    with pytest.raises(RuntimeError, match="artifact cache miss"):
        builder.build_prompt_embeds(
            task_type="Base",
            info_dict={
                "text": ["hello"],
                "ref_ids": torch.arange(8, dtype=torch.long).reshape(1, -1),
                "non_streaming_mode": [False],
                REF_AUDIO_CACHE_KEY: ["missing-ref"],
            },
        )


def test_evicted_ref_audio_cache_key_with_payload_recomputes_artifacts():
    """A serving-side artifact key whose engine entry was evicted is only a cache miss."""
    builder = _make_minimal_builder()
    builder._ref_audio_artifact_cache_max_entries = 1
    device_param = torch.nn.Parameter(torch.empty(0))
    builder._text_embedding = _stub_text_embedding(device_param)
    builder._text_projection = lambda embeds: embeds
    builder._codec_embed = lambda ids: torch.zeros((*ids.shape, 4), device=ids.device)

    class FakeTokenizer:
        def __call__(self, *_args, **_kwargs):
            return {"input_ids": torch.arange(8, dtype=torch.long).reshape(1, -1)}

    builder._text_tokenizer = FakeTokenizer()
    builder._generate_icl_prompt = lambda **kwargs: (
        torch.ones((1, 2, 4), device=kwargs["ref_code"].device),
        torch.ones((1, 4), device=kwargs["ref_code"].device),
    )
    ref_audio = np.arange(1024, dtype=np.float32)
    builder.normalize_ref_audio = lambda _raw: (ref_audio, 16000)
    encoded = []
    builder._encode_ref_audio_batch_fn = lambda wavs, _sr, *, device: [
        encoded.append(len(w)) or torch.ones((2, 2), dtype=torch.long) for w in wavs
    ]
    builder.extract_speaker_embedding = lambda _wav, _sr: torch.ones(4, dtype=torch.bfloat16)

    stale = {"ref_code": torch.zeros((3, 2), dtype=torch.long), "ref_spk_embedding": torch.zeros(4)}
    builder.put_ref_audio_artifacts("ref-a", **stale)
    builder.put_ref_audio_artifacts("ref-b", **stale)
    assert builder.get_ref_audio_artifacts("ref-a") is None

    _prompt, _trailing, ref_code_len, ref_code = builder.build_prompt_embeds(
        task_type="Base",
        info_dict={
            "text": ["hello"],
            "ref_audio": ["ref.wav"],
            "ref_ids": torch.arange(8, dtype=torch.long).reshape(1, -1),
            "non_streaming_mode": [False],
            REF_AUDIO_CACHE_KEY: ["ref-a"],
        },
    )

    assert encoded == [1024]
    assert ref_code_len == 2
    assert torch.equal(ref_code, torch.ones((2, 2), dtype=torch.long))
    refilled = builder.get_ref_audio_artifacts("ref-a")
    assert refilled is not None
    assert torch.equal(refilled["ref_code"], ref_code)


def test_serving_ref_audio_wire_form_survives_engine_ipc_and_normalizes():
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    from vllm_omni.engine import AdditionalInformationPayload
    from vllm_omni.engine.serialization import deserialize_additional_information, serialize_additional_information
    from vllm_omni.entrypoints.openai.tts_adapters.qwen3_tts import _ref_audio_wire_waveform

    wav = np.random.default_rng(0).standard_normal(24000).astype(np.float32)
    wire = serialize_additional_information({"ref_audio": [[_ref_audio_wire_waveform(wav), 24000]]})
    decoded = MsgpackDecoder(AdditionalInformationPayload).decode(MsgpackEncoder().encode(wire))
    info = deserialize_additional_information(decoded)

    out_wav, out_sr = _make_minimal_builder().normalize_ref_audio(info["ref_audio"][0])

    assert out_sr == 24000
    assert out_wav.dtype == np.float32
    assert np.array_equal(out_wav, wav)
