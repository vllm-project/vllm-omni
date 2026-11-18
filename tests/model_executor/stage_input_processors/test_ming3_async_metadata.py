# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from typing import cast

import pytest

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.model_executor.stage_input_processors.ming3_audio import thinker2talker_async_chunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Tokenizer:
    is_fast = True

    def __len__(self):
        return 0x110000

    def convert_ids_to_tokens(self, token_ids, *, skip_special_tokens):
        return [chr(token_id) for token_id in token_ids]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def _manager() -> SimpleNamespace:
    return SimpleNamespace(_ming3_text_tokenizer=_Tokenizer(), _ming3_chunk_state={})


def _request(text: str, info=None) -> OmniEngineCoreRequest:
    return cast(
        OmniEngineCoreRequest,
        SimpleNamespace(
            external_req_id="request",
            prompt_token_ids=[99],
            all_token_ids=[99, *map(ord, text)],
            additional_information=info,
        ),
    )


def test_async_preserves_serialized_talker_controls():
    controls = {
        "tn_mode": "upstream",
        "normalize_text": False,
        "scrub_markup": True,
        "stream_decode": False,
        "seed": 123,
        "voice_name": "灵小柔",
    }
    info = serialize_additional_information(controls)
    output = thinker2talker_async_chunk(_manager(), None, _request("Chinese\t你好。", info), is_finished=True)
    assert output is not None
    assert output.additional_information["text"] == "你好。"
    assert output.additional_information["ming_text_fragments"] == ["你好。"]
    assert output.additional_information["ip_name"] == "灵小柔"
    assert output.additional_information["reference_audio"] is None
    for key, value in controls.items():
        if key != "voice_name":
            assert output.additional_information[key] == value


@pytest.mark.parametrize("text", ["  Chinese\t  ", "<script>secret()</script>"])
def test_empty_terminal_still_completes_and_cleans_request_state(text):
    manager = _manager()
    output = thinker2talker_async_chunk(manager, None, _request(text), is_finished=True)
    assert output is not None
    assert output.codes.audio.numel() == 0
    assert output.additional_information["ming_text_fragments"] == []
    assert manager._ming3_chunk_state == {}


def test_raw_prefix_state_survives_language_stripping_and_terminal_tail():
    manager = _manager()
    first_text = "Chinese\t这里是一段超过十二个字的正常回答。"
    first = thinker2talker_async_chunk(manager, None, _request(first_text))
    assert first is not None
    assert first.additional_information["text"] == first_text.split("\t", 1)[1]
    assert len(manager._ming3_chunk_state["request"].token_ids) == len(first_text)
    assert thinker2talker_async_chunk(manager, None, _request(first_text)) is None
    final = thinker2talker_async_chunk(manager, None, _request(first_text + "剩余文字"), is_finished=True)
    assert final is not None
    assert final.additional_information["text"] == "剩余文字。"
    assert manager._ming3_chunk_state == {}


def test_incremental_input_keeps_numeric_url_and_abbreviation_context():
    text = "Dr. Smith gave e.g. 日期2026.10.16，金额12.5元，IP192.168.1.1，https://example.com/path?a=1 这个例子!"
    complete = thinker2talker_async_chunk(_manager(), None, _request(text), is_finished=True)
    manager = _manager()
    fragments = []
    for length in range(1, len(text) + 1):
        output = thinker2talker_async_chunk(manager, None, _request(text[:length]))
        if output is not None:
            fragments.extend(output.additional_information["ming_text_fragments"])
    final = thinker2talker_async_chunk(manager, None, _request(text), is_finished=True)
    fragments.extend(final.additional_information["ming_text_fragments"])
    assert fragments == complete.additional_information["ming_text_fragments"]
    assert manager._ming3_chunk_state == {}


def test_prompt_and_selected_ip_remain_stable_after_first_speakable_fragment():
    manager = _manager()
    controls = serialize_additional_information({"voice_name": "US_Stella"})
    text = "This is a long enough answer with many ordinary English words for speech!"
    first = thinker2talker_async_chunk(manager, None, _request(text, controls))
    assert first is not None
    final = thinker2talker_async_chunk(
        manager, None, _request(text + "接下来换成中文回答。", controls), is_finished=True
    )
    assert first.additional_information["ip_name"] == "US_Stella"
    assert final.additional_information["ip_name"] == "US_Stella"
    assert first.additional_information["prompt"] == final.additional_information["prompt"]


@pytest.mark.parametrize("override", [{"scrub_markup": False}, {"max_text_length": 5}])
def test_frontend_controls_cannot_change_mid_stream(override):
    manager = _manager()
    assert thinker2talker_async_chunk(manager, None, _request("等待足够的文字")) is None
    with pytest.raises(ValueError, match="cannot change"):
        thinker2talker_async_chunk(
            manager, None, _request("等待足够的文字。", serialize_additional_information(override))
        )


def test_incomplete_utf8_is_not_committed_as_spoken_text():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ["你好世界"],
        trainers.BpeTrainer(vocab_size=256, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    manager = SimpleNamespace(_ming3_text_tokenizer=tokenizer, _ming3_chunk_state={})
    token_ids = tokenizer.encode("你好世界", add_special_tokens=False)
    request = _request("")
    for length in range(1, len(token_ids)):
        request.all_token_ids = [99, *token_ids[:length]]
        assert thinker2talker_async_chunk(manager, None, request) is None
    request.all_token_ids = [99, *token_ids]
    output = thinker2talker_async_chunk(manager, None, request, is_finished=True)
    assert output.additional_information["text"] == "你好世界。"


def test_each_token_is_converted_once_without_full_decode():
    class CountingTokenizer(_Tokenizer):
        converted = 0

        def decode(self, *args, **kwargs):
            raise AssertionError("full-history decode was used")

        def convert_ids_to_tokens(self, token_ids, *, skip_special_tokens):
            self.converted += len(token_ids)
            return super().convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

    tokenizer = CountingTokenizer()
    manager = SimpleNamespace(_ming3_text_tokenizer=tokenizer, _ming3_chunk_state={})
    for length in range(1, 2049):
        thinker2talker_async_chunk(manager, None, _request("a" * length))
    assert tokenizer.converted == 2048


def test_shrinking_token_history_is_rejected():
    manager = _manager()
    thinker2talker_async_chunk(manager, None, _request("hello"))
    with pytest.raises(RuntimeError, match="history shrank"):
        thinker2talker_async_chunk(manager, None, _request("he"))
