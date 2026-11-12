# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""HTTP admission and native speech must agree on the local text tokenizer."""

from unittest.mock import Mock

import pytest

from vllm_omni.entrypoints.openai.drvexo_speech import load_speech_tokenizer
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.drvexo import DRVEXOTTSAdapter
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import Qwen3TTSPromptEmbedsBuilder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("version,needs_regex_fix", [("4.57.1", True), ("5.0.0rc1", False), ("10.0.0", False)])
def test_http_and_native_tokenizers_use_the_same_version_settings(monkeypatch, tmp_path, version, needs_regex_fix):
    import transformers

    monkeypatch.setattr(transformers, "__version__", version)
    loader = Mock(side_effect=lambda *_args, **_kwargs: Mock())
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", loader)
    builder = object.__new__(Qwen3TTSPromptEmbedsBuilder)
    builder._model_path = str(tmp_path / "talker")
    builder._text_tokenizer = None
    http_tokenizer = load_speech_tokenizer(tmp_path)
    native_tokenizer = builder.get_text_tokenizer()
    assert http_tokenizer.padding_side == native_tokenizer.padding_side == "left"
    server = Mock(spec=OmniOpenAIServingSpeech, _tts_tokenizer=None)
    adapter = DRVEXOTTSAdapter.__new__(DRVEXOTTSAdapter)
    adapter.ctx = SpeechServingContext(server=server, engine_client=Mock(model_config=Mock(model=str(tmp_path))))
    adapter._load_text_tokenizer()
    assert server._tts_tokenizer.padding_side == "left"
    assert loader.call_count == 3
    for call in loader.call_args_list:
        assert call.args == (str(tmp_path / "talker"),)
        assert call.kwargs["trust_remote_code"] is True
        assert call.kwargs["use_fast"] is True
        assert (call.kwargs.get("fix_mistral_regex") is True) is needs_regex_fix
    assert loader.call_args_list[0].kwargs["local_files_only"] is True
    assert loader.call_args_list[2].kwargs["local_files_only"] is True
    assert builder.get_text_tokenizer() is native_tokenizer
    adapter._load_text_tokenizer()
    assert loader.call_count == 3


def test_http_and_native_tokenizers_preserve_identical_lexical_ids(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    from vllm_omni.model_executor.models.drvexo_speech_8b.lexical import fixed_text_ids

    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2, "<|im_start|>": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    target = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", additional_special_tokens=["<|im_start|>"]
    )
    target.save_pretrained(tmp_path / "talker")
    http = load_speech_tokenizer(tmp_path)
    builder = object.__new__(Qwen3TTSPromptEmbedsBuilder)
    builder._model_path = str(tmp_path / "talker")
    builder._text_tokenizer = None
    native = builder.get_text_tokenizer()
    assert http.get_vocab() == native.get_vocab()
    assert http.all_special_ids == native.all_special_ids
    for text in ("hello", "hello world"):
        assert fixed_text_ids(text, http) == fixed_text_ids(text, native)
    for text in ("hello <|im_start|>", "hello   world"):
        for loaded in (http, native):
            with pytest.raises(ValueError):
                fixed_text_ids(text, loaded)
