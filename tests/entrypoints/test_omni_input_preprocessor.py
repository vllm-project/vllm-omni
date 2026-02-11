import pytest

from vllm_omni.inputs.preprocess import OmniInputPreprocessor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_preprocessor(monkeypatch):
    preprocessor = object.__new__(OmniInputPreprocessor)
    monkeypatch.setattr(preprocessor, "_truncate_inputs", lambda tokens, tokenization_kwargs=None: tokens)
    monkeypatch.setattr(
        preprocessor,
        "_process_multimodal",
        lambda *args, **kwargs: {"prompt_token_ids": [1, 2, 3]},
    )
    monkeypatch.setattr(preprocessor, "_tokenize_prompt", lambda prompt_text, tokenization_kwargs=None: [9, 8, 7])
    return preprocessor


def test_process_tokens_keeps_additional_information(monkeypatch):
    preprocessor = _make_preprocessor(monkeypatch)
    parsed = {
        "prompt_token_ids": [1, 2, 3],
        "prompt_embeds": "embeds",
        "additional_information": {"task": ["tts"], "lang": ["auto"]},
    }

    inputs = OmniInputPreprocessor._process_tokens(preprocessor, parsed)

    assert inputs["prompt_token_ids"] == [1, 2, 3]
    assert inputs["prompt_embeds"] == "embeds"
    assert inputs["additional_information"] == {"task": ["tts"], "lang": ["auto"]}


def test_process_text_keeps_additional_information(monkeypatch):
    preprocessor = _make_preprocessor(monkeypatch)
    parsed = {
        "prompt": "hello",
        "prompt_embeds": "embeds",
        "additional_information": {"speaker": ["alice"]},
    }

    inputs = OmniInputPreprocessor._process_text(preprocessor, parsed)

    assert inputs["prompt_token_ids"] == [9, 8, 7]
    assert inputs["prompt_embeds"] == "embeds"
    assert inputs["additional_information"] == {"speaker": ["alice"]}


def test_process_text_multimodal_skips_empty_payloads(monkeypatch):
    preprocessor = _make_preprocessor(monkeypatch)
    parsed = {
        "prompt": "hello",
        "multi_modal_data": {"image": "fake"},
        "prompt_embeds": None,
        "additional_information": None,
    }

    inputs = OmniInputPreprocessor._process_text(preprocessor, parsed)

    assert inputs["prompt_token_ids"] == [1, 2, 3]
    assert "prompt_embeds" not in inputs
    assert "additional_information" not in inputs
