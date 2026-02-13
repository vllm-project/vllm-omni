from types import SimpleNamespace

import pytest

from vllm_omni.inputs.apertus_preprocess import (
    ApertusOmniInputPreprocessor,
    is_apertus_model_config,
)


def _make_apertus_preprocessor(monkeypatch):
    preprocessor = object.__new__(ApertusOmniInputPreprocessor)
    preprocessor.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="apertus", architectures=["ApertusForCausalLM"])
    )
    monkeypatch.setattr(preprocessor, "_tokenize_prompt", lambda prompt_text, tokenization_kwargs=None: [9, 8, 7])
    monkeypatch.setattr(
        preprocessor,
        "_process_apertus_text_with_images",
        lambda *args, **kwargs: {"prompt_token_ids": [4, 5, 6]},
    )
    return preprocessor


def test_is_apertus_model_config_true():
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="apertus", architectures=["ApertusForCausalLM"])
    )
    assert is_apertus_model_config(model_config) is True


def test_is_apertus_model_config_false():
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="qwen", architectures=["QwenForCausalLM"])
    )
    assert is_apertus_model_config(model_config) is False


def test_apertus_adapter_rejects_unsupported_modalities():
    preprocessor = object.__new__(ApertusOmniInputPreprocessor)
    with pytest.raises(ValueError, match="text and image inputs only"):
        preprocessor._is_apertus_text_image_input(
            {"image": [object()], "audio": [object()]},
        )


def test_process_text_uses_apertus_adapter_path(monkeypatch):
    preprocessor = _make_apertus_preprocessor(monkeypatch)
    parsed = {
        "prompt": "hello <|image|>",
        "multi_modal_data": {"image": ["fake"]},
        "mm_processor_kwargs": {},
    }

    inputs = ApertusOmniInputPreprocessor._process_text(preprocessor, parsed)
    assert inputs["prompt_token_ids"] == [4, 5, 6]
