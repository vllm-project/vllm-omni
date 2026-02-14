from types import SimpleNamespace

import pytest
import torch

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


def test_build_apertus_image_prompt_uses_emu35_format_without_eof():
    preprocessor = object.__new__(ApertusOmniInputPreprocessor)
    preprocessor.tokenizer = SimpleNamespace(
        boi_token="<|image start|>",
        img_token="<|image token|>",
        eol_token="<|extra_200|>",
        eoi_token="<|image end|>",
    )

    token_grid = torch.tensor([[5, 6], [7, 8]])
    image_prompt = ApertusOmniInputPreprocessor._build_apertus_image_prompt(
        preprocessor,
        token_grid,
    )

    assert image_prompt == (
        "<|image start|>2*2<|image token|>"
        "<|visual token 5|><|visual token 6|><|extra_200|>"
        "<|visual token 7|><|visual token 8|>"
        "<|image end|>"
    )
    assert "<|img_end_of_frame|>" not in image_prompt


def test_extract_emu35_token_grid_handles_nested_none_tuple():
    token_ids = torch.arange(6, dtype=torch.int64)
    encode_out = (torch.zeros(1), None, (None, None, token_ids))
    image_token_grid = ApertusOmniInputPreprocessor._extract_emu35_token_grid(
        encode_out,
        token_height=2,
        token_width=3,
    )

    assert tuple(image_token_grid.shape) == (2, 3)
    assert image_token_grid.tolist() == [[0, 1, 2], [3, 4, 5]]


def test_process_apertus_text_sets_add_special_tokens_false_by_default(monkeypatch):
    preprocessor = object.__new__(ApertusOmniInputPreprocessor)
    preprocessor.tokenizer = SimpleNamespace()
    monkeypatch.setattr(
        preprocessor,
        "_normalize_apertus_images",
        lambda image_data: [object()],
    )
    monkeypatch.setattr(
        preprocessor,
        "_encode_apertus_images_to_strings",
        lambda images, mm_processor_kwargs: ["<img_prompt>"],
    )
    monkeypatch.setattr(
        preprocessor,
        "_resolve_apertus_image_placeholder",
        lambda prompt_text, mm_processor_kwargs: "<|image|>",
    )
    captured = {}

    def _fake_tokenize(prompt_text, tokenization_kwargs=None):
        captured["prompt_text"] = prompt_text
        captured["tokenization_kwargs"] = tokenization_kwargs
        return [1, 2, 3]

    monkeypatch.setattr(preprocessor, "_tokenize_prompt", _fake_tokenize)

    inputs = ApertusOmniInputPreprocessor._process_apertus_text_with_images(
        preprocessor,
        prompt_text="hello <|image|>",
        multi_modal_data={"image": [object()]},
        mm_processor_kwargs={},
        tokenization_kwargs=None,
    )

    assert inputs["prompt_token_ids"] == [1, 2, 3]
    assert captured["prompt_text"] == "hello <img_prompt>"
    assert captured["tokenization_kwargs"]["add_special_tokens"] is False
