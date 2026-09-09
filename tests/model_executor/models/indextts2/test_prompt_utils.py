# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.indextts2 import prompt_utils
from vllm_omni.model_executor.models.indextts2.tokenizer_v2_5 import (
    INDEXTTS25_TOKENIZER_FILE,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _load_indextts2_offline_example():
    example_path = (
        Path(__file__).resolve().parents[4]
        / "examples"
        / "offline_inference"
        / "text_to_speech"
        / "indextts2"
        / "end2end.py"
    )
    spec = importlib.util.spec_from_file_location(
        "indextts2_offline_example_test",
        example_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_main_uses_resolved_stage0_tokenizer_file(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    example = _load_indextts2_offline_example()
    stage0 = SimpleNamespace(engine_args=SimpleNamespace(hf_overrides={"tokenizer_file": "custom-tokenizer.tiktoken"}))
    fake_omni = SimpleNamespace(
        stage_configs=[stage0],
        generate=lambda *args, **kwargs: [],
    )
    captured = {}

    monkeypatch.setattr(example, "Omni", lambda **kwargs: fake_omni)
    monkeypatch.setattr(example, "SamplingParams", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        example,
        "build_request",
        lambda **kwargs: captured.update(kwargs) or {"additional_information": {}},
    )

    example.main(
        SimpleNamespace(
            model="/model",
            model_version="2.5",
            deploy_config="/deploy/custom.yaml",
            use_gpt_latent=False,
            stage_init_timeout=300,
            seed=42,
            output_dir=str(tmp_path),
            text="hello",
            ref_audio=None,
            emo_audio=None,
            emo_text=None,
            emo_vector=None,
            emo_alpha=None,
            use_emo_text=False,
            use_random=False,
            lang="en",
            text_normalization=True,
        )
    )

    assert captured["tokenizer_file"] == "custom-tokenizer.tiktoken"


def test_indextts25_prompt_len_uses_three_conditioning_tokens(monkeypatch):
    captured = {}

    def fake_prepare(*args, **kwargs):
        captured.update(kwargs)
        return [11, 12, 13, 14], 0

    monkeypatch.setattr(
        prompt_utils,
        "prepare_indextts25_text",
        fake_prepare,
    )

    prompt_len = prompt_utils.estimate_indextts2_prefill_prompt_len(
        "/model",
        "hello",
        model_type="indextts2_5",
        lang="en",
        text_normalization=False,
    )

    # Official prepare_gpt_inputs wraps the language-prefixed tokenizer IDs
    # with start_text=0 and stop_text=1 before appending start_mel.
    assert prompt_len == 3 + (4 + 2) + 1
    assert captured["text_normalization"] is False
    assert captured["tokenizer_file"] == INDEXTTS25_TOKENIZER_FILE


def test_indextts25_prompt_len_uses_custom_tokenizer_file(monkeypatch):
    captured = {}

    def fake_prepare(*args, **kwargs):
        captured.update(kwargs)
        return [11, 12], 0

    monkeypatch.setattr(
        prompt_utils,
        "prepare_indextts25_text",
        fake_prepare,
    )

    prompt_utils.estimate_indextts2_prefill_prompt_len(
        "/model",
        "hello",
        model_type="indextts2_5",
        tokenizer_file="custom-tokenizer.tiktoken",
    )

    assert captured["tokenizer_file"] == "custom-tokenizer.tiktoken"


def test_indextts25_prompt_len_filters_existing_text_wrapper_ids(monkeypatch):
    monkeypatch.setattr(
        prompt_utils,
        "prepare_indextts25_text",
        lambda *args, **kwargs: ([0, 58838, 1, 42, 0], 0),
    )

    prompt_len = prompt_utils.estimate_indextts2_prefill_prompt_len(
        "/model",
        "hello",
        model_type="indextts2_5",
        lang="en",
    )

    assert prompt_len == 3 + (2 + 2) + 1


def test_indextts2_prompt_len_ignores_v25_tokenizer_file(monkeypatch):
    class FakeTokenizer:
        def encode(self, text, *, add_special_tokens):
            assert text == "hello"
            assert add_special_tokens is False
            return [11, 12]

    def fail_prepare(*args, **kwargs):
        raise AssertionError("IndexTTS 2 must not use the IndexTTS 2.5 tokenizer")

    monkeypatch.setattr(
        prompt_utils,
        "_get_text_tokenizer",
        lambda model_id_or_path: FakeTokenizer(),
    )
    monkeypatch.setattr(
        prompt_utils,
        "prepare_indextts25_text",
        fail_prepare,
    )

    prompt_len = prompt_utils.estimate_indextts2_prefill_prompt_len(
        "/model",
        "hello",
        model_type="indextts2",
        tokenizer_file="custom.tiktoken",
    )

    assert prompt_len == 34 + (2 + 2) + 1


def test_indextts25_prompt_ids_preserve_placeholder_value(monkeypatch):
    captured = {}

    def fake_estimate(*args, **kwargs):
        captured.update(kwargs)
        return 10

    monkeypatch.setattr(
        prompt_utils,
        "estimate_indextts2_prefill_prompt_len",
        fake_estimate,
    )

    prompt_ids = prompt_utils.build_indextts2_prefill_prompt_ids(
        "/model",
        "hello",
        model_type="indextts2_5",
        lang="en",
        tokenizer_file="custom-tokenizer.tiktoken",
        placeholder_token_id=17,
    )

    assert prompt_ids == [17] * 10
    assert captured["tokenizer_file"] == "custom-tokenizer.tiktoken"
