# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace

import pytest
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import ProcessorInputs, TimingContext

from vllm_omni.model_executor.models.neutts_air import neutts_air_talker
from vllm_omni.model_executor.models.neutts_air.neutts_air_talker import (
    NEUTTS_SPEECH_TOKEN_OFFSET,
    NeuTTSAirForCausalLM,
    NeuTTSAirMultiModalProcessor,
    build_neutts_air_prompt_token_ids,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeTokenizer:
    special_ids = {
        "<|TEXT_REPLACE|>": 10,
        "<|TEXT_PROMPT_START|>": 11,
        "<|TEXT_PROMPT_END|>": 12,
        "<|SPEECH_REPLACE|>": 13,
        "<|SPEECH_GENERATION_START|>": 14,
    }

    def encode(self, text, add_special_tokens=True):
        del add_special_tokens
        if text.startswith("user: Convert"):
            return [100, 10, 101, 13, 102]
        if text == "REF TARGET":
            return [200, 201]
        raise AssertionError(f"Unexpected tokenizer input: {text}")

    def convert_tokens_to_ids(self, token):
        return self.special_ids[token]


class FakePhonemizer:
    def phonemize(self, texts):
        values = {"reference": "REF", "target": "TARGET"}
        return [values[text] for text in texts]


def test_prompt_builder_matches_the_official_neutts_layout():
    prompt_ids = build_neutts_air_prompt_token_ids(
        FakeTokenizer(),
        ref_codes=[0, 29],
        ref_text="reference",
        target_text="target",
        phonemize=lambda text: {"reference": "REF", "target": "TARGET"}[text],
    )

    assert prompt_ids == [
        100,
        11,
        200,
        201,
        12,
        101,
        14,
        NEUTTS_SPEECH_TOKEN_OFFSET,
        NEUTTS_SPEECH_TOKEN_OFFSET + 29,
    ]


def test_prompt_builder_rejects_out_of_range_reference_codes():
    with pytest.raises(ValueError, match="must be in"):
        build_neutts_air_prompt_token_ids(
            FakeTokenizer(),
            ref_codes=[65536],
            ref_text="reference",
            target_text="target",
            phonemize=str.upper,
        )


def test_processor_consumes_reference_codes_into_an_ordinary_token_prompt(
    monkeypatch,
):
    monkeypatch.setattr(neutts_air_talker, "_PHONEMIZER", FakePhonemizer())
    processor = object.__new__(NeuTTSAirMultiModalProcessor)
    processor.info = SimpleNamespace(ctx=SimpleNamespace(tokenizer=FakeTokenizer()))
    inputs = ProcessorInputs(
        prompt="target",
        mm_data_items=MultiModalDataItems({}),
        hf_processor_mm_kwargs={
            "ref_text": "reference",
            "ref_codes": [0, 29],
        },
    )

    output = processor.apply(inputs, TimingContext(enabled=False))

    assert output["prompt_token_ids"][-2:] == [
        NEUTTS_SPEECH_TOKEN_OFFSET,
        NEUTTS_SPEECH_TOKEN_OFFSET + 29,
    ]
    assert output["mm_kwargs"] == {}
    assert output["mm_placeholders"] == {}


def test_talker_is_a_thin_native_qwen2_subclass():
    assert issubclass(NeuTTSAirForCausalLM, Qwen2ForCausalLM)
    assert NeuTTSAirForCausalLM.requires_raw_input_tokens


def test_explicit_espeak_assets_configure_exact_paths(monkeypatch, tmp_path):
    library = tmp_path / "libespeak-ng.so"
    library.write_bytes(b"fake")
    data_path = tmp_path / "espeak-ng-data"
    data_path.mkdir()
    configured = {}

    wrapper = SimpleNamespace(
        set_library=lambda path: configured.setdefault("library", path),
        set_data_path=lambda path: configured.setdefault("data_path", path),
    )
    monkeypatch.setenv(
        neutts_air_talker.NEUTTS_ESPEAK_LIBRARY_ENV,
        str(library),
    )
    monkeypatch.setenv(
        neutts_air_talker.NEUTTS_ESPEAK_DATA_PATH_ENV,
        str(data_path),
    )

    assert neutts_air_talker._configure_explicit_espeak_assets(wrapper)
    assert configured == {
        "library": str(library),
        "data_path": str(data_path),
    }


def test_explicit_espeak_assets_require_both_paths(monkeypatch, tmp_path):
    library = tmp_path / "libespeak-ng.so"
    library.write_bytes(b"fake")
    monkeypatch.setenv(
        neutts_air_talker.NEUTTS_ESPEAK_LIBRARY_ENV,
        str(library),
    )
    monkeypatch.delenv(
        neutts_air_talker.NEUTTS_ESPEAK_DATA_PATH_ENV,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="must be set together"):
        neutts_air_talker._configure_explicit_espeak_assets(SimpleNamespace())


def _install_fake_phonemizer(monkeypatch, version):
    class FakeEspeakBackend:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def version(self):
            return version

    class FakeEspeakWrapper:
        pass

    phonemizer_module = ModuleType("phonemizer")
    backend_module = ModuleType("phonemizer.backend")
    espeak_module = ModuleType("phonemizer.backend.espeak")
    wrapper_module = ModuleType("phonemizer.backend.espeak.wrapper")

    phonemizer_module.__path__ = []
    backend_module.__path__ = []
    espeak_module.__path__ = []
    backend_module.EspeakBackend = FakeEspeakBackend
    wrapper_module.EspeakWrapper = FakeEspeakWrapper

    monkeypatch.setitem(sys.modules, "phonemizer", phonemizer_module)
    monkeypatch.setitem(sys.modules, "phonemizer.backend", backend_module)
    monkeypatch.setitem(sys.modules, "phonemizer.backend.espeak", espeak_module)
    monkeypatch.setitem(
        sys.modules,
        "phonemizer.backend.espeak.wrapper",
        wrapper_module,
    )
    monkeypatch.setattr(
        neutts_air_talker,
        "_configure_explicit_espeak_assets",
        lambda wrapper: True,
    )
    monkeypatch.setattr(neutts_air_talker, "_PHONEMIZER", None)


def test_phonemizer_accepts_and_caches_the_official_espeak_version(monkeypatch):
    expected = neutts_air_talker.EXPECTED_NEUTTS_ESPEAK_VERSION
    _install_fake_phonemizer(monkeypatch, expected)

    phonemizer = neutts_air_talker._get_english_phonemizer()

    assert tuple(phonemizer.version()) == expected
    assert neutts_air_talker._get_english_phonemizer() is phonemizer


def test_phonemizer_rejects_an_unexpected_espeak_version(monkeypatch):
    _install_fake_phonemizer(monkeypatch, (1, 52, 0))

    with pytest.raises(RuntimeError, match="official eSpeak assets"):
        neutts_air_talker._get_english_phonemizer()
