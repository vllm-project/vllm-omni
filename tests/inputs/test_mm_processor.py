# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from transformers import BatchFeature
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.multimodal.parse import AudioProcessorItems, ImageProcessorItems, MultiModalDataItems
from vllm.multimodal.processing import BaseMultiModalProcessor

from vllm_omni.inputs.mm_processor import OmniDummyInputsBuilder, OmniMultiModalProcessor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("counts", [{}, {"audio": 1, "image": 0}])
@pytest.mark.parametrize("original_text", [None, "", "voice cloning text"])
def test_custom_processor_bridge_preserves_prompt_kwargs_and_passthrough(counts, original_text):
    class Processor(OmniMultiModalProcessor):
        def _get_mm_fields_config(self, *args):
            return {}

        def _get_prompt_updates(self, *args):
            return []

    processor = object.__new__(Processor)
    processor.dummy_inputs = SimpleNamespace(get_dummy_text=Mock(return_value="dummy"))
    processor._call_hf_processor = Mock(return_value=BatchFeature({"features": "processed"}))

    class AudioItems(AudioProcessorItems):
        def get_passthrough_data(self):
            return {"embedding": "preserve"}

    items = MultiModalDataItems({"audio": AudioItems(["wave"]), "image": ImageProcessorItems([])} if counts else {})
    items.select = Mock(wraps=items.select)
    kwargs = {"sampling_rate": 16000}
    if original_text is not None:
        kwargs[processor._OMNI_PROMPT_TEXT_KEY] = original_text
    before = dict(kwargs)
    result = processor._apply_hf_processor_main(items, kwargs)
    items.select.assert_called_once_with({k for k, v in counts.items() if v > 0})
    processor._call_hf_processor.assert_called_once_with(
        "dummy" if original_text is None else original_text,
        {"audios": ["wave"]} if counts else {},
        {"sampling_rate": 16000},
        {},
    )
    assert dict(result) == ({"features": "processed", "embedding": "preserve"} if counts else {"features": "processed"})
    assert kwargs == before


@pytest.mark.parametrize(
    "module_name,builder_name,processor_name,expected_kwargs",
    [
        (
            "cosyvoice3",
            "CosyVoice3DummyInputsBuilder",
            "CosyVoice3MultiModalProcessor",
            {"prompt_text": "Testing my voices. Why should I not?"},
        ),
        (
            "glm_tts",
            "GLMTTSDummyInputsBuilder",
            "GLMTTSMultiModalProcessor",
            {"prompt_text": "This is the reference voice."},
        ),
        (
            "omnivoice",
            "OmniVoiceDummyInputsBuilder",
            "OmniVoiceMultiModalProcessor",
            {"ref_text": "Testing voice cloning."},
        ),
        ("mimo_audio", "MiMoAudioLLMDummyInputsBuilder", "MiMoAudioLLMMultiModalProcessor", {}),
    ],
)
def test_dummy_budget_preserves_model_conditioning(module_name, builder_name, processor_name, expected_kwargs):
    """The v0.31 budget path must reach each model's custom builder hook."""
    module = import_module(f"vllm_omni.model_executor.models.{module_name}.{module_name}")
    builder = object.__new__(getattr(module, builder_name))
    assert isinstance(builder, OmniDummyInputsBuilder)
    items = MultiModalDataItems({})
    builder.info = SimpleNamespace(ctx=SimpleNamespace(tokenizer=None), parse_mm_data=Mock(return_value=items))
    builder.get_dummy_mm_data = Mock(return_value={})
    processor = object.__new__(getattr(module, processor_name))
    processor.dummy_inputs = builder

    inputs = processor.get_dummy_inputs(128, {"audio": 1}, MultiModalDummyOptions())

    assert inputs.mm_data_items is items
    assert inputs.hf_processor_mm_kwargs == expected_kwargs
    assert inputs.prompt == ("<|empty|>" if module_name == "mimo_audio" else [])


def test_dummy_budget_keeps_upstream_path_for_standard_builders(monkeypatch):
    class Processor(OmniMultiModalProcessor):
        def _get_mm_fields_config(self, *args):
            return {}

        def _get_prompt_updates(self, *args):
            return []

    processor = object.__new__(Processor)
    processor.dummy_inputs = SimpleNamespace()
    expected = object()
    upstream = Mock(return_value=expected)
    monkeypatch.setattr(BaseMultiModalProcessor, "get_dummy_inputs", upstream)
    options = MultiModalDummyOptions()

    assert processor.get_dummy_inputs(128, {"audio": 1}, options) is expected
    upstream.assert_called_once_with(128, {"audio": 1}, options)


def test_voxtral_dummy_budget_uses_audio_token_prompt():
    from vllm_omni.model_executor.models.voxtral_tts.voxtral_tts_audio_generation import VoxtralTTSMultiModalProcessor

    processor = object.__new__(VoxtralTTSMultiModalProcessor)
    expected = SimpleNamespace(prompt=[1, 25, 24, 35])
    builder = SimpleNamespace(get_dummy_processor_inputs=Mock(return_value=expected))
    processor.dummy_inputs = builder
    options = MultiModalDummyOptions()

    assert processor.get_dummy_inputs(128, {"audio": 1}, options) is expected
    builder.get_dummy_processor_inputs.assert_called_once_with(128, {"audio": 1}, options)
