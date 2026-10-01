# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OmniVoice processor: the prompt is tokenized exactly once per request.

Upstream ``_cached_apply_hf_processor`` calls ``_apply_hf_processor_main`` for
every request, including text-only requests and requests whose reference audio
is already cached (empty ``mm_items``). These tests pin that only ``apply``
touches the text tokenizer and that the media path only encodes the audio.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import ProcessorInputs

from vllm_omni.inputs.mm_processor import OmniMultiModalProcessor
from vllm_omni.model_executor.models.omnivoice.omnivoice import OmniVoiceMultiModalProcessor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PROMPT_IDS = [5, 6, 7]


def _make_processor(monkeypatch: pytest.MonkeyPatch) -> OmniVoiceMultiModalProcessor:
    processor = object.__new__(OmniVoiceMultiModalProcessor)
    tokenizer = SimpleNamespace(decode=Mock(return_value="hello world"))
    processor.info = SimpleNamespace(
        ctx=SimpleNamespace(
            get_hf_config=lambda *_args, **_kwargs: SimpleNamespace(sample_rate=24000),
            model_config=SimpleNamespace(model="fake-model"),
        ),
        get_tokenizer=lambda: tokenizer,
    )
    processor.text_tokenizer = Mock(return_value=SimpleNamespace(input_ids=torch.tensor([_PROMPT_IDS])))
    processor.audio_tokenizer = SimpleNamespace(encode=Mock(return_value=torch.zeros(8, 4, dtype=torch.long)))
    processor.feature_extractor = None
    monkeypatch.setattr(processor, "_ensure_cached_runtime_components", lambda *_args, **_kwargs: None)
    return processor


def _items(counts: dict[str, int], mm_data: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        get_all_counts=lambda: counts,
        select=lambda _keys: mm_data,
    )


def test_apply_tokenizes_the_prompt_once(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = _make_processor(monkeypatch)
    captured: dict[str, ProcessorInputs] = {}

    def fake_parent_apply(self, inputs, timing_ctx):
        captured["inputs"] = inputs
        return "sentinel"

    monkeypatch.setattr(OmniMultiModalProcessor, "apply", fake_parent_apply)
    inputs = ProcessorInputs(
        prompt=[1, 2],
        mm_data_items=MultiModalDataItems({}),
        hf_processor_mm_kwargs={"ref_text": "reference"},
    )

    assert processor.apply(inputs, timing_ctx=None) == "sentinel"
    assert captured["inputs"].prompt == _PROMPT_IDS
    assert processor.text_tokenizer.call_count == 1
    (text_prompt,), _ = processor.text_tokenizer.call_args
    assert text_prompt.endswith("<|text_start|>reference hello world<|text_end|>")
    assert captured["inputs"].hf_processor_mm_kwargs["ref_text"] == "reference"


def test_media_path_encodes_audio_without_retokenizing(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = _make_processor(monkeypatch)
    audio = np.zeros(2400, dtype=np.float32)
    monkeypatch.setattr(processor, "_get_hf_mm_data", lambda _items: ({"audios": [audio]}, {}))

    out = processor._apply_hf_processor_main(_items({"audio": 1}, {}), {"ref_text": "reference"})

    assert tuple(out["ref_audio_tokens"].shape) == (8, 4)
    assert list(out["ref_audio_len"]) == [4]
    assert "input_ids" not in out
    assert processor.audio_tokenizer.encode.call_count == 1
    assert processor.text_tokenizer.call_count == 0


def test_media_path_with_no_audio_touches_no_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Text-only requests and fully cached audio both reach this path with empty items."""
    processor = _make_processor(monkeypatch)
    monkeypatch.setattr(processor, "_get_hf_mm_data", lambda _items: ({}, {}))

    out = processor._apply_hf_processor_main(_items({}, {}), {"ref_text": "reference"})

    assert len(out.keys()) == 0
    assert processor.text_tokenizer.call_count == 0
    assert processor.audio_tokenizer.encode.call_count == 0


def test_legacy_call_hf_processor_is_gone() -> None:
    assert not hasattr(OmniVoiceMultiModalProcessor, "_call_hf_processor")
