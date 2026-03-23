# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniThinkerMultiModalProcessor as UpstreamQwen2_5OmniThinkerMultiModalProcessor,
)
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.processor import PlaceholderFeaturesInfo

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerMultiModalProcessor,
)


class _DummyTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {"<|audio_pad|>": 2}


class _DummyHFProcessor:
    audio_token = "<|audio_pad|>"


class _DummyInfo:
    def get_tokenizer(self) -> _DummyTokenizer:
        return _DummyTokenizer()

    def get_hf_processor(self) -> _DummyHFProcessor:
        return _DummyHFProcessor()


def test_normalize_mm_item_counts_for_use_audio_in_video():
    normalized = Qwen3OmniMoeThinkerMultiModalProcessor._normalize_mm_item_counts(
        {"video": 2},
        use_audio_in_video=True,
    )
    assert normalized == {"video": 2, "audio": 2}

    unchanged = Qwen3OmniMoeThinkerMultiModalProcessor._normalize_mm_item_counts(
        {"video": 2},
        use_audio_in_video=False,
    )
    assert unchanged == {"video": 2}


def test_derive_audio_placeholders_when_audio_count_missing():
    processor = object.__new__(Qwen3OmniMoeThinkerMultiModalProcessor)
    processor.info = _DummyInfo()

    placeholders = {
        "video": [
            PlaceholderFeaturesInfo(
                modality="video",
                item_idx=0,
                start_idx=0,
                tokens=[10, 2, 11],
                is_embed=None,
            )
        ]
    }

    result = processor._derive_audio_from_video_placeholders(
        placeholders,
        {"video": 1},
    )
    assert "audio" in result
    assert len(result["audio"]) == 1
    assert torch.equal(result["audio"][0].is_embed, torch.tensor([False, True, False]))


def test_maybe_apply_prompt_updates_normalizes_counts(monkeypatch):
    processor = object.__new__(Qwen3OmniMoeThinkerMultiModalProcessor)
    processor.info = _DummyInfo()

    placeholders = {
        "video": [
            PlaceholderFeaturesInfo(
                modality="video",
                item_idx=0,
                start_idx=0,
                tokens=[10, 2, 11],
                is_embed=None,
            )
        ]
    }

    class _DummyItems:
        def get_all_counts(self):
            return {"video": 1}

    validate_calls: list[dict[str, int]] = []

    def fake_validate_mm_kwargs(mm_kwargs, mm_item_counts):
        validate_calls.append(dict(mm_item_counts))

    processor._validate_mm_kwargs = fake_validate_mm_kwargs
    processor._validate_mm_placeholders = lambda mm_placeholders, mm_item_counts: None
    processor._apply_prompt_updates = lambda prompt_ids, updates: (prompt_ids, placeholders)
    monkeypatch.setattr(
        Qwen3OmniMoeThinkerMultiModalProcessor,
        "_find_mm_placeholders",
        lambda self, prompt_ids, mm_prompt_updates: placeholders,
    )

    video_item = {"use_audio_in_video": SimpleNamespace(data=torch.tensor([True]))}
    _, result_placeholders = processor._maybe_apply_prompt_updates(
        mm_items=_DummyItems(),
        prompt_ids=[1, 2, 3],
        mm_kwargs={"video": [video_item]},
        mm_prompt_updates={"audio": [], "video": [[1]]},
        is_update_applied=False,
    )

    assert validate_calls == [{"video": 1, "audio": 1}]
    assert "audio" in result_placeholders
    assert len(result_placeholders["audio"]) == 1


def test_cached_apply_hf_processor_bypasses_cache_for_use_audio_in_video(monkeypatch):
    processor = object.__new__(Qwen3OmniMoeThinkerMultiModalProcessor)
    called: dict[str, object] = {}

    def fake_apply_hf_processor(*, inputs, timing_ctx):
        called["apply"] = {
            "inputs": inputs,
            "timing_ctx": timing_ctx,
        }
        return [1], "apply", False

    def fake_parent_cached_apply_hf_processor(self, inputs, timing_ctx):
        called["parent"] = True
        return [2], "parent", False

    processor._apply_hf_processor = fake_apply_hf_processor
    monkeypatch.setattr(
        UpstreamQwen2_5OmniThinkerMultiModalProcessor,
        "_cached_apply_hf_processor",
        fake_parent_cached_apply_hf_processor,
    )

    inputs = SimpleNamespace(hf_processor_mm_kwargs={"use_audio_in_video": [torch.tensor([True])]})
    output = processor._cached_apply_hf_processor(
        inputs=inputs,
        timing_ctx=TimingContext(enabled=False),
    )

    assert output == ([1], "apply", False)
    assert "parent" not in called
    assert called["apply"]["inputs"] is inputs


def test_cached_apply_hf_processor_uses_parent_when_not_use_audio_in_video(monkeypatch):
    processor = object.__new__(Qwen3OmniMoeThinkerMultiModalProcessor)
    called: dict[str, object] = {}

    def fake_apply_hf_processor(*, inputs, timing_ctx):
        raise AssertionError("_apply_hf_processor should not be called when use_audio_in_video=False")

    def fake_parent_cached_apply_hf_processor(self, inputs, timing_ctx):
        called["inputs"] = inputs
        called["timing_ctx"] = timing_ctx
        return [2], "parent", True

    processor._apply_hf_processor = fake_apply_hf_processor
    monkeypatch.setattr(
        UpstreamQwen2_5OmniThinkerMultiModalProcessor,
        "_cached_apply_hf_processor",
        fake_parent_cached_apply_hf_processor,
    )

    inputs = SimpleNamespace(hf_processor_mm_kwargs={"use_audio_in_video": False})
    timing_ctx = TimingContext(enabled=False)
    output = processor._cached_apply_hf_processor(inputs=inputs, timing_ctx=timing_ctx)

    assert output == ([2], "parent", True)
    assert called["inputs"] is inputs
    assert called["timing_ctx"] is timing_ctx
