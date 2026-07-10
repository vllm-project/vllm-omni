# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from vllm.multimodal.processing.processor import PlaceholderFeaturesInfo

from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_thinker import (
    Qwen2_5OmniThinkerMultiModalProcessor,
    _normalize_use_audio_in_video,
)
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerMultiModalProcessor,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

AUDIO_TOKEN_ID = 10
VIDEO_TOKEN_ID = 20


class _Feature:
    def __init__(self, data: torch.Tensor) -> None:
        self.data = data


def _fake_processor() -> SimpleNamespace:
    tokenizer = SimpleNamespace(get_vocab=lambda: {"<|audio_pad|>": AUDIO_TOKEN_ID})
    processor = SimpleNamespace(audio_token="<|audio_pad|>")
    config = SimpleNamespace(audio_token_id=AUDIO_TOKEN_ID)
    info = SimpleNamespace(
        get_hf_config=lambda: config,
        get_hf_processor=lambda: processor,
        get_tokenizer=lambda: tokenizer,
    )
    return SimpleNamespace(info=info)


def test_normalize_use_audio_in_video_accepts_bool_and_per_video_mask():
    assert _normalize_use_audio_in_video(True, 2) == [True, True]
    assert _normalize_use_audio_in_video(False, 2) == [False, False]
    assert _normalize_use_audio_in_video([True, False], 2) == [True, False]
    assert _normalize_use_audio_in_video(torch.tensor([1, 0]), 2) == [
        True,
        False,
    ]


def test_normalize_use_audio_in_video_rejects_wrong_length():
    with pytest.raises(ValueError, match="one boolean per video"):
        _normalize_use_audio_in_video([True], 2)


def test_get_video_use_audio_in_video_preserves_explicit_false():
    fake_self = _fake_processor()
    mm_kwargs = {
        "video": [
            {"use_audio_in_video": _Feature(torch.tensor(True))},
            {"use_audio_in_video": _Feature(torch.tensor(False))},
        ]
    }

    result = Qwen2_5OmniThinkerMultiModalProcessor._get_video_use_audio_in_video(
        fake_self,
        mm_kwargs,
        {},
    )

    assert result == [True, False]


def test_get_video_use_audio_in_video_treats_missing_item_as_false_with_explicit_mask():
    fake_self = _fake_processor()
    mm_kwargs = {
        "video": [
            {"use_audio_in_video": _Feature(torch.tensor(True))},
            None,
        ]
    }

    result = Qwen2_5OmniThinkerMultiModalProcessor._get_video_use_audio_in_video(
        fake_self,
        mm_kwargs,
        {},
    )

    assert result == [True, False]


def test_get_video_use_audio_in_video_falls_back_to_prompt_updates_for_cache_hits():
    fake_self = _fake_processor()
    update_with_audio = SimpleNamespace(content=SimpleNamespace(full=[VIDEO_TOKEN_ID, AUDIO_TOKEN_ID]))
    update_without_audio = SimpleNamespace(content=SimpleNamespace(full=[VIDEO_TOKEN_ID]))

    result = Qwen2_5OmniThinkerMultiModalProcessor._get_video_use_audio_in_video(
        fake_self,
        {"video": [None, None]},
        {"video": [[update_with_audio], [update_without_audio]]},
    )

    assert result == [True, False]


def test_derive_audio_from_video_placeholders_only_pairs_true_videos():
    fake_self = _fake_processor()
    video_placeholders = [
        PlaceholderFeaturesInfo(
            modality="video",
            item_idx=0,
            start_idx=5,
            tokens=[VIDEO_TOKEN_ID, AUDIO_TOKEN_ID, VIDEO_TOKEN_ID],
            is_embed=torch.tensor([True, True, True]),
        ),
        PlaceholderFeaturesInfo(
            modality="video",
            item_idx=1,
            start_idx=12,
            tokens=[VIDEO_TOKEN_ID, VIDEO_TOKEN_ID],
            is_embed=torch.tensor([True, True]),
        ),
    ]

    result = Qwen2_5OmniThinkerMultiModalProcessor._derive_audio_from_video_placeholders(
        fake_self,
        {"video": video_placeholders},
        {"audio": [[object()]]},
        [True, False],
    )

    assert len(result["audio"]) == 1
    assert result["audio"][0].item_idx == 0
    assert result["audio"][0].start_idx == 5
    assert result["audio"][0].is_embed.tolist() == [False, True, False]

    assert len(result["video"]) == 2
    assert result["video"][0].item_idx == 0
    assert result["video"][0].is_embed.tolist() == [True, False, True]
    assert result["video"][1].item_idx == 1
    assert result["video"][1].is_embed.tolist() == [True, True]


def test_qwen3_processor_inherits_vllm_omni_per_video_helpers():
    assert issubclass(
        Qwen3OmniMoeThinkerMultiModalProcessor,
        Qwen2_5OmniThinkerMultiModalProcessor,
    )
