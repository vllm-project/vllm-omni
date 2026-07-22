# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from vllm.multimodal.inputs import MultiModalBatchedField, MultiModalSharedField
from vllm.multimodal.processing.processor import PlaceholderFeaturesInfo

from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_thinker import (
    _PER_VIDEO_USE_AUDIO_IN_VIDEO_KEY,
    Qwen2_5OmniThinkerMultiModalProcessor,
    _coerce_use_audio_in_video_for_hf_processor,
    _filter_video_use_audio_in_video_for_uncached_items,
    _normalize_use_audio_in_video,
)
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerMultiModalProcessor,
)
from vllm_omni.outputs import OmniModelRunnerOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

AUDIO_TOKEN_ID = 10
VIDEO_TOKEN_ID = 20
AUDIO_BOS_TOKEN_ID = 30
AUDIO_EOS_TOKEN_ID = 31
IMAGE_TOKEN_ID = 40


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


def _fake_qwen2_prompt_processor() -> SimpleNamespace:
    tokenizer = SimpleNamespace(
        get_vocab=lambda: {
            "<|audio_pad|>": AUDIO_TOKEN_ID,
            "<|video_pad|>": VIDEO_TOKEN_ID,
            "<|image_pad|>": IMAGE_TOKEN_ID,
        }
    )
    processor = SimpleNamespace(
        audio_token="<|audio_pad|>",
        video_token="<|video_pad|>",
        image_token="<|image_pad|>",
    )
    image_processor = SimpleNamespace(merge_size=1)
    config = SimpleNamespace(audio_token_id=AUDIO_TOKEN_ID)
    info = SimpleNamespace(
        get_hf_config=lambda: config,
        get_hf_processor=lambda **_kwargs: processor,
        get_image_processor=lambda **_kwargs: image_processor,
        get_tokenizer=lambda: tokenizer,
    )

    def omni_get_updates_use_audio_in_video(**_kwargs):
        return [
            AUDIO_BOS_TOKEN_ID,
            VIDEO_TOKEN_ID,
            AUDIO_TOKEN_ID,
            AUDIO_EOS_TOKEN_ID,
        ]

    return SimpleNamespace(
        info=info,
        omni_get_updates_use_audio_in_video=omni_get_updates_use_audio_in_video,
    )


def _fake_processor_with_spatial_merge_size() -> SimpleNamespace:
    hf_config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
    return SimpleNamespace(info=SimpleNamespace(get_hf_config=lambda: hf_config))


def _fake_qwen3_processor() -> SimpleNamespace:
    tokenizer = SimpleNamespace(
        get_vocab=lambda: {
            "<|audio_pad|>": AUDIO_TOKEN_ID,
            "<|video_pad|>": VIDEO_TOKEN_ID,
        }
    )
    processor = SimpleNamespace(
        audio_token="<|audio_pad|>",
        video_token="<|video_pad|>",
    )
    info = SimpleNamespace(
        get_hf_processor=lambda: processor,
        get_tokenizer=lambda: tokenizer,
    )
    return SimpleNamespace(info=info)


def _fake_qwen3_prompt_processor() -> SimpleNamespace:
    tokenizer = SimpleNamespace(
        get_vocab=lambda: {
            "<|audio_pad|>": AUDIO_TOKEN_ID,
            "<|video_pad|>": VIDEO_TOKEN_ID,
            "<|image_pad|>": IMAGE_TOKEN_ID,
        }
    )
    processor = SimpleNamespace(
        audio_token="<|audio_pad|>",
        video_token="<|video_pad|>",
        image_token="<|image_pad|>",
    )
    image_processor = SimpleNamespace(merge_size=1)
    config = SimpleNamespace(
        audio_token_id=AUDIO_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        audio_start_token_id=AUDIO_BOS_TOKEN_ID,
        audio_end_token_id=AUDIO_EOS_TOKEN_ID,
        vision_config=SimpleNamespace(spatial_merge_size=1),
        position_id_per_seconds=1,
    )
    info = SimpleNamespace(
        get_hf_config=lambda: config,
        get_hf_processor=lambda **_kwargs: processor,
        get_image_processor=lambda **_kwargs: image_processor,
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


def test_coerce_use_audio_in_video_for_hf_processor_keeps_global_bool():
    mm_kwargs = {"use_audio_in_video": True}

    result = _coerce_use_audio_in_video_for_hf_processor({"videos": [object(), object()]}, mm_kwargs)

    assert result is mm_kwargs
    assert result["use_audio_in_video"] is True


def test_coerce_use_audio_in_video_for_hf_processor_hides_per_video_mask_from_hf():
    mm_kwargs = {"use_audio_in_video": [True, False], "other": "kept"}

    result = _coerce_use_audio_in_video_for_hf_processor({"videos": [object(), object()]}, mm_kwargs)

    assert result is not mm_kwargs
    assert result["use_audio_in_video"] is False
    assert result[_PER_VIDEO_USE_AUDIO_IN_VIDEO_KEY] == [True, False]
    assert result["other"] == "kept"
    assert mm_kwargs["use_audio_in_video"] == [True, False]


def test_coerce_use_audio_in_video_for_hf_processor_does_not_validate_against_empty_mm_data():
    result = _coerce_use_audio_in_video_for_hf_processor(
        {},
        {"use_audio_in_video": [True, False]},
    )

    assert result["use_audio_in_video"] is False
    assert result[_PER_VIDEO_USE_AUDIO_IN_VIDEO_KEY] == [True, False]


def test_coerce_use_audio_in_video_for_hf_processor_does_not_validate_against_partial_mm_data():
    result = _coerce_use_audio_in_video_for_hf_processor(
        {"videos": [object()]},
        {"use_audio_in_video": [True, False]},
    )

    assert result["use_audio_in_video"] is False
    assert result[_PER_VIDEO_USE_AUDIO_IN_VIDEO_KEY] == [True, False]


def test_filter_video_use_audio_in_video_for_uncached_items_aligns_partial_cache_mask():
    result = _filter_video_use_audio_in_video_for_uncached_items(
        {
            "use_audio_in_video": [True, False, True],
            "second_per_grid_ts": [0.5, 1.0, 1.5],
        },
        [True, False, False],
    )

    assert result["use_audio_in_video"] == [False, True]
    assert result[_PER_VIDEO_USE_AUDIO_IN_VIDEO_KEY] == [False, True]
    assert result["second_per_grid_ts"] == [1.0, 1.5]


def test_filter_video_use_audio_in_video_for_uncached_items_validates_request_mask_length():
    with pytest.raises(ValueError, match="one boolean per video"):
        _filter_video_use_audio_in_video_for_uncached_items(
            {"use_audio_in_video": [True]},
            [False, False],
        )


def test_mm_fields_config_keeps_global_use_audio_in_video_shared():
    fake_self = _fake_processor_with_spatial_merge_size()
    hf_inputs = {
        "video_grid_thw": torch.tensor([[1, 4, 4], [1, 4, 4]]),
        "use_audio_in_video": torch.tensor(False),
    }

    result = Qwen2_5OmniThinkerMultiModalProcessor._get_mm_fields_config(
        fake_self,
        hf_inputs,
        {},
    )

    assert isinstance(result["use_audio_in_video"].field, MultiModalSharedField)


def test_mm_fields_config_uses_batched_field_for_per_video_audio_mask():
    fake_self = _fake_processor_with_spatial_merge_size()
    hf_inputs = {
        "video_grid_thw": torch.tensor([[1, 4, 4], [1, 4, 4]]),
        "use_audio_in_video": torch.tensor([True, False]),
    }

    result = Qwen2_5OmniThinkerMultiModalProcessor._get_mm_fields_config(
        fake_self,
        hf_inputs,
        {_PER_VIDEO_USE_AUDIO_IN_VIDEO_KEY: [True, False]},
    )

    assert isinstance(result["use_audio_in_video"].field, MultiModalBatchedField)


def test_mm_fields_config_uses_batched_field_for_vector_hf_audio_mask():
    fake_self = _fake_processor_with_spatial_merge_size()
    hf_inputs = {
        "video_grid_thw": torch.tensor([[1, 4, 4], [1, 4, 4]]),
        "use_audio_in_video": torch.tensor([True, False]),
    }

    result = Qwen2_5OmniThinkerMultiModalProcessor._get_mm_fields_config(
        fake_self,
        hf_inputs,
        {},
    )

    assert isinstance(result["use_audio_in_video"].field, MultiModalBatchedField)


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


def test_get_video_use_audio_in_video_falls_back_for_partial_cache_hit_with_explicit_mask():
    fake_self = _fake_processor()
    update_with_audio = SimpleNamespace(content=SimpleNamespace(full=[VIDEO_TOKEN_ID, AUDIO_TOKEN_ID]))
    mm_kwargs = {
        "video": [
            {"use_audio_in_video": _Feature(torch.tensor(True))},
            None,
        ]
    }

    result = Qwen2_5OmniThinkerMultiModalProcessor._get_video_use_audio_in_video(
        fake_self,
        mm_kwargs,
        {"video": [[object()], [update_with_audio]]},
    )

    assert result == [True, True]


def test_qwen2_5_prompt_updates_apply_audio_mask_per_video():
    fake_self = _fake_qwen2_prompt_processor()
    out_mm_kwargs = SimpleNamespace(
        get_data=lambda **_kwargs: {
            "audio_feature_lengths": torch.tensor([100]),
            "video_grid_thw": torch.tensor([[1, 1, 2], [1, 1, 2]]),
        }
    )
    mm_items = SimpleNamespace(
        get_all_counts=lambda: {"video": 2, "audio": 1},
        get_items=lambda *_args, **_kwargs: SimpleNamespace(get=lambda _idx: [0] * 100),
    )

    updates = Qwen2_5OmniThinkerMultiModalProcessor._get_prompt_updates(
        fake_self,
        mm_items,
        {"use_audio_in_video": [True, False]},
        out_mm_kwargs,
    )
    video_update = next(update for update in updates if update.modality == "video")

    assert video_update.resolve(0).content.full == [
        AUDIO_BOS_TOKEN_ID,
        VIDEO_TOKEN_ID,
        AUDIO_TOKEN_ID,
        AUDIO_EOS_TOKEN_ID,
    ]
    assert video_update.resolve(1).content.full == [VIDEO_TOKEN_ID, VIDEO_TOKEN_ID]


def test_qwen2_5_prompt_updates_prefer_hf_second_per_grid_ts_from_outputs():
    captured = {}
    fake_self = _fake_qwen2_prompt_processor()

    def omni_get_updates_use_audio_in_video(**kwargs):
        captured["video_second_per_grid_t"] = kwargs["video_second_per_grid_t"]
        return [VIDEO_TOKEN_ID]

    fake_self.omni_get_updates_use_audio_in_video = omni_get_updates_use_audio_in_video
    out_mm_kwargs = SimpleNamespace(
        get_data=lambda **_kwargs: {
            "audio_feature_lengths": torch.tensor([100]),
            "video_grid_thw": torch.tensor([[1, 1, 1]]),
            "second_per_grid_ts": torch.tensor([0.25]),
        }
    )
    mm_items = SimpleNamespace(
        get_all_counts=lambda: {"video": 1, "audio": 1},
        get_items=lambda *_args, **_kwargs: SimpleNamespace(get=lambda _idx: [0] * 100),
    )

    updates = Qwen2_5OmniThinkerMultiModalProcessor._get_prompt_updates(
        fake_self,
        mm_items,
        {"use_audio_in_video": [True], "second_per_grid_ts": torch.tensor([0.5])},
        out_mm_kwargs,
    )
    video_update = next(update for update in updates if update.modality == "video")
    video_update.resolve(0)

    assert captured["video_second_per_grid_t"] == 0.25


def test_qwen2_5_prompt_updates_accept_tensor_second_per_grid_ts_from_kwargs():
    captured = {}
    fake_self = _fake_qwen2_prompt_processor()

    def omni_get_updates_use_audio_in_video(**kwargs):
        captured["video_second_per_grid_t"] = kwargs["video_second_per_grid_t"]
        return [VIDEO_TOKEN_ID]

    fake_self.omni_get_updates_use_audio_in_video = omni_get_updates_use_audio_in_video
    out_mm_kwargs = SimpleNamespace(
        get_data=lambda **_kwargs: {
            "audio_feature_lengths": torch.tensor([100]),
            "video_grid_thw": torch.tensor([[1, 1, 1]]),
        }
    )
    mm_items = SimpleNamespace(
        get_all_counts=lambda: {"video": 1, "audio": 1},
        get_items=lambda *_args, **_kwargs: SimpleNamespace(get=lambda _idx: [0] * 100),
    )

    updates = Qwen2_5OmniThinkerMultiModalProcessor._get_prompt_updates(
        fake_self,
        mm_items,
        {"use_audio_in_video": [True], "second_per_grid_ts": torch.tensor([0.75, 1.0])},
        out_mm_kwargs,
    )
    video_update = next(update for update in updates if update.modality == "video")
    video_update.resolve(0)

    assert captured["video_second_per_grid_t"] == 0.75


def test_qwen2_5_prompt_updates_validate_mask_against_request_video_count():
    fake_self = _fake_qwen2_prompt_processor()
    out_mm_kwargs = SimpleNamespace(
        get_data=lambda **_kwargs: {
            "video_grid_thw": torch.tensor([[1, 1, 2], [1, 1, 2]]),
        }
    )
    mm_items = SimpleNamespace(get_all_counts=lambda: {"video": 2})

    with pytest.raises(ValueError, match="one boolean per video"):
        Qwen2_5OmniThinkerMultiModalProcessor._get_prompt_updates(
            fake_self,
            mm_items,
            {"use_audio_in_video": [True]},
            out_mm_kwargs,
        )


def test_qwen2_5_mm_only_dummy_counts_subtract_only_videos_using_audio():
    captured_counts = {}
    fake_self = SimpleNamespace(
        dummy_inputs=SimpleNamespace(get_dummy_text=lambda counts: captured_counts.update(counts) or "dummy"),
        _apply_hf_processor_text_mm=lambda **_kwargs: ([], {}, False),
    )
    mm_items = SimpleNamespace(get_all_counts=lambda: {"video": 2, "audio": 1})

    Qwen2_5OmniThinkerMultiModalProcessor._apply_hf_processor_mm_only(
        fake_self,
        mm_items,
        {"use_audio_in_video": [True, False]},
        {},
    )

    assert captured_counts == {"video": 2, "audio": 0}


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


def test_qwen3_prompt_updates_prefer_hf_second_per_grid_ts_from_outputs():
    captured = {}
    fake_self = _fake_qwen3_prompt_processor()

    def get_updates_use_audio_in_video(**kwargs):
        captured["video_second_per_grid_t"] = kwargs["video_second_per_grid_t"]
        return [VIDEO_TOKEN_ID]

    fake_self.get_updates_use_audio_in_video = get_updates_use_audio_in_video
    out_mm_kwargs = SimpleNamespace(
        get_data=lambda **_kwargs: {
            "audio_feature_lengths": torch.tensor([100]),
            "video_grid_thw": torch.tensor([[1, 1, 1]]),
            "second_per_grid_ts": torch.tensor([0.25]),
        }
    )
    mm_items = SimpleNamespace(
        get_items=lambda *_args, **_kwargs: SimpleNamespace(get=lambda _idx: [0] * 100),
    )

    updates = Qwen3OmniMoeThinkerMultiModalProcessor._get_prompt_updates(
        fake_self,
        mm_items,
        {"use_audio_in_video": [True], "second_per_grid_ts": torch.tensor([0.5])},
        out_mm_kwargs,
    )
    video_update = next(update for update in updates if update.modality == "video")
    video_update.resolve(0)

    assert captured["video_second_per_grid_t"] == 0.25


def test_qwen3_prompt_updates_accept_tensor_second_per_grid_ts_from_kwargs():
    captured = {}
    fake_self = _fake_qwen3_prompt_processor()

    def get_updates_use_audio_in_video(**kwargs):
        captured["video_second_per_grid_t"] = kwargs["video_second_per_grid_t"]
        return [VIDEO_TOKEN_ID]

    fake_self.get_updates_use_audio_in_video = get_updates_use_audio_in_video
    out_mm_kwargs = SimpleNamespace(
        get_data=lambda **_kwargs: {
            "audio_feature_lengths": torch.tensor([100]),
            "video_grid_thw": torch.tensor([[1, 1, 1]]),
        }
    )
    mm_items = SimpleNamespace(
        get_items=lambda *_args, **_kwargs: SimpleNamespace(get=lambda _idx: [0] * 100),
    )

    updates = Qwen3OmniMoeThinkerMultiModalProcessor._get_prompt_updates(
        fake_self,
        mm_items,
        {"use_audio_in_video": [True], "second_per_grid_ts": torch.tensor([0.75, 1.0])},
        out_mm_kwargs,
    )
    video_update = next(update for update in updates if update.modality == "video")
    video_update.resolve(0)

    assert captured["video_second_per_grid_t"] == 0.75


def test_qwen3_derive_audio_from_video_placeholders_excludes_audio_wrappers_from_embeds():
    fake_self = _fake_qwen3_processor()
    video_placeholders = [
        PlaceholderFeaturesInfo(
            modality="video",
            item_idx=0,
            start_idx=5,
            tokens=[
                AUDIO_BOS_TOKEN_ID,
                VIDEO_TOKEN_ID,
                AUDIO_TOKEN_ID,
                AUDIO_EOS_TOKEN_ID,
                VIDEO_TOKEN_ID,
            ],
            is_embed=torch.ones(5, dtype=torch.bool),
        )
    ]

    result = Qwen3OmniMoeThinkerMultiModalProcessor._derive_audio_from_video_placeholders(
        fake_self,
        {"video": video_placeholders},
        {"audio": [[object()]]},
        [True],
    )

    assert result["audio"][0].is_embed.tolist() == [
        False,
        False,
        True,
        False,
        False,
    ]
    assert result["video"][0].is_embed.tolist() == [
        False,
        True,
        False,
        False,
        True,
    ]


def test_qwen3_mrope_positions_match_plain_video_placeholder_length():
    fake_self = SimpleNamespace(
        iter_mm_features=lambda _: iter(
            [
                (
                    2,
                    "video",
                    {
                        "grid_t": 1,
                        "grid_h": 2,
                        "grid_w": 2,
                        "t_factor": 1.0,
                        "use_audio_in_video": False,
                        "placeholder_len": 4,
                    },
                )
            ]
        )
    )

    positions, _ = Qwen3OmniMoeThinkerForConditionalGeneration.get_mrope_input_positions(
        fake_self,
        list(range(8)),
        [],
    )

    assert positions.shape == (3, 8)


def test_qwen3_mrope_positions_count_single_interleaved_audio_wrappers():
    fake_self = SimpleNamespace(
        iter_mm_features=lambda _: iter(
            [
                (
                    2,
                    "video",
                    {
                        "grid_t": 1,
                        "grid_h": 2,
                        "grid_w": 2,
                        "t_factor": 1.0,
                        "use_audio_in_video": True,
                        "placeholder_len": 8,
                    },
                )
            ]
        ),
        _compute_interleaved_positions=lambda _start_idx, _data: (
            torch.zeros((3, 6), dtype=torch.long).numpy(),
            6,
        ),
    )

    positions, _ = Qwen3OmniMoeThinkerForConditionalGeneration.get_mrope_input_positions(
        fake_self,
        list(range(10)),
        [],
    )

    assert positions.shape == (3, 10)


def test_qwen3_interleaved_mrope_prefers_placeholder_length_for_audio_count():
    fake_self = SimpleNamespace(
        _compute_audio_token_count=lambda _audio_feature_length: 100,
    )
    data = {
        "grid_t": 1,
        "grid_h": 2,
        "grid_w": 2,
        "t_factor": 1.0,
        "audio_feature_length": 999,
        "placeholder_len": 8,
    }

    positions, token_count = Qwen3OmniMoeThinkerForConditionalGeneration._compute_interleaved_positions(
        fake_self,
        0,
        data,
    )

    assert token_count == 6
    assert positions.shape == (3, 6)


def test_omni_model_runner_output_can_carry_kv_connector_output_only():
    marker = object()

    output = OmniModelRunnerOutput.with_kv_conn_output_only(marker)

    assert output.req_ids == []
    assert output.req_id_to_index == {}
    assert output.sampled_token_ids == []
    assert output.kv_connector_output is marker
