# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3 video profiling uses the native dummy builder's sizing contract."""

from types import SimpleNamespace

import pytest
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import Qwen3OmniMoeProcessorKwargs
from vllm.config.multimodal import VideoDummyOptions
from vllm.multimodal.parse import ImageSize

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerDummyInputsBuilder,
    Qwen3OmniMoeThinkerProcessingInfo,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def video_dummy_builder(mocker):
    info = object.__new__(Qwen3OmniMoeThinkerProcessingInfo)
    video_processor = SimpleNamespace(size={"shortest_edge": 784, "longest_edge": 50176}, do_resize=True)
    processor = SimpleNamespace(
        feature_extractor=SimpleNamespace(chunk_length=30, sampling_rate=16000),
        video_processor=video_processor,
        tokenizer=SimpleNamespace(init_kwargs={}),
        _merge_kwargs=mocker.Mock(return_value={"videos_kwargs": {"size": video_processor.size}}),
    )
    info.ctx = SimpleNamespace(get_merged_mm_kwargs=lambda _: {})
    mocker.patch.object(info, "get_hf_processor", return_value=processor)
    mocker.patch.object(
        info,
        "get_hf_config",
        return_value=SimpleNamespace(vision_config=SimpleNamespace(patch_size=14, spatial_merge_size=2)),
    )
    mocker.patch.object(info, "get_image_size_with_most_features", return_value=ImageSize(width=224, height=224))
    mocker.patch.object(info, "get_num_frames_with_most_features", return_value=2)
    builder = Qwen3OmniMoeThinkerDummyInputsBuilder(info)
    for name in ("_get_dummy_audios", "_get_dummy_images", "_get_dummy_videos"):
        mocker.patch.object(builder, name, return_value=[])
    return builder, processor


@pytest.mark.parametrize(
    ("dimensions", "expected_size"),
    [
        ({"width": 112}, (112, 224)),
        ({"height": 84}, (224, 84)),
        ({"width": 112, "height": 84}, (112, 84)),
    ],
    ids=["width-only", "height-only", "width-and-height"],
)
def test_video_dummy_builder_honors_dimension_overrides(video_dummy_builder, dimensions, expected_size):
    builder, _ = video_dummy_builder
    options = VideoDummyOptions(count=1, num_frames=2, **dimensions)

    builder.get_dummy_mm_data(128, {"video": 1}, {"video": options})

    received = builder._get_dummy_videos.call_args.kwargs
    assert (received["width"], received["height"]) == expected_size
    assert received["num_videos"] == 1
    assert received["num_frames"] == 2
    assert received["overrides"].count == 1
    assert received["overrides"].num_frames == 2
    assert received["overrides"].width is None
    assert received["overrides"].height is None


@pytest.mark.parametrize("do_resize,expected_size", [(False, (111, 83)), (True, (112, 84))])
def test_video_frame_size_respects_do_resize(video_dummy_builder, do_resize, expected_size):
    builder, processor = video_dummy_builder
    processor.video_processor.do_resize = not do_resize
    processor._merge_kwargs.return_value["videos_kwargs"]["do_resize"] = do_resize

    size = builder.info.get_video_frame_size(width=111, height=83)

    assert (size.width, size.height) == expected_size


def test_video_dummy_builder_preserves_default_sizing_and_qwen3_kwargs(video_dummy_builder):
    builder, processor = video_dummy_builder
    options = VideoDummyOptions(count=1, num_frames=2)

    builder.get_dummy_mm_data(128, {"video": 1}, {"video": options})

    received = builder._get_dummy_videos.call_args.kwargs
    assert (received["width"], received["height"]) == (224, 224)
    assert received["overrides"] is options
    builder.info.get_image_size_with_most_features.assert_any_call(max_pixels=50176)
    assert processor._merge_kwargs.call_args.args[0] is Qwen3OmniMoeProcessorKwargs
    assert builder.info.processor_kwargs_cls is Qwen3OmniMoeProcessorKwargs
