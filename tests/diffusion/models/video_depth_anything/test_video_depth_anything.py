# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for Video Depth Anything: checkpoint recognition, preprocessing, windowing and stitching.

Numerical parity with the reference implementation needs its code and real
weights; see ``recipes/depth-anything/Video-Depth-Anything-H200.md``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.video_depth_anything import pipeline_video_depth_anything as P
from vllm_omni.diffusion.models.video_depth_anything.checkpoints import (
    HUB_CHECKPOINTS,
    checkpoint_variant,
    resolve_checkpoint,
)
from vllm_omni.diffusion.models.video_depth_anything.vda_model import VideoDepthAnything, VideoDepthAnythingConfig
from vllm_omni.diffusion.utils.hf_utils import resolve_native_diffusion_model_class

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_released_checkpoints_are_recognized_by_name(tmp_path):
    assert len(HUB_CHECKPOINTS) == 6
    assert resolve_checkpoint("depth-anything/Video-Depth-Anything-Small") == "video_depth_anything_vits.pth"
    assert resolve_checkpoint("depth-anything/metric-video-depth-anything-large/") == (
        "metric_video_depth_anything_vitl.pth"
    )
    assert checkpoint_variant("metric_video_depth_anything_vitb.pth") == ("vitb", True)
    assert checkpoint_variant("/models/video_depth_anything_vits.pth") == ("vits", False)
    with pytest.raises(ValueError, match="not a released"):
        checkpoint_variant("video_depth_anything_vitg.pth")
    assert resolve_native_diffusion_model_class("depth-anything/Video-Depth-Anything-Large") == (
        "VideoDepthAnythingPipeline"
    )
    # A local copy is recognized by its single checkpoint; several variants are ambiguous.
    (tmp_path / "video_depth_anything_vitb.pth").touch()
    assert resolve_checkpoint(str(tmp_path)) == "video_depth_anything_vitb.pth"
    (tmp_path / "metric_video_depth_anything_vitb.pth").touch()
    assert resolve_checkpoint(str(tmp_path)) is None
    assert resolve_checkpoint("depth-anything/Depth-Anything-V2-Small") is None


@pytest.mark.parametrize(
    ("height", "width", "input_size", "expected"),
    [
        # Expected sizes come from the reference ``Resize`` (lower bound, multiple of 14).
        (540, 960, 518, (518, 924)),
        (720, 1920, 518, (350, 938)),  # wider than 16:9: the reference shrinks input_size first
        (1080, 1080, 518, (518, 518)),
        (480, 640, 392, (392, 518)),
        (1280, 720, 518, (924, 518)),
    ],
)
def test_network_size_follows_the_reference_resize(height, width, input_size, expected):
    assert P.network_size(height, width, input_size) == expected


def test_options_are_validated():
    assert P.parse_options({}) == (518, 1280)
    assert P.parse_options({"input_size": 392, "max_res": -1}) == (392, -1)
    for extra, message in (
        ({"input_size": 500}, "multiple of 14"),
        ({"input_size": 1036}, "multiple of 14"),
        ({"input_size": True}, "integer"),
        ({"max_res": 0}, "max_res"),
    ):
        with pytest.raises(ValueError, match=message):
            P.parse_options(extra)


def test_scale_and_shift_is_the_least_squares_fit():
    prediction = np.linspace(0.5, 3.0, 64, dtype=np.float32).reshape(8, 8)
    scale, shift = P.scale_and_shift(prediction, 2.5 * prediction - 0.75)
    assert scale == pytest.approx(2.5, rel=1e-5) and shift == pytest.approx(-0.75, abs=1e-5)
    assert P.scale_and_shift(np.ones((2, 2), np.float32), np.ones((2, 2), np.float32)) == (1.0, 0.0)


def _pipeline(*, metric: bool = False) -> P.VideoDepthAnythingPipeline:
    pipeline = object.__new__(P.VideoDepthAnythingPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.config = VideoDepthAnythingConfig("vits", metric)
    pipeline.model = VideoDepthAnything(pipeline.config).float().eval()
    pipeline.od_config = SimpleNamespace(dtype=torch.float32)
    return pipeline


@pytest.mark.parametrize(("frames", "windows"), [(10, 1), (22, 1), (23, 2), (32, 2), (54, 3), (70, 4)])
def test_windows_cover_the_clip_like_the_reference(frames, windows, monkeypatch):
    """The reference runs one 32-frame window per 22 frames, padding the clip with its last frame."""
    pipeline = _pipeline()
    seen = []

    def window_depth(inputs, height, width):
        # Encode each input frame's identity (its constant pixel value) as its depth.
        seen.append(inputs[0, :, 0, 0, 0].clone())
        return inputs[0, :, :1].mean(dim=(2, 3), keepdim=True).expand(-1, -1, height, width)[:, 0].numpy()

    monkeypatch.setattr(pipeline, "_window_depth", window_depth)
    video = np.stack([np.full((4, 6, 3), 3 * index, np.uint8) for index in range(frames)])
    result = pipeline.estimate(video, input_size=14, max_res=-1)
    assert len(seen) == windows and result["num_frames"] == frames
    assert result["depth"].shape == (frames, 4, 6) and result["depth"].dtype == np.float16
    if windows > 1:
        # Window 1 starts with window 0's keyframes: its inputs 0, 12 and 24-31.
        torch.testing.assert_close(seen[1][:10], seen[0][[0, 12, *range(24, 32)]])


def test_relative_windows_are_aligned_and_cross_faded():
    pipeline = _pipeline()
    # Window 1 sees the same scene at twice the scale plus one: alignment must undo it.
    truth = np.arange(54, dtype=np.float32)[:, None, None] * np.ones((1, 2, 3), np.float32) + 1
    first = truth[:32]
    second = np.concatenate([truth[[0, 12]], truth[24:32], truth[32:54]]) * 2 + 1
    aligned = pipeline._align([first, second], 54)
    np.testing.assert_allclose(aligned, truth, rtol=1e-5, atol=1e-4)


def test_metric_windows_are_stitched_without_rescaling():
    pipeline = _pipeline(metric=True)
    first = np.full((32, 2, 2), 2.0, np.float32)
    second = np.full((32, 2, 2), 4.0, np.float32)
    aligned = pipeline._align([first, second], 54)
    # Frames 24-31 fade from the first window's 2.0 to the second's 4.0; frames 32+ keep 4.0 unscaled.
    np.testing.assert_allclose(aligned[24:32, 0, 0], 2.0 + 2.0 * np.arange(8) / 7, rtol=1e-6)
    assert (aligned[32:] == 4.0).all() and (aligned[:24] == 2.0).all()


def test_pipeline_runs_end_to_end_on_random_weights():
    pipeline = _pipeline()
    video = np.random.default_rng(0).integers(0, 255, (35, 20, 30, 3), dtype=np.uint8)
    result = pipeline.estimate(video, input_size=28, max_res=-1)
    assert result["depth"].shape == (35, 20, 30) and result["metric"] is False
    assert np.isfinite(result["depth"]).all() and (result["depth"] >= 0).all()


def test_forward_rejects_requests_without_a_video():
    pipeline = _pipeline()
    request = SimpleNamespace(prompts=[{"prompt": ""}], sampling_params=SimpleNamespace(extra_args={}))
    output = pipeline.forward(request)
    assert output.error_status_code == 400 and "requires an input video" in output.error
    request = SimpleNamespace(
        prompts=[{"multi_modal_data": {"video": np.zeros((2, 4, 6, 3), np.uint8)}}],
        sampling_params=SimpleNamespace(extra_args={"input_size": 500}),
    )
    output = pipeline.forward(request)
    assert output.error_status_code == 400 and "multiple of 14" in output.error
