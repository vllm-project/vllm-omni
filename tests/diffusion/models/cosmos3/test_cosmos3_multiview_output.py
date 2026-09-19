# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import sys
import types
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers.video_processor import VideoProcessor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decode_releases_each_camera_before_decoding_the_next(monkeypatch, dtype):
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import Cosmos3MultiviewPipeline

    pipeline = object.__new__(Cosmos3MultiviewPipeline)
    torch.nn.Module.__init__(pipeline)
    latents = torch.arange(144, dtype=dtype).reshape(1, 3, 6, 2, 4)
    expected = latents.repeat_interleave(2, dim=2)[..., ::2]
    decoded_refs = []

    def decode(view_latents):
        assert all(ref() is None for ref in decoded_refs), "Previous camera decode is still live"
        decoded = view_latents.repeat_interleave(2, dim=2)[..., ::2]
        assert not decoded.is_contiguous()
        decoded_refs.append(weakref.ref(decoded))
        return decoded

    def reject_cat(*args, **kwargs):
        pytest.fail("Decoded cameras must be copied into their final buffer without concatenation")

    pipeline._decode_latents = decode
    monkeypatch.setattr(torch, "cat", reject_cat)
    actual = pipeline._decode_multiview_latents(latents, num_views=3, latent_frames_per_view=2)

    assert len(decoded_refs) == 3
    assert all(ref() is None for ref in decoded_refs)
    assert actual.device.type == "cpu"
    assert actual.dtype == dtype
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("shape", "num_views", "latent_frames_per_view"),
    [((1, 3, 6, 2, 2), 0, 2), ((1, 3, 6, 2, 2), 3, 0), ((1, 3, 5, 2, 2), 3, 2), ((3, 6, 2, 2), 3, 2)],
)
def test_decode_rejects_invalid_camera_geometry(shape, num_views, latent_frames_per_view):
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import Cosmos3MultiviewPipeline

    pipeline = object.__new__(Cosmos3MultiviewPipeline)
    with pytest.raises(ValueError, match="camera-major"):
        pipeline._decode_multiview_latents(
            torch.zeros(shape), num_views=num_views, latent_frames_per_view=latent_frames_per_view
        )


@pytest.mark.parametrize("mismatch", ["shape", "dtype"])
def test_decode_rejects_inconsistent_camera_outputs(mismatch):
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import Cosmos3MultiviewPipeline

    pipeline = object.__new__(Cosmos3MultiviewPipeline)
    torch.nn.Module.__init__(pipeline)
    clips = iter(
        [torch.zeros(1, 3, 4, 2, 2), torch.zeros(1, 3, 1, 2, 2)]
        if mismatch == "shape"
        else [torch.zeros(1, 3, 4, 2, 2), torch.zeros(1, 3, 4, 2, 2, dtype=torch.bfloat16)]
    )
    pipeline._decode_latents = lambda _: next(clips)
    with pytest.raises(ValueError, match="matching shapes and dtypes"):
        pipeline._decode_multiview_latents(torch.zeros(1, 3, 4, 2, 2), num_views=2, latent_frames_per_view=2)


@pytest.fixture
def guardrails(monkeypatch):
    module = types.ModuleType("vllm_omni.diffusion.models.cosmos3.guardrails")
    module.is_guardrails_enabled = lambda *_: False
    module.check_video_safety = lambda video: video
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


@pytest.mark.parametrize("output_type", ["np", "pt", "pil"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("guardrails_enabled", [False, True])
@pytest.mark.parametrize("num_views", [1, 3])
def test_postprocess_preserves_output_with_one_camera_live_at_a_time(
    monkeypatch, guardrails, output_type, dtype, guardrails_enabled, num_views
):
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import get_cosmos3_post_process_func

    video = torch.linspace(-1, 1, 48 * num_views).reshape(1, 3, 2 * num_views, 2, 4).to(dtype)[..., ::2]
    original = video.clone()
    lidar = torch.full((1, 3, 2, 2, 2), 60.0)
    metadata = {
        "multiview": {"cameras": ["front", "left", "rear"][:num_views], "frames_per_view": 2, "fps": 30},
        "lidar": {"fps": 10},
        "internal": {"hidden": True},
    }

    def apply_guardrail(view):
        # Model the real guardrail's FP32 return and a framewise pixel edit.
        return view.float().mul(0.5)

    expected = VideoProcessor(vae_scale_factor=16).postprocess_video(
        apply_guardrail(video) if guardrails_enabled else video, output_type=output_type
    )
    checked_refs = []
    processed_refs = []
    guardrail_calls = []
    processed_calls = []
    postprocess_video = VideoProcessor.postprocess_video

    def check_video_safety(view):
        assert all(ref() is None for ref in checked_refs), "Previous guardrailed camera is still live"
        assert view.device.type == "cpu"
        start = len(guardrail_calls) * 2
        torch.testing.assert_close(view, original[:, :, start : start + 2], rtol=0, atol=0)
        guardrail_calls.append(tuple(view.shape))
        checked = apply_guardrail(view)
        checked_refs.append(weakref.ref(checked))
        return checked

    def postprocess_one_camera(self, view, output_type="np"):
        assert all(ref() is None for ref in processed_refs), "Previous processed camera is still live"
        assert view.device.type == "cpu"
        assert view.shape == (1, 3, 2, 2, 2)
        processed_calls.append(tuple(view.shape))
        processed = postprocess_video(self, view, output_type=output_type)
        if output_type != "pil":
            processed_refs.append(weakref.ref(processed))
        return processed

    guardrails.is_guardrails_enabled = lambda *_: guardrails_enabled
    guardrails.check_video_safety = check_video_safety
    monkeypatch.setattr(VideoProcessor, "postprocess_video", postprocess_one_camera)
    result = get_cosmos3_post_process_func(SimpleNamespace())(
        {"payload": {"video": video, "lidar": lidar}, "metadata": metadata}, output_type=output_type
    )
    actual = result["payload"]["video"]

    assert len(processed_calls) == num_views
    assert len(guardrail_calls) == (num_views if guardrails_enabled else 0)
    assert all(ref() is None for ref in checked_refs + processed_refs)
    if output_type == "np":
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    elif output_type == "pt":
        assert actual.device.type == "cpu"
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        assert len(actual) == len(expected) == 1
        assert len(actual[0]) == len(expected[0]) == 2 * num_views
        for actual_frame, expected_frame in zip(actual[0], expected[0], strict=True):
            np.testing.assert_array_equal(np.asarray(actual_frame), np.asarray(expected_frame))
    torch.testing.assert_close(result["payload"]["lidar"], lidar, rtol=0, atol=0)
    torch.testing.assert_close(video, original, rtol=0, atol=0)
    assert result["metadata"] == {key: value for key, value in metadata.items() if key != "internal"}
    assert "internal" in metadata


@pytest.mark.parametrize("frames_per_view", [0, -1, True, 1.5, None, 3])
def test_postprocess_rejects_invalid_multiview_metadata(guardrails, frames_per_view):
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import get_cosmos3_post_process_func

    process = get_cosmos3_post_process_func(SimpleNamespace())
    with pytest.raises(ValueError, match="multiview"):
        process(
            {
                "payload": {"video": torch.zeros(1, 3, 4, 2, 2)},
                "metadata": {"multiview": {"cameras": ["front", "rear"], "frames_per_view": frames_per_view}},
            }
        )


def test_multiview_latent_output_bypasses_postprocessing(guardrails):
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import get_cosmos3_post_process_func

    output = {"payload": {"video": torch.zeros(1, 3, 4, 2, 2)}, "metadata": {"multiview": {}}}
    assert get_cosmos3_post_process_func(SimpleNamespace())(output, output_type="latent") is output
