# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU correctness/size tests for device-side video reduction.

The reduction must be byte-identical to the current production output (SHM
widens bf16->f32, then diffusers postprocess, then the API server's ``*255``
rounding), otherwise enabling it would silently change pixels.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.postprocess.device_reduction import reduce_video_to_uint8_frames

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
]

_GPU = torch.accelerator.is_available() if hasattr(torch, "accelerator") else torch.cuda.is_available()
requires_gpu = pytest.mark.skipif(not _GPU, reason="device reduction is a GPU path")


def _device() -> str:
    return "cuda:0"


def _reference_uint8(video: torch.Tensor, *, compute_dtype: torch.dtype | None = torch.float32) -> np.ndarray:
    """The diffusers postprocess plus the API server's rounding, for the same input.

    ``compute_dtype`` selects the precision the denormalize runs at.
    ``torch.float32`` is the precision the device reduction uses; ``None`` keeps
    the input's own dtype, which is what the engine's postprocess actually does.
    """
    from diffusers.video_processor import VideoProcessor

    source = video if compute_dtype is None else video.to(compute_dtype)
    processed = VideoProcessor(vae_scale_factor=8).postprocess_video(source, output_type="np")
    scaled = np.clip(processed.astype(np.float32), 0.0, 1.0) * 255.0
    np.rint(scaled, out=scaled)
    return scaled.astype(np.uint8)


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_device_reduction_matches_a_float32_reference(dtype: torch.dtype) -> None:
    """The reduction must be the float32 computation, exactly, for every input dtype."""
    torch.manual_seed(0)
    # Values slightly outside [-1, 1] so the clamp is actually exercised.
    video = (torch.rand(1, 3, 8, 64, 96, device=_device()) * 2.4 - 1.2).to(dtype)

    reference = _reference_uint8(video)
    produced = reduce_video_to_uint8_frames(video).cpu().numpy()

    assert produced.shape == reference.shape == (1, 8, 64, 96, 3)
    assert produced.dtype == np.uint8
    np.testing.assert_array_equal(produced, reference)


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_device_reduction_deviates_from_a_narrow_postprocess_by_at_most_one(dtype: torch.dtype) -> None:
    """Pin the cost of computing in float32: at most one 255th.

    A postprocess that denormalizes in the VAE's own dtype (WAN, HunyuanVideo,
    LTX-2, Cosmos3) does not match this reduction bit for bit; ~20% of pixels
    differ by one on a real 480x832x81 WAN generation. Keep that a bound rather
    than a surprise, so a regression widening it fails here.
    """
    torch.manual_seed(0)
    video = (torch.rand(1, 3, 8, 64, 96, device=_device()) * 2.4 - 1.2).to(dtype)

    narrow = _reference_uint8(video, compute_dtype=None)
    produced = reduce_video_to_uint8_frames(video).cpu().numpy()

    deviation = np.abs(produced.astype(np.int16) - narrow.astype(np.int16))
    assert deviation.max() <= 1
    # The narrow path really is different, so the bound above is not vacuous.
    assert deviation.any()


@requires_gpu
def test_device_reduction_shrinks_hop1_payload_4x() -> None:
    """The whole point: the D2H payload drops from float32 to uint8 (4x)."""
    video = torch.rand(1, 3, 8, 704, 1280, device=_device(), dtype=torch.bfloat16) * 2 - 1

    # Production transports bf16 widened to float32 over SHM.
    widened_f32_bytes = video.numel() * 4
    reduced = reduce_video_to_uint8_frames(video)
    reduced_bytes = reduced.numel() * reduced.element_size()

    assert reduced.dtype == torch.uint8
    assert widened_f32_bytes / reduced_bytes == pytest.approx(4.0, rel=1e-6)


@requires_gpu
def test_device_reduction_stays_on_device() -> None:
    """Reduction must happen before D2H, so the result is still on the GPU."""
    video = torch.rand(1, 3, 4, 32, 48, device=_device(), dtype=torch.bfloat16) * 2 - 1
    reduced = reduce_video_to_uint8_frames(video)
    assert reduced.device.type == "cuda"
    assert reduced.is_contiguous()


@requires_gpu
def test_device_reduction_without_denormalize_matches_already_unit_range() -> None:
    """Pipelines with do_normalize=False emit [0,1]; skip the denorm step."""
    video = torch.rand(1, 3, 4, 16, 16, device=_device(), dtype=torch.float32)  # already [0,1]

    produced = reduce_video_to_uint8_frames(video, do_denormalize=False).cpu().numpy()

    expected = video.permute(0, 2, 3, 4, 1).cpu().numpy()
    expected = np.rint(np.clip(expected, 0, 1) * 255.0).astype(np.uint8)
    np.testing.assert_array_equal(produced, expected)


def test_device_reduction_rejects_non_5d() -> None:
    """A wrong-rank tensor is a programming error, not a silent reshape."""
    with pytest.raises(ValueError, match=r"\[B, C, F, H, W\]"):
        reduce_video_to_uint8_frames(torch.rand(3, 8, 64, 64))
