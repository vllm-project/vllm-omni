# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for opt-in RIFE frame interpolation in the Helios pipeline.

Covers the issue #8173 / A6 wiring: ``post_decode`` interpolates the decoded
chunk worker-side and emits an output envelope whose ``video_fps_multiplier``
metadata survives the engine-side post-process step.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture

import vllm_omni.diffusion.models.helios.pipeline_helios as helios_module
from vllm_omni.diffusion.models.helios.pipeline_helios import (
    get_helios_post_process_func,
)
from vllm_omni.diffusion.output_formatter import normalize_diffusion_postprocess_output
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_BATCH, _LATENT_CH, _LATENT_T, _LATENT_H, _LATENT_W = 1, 16, 9, 2, 2
_PIXEL_T, _PIXEL_H, _PIXEL_W = 4, 16, 16

# Real-RIFE tests need local weights (layout: <path>/flownet.pkl); they are
# skipped when absent so hermetic CI runs still cover the wiring tests.
_LOCAL_RIFE_PATH = os.environ.get("RIFE_TEST_MODEL_PATH", "/root/autodl-tmp/models/RIFE-4.22.lite")
_RIFE_AVAILABLE = os.path.isfile(os.path.join(_LOCAL_RIFE_PATH, "flownet.pkl"))


def _fake_rife(video: torch.Tensor, exp: int, scale: float, model_path):
    """Deterministic stand-in matching the RIFE output contract.

    Original frames pass through unchanged at stride ``2**exp``; the other
    slots hold synthesized frames (simple midpoints). Frame count follows the
    documented ``(N - 1) * 2**exp + 1`` formula for ``exp`` doubling steps.
    """
    for _ in range(exp):
        frames = [video[:, :, idx] for idx in range(video.shape[2])]
        doubled = [frames[0]]
        for left, right in zip(frames, frames[1:]):
            doubled.extend([(left + right) / 2, right])
        video = torch.stack(doubled, dim=2)
    return video, 2**exp


def _fake_rife_with_range_roundtrip(video: torch.Tensor, exp: int, scale: float, model_path):
    """``_fake_rife`` wrapped in the interpolator's float32 range round-trip.

    Weight-free stand-in for ``interpolate_video_tensor`` that reproduces its
    ``[-1, 1] -> [0, 1] -> [-1, 1]`` conversion, which perturbs small bf16
    pixels in the original-frame slots.
    """
    normalized = video.to(torch.float32) * 0.5 + 0.5
    interpolated, multiplier = _fake_rife(normalized, exp, scale, model_path)
    return (interpolated * 2.0 - 1.0).to(video.dtype), multiplier


def _make_decoded_video(num_frames: int = _PIXEL_T) -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, _BATCH * 3 * num_frames * _PIXEL_H * _PIXEL_W).reshape(
        _BATCH, 3, num_frames, _PIXEL_H, _PIXEL_W
    )


# Exactly representable in bf16, yet the interpolator's range round-trip
# returns ``2**-20`` for it.
_NEAR_ZERO_BF16 = 129 * 2**-27


def _make_mixed_sign_bf16_video(num_frames: int = _PIXEL_T) -> torch.Tensor:
    """bf16 video (the decoded-chunk dtype) that exercises the lossy range round-trip.

    A negative pixel selects the ``[-1, 1] -> [0, 1]`` normalization branch,
    and the near-zero value is planted in the first and last generated frame.
    The plain linspace fixture cannot trigger the precision loss: none of its
    values is small enough.
    """
    decoded = _make_decoded_video(num_frames=num_frames).to(torch.bfloat16)
    decoded[0, 0, 0, 0, 0] = -0.5
    decoded[0, 0, 0, 0, 1] = _NEAR_ZERO_BF16
    decoded[0, 0, num_frames - 1, 0, 1] = _NEAR_ZERO_BF16
    return decoded


def _make_pipeline(decoded_video: torch.Tensor) -> helios_module.HeliosPipeline:
    pipeline = object.__new__(helios_module.HeliosPipeline)
    pipeline.vae = SimpleNamespace(decode=lambda latents, return_dict: (decoded_video,))
    return pipeline


def _make_state(
    sampling: OmniDiffusionSamplingParams,
    output_type: str = "np",
) -> StepRequestState:
    state = StepRequestState(request_id="req-0", sampling=sampling)
    state.latents = torch.zeros(_BATCH, _LATENT_CH, _LATENT_T, _LATENT_H, _LATENT_W)
    state.extra = {
        "is_first_chunk": False,
        "is_second_chunk": False,
        "is_skip_first_chunk": False,
        "keep_first_frame": False,
        "image_latents": None,
        "total_generated_latent_frames": 0,
        "history_latents": torch.zeros(_BATCH, _LATENT_CH, 0, _LATENT_H, _LATENT_W),
        "num_latent_frames_per_chunk": _LATENT_T,
        "vae_dtype": torch.float32,
        "latents_std": 1.0,
        "latents_mean": 0.0,
        "output_type": output_type,
        "history_video": None,
    }
    # Post-decode state of the final chunk of a single-chunk request.
    state.chunk_index = 1
    state.total_chunks = 1
    state.chunk_num_steps = 1
    return state


@pytest.fixture(autouse=True)
def _no_platform_cache(mocker: MockerFixture):
    mocker.patch.object(helios_module, "current_omni_platform", SimpleNamespace(is_available=lambda: False))


class TestPostDecodeFrameInterpolation:
    def test_disabled_returns_decoded_tensor(self) -> None:
        decoded = _make_decoded_video()
        pipeline = _make_pipeline(decoded)
        state = _make_state(OmniDiffusionSamplingParams(enable_frame_interpolation=False))

        result = pipeline.post_decode(state)

        assert isinstance(result.output, torch.Tensor)
        assert result.output.shape == decoded.shape
        assert torch.equal(result.output, decoded)

    def test_enabled_emits_envelope_with_fps_multiplier(self, mocker: MockerFixture) -> None:
        interpolate = mocker.patch.object(
            helios_module,
            "interpolate_video_tensor",
            side_effect=_fake_rife,
        )
        decoded = _make_decoded_video()
        pipeline = _make_pipeline(decoded)
        sampling = OmniDiffusionSamplingParams(
            enable_frame_interpolation=True,
            frame_interpolation_exp=2,
            frame_interpolation_scale=0.75,
            frame_interpolation_model_path="/tmp/rife",
        )
        state = _make_state(sampling)

        result = pipeline.post_decode(state)

        assert isinstance(result.output, dict)
        assert set(result.output) == {"payload", "metadata"}
        video = result.output["payload"]["video"]
        assert video.shape[2] == (decoded.shape[2] - 1) * 4 + 1
        assert result.output["metadata"]["video"]["video_fps_multiplier"] == 4
        interpolate.assert_called_once_with(
            decoded,
            exp=2,
            scale=0.75,
            model_path="/tmp/rife",
        )

    @pytest.mark.parametrize("exp", [1, 2])
    def test_generated_frames_restored_bit_exactly(self, mocker: MockerFixture, exp: int) -> None:
        # Weight-free regression: the interpolator's range round-trip perturbs
        # small bf16 pixels, so the generated frames must be restored.
        mocker.patch.object(
            helios_module,
            "interpolate_video_tensor",
            side_effect=_fake_rife_with_range_roundtrip,
        )
        decoded = _make_mixed_sign_bf16_video(num_frames=5)
        pipeline = _make_pipeline(decoded)
        sampling = OmniDiffusionSamplingParams(enable_frame_interpolation=True, frame_interpolation_exp=exp)
        state = _make_state(sampling)

        result = pipeline.post_decode(state)

        raw, _ = _fake_rife_with_range_roundtrip(decoded, exp, 1.0, None)
        # The stand-in must really lose the planted value; otherwise this
        # test would pass without the frame restore.
        assert not torch.equal(raw[:, :, :: 2**exp], decoded), "fixture no longer triggers the precision loss"
        video = result.output["payload"]["video"]
        assert video.shape[2] == (5 - 1) * 2**exp + 1
        # Generated frames are bit-identical to the decoded chunk ...
        assert torch.equal(video[:, :, :: 2**exp], decoded)
        # ... and every other slot is still what the interpolator synthesized.
        raw[:, :, :: 2**exp] = decoded
        assert torch.equal(video, raw)

    @pytest.mark.skipif(not _RIFE_AVAILABLE, reason="local RIFE weights not available")
    def test_generated_frames_pass_through_unchanged(self) -> None:
        decoded = _make_mixed_sign_bf16_video(num_frames=5)
        pipeline = _make_pipeline(decoded)
        state = _make_state(
            OmniDiffusionSamplingParams(
                enable_frame_interpolation=True,
                frame_interpolation_exp=1,
                frame_interpolation_model_path=_LOCAL_RIFE_PATH,
            )
        )

        result = pipeline.post_decode(state)

        video = result.output["payload"]["video"]
        assert video.shape[2] == (5 - 1) * 2 + 1
        # Real RIFE: the restore must line up with the interpolator's actual
        # frame layout. RIFE runs on the worker accelerator; compare on CPU.
        assert torch.equal(video[:, :, ::2].cpu(), decoded)

    def test_latent_output_skips_interpolation(self, mocker: MockerFixture) -> None:
        interpolate = mocker.patch.object(helios_module, "interpolate_video_tensor")
        decoded = _make_decoded_video()
        pipeline = _make_pipeline(decoded)
        state = _make_state(
            OmniDiffusionSamplingParams(enable_frame_interpolation=True),
            output_type="latent",
        )

        result = pipeline.post_decode(state)

        assert isinstance(result.output, torch.Tensor)
        assert result.output.shape == (_BATCH, _LATENT_CH, _LATENT_T, _LATENT_H, _LATENT_W)
        interpolate.assert_not_called()

    @pytest.mark.skipif(not _RIFE_AVAILABLE, reason="local RIFE weights not available")
    def test_history_video_keeps_uninterpolated_frames(self) -> None:
        decoded = _make_decoded_video()
        pipeline = _make_pipeline(decoded)
        state = _make_state(
            OmniDiffusionSamplingParams(
                enable_frame_interpolation=True,
                frame_interpolation_exp=1,
                frame_interpolation_model_path=_LOCAL_RIFE_PATH,
            )
        )

        pipeline.post_decode(state)

        history = state.extra["history_video"]
        assert history.shape == decoded.shape
        assert torch.equal(history, decoded)


class TestHeliosPostProcessFunc:
    def test_plain_tensor_keeps_legacy_behavior(self) -> None:
        post_process = get_helios_post_process_func(od_config=None)

        video = post_process(_make_decoded_video(), output_type="np")

        assert isinstance(video, np.ndarray)
        assert video.shape == (_BATCH, _PIXEL_T, _PIXEL_H, _PIXEL_W, 3)

    def test_envelope_postprocesses_payload_and_keeps_metadata(self) -> None:
        post_process = get_helios_post_process_func(od_config=None)
        envelope = {
            "payload": {"video": _make_decoded_video(num_frames=9)},
            "metadata": {"video": {"video_fps_multiplier": 2}},
        }

        processed = post_process(envelope, output_type="np")

        assert processed["metadata"] == {"video": {"video_fps_multiplier": 2}}
        video = processed["payload"]["video"]
        assert video.shape == (_BATCH, 9, _PIXEL_H, _PIXEL_W, 3)
        assert 0.0 <= float(video.min()) and float(video.max()) <= 1.0

    def test_envelope_metadata_reaches_serving_layer(self) -> None:
        post_process = get_helios_post_process_func(od_config=None)
        envelope = {
            "payload": {"video": _make_decoded_video(num_frames=9)},
            "metadata": {"video": {"video_fps_multiplier": 4}},
        }

        normalized = normalize_diffusion_postprocess_output(post_process(envelope, output_type="np"))

        assert normalized.metadata["video"]["video_fps_multiplier"] == 4
        assert normalized.primary_key == "video"
        assert "video" in normalized.outputs
