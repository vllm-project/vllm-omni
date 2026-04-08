# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for MagiHuman pipeline via vLLM-Omni."""

import io

import av
import numpy as np
import pytest

from tests.utils import hardware_test
from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def _validate_mp4(video_bytes: bytes, min_frames: int = 10) -> None:
    """Validate that the MP4 contains meaningful video and audio tracks."""
    container = av.open(io.BytesIO(video_bytes))

    v_streams = [s for s in container.streams if s.type == "video"]
    assert len(v_streams) >= 1, "No video stream found in MP4"

    a_streams = [s for s in container.streams if s.type == "audio"]
    assert len(a_streams) >= 1, "No audio stream found in MP4"

    v_stream = v_streams[0]
    assert v_stream.width >= 1080, f"Unexpected video width: {v_stream.width}"
    assert v_stream.height >= 1056, f"Unexpected video height: {v_stream.height}"

    frame_count = 0
    for frame in container.decode(video=0):
        frame_count += 1
        if frame_count >= min_frames:
            break
    assert frame_count >= min_frames, f"Video has only {frame_count} frames (expected >= {min_frames})"

    container.close()


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_magi_human_e2e(run_level):
    """End-to-end test for MagiHuman generating video and audio."""
    if run_level != "advanced_model":
        pytest.skip("MagiHuman e2e test requires advanced_model run level with real weights.")

    model_path = "SII-GAIR/daVinci-MagiHuman-Base-1080p"

    omni = Omni(
        model=model_path,
        init_timeout=1200,
        tensor_parallel_size=2,
    )

    prompt = (
        "A young woman with long, wavy golden blonde hair and bright blue eyes, "
        "wearing a fitted ivory silk blouse with a delicate lace collar, sits "
        "stationary in front of a softly lit, blurred warm-toned interior. Her "
        "overall disposition is warm, composed, and gently confident. The camera "
        "holds a static medium close-up, framing her from the shoulders up, "
        "with shallow depth of field keeping her face in sharp focus. Soft "
        "directional key light falls from the upper left, casting a gentle "
        "highlight along her cheekbone and nose bridge. She draws a quiet breath, "
        "the levator labii superiors relaxing as her lips part. She speaks in "
        "clear, warm, unhurried American English: "
        "\"The most beautiful things in life aren't things at all — "
        "they're moments, feelings, and the people who make you feel truly alive.\" "
        "Her jaw descends smoothly on each stressed syllable; the orbicularis oris "
        "shapes each vowel with precision. A faint, genuine smile engages the "
        "zygomaticus major, lifting her lip corners fractionally. Her brows rest "
        "in a soft, neutral arch throughout. She maintains steady, forward-facing "
        "eye contact. Head position remains level; no torso displacement occurs.\n\n"
        "Dialogue:\n"
        "<Young blonde woman, American English>: "
        "\"The most beautiful things in life aren't things at all — "
        "they're moments, feelings, and the people who make you feel truly alive.\"\n\n"
        "Background Sound:\n"
        "<Soft, warm indoor ambience with a faint distant piano melody>"
    )

    sampling_params = OmniDiffusionSamplingParams(
        height=256,
        width=448,
        num_inference_steps=8,
        seed=52,
        extra_args={
            "seconds": 5,
            "sr_height": 1080,
            "sr_width": 1920,
            "sr_num_inference_steps": 5,
        },
    )

    try:
        outputs = list(
            omni.generate(
                prompts=[prompt],
                sampling_params_list=[sampling_params],
            )
        )

        assert len(outputs) > 0, "No outputs returned"
        first = outputs[0]

        assert hasattr(first, "images") and first.images, "No video frames in output"
        video_frames = first.images[0]
        assert isinstance(video_frames, np.ndarray), f"Expected numpy array, got {type(video_frames)}"
        assert video_frames.ndim == 4, f"Expected 4D array (T,H,W,3), got shape {video_frames.shape}"

        audio_waveform = None
        if hasattr(first, "multimodal_output") and first.multimodal_output:
            audio_waveform = first.multimodal_output.get("audio")
        assert audio_waveform is not None, "No audio waveform in multimodal_output"

        video_bytes = mux_video_audio_bytes(
            video_frames,
            audio_waveform,
            fps=25.0,
            audio_sample_rate=44100,
        )
        assert isinstance(video_bytes, bytes), f"Expected MP4 bytes, got {type(video_bytes)}"
        assert len(video_bytes) > 1000, f"MP4 too small ({len(video_bytes)} bytes)"

        _validate_mp4(video_bytes)
    finally:
        omni.close()
