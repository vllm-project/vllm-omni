# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The S2V muxed-bytes post-process output carries the audio sample rate."""

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_s2v import (
    S2V_OUTPUT_FPS,
    get_wan22_s2v_post_process_func,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_muxed_bytes_carry_the_sample_rate() -> None:
    # The worker muxes the waveform into the MP4 bytes and hands post-process a
    # (video, None, sample_rate) tuple; the returned dict must keep the rate so
    # the response is labeled instead of falling back to a default.
    post_process = get_wan22_s2v_post_process_func(SimpleNamespace())

    result = post_process(([b"mp4-chunk-0", b"mp4-chunk-1"], None, 44100), output_type="np")

    assert result == {
        "video": [b"mp4-chunk-0", b"mp4-chunk-1"],
        "audio": None,
        "audio_sample_rate": 44100,
        "fps": S2V_OUTPUT_FPS,
    }


def test_muxed_bytes_without_a_rate_stay_unlabeled() -> None:
    # A bare muxed-bytes list (no tuple) keeps the historical shape and simply
    # carries no rate; normalize treats that as unlabeled audio.
    post_process = get_wan22_s2v_post_process_func(SimpleNamespace())

    result = post_process([b"mp4-chunk-0"], output_type="np")

    assert result == {
        "video": [b"mp4-chunk-0"],
        "audio": None,
        "fps": S2V_OUTPUT_FPS,
    }
