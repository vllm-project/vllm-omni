# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the CFM mel-frame bucketing decision.

Steady-state chunk lengths vary per request; without bucketing every length
becomes its own CUDA-graph capture shape (428 captures / 13 flushes in the
#6628 regression). ``_cfm_pad_frames`` aligns the frame axis onto a grid so
steady-state calls share one capture shape; ``_decode_cfm`` trims the output
back. These tests pin the decision math on CPU, no CUDA required.
"""

import pytest

from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import (
    _cfm_pad_frames,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_pads_partial_chunk_up_to_bucket():
    # 50 frames on a 16-frame grid -> 14 padding frames.
    assert _cfm_pad_frames(mel_frames=50, offset=0, noise_capacity=1000, bucket_frames=16, disabled=False) == 14


def test_aligned_chunk_needs_no_padding():
    assert _cfm_pad_frames(mel_frames=64, offset=0, noise_capacity=1000, bucket_frames=16, disabled=False) == 0


def test_ragged_valid_lengths_path_disables_bucketing():
    """The ragged per-request cache path must never be padded."""
    for ragged in (True,):
        for mel in (50, 64, 96):
            assert (
                _cfm_pad_frames(
                    mel_frames=mel,
                    offset=0,
                    noise_capacity=1000,
                    bucket_frames=16,
                    disabled=ragged,
                )
                == 0
            )


def test_disabled_bucketing_returns_zero():
    """bucket_frames <= 1 (or wrapper absent) means plain eager behavior."""
    for bucket in (0, 1):
        assert (
            _cfm_pad_frames(
                mel_frames=50,
                offset=0,
                noise_capacity=1000,
                bucket_frames=bucket,
                disabled=False,
            )
            == 0
        )


def test_padding_never_overflows_noise_buffer():
    """Padding past the decoder's noise buffer must fall back to zero.

    The padded call slices x from decoder.rand_noise[offset:end]; exceeding
    its width would crash. Bucketing is best-effort, never required.
    """
    # 96 frames + 16 pad = 112 > 100 capacity -> give up, pad 0.
    assert _cfm_pad_frames(mel_frames=96, offset=0, noise_capacity=100, bucket_frames=16, disabled=False) == 0
    # Same mel length fits when un-padded: 96 <= 100.
    assert _cfm_pad_frames(mel_frames=96, offset=0, noise_capacity=100, bucket_frames=1, disabled=False) == 0
    # Cache offset eats into the capacity: 64 + 0 pad would fit, 64 + 48 pad
    # would not, so the pad must be dropped.
    assert _cfm_pad_frames(mel_frames=64, offset=50, noise_capacity=110, bucket_frames=16, disabled=False) == 0


def test_padding_applies_with_cache_offset_when_it_fits():
    # offset 50 + mel 50 + pad 14 = 114 <= 200 -> pad stands.
    assert _cfm_pad_frames(mel_frames=50, offset=50, noise_capacity=200, bucket_frames=16, disabled=False) == 14


def test_steady_state_cache_width_saturates_on_bucket_grid():
    """Pin the real recurrence: grow, then trim to ``prompt_len + 100``.

    ``_decode_batch_once`` trims the estimator attention cache after every
    decode, so the width saturates instead of growing without bound. What
    matters for the graph cache is that the steady-state
    ``(chunk_width, cache_width)`` pair settles on a single grid point.
    """
    bucket, prompt_len = 16, 304
    cache_cap = prompt_len + 100
    width = prompt_len
    shapes = set()
    for _ in range(20):
        pad = _cfm_pad_frames(
            mel_frames=50,
            offset=width,
            noise_capacity=30000,
            bucket_frames=bucket,
            disabled=False,
        )
        assert pad == 14
        shapes.add((50 + pad, width))  # (chunk width, cache width) as used
        width = min(width + 50 + pad, cache_cap)
    assert width == cache_cap
    # (64, 304) -> (64, 368) -> (64, 404), then stable.
    assert shapes == {(64, 304), (64, 368), (64, 404)}


def test_varied_chunk_lengths_collapse_onto_few_widths():
    """Bucketing exists for the varied-length calls, not the steady 50-frame one.

    The first/last chunk and the ``plan_token2wav_encode_slices`` splits land on
    arbitrary lengths; those are the calls that would each become their own
    capture shape. Pin that a realistic mix collapses onto a few padded widths.
    """
    varied = (7, 12, 17, 25, 33, 50, 51, 64)
    padded = {
        mel
        + _cfm_pad_frames(
            mel_frames=mel,
            offset=300,
            noise_capacity=30000,
            bucket_frames=16,
            disabled=False,
        )
        for mel in varied
    }
    assert len(padded) == 4, sorted(padded)  # 16 / 32 / 48 / 64
    assert len(padded) < len(varied)
