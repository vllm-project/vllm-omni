# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the CFM mel-frame bucketing decision.

Steady-state chunk lengths vary per request; without bucketing every length
becomes its own CUDA-graph capture shape (428 captures / 13 flushes in the
#6628 regression). ``_cfm_pad_frames`` aligns the frame axis onto a grid so
steady-state calls share one capture shape; ``_decode_cfm`` trims the output
back. These tests pin the decision math on CPU, no CUDA required.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import (
    _cfm_pad_frames,
    _zero_padded_frames,
)
from vllm_omni.model_executor.models.minicpmo_4_5.cuda_graph_wrapper import (
    WholeEulerCFMGraphWrapper,
    _att_keep_ranges,
    _build_capture_mask,
    _capture_query_width,
    _whole_euler_att_segments,
    _zero_padded_cnn_cache,
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
    # The original chunk fits while the padded one does not: 50 + 50 = 100
    # <= 110, but 100 + 14 = 114 > 110, so the pad must be dropped.
    assert _cfm_pad_frames(mel_frames=50, offset=50, noise_capacity=110, bucket_frames=16, disabled=False) == 0
    # Same numbers with room for the padding: 114 <= 120, so it stands.
    assert _cfm_pad_frames(mel_frames=50, offset=50, noise_capacity=120, bucket_frames=16, disabled=False) == 14


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


def test_capture_query_width_collapses_decode_grid_onto_one_bucket():
    """Whole-Euler capture bucket is the #7416 pad helper with a 64-wide decode grid."""
    assert _capture_query_width(16, 0) == 16
    assert _capture_query_width(16, 1) == 16
    assert {_capture_query_width(w, 64) for w in (16, 32, 48, 50, 64)} == {64}
    assert _capture_query_width(304, 64) == 320
    assert _capture_query_width(320, 64) == 320


def test_att_keep_ranges_match_the_streaming_trim():
    """Past ``prompt_len + 100`` frames the cache keeps its first ``prompt_len`` and last 100."""
    assert _att_keep_ranges(368, (300, 100)) == [(0, 368)]
    assert _att_keep_ranges(400, (300, 100)) == [(0, 400)]
    assert _att_keep_ranges(464, (300, 100)) == [(0, 300), (364, 100)]
    assert _att_keep_ranges(464, None) == [(0, 464)]


def test_whole_euler_att_segments_skip_capture_padding():
    # Steady chunk: capture width == chunk width, so only the trim applies.
    assert _whole_euler_att_segments(mel_width=64, query_cap=64, offset=400, keep=(300, 100)) == [
        (0, 300),
        (364, 100),
    ]
    # A 32-wide chunk on the 64 capture: the graph wrote [current 64 | cache 400];
    # frames 32:64 are capture padding and the cache starts at 64.
    assert _whole_euler_att_segments(mel_width=32, query_cap=64, offset=400, keep=None) == [(0, 32), (64, 400)]
    assert _whole_euler_att_segments(mel_width=32, query_cap=64, offset=400, keep=(300, 100)) == [
        (0, 32),
        (64, 268),
        (364, 100),
    ]
    # First chunk: no cache, only the tail of the capture width is dropped.
    assert _whole_euler_att_segments(mel_width=304, query_cap=320, offset=0, keep=None) == [(0, 304)]


def test_capture_mask_moves_the_cache_block_past_capture_padding():
    width, cap, offset, valid = 32, 64, 8, 30
    caller = torch.ones(2, width, width + offset, dtype=torch.bool)
    caller[:, :, valid:width] = False
    mask = _build_capture_mask(
        attn_mask=caller,
        batch_size=1,
        query_cap=cap,
        offset=offset,
        mel_width=width,
        mel_frames=valid,
        device=torch.device("cpu"),
    )
    assert mask.shape == (2, cap, cap + offset)
    assert mask[:, :, :valid].all()
    assert not mask[:, :, valid:cap].any()
    assert mask[:, :, cap:].all()
    # Capture-padding query rows are never empty, so SDPA cannot produce NaN there.
    assert mask.any(dim=-1).all()

    implicit = _build_capture_mask(
        attn_mask=None,
        batch_size=1,
        query_cap=cap,
        offset=offset,
        mel_width=width,
        mel_frames=width,
        device=torch.device("cpu"),
    )
    assert implicit[:, :, :width].all()
    assert not implicit[:, :, width:cap].any()
    assert implicit[:, :, cap:].all()


def test_zero_padded_frames_survives_an_integration_step():
    """The padded region must stay zero after ``x = x + dt * velocity``.

    Zeroing once before the CFM loop is not enough: the update touches every
    column, so the padded region becomes non-zero again (0 -> 0.010 -> 0.020
    over the first steps). Pin the helper that re-zeroes it each step.
    """
    mel, pad = 50, 14
    x = torch.zeros(2, 4, mel + pad)
    x[..., :mel] = 0.05
    _zero_padded_frames(x, mel)
    assert torch.all(x[..., mel:] == 0.0)
    for _ in range(3):
        x = x + 0.02 * torch.ones_like(x)  # the integration step
        _zero_padded_frames(x, mel)
        assert torch.all(x[..., mel:] == 0.0)
        assert torch.all(x[..., :mel] > 0.0)


def test_zero_padded_frames_is_a_noop_without_padding():
    x = torch.ones(1, 2, 32)
    _zero_padded_frames(x, None)
    _zero_padded_frames(x, 32)
    assert torch.all(x == 1.0)


def _cnn_cache_estimator(widths):
    return SimpleNamespace(
        blocks=[
            SimpleNamespace(conv=SimpleNamespace(block=[None, SimpleNamespace(causal_padding=(width, 0))]))
            for width in widths
        ]
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize(
    "widths,cache_width,pad_frames,cache_depth",
    [
        ((4, 4), 4, 1, 2),
        ((4, 4), 4, 0, 2),
        ((4, 4), 4, 7, 2),
        ((4, 2), 4, 1, 2),
        ((2, 2), 4, 1, 2),
        ((6, 6), 4, 1, 2),
        ((0, 4, -1), 4, 1, 3),
        ((0, -1), 4, 7, 2),
        ((4, 4), 4, -1, 2),
        ((4, 4), 4, 1, 3),
        ((), 4, 1, 1),
    ],
)
def test_zero_padded_cnn_cache_matches_per_block_writes(
    dtype, noncontiguous, widths, cache_width, pad_frames, cache_depth
):
    # Compare all storage, including the untouched columns of a strided view.
    storage_width = cache_width * (2 if noncontiguous else 1)
    original = torch.arange(cache_depth * 2 * 3 * storage_width, dtype=torch.float32)
    original = (original.reshape(cache_depth, 2, 3, storage_width) + 1).to(dtype)
    expected_storage = original.clone()
    actual_storage = original.clone()
    expected = expected_storage[..., ::2] if noncontiguous else expected_storage
    actual = actual_storage[..., ::2] if noncontiguous else actual_storage
    for index, width in enumerate(widths):
        if width > 0:
            zero_from = max(0, width - pad_frames)
            if zero_from < width:
                expected[index][..., zero_from:] = 0.0

    result = _zero_padded_cnn_cache(actual, _cnn_cache_estimator(widths), pad_frames)

    assert result is None
    assert torch.equal(actual_storage, expected_storage)
    assert actual.dtype == dtype
    assert actual.stride() == expected.stride()


def test_zero_padded_cnn_cache_uniform_blocks_use_one_write():
    cache = torch.ones(16, 2, 3, 4)
    version = cache._version

    _zero_padded_cnn_cache(cache, _cnn_cache_estimator([4] * 16), 1)

    # CPU mutation count pins the mechanism without making a CUDA timing claim.
    assert cache._version - version == 1
    assert torch.equal(cache[..., :3], torch.ones(16, 2, 3, 3))
    assert torch.count_nonzero(cache[..., 3:]) == 0


@pytest.mark.parametrize("max_graphs", [15, 32])
def test_precapture_covers_final_chunk_without_displacing_steady_graphs(monkeypatch, max_graphs):
    from vllm_omni.model_executor.models.minicpmo_4_5 import cuda_graph_wrapper as graph_module

    captured = []
    stats = {"captures": 0}

    def entry(*, graph_batch, query_cap, offset, **kwargs):
        captured.append((graph_batch, query_cap, offset))
        stats["captures"] += 1
        return ()

    wrapper = SimpleNamespace(
        enabled=True,
        att_slots=0,
        _att_capacity=0,
        offset_bucket_frames=50,
        query_widths=(50,),
        _graph_batches=lambda: [1, 2, 4, 8, 16],
        max_graphs=max_graphs,
        _cache={},
        _slot_graphs={},
        device=torch.device("cpu"),
        dtype=torch.float32,
        _entry=entry,
        _precapture_fill=lambda statics: None,
        _stats=stats,
        arena=SimpleNamespace(att_bytes=lambda: 0),
    )
    monkeypatch.setattr(graph_module, "_memory_snapshot", lambda device: None)
    count = WholeEulerCFMGraphWrapper.precapture(
        wrapper, offsets=range(300, 401), steady=400, channels=4, spk_dim=4, tail_frames=6
    )
    assert captured[:15] == [(b, 50, o) for b in [16, 8, 4, 2, 1] for o in [400, 350, 300]]
    assert count == min(max_graphs, 30)
    assert wrapper.query_widths == (50,)
    if max_graphs == 32:
        assert (1, 100, 350) in captured
        assert captured[15:] == [(b, 100, o) for b in [16, 8, 4, 2, 1] for o in [400, 350, 300]]
