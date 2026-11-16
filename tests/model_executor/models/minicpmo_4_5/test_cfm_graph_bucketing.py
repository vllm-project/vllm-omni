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
    SharedPromptAttCache,
    WholeEulerCFMGraphWrapper,
    _att_keep_ranges,
    _build_capture_mask,
    _capture_query_width,
    _whole_euler_att_segments,
    _zero_padded_cnn_cache,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_shared_prompt_attention_cache_is_bit_exact_and_range_copyable():
    prefix = torch.arange(2 * 2 * 2 * 2 * 1 * 3 * 4, dtype=torch.float32).reshape(2, 2, 2, 2, 1, 3, 4) + 1000
    prompt = torch.arange(2 * 2 * 2 * 2 * 1 * 5 * 4, dtype=torch.float32).reshape(2, 2, 2, 2, 1, 5, 4)
    shared = SharedPromptAttCache(prefix, prompt)
    assert shared.ndim == 7
    assert shared.shape == (2, 2, 2, 2, 1, 8, 4)
    torch.testing.assert_close(shared.materialize(), torch.cat((prefix, prompt), dim=5), rtol=0, atol=0)
    copied = torch.empty_like(shared.materialize())
    shared.copy_range_to(copied[..., :4, :], 0, 4)
    shared.copy_range_to(copied[..., 4:, :], 4, 4)
    torch.testing.assert_close(copied, shared.materialize(), rtol=0, atol=0)

    with pytest.raises(ValueError, match="7-D"):
        SharedPromptAttCache(torch.zeros(2, 2, 2, 1, 3, 4), torch.zeros(2, 2, 2, 1, 5, 4))


def test_whole_euler_plan_groups_handles_grid_without_singleton(monkeypatch):
    block0 = SimpleNamespace(
        conv=SimpleNamespace(in_channels=2, out_channels=2, block=[None, SimpleNamespace(causal_padding=(4, 0))]),
        attn=SimpleNamespace(num_heads=2, head_dim=4),
    )
    estimator = SimpleNamespace(blocks=[block0], parameters=lambda: iter((torch.empty(0),)))

    monkeypatch.setenv("VLLM_OMNI_CFM_GRAPH_GRID", "16")
    wrapper_16 = WholeEulerCFMGraphWrapper(estimator=estimator, micro_batch_size=32, pad_max_rows=2)
    assert wrapper_16._graph_batches() == [16]
    assert wrapper_16._plan_groups(1) == [(16, 1)]
    assert wrapper_16._plan_groups(17) == [(16, 16), (16, 1)]

    monkeypatch.setenv("VLLM_OMNI_CFM_GRAPH_GRID", "4")
    wrapper_4 = WholeEulerCFMGraphWrapper(estimator=estimator, micro_batch_size=4, pad_max_rows=1)
    assert wrapper_4._graph_batches() == [4]
    assert wrapper_4._plan_groups(1) == [(4, 1)]


def test_whole_euler_arena_rows_from_graph_grid_and_env(monkeypatch):
    """Verify that VLLM_OMNI_CFM_GRAPH_GRID pins graph batches and decouples arena rows."""
    monkeypatch.setenv("VLLM_OMNI_CFM_GRAPH_GRID", "1,4,8,16")
    block0 = SimpleNamespace(
        conv=SimpleNamespace(in_channels=2, out_channels=2, block=[None, SimpleNamespace(causal_padding=(4, 0))]),
        attn=SimpleNamespace(num_heads=2, head_dim=4),
    )
    estimator = SimpleNamespace(blocks=[block0], parameters=lambda: iter((torch.empty(0),)))
    wrapper = WholeEulerCFMGraphWrapper(
        estimator=estimator,
        micro_batch_size=32,
        arena_rows_from_graph_grid=True,
    )
    # Sizes above micro_batch_size are dropped; largest pinned size in grid becomes arena rows
    assert wrapper._graph_batches() == [1, 4, 8, 16]
    assert wrapper._arena_rows == 16


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


@pytest.mark.parametrize("width,valid", [(0, 0), (3, 2), (8, 8)])
def test_uniform_capture_mask_cache_reuses_only_key_storage(width, valid):
    cache = {}
    args = dict(
        attn_mask=None,
        query_cap=8,
        offset=2,
        mel_width=width,
        mel_frames=valid,
        device=torch.device("cpu"),
        offset_cap=4,
        key_masks=cache,
    )
    first = _build_capture_mask(batch_size=1, **args)
    larger = _build_capture_mask(batch_size=4, **args)
    expected = torch.zeros(8, 8, 12, dtype=torch.bool)
    expected[:, :width, :valid] = True
    expected[:, :width, 8:10] = True
    if width < 8:
        expected[:, width:] = expected[:, :1]
    torch.testing.assert_close(larger, expected)
    assert first.data_ptr() == larger.data_ptr()
    assert larger.untyped_storage().nbytes() == 12
    # The CFG split used when filling graph inputs must preserve broadcasting.
    target = torch.empty_like(larger)
    target.unflatten(0, (2, 4)).copy_(larger.unflatten(0, (2, 4)))
    torch.testing.assert_close(target, expected)


def test_capture_key_mask_cache_is_bounded_and_excludes_dynamic_masks():
    cache = {}
    args = dict(batch_size=1, query_cap=4, mel_width=3, mel_frames=2, device=torch.device("cpu"), key_masks=cache)
    first = _build_capture_mask(attn_mask=None, offset=0, **args)
    retained = first.clone()
    for offset in range(70):
        _build_capture_mask(attn_mask=None, offset=offset, **args)
    assert len(cache) == 64
    torch.testing.assert_close(first, retained)
    caller = torch.ones(2, 3, 3, dtype=torch.bool)
    a = _build_capture_mask(attn_mask=caller, offset=0, **args)
    caller[0, :, 1] = False
    b = _build_capture_mask(attn_mask=caller, offset=0, **args)
    assert a[0, :, 1].all() and not b[0, :, 1].any()
    assert len(cache) == 64


@pytest.mark.parametrize("device_type", ["cuda", "npu", "cpu"])
def test_graph_memory_telemetry_uses_device_allocator(monkeypatch, mocker, device_type):
    from vllm_omni.model_executor.models.minicpmo_4_5 import cuda_graph_wrapper as graph_module

    device = SimpleNamespace(type=device_type)
    api = mocker.Mock()
    api.memory_allocated.return_value = 10
    api.memory_reserved.return_value = 20
    api.max_memory_allocated.return_value = 30
    api.max_memory_reserved.return_value = 40
    monkeypatch.setattr(torch, "npu", api if device_type == "npu" else mocker.Mock(), raising=False)
    monkeypatch.setattr(torch, "accelerator", api if device_type == "cuda" else mocker.Mock())
    if device_type == "cpu":
        assert graph_module._memory_snapshot(device) is None
        assert graph_module._memory_peak(device) is None
        assert not api.mock_calls
    else:
        assert graph_module._memory_snapshot(device) == (10, 20)
        assert graph_module._memory_peak(device) == (30, 40)
        for name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
            getattr(api, name).assert_called_once_with(device)


def test_npu_precapture_stops_after_graph_exceeds_memory_budget(monkeypatch, mocker):
    from vllm_omni.model_executor.models.minicpmo_4_5 import cuda_graph_wrapper as graph_module

    api = mocker.Mock()
    api.memory_allocated.return_value = 0
    api.memory_reserved.side_effect = [0, 2048]
    monkeypatch.setattr(torch, "npu", api, raising=False)
    # Keep tensor allocation on CPU; exercise the real NPU telemetry and
    # precapture stop policy without requiring a device or graph capture.
    empty = torch.empty
    monkeypatch.setattr(graph_module.torch, "empty", lambda shape, **kwargs: empty(shape))
    wrapper = WholeEulerCFMGraphWrapper.__new__(WholeEulerCFMGraphWrapper)
    wrapper.enabled = True
    wrapper.att_slots = 0
    wrapper._att_capacity = 0
    wrapper.offset_bucket_frames = 50
    wrapper.query_widths = (50,)
    wrapper._graph_batches = lambda: [1]
    wrapper.max_graphs = 10
    wrapper._cache = {}
    wrapper._slot_graphs = {}
    wrapper.device = SimpleNamespace(type="npu")
    wrapper.dtype = torch.float32
    wrapper._stats = {"captures": 0}
    wrapper.arena = mocker.Mock()
    wrapper.arena.att_bytes.return_value = 0

    def entry(**kwargs):
        wrapper._stats["captures"] += 1
        return ()

    wrapper._entry = entry
    assert wrapper.precapture(offsets=[50, 100, 150], steady=150, channels=4, spk_dim=4, max_graph_bytes=1024) == 1
    assert api.memory_reserved.call_count == 2


def _make_history_preserving_token2wav(prompt_len: int = 10):
    import torch.nn as nn

    class _HistoryEncoder(nn.Module):
        def forward_chunk(self, xs, last_chunk=False, cnn_cache=None, att_cache=None):
            del last_chunk, cnn_cache
            batch, length, _ = xs.shape
            old_length = 0 if att_cache is None else att_cache.shape[3]
            output = xs[:, : max(1, length - 1)]
            cnn = xs[:, :1, :].transpose(1, 2).contiguous()
            cur = output.unsqueeze(0).unsqueeze(2).expand(1, batch, 1, output.shape[1], 1)
            att = cur if att_cache is None else torch.cat((cur, att_cache), dim=3)
            assert att.shape[3] == old_length + output.shape[1]
            return output, cnn, att

    class _HistoryEstimator(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = True
            conv1 = SimpleNamespace(causal_padding=(1, 0))
            self.blocks = [
                SimpleNamespace(
                    conv=SimpleNamespace(in_channels=1, out_channels=1, block=[None, conv1]),
                    attn=SimpleNamespace(num_heads=1, head_dim=1),
                )
            ]
            self.register_buffer("att_cache_buffer", torch.ones(1), persistent=False)
            self.register_buffer("cnn_cache_buffer", torch.ones(1), persistent=False)

        def t_embedder(self, time):
            return time[:, None]

        def blocks_forward_chunk(self, inputs, time, mask, cnn_cache, att_cache, cnn_out, att_out):
            del time, cnn_cache
            batch_2x, _, frames = inputs.shape
            x_feat = inputs[:, :1, :].reshape(1, batch_2x, 1, frames, 1)
            new_kv = torch.cat((x_feat, x_feat + 0.5), dim=-1)
            if isinstance(att_cache, torch.Tensor):
                att_out[:, :, :, :frames].copy_(new_kv)
                att_out[:, :, :, frames:].copy_(att_cache)
            else:
                att_out.copy_(new_kv)
            kv_sum = (att_out * 0.05).mean(dim=(0, 2, 3, 4)).reshape(batch_2x, 1, 1)
            if mask is not None:
                kv_sum = kv_sum * mask.to(inputs.dtype).mean(dim=(1, 2)).reshape(batch_2x, 1, 1)
            cnn_out.copy_(inputs[:, :1, :1].reshape(1, batch_2x, 1, 1).expand_as(cnn_out))
            return inputs[:, 1:2] + kv_sum

    class _HistoryDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.estimator = _HistoryEstimator()
            self.inference_cfg_rate = 0.7
            self.register_buffer("rand_noise", torch.zeros(1, 1, 512), persistent=False)

    class _HistoryFlow(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _HistoryEncoder()
            self.encoder_proj = nn.Identity()
            self.decoder = _HistoryDecoder()
            self.spk_embed_affine_layer = nn.Identity()

        def input_embedding(self, tokens):
            return tokens.to(torch.float32).unsqueeze(-1)

    class _HistoryHiFT(nn.Module):
        def inference(self, mel, source):
            del source
            speech = mel[:, 0].repeat_interleave(2, dim=1)
            return speech, speech[:, None]

    class _HistoryToken2Wav:
        def __init__(self):
            self.flow = _HistoryFlow()
            self.hift = _HistoryHiFT()
            self.float16 = False
            self.n_timesteps = 2
            self.mel_cache_len = 1
            self.source_cache_len = 2
            self.speech_window = torch.hamming_window(4, periodic=False)
            self._prompt_len = prompt_len

        def _prepare_prompt(self, prompt_wav):
            seed = float(sum(ord(ch) for ch in str(prompt_wav)) % 7 + 1)
            # _create_initial_states appends 3 lookahead tokens and _HistoryEncoder trims 1 frame.
            tok_len = max(1, self._prompt_len - 2)
            tokens = torch.arange(tok_len, dtype=torch.long).unsqueeze(0) + int(seed)
            mels = torch.arange(self._prompt_len, dtype=torch.float32).reshape(1, self._prompt_len, 1) * 0.1 + seed
            return (
                tokens,
                torch.tensor([tok_len], dtype=torch.int32),
                torch.full((1, 1), seed),
                mels,
                torch.tensor([self._prompt_len], dtype=torch.int32),
            )

    return _HistoryToken2Wav()


def test_batched_token2wav_prompt_att_sharing_matches_unshared_across_trims(monkeypatch, caplog):
    import logging

    from vllm_omni.model_executor.models.minicpmo_4_5 import batched_token2wav as btw_mod
    from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import (
        BatchedToken2Wav,
        _trim_streaming_cache,
    )

    # Set trim suffix < prompt_len (6 < 10) so multi-chunk decoding crosses
    # prompt_len + _CACHE_TRIM_SUFFIX and exercises _att_keep_ranges eviction.
    monkeypatch.setattr(btw_mod, "_CACHE_TRIM_SUFFIX", 6)

    unshared = BatchedToken2Wav(_make_history_preserving_token2wav(prompt_len=10))
    shared = BatchedToken2Wav(
        _make_history_preserving_token2wav(prompt_len=10),
        cfm_graph_config={"prompt_att_sharing": True},
    )

    feat_u = unshared.prepare_prompt("voice-a", "/fake/a.wav")
    feat_s = shared.prepare_prompt("voice-a", "/fake/a.wav")

    states_u = unshared.setup_batch(feat_u, 2)
    states_s = shared.setup_batch(feat_s, 2)
    # Initial states share the raw setup cache before any chunk runs.
    prompt_suffix = shared._prompt_att_caches[feat_s.cache_key]
    assert prompt_suffix.shape[5] == 6
    assert torch.equal(prompt_suffix, states_s[0].flow_cache["estimator_att_cache"][..., -6:, :])

    with caplog.at_level(logging.INFO, logger=btw_mod.logger.name):
        for step in range(4):
            toks = torch.full((2, 5), 10 + step * 3, dtype=torch.long)
            toks[1] += 1
            aud_u, states_u = unshared.decode_batch(toks, feat_u, states_u, last_chunk=(step == 3))
            aud_s, states_s = shared.decode_batch(toks, feat_s, states_s, last_chunk=(step == 3))
            for row in range(2):
                torch.testing.assert_close(aud_s[row], aud_u[row], rtol=0, atol=0)
                att_s = states_s[row].flow_cache["estimator_att_cache"]
                assert isinstance(att_s, SharedPromptAttCache)
                assert att_s.prompt.data_ptr() == prompt_suffix.data_ptr()
                torch.testing.assert_close(
                    att_s.materialize(),
                    states_u[row].flow_cache["estimator_att_cache"],
                    rtol=0,
                    atol=0,
                )

    assert any("CFM prompt attention suffix verified bit-exact" in rec.message for rec in caplog.records)
    assert any(rec.levelname == "INFO" and "prompt=voice-a" in rec.message for rec in caplog.records)
    assert states_s[0].flow_cache["estimator_att_cache"].prefix.data_ptr() != (
        states_s[1].flow_cache["estimator_att_cache"].prefix.data_ptr()
    )
    # _trim_streaming_cache keeps SharedPromptAttCache valid when trimming further.
    extra_trimmed = _trim_streaming_cache(states_s[0].flow_cache["estimator_att_cache"], 4)
    assert isinstance(extra_trimmed, SharedPromptAttCache)
    assert extra_trimmed.shape[5] == 10

    # Inject a corrupted prompt suffix on chunk 2+ to confirm per-call validation
    # catches mismatches even after initial verification succeeded.
    corrupted = states_s[0].flow_cache["estimator_att_cache"].materialize()
    corrupted[..., -1:, :] += 9.0
    rejected_rows = shared._share_prompt_att_rows(feat_s.cache_key, prompt_suffix, [corrupted])
    assert isinstance(rejected_rows[0], torch.Tensor)
    assert feat_s.cache_key in shared._prompt_att_rejected
    assert any("CFM prompt attention suffix differs" in rec.message for rec in caplog.records)


def test_decode_ragged_batch_shares_prompt_suffix_single_and_mixed_voices(monkeypatch):
    from vllm_omni.model_executor.models.minicpmo_4_5 import batched_token2wav as btw_mod
    from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import BatchedToken2Wav
    from vllm_omni.model_executor.models.minicpmo_4_5.cuda_graph_wrapper import (
        _pack_att_cache,
        _packed_att_buffer,
        _unpack_att_cache,
    )

    monkeypatch.setattr(btw_mod, "_CACHE_TRIM_SUFFIX", 6)

    def _mock_ragged_body(estimator, x, time_emb, mask, old_cnn, old_att, cnn_out, att_out, valid_lengths):
        del valid_lengths
        packed_out = _packed_att_buffer(att_out)
        out = estimator.blocks_forward_chunk(x, time_emb, mask, old_cnn, _pack_att_cache(old_att), cnn_out, packed_out)
        _unpack_att_cache(packed_out, att_out)
        return out

    # Single-voice ragged batch
    unshared = BatchedToken2Wav(_make_history_preserving_token2wav(prompt_len=8))
    shared = BatchedToken2Wav(
        _make_history_preserving_token2wav(prompt_len=8),
        cfm_graph_config={"prompt_att_sharing": True},
    )
    monkeypatch.setattr(unshared, "_ragged_body", _mock_ragged_body)
    monkeypatch.setattr(shared, "_ragged_body", _mock_ragged_body)
    feat_u = unshared.prepare_prompt("voice-r", "/fake/r.wav")
    feat_s = shared.prepare_prompt("voice-r", "/fake/r.wav")
    states_u = unshared.setup_batch(feat_u, 2)
    states_s = shared.setup_batch(feat_s, 2)
    ragged_toks = [torch.tensor([11, 12, 13]), torch.tensor([21, 22, 23, 24])]
    aud_u, next_u = unshared.decode_ragged_batch(ragged_toks, feat_u, states_u, last_chunks=[False, True])
    aud_s, next_s = shared.decode_ragged_batch(ragged_toks, feat_s, states_s, last_chunks=[False, True])
    for row in range(2):
        torch.testing.assert_close(aud_s[row], aud_u[row], rtol=0, atol=0)
        att_s = next_s[row].flow_cache["estimator_att_cache"]
        assert isinstance(att_s, SharedPromptAttCache)
        assert att_s.prompt.data_ptr() == shared._prompt_att_caches[feat_s.cache_key].data_ptr()
        torch.testing.assert_close(att_s.materialize(), next_u[row].flow_cache["estimator_att_cache"], rtol=0, atol=0)

    # Mixed-voice ragged batch via setup_prompts + list[PromptFeatures]
    feat_u_pair = [
        unshared.prepare_prompt("voice-m1", "/fake/m1.wav"),
        unshared.prepare_prompt("voice-m2", "/fake/m2.wav"),
    ]
    feat_s_pair = [
        shared.prepare_prompt("voice-m1", "/fake/m1.wav"),
        shared.prepare_prompt("voice-m2", "/fake/m2.wav"),
    ]
    mixed_u = unshared.setup_prompts(feat_u_pair, max_batch=2)
    mixed_s = shared.setup_prompts(feat_s_pair, max_batch=2)
    m_aud_u, m_next_u = unshared.decode_ragged_batch(ragged_toks, feat_u_pair, mixed_u, last_chunks=[False, False])
    m_aud_s, m_next_s = shared.decode_ragged_batch(ragged_toks, feat_s_pair, mixed_s, last_chunks=[False, False])
    for row, feat in enumerate(feat_s_pair):
        torch.testing.assert_close(m_aud_s[row], m_aud_u[row], rtol=0, atol=0)
        att_s = m_next_s[row].flow_cache["estimator_att_cache"]
        assert isinstance(att_s, SharedPromptAttCache)
        assert att_s.prompt.data_ptr() == shared._prompt_att_caches[feat.cache_key].data_ptr()
        torch.testing.assert_close(att_s.materialize(), m_next_u[row].flow_cache["estimator_att_cache"], rtol=0, atol=0)

    # can_merge_state_shapes accepts mixed SharedPromptAttCache and Tensor rows
    monkeypatch.setattr(shared, "_whole_euler_ragged_active", lambda: True)
    mixed_states = [m_next_s[0], mixed_s[1]]
    assert shared.can_merge_state_shapes(mixed_states, [3, 4]) is True
