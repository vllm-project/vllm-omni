# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Acoustic-chunk arithmetic and attention-mask tests for the YuE2 NAR port.

Two contracts from the reference implementation are pinned here, because both
are silent-failure surfaces:

* ``song_chunks`` draws the WHOLE-song noise once from the request seed and
  then cuts it at the historical chunk boundaries — chunk-local draws would
  change the song for the same seed whenever the chunk geometry changes;
* the hybrid attention in ``_attention`` gives NAR queries bidirectional
  visibility and the AR prefill causal visibility, including across query
  tiling blocks (a rectangular Q/K with ``is_causal`` alone uses the
  upper-left triangle and is wrong for any block but the first).

Both are pure functions over tensors: no weights, no GPU.
"""

import pytest
import torch

from vllm_omni.model_executor.models.yue2.constants import CODEC_OFFSET, CODEC_SIZE, CONTEXT, MUSIC_END
from vllm_omni.model_executor.models.yue2.nar import _attention, chunk_ranges, song_chunks

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestChunkRanges:
    def test_short_song_is_one_chunk(self):
        assert chunk_ranges(500, 63) == [(0, 500)]

    def test_chunk_size_is_half_the_free_context(self):
        # (24576 - 63 - 3) // 2 = 12255: one AR prefix + one NAR chunk of the
        # same order must fit the window together, hence the halving.
        frames, prefix = 30000, 63
        size = (CONTEXT - prefix - 3) // 2
        ranges = chunk_ranges(frames, prefix)
        assert ranges[0] == (0, size)
        assert ranges == [(a, min(a + size, frames)) for a in range(0, frames, size)]

    def test_ranges_cover_all_frames_exactly_once(self):
        for frames, prefix in [(1, 63), (12255, 63), (12256, 63), (40000, 500)]:
            ranges = chunk_ranges(frames, prefix)
            assert ranges[0][0] == 0 and ranges[-1][1] == frames
            for (_, end), (start, _) in zip(ranges, ranges[1:]):
                assert end == start  # no gap, no overlap

    @pytest.mark.parametrize("frames,prefix", [(0, 63), (-5, 63), (10, CONTEXT), (10, CONTEXT - 2)])
    def test_invalid_geometries_raise(self, frames, prefix):
        with pytest.raises(ValueError):
            chunk_ranges(frames, prefix)


class TestSongChunks:
    def test_chunk_tokens_reconstruct_the_model_input(self):
        prefix, codec = [1, 2, 3], [5, 6, 7]
        (chunk,) = song_chunks(prefix, codec, seed=42)
        assert chunk.ar_tokens == [1, 2, 3, CODEC_OFFSET + 5, CODEC_OFFSET + 6, CODEC_OFFSET + 7, MUSIC_END]

    def test_whole_song_noise_is_drawn_once_then_cut(self):
        frames, prefix, seed = 30000, [0] * 63, 831001
        chunks = song_chunks(prefix, list(range(frames)), seed=seed)
        # Reference: one draw for the whole song, sliced at the same cuts.
        generator = torch.Generator().manual_seed(seed)
        full = torch.randn((frames, 64), dtype=torch.float32, generator=generator)
        for (start, end), chunk in zip(chunk_ranges(frames, len(prefix)), chunks):
            assert torch.equal(chunk.noise, full[start:end])

    def test_same_seed_same_noise_different_seed_differs(self):
        prefix, codec = [1], list(range(32))
        a = song_chunks(prefix, codec, seed=7)
        b = song_chunks(prefix, codec, seed=7)
        c = song_chunks(prefix, codec, seed=8)
        assert torch.equal(a[0].noise, b[0].noise)
        assert not torch.equal(a[0].noise, c[0].noise)

    def test_noise_layout_is_fp32_frames_by_64(self):
        (chunk,) = song_chunks([1], list(range(64)), seed=1)
        assert chunk.noise.dtype == torch.float32
        assert chunk.noise.shape == (64, 64)

    @pytest.mark.parametrize("codec", [[-1, 5], [CODEC_SIZE], [CODEC_SIZE + 3]])
    def test_out_of_range_codec_raises(self, codec):
        with pytest.raises(ValueError):
            song_chunks([1], codec, seed=1)


def _reference_attention(q, k, v, *, causal: bool) -> torch.Tensor:
    """[T, H, D] tensors; explicit GQA expansion + float64 softmax reference."""
    groups = q.shape[1] // k.shape[1]
    k_ref = k.repeat_interleave(groups, dim=1)
    v_ref = v.repeat_interleave(groups, dim=1)
    scores = torch.einsum("thd,shd->hts", q.double(), k_ref.double()) / (q.shape[-1] ** 0.5)
    if causal:
        mask = torch.triu(torch.ones(q.shape[0], k.shape[0], dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    out = torch.softmax(scores, dim=-1)
    return torch.einsum("hts,shd->thd", out, v_ref.double())


def _qkv(tokens: int, heads: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(tokens, heads, 8, generator=g)


class TestHybridAttention:
    def test_prefill_attention_is_causal(self):
        q, k, v = _qkv(7, 4, 1), _qkv(7, 2, 2), _qkv(7, 2, 3)
        got = _attention(q, k, v, causal=True)
        ref = _reference_attention(q, k, v, causal=True)
        assert torch.allclose(got.double(), ref, atol=1e-5)

    def test_nar_attention_is_bidirectional(self):
        # Every NAR query sees every key: the song latents are solved jointly.
        q, k, v = _qkv(5, 4, 4), _qkv(9, 2, 5), _qkv(9, 2, 6)
        got = _attention(q, k, v, causal=False)
        ref = _reference_attention(q, k, v, causal=False)
        assert torch.allclose(got.double(), ref, atol=1e-5)

    def test_causal_attention_survives_query_tiling(self):
        """Blocks after the first need their absolute-position mask."""
        q, k, v = _qkv(11, 4, 7), _qkv(11, 2, 8), _qkv(11, 2, 9)
        tiled = _attention(q, k, v, causal=True, query_chunk_size=4)
        whole = _attention(q, k, v, causal=True)
        ref = _reference_attention(q, k, v, causal=True)
        assert torch.allclose(tiled.double(), ref, atol=1e-5)
        assert torch.allclose(whole.double(), ref, atol=1e-5)

    def test_bidirectional_attention_survives_query_tiling(self):
        q, k, v = _qkv(10, 4, 10), _qkv(10, 2, 11), _qkv(10, 2, 12)
        tiled = _attention(q, k, v, causal=False, query_chunk_size=3)
        ref = _reference_attention(q, k, v, causal=False)
        assert torch.allclose(tiled.double(), ref, atol=1e-5)

    def test_mismatched_shapes_are_rejected(self):
        with pytest.raises(ValueError):
            _attention(torch.zeros(4, 4, 8), torch.zeros(5, 2, 8), torch.zeros(5, 2, 8), causal=True)
