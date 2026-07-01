# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HiDream-O1-Image's packed-sequence / masking / MRoPE
position-id construction (utils_hidream_o1.py).

These functions have no learned parameters, so they're tested in isolation
from real model weights -- per the project roadmap, this is the
highest-value test in Phase 1 since position-id/masking bugs otherwise only
show up as "the image looks subtly wrong," which is expensive to debug.
"""

import pytest
import torch

from vllm_omni.diffusion.models.hidream_o1_image.utils_hidream_o1 import (
    build_packed_attention_metadata,
    depatchify,
    find_closest_resolution,
    get_rope_index_fix_point,
    patchify,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
VISION_START_TOKEN_ID = 151652


def _build_t2i_input_ids(txt_len: int, image_len: int) -> torch.Tensor:
    """Synthetic packed sequence: txt_len plain text ids, then one
    vision-start-tagged image-token span of length image_len."""
    text = torch.arange(100, 100 + txt_len, dtype=torch.long)
    image_span = torch.full((image_len,), IMAGE_TOKEN_ID, dtype=torch.long)
    image_span[0] = VISION_START_TOKEN_ID
    return torch.cat([text, image_span]).unsqueeze(0)


class TestGetRopeIndexFixPoint:
    def test_text_only_sequence_is_sequential(self):
        input_ids = torch.arange(100, 110, dtype=torch.long).unsqueeze(0)
        position_ids, deltas = get_rope_index_fix_point(
            spatial_merge_size=1,
            image_token_id=IMAGE_TOKEN_ID,
            video_token_id=VIDEO_TOKEN_ID,
            vision_start_token_id=VISION_START_TOKEN_ID,
            input_ids=input_ids,
        )
        assert position_ids.shape == (3, 1, 10)
        expected = torch.arange(10).view(1, 1, -1).expand(3, 1, -1)
        assert torch.equal(position_ids, expected)
        assert deltas.shape == (1, 1)

    def test_image_span_anchored_at_fix_point_not_contiguous(self):
        """The key behavior under test: an image span flagged with
        skip_vision_start_token=True must NOT be placed immediately after
        the preceding text (contiguous offset) -- it must jump to the
        fix_point anchor instead.
        """
        txt_len = 8
        h, w = 3, 3
        # _build_t2i_input_ids's image_len is the TOTAL vision-token span
        # (matching the real packed-sequence convention: the vision-start
        # token overwrites grid slot 0, it isn't an extra token on top of
        # h*w) -- so this must be h*w, not h*w + 1.
        input_ids = _build_t2i_input_ids(txt_len, h * w)
        image_grid_thw = torch.tensor([[1, h, w]], dtype=torch.int64)

        position_ids, _ = get_rope_index_fix_point(
            spatial_merge_size=1,
            image_token_id=IMAGE_TOKEN_ID,
            video_token_id=VIDEO_TOKEN_ID,
            vision_start_token_id=VISION_START_TOKEN_ID,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            skip_vision_start_token=[1],
            fix_point=4096,
        )

        # Text positions: sequential starting at 0.
        text_positions = position_ids[0, 0, :txt_len]
        assert torch.equal(text_positions, torch.arange(txt_len))

        # Image span (including the vision-start token slot) starts at the
        # fix_point anchor (minus the text-length offset baked into the
        # anchor calc), NOT at txt_len (which would be the naive
        # contiguous placement).
        image_t_positions = position_ids[0, 0, txt_len:]
        naive_contiguous_start = txt_len
        assert int(image_t_positions.min()) != naive_contiguous_start
        assert int(image_t_positions.min()) >= naive_contiguous_start  # anchor is always >= text length here

    def test_image_grid_h_w_channels_vary_spatially(self):
        h, w = 2, 3
        txt_len = 4
        input_ids = _build_t2i_input_ids(txt_len, h * w)
        image_grid_thw = torch.tensor([[1, h, w]], dtype=torch.int64)

        position_ids, _ = get_rope_index_fix_point(
            spatial_merge_size=1,
            image_token_id=IMAGE_TOKEN_ID,
            video_token_id=VIDEO_TOKEN_ID,
            vision_start_token_id=VISION_START_TOKEN_ID,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            skip_vision_start_token=[1],
        )

        h_chan = position_ids[1, 0, txt_len:]
        w_chan = position_ids[2, 0, txt_len:]
        # H channel should take exactly h distinct values, W channel exactly w.
        assert len(torch.unique(h_chan)) == h
        assert len(torch.unique(w_chan)) == w


class TestBuildPackedAttentionMetadata:
    def test_text_only_is_pure_causal(self):
        token_types = torch.zeros((1, 6), dtype=torch.long)
        dense_mask, spans = build_packed_attention_metadata(token_types)

        assert dense_mask.shape == (1, 1, 6, 6)
        expected_causal = torch.ones(6, 6, dtype=torch.bool).tril()
        assert torch.equal(dense_mask[0, 0], expected_causal)
        assert spans == [[]]

    def test_trailing_image_span_is_bidirectional_and_excluded_from_text_view(self):
        # 3 text tokens (causal), then 4 "gen" tokens (bidirectional).
        token_types = torch.tensor([[0, 0, 0, 1, 1, 1, 1]])
        dense_mask, spans = build_packed_attention_metadata(token_types)

        assert spans == [[(3, 7)]]

        # Text rows: causal only, and crucially must NOT attend into the image span.
        for row in range(3):
            assert dense_mask[0, 0, row, : row + 1].all()
            assert not dense_mask[0, 0, row, row + 1 :].any()

        # Gen rows: attend to everything (text AND other gen positions).
        for row in range(3, 7):
            assert dense_mask[0, 0, row, :].all()

    def test_multiple_disjoint_gen_spans_detected(self):
        # text(2) - ref-image(2) - text(1) - target-image(3): two separate
        # gen spans, matching Phase 2's multi-reference-image shape.
        token_types = torch.tensor([[0, 0, 1, 1, 0, 1, 1, 1]])
        _, spans = build_packed_attention_metadata(token_types)
        assert spans == [[(2, 4), (5, 8)]]


class TestPatchifyRoundTrip:
    def test_patchify_depatchify_is_identity(self):
        torch.manual_seed(0)
        image = torch.randn(1, 3, 64, 96)
        patches = patchify(image, patch_size=32)
        assert patches.shape == (1, (64 // 32) * (96 // 32), 3 * 32 * 32)
        restored = depatchify(patches, h_patches=2, w_patches=3, patch_size=32)
        assert torch.allclose(image, restored)


class TestFindClosestResolution:
    def test_already_aligned_resolution_is_unchanged(self):
        assert find_closest_resolution(1024, 1024) == (1024, 1024)

    def test_unaligned_resolution_snaps_to_patch_multiple(self):
        w, h = find_closest_resolution(1000, 1000)
        assert w % 32 == 0
        assert h % 32 == 0


# ---------------------------------------------------------------------------
# Phase 2 — Reference-image sequence / masking / position-ID tests
# ---------------------------------------------------------------------------

from PIL import Image as PILImage  # noqa: E402

from vllm_omni.diffusion.models.hidream_o1_image.utils_hidream_o1 import (  # noqa: E402
    adaptive_ref_max_size,
    preprocess_ref_patches,
)


class TestAdaptiveRefMaxSize:
    def test_single_ref_full_resolution(self):
        assert adaptive_ref_max_size(1, 1024) == 1024

    def test_two_refs_smaller(self):
        assert adaptive_ref_max_size(2, 1024) < 1024

    def test_sizes_monotonically_decrease(self):
        sizes = [adaptive_ref_max_size(k, 1024) for k in range(1, 12)]
        # Each jump at 1, 2, 4, 8 boundaries should be non-increasing
        for a, b in zip(sizes, sizes[1:]):
            assert a >= b

    def test_minimum_is_positive(self):
        assert adaptive_ref_max_size(20, 256) > 0


class TestPreprocessRefPatches:
    def test_single_ref_shape(self):
        pil = PILImage.new("RGB", (64, 64), color=(128, 64, 32))
        patches, lens = preprocess_ref_patches([pil], max_size=64, patch_size=32)
        assert patches.shape == (1, (64 // 32) * (64 // 32), 3 * 32 * 32)
        assert lens == [(64 // 32) * (64 // 32)]

    def test_two_refs_concat(self):
        pil_a = PILImage.new("RGB", (64, 32), color=(0, 0, 0))
        pil_b = PILImage.new("RGB", (32, 64), color=(255, 255, 255))
        patches, lens = preprocess_ref_patches([pil_a, pil_b], max_size=128, patch_size=32)
        assert patches.shape[0] == 1
        assert patches.shape[1] == sum(lens)
        assert len(lens) == 2

    def test_normalized_to_minus_one_one(self):
        # Solid white image → all pixels 1.0 after /255 → (1.0 - 0.5) / 0.5 = 1.0
        pil = PILImage.new("RGB", (32, 32), color=(255, 255, 255))
        patches, _ = preprocess_ref_patches([pil], max_size=32, patch_size=32, dtype=torch.float32)
        assert patches.max().item() == pytest.approx(1.0, abs=1e-4)
        # Solid black image → all pixels 0.0 → (0.0 - 0.5) / 0.5 = -1.0
        pil_black = PILImage.new("RGB", (32, 32), color=(0, 0, 0))
        patches_b, _ = preprocess_ref_patches([pil_black], max_size=32, patch_size=32, dtype=torch.float32)
        assert patches_b.min().item() == pytest.approx(-1.0, abs=1e-4)


class TestEditSampleTokenTypes:
    """Tests for the token_type assignment in a synthetic edit-like sequence.
    Uses build_packed_attention_metadata directly, mimicking the layout that
    _build_edit_sample would produce for K=1 reference image.
    """

    def _make_edit_token_types(self, txt_len: int, tgt_len: int, ref_len: int) -> torch.Tensor:
        """Synthetic token_types_raw for:
            [text(0)...txt_len-1 tokens][tms(3)][target(1)...tgt_len][ref_pixel(2)...ref_len]
        The tms token is at position txt_len-1 (last text position).
        """
        total = txt_len + tgt_len + ref_len
        raw = torch.zeros((1, total), dtype=torch.long)
        raw[0, txt_len - 1] = 3          # tms
        raw[0, txt_len: txt_len + tgt_len] = 1   # target patches
        raw[0, txt_len + tgt_len:] = 2           # ref pixel patches
        return raw

    def test_vinput_mask_excludes_ref_pixel_patches(self):
        """vinput_mask must be True ONLY for type=1 (target) positions."""
        raw = self._make_edit_token_types(txt_len=5, tgt_len=4, ref_len=3)
        vinput_mask = raw == 1
        assert vinput_mask[0, :5].sum() == 0     # text: no
        assert vinput_mask[0, 5:9].sum() == 4    # target: all True
        assert vinput_mask[0, 9:].sum() == 0     # ref pixel: no

    def test_single_contiguous_gen_span(self):
        """tms + target + ref_pixel form one contiguous gen span (no multi-span needed)."""
        raw = self._make_edit_token_types(txt_len=5, tgt_len=4, ref_len=3)
        token_types = (raw > 0).long()
        _, spans = build_packed_attention_metadata(token_types)
        # tms is at position 4, gen ends at position 11 (4+1+4+3=12 total, idx 11 incl.)
        assert len(spans[0]) == 1
        start, end = spans[0][0]
        assert start == 4   # tms position (txt_len - 1)
        assert end == 12    # exclusive end (txt_len - 1 + 1 + tgt_len + ref_len = 12)

    def test_ref_pixel_rows_attend_to_everything(self):
        """Gen rows (including ref pixel patches) must be fully True (bidirectional)."""
        raw = self._make_edit_token_types(txt_len=5, tgt_len=4, ref_len=3)
        token_types = (raw > 0).long()
        dense_mask, _ = build_packed_attention_metadata(token_types)
        # Check a ref pixel row (e.g., row 9 = first ref pixel token)
        ref_pixel_row = dense_mask[0, 0, 9, :]
        assert ref_pixel_row.all(), "Ref pixel token must attend to the full sequence"

    def test_text_rows_do_not_attend_to_gen_tokens(self):
        """Text rows must be causally masked -- they cannot see tms / target / ref pixels."""
        raw = self._make_edit_token_types(txt_len=5, tgt_len=4, ref_len=3)
        token_types = (raw > 0).long()
        dense_mask, _ = build_packed_attention_metadata(token_types)
        # Text row at position 3 (before tms at 4): should not attend to 4..11
        text_row = dense_mask[0, 0, 3, :]
        assert not text_row[4:].any(), "Text row must not attend to gen positions"
