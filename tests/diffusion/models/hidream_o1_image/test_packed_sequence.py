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
