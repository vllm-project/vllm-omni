# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_video_patchify_round_trip_preserves_values():
    from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
        minimax_h3_patchify_video_latent,
        minimax_h3_unpatchify_video_tokens,
    )

    latent = torch.arange(2 * 3 * 2 * 4 * 6).reshape(2, 3, 2, 4, 6)
    rows = minimax_h3_patchify_video_latent(
        latent,
        patch_size=(1, 2, 2),
    )
    restored = minimax_h3_unpatchify_video_tokens(
        rows,
        latent_shape=(2, 2, 3, 3),
        patch_size=(1, 2, 2),
    )

    torch.testing.assert_close(restored, latent)


def test_audio_pack_round_trip_preserves_channel_major_order():
    from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
        minimax_h3_pack_audio_latent,
        minimax_h3_unpack_audio_tokens,
    )

    latent = torch.arange(2 * 4 * 5).reshape(2, 4, 5)
    rows = minimax_h3_pack_audio_latent(latent)
    restored = minimax_h3_unpack_audio_tokens(
        rows,
        audio_t=10,
        audio_channel=2,
    )

    torch.testing.assert_close(restored, latent)


def test_t2va_and_fl2va_packing_keep_update_rows_separate():
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence,
    )

    common = dict(
        text_len=4,
        latent_t=2,
        latent_h=4,
        latent_w=6,
        audio_t=3,
    )
    t2va = minimax_h3_packed_sequence(
        **common,
        include_keyframe_cond=False,
    )
    fl2va = minimax_h3_packed_sequence(
        **common,
        include_keyframe_cond=True,
        keyframe_frame_indices=[0],
        frame_count=5,
    )

    assert int(t2va["seq_len"]) == 64
    assert t2va["img_pos"].numel() == 12
    assert t2va["update_mask"].all()
    assert fl2va["img_pos"].numel() == 18
    assert fl2va["update_mask"].sum().item() == 12
    assert (~fl2va["update_mask"]).sum().item() == 6
    assert t2va["audio_pos"].numel() == fl2va["audio_pos"].numel() == 6
    assert t2va["video_spans"] == ({"start": 10, "latent_grid": (2, 2, 3), "role": "target"},)
    assert fl2va["video_spans"] == ({"start": 16, "latent_grid": (2, 2, 3), "role": "target"},)


def test_ref2va_packing_tracks_video_and_audio_update_masks():
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence_ref2va_blocks,
    )

    packed = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=4,
        latent_t=2,
        latent_h=4,
        latent_w=6,
        audio_t=3,
        ref_blocks=[
            {"kind": "image", "latent_h": 4, "latent_w": 4},
            {"kind": "audio", "ref_audio_t": 2},
        ],
    )

    assert int(packed["seq_len"]) == 64
    assert packed["img_pos"].numel() == 16
    assert packed["update_mask"].sum().item() == 12
    assert packed["audio_pos"].numel() == 10
    assert packed["audio_update_mask"].sum().item() == 6
    assert packed["cu_seqlens"].tolist() == [0, 30, 64]


def test_ref2va_packing_publishes_physical_video_spans_only():
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence_ref2va_blocks,
    )

    packed = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=4,
        latent_t=2,
        latent_h=4,
        latent_w=6,
        audio_t=3,
        ref_blocks=[
            {"kind": "image", "latent_h": 4, "latent_w": 4},
            {"kind": "video_audio", "ref_audio_t": 2, "latent_t": 2, "latent_h": 4, "latent_w": 6},
            {"kind": "audio", "ref_audio_t": 1},
            {"kind": "video", "ref_audio_t": 0, "latent_t": 1, "latent_h": 4, "latent_w": 4},
        ],
    )

    # Image and all audio rows intentionally stay out of the sparse spans.
    assert packed["video_spans"] == (
        {"start": 12, "latent_grid": (2, 2, 3), "role": "reference"},
        {"start": 26, "latent_grid": (1, 2, 2), "role": "reference"},
        {"start": 36, "latent_grid": (2, 2, 3), "role": "target"},
    )


def test_condition_noise_is_seeded_and_keeps_clean_anchor_at_timestep_one():
    from vllm_omni.diffusion.models.minimax_h3.condition_noise import (
        minimax_h3_audio_cond_noise_aug_rows,
        minimax_h3_imgvid_cond_noise_aug_rows,
    )

    image_rows = torch.randn(4, 96)
    audio_rows = torch.randn(6, 32)

    torch.testing.assert_close(
        minimax_h3_imgvid_cond_noise_aug_rows(
            image_rows,
            condition_shapes=[(1, 4, 4)],
            target_latent_t=2,
            imgvid_cond_num_frames=1,
            seed=42,
            noise_aug=1.0,
        ),
        image_rows,
    )
    torch.testing.assert_close(
        minimax_h3_audio_cond_noise_aug_rows(
            audio_rows,
            condition_audio_t=[3],
            seed=42,
            noise_aug=1.0,
        ),
        audio_rows,
    )

    first = minimax_h3_audio_cond_noise_aug_rows(
        audio_rows,
        condition_audio_t=[3],
        seed=42,
        noise_aug=0.25,
    )
    second = minimax_h3_audio_cond_noise_aug_rows(
        audio_rows,
        condition_audio_t=[3],
        seed=42,
        noise_aug=0.25,
    )
    torch.testing.assert_close(first, second)


def test_condition_noise_accepts_a_reference_video_longer_than_the_target():
    from vllm_omni.diffusion.models.minimax_h3.condition_noise import (
        minimax_h3_imgvid_cond_noise_aug_rows,
    )

    rows = torch.zeros(4 * 2 * 2, 96)
    result = minimax_h3_imgvid_cond_noise_aug_rows(
        rows,
        condition_shapes=[(4, 4, 4)],
        target_latent_t=2,
        imgvid_cond_num_frames=1,
        seed=7,
        noise_aug=0.5,
    )
    assert result.shape == rows.shape


def test_guides_preserve_insertion_order_overlaps_and_reference_adjusted_origins():
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        MINIMAX_H3_AUDIO_REF_COND_ID,
        MINIMAX_H3_IMGVID_COND_ID,
        MINIMAX_H3_PAD_ID,
        minimax_h3_packed_sequence_ref2va_blocks,
    )

    image = {"kind": "image", "frame_index": 36, "latent_h": 4, "latent_w": 6}
    packed = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=3,
        latent_t=2,
        latent_h=4,
        latent_w=6,
        audio_t=5,
        guide_blocks=[
            image,
            {"kind": "video_audio", "frame_index": 1, "latent_t": 6, "latent_h": 2, "latent_w": 4, "ref_audio_t": 3},
            {"kind": "audio", "frame_index": 0, "ref_audio_t": 2},
            {"kind": "video", "frame_index": 1, "latent_t": 2, "latent_h": 4, "latent_w": 2, "ref_audio_t": 0},
            image,
        ],
        ref_blocks=[
            {"kind": "image", "latent_h": 2, "latent_w": 2},
            {"kind": "video_audio", "latent_t": 2, "latent_h": 2, "latent_w": 4, "ref_audio_t": 3},
            {"kind": "audio", "ref_audio_t": 2},
            {"kind": "video", "latent_t": 1, "latent_h": 2, "latent_w": 2, "ref_audio_t": 0},
        ],
    )

    assert packed["cu_seqlens"].tolist() == [0, 79, 128]
    assert packed["video_row_start"].item() == 67
    assert packed["video_spans"] == (
        {"start": 9, "latent_grid": (6, 1, 2), "role": "reference"},
        {"start": 31, "latent_grid": (2, 2, 1), "role": "reference"},
        {"start": 48, "latent_grid": (2, 1, 2), "role": "reference"},
        {"start": 56, "latent_grid": (1, 1, 1), "role": "reference"},
        {"start": 67, "latent_grid": (2, 2, 3), "role": "target"},
    )
    visual_anchors = [*range(3, 21), *range(31, 42), *range(48, 52), 56]
    audio_anchors = [*range(21, 31), *range(42, 48), *range(52, 56)]
    assert packed["img_pos"].tolist() == visual_anchors + list(range(67, 79))
    assert packed["audio_pos"].tolist() == audio_anchors + list(range(57, 67))
    assert packed["update_mask"].tolist() == [False] * 34 + [True] * 12
    assert packed["audio_update_mask"].tolist() == [False] * 20 + [True] * 10
    assert (packed["input_ids"][visual_anchors] == MINIMAX_H3_IMGVID_COND_ID).all()
    assert (packed["input_ids"][audio_anchors] == MINIMAX_H3_AUDIO_REF_COND_ID).all()
    assert packed["text_pos"].tolist() == [0, 1, 2]
    for modality, tag in (("img", 0), ("audio", 2)):
        positions = packed[f"{modality}_pos"]
        mask = packed["image_mask" if modality == "img" else "audio_mask"]
        assert torch.equal(mask.nonzero().flatten(), positions)
        assert (packed["token_tags"][positions] == tag).all()
    assert (packed["token_tags"][:3] == 1).all()
    assert (packed["input_ids"][79:] == MINIMAX_H3_PAD_ID).all()
    assert (packed["token_tags"][79:] == -1).all()
    assert packed["document_id"].tolist() == [0] * 79 + [1] * 49
    assert not packed["img_position_ids"][79:].any()

    g = packed["img_position_ids"]
    assert g.dtype == torch.float64
    scale = 5.0 / 3.0
    origin = 3.0 + 1.0 + (scale + 4 * scale) + 2.0 + scale
    assert g[57, 0].item() == origin
    assert g[67, 0].item() == origin
    # References retain their own timeline even though they follow guides physically.
    assert g[41, 0].item() == 3.0
    assert g[42, 0].item() == g[48, 0].item() == 4.0
    assert g[52, 0].item() == 4.0 + scale + 4 * scale
    assert g[56, 0].item() == 4.0 + scale + 4 * scale + 2.0
    assert (g[3:9, 0] == origin + 36 * scale).all()
    assert torch.equal(g[3:9], g[35:41])  # repeated images are not deduplicated
    assert torch.equal(g[3:9, 1:], g[67:73, 1:])  # local still spatial grid
    clip_t = g[9:21, 0].reshape(6, 2)
    expected_t = origin + scale + torch.tensor([0, 1, 5, 9, 13, 17], dtype=torch.float64) * scale
    torch.testing.assert_close(clip_t[:, 0], expected_t, rtol=0, atol=1e-14)
    assert torch.equal(clip_t[:, 0], clip_t[:, 1])
    assert g[9, 0].item() == g[21, 0].item() == g[31, 0].item() == origin + scale
    torch.testing.assert_close(
        g[21:27, 0], (origin + scale + torch.arange(3, dtype=torch.float64)).repeat(2), rtol=0, atol=0
    )
    torch.testing.assert_close(g[27:31, 0], (origin + torch.arange(2, dtype=torch.float64)).repeat(2), rtol=0, atol=0)
    # AV audio uses its visual width grid, standalone audio uses the target grid.
    assert (g[21:24, 2] == g[9:11, 2].min()).all()
    assert (g[24:27, 2] == g[9:11, 2].max()).all()
    assert (g[27:29, 2] == g[67:73, 2].min()).all()
    assert (g[29:31, 2] == g[67:73, 2].max()).all()


@pytest.mark.parametrize("seq_len", [None, 25, 80])
def test_guide_only_fractional_origin_and_explicit_padding(seq_len):
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence_ref2va_blocks,
    )

    packed = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=3,
        latent_t=2,
        latent_h=4,
        latent_w=4,
        audio_t=3,
        ref_blocks=[],
        seq_len=seq_len,
        guide_blocks=[
            {"kind": "image", "frame_index": 1, "latent_h": 4, "latent_w": 4},
            {"kind": "audio", "frame_index": 2, "ref_audio_t": 2},
        ],
    )
    assert packed["cu_seqlens"].tolist() == [0, 25, 64 if seq_len is None else seq_len]
    g = packed["img_position_ids"]
    assert (g[3:7, 0] == 3.0 + 5.0 / 3.0).all()
    assert g[7, 0].item() == 3.0 + 2 * (5.0 / 3.0)
    assert g[11, 0].item() == g[17, 0].item() == 3.0


@pytest.mark.parametrize("guides", [None, []])
@pytest.mark.parametrize(
    "ref_blocks",
    [
        [],
        [
            {"kind": "image", "latent_h": 2, "latent_w": 4},
            {"kind": "audio", "ref_audio_t": 0},
            {"kind": "video", "ref_audio_t": 40, "latent_t": 17, "latent_h": 4, "latent_w": 2},
            {"kind": "video_audio", "ref_audio_t": 0, "latent_t": 1, "latent_h": 2, "latent_w": 2},
        ],
    ],
)
def test_empty_guides_are_identical_to_omitted_guides(guides, ref_blocks):
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence_ref2va_blocks,
    )

    kwargs = dict(text_len=4, latent_t=17, latent_h=4, latent_w=6, audio_t=5, ref_blocks=ref_blocks)
    legacy = minimax_h3_packed_sequence_ref2va_blocks(**kwargs)
    explicit = minimax_h3_packed_sequence_ref2va_blocks(**kwargs, guide_blocks=guides)
    assert legacy.keys() == explicit.keys()
    for key, value in legacy.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, explicit[key]), key
        else:
            assert value == explicit[key], key


@pytest.mark.parametrize(
    ("block", "message"),
    [
        (None, "must be an object"),
        ({"kind": "image", "frame_index": -1}, "frame_index must be non-negative"),
        ({"kind": "image", "frame_index": True}, "frame_index must be an integer"),
        ({"kind": "image", "frame_index": 1.5}, "frame_index must be an integer"),
        ({"kind": "image"}, "frame_index must be an integer"),
        ({"kind": "other", "frame_index": 0}, "kind unsupported"),
        ({"kind": "image", "frame_index": 0, "latent_h": 3, "latent_w": 4}, "latent_h must be divisible"),
        ({"kind": "image", "frame_index": 0, "latent_h": 4, "latent_w": 0}, "latent_w must be positive"),
        ({"kind": "audio", "frame_index": 0, "ref_audio_t": 0}, "ref_audio_t must be positive"),
        ({"kind": "audio", "frame_index": 0, "ref_audio_t": 1.5}, "ref_audio_t must be an integer"),
        (
            {"kind": "video", "frame_index": 0, "latent_t": 1, "latent_h": 2, "latent_w": 2, "ref_audio_t": 1},
            "ref_audio_t must be zero",
        ),
        (
            {"kind": "video_audio", "frame_index": 0, "latent_t": 1, "latent_h": 2, "latent_w": 2, "ref_audio_t": 0},
            "ref_audio_t must be positive",
        ),
        (
            {"kind": "video", "frame_index": 0, "latent_t": 0, "latent_h": 2, "latent_w": 2, "ref_audio_t": 0},
            "latent_t must be positive",
        ),
    ],
)
def test_guide_block_validation(block, message):
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence_ref2va_blocks,
    )

    with pytest.raises(ValueError, match=message):
        minimax_h3_packed_sequence_ref2va_blocks(
            text_len=3,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=3,
            ref_blocks=[],
            guide_blocks=[block],
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"text_len": -1}, "text_len must be non-negative"),
        ({"latent_t": 0}, "latent_t must be positive"),
        ({"latent_h": 3}, "must be divisible"),
        ({"latent_w": True}, "latent_w must be an integer"),
        ({"audio_t": 0}, "audio_t must be positive"),
        ({"audio_channel": 0}, "audio_channel must be positive"),
        ({"seq_len": 1.5}, "seq_len must be an integer"),
        ({"seq_len": 20}, "< used rows 21"),
        ({"guide_blocks": {}}, "guide_blocks must be a sequence"),
        ({"guide_blocks": "image"}, "guide_blocks must be a sequence"),
        ({"ref_blocks": [{"kind": "image", "latent_h": 4, "latent_w": 3}]}, "latent_w must be divisible"),
    ],
)
def test_guide_packing_dimensions_and_capacity_validation(overrides, message):
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence_ref2va_blocks,
    )

    kwargs = dict(
        text_len=3,
        latent_t=2,
        latent_h=4,
        latent_w=4,
        audio_t=3,
        ref_blocks=[],
        guide_blocks=[{"kind": "image", "frame_index": 0, "latent_h": 4, "latent_w": 4}],
    )
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        minimax_h3_packed_sequence_ref2va_blocks(**kwargs)


@pytest.mark.parametrize(
    ("kind", "expected_guide_rows", "expected_padded_rows"),
    [("image", 4032, 42048), ("video", 7056, 45056)],
    ids=["four-images", "22-frame-tail-clip"],
)
def test_native_canvas_guides_fit_default_row_budgets(kind, expected_guide_rows, expected_padded_rows):
    """Synthetic CPU layout acceptance only, not measured safe GPU capacity."""
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence_ref2va_blocks,
    )
    from vllm_omni.diffusion.models.minimax_h3.time_request import MINIMAX_H3_SHAPE_PLANNER
    from vllm_omni.model_executor.models.minimax_h3.timeline_guides import TimelineGuideLimits

    width, height, num_frames, text_len = 1344, 768, 124, 256
    latent_h, latent_w = height // 16, width // 16
    latent_t = MINIMAX_H3_SHAPE_PLANNER.video_latent_t(num_frames)
    audio_t = MINIMAX_H3_SHAPE_PLANNER.audio_latent_t(num_frames / 24)
    assert (latent_t, latent_h, latent_w, audio_t) == (37, 48, 84, 207)
    starts = [0, 36, 72, 123] if kind == "image" else [num_frames - 22]
    guide_blocks = [
        {"kind": kind, "frame_index": start, "latent_h": latent_h, "latent_w": latent_w} for start in starts
    ]
    if kind == "video":
        guide_blocks[0].update(latent_t=MINIMAX_H3_SHAPE_PLANNER.video_latent_t(22), ref_audio_t=0)
        assert guide_blocks[0]["latent_t"] == 7

    packed = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=text_len,
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        ref_blocks=[],
        guide_blocks=guide_blocks,
    )
    guide_rows = int((~packed["update_mask"]).sum())
    target_video_rows, target_audio_rows = 37296, 414
    used = text_len + guide_rows + target_audio_rows + target_video_rows
    video_start = text_len + guide_rows + target_audio_rows
    limits = TimelineGuideLimits()
    assert len(guide_blocks) <= limits.max_entries
    assert guide_rows == expected_guide_rows <= limits.max_guide_rows
    assert packed["seq_len"].item() == expected_padded_rows <= limits.max_packed_rows
    assert packed["cu_seqlens"].tolist() == [0, used, expected_padded_rows]
    assert 0 < expected_padded_rows - used < 64
    assert expected_padded_rows % 64 == 0
    assert packed["img_position_ids"].device.type == "cpu"
    assert packed["image_mask"].sum().item() == guide_rows + target_video_rows
    assert packed["audio_mask"].sum().item() == target_audio_rows
    assert packed["update_mask"].sum().item() == target_video_rows
    assert not packed["update_mask"][:guide_rows].any()
    assert packed["update_mask"][guide_rows:].all()
    assert packed["audio_update_mask"].all()
    assert packed["audio_pos"].numel() == target_audio_rows
    assert torch.equal(packed["img_pos"][packed["update_mask"]], torch.arange(video_start, used))
    assert torch.equal(packed["audio_pos"], torch.arange(text_len + guide_rows, video_start))
    assert packed["text_pos"].numel() == text_len
    assert packed["latent_grid"].tolist() == [37, 24, 42]
    assert packed["video_row_start"].item() == video_start
    guide_spans = ({"start": text_len, "latent_grid": (7, 24, 42), "role": "reference"},) if kind == "video" else ()
    assert packed["video_spans"] == guide_spans + (
        {"start": video_start, "latent_grid": (37, 24, 42), "role": "target"},
    )
    assert packed["img_position_ids"][video_start, 0].item() == float(text_len)
    assert (packed["token_tags"][packed["image_mask"]] == 0).all()
    assert (packed["token_tags"][packed["audio_mask"]] == 2).all()
    assert (packed["token_tags"][used:] == -1).all()
    assert not packed["document_id"][:used].any()
    assert (packed["document_id"][used:] == 1).all()


def test_explicit_seq_len_pins_one_shape_across_prompt_lengths():
    """Prompts of different token counts must land on the same packed length.

    Without the pin they fall into different 64-row buckets, and each bucket is
    a packed shape the compiled transformer has not seen before.
    """
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence,
    )

    common = dict(
        latent_t=2,
        latent_h=4,
        latent_w=6,
        audio_t=3,
        include_keyframe_cond=False,
    )
    # used = text_len + 18 rows: 22 -> bucket 64; 118 -> bucket 128.
    short = minimax_h3_packed_sequence(text_len=4, **common)
    long = minimax_h3_packed_sequence(text_len=100, **common)
    assert int(short["seq_len"]) == 64
    assert int(long["seq_len"]) == 128

    short_pinned = minimax_h3_packed_sequence(text_len=4, seq_len=192, **common)
    long_pinned = minimax_h3_packed_sequence(text_len=100, seq_len=192, **common)
    assert int(short_pinned["seq_len"]) == int(long_pinned["seq_len"]) == 192
    # The pin only adds padding rows; the used prefix is untouched.
    assert int(short_pinned["cu_seqlens"][1]) == int(short["cu_seqlens"][1]) == 22
    assert int(long_pinned["cu_seqlens"][1]) == int(long["cu_seqlens"][1]) == 118
    assert short_pinned["img_pos"].tolist() == short["img_pos"].tolist()


def test_the_ref2va_packer_pins_one_shape_across_prompt_lengths_too():
    """Both packers share one resolver, so both honour the pin."""
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence_ref2va_blocks,
    )

    common = dict(
        latent_t=2,
        latent_h=4,
        latent_w=6,
        audio_t=3,
        ref_blocks=[{"kind": "audio", "ref_audio_t": 1}],
    )
    short = minimax_h3_packed_sequence_ref2va_blocks(text_len=4, **common)
    long = minimax_h3_packed_sequence_ref2va_blocks(text_len=100, **common)
    assert int(short["seq_len"]) != int(long["seq_len"])

    short_pinned = minimax_h3_packed_sequence_ref2va_blocks(text_len=4, seq_len=192, **common)
    long_pinned = minimax_h3_packed_sequence_ref2va_blocks(text_len=100, seq_len=192, **common)
    assert int(short_pinned["seq_len"]) == int(long_pinned["seq_len"]) == 192
    assert int(short_pinned["cu_seqlens"][1]) == int(short["cu_seqlens"][1])
    assert int(long_pinned["cu_seqlens"][1]) == int(long["cu_seqlens"][1])


def test_explicit_seq_len_below_used_rows_is_a_client_error():
    """Both packers reject an under-sized pin as a 4xx, not a server error."""
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence,
        minimax_h3_packed_sequence_ref2va_blocks,
    )
    from vllm_omni.errors import OmniClientError

    with pytest.raises(OmniClientError, match="used rows"):
        minimax_h3_packed_sequence(
            text_len=4,
            latent_t=2,
            latent_h=4,
            latent_w=6,
            audio_t=3,
            include_keyframe_cond=False,
            seq_len=8,
        )

    with pytest.raises(OmniClientError, match="used rows"):
        minimax_h3_packed_sequence_ref2va_blocks(
            text_len=4,
            latent_t=2,
            latent_h=4,
            latent_w=6,
            audio_t=3,
            ref_blocks=[{"kind": "audio", "ref_audio_t": 1}],
            seq_len=8,
        )
