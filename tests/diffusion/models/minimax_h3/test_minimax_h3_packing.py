# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
