# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Focused CPU tests for MiniMax H3 per-token latent editing."""

from __future__ import annotations

from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_video_mask_accepts_supported_shapes_and_pools_full_grid():
    from vllm_omni.diffusion.models.minimax_h3.latent_mask import (
        minimax_h3_video_edit_masks,
        minimax_h3_video_mask_rows,
    )

    shape = dict(latent_t=2, latent_h=4, latent_w=6)
    torch.testing.assert_close(
        minimax_h3_video_mask_rows(0.25, **shape),
        torch.full((12,), 0.25),
    )

    flat = torch.arange(12, dtype=torch.float32) / 12.0
    quantized_flat = torch.ceil(flat * 256.0) / 256.0
    torch.testing.assert_close(
        minimax_h3_video_mask_rows(flat[None, None], **shape),
        quantized_flat,
    )
    token_grid = flat.reshape(2, 2, 3)
    parsed_token = minimax_h3_video_edit_masks(token_grid[None], **shape)
    torch.testing.assert_close(parsed_token.model_mask_rows, quantized_flat)
    torch.testing.assert_close(parsed_token.restore_mask_rows, flat)

    full_grid = torch.arange(48, dtype=torch.float32).reshape(2, 4, 6) / 48.0
    patch_cells = full_grid.reshape(2, 2, 2, 3, 2).permute(0, 1, 3, 2, 4)
    pooled = patch_cells.amax(dim=(3, 4)).reshape(-1)
    expected_restore = patch_cells.unsqueeze(3).expand(2, 2, 3, 24, 2, 2).reshape(12, 96)
    parsed_full = minimax_h3_video_edit_masks(full_grid[None, None], **shape)
    torch.testing.assert_close(parsed_full.model_mask_rows, torch.ceil(pooled * 256.0) / 256.0)
    torch.testing.assert_close(parsed_full.restore_mask_rows, expected_restore)
    torch.testing.assert_close(
        minimax_h3_video_mask_rows(full_grid[None, None], **shape),
        parsed_full.model_mask_rows,
    )


def test_audio_mask_uses_channel_major_row_order():
    from vllm_omni.diffusion.models.minimax_h3.latent_mask import (
        minimax_h3_audio_edit_masks,
        minimax_h3_audio_mask_rows,
    )

    temporal = torch.tensor([0.0, 0.1, 1.0])
    expected = torch.tensor([0.0, 0.1, 1.0, 0.0, 0.1, 1.0])
    quantized = torch.ceil(expected * 256.0) / 256.0
    torch.testing.assert_close(
        minimax_h3_audio_mask_rows(temporal[None, None], audio_t=3),
        quantized,
    )
    torch.testing.assert_close(
        minimax_h3_audio_mask_rows(expected, audio_t=3),
        quantized,
    )
    parsed = minimax_h3_audio_edit_masks(expected.reshape(2, 3)[None], audio_t=3)
    torch.testing.assert_close(parsed.model_mask_rows, quantized)
    torch.testing.assert_close(parsed.restore_mask_rows, expected)
    torch.testing.assert_close(
        minimax_h3_audio_mask_rows(0.5, audio_t=3),
        torch.full((6,), 0.5),
    )


@pytest.mark.parametrize(
    ("kind", "mask", "match"),
    [
        ("video", torch.zeros(2, 3), "video mask shape"),
        ("video", torch.tensor([float("nan")]), "finite"),
        ("audio", torch.zeros(4), "audio mask shape"),
        ("audio", torch.tensor([-0.1, 0.0, 0.0]), r"\[0, 1\]"),
    ],
)
def test_mask_parsers_reject_invalid_values_and_shapes(kind, mask, match):
    from vllm_omni.diffusion.models.minimax_h3.latent_mask import (
        minimax_h3_audio_mask_rows,
        minimax_h3_video_mask_rows,
    )

    with pytest.raises(ValueError, match=match):
        if kind == "video":
            minimax_h3_video_mask_rows(mask, latent_t=2, latent_h=4, latent_w=6)
        else:
            minimax_h3_audio_mask_rows(mask, audio_t=3)


def test_latent_edit_validates_quantizes_and_canonicalizes_all_generate():
    from vllm_omni.diffusion.models.minimax_h3.latent_mask import MiniMaxH3LatentEdit

    clean = torch.zeros(3, 2)
    anchor = torch.ones(3, 2)
    edit = MiniMaxH3LatentEdit.from_rows(
        clean,
        anchor,
        torch.tensor([0.0, 1.0 / 512.0, 0.5]),
    )
    assert edit is not None
    torch.testing.assert_close(edit.mask_rows, torch.tensor([0.0, 1.0 / 256.0, 0.5]))
    assert edit.restore_mask_rows is not None
    torch.testing.assert_close(edit.restore_mask_rows, torch.tensor([0.0, 1.0 / 512.0, 0.5]))

    almost_one = MiniMaxH3LatentEdit.from_rows(clean, anchor, torch.full((3,), 0.999))
    assert almost_one is not None
    assert almost_one.mask_rows.eq(1.0).all()
    assert almost_one.restore_mask_rows is not None
    assert almost_one.restore_mask_rows.eq(0.999).all()
    assert MiniMaxH3LatentEdit.from_rows(clean, anchor, torch.ones(3)) is None

    with pytest.raises(ValueError, match="shapes must match"):
        MiniMaxH3LatentEdit.from_rows(clean, torch.ones(2, 2), torch.zeros(3))
    with pytest.raises(ValueError, match="mask_rows must have shape"):
        MiniMaxH3LatentEdit.from_rows(clean, anchor, torch.zeros(2))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        MiniMaxH3LatentEdit.from_rows(clean, anchor, torch.tensor([0.0, 1.1, 0.0]))
    with pytest.raises(ValueError, match="restore_mask_rows must have shape"):
        MiniMaxH3LatentEdit.from_rows(clean, anchor, torch.zeros(3), torch.zeros(3, 3))
    bad_clean = clean.clone()
    bad_clean[0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        MiniMaxH3LatentEdit.from_rows(bad_clean, anchor, torch.zeros(3))


def test_latent_edit_zero_fractional_and_one_semantics():
    from vllm_omni.diffusion.models.minimax_h3.latent_mask import MiniMaxH3LatentEdit

    clean = torch.full((3, 2), 10.0)
    anchor = torch.tensor([[8.0, 8.0], [6.0, 6.0], [4.0, 4.0]])
    state = torch.tensor([[2.0, 2.0], [4.0, 4.0], [6.0, 6.0]])
    velocity = torch.full((3, 2), 2.0)
    mask = torch.tensor([0.0, 0.5, 1.0])
    edit = MiniMaxH3LatentEdit.from_rows(clean, anchor, mask)
    assert edit is not None

    model_rows = edit.model_rows(state)
    expected_model_rows = anchor + mask[:, None] * (state - anchor)
    torch.testing.assert_close(model_rows, expected_model_rows)
    torch.testing.assert_close(
        edit.target_timesteps(0.25, 0.999, sigma=0.75),
        torch.tensor([0.999, 0.625, 0.25]),
    )

    predicted = model_rows + 0.75 * velocity * mask[:, None]
    expected_x0 = clean + mask[:, None] * (predicted - clean)
    x0 = edit.x0(model_rows, velocity, 0.25)
    torch.testing.assert_close(x0, expected_x0)
    torch.testing.assert_close(x0[0], clean[0])
    torch.testing.assert_close(x0[2], state[2] + 0.75 * velocity[2])


def test_full_grid_video_mask_restores_individual_cells_inside_model_active_token():
    from vllm_omni.diffusion.models.minimax_h3.latent_mask import (
        MiniMaxH3LatentEdit,
        minimax_h3_video_edit_masks,
    )

    # One generated latent cell activates the whole 2x2 model token, but final
    # restoration must leave the other three cells clean in all 24 channels.
    cell_mask = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    parsed = minimax_h3_video_edit_masks(cell_mask, latent_t=1, latent_h=2, latent_w=2)
    torch.testing.assert_close(parsed.model_mask_rows, torch.ones(1))
    assert parsed.restore_mask_rows.shape == (1, 96)
    expected_restore = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(24).reshape(1, 96)
    torch.testing.assert_close(parsed.restore_mask_rows, expected_restore)

    clean = torch.full((1, 96), 10.0)
    anchor = torch.full((1, 96), 8.0)
    state = torch.full((1, 96), 2.0)
    velocity = torch.full((1, 96), 4.0)
    edit = MiniMaxH3LatentEdit.from_rows(
        clean,
        anchor,
        parsed.model_mask_rows,
        parsed.restore_mask_rows,
    )
    assert edit is not None
    # The pooled model mask is one, so the model sees the live sampler state
    # and the token receives the global timestep.
    torch.testing.assert_close(edit.model_rows(state), state)
    torch.testing.assert_close(edit.target_timesteps(0.25, 0.999), torch.tensor([0.25]))

    predicted = state + 0.75 * velocity
    expected_x0 = torch.where(expected_restore.bool(), predicted, clean)
    torch.testing.assert_close(edit.x0(state, velocity, 0.25), expected_x0)


class _CaptureVelocityModel:
    def __init__(self, video_velocity: float, audio_velocity: float):
        self.video_velocity = video_velocity
        self.audio_velocity = audio_velocity
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        video_rows = kwargs["x"][0, kwargs["img_pos_info"]["position_ids"]]
        audio_rows = kwargs["audio_x"][0, kwargs["audio_pos_info"]["position_ids"]]
        return (
            torch.full_like(video_rows, self.video_velocity),
            torch.full_like(audio_rows, self.audio_velocity),
        )


def _branch(*, latent_t: int, latent_h: int, latent_w: int, audio_t: int):
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import MiniMaxH3DenoiseBranch
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence

    packed = minimax_h3_packed_sequence(
        text_len=2,
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        include_keyframe_cond=False,
    )
    return MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(2, 4),
        token_tags=packed["token_tags"],
        device=torch.device("cpu"),
    )


def test_denoise_loop_uses_masked_model_rows_timesteps_and_original_euler_state():
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import minimax_h3_denoise_loop
    from vllm_omni.diffusion.models.minimax_h3.latent_mask import MiniMaxH3LatentEdit

    branch = _branch(latent_t=3, latent_h=2, latent_w=2, audio_t=2)
    video_state = torch.tensor([2.0, 4.0, 6.0])[:, None].expand(-1, 96).clone()
    audio_state = torch.tensor([1.0, 3.0, 5.0, 7.0])[:, None].expand(-1, 32).clone()
    video_clean = torch.full_like(video_state, 10.0)
    video_anchor = torch.full_like(video_state, 8.0)
    audio_clean = torch.full_like(audio_state, 9.0)
    audio_anchor = audio_clean.clone()
    video_mask = torch.tensor([0.0, 0.5, 1.0])
    audio_mask = torch.tensor([0.0, 0.25, 0.5, 1.0])
    video_edit = MiniMaxH3LatentEdit.from_rows(video_clean, video_anchor, video_mask)
    audio_edit = MiniMaxH3LatentEdit.from_rows(audio_clean, audio_anchor, audio_mask)
    assert video_edit is not None and audio_edit is not None

    model = _CaptureVelocityModel(video_velocity=2.0, audio_velocity=4.0)
    final_video, final_audio = minimax_h3_denoise_loop(
        model=model,
        positive=branch,
        initial_video_rows=video_state,
        initial_audio_rows=audio_state,
        keyframe_cond_rows=None,
        video_edit=video_edit,
        audio_edit=audio_edit,
        sigmas_video=[0.75, 0.25],
        sigmas_audio=[0.5, 0.2],
        device=torch.device("cpu"),
    )

    call = model.calls[0]
    model_video = call["x"][0, call["img_pos_info"]["position_ids"]]
    model_audio = call["audio_x"][0, call["audio_pos_info"]["position_ids"]]
    expected_model_video = video_anchor + video_mask[:, None] * (video_state - video_anchor)
    expected_model_audio = audio_anchor + audio_mask[:, None] * (audio_state - audio_anchor)
    torch.testing.assert_close(model_video, expected_model_video)
    torch.testing.assert_close(model_audio, expected_model_audio)

    row_t = call["unique_timesteps"][call["inverse_indices"]]
    torch.testing.assert_close(
        row_t[call["img_pos_info"]["position_ids"]],
        torch.tensor([0.999, 0.625, 0.25]),
    )
    torch.testing.assert_close(
        row_t[call["audio_pos_info"]["position_ids"]],
        torch.tensor([1.0, 0.875, 0.75, 0.5]),
    )

    # The Euler source is the original state; the blended rows are only the
    # model input. Nonzero next sigmas make that distinction observable.
    video_predicted = expected_model_video + 0.75 * 2.0 * video_mask[:, None]
    audio_predicted = expected_model_audio + 0.5 * 4.0 * audio_mask[:, None]
    video_x0 = video_clean + video_mask[:, None] * (video_predicted - video_clean)
    audio_x0 = audio_clean + audio_mask[:, None] * (audio_predicted - audio_clean)
    expected_video = (0.25 / 0.75) * video_state + (1.0 - 0.25 / 0.75) * video_x0
    expected_audio = (0.2 / 0.5) * audio_state + (1.0 - 0.2 / 0.5) * audio_x0
    torch.testing.assert_close(final_video, expected_video)
    torch.testing.assert_close(final_audio, expected_audio)


def test_batched_forward_supports_per_request_target_timesteps_and_legacy_defaults():
    from vllm_omni.diffusion.models.minimax_h3.batched_packing import minimax_h3_batched_forward_kwargs

    first = _branch(latent_t=2, latent_h=2, latent_w=2, audio_t=2)
    second = _branch(latent_t=3, latent_h=2, latent_w=2, audio_t=1)
    branches = [first, second]
    video_rows = [torch.zeros(first.img_pos.numel(), 96), torch.zeros(second.img_pos.numel(), 96)]
    audio_rows = [torch.zeros(first.audio_pos.numel(), 32), torch.zeros(second.audio_pos.numel(), 32)]
    common = dict(
        branches=branches,
        video_rows=video_rows,
        audio_rows=audio_rows,
        t_video=[0.2, 0.3],
        t_audio=[0.4, 0.5],
        imgvid_cond_timesteps=[0.999, 0.999],
        audio_ref_cond_timesteps=[1.0, 1.0],
    )

    legacy = minimax_h3_batched_forward_kwargs(**common)
    legacy_rows = legacy["unique_timesteps"][legacy["inverse_indices"]]
    first_video_pos = legacy["img_pos_info"]["position_ids"][: first.img_pos.numel()]
    second_video_pos = legacy["img_pos_info"]["position_ids"][first.img_pos.numel() :]
    assert torch.all(legacy_rows[first_video_pos] == 0.2)
    assert torch.all(legacy_rows[second_video_pos] == 0.3)

    video_times = [torch.tensor([0.9, 0.2]), torch.tensor([0.7, 0.6, 0.3])]
    audio_times = [torch.tensor([1.0, 0.8, 1.0, 0.4]), torch.tensor([0.75, 0.5])]
    masked = minimax_h3_batched_forward_kwargs(
        **common,
        video_target_timesteps=video_times,
        audio_target_timesteps=audio_times,
    )
    masked_rows = masked["unique_timesteps"][masked["inverse_indices"]]
    img_pos = masked["img_pos_info"]["position_ids"]
    audio_pos = masked["audio_pos_info"]["position_ids"]
    torch.testing.assert_close(masked_rows[img_pos], torch.cat(video_times))
    torch.testing.assert_close(masked_rows[audio_pos], torch.cat(audio_times))


def test_batched_fractional_edit_keeps_ref2va_condition_rows_request_local():
    """Request offsets must not move edit timesteps onto Ref2VA anchors."""
    from vllm_omni.diffusion.models.minimax_h3.batched_packing import minimax_h3_batched_forward_kwargs
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import MiniMaxH3DenoiseBranch
    from vllm_omni.diffusion.models.minimax_h3.latent_mask import MiniMaxH3LatentEdit
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence_ref2va_blocks,
    )

    def ref2va_branch(
        *,
        text_len: int,
        latent_t: int,
        audio_t: int,
        ref_latent_t: int,
        ref_audio_t: int,
        seq_len: int,
    ) -> MiniMaxH3DenoiseBranch:
        packed = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=text_len,
            latent_t=latent_t,
            latent_h=2,
            latent_w=2,
            audio_t=audio_t,
            ref_blocks=[
                {
                    "kind": "video_audio",
                    "ref_audio_t": ref_audio_t,
                    "latent_t": ref_latent_t,
                    "latent_h": 2,
                    "latent_w": 2,
                }
            ],
            seq_len=seq_len,
        )
        return MiniMaxH3DenoiseBranch(
            packed=packed,
            text_embeddings=torch.zeros(text_len, 4),
            token_tags=packed["token_tags"],
            device=torch.device("cpu"),
        )

    # Different target/reference lengths and padded sequence lengths make a
    # missing request offset observable in both media streams.
    branches = [
        ref2va_branch(
            text_len=2,
            latent_t=2,
            audio_t=2,
            ref_latent_t=1,
            ref_audio_t=1,
            seq_len=64,
        ),
        ref2va_branch(
            text_len=3,
            latent_t=3,
            audio_t=1,
            ref_latent_t=2,
            ref_audio_t=2,
            seq_len=128,
        ),
    ]
    video_masks = [
        torch.tensor([0.0, 0.5]),
        torch.tensor([0.25, 0.6, 1.0]),
    ]
    audio_masks = [
        torch.tensor([0.0, 0.25, 0.75, 1.0]),
        torch.tensor([0.4, 0.8]),
    ]
    t_video = [0.2, 0.35]
    t_audio = [0.4, 0.55]
    imgvid_cond_timesteps = [0.91, 0.97]
    audio_ref_cond_timesteps = [0.93, 0.99]

    video_rows: list[torch.Tensor] = []
    audio_rows: list[torch.Tensor] = []
    video_target_timesteps: list[torch.Tensor] = []
    audio_target_timesteps: list[torch.Tensor] = []
    for index, branch in enumerate(branches):
        base = float(100 * (index + 1))
        request_video = (
            torch.arange(branch.img_pos.numel(), dtype=torch.float32).unsqueeze(1).expand(-1, 96).clone() + base
        )
        request_audio = (
            torch.arange(branch.audio_pos.numel(), dtype=torch.float32).unsqueeze(1).expand(-1, 32).clone()
            + base
            + 50.0
        )

        video_state = request_video[branch.update_mask]
        video_clean = torch.full_like(video_state, base + 80.0)
        video_anchor = torch.full_like(video_state, base + 40.0)
        video_edit = MiniMaxH3LatentEdit.from_rows(
            video_clean,
            video_anchor,
            video_masks[index],
        )
        audio_state = request_audio[branch.audio_update_mask]
        audio_clean = torch.full_like(audio_state, base + 90.0)
        audio_anchor = torch.full_like(audio_state, base + 60.0)
        audio_edit = MiniMaxH3LatentEdit.from_rows(
            audio_clean,
            audio_anchor,
            audio_masks[index],
        )
        assert video_edit is not None and audio_edit is not None

        expected_video_targets = torch.lerp(
            video_anchor,
            video_state,
            video_edit.mask_rows[:, None],
        )
        expected_audio_targets = torch.lerp(
            audio_anchor,
            audio_state,
            audio_edit.mask_rows[:, None],
        )
        request_video[branch.update_mask] = video_edit.model_rows(video_state)
        request_audio[branch.audio_update_mask] = audio_edit.model_rows(audio_state)
        torch.testing.assert_close(request_video[branch.update_mask], expected_video_targets)
        torch.testing.assert_close(request_audio[branch.audio_update_mask], expected_audio_targets)

        video_rows.append(request_video)
        audio_rows.append(request_audio)
        video_target_timesteps.append(
            video_edit.target_timesteps(
                t_video[index],
                imgvid_cond_timesteps[index],
            )
        )
        audio_target_timesteps.append(
            audio_edit.target_timesteps(
                t_audio[index],
                audio_ref_cond_timesteps[index],
            )
        )

    packed = minimax_h3_batched_forward_kwargs(
        branches=branches,
        video_rows=video_rows,
        audio_rows=audio_rows,
        t_video=t_video,
        t_audio=t_audio,
        imgvid_cond_timesteps=imgvid_cond_timesteps,
        audio_ref_cond_timesteps=audio_ref_cond_timesteps,
        video_target_timesteps=video_target_timesteps,
        audio_target_timesteps=audio_target_timesteps,
    )
    row_timesteps = packed["unique_timesteps"][packed["inverse_indices"]]
    assert packed["packed_seq_params"]["cu_seqlens_q"].tolist() == [0, 11, 64, 78, 192]

    expected_img_pos: list[torch.Tensor] = []
    expected_audio_pos: list[torch.Tensor] = []
    seq_offset = 0
    for index, branch in enumerate(branches):
        img_pos = branch.img_pos_dev + seq_offset
        audio_pos = branch.audio_pos_dev + seq_offset
        expected_img_pos.append(img_pos)
        expected_audio_pos.append(audio_pos)

        # Model rows and per-token edit timesteps affect target rows only.
        # Ref2VA visual/audio anchors keep their own request's condition pin.
        torch.testing.assert_close(packed["x"][0, img_pos], video_rows[index])
        torch.testing.assert_close(packed["audio_x"][0, audio_pos], audio_rows[index])
        torch.testing.assert_close(
            row_timesteps[img_pos[branch.update_mask_dev]],
            video_target_timesteps[index],
        )
        torch.testing.assert_close(
            row_timesteps[audio_pos[branch.audio_update_mask_dev]],
            audio_target_timesteps[index],
        )
        assert (~branch.update_mask_dev).any()
        assert (~branch.audio_update_mask_dev).any()
        torch.testing.assert_close(
            row_timesteps[img_pos[~branch.update_mask_dev]],
            torch.full_like(img_pos[~branch.update_mask_dev], imgvid_cond_timesteps[index], dtype=torch.float32),
        )
        torch.testing.assert_close(
            row_timesteps[audio_pos[~branch.audio_update_mask_dev]],
            torch.full_like(
                audio_pos[~branch.audio_update_mask_dev],
                audio_ref_cond_timesteps[index],
                dtype=torch.float32,
            ),
        )
        torch.testing.assert_close(
            row_timesteps[branch.text_pos_dev + seq_offset],
            torch.full((branch.text_len,), t_video[index]),
        )
        seq_offset += branch.seq_len

    torch.testing.assert_close(
        packed["img_pos_info"]["position_ids"],
        torch.cat(expected_img_pos),
    )
    torch.testing.assert_close(
        packed["audio_pos_info"]["position_ids"],
        torch.cat(expected_audio_pos),
    )
    torch.testing.assert_close(
        packed["update_mask"],
        torch.cat([branch.update_mask_dev for branch in branches]),
    )
