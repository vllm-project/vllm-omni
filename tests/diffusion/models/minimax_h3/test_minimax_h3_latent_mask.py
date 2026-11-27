# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Focused CPU tests for MiniMax H3 latent-mask editing."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_video_full_grid_quantization_and_cell_level_restore():
    from vllm_omni.diffusion.models.minimax_h3.latent_mask import minimax_h3_video_edit_masks

    shape = dict(latent_t=2, latent_h=4, latent_w=6)
    full = torch.arange(48, dtype=torch.float32).reshape(2, 4, 6) / 48
    cells = full.reshape(2, 2, 2, 3, 2).permute(0, 1, 3, 2, 4)
    parsed = minimax_h3_video_edit_masks(full, **shape)
    expected_model = torch.ceil(cells.amax(dim=(3, 4)).reshape(-1) * 256) / 256
    expected_restore = cells.unsqueeze(3).expand(2, 2, 3, 24, 2, 2).reshape(12, 96)
    torch.testing.assert_close(parsed.model_mask_rows, expected_model)
    torch.testing.assert_close(parsed.restore_mask_rows, expected_restore)


def test_audio_grid_uses_channel_major_order_and_preserves_raw_values():
    from vllm_omni.diffusion.models.minimax_h3.latent_mask import minimax_h3_audio_edit_masks

    grid = torch.tensor([[0.0, 0.1, 1.0], [0.25, 0.5, 0.75]])
    expected = grid.flatten()
    parsed = minimax_h3_audio_edit_masks(grid, audio_t=3)
    torch.testing.assert_close(parsed.model_mask_rows, torch.ceil(expected * 256) / 256)
    torch.testing.assert_close(parsed.restore_mask_rows, expected)


def test_zero_fractional_one_semantics_and_cell_level_restore():
    from vllm_omni.diffusion.models.minimax_h3.latent_mask import (
        MiniMaxH3LatentEdit,
        minimax_h3_video_edit_masks,
    )

    clean = torch.full((3, 2), 10.0)
    anchor = torch.tensor([[8.0, 8.0], [6.0, 6.0], [4.0, 4.0]])
    state = torch.tensor([[2.0, 2.0], [4.0, 4.0], [6.0, 6.0]])
    velocity = torch.full((3, 2), 2.0)
    mask = torch.tensor([0.0, 0.5, 1.0])
    edit = MiniMaxH3LatentEdit.from_rows(clean, anchor, mask, mask)
    assert edit is not None

    model_rows, target_timesteps = edit.prepare(state, 0.25, 0.999, sigma=0.75)
    torch.testing.assert_close(model_rows, torch.lerp(anchor, state, mask[:, None]))
    torch.testing.assert_close(
        target_timesteps,
        torch.tensor([0.999, 0.625, 0.25]),
    )
    predicted = model_rows + 0.75 * velocity * mask[:, None]
    torch.testing.assert_close(edit.x0(model_rows, velocity, 0.25), torch.lerp(clean, predicted, mask[:, None]))

    # A single active cell drives the full model token, while raw restoration
    # preserves the other three cells independently in every packed channel.
    parsed = minimax_h3_video_edit_masks(
        torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
        latent_t=1,
        latent_h=2,
        latent_w=2,
    )
    restore = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(24).reshape(1, 96)
    torch.testing.assert_close(parsed.model_mask_rows, torch.ones(1))
    torch.testing.assert_close(parsed.restore_mask_rows, restore)

    cell_edit = MiniMaxH3LatentEdit.from_rows(
        torch.full((1, 96), 10.0),
        torch.full((1, 96), 8.0),
        parsed.model_mask_rows,
        parsed.restore_mask_rows,
    )
    assert cell_edit is not None
    cell_state = torch.full((1, 96), 2.0)
    cell_velocity = torch.full((1, 96), 4.0)
    expected = torch.where(restore.bool(), cell_state + 0.75 * cell_velocity, cell_edit.clean_rows)
    torch.testing.assert_close(cell_edit.x0(cell_state, cell_velocity, 0.25), expected)
