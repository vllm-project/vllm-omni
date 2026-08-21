# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.forward_context import (
    ForwardContext,
    get_forward_context,
    override_forward_context,
)
from vllm_omni.diffusion.models.ltx2.ltx2_tiled_data_parallel import (
    _blend_tile_output,
    build_spatial_tiling_plan,
    forward_tiled_data_parallel,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


class _NoopGroup:
    @staticmethod
    def all_reduce(value):
        return value


def test_four_gpu_4k_latent_tile_geometry_and_partition_of_unity():
    plan = build_spatial_tiling_plan(
        num_frames=16,
        height=68,
        width=120,
        world_size=4,
        rank=0,
        overlap=5,
    )

    assert (plan.grid_rows, plan.grid_columns) == (2, 2)
    assert [(tile.height.start, tile.height.end) for tile in plan.tiles] == [
        (0, 37),
        (0, 37),
        (32, 68),
        (32, 68),
    ]
    assert [(tile.width.start, tile.width.end) for tile in plan.tiles] == [
        (0, 63),
        (58, 120),
        (0, 63),
        (58, 120),
    ]
    assert [tile.token_count for tile in plan.tiles] == [37296, 36704, 36288, 35712]

    weights = torch.zeros(1, plan.token_count, 1)
    for tile in plan.tiles:
        _blend_tile_output(weights, tile, torch.ones(1, tile.token_count, 1))
    torch.testing.assert_close(weights, torch.ones_like(weights))


def test_blending_identity_tiles_reconstructs_packed_video():
    plan = build_spatial_tiling_plan(
        num_frames=2,
        height=12,
        width=20,
        world_size=4,
        rank=0,
        overlap=3,
    )
    packed = torch.arange(plan.token_count, dtype=torch.float32).view(1, -1, 1)
    reconstructed = torch.zeros_like(packed)
    for tile in plan.tiles:
        _blend_tile_output(reconstructed, tile, packed.index_select(1, tile.token_indices))
    torch.testing.assert_close(reconstructed, packed)


def test_tile_grid_matches_official_balanced_factor_split_for_portrait_input():
    plan = build_spatial_tiling_plan(
        num_frames=1,
        height=80,
        width=40,
        world_size=8,
        rank=0,
        overlap=2,
    )

    assert (plan.grid_rows, plan.grid_columns) == (2, 4)


def test_single_tile_forward_matches_direct_transformer_and_normalizes_coords():
    plan = build_spatial_tiling_plan(
        num_frames=2,
        height=3,
        width=4,
        world_size=1,
        rank=0,
        overlap=5,
    )
    hidden_states = torch.randn(1, plan.token_count, 3)
    audio_hidden_states = torch.randn(1, 5, 2)
    video_coords = torch.arange(1 * 3 * plan.token_count * 2, dtype=torch.float32).view(1, 3, plan.token_count, 2)

    class _Transformer:
        config = SimpleNamespace()

        def __call__(self, **kwargs):
            assert get_forward_context().sequence_parallel_enabled is False
            assert get_forward_context().sp_active is False
            assert kwargs["height"] == 3
            assert kwargs["width"] == 4
            assert torch.all(kwargs["video_coords"][..., 0].amin(dim=2) == 0)
            return kwargs["hidden_states"] + 1, kwargs["audio_hidden_states"] + 2

    context = ForwardContext(sp_plan_hooks_applied=True, _sp_shard_depth=2)
    with override_forward_context(context):
        video, audio = forward_tiled_data_parallel(
            _Transformer(),
            {
                "hidden_states": hidden_states,
                "audio_hidden_states": audio_hidden_states,
                "video_coords": video_coords,
                "height": 3,
                "width": 4,
                "timestep": torch.ones(1, plan.token_count),
                "keyframes_mask": torch.ones(1, plan.token_count, 1),
            },
            plan,
            _NoopGroup(),
        )
        assert context.sequence_parallel_enabled is True
        assert context._sp_shard_depth == 2

    torch.testing.assert_close(video, hidden_states + 1)
    torch.testing.assert_close(audio, audio_hidden_states + 2)
