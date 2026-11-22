# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Token embedding of the Sim-Auto transformer: physical rig identity and timestep gating."""

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.geometry import Chunk, SensorGrid
from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.layout import build_chunk_items
from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.transformer_cosmos3_nano_sim_auto import (
    Cosmos3NanoSimAutoTransformer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

CHANNELS, HIDDEN, RIG_ROWS, LIDAR_ROW = 3, 5, 12, 11


class _Host(nn.Module):
    """The attributes ``embed_items`` and ``validate_loaded_weights`` read, with 1x1 patches."""

    def __init__(self, *, rig: bool) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.proj_in = nn.Linear(CHANNELS, HIDDEN)
        self.lidar_proj_in = nn.Linear(CHANNELS, HIDDEN)
        self.rig_view_embed = nn.Embedding(RIG_ROWS, HIDDEN) if rig else None
        self.rig_lidar_id = LIDAR_ROW if rig else None
        self.time = torch.randn(HIDDEN, generator=generator)

    def _item_patch(self, item) -> tuple[int, int]:
        return (1, 1)

    def _embed_timestep(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return self.time.to(dtype).view(1, -1)


def _joint_items():
    chunk = Chunk(step=1, rgb_frames=(1, 2), lidar_sweeps=(1,), rgb_prefix_end=3, lidar_prefix_end=2)
    items = build_chunk_items(
        chunk,
        num_views=2,
        rgb_grid=SensorGrid(2, 2, 1, 1),
        rgb_seconds_per_frame=4 / 30,
        lidar_grid=SensorGrid(1, 3, 1, 1),
        lidar_seconds_per_frame=0.1,
        rgb_condition_frames={1},
    )
    generator = torch.Generator().manual_seed(1)
    latents = []
    for item in items:
        shape = (1, CHANNELS, len(item.view_ids) * len(item.frames), item.grid.latent_height, item.grid.latent_width)
        latents.append(torch.randn(shape, generator=generator))
    return items, latents


@torch.no_grad()
def test_rig_rows_follow_request_cameras_and_lidar_row_before_timestep() -> None:
    host = _Host(rig=True)
    items, latents = _joint_items()
    rig_view_ids = torch.tensor([5, 0])  # request order: cross-left, then front-wide
    hidden = Cosmos3NanoSimAutoTransformer.embed_items(
        host, items, latents, timestep=torch.tensor([500.0]), rig_view_ids=rig_view_ids
    )

    expected = []
    for item, latent in zip(items, latents, strict=True):
        project = host.lidar_proj_in if item.is_lidar else host.proj_in
        tokens = project(latent[0].permute(1, 2, 3, 0).reshape(-1, CHANNELS))  # (frame, h, w) order
        per_view = tokens.shape[0] // len(item.view_ids)
        if item.is_lidar:
            rows = host.rig_view_embed.weight[LIDAR_ROW].expand_as(tokens)
        else:
            rows = host.rig_view_embed(rig_view_ids).repeat_interleave(per_view, dim=0)
        tokens = tokens + rows
        if not item.is_control:
            frame_of_token = (
                torch.tensor(item.frames).repeat_interleave(item.tokens_per_frame).repeat(len(item.view_ids))
            )
            generated = torch.tensor([int(frame) not in item.condition_frames for frame in frame_of_token])
            tokens = tokens + host.time * generated[:, None]
        expected.append(tokens)
    torch.testing.assert_close(hidden[0], torch.cat(expected))


@torch.no_grad()
def test_rig_view_ids_are_required_exactly_with_a_rig_table() -> None:
    items, latents = _joint_items()
    with pytest.raises(ValueError, match="exactly when the artifact declares a rig table"):
        Cosmos3NanoSimAutoTransformer.embed_items(_Host(rig=True), items, latents, timestep=None)
    with pytest.raises(ValueError, match="exactly when the artifact declares a rig table"):
        Cosmos3NanoSimAutoTransformer.embed_items(
            _Host(rig=False), items, latents, timestep=None, rig_view_ids=torch.tensor([0, 1])
        )
    host = _Host(rig=False)
    hidden = Cosmos3NanoSimAutoTransformer.embed_items(host, items, latents, timestep=None)
    plain = [
        (host.lidar_proj_in if item.is_lidar else host.proj_in)(latent[0].permute(1, 2, 3, 0).reshape(-1, CHANNELS))
        for item, latent in zip(items, latents, strict=True)
    ]
    torch.testing.assert_close(hidden[0], torch.cat(plain))


def test_rig_weights_must_match_the_declared_table() -> None:
    with_rig = _Host(rig=True)
    names = {f"transformer.{name}" for name, _ in with_rig.named_parameters()}
    Cosmos3NanoSimAutoTransformer.validate_loaded_weights(with_rig, names)
    with pytest.raises(ValueError, match="missing required transformer weights: transformer.rig_view_embed.weight"):
        Cosmos3NanoSimAutoTransformer.validate_loaded_weights(with_rig, names - {"transformer.rig_view_embed.weight"})
    without_rig = _Host(rig=False)
    names = {f"transformer.{name}" for name, _ in without_rig.named_parameters()}
    with pytest.raises(ValueError, match="declares no rig_view_embedding"):
        Cosmos3NanoSimAutoTransformer.validate_loaded_weights(
            without_rig, names | {"transformer.rig_view_embed.weight"}
        )
