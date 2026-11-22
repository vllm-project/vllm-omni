# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packed item order, token roles and caption documents."""

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.geometry import SensorGrid, build_chunk_schedule
from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.layout import (
    CAPTION_SCOPE_ALL,
    CAPTION_SCOPE_SAME_VIEW,
    ROLE_CLEAN_TARGET,
    ROLE_CONTROL,
    ROLE_CURRENT_TARGET,
    ROLE_TARGET_CONDITION,
    CaptionDocument,
    build_chunk_items,
    caption_metadata,
    chunk_token_metadata,
    committed_metadata,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

RGB = SensorGrid(2, 2, 2, 2)  # 1 token per frame
LIDAR = SensorGrid(2, 4, 2, 2)  # 2 tokens per sweep


def schedule():
    return build_chunk_schedule(
        rgb_latents=7, lidar_sweeps=13, frames_per_chunk=2, rgb_seconds_per_frame=0.2, lidar_seconds_per_frame=0.1
    )


def test_item_order_and_token_counts() -> None:
    chunk = schedule()[1]
    items = build_chunk_items(
        chunk,
        num_views=2,
        rgb_grid=RGB,
        rgb_seconds_per_frame=0.2,
        lidar_grid=LIDAR,
        lidar_seconds_per_frame=0.1,
    )
    assert [item.kind for item in items] == ["rgb_control", "rgb_target", "lidar_control", "lidar_target"]
    assert items[0].view_ids == (0, 1) and items[2].view_ids == (2,)
    assert items[1].frames == (1, 2) and items[3].frames == (1, 2, 3, 4)
    assert [item.num_tokens for item in items] == [4, 4, 8, 8]
    assert items[1].caption_scope == CAPTION_SCOPE_SAME_VIEW and items[3].caption_scope == CAPTION_SCOPE_ALL
    metadata, offsets = chunk_token_metadata(items, step=1, pass_kind="noisy")
    assert offsets == (0, 4, 8, 16) and metadata.num_tokens == 24
    # Camera-major: both frames of view 0 precede view 1.
    assert metadata.view_id[:4].tolist() == [0, 0, 1, 1]
    assert metadata.frame_id[:4].tolist() == [1, 2, 1, 2]
    assert torch.allclose(metadata.timestamp[4:8], torch.tensor([0.2, 0.4, 0.2, 0.4]))
    assert torch.allclose(metadata.timestamp[16:18], torch.tensor([0.1, 0.1]))
    assert metadata.step.unique().tolist() == [1]


def test_roles_condition_noisy_clean_and_commit() -> None:
    chunk0 = schedule()[0]
    items = build_chunk_items(
        chunk0,
        num_views=2,
        rgb_grid=RGB,
        rgb_seconds_per_frame=0.2,
        lidar_grid=LIDAR,
        lidar_seconds_per_frame=0.1,
        rgb_condition_frames=(0,),
    )
    noisy, _ = chunk_token_metadata(items, step=0, pass_kind="noisy")
    roles = noisy.role.tolist()
    assert roles[:2] == [ROLE_CONTROL, ROLE_CONTROL]
    assert roles[2:4] == [ROLE_TARGET_CONDITION, ROLE_TARGET_CONDITION]
    assert roles[4:6] == [ROLE_CONTROL, ROLE_CONTROL]
    assert roles[6:8] == [ROLE_CURRENT_TARGET, ROLE_CURRENT_TARGET]
    clean, _ = chunk_token_metadata(items, step=0, pass_kind="clean")
    assert clean.role.tolist()[6:8] == [ROLE_CLEAN_TARGET, ROLE_CLEAN_TARGET]
    assert clean.role.tolist()[2:4] == [ROLE_TARGET_CONDITION, ROLE_TARGET_CONDITION]
    committed = committed_metadata(noisy)
    assert torch.equal(committed.role, clean.role)
    assert not bool(committed.is_noisy.any())


def test_lidar_condition_sweep_role() -> None:
    chunk0 = schedule()[0]
    items = build_chunk_items(
        chunk0,
        num_views=1,
        rgb_grid=RGB,
        rgb_seconds_per_frame=0.2,
        lidar_grid=LIDAR,
        lidar_seconds_per_frame=0.1,
        rgb_condition_frames=(0,),
        lidar_condition_sweeps=(0,),
    )
    metadata, _ = chunk_token_metadata(items, step=0, pass_kind="noisy")
    assert metadata.role[-2:].tolist() == [ROLE_TARGET_CONDITION, ROLE_TARGET_CONDITION]


def test_caption_documents_pack_sink_then_caption_per_view() -> None:
    docs = (
        CaptionDocument(0, 2, is_sink=True),
        CaptionDocument(0, 3, start_seconds=0.0, end_seconds=1.4),
        CaptionDocument(1, 2, is_sink=True),
        CaptionDocument(1, 4, start_seconds=0.0, end_seconds=1.4),
    )
    captions = caption_metadata(docs)
    assert captions.num_tokens == 11 and captions.doc_offsets == (0, 2, 5, 7)
    assert captions.view_id.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    assert captions.is_sink.tolist() == [True, True, False, False, False, True, True, False, False, False, False]
    assert captions.start_seconds[0] == float("-inf") and captions.end_seconds[0] == float("inf")
    assert captions.start_seconds[2] == 0.0 and float(captions.end_seconds[2]) == pytest.approx(1.4)
    with pytest.raises(ValueError, match="visible at every time"):
        CaptionDocument(0, 2, is_sink=True, start_seconds=0.0)
