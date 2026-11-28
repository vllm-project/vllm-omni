# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chunk clock and token geometry against the reference numbers (297 frames, 7 views, 480x832)."""

import pytest

from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.geometry import (
    build_chunk_schedule,
    chunk_token_count,
    latent_frames_from_pixel_frames,
    lidar_grid,
    lidar_num_sweeps,
    max_chunk_tokens,
    max_sweeps_per_chunk,
    paged_frame_unit,
    rgb_grid,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

RGB_SPF = 4 / 30
LIDAR_SPF = 0.1


def reference_schedule(joint: bool):
    latents = latent_frames_from_pixel_frames(297)
    sweeps = lidar_num_sweeps(297, camera_fps=30, lidar_fps=10) if joint else None
    return build_chunk_schedule(
        rgb_latents=latents,
        lidar_sweeps=sweeps,
        frames_per_chunk=3,
        rgb_seconds_per_frame=RGB_SPF,
        lidar_seconds_per_frame=LIDAR_SPF if joint else None,
    )


def test_reference_horizon_counts() -> None:
    assert latent_frames_from_pixel_frames(297) == 75
    assert lidar_num_sweeps(297, camera_fps=30, lidar_fps=10) == 99


def test_joint_schedule_singleton_regular_and_partial_tail() -> None:
    schedule = reference_schedule(joint=True)
    assert len(schedule) == 26
    assert schedule[0].rgb_frames == (0,) and schedule[0].lidar_sweeps == (0,)
    assert schedule[1].rgb_frames == (1, 2, 3) and schedule[1].lidar_sweeps == (1, 2, 3, 4)
    assert schedule[24].rgb_frames == (70, 71, 72) and schedule[24].lidar_sweeps == (93, 94, 95, 96)
    assert schedule[25].rgb_frames == (73, 74) and schedule[25].lidar_sweeps == (97, 98)
    assert schedule[25].rgb_prefix_end == 75 and schedule[25].lidar_prefix_end == 99
    assert [chunk.step for chunk in schedule] == list(range(26))


def test_rgb_only_schedule_has_no_sweeps() -> None:
    schedule = reference_schedule(joint=False)
    assert len(schedule) == 26
    assert all(chunk.lidar_sweeps == () and chunk.lidar_prefix_end == 0 for chunk in schedule)


def test_joint_schedule_requires_both_sensors_per_step() -> None:
    # One sweep per 0.4 s chunk cannot fill step 1 (sweeps at 0 s and 1 s only).
    with pytest.raises(ValueError, match="both RGB latents and LiDAR sweeps"):
        build_chunk_schedule(
            rgb_latents=4,
            lidar_sweeps=2,
            frames_per_chunk=3,
            rgb_seconds_per_frame=RGB_SPF,
            lidar_seconds_per_frame=1.0,
        )


@pytest.mark.parametrize(
    ("camera_fps", "lidar_fps", "expected"),
    [
        (30.0, 10.0, 4),  # the reference clock
        (30.0, 11.0, 5),  # 4.4 sweeps per chunk: rounding would size for four
        (30.0, 12.0, 5),  # 4.8
        (25.0, 10.0, 5),  # 4.8
        (24.0, 10.0, 5),  # exactly 5
        (30.0, 20.0, 8),
        (10.0, 10.0, 12),
    ],
)
def test_max_sweeps_per_chunk_bounds_every_schedule(camera_fps: float, lidar_fps: float, expected: int) -> None:
    bound = max_sweeps_per_chunk(
        frames_per_chunk=3, rgb_seconds_per_frame=4 / camera_fps, lidar_seconds_per_frame=1 / lidar_fps
    )
    assert bound == expected
    largest = 0
    for num_frames in range(5, 400, 4):
        schedule = build_chunk_schedule(
            rgb_latents=latent_frames_from_pixel_frames(num_frames),
            lidar_sweeps=lidar_num_sweeps(num_frames, camera_fps=camera_fps, lidar_fps=lidar_fps),
            frames_per_chunk=3,
            rgb_seconds_per_frame=4 / camera_fps,
            lidar_seconds_per_frame=1 / lidar_fps,
        )
        largest = max(largest, *(len(chunk.lidar_sweeps) for chunk in schedule))
    # Every horizon fits, and long ones reach the bound.
    assert largest == bound


def test_pixel_frames_must_be_wan_aligned() -> None:
    with pytest.raises(ValueError, match="4k \\+ 1"):
        latent_frames_from_pixel_frames(296)


def test_token_grids_and_frame_unit() -> None:
    rgb = rgb_grid(480, 832)
    lidar = lidar_grid()
    assert (rgb.grid_height, rgb.grid_width, rgb.tokens_per_frame) == (15, 26, 390)
    assert (lidar.grid_height, lidar.grid_width, lidar.tokens_per_frame) == (4, 57, 228)
    joint = reference_schedule(joint=True)
    assert chunk_token_count(joint[0], num_views=7, rgb_tokens_per_frame=390, lidar_tokens_per_sweep=228) == 5916
    assert chunk_token_count(joint[1], num_views=7, rgb_tokens_per_frame=390, lidar_tokens_per_sweep=228) == 18204
    assert max_chunk_tokens(joint, num_views=7, rgb_tokens_per_frame=390, lidar_tokens_per_sweep=228) == 18204
    assert paged_frame_unit(18204) == 18208
    rgb_only = reference_schedule(joint=False)
    assert max_chunk_tokens(rgb_only, num_views=7, rgb_tokens_per_frame=390, lidar_tokens_per_sweep=None) == 16380
    assert paged_frame_unit(16380) == 16384
    assert paged_frame_unit(16384) == 16384
