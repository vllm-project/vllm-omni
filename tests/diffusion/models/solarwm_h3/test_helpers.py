# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib
import sys
import types
from pathlib import Path

import pytest
import torch

# The math helpers deliberately have no vLLM/CUDA dependency. Keep their CPU
# coverage runnable even in an environment that cannot initialize vLLM.
_PACKAGE = "_solarwm_h3_helper_tests"
_ROOT = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/solarwm_h3"
package = types.ModuleType(_PACKAGE)
package.__path__ = [str(_ROOT)]
sys.modules.setdefault(_PACKAGE, package)
camera = importlib.import_module(f"{_PACKAGE}.camera")
layout = importlib.import_module(f"{_PACKAGE}.layout")
sampling = importlib.import_module(f"{_PACKAGE}.sampling")

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("frames", [22, 23, 158, 960, 2000])
def test_source_geometry_preserves_native_cadence_and_tail(frames):
    geometry = layout.full_length_geometry(frames)
    assert geometry.decoded_frames >= frames
    assert geometry.decoded_frames % 17 == 5
    assert geometry.decode_latents == 5 * ((geometry.decoded_frames - 5) // 17) + 2
    assert geometry.rollout_latents % 5 == 0
    valid = [
        period + offset for period in range(0, frames, 17) for offset in (0, 1, 5, 9, 13) if period + offset < frames
    ]
    expected = valid[: geometry.rollout_latents]
    expected += [expected[-1]] * (geometry.rollout_latents - len(expected))
    assert geometry.camera_indices == tuple(expected)


def test_158_frame_anchor_is_separate_from_47_generated_latents():
    geometry = layout.full_length_geometry(158)
    assert (geometry.decode_latents, geometry.rollout_latents, geometry.tail_camera_rows) == (47, 50, 3)
    assert geometry.camera_indices[-4:] == (154, 154, 154, 154)
    plan = layout.build_layout(torch.tensor([1, 0, 0, 1]), latent_height=4, latent_width=6, num_audio_latents=3)
    assert plan.condition_indices.tolist() == list(range(10))
    assert plan.audio_indices.tolist() == list(range(10, 16))
    assert plan.video_indices.tolist() == list(range(4, 10)) + list(range(16, 46))
    assert plan.noisy_video_output_slice == slice(6, 36)
    unique, inverse = layout.build_row_timesteps(plan, 0.2, 0.3)
    rows = unique[inverse]
    torch.testing.assert_close(rows[plan.text_indices], torch.ones(4))
    torch.testing.assert_close(rows[plan.condition_video_indices], torch.full((6,), 0.999))
    torch.testing.assert_close(rows[plan.audio_indices], torch.full((6,), 0.3))
    torch.testing.assert_close(rows[plan.noisy_video_indices], torch.full((30,), 0.2))
    # Qwen image rows are tag zero but have clean text time, not anchor/noisy time.
    assert plan.token_tags[1].item() == 0 and rows[1].item() == 1


def test_native_positions_have_nonuniform_time_and_endpoint_excluded_space():
    positions = layout.native_video_position_grid(6, 4, 4, origin=7).reshape(6, 4, 3)
    expected = torch.tensor([7, 7 + 5 / 3, 7 + 25 / 3, 22, 7 + 65 / 3, 7 + 85 / 3], dtype=torch.float64)
    torch.testing.assert_close(positions[:, 0, 0], expected)
    torch.testing.assert_close(
        positions[0, :, 1:], torch.tensor([[0, 0], [0, 16], [16, 0], [16, 16]], dtype=torch.float64)
    )
    assert positions.dtype == torch.float64


def test_video_patch_roundtrip_is_spatially_and_channel_ordered():
    value = torch.arange(2 * 3 * 5 * 4 * 6).reshape(2, 3, 5, 4, 6)
    patches = layout.patchify_video(value)
    assert patches[0, 0].tolist() == value[0, :, 0, :2, :2].reshape(-1).tolist()
    torch.testing.assert_close(layout.unpatchify_video(patches, 5, 4, 6), value)
    with pytest.raises(ValueError, match="match"):
        layout.unpatchify_video(patches[:, :-1], 5, 4, 6)


def test_camera_preparation_uses_first_relative_pose_and_latent_camera_tail():
    poses = torch.eye(4).repeat(158, 1, 1)
    poses[:, 0, 3] = torch.arange(158) + 10
    intrinsics = torch.tensor([[1000.0, 0, 672], [0, 900, 384], [0, 0, 1]])
    prepared = camera.prepare_camera(poses, intrinsics, num_frames=158)
    assert prepared.viewmats.shape == (1, 51, 4, 4)
    torch.testing.assert_close(prepared.viewmats[0, 0], torch.eye(4))
    torch.testing.assert_close(prepared.viewmats[0, 1], torch.eye(4))
    assert prepared.viewmats[0, 2, 0, 3].item() == -1
    assert prepared.viewmats[0, -1, 0, 3].item() == -154
    assert prepared.intrinsics[0, 0, 0, 0].item() == pytest.approx(1000 / 1344)
    # A change of world coordinates cannot change first-frame-relative poses.
    world = torch.eye(4)
    world[:2, :2] = torch.tensor([[0.0, -1], [1, 0]])
    world[:3, 3] = torch.tensor([3.0, 5, -2])
    moved = camera.prepare_camera(world @ poses, intrinsics, num_frames=158)
    torch.testing.assert_close(moved.viewmats, prepared.viewmats)


@pytest.mark.parametrize("invalid", ["reflection", "scale", "nan", "misaligned"])
def test_invalid_camera_trajectories_fail_before_model_work(invalid):
    poses = torch.eye(4).repeat(22, 1, 1)
    if invalid == "reflection":
        poses[5, 0, 0] = -1
    elif invalid == "scale":
        poses[5, 0, 0] = 2
    elif invalid == "nan":
        poses[5, 0, 3] = float("nan")
    else:
        poses = poses[:-1]
    with pytest.raises(ValueError, match="camera c2w"):
        camera.prepare_camera(poses, torch.eye(3), num_frames=22)


def test_projective_suffix_preserves_dot_products_and_restores_value_basis():
    generator = torch.Generator().manual_seed(4)
    q, k, v = [torch.randn(1, 2, 3, 128, generator=generator) for _ in range(3)]
    views = torch.eye(4).repeat(1, 3, 1, 1)
    views[..., 0, 3] = 4
    intrinsics = torch.eye(3).repeat(1, 3, 1, 1)
    transformed_q, transformed_k, transformed_v, restore = camera.prope_qkv_separate(
        q, k, v, query_viewmats=views, query_K=intrinsics, key_viewmats=views, key_K=intrinsics
    )
    torch.testing.assert_close(transformed_q[..., :96], q[..., :96], rtol=0, atol=0)
    torch.testing.assert_close(transformed_k[..., :96], k[..., :96], rtol=0, atol=0)
    torch.testing.assert_close(restore(transformed_v), v, rtol=2e-6, atol=1e-6)
    torch.testing.assert_close((transformed_q * transformed_k).sum(-1), (q * k).sum(-1), rtol=2e-6, atol=2e-6)
    # The last suffix element carries translation through homogeneous projective coordinates.
    matrix = torch.eye(4)
    matrix[0, 0], matrix[1, 1] = camera.WAN_FIXED_FX, camera.WAN_FIXED_FY
    matrix[0, 3] = camera.WAN_FIXED_FX * torch.log(torch.tensor(5.0)) / 4
    expected = torch.einsum("ij,bhspj->bhspi", matrix.T, q[..., 96:].reshape(1, 2, 3, 8, 4)).flatten(-2)
    torch.testing.assert_close(transformed_q[..., 96:], expected)


def test_bfloat16_fixed_focal_uses_released_reciprocal_rounding():
    # Golden identity-camera values from the pinned upstream implementation.
    # Inverting Python focal constants before the BF16 cast changes both KV
    # focal entries, even though an FP32 projective-invariant test still passes.
    rows = torch.ones(1, 1, 1, 128, dtype=torch.bfloat16)
    view = torch.eye(4).reshape(1, 1, 4, 4)
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3)
    q, k, v, restore = camera.prope_qkv_separate(
        rows, rows, rows, query_viewmats=view, query_K=intrinsics, key_viewmats=view, key_K=intrinsics
    )
    expected_q = torch.tensor([0.50390625, 0.8984375, 1, 1], dtype=torch.bfloat16).repeat(8)
    expected_kv = torch.tensor([1.984375, 1.109375, 1, 1], dtype=torch.bfloat16).repeat(8)
    torch.testing.assert_close(q[..., 96:].flatten(), expected_q, atol=0, rtol=0)
    torch.testing.assert_close(k[..., 96:].flatten(), expected_kv, atol=0, rtol=0)
    torch.testing.assert_close(v[..., 96:].flatten(), expected_kv, atol=0, rtol=0)
    torch.testing.assert_close(restore(rows)[..., 96:].flatten(), expected_q, atol=0, rtol=0)


def test_stage2_uses_four_stochastic_evaluations_then_clean_commit():
    noise = torch.full((1, 2, 5, 2, 2), 0.2)
    calls: list[tuple[torch.Tensor, float]] = []
    noises: list[torch.Tensor] = []
    commits: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(current, time):
        calls.append((current.clone(), time.item()))
        return current * 0.125 + time

    def draw(clean):
        value = torch.full_like(clean, len(noises) + 1)
        noises.append(value)
        return value

    generated = sampling.sample_chunk(
        forward, noise, commit=lambda clean, time: commits.append((clean, time)), noise_fn=draw
    )
    times = sampling.stage2_timesteps()
    torch.testing.assert_close(times, torch.tensor([0, 1 / 37, 1 / 13, 1 / 5]), rtol=2e-6, atol=1e-7)
    current = noise
    for index, time in enumerate(times):
        clean = current + (1 - time) * (current * 0.125 + time)
        torch.testing.assert_close(calls[index][0], current)
        if index < 3:
            current = times[index + 1] * clean + (1 - times[index + 1]) * noises[index]
    torch.testing.assert_close(generated, clean)
    assert len(calls) == 4 and len(noises) == 3 and len(commits) == 1
    assert commits[0][0] is generated and commits[0][1].item() == 1


def test_nonfinite_chunk_does_not_commit_partial_history():
    commits = []
    with pytest.raises(FloatingPointError, match="chunk=6, denoise=0"):
        sampling.sample_chunk(
            lambda value, _: torch.full_like(value, float("nan")),
            torch.zeros(1, 1, 5, 2, 2),
            commit=lambda *args: commits.append(args),
            chunk_index=6,
        )
    assert commits == []
