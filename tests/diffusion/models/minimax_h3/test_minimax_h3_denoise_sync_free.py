# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The denoise step's index-based row updates and host-built timestep unique match the boolean-mask forms."""

import random

import pytest
import torch

from tests.helpers.mark import hardware_marks, hardware_test
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import MiniMaxH3DenoiseBranch
from vllm_omni.diffusion.models.minimax_h3.sampling import MiniMaxH3Model, _assign_rows
from vllm_omni.diffusion.models.minimax_h3.scheduling_minimax_h3_euler_ancestral import minimax_h3_rf_v_to_x0
from vllm_omni.diffusion.sampler import SamplingState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _devices() -> list:
    devices = [pytest.param("cpu", marks=pytest.mark.cpu, id="cpu")]
    if torch.cuda.is_available():
        devices.append(pytest.param("cuda", marks=hardware_marks(res={"cuda": "L4"}, num_cards=1), id="cuda"))
    return devices


def _branch(device: str) -> MiniMaxH3DenoiseBranch:
    generator = torch.Generator().manual_seed(0)
    text, video, audio, pad = 40, 300, 120, 17
    seq_len = text + video + audio + pad
    order = torch.randperm(text + video + audio, generator=generator)
    update = torch.ones(video, dtype=torch.bool)
    update[torch.randperm(video, generator=generator)[:60]] = False
    audio_update = torch.ones(audio, dtype=torch.bool)
    audio_update[:25] = False
    packed = {
        "seq_len": seq_len,
        "img_pos": order[text : text + video],
        "audio_pos": order[text + video :],
        "update_mask": update,
        "audio_update_mask": audio_update,
        "text_pos": order[:text],
        "cu_seqlens": torch.tensor([0, seq_len - pad, seq_len], dtype=torch.int32),
        "img_position_ids": torch.zeros(3, video, dtype=torch.long),
        "latent_grid": torch.tensor([3, 10, 10]),
        "video_row_start": 0,
    }
    return MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(text, 8),
        token_tags=torch.zeros(seq_len, dtype=torch.long),
        device=torch.device(device),
    )


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("locked_audio", [False, True])
def test_host_built_unique_matches_fill_and_torch_unique(device: str, locked_audio: bool) -> None:
    branch = _branch(device)
    branch.locked_audio_rows = torch.zeros(1) if locked_audio else None
    rng = random.Random(0)
    pool = [0.0, 1.0, 0.9999999, 1e-9, 0.25, 0.5]
    for _ in range(200):
        scalars = {
            "t_video": rng.choice(pool + [rng.random()]),
            "t_audio": rng.choice(pool + [rng.random()]),
            "imgvid_cond_timestep": rng.choice(pool + [rng.random()]),
            "audio_ref_cond_timestep": rng.choice(pool + [rng.random()]),
        }
        timesteps = torch.empty(branch.seq_len, dtype=torch.float32, device=device)
        branch.fill_timesteps(timesteps, **scalars)
        expected_unique, expected_inverse = torch.unique(timesteps, sorted=True, return_inverse=True)
        unique, inverse = branch._unique_step_timesteps(**scalars)
        assert unique.dtype == expected_unique.dtype and inverse.dtype == expected_inverse.dtype
        assert torch.equal(unique, expected_unique)
        assert torch.equal(inverse, expected_inverse)


@pytest.mark.parametrize("device", _devices())
def test_index_row_updates_match_boolean_masks(device: str) -> None:
    branch = _branch(device)
    cases = (
        (branch.update_mask_dev, branch.update_idx_dev),
        (~branch.update_mask_dev, branch.cond_idx_dev),
        (branch.audio_update_mask_dev, branch.audio_update_idx_dev),
        (~branch.audio_update_mask_dev, branch.audio_cond_idx_dev),
    )
    for mask, index in cases:
        rows = torch.randn(mask.shape[0], 48, device=device)
        values = torch.randn(int(mask.sum()), 48, device=device)
        expected, actual = rows.clone(), rows.clone()
        expected[mask] = values
        _assign_rows(actual, index, values)
        assert torch.equal(actual, expected)
        assert torch.equal(rows.index_select(0, index), rows[mask])


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("values_shape", [(1, 48), (48,), (59, 48), (60, 47)])
def test_assign_rows_rejects_values_not_sized_to_target_rows(device: str, values_shape: tuple) -> None:
    rows = torch.zeros(100, 48, device=device)
    index = torch.arange(60, device=device)
    with pytest.raises(ValueError, match="target row values have shape"):
        _assign_rows(rows, index, torch.ones(values_shape, device=device))
    assert not rows.any()


@hardware_test(res={"cuda": "L4"})
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_step_kwargs_do_not_synchronize() -> None:
    branch = _branch("cuda")
    branch.x_base = torch.zeros(1, branch.seq_len, 64, device="cuda")
    branch.audio_x_base = torch.zeros(1, branch.seq_len, 16, device="cuda")
    video_rows = torch.randn(branch.img_pos.shape[0], 64, device="cuda")
    audio_rows = torch.randn(branch.audio_pos.shape[0], 16, device="cuda")
    torch.accelerator.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        branch.forward_kwargs(
            video_rows=video_rows,
            audio_rows=audio_rows,
            t_video=0.3,
            t_audio=0.4,
            imgvid_cond_timestep=0.9,
            audio_ref_cond_timestep=0.95,
        )
        target = video_rows.index_select(0, branch.update_idx_dev)
        _assign_rows(video_rows, branch.update_idx_dev, target * 0.5)
    finally:
        torch.cuda.set_sync_debug_mode("default")


@hardware_test(res={"cuda": "L4"})
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_sampler_adapter_row_updates_do_not_synchronize() -> None:
    branch = _branch("cuda")
    video_rows = torch.randn(branch.img_pos.shape[0], 64, device="cuda")
    audio_rows = torch.randn(branch.audio_pos.shape[0], 16, device="cuda")
    adapter = MiniMaxH3Model(None, branch, (video_rows, audio_rows))
    targets = (
        torch.randn(branch.update_idx_dev.shape[0], 64, device="cuda"),
        torch.randn(branch.audio_update_idx_dev.shape[0], 16, device="cuda"),
    )
    velocities = (torch.randn_like(video_rows), torch.randn_like(audio_rows))
    expected_rows = [row.clone() for row in adapter.model_rows]
    for row, mask, target in zip(expected_rows, adapter.masks, targets, strict=True):
        row[mask] = target
    adapter.indices
    torch.accelerator.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        rows = adapter.rows_for(targets)
        adapter.write_back(SamplingState(targets, ((1.0, 0.0), (1.0, 0.0))))
    finally:
        torch.cuda.set_sync_debug_mode("default")
    denoised = adapter.to_denoised((rows, [(row, None) for row in rows], (0.3, 0.4), None), velocities)
    assert all(torch.equal(a, b) for a, b in zip(rows, expected_rows, strict=True))
    for row, mask, v, t, x0 in zip(rows, adapter.masks, velocities, (0.3, 0.4), denoised, strict=True):
        assert torch.equal(x0, minimax_h3_rf_v_to_x0(row[mask], v.float()[mask], row.new_tensor(t)))
