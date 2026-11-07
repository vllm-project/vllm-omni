# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import MiniMaxH3DenoiseBranch


def student_forward_kwargs(
    branch: "MiniMaxH3DenoiseBranch",
    *,
    video_rows: torch.Tensor,
    audio_rows: torch.Tensor,
    t_video: float,
    t_audio: float,
) -> dict[str, Any]:
    """Host-side ``forward_kwargs`` for the student and clean-commit forwards.

    Matches ``MiniMaxH3DenoiseBranch.forward_kwargs`` only under the TaoMate
    call sites' conditioning: ``imgvid_cond_timestep == t_video`` and
    ``audio_ref_cond_timestep == 1.0``, with no per-row
    ``video_target_timesteps``/``audio_target_timesteps``. Under that invariant
    every video-condition row already holds ``t_video`` — the same value the
    non-media rows get — so ``fill_timesteps`` reduces to the audio-row writes
    below. The sorted-unique timesteps and inverse indices are computed on the
    host and uploaded, keeping the device free of the ``torch.unique``. A
    caller that needs different conditioning timesteps or per-row targets must
    call ``branch.forward_kwargs`` directly.
    """
    # Layout and masks are already on the CPU; keep the native float32 unique cardinality and ordering.
    timesteps = torch.full((branch.seq_len,), float(t_video), dtype=torch.float32, device="cpu")
    audio_pos = branch.audio_pos
    audio_update = branch.audio_update_mask
    timesteps[audio_pos[audio_update]] = float(t_audio) if branch.locked_audio_rows is None else 1.0
    timesteps[audio_pos[~audio_update]] = 1.0
    unique, inverse = torch.unique(timesteps, sorted=True, return_inverse=True)
    x = branch.x_base.clone()
    x[0].index_copy_(0, branch.img_pos_dev, video_rows)
    audio_x = branch.audio_x_base.clone()
    audio_x[0].index_copy_(0, branch.audio_pos_dev, audio_rows)
    return {
        **branch.static_kwargs,
        "x": x,
        "audio_x": audio_x,
        "unique_timesteps": unique.to(branch.device, non_blocking=True),
        "inverse_indices": inverse.to(branch.device, non_blocking=True),
    }
