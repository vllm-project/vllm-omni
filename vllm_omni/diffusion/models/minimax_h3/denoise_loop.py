# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 cfg-distilled full denoise loop.

Per step, the positive presentation is forwarded exactly once. Video and audio
target rows chain through the Euler-eta0 update while visual and audio condition
rows stay pinned to their noised step-0 anchors.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch

from vllm_omni.diffusion.attention.backends.abstract import VideoTokenLayout
from vllm_omni.diffusion.forward_context import (
    set_forward_context_denoise_step_idx,
    set_forward_context_denoise_timestep,
)

from .scheduling_minimax_h3_euler_ancestral import (
    minimax_h3_euler_eta0_step,
    minimax_h3_rf_v_to_x0,
)

MINIMAX_H3_IMGVID_COND_TIMESTEP = 0.999
# ref2va audio reference anchor timestep
MINIMAX_H3_AUDIO_REF_COND_TIMESTEP = 1.0
# Packed row widths: video rows are [1,2,2]-patchified 24-channel latents
# (24 * 1 * 2 * 2 = 96); audio rows carry the 32-dim audio latent.
MINIMAX_H3_VIDEO_ROW_WIDTH = 96
MINIMAX_H3_AUDIO_ROW_WIDTH = 32


def minimax_h3_publish_denoise_progress(step: int | None, sigma_video: float | None) -> None:
    """Publish denoise progress for step-gated attention features.

    Both execution modes must publish the same pair: the step index drives the
    dense warmup of RAINFUSION_ATTN, and the normalized descending timestep
    drives the TRTLLM_ATTN skip gate, which stays dense while it is unset.
    """
    set_forward_context_denoise_step_idx(step)
    set_forward_context_denoise_timestep(sigma_video)


class MiniMaxH3DenoiseBranch:
    """Static per-branch state: packed layout + fixed forward kwargs.

    `packed` is a minimax_h3_packed_sequence(...) result (or equivalent layout
    dict); `text_embeddings` is the branch's [text_len, 5120] hidden states;
    `token_tags` must already carry any fl2va vision-span overrides.
    """

    def __init__(
        self,
        *,
        packed: dict[str, torch.Tensor],
        text_embeddings: torch.Tensor,
        token_tags: torch.Tensor,
        device: torch.device,
    ) -> None:
        seq_len = int(packed["seq_len"])
        self.seq_len = seq_len
        self.img_pos = packed["img_pos"].view(-1).to(torch.long)
        self.audio_pos = packed["audio_pos"].view(-1).to(torch.long)
        self.update_mask = packed["update_mask"].view(-1).to(torch.bool)
        # ref2va: audio_pos may include reference-audio anchor rows
        # (audio_update_mask False); absent means all rows are targets.
        if "audio_update_mask" in packed:
            self.audio_update_mask = packed["audio_update_mask"].view(-1).to(torch.bool)
        else:
            self.audio_update_mask = torch.ones(self.audio_pos.shape[0], dtype=torch.bool)
        text_len = int(packed["text_pos"].view(-1).shape[0])
        if list(text_embeddings.shape)[0] != text_len:
            raise ValueError(f"text_embeddings rows {list(text_embeddings.shape)} != packed text_len {text_len}")
        if int(token_tags.view(-1).shape[0]) != seq_len:
            raise ValueError(f"token_tags length {int(token_tags.view(-1).shape[0])} != seq_len {seq_len}")
        cu = packed["cu_seqlens"].to(torch.int32)
        self.device = device
        # ``used_len`` is the real (non-padding) document length; step-mode
        # batching concatenates layouts and needs both bounds as Python ints so
        # it can rebuild ``cu_seqlens`` without a device sync per step.
        self.used_len = int(cu[1])
        self.text_len = text_len
        self.img_pos_dev = self.img_pos.to(device)
        self.audio_pos_dev = self.audio_pos.to(device)
        self.update_mask_dev = self.update_mask.to(device)
        self.audio_update_mask_dev = self.audio_update_mask.to(device)
        self.x_base = torch.zeros(1, seq_len, MINIMAX_H3_VIDEO_ROW_WIDTH, dtype=torch.float32, device=device)
        self.audio_x_base = torch.zeros(1, seq_len, MINIMAX_H3_AUDIO_ROW_WIDTH, dtype=torch.float32, device=device)
        self.text_pos_dev = packed["text_pos"].view(-1).to(torch.long).to(device)
        self.token_tags_dev = token_tags.view(-1).to(torch.long).to(device)
        self.img_position_ids_dev = packed["img_position_ids"].to(device)
        self.text_embeddings_dev = text_embeddings.to(device)
        self.static_kwargs: dict[str, Any] = {
            "img_position_ids": self.img_position_ids_dev[None],
            "update_mask": self.update_mask_dev,
            "token_tags": self.token_tags_dev,
            "skip_mask_out_condition": False,
            "prompt_embeds": self.text_embeddings_dev,
            "img_pos_info": {"position_ids": self.img_pos_dev},
            "audio_pos_info": {"position_ids": self.audio_pos_dev},
            "text_pos_info": {"position_ids": self.text_pos_dev},
            "img_pos_for_infer_output_info": {"position_ids": self.img_pos_dev},
            "packed_seq_params": {
                "cu_seqlens_q": cu.to(device),
                "max_seqlen_q": self.used_len,
                # One request: valid rows are a prefix, so attention may use a
                # KV prefix length or a 1-D pad mask. See batched_packing for the
                # co-batched layout, where neither can describe the valid rows.
                "num_requests": 1,
            },
            "refiner_packed_seq_params": {
                "cu_seqlens_q": torch.tensor([0, text_len, text_len], dtype=torch.int32, device=device),
                "max_seqlen_q": text_len,
            },
        }
        # Where the video segment sits in the packed sequence. Resolved to plain
        # ints here so the attention layers never sync on it per step.
        grid = packed["latent_grid"].tolist()
        self.static_kwargs["video_token_layout"] = VideoTokenLayout(
            prefix_len=int(packed["video_row_start"]),
            latent_grid=(int(grid[0]), int(grid[1]), int(grid[2])),
        )

    def forward_kwargs(
        self,
        *,
        video_rows: torch.Tensor,
        audio_rows: torch.Tensor,
        t_video: float,
        t_audio: float,
        imgvid_cond_timestep: float,
        audio_ref_cond_timestep: float,
    ) -> dict[str, Any]:
        x = self.x_base.clone()
        x[0].index_copy_(0, self.img_pos_dev, video_rows)
        audio_x = self.audio_x_base.clone()
        audio_x[0].index_copy_(0, self.audio_pos_dev, audio_rows)
        # Packed-sequence timestep semantics: non-media rows (text and
        # padding) inherit the current video timestep. Later steps must reuse
        # the previous step's updated rows; re-initializing from zeros is only
        # valid at step 0.
        timesteps = torch.full(
            (self.seq_len,),
            float(t_video),
            dtype=torch.float32,
            device=x.device,
        )
        timesteps[self.img_pos_dev[self.update_mask_dev]] = t_video
        timesteps[self.img_pos_dev[~self.update_mask_dev]] = imgvid_cond_timestep
        timesteps[self.audio_pos_dev[self.audio_update_mask_dev]] = t_audio
        timesteps[self.audio_pos_dev[~self.audio_update_mask_dev]] = audio_ref_cond_timestep
        unique_timesteps, inverse_indices = torch.unique(timesteps, sorted=True, return_inverse=True)
        return {
            **self.static_kwargs,
            "x": x,
            "audio_x": audio_x,
            "unique_timesteps": unique_timesteps,
            "inverse_indices": inverse_indices,
        }


def minimax_h3_prepare_denoise_rows(
    *,
    positive: MiniMaxH3DenoiseBranch,
    initial_video_rows: torch.Tensor,
    initial_audio_rows: torch.Tensor,
    keyframe_cond_rows: torch.Tensor | None,
    audio_ref_rows: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Validate the initial rows against the layout and pin the condition rows.

    Returns ``(video_rows, audio_rows, cond_anchor, audio_anchor)`` with the
    anchors already written into their rows, which is the state both the
    request-mode loop and step-mode ``prepare_encode()`` start from.
    """
    n_cond = int((~positive.update_mask).sum())
    if keyframe_cond_rows is None:
        if n_cond != 0:
            raise ValueError(f"layout has {n_cond} cond rows but keyframe_cond_rows is None")
    else:
        if int(keyframe_cond_rows.shape[0]) != n_cond:
            raise ValueError(f"keyframe_cond_rows {int(keyframe_cond_rows.shape[0])} != layout cond rows {n_cond}")
    video_rows = initial_video_rows.to(device=device, dtype=torch.float32).clone()
    audio_rows = initial_audio_rows.to(device=device, dtype=torch.float32).clone()
    update = positive.update_mask_dev
    audio_update = positive.audio_update_mask_dev
    if int(video_rows.shape[0]) != int(positive.img_pos.shape[0]):
        raise ValueError(
            f"initial video rows {int(video_rows.shape[0])} != positive layout rows {int(positive.img_pos.shape[0])}"
        )
    if int(audio_rows.shape[0]) != int(positive.audio_pos.shape[0]):
        raise ValueError(
            f"initial audio rows {int(audio_rows.shape[0])} != positive layout rows {int(positive.audio_pos.shape[0])}"
        )
    n_audio_ref = int((~positive.audio_update_mask).sum())
    if audio_ref_rows is None:
        if n_audio_ref != 0:
            raise ValueError(f"layout has {n_audio_ref} audio ref rows but audio_ref_rows is None")
        audio_anchor = None
    else:
        if int(audio_ref_rows.shape[0]) != n_audio_ref:
            raise ValueError(f"audio_ref_rows {int(audio_ref_rows.shape[0])} != layout audio ref rows {n_audio_ref}")
        audio_anchor = audio_ref_rows.to(device=device, dtype=torch.float32)
    cond_anchor = keyframe_cond_rows.to(device=device, dtype=torch.float32) if keyframe_cond_rows is not None else None
    if cond_anchor is not None:
        video_rows[~update] = cond_anchor
    if audio_anchor is not None:
        audio_rows[~audio_update] = audio_anchor
    return video_rows, audio_rows, cond_anchor, audio_anchor


def minimax_h3_denoise_loop(
    *,
    model: Any,
    positive: MiniMaxH3DenoiseBranch,
    initial_video_rows: torch.Tensor,
    initial_audio_rows: torch.Tensor,
    keyframe_cond_rows: torch.Tensor | None,
    audio_ref_rows: torch.Tensor | None = None,
    sigmas_video: list[float],
    sigmas_audio: list[float],
    device: torch.device,
    imgvid_cond_noise_aug_for_inference: float = MINIMAX_H3_IMGVID_COND_TIMESTEP,
    audio_cond_noise_aug_for_inference: float = MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    on_step: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
    step_profiler: Callable[[int], AbstractContextManager] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full denoise loop; returns final (video_rows, audio_rows).

    ``initial_video_rows`` covers all image rows of the positive layout. For a
    conditional task, pass ``keyframe_cond_rows`` and/or ``audio_ref_rows`` to
    pin those rows across every step. The model's raw positive velocity is the
    update signal; MiniMax H3 only supports cfg-distilled checkpoints.
    """
    if len(sigmas_video) != len(sigmas_audio):
        raise ValueError("video/audio sigma schedules must have equal length")
    if len(sigmas_video) < 2:
        raise ValueError("sigma schedules need at least 2 entries")
    video_rows, audio_rows, cond_anchor, audio_anchor = minimax_h3_prepare_denoise_rows(
        positive=positive,
        initial_video_rows=initial_video_rows,
        initial_audio_rows=initial_audio_rows,
        keyframe_cond_rows=keyframe_cond_rows,
        audio_ref_rows=audio_ref_rows,
        device=device,
    )
    update = positive.update_mask_dev
    audio_update = positive.audio_update_mask_dev

    num_steps = len(sigmas_video) - 1
    for step in range(num_steps):
        step_cm = step_profiler(step) if step_profiler is not None else nullcontext()
        with step_cm:
            s_v, s_v_next = sigmas_video[step], sigmas_video[step + 1]
            s_a, s_a_next = sigmas_audio[step], sigmas_audio[step + 1]
            # Publish where we are so step-gated attention features (the dense
            # warmup of RAINFUSION_ATTN, the timestep gate of TRTLLM_ATTN) can
            # see it. Gates use the scheduler-style descending timestep, which
            # for a rectified-flow schedule is the video sigma.
            minimax_h3_publish_denoise_progress(step, s_v)
            t_v, t_a = 1.0 - s_v, 1.0 - s_a
            imgvid_cond_t = max(t_v, float(imgvid_cond_noise_aug_for_inference))
            audio_ref_cond_t = max(t_a, float(audio_cond_noise_aug_for_inference))

            fk = positive.forward_kwargs(
                video_rows=video_rows,
                audio_rows=audio_rows,
                t_video=t_v,
                t_audio=t_a,
                imgvid_cond_timestep=imgvid_cond_t,
                audio_ref_cond_timestep=audio_ref_cond_t,
            )
            with torch.inference_mode():
                v_video, v_audio = model(**fk)
            mv_video_t = v_video.float()[update]
            mv_audio_t = v_audio.float()[audio_update]

            x0_video = minimax_h3_rf_v_to_x0(
                video_rows[update],
                mv_video_t,
                torch.tensor(t_v, dtype=torch.float32, device=device),
            )
            new_target = minimax_h3_euler_eta0_step(video_rows[update], x0_video, sigma_curr=s_v, sigma_next=s_v_next)
            video_rows = video_rows.clone()
            video_rows[update] = new_target
            if cond_anchor is not None:
                video_rows[~update] = cond_anchor  # per-step imgvid cond reset

            x0_audio = minimax_h3_rf_v_to_x0(
                audio_rows[audio_update],
                mv_audio_t,
                torch.tensor(t_a, dtype=torch.float32, device=device),
            )
            new_audio = minimax_h3_euler_eta0_step(
                audio_rows[audio_update], x0_audio, sigma_curr=s_a, sigma_next=s_a_next
            )
            audio_rows = audio_rows.clone()
            audio_rows[audio_update] = new_audio
            if audio_anchor is not None:
                audio_rows[~audio_update] = audio_anchor  # per-step audio ref reset
            if on_step is not None:
                on_step(step, video_rows, audio_rows)

    minimax_h3_publish_denoise_progress(None, None)
    return video_rows, audio_rows


__all__ = [
    "MINIMAX_H3_AUDIO_REF_COND_TIMESTEP",
    "MINIMAX_H3_AUDIO_ROW_WIDTH",
    "MINIMAX_H3_IMGVID_COND_TIMESTEP",
    "MINIMAX_H3_VIDEO_ROW_WIDTH",
    "MiniMaxH3DenoiseBranch",
    "minimax_h3_denoise_loop",
    "minimax_h3_prepare_denoise_rows",
    "minimax_h3_publish_denoise_progress",
]
