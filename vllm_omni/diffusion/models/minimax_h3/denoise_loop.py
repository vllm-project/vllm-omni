# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 cfg-distilled full denoise loop.

Per step, the positive presentation is forwarded exactly once. Video and audio
target rows chain through the Euler-eta0 update while visual and audio condition
rows stay pinned to their noised step-0 anchors.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from .adaln_schedule_cache import (
    minimax_h3_adaln_schedule_cache_enabled,
    minimax_h3_adaln_schedule_key,
    minimax_h3_adaln_weight_offload_enabled,
    minimax_h3_float32_bits,
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


@dataclass(frozen=True)
class MiniMaxH3StepTimestepMetadata:
    """Device-ready timestep routing for one denoise forward."""

    unique_timesteps: torch.Tensor
    inverse_indices: torch.Tensor
    timestep_bits: tuple[int, ...]


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
        self.device = device
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
        self.img_pos_dev = self.img_pos.to(device)
        self.audio_pos_dev = self.audio_pos.to(device)
        self.update_mask_dev = self.update_mask.to(device)
        self.audio_update_mask_dev = self.audio_update_mask.to(device)
        timestep_category_ids = torch.zeros(seq_len, dtype=torch.long)
        timestep_category_ids[self.img_pos[~self.update_mask]] = 1
        timestep_category_ids[self.audio_pos[self.audio_update_mask]] = 2
        timestep_category_ids[self.audio_pos[~self.audio_update_mask]] = 3
        self.timestep_category_ids = timestep_category_ids
        self.used_timestep_categories = torch.unique(timestep_category_ids, sorted=True)
        self.x_base = torch.zeros(1, seq_len, MINIMAX_H3_VIDEO_ROW_WIDTH, dtype=torch.float32, device=device)
        self.audio_x_base = torch.zeros(1, seq_len, MINIMAX_H3_AUDIO_ROW_WIDTH, dtype=torch.float32, device=device)
        text_pos_dev = packed["text_pos"].view(-1).to(torch.long).to(device)
        self.static_kwargs: dict[str, Any] = {
            "img_position_ids": packed["img_position_ids"][None].to(device),
            "update_mask": self.update_mask_dev,
            "token_tags": token_tags.view(-1).to(torch.long).to(device),
            "skip_mask_out_condition": False,
            "prompt_embeds": text_embeddings.to(device),
            "img_pos_info": {"position_ids": self.img_pos_dev},
            "audio_pos_info": {"position_ids": self.audio_pos_dev},
            "text_pos_info": {"position_ids": text_pos_dev},
            "img_pos_for_infer_output_info": {"position_ids": self.img_pos_dev},
            "packed_seq_params": {
                "cu_seqlens_q": cu.to(device),
                "max_seqlen_q": int(cu[1]),
            },
            "refiner_packed_seq_params": {
                "cu_seqlens_q": torch.tensor([0, text_len, text_len], dtype=torch.int32, device=device),
                "max_seqlen_q": text_len,
            },
        }
        self._adaln_schedule_cache_active = False

    def prepare_timestep_metadata(
        self,
        *,
        sigmas_video: list[float],
        sigmas_audio: list[float],
        imgvid_cond_noise_aug_for_inference: float,
        audio_cond_noise_aug_for_inference: float,
    ) -> list[MiniMaxH3StepTimestepMetadata]:
        """Precompute the full schedule's packed-row timestep routing on CPU."""
        if len(sigmas_video) != len(sigmas_audio):
            raise ValueError("video/audio sigma schedules must have equal length")
        if len(sigmas_video) < 2:
            raise ValueError("sigma schedules need at least 2 entries")

        result: list[MiniMaxH3StepTimestepMetadata] = []
        for step in range(len(sigmas_video) - 1):
            t_video = 1.0 - float(sigmas_video[step])
            t_audio = 1.0 - float(sigmas_audio[step])
            category_timesteps = torch.tensor(
                [
                    t_video,
                    max(t_video, float(imgvid_cond_noise_aug_for_inference)),
                    t_audio,
                    max(t_audio, float(audio_cond_noise_aug_for_inference)),
                ],
                dtype=torch.float32,
            )
            active_timesteps = category_timesteps.index_select(0, self.used_timestep_categories)
            unique_timesteps, active_inverse = torch.unique(
                active_timesteps,
                sorted=True,
                return_inverse=True,
            )
            category_inverse = torch.zeros(4, dtype=torch.long)
            category_inverse.index_copy_(0, self.used_timestep_categories, active_inverse)
            inverse_indices = category_inverse.index_select(0, self.timestep_category_ids)
            result.append(
                MiniMaxH3StepTimestepMetadata(
                    unique_timesteps=unique_timesteps.to(self.device),
                    inverse_indices=inverse_indices.to(self.device),
                    timestep_bits=minimax_h3_float32_bits(unique_timesteps),
                )
            )
        return result

    def prepare_adaln_schedule_cache(
        self,
        model: Any,
        timestep_metadata: list[MiniMaxH3StepTimestepMetadata],
        *,
        enabled: bool,
    ) -> Any | None:
        """Prepare exact model-local modulation tables for this schedule."""
        self._adaln_schedule_cache_active = False
        if not enabled:
            clear = getattr(model, "clear_adaln_schedule_cache", None)
            if clear is not None:
                clear()
            return None
        prepare = getattr(model, "prepare_adaln_schedule_cache", None)
        if prepare is None:
            return None
        with torch.inference_mode():
            cache = prepare(
                unique_timestep_plan=tuple(step.unique_timesteps for step in timestep_metadata),
                schedule_key=minimax_h3_adaln_schedule_key(timestep_metadata),
                offload_weights=minimax_h3_adaln_weight_offload_enabled(),
            )
        self._adaln_schedule_cache_active = cache is not None
        return cache

    def forward_kwargs(
        self,
        *,
        video_rows: torch.Tensor,
        audio_rows: torch.Tensor,
        timestep_metadata: MiniMaxH3StepTimestepMetadata,
        step_index: int | None = None,
    ) -> dict[str, Any]:
        x = self.x_base.clone()
        x[0].index_copy_(0, self.img_pos_dev, video_rows)
        audio_x = self.audio_x_base.clone()
        audio_x[0].index_copy_(0, self.audio_pos_dev, audio_rows)
        dynamic_kwargs = {
            **self.static_kwargs,
            "x": x,
            "audio_x": audio_x,
            "unique_timesteps": timestep_metadata.unique_timesteps,
            "inverse_indices": timestep_metadata.inverse_indices,
        }
        if self._adaln_schedule_cache_active:
            if step_index is None:
                raise ValueError("step_index is required while the AdaLN schedule cache is active")
            dynamic_kwargs["adaln_step_index"] = step_index
        return dynamic_kwargs


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
    timestep_metadata = positive.prepare_timestep_metadata(
        sigmas_video=sigmas_video,
        sigmas_audio=sigmas_audio,
        imgvid_cond_noise_aug_for_inference=imgvid_cond_noise_aug_for_inference,
        audio_cond_noise_aug_for_inference=audio_cond_noise_aug_for_inference,
    )
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

    prepare_for_request_caches = getattr(model, "prepare_for_request_caches", None)
    if prepare_for_request_caches is not None:
        with torch.inference_mode():
            prepare_for_request_caches()
    positive.prepare_adaln_schedule_cache(
        model,
        timestep_metadata,
        enabled=minimax_h3_adaln_schedule_cache_enabled(),
    )

    num_steps = len(sigmas_video) - 1
    for step in range(num_steps):
        step_cm = step_profiler(step) if step_profiler is not None else nullcontext()
        with step_cm:
            s_v, s_v_next = sigmas_video[step], sigmas_video[step + 1]
            s_a, s_a_next = sigmas_audio[step], sigmas_audio[step + 1]
            t_v, t_a = 1.0 - s_v, 1.0 - s_a

            fk = positive.forward_kwargs(
                video_rows=video_rows,
                audio_rows=audio_rows,
                timestep_metadata=timestep_metadata[step],
                step_index=step,
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

    return video_rows, audio_rows


__all__ = [
    "MINIMAX_H3_AUDIO_REF_COND_TIMESTEP",
    "MINIMAX_H3_AUDIO_ROW_WIDTH",
    "MINIMAX_H3_IMGVID_COND_TIMESTEP",
    "MINIMAX_H3_VIDEO_ROW_WIDTH",
    "MiniMaxH3DenoiseBranch",
    "MiniMaxH3StepTimestepMetadata",
    "minimax_h3_denoise_loop",
]
