# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax H3 cfg-distilled full denoise loop.

The sampler owns model evaluations and numerical integration. Video and audio
target rows chain through the selected solver while visual and audio condition
rows stay pinned to their noised step-0 anchors.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch

from vllm_omni.diffusion.attention.backends.abstract import VideoTokenLayout, VideoTokenSpan
from vllm_omni.diffusion.cancellation import check_request_cancellation
from vllm_omni.diffusion.forward_context import (
    set_forward_context_denoise_step_idx,
    set_forward_context_denoise_timestep,
    set_forward_context_denoise_total_steps,
)
from vllm_omni.diffusion.sampler import SamplingState, sampler_object
from vllm_omni.platforms import current_omni_platform

from .latent_mask import MiniMaxH3LatentEdit
from .sampling import MiniMaxH3Model, MiniMaxH3ModelBatch

MINIMAX_H3_IMGVID_COND_TIMESTEP = 0.999
# ref2va audio reference anchor timestep
MINIMAX_H3_AUDIO_REF_COND_TIMESTEP = 1.0
# Packed row widths: video rows are [1,2,2]-patchified 24-channel latents
# (24 * 1 * 2 * 2 = 96); audio rows carry the 32-dim audio latent.
MINIMAX_H3_VIDEO_ROW_WIDTH = 96
MINIMAX_H3_AUDIO_ROW_WIDTH = 32


def minimax_h3_publish_denoise_progress(
    step: int | None, sigma_video: float | None, total_steps: int | None = None
) -> None:
    """Publish denoise progress for step-gated attention features.

    Both execution modes must publish the same trio: the step index drives the
    dense warmup of RAINFUSION_ATTN, the normalized descending timestep
    drives the TRTLLM_ATTN skip gate (which stays dense while it is unset),
    and the total step count enables the ``end_step`` tail fallback.
    """
    set_forward_context_denoise_step_idx(step)
    set_forward_context_denoise_timestep(sigma_video)
    set_forward_context_denoise_total_steps(total_steps)


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
        self.locked_audio_rows: torch.Tensor | None = None
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
        raw_spans = packed.get("video_spans")
        if raw_spans:
            spans = tuple(
                VideoTokenSpan(
                    start=int(span["start"]),
                    latent_grid=tuple(int(dim) for dim in span["latent_grid"]),
                    role=str(span["role"]),
                )
                for span in raw_spans
            )
            target_start = next(span.start for span in reversed(spans) if span.role == "target")
            prefix_tags = token_tags[:target_start]
            prefix_segments: list[int] = []
            if prefix_tags.numel():
                # Run-lengths preserve modality/reference boundaries without
                # carrying CUDA tensors into every attention layer.
                boundaries = torch.nonzero(prefix_tags[1:] != prefix_tags[:-1]).flatten().tolist()
                starts = [0, *(index + 1 for index in boundaries)]
                stops = [*(index + 1 for index in boundaries), target_start]
                prefix_segments = [stop - start for start, stop in zip(starts, stops, strict=True)]
            self.static_kwargs["video_token_layout"] = VideoTokenLayout(
                used_len=int(cu[1]),
                video_spans=spans,
            )
            # FastH3-specific prefix geometry belongs to MiniMax-H3's packed
            # attention contract, not the shared VideoTokenLayout interface.
            self.static_kwargs["packed_seq_params"]["vsa_prefix_segments"] = tuple(prefix_segments)
        else:
            grid = packed["latent_grid"].tolist()
            self.static_kwargs["video_token_layout"] = VideoTokenLayout(
                prefix_len=int(packed["video_row_start"]),
                latent_grid=(int(grid[0]), int(grid[1]), int(grid[2])),
            )

    def prepare_rope_table(self, model: Any) -> None:
        """Materialize the branch-local DiT RoPE table once per denoise run.

        ``img_position_ids`` is immutable for this branch while latents and
        timesteps change every scheduler step. Keeping the table in
        ``static_kwargs`` makes every model call reuse the exact same BF16
        tensor without extending its lifetime beyond this request branch.
        """
        prepare = getattr(model, "prepare_rope_table", None)
        if not callable(prepare):
            return
        self.static_kwargs["rope_table"] = prepare(
            self.static_kwargs["img_position_ids"],
            seq_len=self.seq_len,
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
        video_target_timesteps: torch.Tensor | None = None,
        audio_target_timesteps: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        x = self.x_base.clone()
        x[0].index_copy_(0, self.img_pos_dev, video_rows)
        audio_x = self.audio_x_base.clone()
        audio_x[0].index_copy_(0, self.audio_pos_dev, audio_rows)
        timesteps = torch.empty(self.seq_len, dtype=torch.float32, device=self.device)
        self.fill_timesteps(
            timesteps,
            t_video=t_video,
            t_audio=t_audio,
            imgvid_cond_timestep=imgvid_cond_timestep,
            audio_ref_cond_timestep=audio_ref_cond_timestep,
            video_target_timesteps=video_target_timesteps,
            audio_target_timesteps=audio_target_timesteps,
        )
        unique_timesteps, inverse_indices = torch.unique(timesteps, sorted=True, return_inverse=True)
        return {
            **self.static_kwargs,
            "x": x,
            "audio_x": audio_x,
            "unique_timesteps": unique_timesteps,
            "inverse_indices": inverse_indices,
        }

    def fill_timesteps(
        self,
        timesteps: torch.Tensor,
        *,
        t_video: float,
        t_audio: float,
        imgvid_cond_timestep: float,
        audio_ref_cond_timestep: float,
        video_target_timesteps: torch.Tensor | None = None,
        audio_target_timesteps: torch.Tensor | None = None,
    ) -> None:
        """Fill one request's packed row timesteps in an existing tensor."""
        if timesteps.shape != (self.seq_len,):
            raise ValueError(f"timesteps must have shape ({self.seq_len},)")
        # Packed-sequence timestep semantics: non-media rows (text and
        # padding) inherit the current video timestep. Later steps must reuse
        # the previous step's updated rows; re-initializing from zeros is only
        # valid at step 0.
        timesteps.fill_(float(t_video))
        timestep_groups = (
            (
                "video_target_timesteps",
                self.img_pos_dev,
                self.update_mask_dev,
                video_target_timesteps,
                t_video,
                imgvid_cond_timestep,
            ),
            (
                "audio_target_timesteps",
                self.audio_pos_dev,
                self.audio_update_mask_dev,
                audio_target_timesteps,
                t_audio if self.locked_audio_rows is None else 1.0,
                audio_ref_cond_timestep,
            ),
        )
        for name, positions, update_mask, target_timesteps, current_timestep, condition_timestep in timestep_groups:
            target_positions = positions[update_mask]
            if target_timesteps is None:
                timesteps[target_positions] = current_timestep
            else:
                target_timesteps = target_timesteps.to(device=self.device, dtype=torch.float32).reshape(-1)
                if target_timesteps.shape[0] != target_positions.shape[0]:
                    raise ValueError(
                        f"{name} rows {target_timesteps.shape[0]} != target rows {target_positions.shape[0]}"
                    )
                timesteps[target_positions] = target_timesteps
            timesteps[positions[~update_mask]] = condition_timestep


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
    video_edit: MiniMaxH3LatentEdit | None = None,
    audio_edit: MiniMaxH3LatentEdit | None = None,
    sigmas_video: list[float],
    sigmas_audio: list[float],
    device: torch.device,
    imgvid_cond_noise_aug_for_inference: float = MINIMAX_H3_IMGVID_COND_TIMESTEP,
    audio_cond_noise_aug_for_inference: float = MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    update_progress: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
    step_profiler: Callable[[int], AbstractContextManager] | None = None,
    sampler: str = "euler",
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
    # Keep CUDA/CPU on the reference path: its accuracy CI compares the
    # generated video against the official artifact. The table reuse is an
    # Ascend-specific performance optimization and must not alter that path.
    if current_omni_platform.is_npu():
        positive.prepare_rope_table(model)
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
    if video_edit is not None:
        video_edit = video_edit.to(device=device, dtype=torch.float32)
    if audio_edit is not None:
        audio_edit = audio_edit.to(device=device, dtype=torch.float32)

    adapter = MiniMaxH3Model(
        model,
        positive,
        (video_rows, audio_rows),
        (video_edit, audio_edit),
        (imgvid_cond_noise_aug_for_inference, audio_cond_noise_aug_for_inference),
    )
    state = SamplingState((video_rows[update], audio_rows[audio_update]), (sigmas_video, sigmas_audio))

    def write_back_and_update_progress(state):
        nonlocal video_rows, audio_rows
        adapter.validate_write_back(state)
        video_rows, audio_rows = adapter.write_back(state)
        if update_progress is not None:
            update_progress(state.step_index - 1, video_rows, audio_rows)
        check_request_cancellation(synchronize=True)

    def step_context(states):
        check_request_cancellation()
        return step_profiler(states[0].step_index) if step_profiler is not None else nullcontext()

    try:
        with torch.inference_mode():
            sampler_object(sampler).sample(
                MiniMaxH3ModelBatch({state: adapter}),
                [state],
                callback=write_back_and_update_progress,
                step_context=step_context,
            )
    finally:
        minimax_h3_publish_denoise_progress(None, None, None)
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
