# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DreamZero pipeline persistent state."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Number of frames per chunk for subsequent calls (first call uses 1)
FRAMES_PER_CHUNK = 4


@dataclass
class DreamZeroStageCarrier:
    """DreamZero-private inter-phase carrier for disaggregated execution (RFC #4590).

    ``forward()`` is refactored into three phase methods
    (``_run_encode_phase`` / ``_run_denoise_phase`` / ``_run_decode_phase``) that
    read and write this carrier. In the monolithic path all three run in one
    process on one carrier, so the composition is byte-identical to the original
    forward. In the disaggregated path each phase runs in its own worker and the
    carrier's stable fields are marshalled through a ``StagePayload`` by the
    pipeline's ``pack_stage_state`` / ``unpack_stage_state`` hooks.

    This is model-private state (kept out of the generic ``DiffusionRequestState``
    per RFC §2.3): it is stored on ``DiffusionRequestState.extra`` and interpreted
    only by DreamZero. The AR-Diffusion KV / live session objects are NEVER placed
    here — they stay on the denoise worker.
    """

    # -- identity / control (crosses every boundary) --
    session_id: str = "default"
    embodiment_name: str = ""
    # The exact key passed to get_transform() at encode time, so decode selects
    # the identical transform (preserves monolithic-forward behavior exactly).
    transform_embodiment: str = ""
    # Reset decision computed at encode from the encode-session state; the
    # denoise worker applies the SAME reset to its engine KV window.
    reset_reason: str | None = None
    explicit_reset: bool = False
    do_true_cfg: bool = False
    # Window position at the start of this forward. Both encode and denoise track
    # csf on their own DreamZeroState; it is carried so the denoise worker aligns
    # its KV window with the conditions the encode worker produced.
    current_start_frame: int = 0

    # -- geometry (encode -> denoise) --
    height: int = 0
    width: int = 0
    seq_len: int = 0
    frame_seqlen: int = 0

    # -- encoded conditions (encode -> denoise), stable/read-only during denoise --
    prompt_embeds: torch.Tensor | None = None
    negative_prompt_embeds: torch.Tensor | None = None
    clip_feas: torch.Tensor | None = None
    ys: torch.Tensor | None = None
    image_latent: torch.Tensor | None = None
    state_features: torch.Tensor | None = None
    embodiment_id: torch.Tensor | None = None

    # -- initial noise (encode -> denoise). Carried explicitly so the denoise
    #    worker uses exactly the encode-seeded noise (deterministic parity). --
    noise_obs: torch.Tensor | None = None
    noise_action: torch.Tensor | None = None

    # -- timestep schedule params (encode -> denoise) --
    num_inference_steps: int = 0
    sigma_shift: float = 1.0

    # -- postprocess inputs (encode -> ... -> decode) --
    # Raw padded observation state for relative->absolute action recovery.
    state_for_postprocess: torch.Tensor | None = None

    # -- denoise outputs (denoise -> decode) --
    video_out: torch.Tensor | None = None
    action_out: torch.Tensor | None = None

    # -- opaque extras for forward-compat --
    extra: dict[str, Any] = field(default_factory=dict)


class DreamZeroState:
    """Pipeline persistent state across forward() calls.

    Lifecycle:
        - Created once in DreamZeroPipeline.__init__()
        - Mutated every forward() call (frame append, KV cache grow)
        - reset() on new session / language change
        - reset(clear_video_latents=False) on local_attn_size exceeded (KV only)
    """

    def __init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------
    # Frame accumulation (single stitched buffer)
    # Transform outputs stitched single frame per call.
    # We accumulate here to build multi-frame video for AR inference.
    # ------------------------------------------------------------------

    def accumulate_frames(self, stitched: np.ndarray) -> np.ndarray:
        """Accumulate stitched frames and return multi-frame video.

        Args:
            stitched: (H, W, C) single frame or (T, H, W, C) multi-frame,
                      already stitched by transform.

        Returns:
            (T, H, W, C) ndarray. T=1 for first call, T=FRAMES_PER_CHUNK(4) after.
        """
        if stitched.ndim == 3:
            self.stitched_buffer.append(stitched)
        elif stitched.ndim == 4:
            self.stitched_buffer.extend(list(stitched))
        else:
            raise ValueError(f"Expected 3D or 4D stitched, got {stitched.ndim}D")

        num_frames = 1 if self.call_count == 0 else FRAMES_PER_CHUNK

        buffer_frames = list(self.stitched_buffer)
        if len(buffer_frames) >= num_frames:
            frames = buffer_frames[-num_frames:]
        else:
            frames = buffer_frames
            while len(frames) < num_frames:
                frames.insert(0, buffer_frames[0])

        self.call_count += 1
        return np.stack(frames, axis=0)  # (T, H, W, C)

    def append_video_latents(self, video_out: torch.Tensor) -> None:
        """Append one AR chunk of normalized video latents for later decode.

        Args:
            video_out: ``(B, T, C, H, W)`` tensor returned by the denoise loop
                (before the ``transpose(1, 2)`` stored in ``DiffusionOutput``).
        """
        if video_out.dim() != 5:
            raise ValueError(f"Expected 5D video_out, got shape {tuple(video_out.shape)}")
        # Upstream ``torch.cat(self.video_across_time, dim=2)`` uses (B, C, T, H, W).
        chunk = video_out.transpose(1, 2).detach().cpu()
        self.video_latents_across_time.append(chunk)
        logger.info(
            "append_video_latents: chunk_shape=%s total_chunks=%d total_latent_t=%d",
            tuple(chunk.shape),
            len(self.video_latents_across_time),
            int(sum(c.shape[2] for c in self.video_latents_across_time)),
        )

    def get_concatenated_video_latents(self) -> torch.Tensor | None:
        """Return all accumulated chunks concatenated along the time dimension."""
        if not self.video_latents_across_time:
            return None
        if len(self.video_latents_across_time) == 1:
            return self.video_latents_across_time[0]
        return torch.cat(self.video_latents_across_time, dim=2)

    def clear_video_latents(self) -> None:
        """Drop accumulated video latents without resetting KV/frame state."""
        self.video_latents_across_time = []

    # ------------------------------------------------------------------
    # Reset / reset_reason
    # ------------------------------------------------------------------

    def reset(self, *, clear_video_latents: bool = True) -> None:
        """Clear session state.

        Args:
            clear_video_latents: When ``False``, keep ``video_latents_across_time``
                so offline/online export can decode the full AR rollout.
        """
        saved_video_latents = [] if clear_video_latents else list(self.video_latents_across_time)
        self.stitched_buffer = deque(maxlen=FRAMES_PER_CHUNK)
        self.call_count = 0
        # Normalized VAE latents per AR chunk, each (B, C, T, H, W) on CPU.
        # Concatenated along T before decode (matches upstream video_across_time).
        self.video_latents_across_time = saved_video_latents

        self.current_start_frame = 0

        self.clip_feas = None
        self.ys = None
        if clear_video_latents:
            # Session reset: drop the prompt-embed cache and VAE encoder stream.
            self.language = None
            self.prompt_embeds = None
            self.reset_vae_encoder_stream()
        # Window ("inference") resets keep both: the prompt is unchanged, and the
        # Wan feat_cache history is independent of the DiT attention window.

    def reset_vae_encoder_stream(self) -> None:
        """Clear incremental Wan VAE encoder state used across AR steps."""
        self.vae_stream_initialized = False
        self.vae_enc_feat_map: list[torch.Tensor | None] | None = None
        self.vae_encoder_out: torch.Tensor | None = None
        self.vae_pending_body_frames: torch.Tensor | None = None

    def reset_reason(
        self,
        text_tokens: torch.Tensor | None,
        num_video_frames: int,
        local_attn_size: int,
    ) -> str | None:
        """Return why state should reset before the next forward(), if any."""
        if self.language is None:
            logger.info("language is None, resetting")
            return "session"

        if text_tokens is not None and not torch.equal(self.language, text_tokens):
            logger.info("language changed, resetting")
            return "session"

        # NOTE: after accumulate_frames, num_video_frames is the accumulated T
        # (1 for first call, 4 for subsequent). Only reset on true single-frame
        # which happens when the stitched_buffer was cleared externally.
        if num_video_frames == 1 and self.call_count > 1:
            logger.info("single frame input after first call, resetting")
            return "session"

        if local_attn_size != -1 and self.current_start_frame >= local_attn_size:
            logger.info(
                "current_start_frame %d >= local_attn_size %d, resetting inference state",
                self.current_start_frame,
                local_attn_size,
            )
            return "inference"

        return None
