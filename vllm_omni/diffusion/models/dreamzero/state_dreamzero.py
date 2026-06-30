# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DreamZero pipeline persistent state."""

from __future__ import annotations

import logging
from collections import deque

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Number of frames per chunk for subsequent calls (first call uses 1)
FRAMES_PER_CHUNK = 4


class DreamZeroState:
    """Pipeline persistent state across forward() calls.

    Lifecycle:
        - Created once in DreamZeroPipeline.__init__()
        - Mutated every forward() call (frame append, KV cache grow)
        - reset() on new session / language change
        - reset_inference_state() on local_attn_size exceeded (KV only)
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
    # Reset / should_reset
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

        # KV cache once robot-policy diffusion supports that integration.
        self.kv_cache = None
        self.kv_cache_neg = None
        self.crossattn_cache = None
        self.crossattn_cache_neg = None
        self.current_start_frame = 0

        self.clip_feas = None
        self.ys = None
        if clear_video_latents:
            self.language = None

    def reset_inference_state(self) -> None:
        """Reset KV/frame state after local attention rolls without dropping video latents."""
        self.reset(clear_video_latents=False)

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

    def should_reset(self, text_tokens: torch.Tensor | None, num_video_frames: int, local_attn_size: int) -> bool:
        """Determine if state should be reset before this forward()."""
        return self.reset_reason(text_tokens, num_video_frames, local_attn_size) is not None

    # ------------------------------------------------------------------
    # KV cache management
    # ------------------------------------------------------------------

    def create_kv_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        num_layers: int,
        num_heads: int,
        head_dim: int,
    ) -> None:
        """Initialize empty KV caches and cross-attention caches."""
        self.kv_cache = [
            torch.zeros(2, batch_size, 0, num_heads, head_dim, dtype=dtype, device=device) for _ in range(num_layers)
        ]
        self.kv_cache_neg = [
            torch.zeros(2, batch_size, 0, num_heads, head_dim, dtype=dtype, device=device) for _ in range(num_layers)
        ]

        self.crossattn_cache = [
            {"is_init": False, "k": None, "v": None, "k_img": None, "v_img": None} for _ in range(num_layers)
        ]
        self.crossattn_cache_neg = [
            {"is_init": False, "k": None, "v": None, "k_img": None, "v_img": None} for _ in range(num_layers)
        ]

    def update_kv_cache(
        self,
        layer_index: int,
        updated_kv: torch.Tensor,
        is_negative: bool = False,
    ) -> None:
        """Update a single layer's KV cache after prefill."""
        cache = self.kv_cache_neg if is_negative else self.kv_cache
        if cache is None:
            raise RuntimeError("KV caches not initialized, call create_kv_caches first.")
        cache[layer_index] = updated_kv.clone()

    def get_kv_caches(self, is_negative: bool = False) -> list[torch.Tensor]:
        """Get KV caches for the specified branch."""
        cache = self.kv_cache_neg if is_negative else self.kv_cache
        if cache is None:
            raise RuntimeError("KV caches not initialized.")
        return cache

    def get_crossattn_caches(self, is_negative: bool = False) -> list[dict[str, bool | torch.Tensor | None]]:
        """Get cross-attention caches for the specified branch."""
        cache = self.crossattn_cache_neg if is_negative else self.crossattn_cache
        if cache is None:
            raise RuntimeError("Cross-attn caches not initialized.")
        return cache
