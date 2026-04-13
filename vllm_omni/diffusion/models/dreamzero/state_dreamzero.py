# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DreamZero pipeline persistent state.

Consolidates all cross-forward() state that was originally scattered across:
- ARDroidRoboarenaPolicy._frame_buffers   (socket_test_optimized_AR.py)
- WANPolicyHead.kv_cache1/kv_cache_neg    (wan_flow_matching_action_tf.py)
- WANPolicyHead.clip_feas/ys              (wan_flow_matching_action_tf.py)
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Number of frames per chunk for subsequent calls (first call uses 1)
# Corresponds to: ARDroidRoboarenaPolicy.FRAMES_PER_CHUNK = 4
FRAMES_PER_CHUNK = 4


class DreamZeroState:
    """Pipeline persistent state across forward() calls.

    Lifecycle:
        - Created once in DreamZeroPipeline.__init__()
        - Mutated every forward() call (frame append, KV cache grow)
        - reset() on new session / language change / local_attn_size exceeded
    """

    def __init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------
    # Frame accumulation (single stitched buffer)
    # Transform outputs stitched single frame per call.
    # We accumulate here to build multi-frame video for AR inference.
    # Source: socket_test_optimized_AR.py L110-144 (adapted from per-camera to stitched)
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

        if len(self.stitched_buffer) >= num_frames:
            frames = self.stitched_buffer[-num_frames:]
        else:
            # Pad by repeating first frame
            frames = list(self.stitched_buffer)
            while len(frames) < num_frames:
                frames.insert(0, self.stitched_buffer[0])

        self.call_count += 1
        return np.stack(frames, axis=0)  # (T, H, W, C)

    # ------------------------------------------------------------------
    # Reset / should_reset
    # Source: wan_flow_matching_action_tf.py L968-981
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all state.

        Source:
        - socket_test_optimized_AR.py L302-330: ARDroidRoboarenaPolicy._reset_state
        - wan_flow_matching_action_tf.py L185-199: WANPolicyHead.__init__ state fields
        """
        # Frame buffer — single stitched buffer
        self.stitched_buffer: list[np.ndarray] = []
        self.call_count: int = 0

        # KV cache — from WANPolicyHead.__init__ L185-188
        self.kv_cache: list[torch.Tensor] | None = None
        self.kv_cache_neg: list[torch.Tensor] | None = None
        self.crossattn_cache: list[dict[str, bool | torch.Tensor | None]] | None = None
        self.crossattn_cache_neg: list[dict[str, bool | torch.Tensor | None]] | None = None
        self.current_start_frame: int = 0  # WANPolicyHead L199

        # Encoding cache — from WANPolicyHead.__init__ L197-200
        self.clip_feas: torch.Tensor | None = None
        self.ys: torch.Tensor | None = None
        self.language: torch.Tensor | None = None  # WANPolicyHead L200

    def should_reset(self, text_tokens: torch.Tensor | None, num_video_frames: int, local_attn_size: int) -> bool:
        """Determine if state should be reset before this forward().

        Source: wan_flow_matching_action_tf.py L968-981
        """
        # L968-971: first call (language not set yet)
        if self.language is None:
            logger.info("language is None, resetting")
            return True

        # L972-975: language changed
        if text_tokens is not None and not torch.equal(self.language, text_tokens):
            logger.info("language changed, resetting")
            return True

        # L976-978: single-frame input (signals new episode in real-world eval)
        # NOTE: after accumulate_frames, num_video_frames is the accumulated T
        # (1 for first call, 4 for subsequent). Only reset on true single-frame
        # which happens when the stitched_buffer was cleared externally.
        if num_video_frames == 1 and self.call_count > 1:
            logger.info("single frame input after first call, resetting")
            return True

        # L979-981: KV cache exceeded local attention window
        if local_attn_size != -1 and self.current_start_frame >= local_attn_size:
            logger.info(
                "current_start_frame %d >= local_attn_size %d, resetting", self.current_start_frame, local_attn_size
            )
            return True

        return False

    # ------------------------------------------------------------------
    # KV cache management
    # Source: wan_flow_matching_action_tf.py L480-512
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
        """Initialize empty KV caches and cross-attention caches.
        Source: wan_flow_matching_action_tf.py L480-512
        """
        self.kv_cache = [
            torch.zeros(2, batch_size, 0, num_heads, head_dim, dtype=dtype, device=device) for _ in range(num_layers)
        ]
        self.kv_cache_neg = [
            torch.zeros(2, batch_size, 0, num_heads, head_dim, dtype=dtype, device=device) for _ in range(num_layers)
        ]

        self.crossattn_cache = [{"is_init": False, "k": None, "v": None} for _ in range(num_layers)]
        self.crossattn_cache_neg = [{"is_init": False, "k": None, "v": None} for _ in range(num_layers)]

    def update_kv_cache(
        self,
        layer_index: int,
        updated_kv: torch.Tensor,
        is_negative: bool = False,
    ) -> None:
        """Update a single layer's KV cache after prefill.
        Source: wan_flow_matching_action_tf.py L856-858
        """
        cache = self.kv_cache_neg if is_negative else self.kv_cache
        assert cache is not None, "KV caches not initialized, call create_kv_caches first"
        cache[layer_index] = updated_kv.clone()

    def get_kv_caches(self, is_negative: bool = False) -> list[torch.Tensor]:
        """Get KV caches for the specified branch."""
        cache = self.kv_cache_neg if is_negative else self.kv_cache
        assert cache is not None, "KV caches not initialized"
        return cache

    def get_crossattn_caches(self, is_negative: bool = False) -> list[dict[str, bool | torch.Tensor | None]]:
        """Get cross-attention caches for the specified branch."""
        cache = self.crossattn_cache_neg if is_negative else self.crossattn_cache
        assert cache is not None, "Cross-attn caches not initialized"
        return cache
