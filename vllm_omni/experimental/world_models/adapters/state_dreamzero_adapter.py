# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter presenting ``DreamZeroState``'s surface over the session manager.

``DreamZeroStateAdapter`` exposes the exact public methods and attributes that
``pipeline_dreamzero.py`` touches on its state object, so the pipeline can use
it interchangeably with the bespoke ``DreamZeroState`` (behind the opt-in flag).
Storage is delegated to typed ``MemoryObject`` instances owned by the
``SessionMemoryManager``:

    * self-attention KV (pos / neg, per layer) -> ``PagedKV``
    * cross-attention KV (pos / neg, per layer) -> ``EncodeOnceKV``
    * the stitched frame buffer                 -> ``LatentBuffer``

The adapter holds no heavy state itself: scalar/tensor metadata lives in the
session's ``attrs`` so a freshly constructed adapter for an existing session
sees the same data (the manager is the single source of truth and the single
LRU authority).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import cast

import numpy as np
import torch

from vllm_omni.diffusion.models.dreamzero.state_dreamzero import FRAMES_PER_CHUNK
from vllm_omni.experimental.world_models.memory.base import MemoryObject
from vllm_omni.experimental.world_models.memory.manager import SessionMemoryManager
from vllm_omni.experimental.world_models.memory.objects import (
    EncodeOnceKV,
    LatentBuffer,
    PagedKV,
)

logger = logging.getLogger(__name__)

_FRAMES = "frames"
_VIDEO_LATENTS = "video_latents"
_META_DEFAULTS: dict[str, object] = {
    "call_count": 0,
    "current_start_frame": 0,
    "clip_feas": None,
    "ys": None,
    "language": None,
}


def _kv_key(layer_index: int, is_negative: bool) -> str:
    return f"kv_neg/{layer_index}" if is_negative else f"kv/{layer_index}"


def _xattn_key(layer_index: int, is_negative: bool) -> str:
    return f"xattn_neg/{layer_index}" if is_negative else f"xattn/{layer_index}"


class DreamZeroStateAdapter:
    """Drop-in replacement for ``DreamZeroState`` backed by the manager."""

    def __init__(self, session_id: str | None, manager: SessionMemoryManager) -> None:
        self._session_id = session_id
        # Pin the session for this adapter's lifetime. The manager may evict the
        # session from its lookup table to bound memory, but an adapter that is
        # mid-generation keeps its own reference, so the in-progress state is not
        # lost -- matching the bespoke DreamZeroState, which the caller holds even
        # after it leaves the table. A fresh adapter is built per forward, so the
        # session is still marked recently-used on each step.
        self._session = manager.get_or_create_session(session_id)
        self._ensure_frame_buffer()

    # -- session / metadata plumbing ------------------------------------

    @staticmethod
    def _new_frame_buffer() -> LatentBuffer:
        buffer = LatentBuffer()
        buffer.allocate(maxlen=FRAMES_PER_CHUNK)
        return buffer

    def _ensure_frame_buffer(self) -> LatentBuffer:
        buffer = self._session.get(_FRAMES)
        if not isinstance(buffer, LatentBuffer) or not buffer.resident:
            buffer = self._new_frame_buffer()
            self._session.put(_FRAMES, buffer)
        return buffer

    def _ensure_video_buffer(self) -> LatentBuffer:
        """The accumulated AR video-latent chunks (unbounded, CPU)."""
        buffer = self._session.get(_VIDEO_LATENTS)
        if not isinstance(buffer, LatentBuffer) or not buffer.resident:
            buffer = LatentBuffer()
            buffer.allocate(maxlen=None)
            self._session.put(_VIDEO_LATENTS, buffer)
        return buffer

    # Metadata getters read with a default so they never raise after a session
    # reset clears attrs; setters write through to the pinned session.
    @property
    def call_count(self) -> int:
        return int(self._session.attrs.get("call_count", _META_DEFAULTS["call_count"]))

    @call_count.setter
    def call_count(self, value: int) -> None:
        self._session.attrs["call_count"] = int(value)

    @property
    def current_start_frame(self) -> int:
        return int(self._session.attrs.get("current_start_frame", _META_DEFAULTS["current_start_frame"]))

    @current_start_frame.setter
    def current_start_frame(self, value: int) -> None:
        self._session.attrs["current_start_frame"] = int(value)

    @property
    def clip_feas(self) -> torch.Tensor | None:
        return cast("torch.Tensor | None", self._session.attrs.get("clip_feas"))

    @clip_feas.setter
    def clip_feas(self, value: torch.Tensor | None) -> None:
        self._session.attrs["clip_feas"] = value

    @property
    def ys(self) -> torch.Tensor | None:
        return cast("torch.Tensor | None", self._session.attrs.get("ys"))

    @ys.setter
    def ys(self, value: torch.Tensor | None) -> None:
        self._session.attrs["ys"] = value

    @property
    def language(self) -> torch.Tensor | None:
        return cast("torch.Tensor | None", self._session.attrs.get("language"))

    @language.setter
    def language(self, value: torch.Tensor | None) -> None:
        self._session.attrs["language"] = value

    @property
    def stitched_buffer(self) -> LatentBuffer:
        return self._ensure_frame_buffer()

    # -- frame accumulation (logic mirrors DreamZeroState) --------------

    def accumulate_frames(self, stitched: np.ndarray) -> np.ndarray:
        """Accumulate stitched frames and return multi-frame video.

        Behaviourally identical to ``DreamZeroState.accumulate_frames``.
        """
        buffer = self.stitched_buffer
        if stitched.ndim == 3:
            buffer.append(stitched)
        elif stitched.ndim == 4:
            buffer.extend(list(stitched))
        else:
            raise ValueError(f"Expected 3D or 4D stitched, got {stitched.ndim}D")

        num_frames = 1 if self.call_count == 0 else FRAMES_PER_CHUNK

        buffer_frames = buffer.view()
        if len(buffer_frames) >= num_frames:
            frames = buffer_frames[-num_frames:]
        else:
            frames = buffer_frames
            while len(frames) < num_frames:
                frames.insert(0, buffer_frames[0])

        self.call_count += 1
        return np.stack(frames, axis=0)

    # -- video-latent accumulation (logic mirrors DreamZeroState) -------

    def append_video_latents(self, video_out: torch.Tensor) -> None:
        """Append one AR chunk of normalized video latents for later decode."""
        if video_out.dim() != 5:
            raise ValueError(f"Expected 5D video_out, got shape {tuple(video_out.shape)}")
        # Upstream ``torch.cat(..., dim=2)`` uses (B, C, T, H, W).
        chunk = video_out.transpose(1, 2).detach().cpu()
        buffer = self._ensure_video_buffer()
        buffer.append(chunk)
        logger.info(
            "append_video_latents: chunk_shape=%s total_chunks=%d total_latent_t=%d",
            tuple(chunk.shape),
            len(buffer),
            int(sum(c.shape[2] for c in buffer.view())),
        )

    def get_concatenated_video_latents(self) -> torch.Tensor | None:
        """Return all accumulated chunks concatenated along the time dimension."""
        buffer = self._session.get(_VIDEO_LATENTS)
        if not isinstance(buffer, LatentBuffer) or not buffer.resident:
            return None
        chunks = buffer.view()
        if not chunks:
            return None
        if len(chunks) == 1:
            return cast("torch.Tensor", chunks[0])
        return torch.cat(chunks, dim=2)

    def clear_video_latents(self) -> None:
        """Drop accumulated video latents without resetting KV/frame state."""
        buffer = LatentBuffer()
        buffer.allocate(maxlen=None)
        self._session.put(_VIDEO_LATENTS, buffer)

    # -- reset / should_reset (logic mirrors DreamZeroState) ------------

    def reset(self, *, clear_video_latents: bool = True) -> None:
        """Clear session state.

        When ``clear_video_latents`` is ``False`` the accumulated video latents
        and the ``language`` token are preserved across the reset, matching
        ``DreamZeroState.reset`` -- this is the ``reset_inference_state`` path
        taken when local attention rolls over but the rollout continues.
        """
        saved_chunks: list[object] = []
        saved_language: torch.Tensor | None = None
        if not clear_video_latents:
            buffer = self._session.get(_VIDEO_LATENTS)
            if isinstance(buffer, LatentBuffer) and buffer.resident:
                saved_chunks = list(buffer.view())
            saved_language = self.language

        # Canonical reset: drop every typed object and clear session metadata.
        self._session.reset()
        # Leave an empty, allocated frame buffer ready for the next call.
        self._ensure_frame_buffer()

        if not clear_video_latents:
            if saved_language is not None:
                self.language = saved_language
            restored = self._ensure_video_buffer()
            if saved_chunks:
                restored.extend(saved_chunks)

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
        language = self.language
        if language is None:
            logger.info("language is None, resetting")
            return "session"

        if text_tokens is not None and not torch.equal(language, text_tokens):
            logger.info("language changed, resetting")
            return "session"

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

    # -- KV cache management --------------------------------------------

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
        session = self._session
        for i in range(num_layers):
            for is_neg in (False, True):
                kv = PagedKV()
                kv.allocate(
                    batch_size=batch_size,
                    dtype=dtype,
                    device=device,
                    num_heads=num_heads,
                    head_dim=head_dim,
                )
                session.put(_kv_key(i, is_neg), kv)

                xattn = EncodeOnceKV()
                xattn.allocate()
                session.put(_xattn_key(i, is_neg), xattn)
        session.attrs["_num_layers"] = num_layers

    def update_kv_cache(self, layer_index: int, updated_kv: torch.Tensor, is_negative: bool = False) -> None:
        """Update a single layer's KV cache after prefill."""
        obj = self._session.get(_kv_key(layer_index, is_negative))
        if obj is None:
            raise RuntimeError("KV caches not initialized, call create_kv_caches first.")
        obj.commit(updated_kv)

    def get_kv_caches(self, is_negative: bool = False) -> list[torch.Tensor]:
        """Get KV caches for the specified branch."""
        return [obj.view() for obj in self._iter_layer_objects(_kv_key, is_negative, "KV caches")]

    def get_crossattn_caches(self, is_negative: bool = False) -> list[dict[str, bool | torch.Tensor | None]]:
        """Get cross-attention caches for the specified branch."""
        return [obj.view() for obj in self._iter_layer_objects(_xattn_key, is_negative, "Cross-attn caches")]

    def _iter_layer_objects(
        self,
        key_fn: Callable[[int, bool], str],
        is_negative: bool,
        what: str,
    ) -> list[MemoryObject]:
        session = self._session
        num_layers = session.attrs.get("_num_layers")
        if num_layers is None:
            raise RuntimeError(f"{what} not initialized.")
        objects: list[MemoryObject] = []
        for i in range(int(num_layers)):
            obj = session.get(key_fn(i, is_negative))
            if obj is None or not obj.resident:
                raise RuntimeError(f"{what} not initialized.")
            objects.append(obj)
        return objects
