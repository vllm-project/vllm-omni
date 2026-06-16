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

from vllm_omni.diffusion.memory.base import MemoryObject
from vllm_omni.diffusion.memory.manager import SessionMemory, SessionMemoryManager
from vllm_omni.diffusion.memory.objects import EncodeOnceKV, LatentBuffer, PagedKV
from vllm_omni.diffusion.models.dreamzero.state_dreamzero import FRAMES_PER_CHUNK

logger = logging.getLogger(__name__)

_FRAMES = "frames"
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
        self._manager = manager
        session = manager.get_or_create_session(session_id)
        for key, default in _META_DEFAULTS.items():
            session.attrs.setdefault(key, default)
        if session.get(_FRAMES) is None:
            self._fresh_frame_buffer(session)

    # -- session / metadata plumbing ------------------------------------

    @property
    def _session(self) -> SessionMemory:
        # Re-fetch each access so the manager's LRU sees the touch and the
        # adapter never caches a stale handle across evictions.
        return self._manager.get_or_create_session(self._session_id)

    @staticmethod
    def _fresh_frame_buffer(session: SessionMemory) -> LatentBuffer:
        buffer = LatentBuffer()
        buffer.allocate(maxlen=FRAMES_PER_CHUNK)
        session.put(_FRAMES, buffer)
        return buffer

    @property
    def call_count(self) -> int:
        return int(self._session.attrs["call_count"])

    @call_count.setter
    def call_count(self, value: int) -> None:
        self._session.attrs["call_count"] = int(value)

    @property
    def current_start_frame(self) -> int:
        return int(self._session.attrs["current_start_frame"])

    @current_start_frame.setter
    def current_start_frame(self, value: int) -> None:
        self._session.attrs["current_start_frame"] = int(value)

    @property
    def clip_feas(self) -> torch.Tensor | None:
        return cast("torch.Tensor | None", self._session.attrs["clip_feas"])

    @clip_feas.setter
    def clip_feas(self, value: torch.Tensor | None) -> None:
        self._session.attrs["clip_feas"] = value

    @property
    def ys(self) -> torch.Tensor | None:
        return cast("torch.Tensor | None", self._session.attrs["ys"])

    @ys.setter
    def ys(self, value: torch.Tensor | None) -> None:
        self._session.attrs["ys"] = value

    @property
    def language(self) -> torch.Tensor | None:
        return cast("torch.Tensor | None", self._session.attrs["language"])

    @language.setter
    def language(self, value: torch.Tensor | None) -> None:
        self._session.attrs["language"] = value

    @property
    def stitched_buffer(self) -> LatentBuffer:
        buffer = self._session.get(_FRAMES)
        if not isinstance(buffer, LatentBuffer):
            buffer = self._fresh_frame_buffer(self._session)
        return buffer

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

    # -- reset / should_reset (logic mirrors DreamZeroState) ------------

    def reset(self) -> None:
        """Clear all state for this session."""
        session = self._session
        # Drop the typed KV/cross-attn objects (recreated by create_kv_caches).
        for name in list(session.names()):
            if name != _FRAMES:
                obj = session.get(name)
                if obj is not None:
                    obj.reset()
        self._fresh_frame_buffer(session)
        session.attrs.update({key: default for key, default in _META_DEFAULTS.items()})

    def should_reset(self, text_tokens: torch.Tensor | None, num_video_frames: int, local_attn_size: int) -> bool:
        """Determine if state should be reset before this forward()."""
        language = self.language
        if language is None:
            logger.info("language is None, resetting")
            return True

        if text_tokens is not None and not torch.equal(language, text_tokens):
            logger.info("language changed, resetting")
            return True

        if num_video_frames == 1 and self.call_count > 1:
            logger.info("single frame input after first call, resetting")
            return True

        if local_attn_size != -1 and self.current_start_frame >= local_attn_size:
            logger.info(
                "current_start_frame %d >= local_attn_size %d, resetting",
                self.current_start_frame,
                local_attn_size,
            )
            return True

        return False

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
