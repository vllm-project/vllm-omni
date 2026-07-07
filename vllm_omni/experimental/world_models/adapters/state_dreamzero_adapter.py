# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter presenting ``DreamZeroState``'s surface over the session manager.

``DreamZeroStateAdapter`` exposes the exact public methods and attributes that
``pipeline_dreamzero.py`` touches on its state object, so the pipeline can use
it interchangeably with the bespoke ``DreamZeroState`` (behind the opt-in flag).

DreamZero's attention KV lives in the AR-Diffusion engine's paged pool
(``self._ar_diffusion_kv_state``, see PR #4534) and is *not* managed here; this
adapter covers the model's non-KV session state. Storage is delegated to typed
``MemoryObject`` instances owned by the ``SessionMemoryManager``:

    * the stitched frame buffer            -> ``LatentBuffer`` (ring)
    * accumulated AR video latents         -> ``LatentBuffer`` (append, CPU)

Scalar/tensor metadata (counters, prompt embeds, incremental VAE encoder
stream) lives in the session's ``attrs`` so a freshly constructed adapter for
an existing session sees the same data (the manager is the single source of
truth and the single LRU authority).
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
import torch

from vllm_omni.diffusion.models.dreamzero.state_dreamzero import FRAMES_PER_CHUNK
from vllm_omni.experimental.world_models.memory.manager import SessionMemoryManager
from vllm_omni.experimental.world_models.memory.objects import LatentBuffer

logger = logging.getLogger(__name__)

_FRAMES = "frames"
_VIDEO_LATENTS = "video_latents"


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
        return int(cast("int", self._session.attrs.get("call_count", 0)))

    @call_count.setter
    def call_count(self, value: int) -> None:
        self._session.attrs["call_count"] = int(value)

    @property
    def current_start_frame(self) -> int:
        return int(cast("int", self._session.attrs.get("current_start_frame", 0)))

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
    def prompt_embeds(self) -> torch.Tensor | None:
        return cast("torch.Tensor | None", self._session.attrs.get("prompt_embeds"))

    @prompt_embeds.setter
    def prompt_embeds(self, value: torch.Tensor | None) -> None:
        self._session.attrs["prompt_embeds"] = value

    @property
    def stitched_buffer(self) -> LatentBuffer:
        return self._ensure_frame_buffer()

    # -- incremental VAE encoder stream (logic mirrors DreamZeroState) --

    @property
    def vae_stream_initialized(self) -> bool:
        return bool(self._session.attrs.get("vae_stream_initialized", False))

    @vae_stream_initialized.setter
    def vae_stream_initialized(self, value: bool) -> None:
        self._session.attrs["vae_stream_initialized"] = bool(value)

    @property
    def vae_enc_feat_map(self) -> list[torch.Tensor | None] | None:
        return cast("list[torch.Tensor | None] | None", self._session.attrs.get("vae_enc_feat_map"))

    @vae_enc_feat_map.setter
    def vae_enc_feat_map(self, value: list[torch.Tensor | None] | None) -> None:
        self._session.attrs["vae_enc_feat_map"] = value

    @property
    def vae_encoder_out(self) -> torch.Tensor | None:
        return cast("torch.Tensor | None", self._session.attrs.get("vae_encoder_out"))

    @vae_encoder_out.setter
    def vae_encoder_out(self, value: torch.Tensor | None) -> None:
        self._session.attrs["vae_encoder_out"] = value

    @property
    def vae_pending_body_frames(self) -> torch.Tensor | None:
        return cast("torch.Tensor | None", self._session.attrs.get("vae_pending_body_frames"))

    @vae_pending_body_frames.setter
    def vae_pending_body_frames(self, value: torch.Tensor | None) -> None:
        self._session.attrs["vae_pending_body_frames"] = value

    def reset_vae_encoder_stream(self) -> None:
        """Clear incremental Wan VAE encoder state used across AR steps."""
        self.vae_stream_initialized = False
        self.vae_enc_feat_map = None
        self.vae_encoder_out = None
        self.vae_pending_body_frames = None

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
        """Drop accumulated video latents without resetting frame/VAE state."""
        buffer = LatentBuffer()
        buffer.allocate(maxlen=None)
        self._session.put(_VIDEO_LATENTS, buffer)

    # -- reset / should_reset (logic mirrors DreamZeroState) ------------

    def reset(self, *, clear_video_latents: bool = True) -> None:
        """Clear session state.

        Mirrors ``DreamZeroState.reset``: a session reset (``True``) drops
        everything including the prompt-embed cache and the incremental VAE
        encoder stream; a window ("inference") reset (``False``) keeps the
        accumulated video latents, ``language``, ``prompt_embeds``, and the
        VAE encoder stream -- the prompt is unchanged and the Wan feat_cache
        history is independent of the DiT attention window.
        """
        saved_attrs: dict[str, object] = {}
        saved_chunks: list[object] = []
        if not clear_video_latents:
            buffer = self._session.get(_VIDEO_LATENTS)
            if isinstance(buffer, LatentBuffer) and buffer.resident:
                saved_chunks = list(buffer.view())
            for key in (
                "language",
                "prompt_embeds",
                "vae_stream_initialized",
                "vae_enc_feat_map",
                "vae_encoder_out",
                "vae_pending_body_frames",
            ):
                if key in self._session.attrs:
                    saved_attrs[key] = self._session.attrs[key]

        # Canonical reset: drop every typed object and clear session metadata.
        self._session.reset()
        # Leave an empty, allocated frame buffer ready for the next call.
        self._ensure_frame_buffer()

        if not clear_video_latents:
            self._session.attrs.update(saved_attrs)
            restored = self._ensure_video_buffer()
            if saved_chunks:
                restored.extend(saved_chunks)

    def reset_inference_state(self) -> None:
        """Reset window/frame state after local attention rolls without dropping video latents."""
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
