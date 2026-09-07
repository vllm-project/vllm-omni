# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter presenting ``DreamZeroState``'s surface over the session manager.

``DreamZeroStateAdapter`` exposes the exact public methods and attributes that
``pipeline_dreamzero.py`` touches on its state object, so the pipeline can use
it interchangeably with the bespoke ``DreamZeroState`` (behind the opt-in flag).

DreamZero's attention KV lives in the AR-Diffusion engine's paged pool
(``self._ar_diffusion_kv_state``) and is *not* managed here; this adapter
covers the model's non-KV session state. Storage is delegated to typed
``StateObject`` instances owned by the ``SessionStateManager``:

    * the stitched frame buffer            -> ``LatentBuffer`` (ring)
    * accumulated AR video latents         -> ``LatentBuffer`` (append, CPU)
    * recent VAE encoder output            -> ``LatentBuffer`` (ring, bounded)

The remaining metadata (counters, conditioning tensors, the Wan encoder
causal-conv cache) lives in the session's ``attrs``, so a freshly constructed
adapter for an existing session sees the same data (the manager is the single
source of truth and the single LRU authority). Which values earn a
``StateObject`` and which stay attributes follows the rule in
``docs/features/session_state_manager.md``: a ``StateObject`` when the manager
must be able to act on the value, an attribute when it only carries it.

The public methods mirror ``DreamZeroState`` statement for statement, so the two
can be diffed to check equivalence. The deviations are forced by the buffer
members being ``StateObject``s rather than a plain deque/list/tensor: a read
uses ``.view()`` where the original reads ``list(...)`` or indexes/iterates the
container, and a wholesale replacement uses ``self._session.put(...)`` where the
original reassigns the attribute. The one behavioural difference is deliberate
and documented on ``_ensure_encoder_buffer``: the encoder output is bounded
here and unbounded in the original, which the pointwise ``quant_conv`` makes
numerically indistinguishable to the only reader.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from vllm_omni.diffusion.models.dreamzero.state_dreamzero import FRAMES_PER_CHUNK
from vllm_omni.experimental.world_models.session_state.attrs import SessionAttr
from vllm_omni.experimental.world_models.session_state.manager import SessionStateManager
from vllm_omni.experimental.world_models.session_state.objects import LatentBuffer

logger = logging.getLogger(__name__)

_FRAMES = "frames"
_VIDEO_LATENTS = "video_latents"
_VAE_ENCODER_OUT = "vae_encoder_out"

# Latent frames of VAE encoder output retained by default. DreamZero passes its
# own ``num_frame_per_block``; this is only the fallback for callers that build
# an adapter directly (tests, other entry points).
DEFAULT_VAE_ENCODER_WINDOW = 2


class DreamZeroStateAdapter:
    """Drop-in replacement for ``DreamZeroState`` backed by the manager.

    Session-scoped metadata is declared once per attribute via ``SessionAttr``
    (reads fall back to the default; writes go through to the pinned session).
    """

    call_count = SessionAttr[int](default=0, coerce=int)
    current_start_frame = SessionAttr[int](default=0, coerce=int)
    clip_feas = SessionAttr[torch.Tensor | None](default=None)
    ys = SessionAttr[torch.Tensor | None](default=None)
    language = SessionAttr[torch.Tensor | None](default=None)
    prompt_embeds = SessionAttr[torch.Tensor | None](default=None)
    # -- incremental VAE encoder stream --
    vae_stream_initialized = SessionAttr[bool](default=False, coerce=bool)
    # The Wan encoder's per-layer causal convolution cache: 24 entries, 603 MiB
    # per session, constant from the first few steps onward and by far the
    # largest thing a DreamZero session holds. It is an attribute all the same:
    # it is replaced wholesale, it has no recompute source (rebuilding it means
    # re-encoding the whole observation history, which is not retained), and
    # freeing it mid-session breaks the stream, so ending the session is the
    # only release -- which reclaims attributes anyway.
    vae_enc_feat_map = SessionAttr[list[torch.Tensor | None] | None](default=None)
    vae_pending_body_frames = SessionAttr[torch.Tensor | None](default=None)

    def __init__(
        self,
        session_id: str | None,
        manager: SessionStateManager,
        *,
        vae_encoder_window: int = DEFAULT_VAE_ENCODER_WINDOW,
    ) -> None:
        self._session_id = session_id
        if vae_encoder_window <= 0:
            raise ValueError(f"vae_encoder_window must be positive, got {vae_encoder_window}")
        self.vae_encoder_window = vae_encoder_window
        # Pin the session for this adapter's lifetime. The manager may evict the
        # session from its lookup table to bound memory, but an adapter that is
        # mid-generation keeps its own reference, so the in-progress state is not
        # lost -- matching the bespoke DreamZeroState, which the caller holds even
        # after it leaves the table. A fresh adapter is built per forward, so the
        # session is still marked recently-used on each step.
        self._session = manager.get_or_create_session(session_id)
        self._ensure_frame_buffer()

    @property
    def session_id(self) -> str | None:
        """The session this adapter views, as passed at construction.

        Lets the pipeline match the alias against a runner-supplied session id
        when the engine signals that session's close/reset.
        """
        return self._session_id

    # -- session / metadata plumbing ------------------------------------

    @staticmethod
    def _new_frame_buffer() -> LatentBuffer[np.ndarray]:
        buffer: LatentBuffer[np.ndarray] = LatentBuffer()
        buffer.allocate(maxlen=FRAMES_PER_CHUNK)
        return buffer

    @staticmethod
    def _new_video_buffer() -> LatentBuffer[torch.Tensor]:
        buffer: LatentBuffer[torch.Tensor] = LatentBuffer()
        buffer.allocate(maxlen=None)
        return buffer

    def _ensure_frame_buffer(self) -> LatentBuffer[np.ndarray]:
        """The stitched-frame ring for this session (created on first use)."""
        return self._session.get_or_create(_FRAMES, self._new_frame_buffer, LatentBuffer)

    def _ensure_video_buffer(self) -> LatentBuffer[torch.Tensor]:
        """The accumulated AR video-latent chunks (unbounded, CPU)."""
        return self._session.get_or_create(_VIDEO_LATENTS, self._new_video_buffer, LatentBuffer)

    def _new_encoder_buffer(self) -> LatentBuffer[torch.Tensor]:
        buffer: LatentBuffer[torch.Tensor] = LatentBuffer()
        buffer.allocate(maxlen=self.vae_encoder_window)
        return buffer

    def _ensure_encoder_buffer(self) -> LatentBuffer[torch.Tensor]:
        """The recent VAE encoder output, as a ring of per-chunk tensors.

        The bespoke state concatenates every encoded chunk into one tensor that
        is cleared only by a *session* reset -- a window reset deliberately
        keeps it -- so it grows for as long as the session lives. Its only
        reader, ``_vae_stream_get_observation_latents``, quantises the whole
        history and then keeps the last ``num_frame_per_block`` latent frames.
        ``quant_conv`` is a ``WanCausalConv3d`` of kernel size 1, pointwise in
        time, so those last frames do not depend on the discarded ones: keeping
        a ring of ``vae_encoder_window`` frames is numerically identical and
        bounded.
        """
        return self._session.get_or_create(_VAE_ENCODER_OUT, self._new_encoder_buffer, LatentBuffer)

    @property
    def vae_encoder_out(self) -> torch.Tensor | None:
        """The retained encoder output as one tensor, or ``None`` when empty."""
        chunks = self._ensure_encoder_buffer().view()
        if not chunks:
            return None
        if len(chunks) == 1:
            return chunks[0]
        return torch.cat(chunks, dim=2)

    @vae_encoder_out.setter
    def vae_encoder_out(self, value: torch.Tensor | None) -> None:
        # Seeding (``_vae_stream_seed``) and clearing (``reset``) both assign
        # wholesale; growth goes through ``append_vae_encoder_chunk``.
        buffer = self._new_encoder_buffer()
        if value is not None:
            buffer.append(value)
        self._session.put(_VAE_ENCODER_OUT, buffer)

    def append_vae_encoder_chunk(self, encoder_chunk: torch.Tensor) -> None:
        """Append one encoded chunk, evicting the oldest beyond the window."""
        self._ensure_encoder_buffer().append(encoder_chunk)

    @property
    def stitched_buffer(self) -> LatentBuffer[np.ndarray]:
        return self._ensure_frame_buffer()

    @property
    def video_latents_across_time(self) -> LatentBuffer[torch.Tensor]:
        # Read access only; a wholesale replace goes through ``_session.put``
        # (see reset / clear_video_latents).
        return self._ensure_video_buffer()

    def reset_vae_encoder_stream(self) -> None:
        """Clear incremental Wan VAE encoder state used across AR steps."""
        self.vae_stream_initialized = False
        self.vae_enc_feat_map = None
        self.vae_encoder_out = None
        self.vae_pending_body_frames = None

    # -- frame accumulation ----------------------------------------------

    def accumulate_frames(self, stitched: np.ndarray) -> np.ndarray:
        """Accumulate stitched frames and return multi-frame video."""
        if stitched.ndim == 3:
            self.stitched_buffer.append(stitched)
        elif stitched.ndim == 4:
            self.stitched_buffer.extend(list(stitched))
        else:
            raise ValueError(f"Expected 3D or 4D stitched, got {stitched.ndim}D")

        num_frames = 1 if self.call_count == 0 else FRAMES_PER_CHUNK

        buffer_frames = self.stitched_buffer.view()
        if len(buffer_frames) >= num_frames:
            frames = buffer_frames[-num_frames:]
        else:
            frames = buffer_frames
            while len(frames) < num_frames:
                frames.insert(0, buffer_frames[0])

        self.call_count += 1
        return np.stack(frames, axis=0)

    # -- video-latent accumulation ---------------------------------------

    def append_video_latents(self, video_out: torch.Tensor) -> None:
        """Append one AR chunk of normalized video latents for later decode."""
        if video_out.dim() != 5:
            raise ValueError(f"Expected 5D video_out, got shape {tuple(video_out.shape)}")
        # Upstream ``torch.cat(..., dim=2)`` uses (B, C, T, H, W).
        chunk = video_out.transpose(1, 2).detach().cpu()
        self.video_latents_across_time.append(chunk)
        logger.info(
            "append_video_latents: chunk_shape=%s total_chunks=%d total_latent_t=%d",
            tuple(chunk.shape),
            len(self.video_latents_across_time),
            int(sum(c.shape[2] for c in self.video_latents_across_time.view())),
        )

    def get_concatenated_video_latents(self) -> torch.Tensor | None:
        """Return all accumulated chunks concatenated along the time dimension."""
        chunks = self.video_latents_across_time.view()
        if not chunks:
            return None
        if len(chunks) == 1:
            return chunks[0]
        return torch.cat(chunks, dim=2)

    def clear_video_latents(self) -> None:
        """Drop accumulated video latents without resetting frame/VAE state."""
        self._session.put(_VIDEO_LATENTS, self._new_video_buffer())

    # -- reset / should_reset --------------------------------------------

    def reset(self, *, clear_video_latents: bool = True) -> None:
        """Clear session state.

        A session reset (``clear_video_latents=True``) drops everything,
        including the prompt-embed cache and the incremental VAE encoder stream.
        A window ("inference") reset (``False``) keeps the accumulated video
        latents, ``language``, ``prompt_embeds``, and the VAE encoder stream --
        the prompt is unchanged and the Wan feat_cache history is independent of
        the DiT attention window. The survivors are kept the way the original
        keeps them: by leaving them untouched inside the skipped ``if`` block.
        """
        saved_video_latents = [] if clear_video_latents else self._ensure_video_buffer().view()
        self._session.put(_FRAMES, self._new_frame_buffer())
        self.call_count = 0
        restored_video = self._new_video_buffer()
        restored_video.extend(saved_video_latents)
        self._session.put(_VIDEO_LATENTS, restored_video)

        self.current_start_frame = 0

        self.clip_feas = None
        self.ys = None
        if clear_video_latents:
            # Session reset: drop the prompt-embed cache and VAE encoder stream.
            self.language = None
            self.prompt_embeds = None
            self.reset_vae_encoder_stream()

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
        if self.language is None:
            logger.info("language is None, resetting")
            return "session"

        if text_tokens is not None and not torch.equal(self.language, text_tokens):
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
