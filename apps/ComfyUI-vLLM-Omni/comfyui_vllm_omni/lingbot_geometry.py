# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Chunk geometry for LingBot World video, in one place.

A LingBot rollout is decoded causally, so the first chunk yields fewer pixel
frames than every later one. Both counts are derived from two weight
properties rather than written down:

- ``vae_scale_factor_temporal`` (4): the VAE expands one steady-state latent
  frame into this many pixel frames.
- ``num_frames_per_block`` (3): how many latent frames one AR block (one
  chunk) covers.

Streaming decode expands the session's opening latent to a single frame and
every later latent to the full factor, so chunk 0 is ``(block - 1) * temporal
+ 1`` pixel frames and every later chunk is ``block * temporal``. That is the
``9 + 12k`` grid the ComfyUI nodes, the WebSocket start message and the
``camera_action_script`` all have to agree on.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Latent frames per generated chunk; ``LingBotCameraControlReducer`` uses 3.
DEFAULT_LATENT_FRAMES_PER_CHUNK = 3
#: ``vae_scale_factor_temporal`` of the shipped Wan-derived VAE.
DEFAULT_TEMPORAL_SCALE = 4


@dataclass(frozen=True)
class LingBotGeometry:
    """How pixel frames map onto generated chunks."""

    latent_frames_per_chunk: int = DEFAULT_LATENT_FRAMES_PER_CHUNK
    temporal_scale: int = DEFAULT_TEMPORAL_SCALE

    def __post_init__(self) -> None:
        if self.latent_frames_per_chunk <= 0:
            raise ValueError("latent_frames_per_chunk must be positive.")
        if self.temporal_scale <= 0:
            raise ValueError("temporal_scale must be positive.")

    @property
    def first_chunk_frames(self) -> int:
        """Pixel frames in chunk 0."""
        return (self.latent_frames_per_chunk - 1) * self.temporal_scale + 1

    @property
    def chunk_frames(self) -> int:
        """Pixel frames in every chunk after the first."""
        return self.latent_frames_per_chunk * self.temporal_scale

    @property
    def grid_step(self) -> int:
        """How far ``num_frames`` steps: the frames one later chunk adds."""
        return self.chunk_frames

    def frames_for(self, chunks: int) -> int:
        """Total pixel frames produced by ``chunks`` chunks."""
        if chunks < 0:
            raise ValueError("chunks must not be negative.")
        if chunks == 0:
            return 0
        return self.first_chunk_frames + (chunks - 1) * self.chunk_frames

    def chunks_for(self, num_frames: int) -> int:
        """Chunks needed for ``num_frames``; rejects a request off the grid."""
        if num_frames < self.first_chunk_frames:
            raise ValueError(
                f"LingBot needs at least {self.first_chunk_frames} frames; got {num_frames}. "
                f"Use num_frames = {self.first_chunk_frames} + {self.grid_step}k."
            )
        offset = num_frames - self.first_chunk_frames
        if offset % self.grid_step:
            nearest = self.first_chunk_frames + self.grid_step * round(offset / self.grid_step)
            raise ValueError(
                f"LingBot needs num_frames = {self.first_chunk_frames} + {self.grid_step}k; "
                f"got {num_frames}. The nearest legal lengths are {nearest - self.grid_step} and {nearest}."
            )
        return offset // self.grid_step + 1

    def chunk_start_frame(self, chunk: int) -> int:
        """First pixel frame of ``chunk``."""
        if chunk < 0:
            raise ValueError("chunk must not be negative.")
        if chunk == 0:
            return 0
        return self.first_chunk_frames + (chunk - 1) * self.chunk_frames

    def sample_frame(self, chunk: int, latent: int) -> int:
        """Pixel frame that stands for ``latent`` inside ``chunk``.

        A chunk's latent frames are evenly spread over its pixel frames, so the
        sample of a chunk that starts at ``start`` is ``start + temporal*latent``.
        The opening chunk is the same rule read from frame 0.
        """
        if not 0 <= latent < self.latent_frames_per_chunk:
            raise ValueError(f"latent must be in [0, {self.latent_frames_per_chunk}); got {latent}.")
        return self.chunk_start_frame(chunk) + self.temporal_scale * latent

    def shot_frames(self, index: int, chunks: int) -> int:
        """Frames a shot of ``chunks`` chunks holds, at ``index`` in a sequence.

        Only the first shot opens a scene, so only it gets the short first
        chunk; every later shot is a whole number of full chunks.
        """
        if chunks <= 0:
            raise ValueError("A shot must last at least one chunk.")
        return self.frames_for(chunks) if index == 0 else self.chunk_frames * chunks

    def shot_chunks(self, index: int, frames: int) -> int:
        """Chunks a shot of ``frames`` pixel frames covers, at ``index``."""
        if index == 0:
            return self.chunks_for(frames)
        if frames < self.chunk_frames:
            raise ValueError(f"A shot must last at least one {self.chunk_frames}-frame chunk; got {frames}.")
        if frames % self.chunk_frames:
            raise ValueError(f"A shot after the first needs whole chunks of {self.chunk_frames} frames; got {frames}.")
        return frames // self.chunk_frames


DEFAULT_GEOMETRY = LingBotGeometry()
