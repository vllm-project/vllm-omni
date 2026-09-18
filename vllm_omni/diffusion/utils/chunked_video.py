# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Turn committed VAE chunks into MP4 bytes without materializing the video.

The producer side is the model's business: a Wan VAE walks a causal
frame-by-frame loop, a MiniMax-H3 VAE walks overlapping clips, and Wan S2V
decodes one clip per autoregressive iteration. What happens to a chunk once it
is final is not: every one of them quantizes to uint8, moves to the host once,
and lands in a bounded encoder. That consumer lives here so a model only has to
publish finished chunks -- :class:`SupportsChunkedVAEDecode` for producers that
can be driven, :class:`ChunkedVideoMP4Session` for producers that drive
themselves.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from vllm_omni.diffusion.models.interface import supports_chunked_vae_decode
from vllm_omni.diffusion.utils.media_utils import ChunkedMP4Encoder


def chunk_to_uint8_frames(chunk: torch.Tensor, value_range: tuple[float, float]) -> np.ndarray:
    """Quantize a ``BCTHW`` chunk to ``BTHWC`` uint8, transferring once.

    ``value_range`` is the interval the producer publishes, which differs per
    checkpoint (see ``SupportsChunkedVAEDecode.chunk_value_range``). Quantizing
    on the accelerator means one transfer moves the final bytes rather than
    float frames.

    That transfer is synchronous: ``.cpu()`` blocks inside the producer's chunk
    callback, so what overlaps the remaining decode today is the CPU H.264
    encode of earlier chunks, not this copy. Splitting device preparation,
    device-to-host copy, and encode into three independently scheduled stages --
    a pool of reusable pinned host slots with non-blocking copies on a dedicated
    stream, leases released by the encoder -- fits behind this same callback
    contract, but is not implemented here.
    """
    low, high = float(value_range[0]), float(value_range[1])
    if high <= low:
        raise ValueError(f"value_range must be increasing, got {value_range!r}")
    scale = 255.0 / (high - low)
    frames = chunk.clamp(low, high).sub(low).mul(scale).round().to(torch.uint8)
    return frames.permute(0, 2, 3, 4, 1).cpu().numpy()


class ChunkedVideoMP4Session:
    """Encode committed video chunks into one progressive MP4 per batch entry.

    Push finished ``BCTHW`` chunks as the producer commits them; each batch
    entry gets its own bounded encoder, so host transfer and H.264 encoding
    overlap whatever the producer is still decoding.

    ``audio_waveforms`` holds one waveform per batch entry (``None`` for a
    silent entry), so a caller with several outputs per prompt repeats a
    request's waveform across its entries. ``batch_frames`` coalesces transfers
    for producers that publish finer than a transfer is worth; ``crop`` trims
    the decoder's padding to the requested output size.
    """

    def __init__(
        self,
        *,
        value_range: tuple[float, float],
        fps: float,
        audio_waveforms: list[np.ndarray | None] | None = None,
        audio_sample_rate: int | None = None,
        batch_frames: int = 1,
        max_pending: int = 2,
        video_codec_options: dict[str, str] | None = None,
        crop: tuple[int, int] | None = None,
    ) -> None:
        if batch_frames <= 0:
            raise ValueError("batch_frames must be positive")
        self._value_range = value_range
        self._fps = fps
        self._audio_waveforms = audio_waveforms
        self._audio_sample_rate = audio_sample_rate
        self._batch_frames = batch_frames
        self._max_pending = max_pending
        self._video_codec_options = video_codec_options
        self._crop = crop
        self._encoders: list[ChunkedMP4Encoder] = []
        self._pending: list[torch.Tensor] = []
        self._pending_frames = 0

    def push(self, chunk: torch.Tensor) -> None:
        """Queue one committed ``BCTHW`` chunk."""
        if self._crop is not None:
            height, width = self._crop
            chunk = chunk[..., :height, :width]
        self._pending.append(chunk)
        self._pending_frames += int(chunk.shape[2])
        if self._pending_frames >= self._batch_frames:
            self._flush()

    def finish(self) -> list[bytes]:
        """Flush what is pending and return one MP4 per batch entry."""
        self._flush()
        return [encoder.finish() for encoder in self._encoders]

    def abort(self) -> None:
        for encoder in self._encoders:
            encoder.abort()

    def _flush(self) -> None:
        if not self._pending:
            return
        frames = chunk_to_uint8_frames(torch.cat(self._pending, dim=2), self._value_range)
        if not self._encoders:
            self._encoders = [
                ChunkedMP4Encoder(
                    width=frames.shape[3],
                    height=frames.shape[2],
                    fps=self._fps,
                    audio_waveform=self._waveform_for(index, frames.shape[0]),
                    audio_sample_rate=self._audio_sample_rate,
                    max_pending=self._max_pending,
                    video_codec_options=self._video_codec_options,
                )
                for index in range(frames.shape[0])
            ]
        for index, encoder in enumerate(self._encoders):
            encoder.push(np.ascontiguousarray(frames[index]))
        self._pending.clear()
        self._pending_frames = 0

    def _waveform_for(self, index: int, batch_size: int) -> np.ndarray | None:
        if self._audio_waveforms is None:
            return None
        if len(self._audio_waveforms) != batch_size:
            raise ValueError(
                f"expected one audio waveform per batch entry, got "
                f"{len(self._audio_waveforms)} for {batch_size} entries"
            )
        return self._audio_waveforms[index]


def decode_to_mp4(vae: Any, z: torch.Tensor, **session_kwargs: Any) -> list[bytes]:
    """Decode ``z`` straight into one progressive MP4 per batch entry.

    Drives a VAE that declares :class:`SupportsChunkedVAEDecode`, so host
    transfer and encoding overlap the remaining decode and the full video is
    never materialized. Ranks that own no decode output receive no chunks and
    get an empty list, matching the empty tensor the full-decode path returns
    there.
    """
    if not supports_chunked_vae_decode(vae):
        raise TypeError(f"{type(vae).__name__} does not expose the chunked VAE decode capability")

    session = ChunkedVideoMP4Session(value_range=vae.chunk_value_range, **session_kwargs)
    try:
        vae.decode_with_chunks(z, on_chunk=session.push)
        return session.finish()
    except BaseException:
        session.abort()
        raise
