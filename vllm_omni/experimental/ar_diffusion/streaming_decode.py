# SPDX-License-Identifier: Apache-2.0
"""Session-owned streaming VAE decode for realtime AR-Diffusion.

A causal video decoder already decodes one latent frame at a time, carrying a
bounded temporal cache between frames. What it does not do today is carry that
cache *between requests*: every decode call clears it, so a chunk can only be
decoded as part of a whole clip.

This module moves the cache from the decoder module to the session, which is
the entire difference between "decode a clip" and "stream a session":

* the cache becomes :class:`StreamingDecodeState`, threaded in and out by the
  caller, so two concurrent sessions cannot overwrite each other's temporal
  context -- a failure that produces a plausible but wrong video while every
  timing metric stays green;
* ``first_chunk`` becomes a property of the session's first frame rather than
  of the call, which is what makes chunk ``N + 1`` continue chunk ``N``.

Resident state is bounded by construction: each causal convolution retains at
most ``CACHE_T`` temporal slices, so the cache is a function of resolution and
never of session length. It is not free -- see :meth:`StreamingDecodeState.nbytes`,
which is the quantity session admission has to account for.

Measured on the reference checkpoint at 832x480 in bf16, on one RTX PRO 6000
with torch 2.11: chunking is neutral -- a session decoded in four chunks is
bit-identical to the same session decoded in one call -- resident decoder
state holds at 1801 MiB across every chunk, and time to first frame drops
from 1.9 s, the whole-clip barrier, to 0.31 s without the session costing more
in total.

Note what "neutral" is measured against. ``AutoencoderKLWan._decode`` applies
``post_quant_conv`` to the whole latent at once, while the tiled path in this
repo -- and this decoder -- apply it per frame. It is a 1x1x1 convolution, so
the two are mathematically identical, but they can select different kernels;
on bf16 CUDA that seeds a 0.004 difference which the causal convolutions
downstream amplify to 0.047. Comparing against ``_decode`` therefore measures
where ``post_quant_conv`` runs, not anything about streaming, and the result
moves with the torch version. The control that isolates chunking is the same
session decoded in a single call through this same path.

Only ``torch`` is imported here so the contract can be exercised without a
device, a checkpoint, or the distributed VAE stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch


@dataclass
class StreamingDecodeState:
    """One session's temporal decoder cache.

    ``feat_map`` holds one entry per causal convolution. Entries are tensors,
    ``None`` before that convolution has run, or a non-tensor sentinel the
    decoder uses for its first upsample step, so byte accounting must skip
    anything that is not a tensor.

    The cache has **no recompute source**: it is the compressed tail of every
    frame decoded so far and cannot be rebuilt from the prompt or the latents.
    Releasing it ends the stream; there is no partial eviction.
    """

    session_id: str
    feat_map: list[Any]
    conv_idx: list[int] = field(default_factory=lambda: [0])
    frames_decoded: int = 0
    chunks_decoded: int = 0

    def nbytes(self) -> int:
        """Resident bytes of the cache, skipping non-tensor entries."""
        return sum(t.numel() * t.element_size() for t in self.feat_map if torch.is_tensor(t))

    def nbytes_by_device(self) -> dict[str, int]:
        """Resident bytes grouped by device, for admission accounting."""
        totals: dict[str, int] = {}
        for tensor in self.feat_map:
            if torch.is_tensor(tensor):
                key = str(tensor.device)
                totals[key] = totals.get(key, 0) + tensor.numel() * tensor.element_size()
        return totals

    @property
    def started(self) -> bool:
        return self.frames_decoded > 0

    def release(self) -> None:
        """Drop the cache and return the session to its pre-stream state."""
        self.feat_map = [None] * len(self.feat_map)
        self.conv_idx = [0]
        self.frames_decoded = 0
        self.chunks_decoded = 0


@runtime_checkable
class SupportsStreamingDecode(Protocol):
    """A decoder that can emit a session's chunks incrementally.

    The lifecycle mirrors the AR session's: ``new_decode_state`` at session
    creation, ``decode_chunk`` per committed chunk, and ``release`` on reset or
    close, so decoder state and KV state are released together rather than
    through two independent lifecycles.
    """

    def new_decode_state(self, session_id: str) -> StreamingDecodeState:
        """Create empty decoder state for a new session."""
        ...

    def decode_chunk(self, latent: torch.Tensor, state: StreamingDecodeState) -> torch.Tensor:
        """Decode one committed chunk, advancing ``state`` in place."""
        ...

    def declared_state_bytes(self, *, height: int, width: int, dtype: torch.dtype) -> int:
        """Upper bound on resident decoder bytes per session at this shape."""
        ...

    def release(self, state: StreamingDecodeState) -> None:
        """Drop ``state``'s cache and return it to its pre-stream state.

        Declared here, not just implemented on :class:`WanStreamingDecoder`,
        so a caller programming against this protocol -- a session runtime
        releasing decoder state alongside KV state on close or failure -- can
        rely on it being present without depending on the concrete class.
        """
        ...


class WanStreamingDecoder:
    """Streaming decode over a Wan-family causal autoencoder.

    Wraps any module exposing ``decoder``, ``post_quant_conv`` and the cached
    causal-convolution counts. It never touches the module's own ``_feat_map``,
    so a module shared by several sessions stays stateless between calls.
    """

    def __init__(self, vae: Any, *, bytes_per_pixel_fp32: float | None = None) -> None:
        for attribute in ("decoder", "post_quant_conv"):
            if not hasattr(vae, attribute):
                raise TypeError(f"Streaming decode requires a Wan-family autoencoder exposing {attribute!r}.")
        self._vae = vae
        # A one-off measured constant for this checkpoint; see declared_state_bytes.
        # Optional at construction because most callers never need admission
        # accounting, but a constructor argument (rather than requiring an
        # external attribute assignment after the fact) means the value can't
        # be silently unset for a caller that does.
        self._bytes_per_pixel_fp32 = bytes_per_pixel_fp32

    @property
    def num_causal_convs(self) -> int:
        counts = getattr(self._vae, "_cached_conv_counts", None)
        if not isinstance(counts, dict) or "decoder" not in counts:
            raise TypeError("The autoencoder does not expose cached decoder convolution counts.")
        return int(counts["decoder"])

    def new_decode_state(self, session_id: str) -> StreamingDecodeState:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string.")
        return StreamingDecodeState(session_id=session_id, feat_map=[None] * self.num_causal_convs)

    def decode_chunk(self, latent: torch.Tensor, state: StreamingDecodeState) -> torch.Tensor:
        """Decode ``latent`` as the continuation of ``state``'s session.

        ``latent`` is ``[B, C, T, H, W]`` in latent space, already rescaled by
        the pipeline's latent statistics. Frames are decoded one at a time, the
        same loop the non-streaming path runs, with the cache carried across
        calls instead of cleared. The returned tensor is ``[B, 3, T', H', W']``
        where ``T'`` is smaller for a session's first chunk: the opening latent
        frame expands to a single raw frame and every later one to the full
        temporal factor.
        """
        if latent.ndim != 5:
            raise ValueError(f"latent must be [B, C, T, H, W]; got shape {tuple(latent.shape)}.")
        if len(state.feat_map) != self.num_causal_convs:
            raise ValueError(
                "Decoder state does not belong to this decoder: "
                f"{len(state.feat_map)} cache slots against {self.num_causal_convs} causal convolutions."
            )
        num_frames = int(latent.shape[2])
        if num_frames == 0:
            raise ValueError("latent must carry at least one frame.")

        decoded_frames = []
        for index in range(num_frames):
            state.conv_idx[0] = 0
            frame = self._vae.post_quant_conv(latent[:, :, index : index + 1])
            # first_chunk marks the session's opening frame, not the call's:
            # that is what makes chunk N + 1 continue chunk N rather than
            # restart the causal expansion.
            decoded_frames.append(
                self._vae.decoder(
                    frame,
                    feat_cache=state.feat_map,
                    feat_idx=state.conv_idx,
                    first_chunk=(state.frames_decoded == 0),
                )
            )
            state.frames_decoded += 1

        out = torch.cat(decoded_frames, dim=2)
        patch_size = getattr(getattr(self._vae, "config", None), "patch_size", None)
        if patch_size is not None:
            from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

            out = unpatchify(out, patch_size=patch_size)
        out = torch.clamp(out, min=-1.0, max=1.0)
        state.chunks_decoded += 1
        return out

    def release(self, state: StreamingDecodeState) -> None:
        state.release()

    def declared_state_bytes(self, *, height: int, width: int, dtype: torch.dtype) -> int:
        """Resident decoder bytes per session at this output shape.

        Every cached tensor is ``[B, C_l, <= CACHE_T, H_l, W_l]`` with ``H_l``
        and ``W_l`` fixed fractions of the output, so the total scales exactly
        with ``height * width``. Callers that need the constant should measure
        it once for their checkpoint and pass it as ``bytes_per_pixel_fp32`` to
        the constructor; this helper scales that measured constant so admission
        does not have to re-measure per resolution.
        """
        constant = self._bytes_per_pixel_fp32
        if constant is None:
            raise RuntimeError(
                "declared_state_bytes needs a measured bytes-per-pixel constant for this checkpoint; "
                "pass bytes_per_pixel_fp32= to WanStreamingDecoder() from a one-off measurement."
            )
        element_ratio = torch.empty((), dtype=dtype).element_size() / 4
        return int(constant * height * width * element_ratio)
