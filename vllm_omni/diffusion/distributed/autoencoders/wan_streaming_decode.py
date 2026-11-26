# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cross-chunk streaming decode for Wan-family causal autoencoders."""

from __future__ import annotations

import threading
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch

from vllm_omni.diffusion.distributed.autoencoders.streaming_decode import StreamingDecodeState


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

    def _execution_context(self) -> AbstractContextManager[None]:
        """The autoencoder's own execution context, when it exposes one.

        Every public compute entry point on ``OmniAutoencoderKLWan`` --
        ``encode``, ``decode``, ``decode_with_chunks``, ``tile_exec`` -- runs
        inside ``_execution_context()``, which is where autocast is entered for
        fp16/bf16 parameters (and on NPU is the only place
        ``torch.npu.amp.autocast`` is entered at all). This decoder drives
        ``post_quant_conv`` and ``decoder`` directly, so it has to enter the
        same context or streamed chunks compute under a different numeric and
        dispatch context from the whole-clip decode they are compared against.

        A plain ``diffusers`` autoencoder has no such hook, and then there is
        nothing to enter.
        """
        context = getattr(self._vae, "_execution_context", None)
        if callable(context):
            return context()
        return nullcontext()

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

    def decode_chunk(
        self,
        latent: torch.Tensor,
        state: StreamingDecodeState,
        *,
        output_format: str = "video",
        produce_output: bool = True,
    ) -> torch.Tensor:
        """Decode ``latent`` as the continuation of ``state``'s session.

        ``latent`` is ``[1, C, T, H, W]`` in latent space, already rescaled by
        the pipeline's latent statistics. Frames are decoded one at a time, the
        same loop the non-streaming path runs, with the cache carried across
        calls instead of cleared. The returned tensor is ``[1, 3, T', H', W']``
        where ``T'`` is smaller for a session's first chunk: the opening latent
        frame expands to a single raw frame and every later one to the full
        temporal factor.
        """
        if output_format not in ("video", "planar_uint8") or type(produce_output) is not bool:
            raise ValueError("Invalid streaming decoder pixel output contract")
        patch_size = getattr(getattr(self._vae, "config", None), "patch_size", None)
        if output_format == "planar_uint8" and patch_size is not None:
            raise ValueError("Planar pixel decode requires an unpatchified decoder output")
        if latent.ndim != 5:
            raise ValueError(f"latent must be [B, C, T, H, W]; got shape {tuple(latent.shape)}.")
        if latent.shape[0] != 1:
            # One state is one session's temporal context, so a batch would
            # thread several samples through a single cache. ``decode()``
            # handles a batch by decoding each sample with its own cleared
            # cache (``use_slicing``); there is no equivalent here, and the
            # streaming contract is one session per state, so refuse rather
            # than silently mix samples.
            raise ValueError(
                "streaming decode takes one sample per session state; "
                f"got batch size {latent.shape[0]}. Decode each sample through its own state."
            )
        if len(state.feat_map) != self.num_causal_convs:
            raise ValueError(
                "Decoder state does not belong to this decoder: "
                f"{len(state.feat_map)} cache slots against {self.num_causal_convs} causal convolutions."
            )
        num_frames = int(latent.shape[2])
        if num_frames == 0:
            raise ValueError("latent must carry at least one frame.")

        # Ascend-only, synchronous-path-only: reclaim the denoise loop's cached
        # allocator blocks before the decode allocates its per-frame buffers.
        # Scoped to NPU because the reclaim was only measured on 910B3; on
        # CUDA/XPU empty_cache here is untested and may regress inter-chunk
        # latency (allocator blocks are returned to the driver and re-faulted
        # on the next allocation). Skipped on background consumer threads
        # (the CausVid async path): a cross-thread allocator flush would
        # steal cached blocks from the concurrently-denoising producer thread,
        # defeating the decode/denoise overlap that path exists for. Placed
        # after input validation so rejected calls acquire no device-wide
        # side effect.
        if latent.device.type == "npu" and threading.current_thread() is threading.main_thread():
            torch.npu.empty_cache()

        decoded_frames = []
        with self._execution_context():
            for index in range(num_frames):
                state.conv_idx[0] = 0
                frame = self._vae.post_quant_conv(latent[:, :, index : index + 1])
                # first_chunk marks the session's opening frame, not the call's:
                # that is what makes chunk N + 1 continue chunk N rather than
                # restart the causal expansion.
                decoded = self._vae.decoder(
                    frame,
                    feat_cache=state.feat_map,
                    feat_idx=state.conv_idx,
                    first_chunk=(state.frames_decoded == 0),
                )
                if produce_output:
                    if output_format == "planar_uint8" and (
                        decoded.dtype != torch.uint8 or decoded.ndim != 4 or decoded.shape[0] != 3
                    ):
                        raise ValueError("Final pixel producer did not return planar uint8")
                    decoded_frames.append(decoded)
                elif decoded.numel() != 0:
                    raise ValueError("Non-output decoder rank must return an empty placeholder")
                state.frames_decoded += 1

        if not produce_output:
            state.chunks_decoded += 1
            return latent.new_empty(0)
        if output_format == "planar_uint8":
            out = torch.cat(decoded_frames, dim=1)
            state.chunks_decoded += 1
            return out
        out = torch.cat(decoded_frames, dim=2)
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
