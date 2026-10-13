# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Temporal arithmetic adapted from SolarWM, Copyright (c) SolarWM contributors.
# Source: Junchao-cs/SolarWM@ce1da4e7705391eda8eeda6016c0fd3f614b975e
"""Session-owned incremental decode using the released H3 temporal clips.

H3 is an overlapping clip decoder, not a causal Wan decoder. Five new latent
frames advance seventeen output frames, but the next two latent frames are
needed before that clip can be decoded. The raw five-frame overlap is blended
before the checkpoint processor reverts the pixels to the unit interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE
    from vllm_omni.diffusion.models.solarwm_h3.vae import SolarH3VideoVAE


@dataclass
class SolarH3DecodeState:
    """A single session's bounded latent lookahead and raw pixel overlap."""

    session_id: str
    max_frames: int | None = None
    pending: torch.Tensor | None = None
    overlap: torch.Tensor | None = None
    latent_frames: int = 0
    decoded_clips: int = 0
    emitted_frames: int = 0
    closed: bool = False
    failed: bool = False

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("Solar H3 decoder requires a nonempty session_id")
        if self.max_frames is not None and (
            isinstance(self.max_frames, bool) or not isinstance(self.max_frames, int) or self.max_frames < 1
        ):
            raise ValueError("Solar H3 decoder max_frames must be a positive integer")

    def nbytes(self) -> int:
        """Count owned tensor storage; buffers never alias a whole input clip."""
        return sum(t.numel() * t.element_size() for t in (self.pending, self.overlap) if t is not None)

    def close(self) -> None:
        self.pending = None
        self.overlap = None
        self.closed = True

    def reset(self) -> None:
        self.close()
        self.latent_frames = self.decoded_clips = self.emitted_frames = 0
        self.closed = self.failed = False


class SolarH3StreamingDecoder:
    """Thread explicit session state through the existing native H3 VAE adapter.

    All ranks participating in native VAE tiling must call push/flush in the same
    order, exactly as they do for complete H3 decode. The adapter's tiling context
    and ``_adaptive_decode`` retain checkpoint spatial tiling and collective rules.
    """

    def __init__(self, vae: MiniMaxH3VideoVAE | SolarH3VideoVAE, *, device: torch.device | str) -> None:
        self.vae = vae
        self.device = torch.device(device)
        model = vae.model
        expected = {
            "tokens_chunk_size": 5,
            "token_drop": 3,
            "token_overlap": 2,
            "vae_ratio_t": 4,
            "frame_pre_padding": 3,
            "frame_overlap": 5,
            "clip_length": 17,
        }
        self._diffusers = callable(getattr(model, "_decode_clip", None))
        for name, value in expected.items():
            if self._diffusers and name == "vae_ratio_t":
                actual = model.temporal_compression_ratio
            elif self._diffusers and name in {"token_drop", "clip_length"}:
                actual = getattr(model.config, name)
            else:
                actual = getattr(model, name, None)
            if actual != value:
                raise ValueError(f"Solar H3 decoder requires {name}={value}")
        if bool(getattr(model, "isolated_last_frame", False)):
            raise ValueError("Solar H3 decoder does not support an isolated final frame")
        decode_name, blend_name = ("_decode_clip", "_blend") if self._diffusers else ("_adaptive_decode", "blend")
        self._native_decode = getattr(model, decode_name, None)
        self._blend = getattr(model, blend_name, None)
        if not callable(self._native_decode) or not callable(self._blend):
            raise ValueError("Solar H3 decoder requires native temporal clip primitives")
        if bool(getattr(vae, "encode_only", False)):
            raise ValueError("Solar H3 decoder requires a decode-capable video VAE")

    def state_bytes(self, height: int, width: int, *, batch_size: int = 1) -> int:
        """Conservative persistent CUDA bytes, independent of session length.

        At return there are at most six FP32 latent frames and five raw RGB
        frames. FP32 is reserved even if the native decoder returns FP16. VAE
        working memory belongs to the per-forward scratch budget separately.
        """
        if min(height, width, batch_size) < 1:
            raise ValueError("Solar H3 decoder dimensions must be positive")
        ratio = int(self.vae.model.spatial_compression_ratio if self._diffusers else self.vae.model.vae_ratio)
        if height % ratio or width % ratio:
            raise ValueError(f"Solar H3 decoder dimensions must be divisible by {ratio}")
        channels = int(self.vae.config_dict["latent_channels"])
        return batch_size * 4 * (6 * channels * (height // ratio) * (width // ratio) + 5 * 3 * height * width)

    @staticmethod
    def _check_active(state: SolarH3DecodeState) -> None:
        if state.closed or state.failed:
            raise RuntimeError("Solar H3 decoder state is closed or failed; reset it before reuse")

    def _decode_clip(self, state: SolarH3DecodeState, clip: torch.Tensor) -> torch.Tensor:
        # Solar's official decoder denormalizes in FP32. The existing adapter
        # controls the native VAE dtype/tiling; do not replace its spatial path.
        clip = clip.to(device=self.device, dtype=torch.float32)
        with self.vae._decode_tiling_context(clip):
            decoded = self._native_decode(self.vae._denormalize_latent(clip))
        if decoded.ndim != 5 or decoded.shape[2] != clip.shape[2] * 4:
            raise RuntimeError("Solar H3 native VAE returned an incompatible temporal clip")
        part = decoded[:, :, 3:20]
        if state.overlap is not None:
            part = self._blend(state.overlap, part, 5, dim=-3)
        # clone prevents a five-frame view from retaining the complete decode.
        state.overlap = decoded[:, :, 23:40].clone()
        state.decoded_clips += 1
        return part

    def _publish(self, state: SolarH3DecodeState, parts: list[torch.Tensor]) -> torch.Tensor | None:
        if not parts:
            return None
        raw = torch.cat(parts, dim=2) if len(parts) > 1 else parts[0]
        count = int(raw.shape[2])
        if state.max_frames is not None:
            count = min(count, max(0, state.max_frames - state.emitted_frames))
        if count == 0:
            return None
        raw = raw[:, :, :count]
        if self._diffusers:
            frames = self.vae._normalize_decoded_frames(raw).contiguous()
        else:
            processor = getattr(self.vae.model, "processor", None)
            decoded = processor.revert_tensor(raw) if processor is not None else raw
            frames = self.vae._normalize_decoded_frames(decoded).contiguous()
        state.emitted_frames += count
        return frames

    @torch.inference_mode()
    def push(self, state: SolarH3DecodeState, normalized_latents: torch.Tensor) -> torch.Tensor | None:
        """Append generated latents and return completed frames, if any.

        The first five-latent block produces no output; the next block makes
        the first seven-latent decode window available and emits 17 frames.
        """
        self._check_active(state)
        try:
            channels = int(self.vae.config_dict["latent_channels"])
            if normalized_latents.ndim != 5 or normalized_latents.shape[1] != channels:
                raise ValueError(f"Solar H3 decoder expects [B,{channels},T,H,W] latents")
            if min(normalized_latents.shape) < 1:
                raise ValueError("Solar H3 decoder cannot append an empty latent tensor")
            latents = normalized_latents.to(device=self.device, dtype=torch.float32)
            if state.pending is not None:
                if state.pending.shape[:2] + state.pending.shape[3:] != latents.shape[:2] + latents.shape[3:]:
                    raise ValueError("Solar H3 decoder latent geometry changed within a session")
                latents = torch.cat((state.pending, latents), dim=2)
            state.latent_frames += int(normalized_latents.shape[2])
            parts = []
            while latents.shape[2] >= 7:
                parts.append(self._decode_clip(state, latents[:, :, :7]))
                latents = latents[:, :, 5:]
            state.pending = latents.clone()
            return self._publish(state, parts)
        except Exception:
            state.failed = True
            state.close()
            raise

    @torch.inference_mode()
    def flush(self, state: SolarH3DecodeState) -> torch.Tensor | None:
        """Finish the official padded final clip, trim padding, and close state."""
        self._check_active(state)
        try:
            if state.pending is None:
                raise ValueError("Solar H3 decoder cannot flush an empty session")
            padding = (-(state.latent_frames + 3)) % 5
            count = (state.latent_frames + 3 + padding) // 5 - 1
            if count < 1:
                raise ValueError("Solar H3 decoder needs at least one complete temporal clip")
            latents = state.pending
            if padding:
                latents = torch.cat((latents, latents[:, :, -1:].repeat(1, 1, padding, 1, 1)), dim=2)
            parts = []
            while state.decoded_clips < count:
                parts.append(self._decode_clip(state, latents[:, :, :7]))
                latents = latents[:, :, 5:]
            if state.overlap is not None:
                parts.append(state.overlap)
            raw = torch.cat(parts, dim=2) if parts else None
            trim = sum(1 if (state.latent_frames + index) % 5 == 0 else 4 for index in range(padding))
            if raw is not None and trim:
                raw = raw[:, :, :-trim]
            result = self._publish(state, [raw] if raw is not None else [])
            state.close()
            return result
        except Exception:
            state.failed = True
            state.close()
            raise
