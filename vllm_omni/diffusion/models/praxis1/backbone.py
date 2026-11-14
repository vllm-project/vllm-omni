# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Architecture adapted from https://github.com/runwayml/praxis-1-oss (Praxis-1).
"""Two-tower video backbone of the chunk-causal window.

The encoder sees ``[registers, anchor frame, text, K noised chunks, K - 1 clean
chunks]`` and returns per-token conditioning plus the taps the action expert
reads. The decoder denoises the noised chunks under that conditioning.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_omni.diffusion.models.praxis1.layers import (
    MaskCache,
    SparseMask,
    TimestepEmbedding,
    Transformer,
    VideoInputProjection,
    ZeroCenteredRMSNorm,
    modulate,
    pad_tags,
)

ROW_STRIDE = 360
FRAME_STRIDE = 129600
TEXT = "TextEncodingTensor"
VIDEO = "VideoTensor"


@dataclass
class TapFeatures:
    """Encoder features of one chunk-causal call for the action expert."""

    taps: list[Tensor]
    embeddings: Tensor
    frames_per_chunk: int
    tokens_per_frame: int
    rope_ids: Tensor
    noised_span: tuple[int, int]
    clean_span: tuple[int, int] | None


@dataclass
class BackboneOutput:
    """Video velocity ``(1, C, T, h, w)`` (None if not decoded) and taps."""

    velocity: Tensor | None
    taps: TapFeatures


def video_positions(frames: int, height: int, width: int, start: int) -> Tensor:
    """Return the rotary positions of one video slot in frame-major order."""
    frame = torch.arange(frames).view(-1, 1, 1) * FRAME_STRIDE
    row = torch.arange(height).view(1, -1, 1) * ROW_STRIDE
    col = torch.arange(width).view(1, 1, -1)
    return (start + 1 + frame + row + col).reshape(-1)


def chunk_causal_positions(
    num_registers: int,
    anchor_hw: tuple[int, int],
    num_text: int,
    noised_thw: tuple[int, int, int],
    num_clean_frames: int,
) -> Tensor:
    """Return the encoder rotary positions of the chunk-causal layout."""
    regs = torch.zeros(num_registers, dtype=torch.int64)
    anchor = video_positions(1, *anchor_hw, start=0)
    text = torch.arange(1, num_text + 1) + FRAME_STRIDE
    noised = video_positions(*noised_thw, start=2 * FRAME_STRIDE)
    tokens_per_frame = noised_thw[1] * noised_thw[2]
    clean = noised[: num_clean_frames * tokens_per_frame]
    return torch.cat([regs, anchor, text, noised, clean])


def _chunk_rows(rows: Tensor, chunk_tokens: Sequence[int]) -> Tensor:
    """Expand row k of ``rows`` (1, K, C) over the ``chunk_tokens[k]`` tokens."""
    return torch.cat([rows[:, k : k + 1].expand(-1, n, -1) for k, n in enumerate(chunk_tokens)], dim=1)


def unpatch(x: Tensor, patch: tuple[int, int]) -> Tensor:
    """Move the channel patch of ``(B, C*ph*pw, T, h, w)`` into space."""
    b, c, t, h, w = x.shape
    ph, pw = patch
    x = x.view(b, c // (ph * pw), ph, pw, t, h, w)
    return x.permute(0, 1, 4, 5, 2, 6, 3).reshape(b, -1, t, h * ph, w * pw)


def patch(x: Tensor, patch: tuple[int, int]) -> Tensor:
    """Move ``(ph, pw)`` spatial patches of ``(B, C, T, H, W)`` into channels."""
    b, c, t, hh, ww = x.shape
    ph, pw = patch
    x = x.view(b, c, t, hh // ph, ph, ww // pw, pw)
    x = x.permute(0, 1, 4, 6, 2, 3, 5)
    return x.reshape(b, c * ph * pw, t, hh // ph, ww // pw)


def observed_context_mask(
    globals_: int, noised: int, clean: int, tokens_per_frame: int, chunk_size: int, device: torch.device
) -> SparseMask:
    """Visibility of the ``[global, noised video, clean video]`` token layout.

    Global tokens are visible to everyone and see only each other. A noised
    chunk sees itself and the clean chunks before it; a clean chunk sees the
    clean chunks up to and including itself.
    """
    length = globals_ + (noised + clean) * tokens_per_frame
    kind = pad_tags(
        torch.cat(
            [
                torch.zeros(globals_, dtype=torch.int32),
                torch.ones(noised * tokens_per_frame, dtype=torch.int32),
                torch.full((clean * tokens_per_frame,), 2, dtype=torch.int32),
            ]
        ),
        length,
    ).to(device)
    chunk = pad_tags(
        torch.cat(
            [
                torch.full((globals_,), -1, dtype=torch.int32),
                torch.arange(noised, dtype=torch.int32).repeat_interleave(tokens_per_frame) // chunk_size,
                torch.arange(clean, dtype=torch.int32).repeat_interleave(tokens_per_frame) // chunk_size,
            ]
        ),
        length,
    ).to(device)

    def mask_mod(b: Tensor | None, h: Tensor | None, q: Tensor, k: Tensor) -> Tensor:
        q_kind, k_kind, q_chunk, k_chunk = kind[q], kind[k], chunk[q], chunk[k]
        return (
            (k_kind == 0)
            | ((q_kind == 1) & (k_kind == 1) & (q_chunk == k_chunk))
            | ((q_kind == 1) & (k_kind == 2) & (k_chunk < q_chunk))
            | ((q_kind == 2) & (k_kind == 2) & (k_chunk <= q_chunk))
        )

    return SparseMask(mask_mod, length, length, device)


class EncoderTower(nn.Module):
    """Encoder tower of the backbone."""

    def __init__(self, cfg: dict, attention_impl: str) -> None:
        super().__init__()
        tower, dim = cfg["encoder"], cfg["dim"]
        time_dim = cfg["timestep_dim"]
        self.registers = nn.Parameter(torch.zeros(1, cfg["num_registers"], dim))
        self.t_embedding = TimestepEmbedding(time_dim)
        self.t_proj = nn.ModuleDict({VIDEO: ReplicatedLinear(time_dim, 2 * dim, return_bias=False)})
        self.input_projections = nn.ModuleDict(
            {
                TEXT: nn.Sequential(
                    ReplicatedLinear(tower["input_projections"][TEXT]["c_in"], 4 * dim, return_bias=False),
                    nn.GELU(),
                    ReplicatedLinear(4 * dim, dim, return_bias=False),
                    ZeroCenteredRMSNorm(dim, fp32=False),
                ),
                VIDEO: VideoInputProjection(tower["input_projections"][VIDEO]["c_in"], dim),
            }
        )
        self.output_projections = nn.ModuleDict({VIDEO: ReplicatedLinear(dim, dim, return_bias=False)})
        self.core = Transformer(cfg, tower, "transformer.backbone.encoder.core", attention_impl)
        self.tap_blocks = tuple(tower["tap_blocks"])
        self._masks = MaskCache()

    def _chunk_modulate(self, flat: Tensor, t_emb: Tensor, chunk_tokens: Sequence[int]) -> Tensor:
        """Modulate the tokens of each chunk with its timestep row."""
        shift, scale = self.t_proj[VIDEO](t_emb).chunk(2, dim=-1)
        return modulate(flat, _chunk_rows(shift, chunk_tokens), _chunk_rows(1 + scale, chunk_tokens))

    def forward_chunk_causal(
        self,
        text: Tensor,
        anchor: Tensor,
        clean: Tensor | None,
        noised: Tensor,
        timesteps: Tensor,
    ) -> tuple[Tensor, TapFeatures]:
        """Encode a chunk-causal window of K chunks."""
        kk = int(timesteps.shape[1])
        frames, height, width = (int(d) for d in noised.shape[2:])
        if frames % kk:
            raise ValueError(f"{frames} noised frames do not split into {kk} chunks")
        chunk = frames // kk
        n_clean_frames = 0 if clean is None else int(clean.shape[2])
        if n_clean_frames != (kk - 1) * chunk:
            raise ValueError(f"{kk} chunks need {(kk - 1) * chunk} clean frames, got {n_clean_frames}")
        tpf = height * width
        anchor_tok = self.input_projections[VIDEO](anchor).flatten(1, -2)
        text_tok = self.input_projections[TEXT](text)
        noised_tok = self.input_projections[VIDEO](noised)
        noised_shape = noised_tok.shape
        noised_tok = self._chunk_modulate(noised_tok.flatten(1, -2), self.t_embedding(timesteps), [chunk * tpf] * kk)
        parts = [self.registers, anchor_tok, text_tok, noised_tok]
        if clean is not None:
            clean_tok = self.input_projections[VIDEO](clean).flatten(1, -2)
            t_clean = self.t_embedding(torch.zeros_like(timesteps))
            parts.append(self._chunk_modulate(clean_tok, t_clean, [chunk * tpf] * (kk - 1)))
        x = torch.cat(parts, dim=1)
        n_reg, n_anchor, n_text = self.registers.size(1), anchor_tok.size(1), text_tok.size(1)
        n_noised, n_clean = frames * tpf, n_clean_frames * tpf
        n_global = n_reg + n_anchor + n_text
        positions = chunk_causal_positions(
            n_reg, (int(anchor.shape[3]), int(anchor.shape[4])), n_text, (frames, height, width), n_clean_frames
        )
        positions = positions.unsqueeze(0).to(x.device)
        key = (n_global, frames, n_clean_frames, tpf, chunk, str(x.device))
        mask = self._masks.get(
            key, lambda: observed_context_mask(n_global, frames, n_clean_frames, tpf, chunk, x.device)
        )
        assert isinstance(mask, SparseMask)
        out, taps = self.core(x, positions, None, mask, self.tap_blocks)
        projected = self.output_projections[VIDEO](out.narrow(1, n_global, n_noised + n_clean))
        cond = projected.narrow(1, 0, n_noised).view(*noised_shape[:-1], -1)
        features = TapFeatures(
            taps=list(taps),
            embeddings=x,
            frames_per_chunk=chunk,
            tokens_per_frame=tpf,
            rope_ids=positions,
            noised_span=(n_global, n_noised),
            clean_span=(n_global + n_noised, n_clean) if n_clean else None,
        )
        return cond, features


class DecoderTower(nn.Module):
    """Decoder tower of the backbone."""

    def __init__(self, cfg: dict, attention_impl: str) -> None:
        super().__init__()
        tower, dim = cfg["decoder"], cfg["dim"]
        video_dim, time_dim = tower["input_projections"][VIDEO]["c_in"], cfg["timestep_dim"]
        self.registers = nn.Parameter(torch.zeros(1, cfg["num_registers"], dim))
        self.t_embedding = TimestepEmbedding(time_dim)
        self.t_proj = nn.ModuleDict({VIDEO: ReplicatedLinear(time_dim, 2 * dim, return_bias=False)})
        self.input_projections = nn.ModuleDict({VIDEO: VideoInputProjection(video_dim, dim)})
        self.output_projections = nn.ModuleDict(
            {
                VIDEO: nn.Sequential(
                    ZeroCenteredRMSNorm(dim, fp32=False), ReplicatedLinear(dim, video_dim, return_bias=False)
                )
            }
        )
        self.core = Transformer(cfg, tower, "transformer.backbone.decoder.core", attention_impl)
        self.patch_size = tuple(tower["unpatch_factor"][1:])
        self._masks = MaskCache()

    def forward_chunk_causal(self, noised: Tensor, cond: Tensor, timesteps: Tensor) -> Tensor:
        """Return the velocity of the K noised chunks of a chunk-causal window."""
        ph, pw = self.patch_size
        kk = int(timesteps.shape[1])
        tok = self.input_projections[VIDEO](unpatch(noised, self.patch_size))
        shape = tok.shape
        _, frames, height, width, _ = shape
        if frames % kk:
            raise ValueError(f"{frames} noised frames do not split into {kk} chunks")
        chunk = frames // kk
        tpf = height * width
        shift, scale = self.t_proj[VIDEO](self.t_embedding(timesteps)).chunk(2, dim=-1)
        rows = [chunk * tpf] * kk
        flat = modulate(tok.flatten(1, -2), _chunk_rows(shift, rows), _chunk_rows(1 + scale, rows))
        x = torch.cat([self.registers, flat], dim=1)
        b, cf, h, w, c = cond.shape
        up = cond[:, :, :, None, :, None, :].expand(b, cf, h, ph, w, pw, c).reshape(b, cf * h * ph * w * pw, c)
        n_reg = self.registers.size(1)
        modulation = torch.cat([torch.zeros(b, n_reg, c, device=up.device, dtype=up.dtype), up], dim=1)
        positions = torch.cat([torch.zeros(n_reg, dtype=torch.int64), video_positions(frames, height, width, 0)])
        positions = positions.unsqueeze(0).to(x.device)
        key = (n_reg, frames, 0, tpf, chunk, str(x.device))
        mask = self._masks.get(key, lambda: observed_context_mask(n_reg, frames, 0, tpf, chunk, x.device))
        assert isinstance(mask, SparseMask)
        out, _ = self.core(x, positions, modulation, mask)
        v = self.output_projections[VIDEO](out.narrow(1, n_reg, flat.size(1)))
        return patch(v.view(*shape[:-1], -1).movedim(-1, 1), self.patch_size)


class VideoBackbone(nn.Module):
    """Encoder tower and decoder tower for one chunk-causal denoise step."""

    def __init__(self, cfg: dict, attention_impl: str = "flex") -> None:
        super().__init__()
        self.encoder = EncoderTower(cfg, attention_impl)
        self.decoder = DecoderTower(cfg, attention_impl)

    def forward_chunk_causal(
        self,
        text: Tensor,
        anchor: Tensor,
        clean: Tensor | None,
        noised: Tensor,
        timesteps: Tensor,
        decode: bool = True,
    ) -> BackboneOutput:
        """Run one denoise step of a chunk-causal window of K chunks."""
        cond, taps = self.encoder.forward_chunk_causal(text, anchor, clean, noised, timesteps)
        if not decode:
            return BackboneOutput(velocity=None, taps=taps)
        return BackboneOutput(velocity=self.decoder.forward_chunk_causal(noised, cond, timesteps), taps=taps)
