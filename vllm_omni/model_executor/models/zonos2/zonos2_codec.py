# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Local-only DAC loading and request-local overlap-add (frozen ZONOS2)."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def shear(codes: torch.Tensor, *, up: bool, pad: int = 1025) -> torch.Tensor:
    """Same-length official shear_down/up of integer [..., T, Q] frames."""
    if codes.ndim < 2 or codes.shape[-1] != 9 or codes.dtype not in (torch.int32, torch.int64):
        raise ValueError("ZONOS2 shear requires integer [...,T,9] frames")
    out = torch.full_like(codes, pad)
    t = codes.shape[-2]
    for j in range(min(9, t)):
        if up:
            out[..., : t - j, j] = codes[..., j:, j]
        else:
            out[..., j:, j] = codes[..., : t - j, j]
    return out


def eos_boundary(codes: torch.Tensor) -> int | None:
    """First EOA, aligned by the highest EOA codebook, as in the talker."""
    hits = (codes == 1024).nonzero()
    if not len(hits):
        return None
    frame = int(hits[0, 0])
    column = int(hits[hits[:, 0] == frame, 1].max())
    return max(0, frame - column)


def local_dac_path(model_path: str | None = None) -> Path:
    explicit = os.environ.get("VLLM_ZONOS2_DAC_PATH")
    candidates = [Path(explicit)] if explicit else []
    if not explicit:
        if model_path:
            candidates.append(Path(model_path) / "dac_44khz.pth")
        candidates.append(Path.home() / ".cache/descript/dac/weights_44khz_8kbps_0.0.1.pth")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "ZONOS2 requires local DAC 44kHz 8kbps weights; set VLLM_ZONOS2_DAC_PATH "
        "to weights_44khz_8kbps_0.0.1.pth (no automatic download)"
    )


class LocalDAC:
    def __init__(self, model_path: str | None = None, device: Any = "cpu"):
        self.model_path, self.device = model_path, device
        self.codec = None

    def load(self):
        if self.codec is not None:
            return self.codec
        path = local_dac_path(self.model_path)
        try:
            import dac
        except (ImportError, OSError) as exc:
            raise ImportError("ZONOS2 DAC requires descript-audio-codec==1.0.0 and descript-audiotools==0.7.2") from exc
        # vLLM sets a model-wide default dtype; DAC's reference is float32.
        old = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            with torch.device("cpu"):
                # Reject executable module/package archives. Only tensors and
                # primitive DAC constructor metadata may cross this boundary.
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                if not isinstance(checkpoint, dict):
                    raise ValueError("DAC checkpoint must contain a tensor state dictionary")
                metadata = checkpoint.get("metadata", {})
                if not isinstance(metadata, dict):
                    raise ValueError("DAC checkpoint metadata must be a dictionary")
                kwargs = metadata.get("kwargs", {})
                if not isinstance(kwargs, dict) or not isinstance(checkpoint.get("state_dict"), dict):
                    raise ValueError("DAC checkpoint needs metadata.kwargs and state_dict")
                codec = dac.DAC(**kwargs)
                codec.load_state_dict(checkpoint["state_dict"], strict=True)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            raise RuntimeError(f"Cannot load ZONOS2 DAC checkpoint {path}: {exc}") from exc
        finally:
            torch.set_default_dtype(old)
        if codec.sample_rate != 44100 or codec.hop_length != 512 or codec.n_codebooks != 9:
            raise ValueError("ZONOS2 requires DAC sample_rate=44100, hop_length=512, n_codebooks=9")
        self.codec = codec.float().eval().requires_grad_(False).to(self.device)
        return self.codec

    @torch.inference_mode()
    def decode(self, codes_qt: torch.Tensor) -> torch.Tensor:
        if codes_qt.ndim != 2 or codes_qt.shape[0] != 9:
            raise ValueError("DAC requires [9,T] codes")
        if codes_qt.shape[1] == 0:
            return torch.empty(0, dtype=torch.float32)
        codec = self.load()
        codes = codes_qt.to(device=self.device, dtype=torch.long).clamp(0, 1023).unsqueeze(0)
        latent = codec.quantizer.from_codes(codes)[0]
        wav = codec.decode(latent).float().squeeze(0).squeeze(0).cpu()
        if wav.ndim != 1 or wav.numel() != codes_qt.shape[1] * 512 or not torch.isfinite(wav).all():
            raise RuntimeError("ZONOS2 DAC returned invalid shape or non-finite audio")
        return wav


@dataclass
class _Stream:
    frames: torch.Tensor
    decoded: int = 0
    tail: torch.Tensor | None = None
    sequence: int = -1


class DACStreamDecoder:
    """Cumulative raw-frame snapshots; duplicates are idempotent until cleanup.

    The connector owns transport buffers. This class owns only codec context
    and withheld tails, keyed by the runner request ID for cancellation.
    """

    min_chunk = 16
    overlap = 4
    hop_length = 512

    def __init__(self, decode: Callable[[torch.Tensor], torch.Tensor]):
        self.decode = decode
        self.states: dict[str, _Stream] = {}
        self.closed: set[str] = set()

    def cleanup(self, request_ids):
        for key in request_ids:
            self.states.pop(str(key), None)
            self.closed.discard(str(key))

    def push(self, key: str, frames: torch.Tensor, *, final: bool, target: int, sequence: int) -> torch.Tensor:
        empty = torch.empty(0, dtype=torch.float32)
        if key in self.closed:
            return empty
        if frames.ndim != 2 or frames.shape[1] != 9 or frames.dtype not in (torch.int32, torch.int64):
            self.cleanup([key])
            raise ValueError("DAC stream requires integer [T,9] frames")
        if not 0 <= target <= len(frames):
            self.cleanup([key])
            raise ValueError("Invalid DAC target frame count")
        state = self.states.get(key)
        if state is None:
            state = _Stream(torch.empty((0, 9), dtype=frames.dtype))
            self.states[key] = state
        if sequence < state.sequence:
            return empty
        frames = frames.detach().cpu()
        if len(frames) < len(state.frames) or not torch.equal(frames[: len(state.frames)], state.frames):
            self.cleanup([key])
            raise ValueError("DAC cumulative frame history changed or regressed")
        # The connector may reuse its snapshot storage on the next chunk.
        state.frames = frames.clone()
        state.sequence = sequence
        try:
            if target < state.decoded:
                raise ValueError("DAC EOS target regressed behind emitted frames")
            if target <= state.decoded:
                wav = state.tail if final and state.tail is not None else empty
            elif not final and target - state.decoded < self.min_chunk:
                return empty
            else:
                overlap = min(self.overlap, state.decoded)
                start = state.decoded - overlap
                end = min(target + 8, len(frames))
                aligned = shear(frames[start:end], up=True)[: target - start]
                # Own PCM before crossfade mutates it in place.
                wav = self.decode(aligned.T.contiguous()).float().cpu().clone()
                if overlap and state.tail is not None:
                    n = min(overlap * 512, len(state.tail), len(wav))
                    fade = 0.5 * (1 - torch.cos(torch.linspace(0, torch.pi, n, dtype=torch.float64)))
                    fade = fade.float()
                    wav[:n] = (1 - fade) * state.tail[-n:] + fade * wav[:n]
                state.decoded = target
                if not final:
                    n = min(4 * 512, len(wav))
                    state.tail = wav[-n:].clone()
                    wav = wav[:-n]
            if final:
                self.states.pop(key, None)
                self.closed.add(key)
            return wav
        except Exception:
            self.cleanup([key])
            raise
