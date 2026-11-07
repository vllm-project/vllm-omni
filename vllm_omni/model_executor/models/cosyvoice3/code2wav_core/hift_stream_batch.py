# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Batched streaming HiFT over request-owned bounded mel windows.

The F0 predictor reads ``trim`` frames ahead and the decoder
``conv_pre_look_right`` frames ahead; every other layer, the oscillator phase
and the position-indexed noise are causal. Windows batch in three layouts:

* right aligned: a settled window (full overlap reaching past the decoder's
  receptive field, full F0 margin, carried phase) emits nothing that depends
  on frames left of its F0 input, so rows end together and the left padding
  never reaches what they emit. Final and non-final windows share these
  batches; a final one emits up to the common right edge.
* left aligned: an early non-final window (still at the stream's start or
  short of history) keeps its left edge; it emits only samples before both
  look-ahead spans and the ISTFT frame held back, so right padding never
  reaches them.
* exact: an early final window emits its tail and needs its own left edge,
  so it runs alone at its own width.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .packed_dit import to_device_async

# Batches pad to a row slot and a frame step so convolution shapes recur:
# cuDNN reuses its plans and loaded kernels instead of building them for each
# new rows-by-width pair, and decode graphs, when on, stay few.
_ROW_SLOTS = (1, 2, 4, 8, 16)
_FRAME_STEP = 64
# Mel frames a decoded sample depends on to its left: CosyVoice3's HiFT reaches
# 29 (conv_pre, upsampling and dilated resblocks); a test measures it.
DECODE_RECEPTIVE_FRAMES = 32


@dataclass
class HiftWindow:
    """One request's streaming HiFT input and the samples it releases."""

    mel: torch.Tensor  # (1, mels, frames): F0 input, its context margin first
    margin: int  # F0 context frames before the decoded window
    phase_acc: torch.Tensor | None  # (1, 1, harmonics) phase carried into this window
    noise_offset: int  # absolute sample index of the decoded window's start
    next_overlap: int  # frames the next window re-reads; the phase carry point
    emit_start: int  # emitted sample range of the decoded window
    emit_end: int
    final: bool  # emits its tail, held-back frames included
    settled: bool  # full overlap, full F0 margin and a carried phase


@contextmanager
def _tf32_disabled():
    """F0 runs in FP32 without TF32, as in CausalHiFTGenerator.inference."""
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


class StreamingHiFTBatcher:
    """Run the streaming HiFT step of several requests as one call.

    Mirrors ``CausalHiFTGenerator.inference`` with ``SineGen2``: F0, the
    harmonic source and the decoder take every row at once; per-row margins,
    noise offsets and phase carries become index tensors instead of host
    loops. Device copies of the fixed noise tables are made on first use.
    """

    def __init__(self, hift: torch.nn.Module) -> None:
        self.hift = hift
        self.scale = int(hift.m_source.l_sin_gen.upsample_scale)
        self.trim = int(hift.f0_predictor.condnet[0].causal_padding)
        self.look_right = int(hift.conv_pre_look_right)
        # A window emits from this many frames before its end (non-final) or
        # this many plus its tail (final); its overlap must also cover the
        # decoder's reach before the first emitted frame.
        self.held_back = self.trim + self.look_right + 1
        self.settled_overlap = DECODE_RECEPTIVE_FRAMES + self.held_back
        self._noise: torch.Tensor | None = None
        self._rand_ini: torch.Tensor | None = None

    def _tables(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._noise is None or self._noise.device != device:
            sine_gen = self.hift.m_source.l_sin_gen
            self._noise = sine_gen.sine_waves[0].to(device)
            self._rand_ini = sine_gen.rand_ini.to(device)
        return self._noise, self._rand_ini

    @staticmethod
    def layout(window: HiftWindow) -> str:
        """The batch layout a window needs: "right", "left" or "exact"."""
        if window.settled:
            return "right"
        if window.final:
            return "exact"
        return "left"

    @torch.inference_mode()
    def run(self, windows: list[HiftWindow], *, layout: str) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
        """Return each window's emitted speech ``(1, 1, samples)`` and the
        phase carried into its next window (None for final windows)."""
        if any(self.layout(window) != layout for window in windows):
            raise ValueError(f"Every window of a {layout} batch must need that layout")
        if layout == "exact" and len(windows) != 1:
            raise ValueError("An early final window emits its tail and runs alone at exact width")
        widths = [window.mel.shape[-1] for window in windows]
        rows, width = len(windows), max(widths)
        if layout != "exact":
            rows = next(slots for slots in _ROW_SLOTS if slots >= rows)
            width = -(-width // _FRAME_STEP) * _FRAME_STEP
        first = windows[0].mel
        device = first.device
        mel = first.new_zeros((rows, first.shape[1], width))
        lefts = [width - frames if layout == "right" else 0 for frames in widths]
        for row, window in enumerate(windows):
            mel[row, :, lefts[row] : lefts[row] + widths[row]] = window.mel[0]
        with _tf32_disabled():
            f0 = self.hift.f0_predictor(mel.float(), finalize=True).to(mel.dtype)
        margins = [window.margin for window in windows] + [0] * (rows - len(windows))
        # Frame of each row's decoded window start within the batch.
        origins = [0] * rows
        if layout == "exact":
            f0, mel = f0[:, margins[0] :], mel[:, :, margins[0] :]
        elif layout == "left":
            if any(margins):
                index = torch.arange(width).unsqueeze(0) + torch.tensor(margins).unsqueeze(1)
                index = to_device_async(index.clamp_(max=width - 1), device)
                f0 = f0.gather(1, index)
                mel = mel.gather(2, index.unsqueeze(1).expand_as(mel))
        else:
            origins = [left + margin for left, margin in zip(lefts, margins)] + [width] * (rows - len(windows))
            # Zero F0 before each window: no phase accrues over the padding and
            # the F0 margin, so the window's phase starts from its carry.
            starts = to_device_async(torch.tensor(origins), device)
            f0 = f0 * (torch.arange(width, device=device).unsqueeze(0) >= starts.unsqueeze(1))
        source, phase = self._source(f0, windows, origins)
        speech = self.hift.decode(x=mel, s=source, finalize=True)
        speeches = []
        for row, window in enumerate(windows):
            start, end = (origins[row] * self.scale + sample for sample in (window.emit_start, window.emit_end))
            speeches.append(speech[row : row + 1, start:end].reshape(1, 1, -1))
        return speeches, self._carry(phase, windows, widths, origins)

    def capture_decode_graphs(self, max_frames: int) -> int:
        """Record the decoder graph of every padded batch up to ``max_frames``
        frames wide; returns how many. Does nothing without decode graphs."""
        graphs = getattr(self.hift, "_decode_graphs", None)
        if graphs is None:
            return 0
        weight = self.hift.m_source.l_linear.weight
        mels = self.hift.conv_pre.in_channels
        count = 0
        for rows in _ROW_SLOTS:
            for width in range(_FRAME_STEP, max_frames + _FRAME_STEP, _FRAME_STEP):
                mel = weight.new_zeros((rows, mels, width))
                source = weight.new_zeros((rows, 1, width * self.scale))
                graphs.capture(mel, source, finalize=True)
                count += 1
        return count

    def warm_fft_plans(self, max_frames: int) -> int:
        """Decode once at every width the streaming step can meet up to
        ``max_frames`` frames; returns how many decodes ran.

        The decoder's STFT and inverse FFT need a cuFFT plan per input size.
        Creating one loads a module, which waits for the GPU's queued work:
        behind a busy step's Flow graphs that took up to ~190 ms, stalling the
        whole step. Early final windows and the per-request path decode one row
        at their own width; batches use the padded row-slot widths. Created
        here on an idle GPU, each plan costs well under a millisecond.
        """
        weight = self.hift.m_source.l_linear.weight
        mels = self.hift.conv_pre.in_channels
        shapes = [(1, width) for width in range(1, max_frames + 1)]
        shapes += [
            (rows, width) for rows in _ROW_SLOTS[1:] for width in range(_FRAME_STEP, max_frames + 1, _FRAME_STEP)
        ]
        for rows, width in shapes:
            mel = weight.new_zeros((rows, mels, width))
            source = weight.new_zeros((rows, 1, width * self.scale))
            self.hift.decode(x=mel, s=source, finalize=True)
        return len(shapes)

    def _source(
        self, f0: torch.Tensor, windows: list[HiftWindow], origins: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """SineGen2 + SourceModuleHnNSF for every row; returns the source and
        the frame-rate phase (mod-1 cycles, carry already added)."""
        m_source = self.hift.m_source
        sine_gen = m_source.l_sin_gen
        rows = f0.shape[0]
        noise_table, rand_ini = self._tables(f0.device)
        upsampled = self.hift.f0_upsamp(f0[:, None]).transpose(1, 2)
        rad = (upsampled * sine_gen.harmonic_ids / sine_gen.sampling_rate) % 1
        fresh = [window.phase_acc is None for window in windows] + [True] * (rows - len(windows))
        if any(fresh):
            # A stream's first window starts at a random harmonic phase; only
            # left and exact layouts hold such windows, which start at frame 0.
            mask = to_device_async(torch.tensor(fresh, dtype=rad.dtype), f0.device)
            rad[:, 0, :] = rad[:, 0, :] + mask[:, None] * rand_ini
        rad = F.interpolate(rad.transpose(1, 2), scale_factor=1 / self.scale, mode="linear").transpose(1, 2)
        carried = [
            rad.new_zeros((1, 1, rad.shape[-1])) if window.phase_acc is None else window.phase_acc for window in windows
        ]
        carried += [rad.new_zeros((1, 1, rad.shape[-1]))] * (rows - len(windows))
        phase = torch.cumsum(rad, dim=1) + torch.cat(carried)
        angle = phase * 2 * np.pi
        angle = F.interpolate(angle.transpose(1, 2) * self.scale, scale_factor=self.scale, mode="nearest")
        sine_waves = torch.sin(angle.transpose(1, 2)) * sine_gen.sine_amp
        uv = (upsampled > sine_gen.voiced_threshold).type(torch.float32)
        noise_amp = uv * sine_gen.noise_std + (1 - uv) * sine_gen.sine_amp / 3
        # Noise is indexed by absolute sample: shift each row by its window's origin.
        offsets = [window.noise_offset - origin * self.scale for window, origin in zip(windows, origins)]
        offsets += [0] * (rows - len(windows))
        offsets = to_device_async(torch.tensor(offsets, dtype=torch.long), f0.device)
        positions = (offsets[:, None] + torch.arange(sine_waves.shape[1], device=f0.device)) % noise_table.shape[0]
        sine_waves = sine_waves * uv + noise_amp * noise_table[positions]
        source = m_source.l_tanh(m_source.l_linear(sine_waves)).transpose(1, 2)
        return source, phase

    def _carry(
        self, phase: torch.Tensor, windows: list[HiftWindow], widths: list[int], origins: list[int]
    ) -> list[torch.Tensor | None]:
        """Phase at the frame before the next window's start, as
        ``_carry_phase_at_boundary`` takes it from a non-final window."""
        carries: list[torch.Tensor | None] = []
        for row, (window, frames) in enumerate(zip(windows, widths)):
            decoded = frames - window.margin
            if window.final:
                carries.append(None)
            elif window.next_overlap >= decoded:
                carries.append(window.phase_acc)
            else:
                index = origins[row] + decoded - window.next_overlap - 1
                carries.append(phase[row : row + 1, index : index + 1] % 1)
        return carries

    def emit_range(self, decoded_frames: int, new_frames: int, *, finalize: bool) -> tuple[int, int]:
        """Samples a window of ``decoded_frames`` releases for ``new_frames``
        new mel frames, matching ``_stream_hift_from_feat``."""
        samples_per_frame = self.scale
        if finalize:
            end = decoded_frames * samples_per_frame
            released = (new_frames + self.held_back) * samples_per_frame
        else:
            end = (decoded_frames - self.held_back) * samples_per_frame
            released = new_frames * samples_per_frame
        return max(0, end - released), end
