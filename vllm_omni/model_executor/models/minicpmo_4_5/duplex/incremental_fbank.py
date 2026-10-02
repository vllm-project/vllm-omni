# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Incremental Stage-0 streaming fbank (HF override ``duplex_incremental_fbank``, default off).

The remote-code ``StreamingMelProcessorExact._extract_full`` runs the Whisper
log-mel front end (``MiniCPMAAudioProcessor._torch_extract_fbank_features``,
on the CPU) over the session's whole streaming buffer, up to its 30 s slide
trigger, on every unit, although only the newest unit's frames change. The
incremental version keeps the power spectrum ``|STFT|^2`` of the frames it
already computed and recomputes only the right edge (the frames whose window
reaches the new samples or the reflect pad) and, after a slide, the new left
edge. The mel projection, clamp, log10, dynamic range (global max) and
scaling still run over the whole buffer with the original's shapes and memory
layout, so the features are bitwise equal to ``_extract_full``:

* The mel projection is not row independent: MKL computes the rows of a
  partial register block with another kernel, so the original itself gives
  one frame different mel values at different buffer lengths. It is not
  cached; it is ~10% of the original's time.
* The STFT and ``abs() ** 2`` are frame independent, except that a batch's
  last frames or elements may take a remainder path. A recompute therefore
  starts on a multiple of ``_ALIGN_FRAMES`` (its remainder then lines up
  with the original's) and its last ``_ALIGN_FRAMES`` frames are not kept.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# Recompute start granularity and the trailing frames of a recompute that are
# not kept (>= the widest remainder of the FFT batch and the abs() vector loop);
# also the left-edge recompute after a slide (>= the 2 frames reading the pad).
_ALIGN_FRAMES = 16
_FEATURE_EXTRACTOR_ATTRS = (
    "n_fft",
    "hop_length",
    "n_samples",
    "sampling_rate",
    "mel_filters",
    "set_spac_log_norm",
    "dynamic_log_norm",
    "dynamic_range_db",
    "log_floor_db",
)
_INCREMENTAL_CLASSES: dict[type, type] = {}
_SELF_CHECKED: dict[tuple[type, type], bool] = {}


def _stft_power(
    buffer: np.ndarray,
    start: int,
    stop: int,
    n_fft: int,
    hop: int,
    window: torch.Tensor,
) -> np.ndarray:
    """``|STFT|^2`` of frames ``[start, stop)`` of ``buffer``, framed as the original's centered STFT.

    The batch is ``[start, stop]`` with its last frame dropped, like the
    original's ``stft(...)[..., :-1].abs() ** 2``. The reflect padding at a
    buffer edge is built explicitly (``np.pad`` reflect is ``torch.stft``'s).
    Returns ``(stop - start, n_fft // 2 + 1)`` rows.
    """
    half = n_fft // 2
    lo, hi = start * hop - half, stop * hop + half
    length = len(buffer)
    segment = buffer[max(lo, 0) : min(hi, length)]
    if lo < 0 or hi > length:
        segment = np.pad(segment, (max(0, -lo), max(0, hi - length)), mode="reflect")
    spec = torch.stft(
        torch.from_numpy(segment)[None],
        n_fft=n_fft,
        hop_length=hop,
        window=window,
        center=False,
        return_complex=True,
    )
    magnitudes = spec[..., :-1].abs() ** 2  # [1, n_freq, stop - start], frame-major in memory
    return magnitudes[0].T.numpy()


def _log_mel(power: torch.Tensor, mel_filters: torch.Tensor, feature_extractor: Any) -> torch.Tensor:
    """``_torch_extract_fbank_features`` from the ``|STFT|^2`` rows, op for op and layout for layout."""
    magnitudes = power.unsqueeze(0).transpose(1, 2)  # [1, n_freq, T], strides of stft(...)[..., :-1]
    mel_spec = mel_filters.T @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    if feature_extractor.dynamic_log_norm:
        max_val_t = log_spec.max(dim=2, keepdim=True)[0]
        max_val_bt = max_val_t.max(dim=1, keepdim=True)[0]
        log_spec = torch.maximum(log_spec, max_val_bt - feature_extractor.dynamic_range_db)
    else:
        floor_tensor = torch.tensor(feature_extractor.log_floor_db, dtype=log_spec.dtype)
        log_spec = torch.maximum(log_spec, floor_tensor)
    return (log_spec + 4.0) / 4.0


class _IncrementalFbankMixin:
    """Overrides of the remote ``StreamingMelProcessorExact``, mixed in by ``enable_incremental_fbank``."""

    #: ``|STFT|^2`` rows, frame t of the current buffer at row t; rows
    #: ``[0, _ifb_valid)`` are exact for the current buffer.
    _ifb_power: np.ndarray | None = None
    _ifb_valid = 0
    #: ``left_samples_dropped`` and the buffer length when the rows were kept.
    _ifb_dropped = 0
    _ifb_length = 0
    #: The ``n_fft`` samples ending where the last kept frame's window ends.
    _ifb_guard: np.ndarray | None = None
    _ifb_window: torch.Tensor | None = None
    _ifb_mel_filters: torch.Tensor | None = None

    def reset(self) -> None:
        super().reset()
        self._ifb_valid = 0

    def restore_snapshot(self, snapshot: dict) -> None:
        super().restore_snapshot(snapshot)
        self._ifb_valid = 0

    def _extract_full(self) -> torch.Tensor:
        fe = self.feature_extractor
        buffer = self.buffer
        if not (
            isinstance(buffer, np.ndarray)
            and buffer.ndim == 1
            and buffer.dtype == np.float32
            and fe.n_fft <= len(buffer) <= fe.n_samples
            and getattr(fe, "dither", 0.0) == 0.0
            and fe.sampling_rate == self.sample_rate
        ):
            # Too short (the original raises), truncated, dithered or misconfigured: the original.
            self._ifb_valid = 0
            return super()._extract_full()
        # The original's log-floor switch, side effect on the feature extractor included.
        if len(buffer) < 5 * self.sample_rate:
            fe.set_spac_log_norm(log_floor_db=-10)
        else:
            fe.set_spac_log_norm(dynamic_range_db=8)
        return _log_mel(self._ifb_update_power(buffer, fe), self._ifb_mel_filters, fe)

    def _ifb_update_power(self, buffer: np.ndarray, fe: Any) -> torch.Tensor:
        n_fft, hop = int(fe.n_fft), int(fe.hop_length)
        if self._ifb_window is None:
            self._ifb_window = torch.hann_window(n_fft)
            self._ifb_mel_filters = torch.from_numpy(fe.mel_filters).to(torch.float32)
        power = self._ifb_power
        if power is None or power.ctypes.data % 64:
            # Rows from a torch allocation: 64-byte aligned like the original's fresh
            # |X|^2 tensor, which the GEMM reads (a deepcopy's numpy array may not be).
            aligned = torch.empty(int(fe.n_samples) // hop + 1, n_fft // 2 + 1, dtype=torch.float32).numpy()
            if power is not None:
                aligned[: self._ifb_valid] = power[: self._ifb_valid]
            self._ifb_power = power = aligned
        frames = len(buffer) // hop  # the original drops the STFT's last frame
        valid = self._ifb_rebase(buffer, n_fft, hop)
        start = valid // _ALIGN_FRAMES * _ALIGN_FRAMES
        power[start:frames] = _stft_power(buffer, start, frames, n_fft, hop, self._ifb_window)
        # Keep frames whose window lies inside the buffer, minus the recompute's tail.
        stable = (len(buffer) - n_fft // 2) // hop + 1
        valid = min(stable, frames - _ALIGN_FRAMES)
        guard_end = (valid - 1) * hop + n_fft // 2
        if guard_end < n_fft:
            valid = 0
        else:
            self._ifb_guard = buffer[guard_end - n_fft : guard_end].copy()
        self._ifb_valid = valid
        self._ifb_dropped = int(self.left_samples_dropped)
        self._ifb_length = len(buffer)
        return torch.from_numpy(power[:frames])

    def _ifb_rebase(self, buffer: np.ndarray, n_fft: int, hop: int) -> int:
        """Rows still exact for ``buffer``: the kept rows (moved down after a slide), or 0."""
        valid = self._ifb_valid
        if valid == 0:
            return 0
        shift = int(self.left_samples_dropped) - self._ifb_dropped
        guard_end = (valid - 1) * hop + n_fft // 2 - shift
        if (
            shift < 0
            or shift % hop
            or len(buffer) + shift < self._ifb_length
            or guard_end < n_fft
            or not np.array_equal(buffer[guard_end - n_fft : guard_end], self._ifb_guard)
        ):
            # Not the same stream (or not a hop-aligned slide of it): start over.
            return 0
        if shift == 0:
            return valid
        drop = shift // hop
        valid -= drop
        if valid <= _ALIGN_FRAMES:
            return 0
        power = self._ifb_power
        power[:valid] = power[drop : drop + valid]
        # The new left edge's frames read the reflect pad.
        power[:_ALIGN_FRAMES] = _stft_power(buffer, 0, _ALIGN_FRAMES, n_fft, hop, self._ifb_window)
        return valid


def _self_check(mel_processor: Any, incremental_cls: type) -> bool:  # Any: remote-code processor
    """Compare the incremental and the original features on a probe stream (with a slide) once per class."""
    fe = mel_processor.feature_extractor
    saved = {name: getattr(fe, name) for name in ("dynamic_log_norm", "dynamic_range_db", "log_floor_db")}
    probe = copy.copy(mel_processor)
    probe.__class__ = incremental_cls
    hop = int(fe.hop_length)
    audio = (0.1 * np.random.default_rng(0).standard_normal(8 * int(mel_processor.sample_rate))).astype(np.float32)
    try:
        # Below and above the 5 s dynamic-range switch, then a slide of 100 hops.
        for dropped, end in ((0, len(audio) // 3 + 37), (0, len(audio) - 11), (100 * hop, len(audio))):
            probe.buffer, probe.left_samples_dropped = audio[dropped:end], dropped
            expected = type(mel_processor)._extract_full(probe)
            if not torch.equal(incremental_cls._extract_full(probe), expected):
                return False
        return True
    except Exception:
        logger.warning("MiniCPM-o incremental fbank self-check raised", exc_info=True)
        return False
    finally:
        for name, value in saved.items():
            setattr(fe, name, value)


def enable_incremental_fbank(mel_processor: Any) -> bool:  # Any: remote-code StreamingMelProcessorExact
    """Switch one session's streaming mel processor to the incremental ``_extract_full`` (idempotent).

    The instance's class becomes a subclass of its remote class with
    ``_IncrementalFbankMixin`` in front, so copies keep the behavior. Returns
    False, leaving the instance on the full recompute, for a processor of
    another shape or one whose features differ from the original's on the
    probe stream.
    """
    if isinstance(mel_processor, _IncrementalFbankMixin):
        return True
    cls = type(mel_processor)
    fe = getattr(mel_processor, "feature_extractor", None)
    if not callable(getattr(cls, "_extract_full", None)) or not all(
        hasattr(fe, name) for name in _FEATURE_EXTRACTOR_ATTRS
    ):
        return False
    if not all(hasattr(mel_processor, name) for name in ("buffer", "sample_rate", "left_samples_dropped")):
        return False
    incremental_cls = _INCREMENTAL_CLASSES.get(cls)
    if incremental_cls is None:
        incremental_cls = type(f"Incremental{cls.__name__}", (_IncrementalFbankMixin, cls), {"__module__": __name__})
        _INCREMENTAL_CLASSES[cls] = incremental_cls
    key = (cls, type(fe))
    if key not in _SELF_CHECKED:
        _SELF_CHECKED[key] = _self_check(mel_processor, incremental_cls)
        if not _SELF_CHECKED[key]:
            logger.warning(
                "MiniCPM-o incremental fbank differs from %s._extract_full on the probe stream; "
                "the streaming fbank stays on the full recompute",
                cls.__name__,
            )
    if not _SELF_CHECKED[key]:
        return False
    mel_processor.__class__ = incremental_cls
    return True
