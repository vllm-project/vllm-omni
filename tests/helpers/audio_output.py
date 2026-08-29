# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parsing helpers for audio carried on ``OmniRequestOutput.multimodal_output``.

These were copy-pasted verbatim across the MOSS-TTS e2e test modules; hoisted
here so the parsing rules (which key holds the audio, how a sample rate is
unwrapped) live in exactly one place and cannot drift between test files.
Currently wired up for MOSS-TTS only — other model families keep their own
helpers until someone has reason to migrate them.

Only the *parsing* is shared. Each test module keeps its own thin
``_collect_audio`` wrapper, because how the per-stage sampling params are
obtained genuinely differs per variant (``Omni`` vs ``OmniRunner``, replicated
defaults vs an explicit list); those wrappers should call
``collect_audio_from_outputs`` with an already-constructed outputs iterable.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


def audio_from_mm(mm: dict | None) -> torch.Tensor | None:
    """Return a 1-D CPU waveform from one output's multimodal_output, or None.

    Audio arrives under ``audio`` or (pre-consolidation) ``model_outputs``, as a
    tensor or a list of per-request tensors. Never use ``a or b`` on tensor
    values — a multi-element tensor raises on truthiness; check ``is None``.
    """
    if not mm:
        return None
    audio = mm.get("audio")
    if audio is None:
        audio = mm.get("model_outputs")
    if audio is None:
        return None
    if isinstance(audio, list):
        parts = [t.reshape(-1) for t in audio if isinstance(t, torch.Tensor) and t.numel() > 0]
        if not parts:
            return None
        audio = torch.cat(parts, dim=0)
    if not isinstance(audio, torch.Tensor):
        return None
    return audio.reshape(-1).cpu()


def sr_from_mm(mm: dict | None) -> int | None:
    """Return the integer sample rate from a multimodal_output, or None.

    The value may be a bare int, a 0-d tensor, or a 1-element list of either.
    """
    if not mm:
        return None
    sr = mm.get("sr")
    if isinstance(sr, (list, tuple)):
        sr = sr[0] if sr else None
    if sr is None:
        return None
    return int(sr.item()) if isinstance(sr, torch.Tensor) else int(sr)


def collect_audio_from_outputs(outputs: Iterable[Any], default_sr: int) -> tuple[torch.Tensor, int]:
    """Concatenate every non-empty audio chunk from ``outputs``, in arrival order.

    ``generate()`` returns a flat list of ``OmniRequestOutput`` (one or more per
    request, as audio streams); each exposes ``.multimodal_output`` directly
    (``.request_output`` is a single inner ``RequestOutput``, not an iterable).

    Returns ``(waveform_cpu_1d, sample_rate)``; ``default_sr`` is used when no
    output reported one. Raises ``AssertionError`` when no audio arrived at all.
    """
    chunks: list[torch.Tensor] = []
    sr = default_sr
    for out in outputs:
        mm = getattr(out, "multimodal_output", None)
        sr_val = sr_from_mm(mm)
        if sr_val is not None:
            sr = sr_val
        chunk = audio_from_mm(mm)
        if chunk is not None and chunk.numel() > 0:
            chunks.append(chunk)
    assert chunks, "No audio received across generate() outputs"
    return torch.cat(chunks, dim=0), sr
