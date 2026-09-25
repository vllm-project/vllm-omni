# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reading a vLLM ``RequestOutput`` the way model-side projection code needs to.

Stage outputs arrive as duck-typed objects (a ``RequestOutput``, an
``OmniRequestOutput`` wrapping one, or a test double): these helpers unwrap
them, read the ``multimodal_output`` mapping, coerce tensor scalars, and turn
cumulative audio/text into the delta since the last read. No duplex
vocabulary lives here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np


def unwrap_request_output(output: object) -> tuple[object, object | None]:
    """Return ``(request_output, first_completion)`` for a stage output or a wrapper around one."""
    inner = getattr(output, "request_output", None)
    if inner is not None and inner is not output:
        output = inner
    return output, first_completion(output)


def first_completion(output: object) -> object | None:
    outputs = getattr(output, "outputs", None)
    return outputs[0] if isinstance(outputs, list) and outputs else None


def multimodal_output(output: object, completion: object | None = None) -> dict[str, object]:
    """The first non-empty ``multimodal_output`` mapping of the output or its completion, copied."""
    for candidate in (
        getattr(output, "multimodal_output", None),
        getattr(completion, "multimodal_output", None) if completion is not None else None,
    ):
        if isinstance(candidate, Mapping) and candidate:
            return dict(candidate)
    return {}


def coerce_int(value: object) -> int | None:
    """``int(value)`` for scalars, one-element tensors/arrays and numeric strings; ``None`` otherwise."""
    detach = getattr(value, "detach", None)
    if callable(detach):
        try:
            flat = detach().cpu().reshape(-1)
            if flat.numel() == 0:
                return None
            value = flat[0].item()
        except (RuntimeError, TypeError, ValueError, IndexError):
            return None
    elif isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.reshape(-1)[0].item()
    try:
        return int(cast(Any, value))  # Any: duck-typed scalar (int/float/str/tensor item)
    except (TypeError, ValueError):
        return None


def coerce_int_list(value: object) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        try:
            value = value.detach().cpu().reshape(-1).tolist()
        except (RuntimeError, TypeError, ValueError):
            return []
    elif isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if not isinstance(value, (list, tuple)):
        return []
    return [token_id for item in value if (token_id := coerce_int(item)) is not None]


def audio_value(multimodal: Mapping[str, object]) -> object | None:
    """The audio carried by a ``multimodal_output`` (``audio`` / ``model_outputs`` / ``latent``)."""
    value = next(
        (multimodal[key] for key in ("audio", "model_outputs", "latent") if key in multimodal),
        None,
    )
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def text_value(multimodal: Mapping[str, object], completion: object | None) -> str:
    for candidate in (
        multimodal.get("text"),
        multimodal.get("llm_output_text"),
        getattr(completion, "text", None) if completion is not None else None,
    ):
        if isinstance(candidate, str) and candidate:
            return candidate
    return ""


def text_delta(text: str, previous: str) -> str:
    """What ``text`` adds over ``previous`` when outputs are cumulative; the whole text on a restart."""
    if not text:
        return ""
    if text == previous:
        return ""
    if previous and text.startswith(previous):
        return text[len(previous) :]
    return text


def audio_sample_count(audio: object | None) -> int | None:
    if audio is None:
        return None
    try:
        import torch

        if isinstance(audio, torch.Tensor):
            return int(audio.numel())
    except ImportError:
        pass
    try:
        return int(np.asarray(audio, dtype=np.float32).size)
    except (TypeError, ValueError):
        return None


def slice_audio_delta(audio: object | None, offset: int) -> object | None:
    """The samples of a cumulative ``audio`` past ``offset``; the whole audio when it restarted."""
    samples = audio_sample_count(audio)
    if samples is None or samples <= 0:
        return None
    if offset <= 0 or samples < offset:
        return audio
    if samples == offset:
        return None
    try:
        import torch

        if isinstance(audio, torch.Tensor):
            return audio.reshape(-1)[offset:].contiguous()
    except ImportError:
        pass
    return np.asarray(audio, dtype=np.float32).reshape(-1)[offset:]


def sample_rate_hz(multimodal: Mapping[str, object], *, default: int) -> int:
    value = multimodal.get("sr", multimodal.get("sample_rate_hz", default))
    if isinstance(value, list) and value:
        value = value[0]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (RuntimeError, TypeError, ValueError):
            value = default
    return int(value) if isinstance(value, int | float) and value > 0 else default


__all__ = [
    "audio_sample_count",
    "audio_value",
    "coerce_int",
    "coerce_int_list",
    "first_completion",
    "multimodal_output",
    "sample_rate_hz",
    "slice_audio_delta",
    "text_delta",
    "text_value",
    "unwrap_request_output",
]
