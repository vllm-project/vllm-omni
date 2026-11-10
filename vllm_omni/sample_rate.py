# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Canonical audio sampling-rate key and model-output payload resolution.

Single source of truth for reading the rate audio models must emit
alongside their waveform. Producers use ``SAMPLE_RATE_KEY`` ("sr");
consumers read it with :func:`get_sampling_rate`. Legacy spellings still
resolve but warn (removed in 0.36). :func:`find_sample_rate` is the
separate lenient, warning-free scan for config objects. Deliberately a
leaf module (stdlib + torch only) to avoid import cycles.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Mapping, Sequence
from types import FrameType
from typing import cast

import torch

# The canonical sampling-rate key for model output payloads.
SAMPLE_RATE_KEY = "sr"

# Legacy spellings still found in payloads; reading them works but warns.
DEPRECATED_SAMPLE_RATE_KEYS: tuple[str, ...] = (
    "sample_rate",
    "audio_sample_rate",
    "sampling_rate",
    "output_sample_rate",
)

# All flat keys that may carry a rate, canonical first (resolution order).
ALL_SAMPLE_RATE_KEYS: tuple[str, ...] = (SAMPLE_RATE_KEY, *DEPRECATED_SAMPLE_RATE_KEYS)

# Version in which deprecated-key support is removed.
SAMPLE_RATE_REMOVAL_VERSION = "0.36"

# Nested diffusion-envelope spelling: metadata["audio"]["sample_rate"].
_NESTED_METADATA_KEYS: tuple[str, ...] = ("metadata", "audio", "sample_rate")

_THIS_FILENAME = __file__


class DeprecatedSampleRateKeyWarning(DeprecationWarning):
    """A legacy sampling-rate key was read from a model output payload."""


class SampleRateError(ValueError):
    """Base class for sampling-rate resolution failures."""


class MissingSampleRateError(SampleRateError):
    """No sampling-rate key is present in the payload."""


class InvalidSampleRateError(SampleRateError):
    """A sampling rate is present but is not a positive integer."""


# ---------------------------------------------------------------------------
# Lenient coercion and generic lookup (no warnings)
# ---------------------------------------------------------------------------


def coerce_sample_rate(value: object) -> int | None:
    """Coerce a rate-like value to a positive int, or ``None`` on failure.

    Accepts ints, 0-d tensors/numpy scalars, and lists/tuples thereof
    (first positive entry wins).
    """
    return _coerce_lenient(value)


def _coerce_lenient(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            coerced = _coerce_lenient(item)
            if coerced is not None:
                return coerced
        return None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except Exception:
            # Multi-element tensors and device-side failures land here.
            return None
    if isinstance(value, bool):
        return None
    try:
        coerced = int(cast("int | float | str", value))
    except (TypeError, ValueError):
        return None
    return coerced if coerced > 0 else None


def find_sample_rate(
    source: Sequence[Mapping[str, object] | object | None] | Mapping[str, object] | object | None,
) -> int | None:
    """Locate a rate on a payload, config object, or sequence of either.

    Probes ``ALL_SAMPLE_RATE_KEYS`` (canonical first); first positive hit
    wins. Emits no warnings — config schemas legitimately use these names.
    """
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        for item in source:
            rate = _find_on_item(item)
            if rate is not None:
                return rate
        return None
    return _find_on_item(source)


def _find_on_item(source: Mapping[str, object] | object | None) -> int | None:
    if not source:
        return None
    for key in ALL_SAMPLE_RATE_KEYS:
        raw = source.get(key) if isinstance(source, Mapping) else getattr(source, key, None)
        rate = _coerce_lenient(raw)
        if rate is not None:
            return rate
    return None


# ---------------------------------------------------------------------------
# Strict model-output payload reading (deprecation warnings, loud failures)
# ---------------------------------------------------------------------------


def get_sampling_rate(payload: Mapping[str, object]) -> int:
    """Return the sampling rate (Hz) carried by a model output payload.

    Accepts any ``Mapping`` (including ``MultimodalPayload``). Legacy
    spellings resolve with :class:`DeprecatedSampleRateKeyWarning`.

    Raises:
        MissingSampleRateError: no key carries a rate; do not substitute
            a default.
        InvalidSampleRateError: the rate is not a positive integer
            (bools, non-integral floats, and conflicting per-request
            rates are rejected).
        TypeError: ``payload`` is not a ``Mapping``.
    """
    rate = _resolve_payload(payload)
    if rate is None:
        raise MissingSampleRateError(
            "model output payload carries no audio sample rate; audio-emitting models must "
            f"emit {SAMPLE_RATE_KEY!r} (tried flat keys {list(ALL_SAMPLE_RATE_KEYS)} and "
            f"nested {'.'.join(_NESTED_METADATA_KEYS)})"
        )
    return rate


def get_sampling_rate_or_none(payload: Mapping[str, object]) -> int | None:
    """Like :func:`get_sampling_rate` but ``None`` when absent.

    Invalid values still raise; only the missing case is softened.
    """
    return _resolve_payload(payload)


def _resolve_payload(payload: Mapping[str, object]) -> int | None:
    if not isinstance(payload, Mapping):
        raise TypeError(f"sample-rate payload must be a Mapping, got {type(payload).__name__}")
    # An explicit {"sr": None} counts as absent; keep probing.
    raw = payload.get(SAMPLE_RATE_KEY)
    if raw is not None:
        return _coerce_strict(raw, SAMPLE_RATE_KEY)
    for key in DEPRECATED_SAMPLE_RATE_KEYS:
        raw = payload.get(key)
        if raw is None:
            continue
        rate = _coerce_strict(raw, key)
        _warn_deprecated(
            f"sample-rate-deprecated-key:{key}",
            f"Reading audio sample rate from legacy key {key!r}. Models should emit "
            f"{SAMPLE_RATE_KEY!r} instead; support for {key!r} is removed in "
            f"vLLM-Omni {SAMPLE_RATE_REMOVAL_VERSION}.",
        )
        return rate
    nested: object = payload
    for key in _NESTED_METADATA_KEYS:
        nested = nested.get(key) if isinstance(nested, Mapping) else None
        if nested is None:
            break
    if nested is not None:
        rate = _coerce_strict(nested, ".".join(_NESTED_METADATA_KEYS))
        _warn_deprecated(
            "sample-rate-deprecated-key:nested",
            f"Reading audio sample rate from nested {'.'.join(_NESTED_METADATA_KEYS)}. Models "
            f"should emit flat {SAMPLE_RATE_KEY!r} instead; the nested spelling is removed in "
            f"vLLM-Omni {SAMPLE_RATE_REMOVAL_VERSION}.",
        )
        return rate
    return None


def _coerce_strict(value: object, source: str) -> int:
    """Coerce to a positive int or raise :class:`InvalidSampleRateError`."""
    if isinstance(value, bool):
        raise InvalidSampleRateError(f"audio sample rate from {source!r} must be an integer, got bool")
    if isinstance(value, int):
        if value <= 0:
            raise InvalidSampleRateError(f"audio sample rate from {source!r} must be positive, got {value}")
        return value
    if isinstance(value, float):
        if not value.is_integer() or value <= 0:
            raise InvalidSampleRateError(f"audio sample rate from {source!r} must be a positive integer, got {value}")
        return int(value)
    if isinstance(value, (list, tuple)):
        if not value:
            raise InvalidSampleRateError(f"audio sample rate list from {source!r} is empty")
        rates = {_coerce_strict(item, source) for item in value}
        if len(rates) != 1:
            raise InvalidSampleRateError(f"conflicting audio sample rates from {source!r}: {sorted(rates)}")
        return rates.pop()
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            raise InvalidSampleRateError(f"audio sample rate tensor from {source!r} is empty")
        try:
            elements = value.reshape(-1).tolist()
        except RuntimeError as exc:
            raise InvalidSampleRateError(f"audio sample rate tensor from {source!r} could not be read: {exc}") from exc
        return _coerce_strict(elements, source)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            unpacked = item()
        except Exception as exc:
            raise InvalidSampleRateError(f"audio sample rate from {source!r} could not be unpacked: {exc}") from exc
        return _coerce_strict(unpacked, source)
    raise InvalidSampleRateError(f"audio sample rate from {source!r} must be an integer, got {type(value).__name__}")


# ---------------------------------------------------------------------------
# Warning plumbing
# ---------------------------------------------------------------------------

# (spelling, file, line) callsites that already warned.
_warned_callsites: set[tuple[str, str, int]] = set()


def _warn_deprecated(callsite_key: str, message: str) -> None:
    """Emit a deprecation warning once per (spelling, callsite).

    Streaming reads hit the same callsite many times per second; dedup
    keeps the log usable.
    """
    frame: FrameType | None = sys._getframe(1)
    while frame is not None and frame.f_code.co_filename == _THIS_FILENAME:
        frame = frame.f_back
    location = (frame.f_code.co_filename, frame.f_lineno) if frame is not None else ("<unknown>", 0)
    callsite = (callsite_key, *location)
    if callsite in _warned_callsites:
        return
    _warned_callsites.add(callsite)
    # stacklevel 4 attributes the warning to the caller of the public getter.
    warnings.warn(message, DeprecatedSampleRateKeyWarning, stacklevel=4)
