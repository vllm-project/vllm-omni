# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Strict parsing for the boolean knobs of the MiniCPM-o 4.5 stage config."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def parse_bool_flag(value: Any, key: str) -> bool:
    """Parse a stage-config boolean strictly, so a typo fails at startup.

    ``bool("false")`` is ``True``, which would silently keep a feature on for a
    quoted YAML value or a JSON/CLI stage override -- exactly the case these
    switches exist to triage. Accepts bools, 0/1 and the ``"true"``/``"false"``/
    ``"1"``/``"0"`` strings; anything else raises.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in ("true", "1"):
            return True
        if token in ("false", "0"):
            return False
    raise ValueError(f"{key} must be a boolean (true/false/1/0), got {value!r}")


def read_bool_flag(extra: Mapping[str, Any], key: str, default: bool) -> bool:
    """Read ``extra[key]`` strictly, keeping ``default`` when the key is unset.

    An explicit ``null`` (``key: ~`` in a YAML override) means "not set" and
    falls back to ``default`` rather than flipping the feature off, so dropping
    a key from a stage config never changes behaviour silently.
    """
    value = extra.get(key, default)
    if value is None:
        return default
    return parse_bool_flag(value, key)


# Legacy environment switches for the Token2Wav TensorRT path (#6232). Each one
# is honoured only while its stage key is unset, and warns once when read.
TOKEN2WAV_TRT_ENV = "MINICPMO_TOKEN2WAV_TRT"
TOKEN2WAV_TRT_DTYPE_ENV = "MINICPMO_TOKEN2WAV_TRT_DTYPE"
TOKEN2WAV_TRT_CACHE_ENV = "MINICPMO_TOKEN2WAV_TRT_CACHE"

_TRT_DTYPE_ALIASES = {"fp16": "fp16", "float16": "fp16", "fp32": "fp32", "float32": "fp32"}


@dataclass(frozen=True)
class Token2WavTrtConfig:
    """Resolved Token2Wav TensorRT settings.

    ``dtype`` is the canonical ``"fp16"`` or ``"fp32"``. ``cache_dir`` is
    ``None`` when neither the stage key nor the legacy env var sets it, so the
    engine builder keeps its own ``$HF_HOME`` default.
    """

    enabled: bool
    dtype: str
    cache_dir: str | None


def _legacy_env(name: str, key: str, *, used: bool) -> str | None:
    """Return a set legacy env var (``None`` if unset/empty) and warn once."""
    raw = os.environ.get(name, "")
    if not raw.strip():
        return None
    if used:
        logger.warning_once(
            f"{name} is deprecated and will be removed in a future release. Set the `{key}` stage config key instead."
        )
    else:
        logger.warning_once(
            f"{name} is deprecated and is ignored because the `{key}` stage config key is set. "
            f"Remove {name} from the environment."
        )
    return raw


def _parse_trt_dtype(value: Any, source: str) -> str:
    dtype = _TRT_DTYPE_ALIASES.get(value.strip().lower()) if isinstance(value, str) else None
    if dtype is None:
        raise ValueError(f"{source} must be one of fp16, float16, fp32 or float32, got {value!r}")
    return dtype


def _parse_trt_cache_dir(value: Any) -> str:
    if not isinstance(value, (str, os.PathLike)) or not os.fspath(value).strip():
        raise ValueError(f"token2wav_trt_cache_dir must be a non-empty path, got {value!r}")
    return os.fspath(value).strip()


def resolve_token2wav_trt_config(extra: Mapping[str, Any], *, cuda: bool) -> Token2WavTrtConfig:
    """Resolve the Token2Wav TensorRT settings from the stage config.

    An explicitly set stage key always wins, including ``token2wav_trt: false``,
    and is validated on every platform so a typo fails at startup.

    A legacy ``MINICPMO_TOKEN2WAV_TRT*`` env var applies only while its key is
    unset (missing or ``null``), and is read only where it would take effect:
    ``MINICPMO_TOKEN2WAV_TRT`` on CUDA (TensorRT is CUDA-only), and the dtype
    and cache variables only when full DiT TensorRT is enabled on CUDA. Reading
    is what logs the deprecation warning, so a deployment where the value is
    ignored gets no advice that would do nothing.
    """
    trt_value = extra.get("token2wav_trt")
    if trt_value is not None:
        enabled = parse_bool_flag(trt_value, "token2wav_trt")
        if cuda:
            _legacy_env(TOKEN2WAV_TRT_ENV, "token2wav_trt", used=False)
    else:
        # Keep the legacy contract: only exactly "1" turned the env switch on.
        enabled = cuda and _legacy_env(TOKEN2WAV_TRT_ENV, "token2wav_trt", used=True) == "1"

    dtype_value = extra.get("token2wav_trt_dtype")
    dtype = "fp16" if dtype_value is None else _parse_trt_dtype(dtype_value, "token2wav_trt_dtype")
    cache_value = extra.get("token2wav_trt_cache_dir")
    cache_dir = None if cache_value is None else _parse_trt_cache_dir(cache_value)

    if cuda and enabled:
        dtype_env = _legacy_env(TOKEN2WAV_TRT_DTYPE_ENV, "token2wav_trt_dtype", used=dtype_value is None)
        if dtype_value is None and dtype_env is not None:
            dtype = _parse_trt_dtype(dtype_env, TOKEN2WAV_TRT_DTYPE_ENV)
        cache_env = _legacy_env(TOKEN2WAV_TRT_CACHE_ENV, "token2wav_trt_cache_dir", used=cache_value is None)
        if cache_value is None and cache_env is not None:
            cache_dir = cache_env.strip()

    return Token2WavTrtConfig(enabled=enabled, dtype=dtype, cache_dir=cache_dir)
