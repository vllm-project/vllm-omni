# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Precedence, validation and deprecation of the Token2Wav TensorRT settings (#6232).

The stage keys ``token2wav_trt``, ``token2wav_trt_dtype`` and
``token2wav_trt_cache_dir`` win whenever they are set and are validated on
every platform. The legacy ``MINICPMO_TOKEN2WAV_TRT*`` env vars apply only
while their key is unset, are read only where they take effect (CUDA, and for
dtype/cache only with full DiT TensorRT enabled), and warn once when read.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vllm_omni.model_executor.models.minicpmo_4_5 import config_flags
from vllm_omni.model_executor.models.minicpmo_4_5.config_flags import (
    Token2WavTrtConfig,
    resolve_token2wav_trt_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_ENV_NAMES = ("MINICPMO_TOKEN2WAV_TRT", "MINICPMO_TOKEN2WAV_TRT_DTYPE", "MINICPMO_TOKEN2WAV_TRT_CACHE")
_DEFAULT = Token2WavTrtConfig(enabled=False, dtype="fp16", cache_dir=None)


@pytest.fixture
def warnings(monkeypatch) -> list[str]:
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    messages: list[str] = []
    monkeypatch.setattr(config_flags.logger, "warning_once", lambda msg, *args: messages.append(msg))
    return messages


@pytest.mark.parametrize("cuda", [True, False])
def test_defaults_without_keys_or_env(warnings, cuda):
    assert resolve_token2wav_trt_config({}, cuda=cuda) == _DEFAULT
    assert warnings == []


@pytest.mark.parametrize("cuda", [True, False])
def test_keys_without_env_do_not_warn(warnings, cuda):
    config = resolve_token2wav_trt_config(
        {"token2wav_trt": "true", "token2wav_trt_dtype": "Float32", "token2wav_trt_cache_dir": Path("/trt")},
        cuda=cuda,
    )
    assert config == Token2WavTrtConfig(enabled=True, dtype="fp32", cache_dir="/trt")
    assert warnings == []


@pytest.mark.parametrize("extra", [{}, {"token2wav_trt": None}], ids=["missing", "null"])
def test_legacy_trt_env_applies_on_cuda_while_key_is_unset(monkeypatch, warnings, extra):
    monkeypatch.setenv("MINICPMO_TOKEN2WAV_TRT", "1")
    assert resolve_token2wav_trt_config(extra, cuda=True).enabled is True
    assert len(warnings) == 1
    assert "MINICPMO_TOKEN2WAV_TRT is deprecated and will be removed" in warnings[0]
    assert "`token2wav_trt`" in warnings[0]


@pytest.mark.parametrize(
    "name,value,key,field,expected",
    [
        ("MINICPMO_TOKEN2WAV_TRT_DTYPE", "FP32", "token2wav_trt_dtype", "dtype", "fp32"),
        ("MINICPMO_TOKEN2WAV_TRT_CACHE", " /env/cache ", "token2wav_trt_cache_dir", "cache_dir", "/env/cache"),
    ],
)
@pytest.mark.parametrize("unset", [{}, None], ids=["missing", "null"])
def test_legacy_engine_env_applies_with_trt_on_cuda(monkeypatch, warnings, name, value, key, field, expected, unset):
    monkeypatch.setenv(name, value)
    extra = {"token2wav_trt": True} if unset == {} else {"token2wav_trt": True, key: None}
    assert getattr(resolve_token2wav_trt_config(extra, cuda=True), field) == expected
    assert len(warnings) == 1
    assert f"{name} is deprecated and will be removed" in warnings[0]
    assert f"`{key}`" in warnings[0]


@pytest.mark.parametrize(
    "name,value,extra,field,expected",
    [
        ("MINICPMO_TOKEN2WAV_TRT", "1", {"token2wav_trt": False}, "enabled", False),
        (
            "MINICPMO_TOKEN2WAV_TRT_DTYPE",
            "fp32",
            {"token2wav_trt": True, "token2wav_trt_dtype": "fp16"},
            "dtype",
            "fp16",
        ),
        (
            "MINICPMO_TOKEN2WAV_TRT_CACHE",
            "/env",
            {"token2wav_trt": True, "token2wav_trt_cache_dir": "/key"},
            "cache_dir",
            "/key",
        ),
    ],
)
def test_set_key_wins_and_warns_that_env_is_ignored(monkeypatch, warnings, name, value, extra, field, expected):
    monkeypatch.setenv(name, value)
    assert getattr(resolve_token2wav_trt_config(extra, cuda=True), field) == expected
    assert len(warnings) == 1
    assert f"{name} is deprecated and is ignored" in warnings[0]


@pytest.mark.parametrize("extra", [{}, {"token2wav_trt": True}, {"token2wav_trt": False}])
def test_env_is_not_read_off_cuda(monkeypatch, warnings, extra):
    """TensorRT is CUDA-only, so a legacy env var there would be advice that does nothing."""
    for name, value in zip(_ENV_NAMES, ("1", "bf16", "/env")):
        monkeypatch.setenv(name, value)
    config = resolve_token2wav_trt_config(extra, cuda=False)
    assert config == Token2WavTrtConfig(enabled=bool(extra.get("token2wav_trt")), dtype="fp16", cache_dir=None)
    assert warnings == []


@pytest.mark.parametrize("extra", [{}, {"token2wav_trt": False}])
def test_engine_env_is_not_read_with_trt_disabled(monkeypatch, warnings, extra):
    """With full DiT TensorRT off, the dtype/cache env vars neither warn nor fail."""
    monkeypatch.setenv("MINICPMO_TOKEN2WAV_TRT_DTYPE", "bf16")
    monkeypatch.setenv("MINICPMO_TOKEN2WAV_TRT_CACHE", "/env")
    assert resolve_token2wav_trt_config(extra, cuda=True) == _DEFAULT
    assert warnings == []


@pytest.mark.parametrize("value", ["0", "true", "yes"])
def test_legacy_trt_env_enables_only_on_exact_1(monkeypatch, warnings, value):
    monkeypatch.setenv("MINICPMO_TOKEN2WAV_TRT", value)
    assert resolve_token2wav_trt_config({}, cuda=True).enabled is False


def test_empty_legacy_env_counts_as_unset(monkeypatch, warnings):
    for name in _ENV_NAMES:
        monkeypatch.setenv(name, "  ")
    assert resolve_token2wav_trt_config({"token2wav_trt": True}, cuda=True) == Token2WavTrtConfig(
        enabled=True, dtype="fp16", cache_dir=None
    )
    assert warnings == []


@pytest.mark.parametrize("cuda", [True, False])
@pytest.mark.parametrize("value", ["bf16", "half", "", 16, True])
def test_invalid_dtype_key_raises_on_every_platform(warnings, cuda, value):
    with pytest.raises(ValueError, match="token2wav_trt_dtype must be one of"):
        resolve_token2wav_trt_config({"token2wav_trt_dtype": value}, cuda=cuda)


def test_invalid_dtype_env_raises_when_used(monkeypatch, warnings):
    monkeypatch.setenv("MINICPMO_TOKEN2WAV_TRT_DTYPE", "bf16")
    with pytest.raises(ValueError, match="MINICPMO_TOKEN2WAV_TRT_DTYPE must be one of"):
        resolve_token2wav_trt_config({"token2wav_trt": True}, cuda=True)


@pytest.mark.parametrize("cuda", [True, False])
@pytest.mark.parametrize("value", ["maybe", 2])
def test_invalid_trt_key_raises_on_every_platform(warnings, cuda, value):
    with pytest.raises(ValueError, match="token2wav_trt must be a boolean"):
        resolve_token2wav_trt_config({"token2wav_trt": value}, cuda=cuda)


@pytest.mark.parametrize("cuda", [True, False])
@pytest.mark.parametrize("value", ["", "   ", 5])
def test_invalid_cache_dir_key_raises_on_every_platform(warnings, cuda, value):
    with pytest.raises(ValueError, match="token2wav_trt_cache_dir must be a non-empty path"):
        resolve_token2wav_trt_config({"token2wav_trt_cache_dir": value}, cuda=cuda)
