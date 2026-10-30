# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Behavior tests for the AMD CosyVoice3 s3tokenizer prefetch step."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_SCRIPT = Path(".buildkite/amd/scripts/prefetch_s3tokenizer.py")


def _load_prefetch():
    spec = importlib.util.spec_from_file_location("prefetch_s3tokenizer", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prefetch_s3tokenizer = _load_prefetch()


def test_prefetch_loads_the_cosyvoice3_checkpoint_once_when_the_download_succeeds() -> None:
    loaded: list[str] = []
    slept: list[float] = []

    prefetch_s3tokenizer.prefetch(load_model=loaded.append, sleep=slept.append)

    assert loaded == ["speech_tokenizer_v3_25hz"]
    assert slept == []


def test_prefetch_retries_an_interrupted_download_before_succeeding() -> None:
    calls: list[str] = []
    slept: list[float] = []

    def flaky_load_model(name: str) -> None:
        calls.append(name)
        if len(calls) < 3:
            raise OSError("connection reset by modelscope.cn")

    prefetch_s3tokenizer.prefetch(
        "speech_tokenizer_v3_25hz", attempts=3, load_model=flaky_load_model, sleep=slept.append
    )

    assert calls == ["speech_tokenizer_v3_25hz"] * 3
    assert slept == [prefetch_s3tokenizer.RETRY_DELAY_S] * 2


def test_prefetch_fails_loudly_once_every_attempt_is_exhausted() -> None:
    calls: list[str] = []

    def broken_load_model(name: str) -> None:
        calls.append(name)
        raise RuntimeError("SHA256 checksum does not match")

    with pytest.raises(RuntimeError, match="after 2 attempts") as excinfo:
        prefetch_s3tokenizer.prefetch(attempts=2, load_model=broken_load_model, sleep=lambda _: None)

    assert calls == ["speech_tokenizer_v3_25hz"] * 2
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_prefetch_rejects_zero_attempts() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        prefetch_s3tokenizer.prefetch(attempts=0, load_model=lambda _: None, sleep=lambda _: None)


def test_main_parses_model_and_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_prefetch(name: str, *, attempts: int) -> None:
        seen["name"] = name
        seen["attempts"] = attempts

    monkeypatch.setattr(prefetch_s3tokenizer, "prefetch", fake_prefetch)

    assert prefetch_s3tokenizer.main(["--model", "speech_tokenizer_v2_25hz", "--attempts", "5"]) == 0
    assert seen == {"name": "speech_tokenizer_v2_25hz", "attempts": 5}
