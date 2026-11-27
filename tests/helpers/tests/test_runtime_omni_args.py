# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU regression tests for iter_omni_server argument assembly.

Verifies that --stage-init-timeout, --init-timeout, and --log-stats are only
added when use_omni=True.  The use_omni=False assertion fails on baseline
da572cddb and passes on this head.
"""

from __future__ import annotations

import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tests.helpers import runtime
from tests.helpers.runtime import OmniServerParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

OMNI_ONLY_FLAGS = {"--stage-init-timeout", "--init-timeout", "--log-stats"}


def _capture_server_args(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    captured: list[str] = []

    def fake_omni_server(model, serve_args, **kwargs):
        del model, kwargs
        captured.extend(serve_args)
        return nullcontext(SimpleNamespace(model="fake-model"))

    monkeypatch.setattr(runtime, "OmniServer", fake_omni_server)
    monkeypatch.setattr(runtime, "_whisper_device_free_around", nullcontext)
    monkeypatch.setattr(
        "tests.helpers.stage_config.stage_config_path_for_run_level",
        lambda _config_path, _run_level: None,
    )
    return captured


def _run_fixture(params: OmniServerParams):
    request = SimpleNamespace(
        param=params,
        node=SimpleNamespace(get_closest_marker=lambda _name: None),
    )
    gen = runtime.iter_omni_server(request, "core_model", threading.Lock())
    next(gen)
    with pytest.raises(StopIteration):
        next(gen)


def test_non_omni_skips_omni_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_server_args(monkeypatch)
    _run_fixture(OmniServerParams(model="fake-model", use_omni=False))
    leaked = OMNI_ONLY_FLAGS & set(captured)
    assert not leaked, f"Omni-only flags leaked into non-Omni args: {leaked}"


def test_omni_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_server_args(monkeypatch)
    _run_fixture(OmniServerParams(model="fake-model", use_omni=True))
    assert captured[captured.index("--stage-init-timeout") + 1] == "600"
    assert captured[captured.index("--init-timeout") + 1] == "900"
    assert "--log-stats" in captured


def test_omni_custom_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_server_args(monkeypatch)
    _run_fixture(OmniServerParams(model="fake-model", use_omni=True, stage_init_timeout=120, init_timeout=300))
    assert captured[captured.index("--stage-init-timeout") + 1] == "120"
    assert captured[captured.index("--init-timeout") + 1] == "300"


def test_omni_disable_log_stats_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_server_args(monkeypatch)
    _run_fixture(OmniServerParams(model="fake-model", use_omni=True, server_args=["--disable-log-stats"]))
    assert "--disable-log-stats" in captured
    assert "--log-stats" not in captured
