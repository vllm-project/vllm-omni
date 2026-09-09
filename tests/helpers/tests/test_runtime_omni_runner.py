# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for OmniRunner startup rollback."""

from __future__ import annotations

import pytest

from tests.helpers.runtime import OmniRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_runner_init_rolls_back_on_omni_startup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """``__exit__`` is skipped when construction raises; rollback must still run."""
    cleaned: list[str] = []
    monkeypatch.setattr(
        "tests.helpers.runtime.cleanup_test_environment",
        lambda: cleaned.append("env"),
    )
    monkeypatch.setattr(
        OmniRunner,
        "_cleanup_process",
        lambda self: cleaned.append("process"),
    )

    class _BoomOmni:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("Orchestrator initialization failed")

    monkeypatch.setattr("vllm_omni.entrypoints.omni.Omni", _BoomOmni)

    with pytest.raises(RuntimeError, match="Orchestrator initialization failed"):
        with OmniRunner("fake-model"):
            raise AssertionError("context body must not run after a failed constructor")

    assert cleaned.count("process") == 1
    # Once at the start of ``__init__``, once from the constructor rollback.
    assert cleaned.count("env") == 2
