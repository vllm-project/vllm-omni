# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VLLM_OMNI_VOXCPM_CODE_PATH still works but warns that it is deprecated (#6232)."""

from pathlib import Path

import pytest

from vllm_omni.model_executor.models.voxcpm2 import voxcpm2_import_utils

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _capture_warnings(monkeypatch) -> list[str]:
    messages: list[str] = []
    monkeypatch.setattr(voxcpm2_import_utils.logger, "warning_once", lambda msg, *args: messages.append(msg))
    return messages


def test_code_path_env_warns_and_is_still_used(monkeypatch, tmp_path: Path):
    messages = _capture_warnings(monkeypatch)
    monkeypatch.setenv("VLLM_OMNI_VOXCPM_CODE_PATH", str(tmp_path))

    candidates = voxcpm2_import_utils._iter_voxcpm2_src_candidates()

    assert candidates[0] == tmp_path
    assert len(messages) == 1
    assert "VLLM_OMNI_VOXCPM_CODE_PATH is deprecated" in messages[0]


def test_no_warning_without_code_path_env(monkeypatch):
    messages = _capture_warnings(monkeypatch)
    monkeypatch.delenv("VLLM_OMNI_VOXCPM_CODE_PATH", raising=False)

    voxcpm2_import_utils._iter_voxcpm2_src_candidates()

    assert messages == []
