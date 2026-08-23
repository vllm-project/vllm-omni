# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for OmniServerStageCli process planning helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers.runtime import OmniServerStageCli

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _write_stage_config(tmp_path: Path) -> str:
    path = tmp_path / "stages.yaml"
    path.write_text(
        """
stages:
  - stage_id: 0
    devices: "0"
  - stage_id: 1
    devices: "1,2,3"
    num_replicas: 3
""".strip(),
        encoding="utf-8",
    )
    return str(path)


def test_stage_cli_builds_headless_replica_cmd(tmp_path: Path) -> None:
    server = OmniServerStageCli("fake-model", _write_stage_config(tmp_path), ["--disable-log-stats"])

    cmd = server._build_stage_cmd(1, headless=True, replica_id=2)

    assert "--headless" in cmd
    assert cmd[cmd.index("--stage-id") + 1] == "1"
    assert cmd[cmd.index("--replica-id") + 1] == "2"
    assert cmd[cmd.index("--deploy-config") + 1] == server.stage_config_path


def test_stage_cli_loads_stage_ids_and_replica_counts(tmp_path: Path) -> None:
    server = OmniServerStageCli("fake-model", _write_stage_config(tmp_path), [])

    assert server.stage_ids == [0, 1]
    assert server.stage_replica_counts == {0: 1, 1: 3}
