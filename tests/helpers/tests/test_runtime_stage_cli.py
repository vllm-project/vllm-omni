# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for OmniServerStageCli process planning helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


def test_stage_cli_enter_rolls_back_launched_stages_on_startup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``__exit__`` is skipped when ``__enter__`` raises; rollback must still run."""
    monkeypatch.setattr("tests.helpers.runtime.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("tests.helpers.runtime.cleanup_test_environment", lambda: None)

    server = OmniServerStageCli("fake-model", _write_stage_config(tmp_path), ["--disable-log-stats"])
    log_path = tmp_path / "stage0.log"
    log_fh = open(log_path, "w", encoding="utf-8")
    log_fh.write("partial launch\n")
    log_fh.flush()

    proc = SimpleNamespace(pid=4242, poll=lambda: None)
    killed: list[int] = []
    monkeypatch.setattr(server, "_kill_process_tree", lambda pid: killed.append(pid))

    def _fake_launch(stage_id: int, *, headless: bool, replica_id: int = 0) -> None:
        stage_key = (stage_id, replica_id)
        server.stage_procs[stage_key] = proc  # type: ignore[assignment]
        server._stage_log_paths[stage_key] = log_path
        server._stage_log_files[stage_key] = log_fh
        if stage_id == 0 and replica_id == 0:
            server.proc = proc  # type: ignore[assignment]

    def _fail_alive() -> None:
        raise RuntimeError("Stage 0 replica 0 exited with code 1")

    monkeypatch.setattr(server, "_launch_stage", _fake_launch)
    monkeypatch.setattr(server, "_ensure_stage_processes_alive", _fail_alive)

    with pytest.raises(RuntimeError, match="exited with code 1"):
        with server:
            raise AssertionError("context body must not run after a failed enter")

    assert killed == [4242]
    assert log_fh.closed
    assert not log_path.exists()
