"""Tests for OmniServerStageCli process planning helpers."""

from __future__ import annotations

from pathlib import Path

from tests.helpers.runtime import OmniServerStageCli


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
    assert cmd[cmd.index("--stage-configs-path") + 1] == server.stage_config_path


def test_stage_cli_splits_devices_per_replica(tmp_path: Path) -> None:
    server = OmniServerStageCli("fake-model", _write_stage_config(tmp_path), [])

    assert server._devices_for_replica(1, "1,2,3", 0) == "1"
    assert server._devices_for_replica(1, "1,2,3", 1) == "2"
    assert server._devices_for_replica(1, "1,2,3", 2) == "3"


def test_stage_cli_maps_visible_devices_after_replica_split(tmp_path: Path) -> None:
    server = OmniServerStageCli(
        "fake-model",
        _write_stage_config(tmp_path),
        [],
        env_dict={"CUDA_VISIBLE_DEVICES": "4,5,6,7"},
    )
    env: dict[str, str] = {}

    server._set_stage_device_env(1, env, "1,2,3", replica_id=1)

    assert env["CUDA_VISIBLE_DEVICES"] == "6"
