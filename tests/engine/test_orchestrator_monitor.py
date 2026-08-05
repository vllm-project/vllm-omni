import json
import time

import pytest

import vllm_omni.engine.orchestrator_monitor as monitor_mod
from vllm_omni.engine.orchestrator_monitor import (
    _NULL_ORCH_MONITOR,
    OrchestratorMonitor,
    create_orch_monitor,
    replica_key,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_replica_key_format():
    assert replica_key(0, 0) == "stage=0,replica=0"
    assert replica_key(2, 1) == "stage=2,replica=1"


def test_create_orch_monitor_disabled_returns_null_singleton():
    monitor = create_orch_monitor(enabled=False, replica_sampler=lambda: {})
    assert monitor is _NULL_ORCH_MONITOR
    assert create_orch_monitor(enabled=False, replica_sampler=lambda: {}) is monitor
    monitor.note_loop(idle=True)
    monitor.flush()


def test_note_loop_and_flush_json(tmp_path, monkeypatch):
    out_path = tmp_path / "orch_monitor.json"
    monkeypatch.setenv("VLLM_OMNI_ORCH_MONITOR_PATH", str(out_path))

    now = [0.0]

    def fake_monotonic() -> float:
        return now[0]

    monkeypatch.setattr(time, "monotonic", fake_monotonic)

    def sampler() -> dict[str, tuple[int, int]]:
        return {replica_key(0, 0): (5, 10)}

    monitor = create_orch_monitor(enabled=True, replica_sampler=sampler)
    monitor.register_replica(0, 0)

    monitor.note_loop(idle=False)
    monitor.note_loop(idle=True)

    now[0] = 1.5
    monitor.flush()

    payload = json.loads(out_path.read_text())
    assert payload["configured_window_s"] == monitor_mod._WINDOW_S
    assert payload["windows"]["loop_idle"] == [1]
    assert payload["windows"]["loop_active"] == [1]
    assert payload["replicas"]["stage=0,replica=0"]["outputs_queue_size"] == [5]
    assert payload["replicas"]["stage=0,replica=0"]["inflight"] == [10]


def test_register_replica_backfills_prior_windows(tmp_path, monkeypatch):
    out_path = tmp_path / "orch_monitor.json"
    monkeypatch.setenv("VLLM_OMNI_ORCH_MONITOR_PATH", str(out_path))

    now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    monitor = OrchestratorMonitor(replica_sampler=lambda: {})
    monitor.note_loop(idle=True)
    now[0] = 1.5
    monitor.register_replica(1, 0)
    monitor.note_loop(idle=False)
    monitor.flush()

    series = json.loads(out_path.read_text())["replicas"]["stage=1,replica=0"]
    assert series["outputs_queue_size"] == [0, 0]
    assert series["inflight"] == [0, 0]


def test_note_loop_survives_sampler_failure():
    def bad_sampler() -> dict[str, tuple[int, int]]:
        raise RuntimeError("sampler failed")

    monitor = OrchestratorMonitor(replica_sampler=bad_sampler)
    monitor.register_replica(0, 0)
    monitor.note_loop(idle=True)
    monitor.flush()


def test_flush_is_idempotent(tmp_path, monkeypatch):
    out_path = tmp_path / "orch_monitor.json"
    monkeypatch.setenv("VLLM_OMNI_ORCH_MONITOR_PATH", str(out_path))

    monitor = OrchestratorMonitor(replica_sampler=lambda: {})
    monitor.register_replica(0, 0)
    monitor.note_loop(idle=False)
    monitor.flush()
    monitor.flush()

    payload = json.loads(out_path.read_text())
    assert len(payload["windows"]["duration_s"]) == 1
