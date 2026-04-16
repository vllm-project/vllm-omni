"""Stability resource monitor helpers."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

STABILITY_DIR = Path(__file__).resolve().parent
RESOURCE_MONITOR_SCRIPT = STABILITY_DIR / "scripts" / "resource_monitor.sh"
REPO_ROOT = STABILITY_DIR.parent.parent.parent


def start_resource_monitor():
    """Start `resource_monitor.sh start` in the background and return `Popen` or `None`."""
    if not RESOURCE_MONITOR_SCRIPT.is_file():
        return None
    try:
        proc = subprocess.Popen(
            ["bash", str(RESOURCE_MONITOR_SCRIPT), "start", "--backend", "gpu"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=2)
            if proc.returncode != 0:
                stderr = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
                if stderr.strip():
                    sys.stderr.write(f"[Stability] Resource monitor failed to start: {stderr.strip()}\n")
                return None
        except subprocess.TimeoutExpired:
            pass
        return proc
    except (FileNotFoundError, OSError):
        return None


def get_monitor_data_root() -> Path:
    data_root = os.environ.get("RESOURCE_MONITOR_DATA_ROOT") or os.environ.get("GPU_MONITOR_DATA_ROOT")
    if data_root:
        return Path(data_root)
    return STABILITY_DIR / "gpu_monitor_data"


def wait_for_run_dir(timeout_sec: int = 10) -> Path | None:
    data_root = get_monitor_data_root()
    run_id_file = data_root / "current_run_id"
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if run_id_file.is_file():
            run_id = run_id_file.read_text(encoding="utf-8").strip()
            if run_id:
                run_dir = data_root / run_id
                if run_dir.is_dir():
                    return run_dir
        time.sleep(0.5)
    return None


def report_latest_gpu_samples(stop_event: threading.Event) -> None:
    """Periodically print the latest sampled GPU line."""
    log_interval = int(
        os.environ.get("RESOURCE_MONITOR_LOG_INTERVAL") or os.environ.get("GPU_MONITOR_LOG_INTERVAL") or "15"
    )
    log_interval = max(log_interval, 1)
    last_line = ""

    time.sleep(min(log_interval, 5))
    while not stop_event.wait(log_interval):
        run_dir = wait_for_run_dir(timeout_sec=1)
        if run_dir is None:
            continue
        csv_file = run_dir / "gpu_metrics.csv"
        if not csv_file.is_file():
            continue
        try:
            lines = csv_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        if len(lines) <= 1:
            continue
        latest = lines[-1].strip()
        if latest and latest != last_line:
            last_line = latest
            sys.stderr.write(f"[GPU] {latest}\n")


def finalize_resource_monitor() -> str | None:
    """
    Run `resource_monitor.sh finalize` for the current run and generate the report.
    Returns the bundle dir path (for this test case's report) if successful, else None.
    """
    if not RESOURCE_MONITOR_SCRIPT.is_file():
        return None
    try:
        result = subprocess.run(
            ["bash", str(RESOURCE_MONITOR_SCRIPT), "finalize", "--backend", "gpu"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            return None
        for line in (result.stdout or "").splitlines():
            if line.startswith("GPU_MONITOR_BUNDLE_DIR=") or line.startswith("RESOURCE_MONITOR_BUNDLE_DIR="):
                _, _, value = line.partition("=")
                return value.strip() if value else None
        return None
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
