"""
Stability-specific conftest: when pytest is executed under this directory,
resource monitoring starts automatically at session start and is finalized
and bundled at session end, so there is no need to wrap pytest with
`bash resource_monitor.sh run -- pytest ...` manually.
"""
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

STABILITY_DIR = Path(__file__).resolve().parent
RESOURCE_MONITOR_SCRIPT = STABILITY_DIR / "scripts" / "resource_monitor.sh"
REPO_ROOT = STABILITY_DIR.parent.parent.parent


def _start_resource_monitor():
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
        # Wait briefly to make sure `start` does not fail immediately.
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


def _get_monitor_data_root() -> Path:
    data_root = os.environ.get("RESOURCE_MONITOR_DATA_ROOT") or os.environ.get("GPU_MONITOR_DATA_ROOT")
    if data_root:
        return Path(data_root)
    return STABILITY_DIR / "gpu_monitor_data"


def _wait_for_run_dir(timeout_sec: int = 10) -> Path | None:
    data_root = _get_monitor_data_root()
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


def _report_latest_gpu_samples(stop_event: threading.Event) -> None:
    """Periodically print the latest sampled GPU line, similar to `run` mode."""
    log_interval = int(
        os.environ.get("RESOURCE_MONITOR_LOG_INTERVAL")
        or os.environ.get("GPU_MONITOR_LOG_INTERVAL")
        or "15"
    )
    log_interval = max(log_interval, 1)
    last_line = ""

    time.sleep(min(log_interval, 5))
    while not stop_event.wait(log_interval):
        run_dir = _wait_for_run_dir(timeout_sec=1)
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


def _finalize_resource_monitor():
    """Run `resource_monitor.sh finalize` to bundle the current run and generate the report."""
    if not RESOURCE_MONITOR_SCRIPT.is_file():
        return
    try:
        subprocess.run(
            ["bash", str(RESOURCE_MONITOR_SCRIPT), "finalize", "--backend", "gpu"],
            cwd=str(REPO_ROOT),
            capture_output=False,
            timeout=60,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        pass


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session: pytest.Session) -> None:
    """Auto-start resource monitoring at session start when the script and bash are available."""
    proc = _start_resource_monitor()
    if proc is not None:
        session._resource_monitor_process = proc  # type: ignore[attr-defined]
        stop_event = threading.Event()
        reporter = threading.Thread(
            target=_report_latest_gpu_samples,
            args=(stop_event,),
            name="stability-resource-monitor-reporter",
            daemon=True,
        )
        reporter.start()
        session._resource_monitor_stop_event = stop_event  # type: ignore[attr-defined]
        session._resource_monitor_reporter = reporter  # type: ignore[attr-defined]
        run_dir = _wait_for_run_dir(timeout_sec=5)
        bundle_root = _get_monitor_data_root()
        sys.stderr.write(
            "[Stability] Resource monitor (gpu) auto-started for this session. "
            "Bundle will be generated at session end.\n"
        )
        if run_dir is not None:
            sys.stderr.write(f"[Stability] Resource monitor run dir: {run_dir}\n")
        else:
            sys.stderr.write(
                f"[Stability] Resource monitor data root: {bundle_root} "
                "(run dir not ready yet)\n"
            )
    else:
        session._resource_monitor_process = None  # type: ignore[attr-defined]
        session._resource_monitor_stop_event = None  # type: ignore[attr-defined]
        session._resource_monitor_reporter = None  # type: ignore[attr-defined]


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Stop monitoring at session end and run finalize to generate `report.html` and related outputs."""
    stop_event = getattr(session, "_resource_monitor_stop_event", None)
    if stop_event is not None:
        stop_event.set()
    reporter = getattr(session, "_resource_monitor_reporter", None)
    if reporter is not None and reporter.is_alive():
        reporter.join(timeout=2)
    proc = getattr(session, "_resource_monitor_process", None)
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    if getattr(session, "_resource_monitor_process", None) is not None:
        sys.stderr.write("[Stability] Finalizing resource monitor (gpu)...\n")
        _finalize_resource_monitor()
