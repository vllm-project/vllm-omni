"""
Stability-specific conftest: when pytest is executed under this directory,
resource monitoring starts automatically at session start and is finalized
and bundled at session end, so there is no need to wrap pytest with
`bash resource_monitor.sh run -- pytest ...` manually.
"""
import subprocess
import sys
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
                return None
        except subprocess.TimeoutExpired:
            pass
        return proc
    except (FileNotFoundError, OSError):
        return None


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
        sys.stderr.write(
            "[Stability] Resource monitor (gpu) auto-started for this session. "
            "Bundle will be generated at session end.\n"
        )
    else:
        session._resource_monitor_process = None  # type: ignore[attr-defined]


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Stop monitoring at session end and run finalize to generate `report.html` and related outputs."""
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
