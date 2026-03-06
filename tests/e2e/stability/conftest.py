"""
Stability 目录专用 conftest：当在此目录下执行 pytest 时，自动在 session 开始时启动资源监控、
session 结束时结束并打包（finalize），无需再手写 bash resource_monitor.sh run -- pytest ...
"""
import subprocess
import sys
from pathlib import Path

import pytest

STABILITY_DIR = Path(__file__).resolve().parent
RESOURCE_MONITOR_SCRIPT = STABILITY_DIR / "scripts" / "resource_monitor.sh"
REPO_ROOT = STABILITY_DIR.parent.parent.parent


def _start_resource_monitor():
    """后台启动 resource_monitor.sh start，返回 Popen 或 None（未启动时）。"""
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
        # 短暂等待，确认 start 未立即失败
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
    """执行 resource_monitor.sh finalize，打包当前 run 并生成 report。"""
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
    """Session 开始时自动启动资源监控（仅当运行本目录下用例且脚本存在、bash 可用时）。"""
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
    """Session 结束时结束监控进程并执行 finalize，生成 report.html 等。"""
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
