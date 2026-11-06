# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Record per-file pytest wall time for automatic_shard planning.

Loaded with ``pytest -p pytest_file_timings`` after
``.buildkite/common/scripts`` is on ``PYTHONPATH``. Each process writes
``timings/jobs/<job>-<pid>.json`` and refreshes it after every test, including
on SIGTERM, so a later pytest in the same Buildkite job does not replace the
first and a job timeout still leaves the files already finished.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import signal
import time
from pathlib import Path

OUTPUT_DIR = Path("timings/jobs")
SCHEMA = "vllm_omni.shard_timings.job.v1"
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
_started: dict[str, float] = {}
_timings: dict[str, float] = {}
_hooks_installed = False


def _output_path() -> Path:
    job = os.environ.get("BUILDKITE_JOB_ID") or "local"
    name = _SAFE.sub("-", f"{job}-{os.getpid()}")
    return OUTPUT_DIR / f"{name}.json"


def _write() -> None:
    if not _timings:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "job_id": os.environ.get("BUILDKITE_JOB_ID", ""),
        "build_id": os.environ.get("BUILDKITE_BUILD_ID", ""),
        "build_number": os.environ.get("BUILDKITE_BUILD_NUMBER", ""),
        "step_key": os.environ.get("BUILDKITE_STEP_KEY", ""),
        "label": os.environ.get("BUILDKITE_LABEL", ""),
        "file_seconds": {key: round(value, 3) for key, value in sorted(_timings.items())},
    }
    _output_path().write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _on_signal(signum: int, _frame) -> None:
    _write()
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def pytest_configure(config) -> None:
    global _hooks_installed
    if _hooks_installed:
        return
    _hooks_installed = True
    atexit.register(_write)
    for signum in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
        if signum is None:
            continue
        try:
            signal.signal(signum, _on_signal)
        except (OSError, ValueError):
            continue


def pytest_runtest_logstart(nodeid: str, location) -> None:
    _started[nodeid] = time.perf_counter()


def pytest_runtest_logfinish(nodeid: str, location) -> None:
    begin = _started.pop(nodeid, None)
    if begin is None:
        return
    file = nodeid.split("::", 1)[0].replace("\\", "/")
    _timings[file] = _timings.get(file, 0.0) + (time.perf_counter() - begin)
    _write()


def pytest_sessionfinish(session, exitstatus: int) -> None:
    _write()
