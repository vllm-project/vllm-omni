# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Record opt-in CI selection and execution without changing pytest verdicts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest


class CaseRecorder:
    def __init__(self, config: pytest.Config, directory: Path) -> None:
        self.controller = bool(config.getoption("numprocesses", default=0)) and not hasattr(config, "workerinput")
        self.nodeids: dict[str, str] = {}
        self.write_failed = False
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{uuid4().hex}.jsonl"
        self.emit(
            {
                "event": "session",
                "schema_version": 1,
                "commit": os.environ.get("BUILDKITE_COMMIT", ""),
                "job_id": os.environ.get("BUILDKITE_JOB_ID", ""),
                "job_label": os.environ.get("BUILDKITE_LABEL", ""),
                "platform": os.environ.get("VLLM_OMNI_CI_CASE_PLATFORM", ""),
                "tier": os.environ.get("VLLM_OMNI_CI_CASE_TIER", ""),
                "blocking": os.environ.get("VLLM_OMNI_CI_CASE_BLOCKING", ""),
                "collect_only": bool(config.option.collectonly),
                "controller": self.controller,
            }
        )

    def emit(self, event: dict[str, object]) -> None:
        # Flush each event so a timeout leaves useful but explicitly incomplete evidence.
        if self.write_failed:
            return
        try:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=True, allow_nan=False) + "\n")
        except OSError as error:
            self.write_failed = True
            print(f"CI case reporting stopped: {error}; evidence is incomplete.", file=sys.stderr)

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        if not self.controller:
            for item in session.items:
                nodeid = item.nodeid
                if session.config.getoption("loadgroup", default=False):
                    names = {
                        str(mark.args[0] if mark.args else mark.kwargs.get("name", "default"))
                        for mark in item.iter_markers("xdist_group")
                    }
                    if names:
                        nodeid = nodeid.removesuffix("@" + "_".join(sorted(names)))
                self.nodeids[item.nodeid] = nodeid
            self.emit({"event": "selection", "nodeids": list(self.nodeids.values())})

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed or report.skipped:
            self.emit({"event": "collection_issue", "nodeid": report.nodeid, "outcome": report.outcome})

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if self.controller:
            return  # Workers record their own reports; do not double-count xdist forwarding.
        event: dict[str, object] = {
            "event": "result",
            "nodeid": self.nodeids.get(report.nodeid, report.nodeid),
            "phase": report.when,
            "outcome": report.outcome,
            "duration_seconds": report.duration,
        }
        if report.skipped:
            event["reason"] = str(report.longrepr)
        if hasattr(report, "wasxfail"):
            event["xfail"] = str(report.wasxfail)
        self.emit(event)

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        self.emit({"event": "finish", "exitstatus": int(exitstatus)})


def pytest_configure(config: pytest.Config) -> None:
    destination = os.environ.get("VLLM_OMNI_CI_CASE_DIR")
    if not destination:
        return
    directory = Path(destination)
    if not directory.is_absolute():
        directory = Path(os.environ.get("BUILDKITE_BUILD_CHECKOUT_PATH", str(config.rootpath))) / directory
    try:
        recorder = CaseRecorder(config, directory)
    except OSError as error:
        print(f"CI case reporting unavailable: {error}; no execution evidence.", file=sys.stderr)
        return
    config.pluginmanager.register(recorder, "ci-case-recorder")
