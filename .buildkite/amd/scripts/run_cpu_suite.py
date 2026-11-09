# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Run one CPU shard with opt-in timing artifacts; leave pytest selection intact."""

import argparse
import json
import os
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Generator
from datetime import datetime, timezone
from pathlib import Path

import pytest


def cpu_context() -> dict[str, object]:
    """Read the same Linux CPU constraints as run-omni-processor-test.sh."""
    context: dict[str, object] = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "cpu_count": os.cpu_count(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
    }
    try:
        context["affinity"] = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    except OSError as error:
        context["affinity"] = {"unavailable": error.strerror}
    for name, path in {
        "cpuset": "/sys/fs/cgroup/cpuset.cpus.effective",
        "cpu_max": "/sys/fs/cgroup/cpu.max",
        "cpu_stat": "/sys/fs/cgroup/cpu.stat",
        "v1_cpuset": "/sys/fs/cgroup/cpuset/cpuset.cpus",
        "v1_quota": "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
        "v1_period": "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
        "load": "/proc/loadavg",
        "cpu_pressure": "/proc/pressure/cpu",
    }.items():
        try:
            context[name] = Path(path).read_text().strip()
        except OSError as error:
            context[name] = {"unavailable": error.strerror}
    return context


class CpuSuiteArtifacts:
    def __init__(self, directory: Path, suite: str | None = None) -> None:
        self.directory = directory
        self.suite = suite
        self.started = time.perf_counter()
        self.collected: list[str] = []
        self.selected: list[str] = []
        self.not_selected: list[str] = []
        self.deselected: list[str] = []
        self.collection_errors: list[str] = []
        self.phases: dict[str, float] = defaultdict(float)
        self.outcomes: dict[str, dict[str, str]] = {}
        self.started_nodes: set[str] = set()
        self.collection_seconds: float | None = None
        self.collection_finished = False
        self.artifact_errors: set[str] = set()
        self.finished_status: int | None = None

    def write(self, name: str, data: object, *, append: bool = False) -> None:
        # Diagnostic failures must not replace a nonzero pytest result.
        try:
            with (self.directory / name).open("a" if append else "w", encoding="utf-8") as stream:
                stream.write(json.dumps(data, sort_keys=True) + "\n")
        except OSError as error:
            self.error(f"{name}: {error}")

    def error(self, message: str) -> None:
        if message not in self.artifact_errors:
            print(f"CPU timing artifact error: {message}", file=sys.stderr)
            self.artifact_errors.add(message)

    def event(self, kind: str, **fields: object) -> None:
        self.write(
            "events.jsonl",
            {"event": kind, "elapsed_seconds": time.perf_counter() - self.started, **fields},
            append=True,
        )

    def pytest_configure(self, config: pytest.Config) -> None:
        weighted = bool(config.getoption("cpu_shard_profile", default=None))
        self.write(
            "invocation.json",
            {
                "schema_version": 1,
                "suite": self.suite,
                "pytest_version": pytest.__version__,
                "args": list(config.invocation_params.args),
                "markers": config.option.markexpr,
                "run_level": config.getoption("run_level", default=None),
                "num_shards": config.getoption("cpu_shard_count" if weighted else "num_shards", default=1),
                "shard_id": config.getoption("cpu_shard_index" if weighted else "shard_id", default=0),
                "buildkite": {
                    name: os.environ.get(f"BUILDKITE_{name}")
                    for name in ("BUILD_NUMBER", "BUILD_URL", "JOB_ID", "COMMIT", "BRANCH", "LABEL", "RETRY_COUNT")
                },
                # The pod exposes a commit tag, not its resolved image digest.
                "configured_image_ref": (
                    f"rocm/vllm-omni:{os.environ['BUILDKITE_COMMIT']}" if os.environ.get("BUILDKITE_COMMIT") else None
                ),
                "image_digest": None,
            },
        )

    @pytest.hookimpl(hookwrapper=True)
    def pytest_collection(self) -> Generator[None, None, None]:
        started = time.perf_counter()
        self.event("collection_start")
        try:
            yield
        finally:
            self.collection_seconds = time.perf_counter() - started
            self.event("collection_end", seconds=self.collection_seconds)

    def pytest_itemcollected(self, item: pytest.Item) -> None:
        self.collected.append(item.nodeid)

    def pytest_deselected(self, items: list[pytest.Item]) -> None:
        self.deselected.extend(item.nodeid for item in items)

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed:
            self.collection_errors.append(report.nodeid)
            self.event("collection_error", nodeid=report.nodeid)

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.selected = [item.nodeid for item in session.items]
        # pytest-shard 0.1.2 removes items without emitting pytest_deselected.
        selected = set(self.selected)
        self.not_selected = [nodeid for nodeid in self.collected if nodeid not in selected]
        self.collection_finished = True
        self.write(
            "selection.json",
            {
                "collected": self.collected,
                "selected": self.selected,
                "not_selected": self.not_selected,
                "deselected": self.deselected,
            },
        )

    def pytest_runtest_logstart(self, nodeid: str) -> None:
        self.started_nodes.add(nodeid)
        self.event("test_start", nodeid=nodeid)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self.phases[report.when] += report.duration
        self.outcomes.setdefault(report.nodeid, {})[report.when] = report.outcome
        self.event(
            "test_report",
            nodeid=report.nodeid,
            phase=report.when,
            outcome=report.outcome,
            seconds=report.duration,
            wasxfail=getattr(report, "wasxfail", None),
        )

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> Generator[None, None, None]:
        """Persist final artifacts before later pytest teardown can exit."""
        yield
        session.exitstatus = self.finish(int(session.exitstatus))

    def finish(self, status: int) -> int:
        if self.finished_status is not None:
            return self.finished_status
        try:
            root = ET.parse(self.directory / "pytest.xml").getroot()
            if root.tag not in ("testsuites", "testsuite"):
                self.error("pytest.xml: invalid JUnit root")
        except (OSError, ET.ParseError) as error:
            self.error(f"pytest.xml: {error}")
        self.write("cpu-after.json", cpu_context())
        counts = Counter()
        for nodeid in self.started_nodes | self.outcomes.keys():
            phases = self.outcomes.get(nodeid, {})
            if "failed" in phases.values():
                counts["failed"] += 1
            elif "skipped" in phases.values():
                counts["skipped"] += 1
            elif phases.get("call") == "passed" and phases.get("teardown") == "passed":
                counts["passed"] += 1
            else:
                counts["incomplete"] += 1
        self.write(
            "summary.json",
            {
                "schema_version": 1,
                "pytest_exit_code": status,
                "collection_finished": self.collection_finished,
                "counts": {
                    "collected": len(self.collected),
                    "selected": len(self.selected),
                    "not_selected": len(self.not_selected),
                    "deselected": len(self.deselected),
                    "started": len(self.started_nodes),
                    "executed": sum("call" in phases for phases in self.outcomes.values()),
                    "collection_errors": len(self.collection_errors),
                    **{name: counts[name] for name in ("passed", "failed", "skipped", "incomplete")},
                },
                "seconds": {
                    "runner_wall": time.perf_counter() - self.started,
                    "collection": self.collection_seconds,
                    **{phase: self.phases[phase] for phase in ("setup", "call", "teardown")},
                },
                "artifact_errors": sorted(self.artifact_errors),
            },
        )
        self.finished_status = status or int(bool(self.artifact_errors))
        return self.finished_status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("model-executor", "other", "diffusion"))
    parser.add_argument("--timing-profile", type=Path, help="Use duration packing instead of hash sharding")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    artifact_root = Path(os.environ.get("BUILDKITE_BUILD_CHECKOUT_PATH", Path.cwd())) / "artifacts/amd-cpu-suites"
    # A fresh directory also isolates retries and repeated local invocations.
    identity = "-".join(
        os.environ.get(name, "local")
        for name in ("BUILDKITE_BUILD_NUMBER", "BUILDKITE_JOB_ID", "BUILDKITE_PARALLEL_JOB")
    )
    identity = "".join(char if char.isalnum() or char in "-_" else "_" for char in identity)
    try:
        artifact_root.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix=f"{args.suite}-{identity}-", dir=artifact_root))
    except OSError as error:
        print(f"Cannot create CPU timing artifacts: {error}", file=sys.stderr)
        return 1
    print(f"CPU timing artifacts: {directory}", flush=True)
    recorder = CpuSuiteArtifacts(directory, args.suite)
    recorder.write("cpu-before.json", cpu_context())
    # Match python -m pytest's import path when this script is invoked by path.
    sys.path.insert(0, str(Path.cwd()))
    pytest_args = list(args.pytest_args)
    if args.timing_profile:
        from tests.helpers.cpu_sharding import load_profile

        try:
            if load_profile(args.timing_profile).suite != args.suite:
                parser.error("Timing profile suite does not match the requested suite")
        except (OSError, ValueError) as error:
            parser.error(str(error))
        # Buildkite assigns the worker count/index; the manifest only supplies costs.
        pytest_args.extend(
            [
                f"--cpu-shard-profile={args.timing_profile}",
                f"--cpu-shard-count={os.environ.get('BUILDKITE_PARALLEL_JOB_COUNT', '1')}",
                f"--cpu-shard-index={os.environ.get('BUILDKITE_PARALLEL_JOB', '0')}",
                f"--cpu-shard-plan={directory / 'plan.json'}",
            ]
        )
    status = int(
        pytest.main([*pytest_args, "--durations=50", f"--junitxml={directory / 'pytest.xml'}"], plugins=[recorder])
    )
    return recorder.finish(status)


if __name__ == "__main__":
    raise SystemExit(main())
