# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Opt-in, bounded pytest phase/GC metadata. No test behavior is replaced."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

_KEY: pytest.StashKey[Observer | None] = pytest.StashKey()
_SUMMARY_RESERVE = 4096
_CHECKPOINT_RESERVE = 4096
_TOTAL_RESERVE = _SUMMARY_RESERVE + 2 * _CHECKPOINT_RESERVE
_VERSION = 2
_OUTPUT_ENV = "NATIVE_PHASE_PROBE_OUTPUT_DIR"


def _write_terminal(config: pytest.Config, message: str) -> None:
    try:
        reporter = config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line("native-phase-probe: " + message)
        else:
            sys.stderr.write("native-phase-probe: " + message + "\n")
    except Exception:
        pass


class Observer:
    def __init__(self, config: pytest.Config) -> None:
        self.config = config
        self.root = Path(config.rootpath).resolve()
        # Environment configuration leaves pytest's original argv and rootdir
        # discovery intact. A custom path-valued CLI option can affect rootdir
        # discovery before an external plugin's options have been registered.
        self.slow = float(os.environ.get("NATIVE_PHASE_PROBE_SLOW_SECONDS", "1.0"))
        self.max_events = int(os.environ.get("NATIVE_PHASE_PROBE_MAX_EVENTS", "2000"))
        self.max_bytes = int(os.environ.get("NATIVE_PHASE_PROBE_MAX_BYTES", str(2 * 1024 * 1024)))
        self.checkpoint_interval = int(os.environ.get("NATIVE_PHASE_PROBE_CHECKPOINT_REPORTS", "1000"))
        if not math.isfinite(self.slow) or self.slow < 0:
            raise ValueError("invalid slow threshold")
        if not 1 <= self.max_events <= 100000:
            raise ValueError("invalid event limit")
        if not 16384 <= self.max_bytes <= 64 * 1024 * 1024:
            raise ValueError("invalid byte limit")
        if not 1 <= self.checkpoint_interval <= 100000:
            raise ValueError("invalid checkpoint interval")
        directory = Path(os.environ[_OUTPUT_ENV]).resolve(strict=True)
        if not directory.is_dir() or directory.stat().st_uid != os.getuid():
            raise ValueError("output must be a caller-owned existing directory")
        if directory == self.root or self.root in directory.parents:
            raise ValueError("output must be outside pytest rootpath")
        # A private new directory prevents overwriting caller files and prevents
        # pre-existing event/summary symlinks from redirecting writes.
        self.directory = Path(tempfile.mkdtemp(prefix="native-phase-probe-", dir=directory))
        self.events = (self.directory / "events.jsonl").open("x", encoding="utf-8")
        self.started_ns = time.perf_counter_ns()
        self.active: dict[str, Any] | None = None
        self.pending: dict[tuple[str, str], dict[str, Any]] = {}
        self.sequence = 0
        self.collected = 0
        self.emitted = 0
        self.eligible_events = 0
        self.last_emitted_offset: float | None = None
        self.first_omitted_offset: float | None = None
        self.event_bytes = 0
        self.omissions = {
            "below_threshold": 0,
            "event_limit": 0,
            "byte_limit": 0,
            "io_error": 0,
            "missing_phase_context": 0,
            "gc_unmatched_start": 0,
            "gc_unmatched_stop": 0,
            "gc_cross_phase": 0,
            "unreported_phase_contexts": 0,
            "checkpoint_error": 0,
        }
        self.phases: dict[str, dict[str, Any]] = {
            name: {
                "reports": 0,
                "outcomes": {"passed": 0, "failed": 0, "skipped": 0},
                "duration_seconds": 0.0,
                "max_duration_seconds": 0.0,
                "gc_count": 0,
                "gc_seconds": 0.0,
            }
            for name in ("setup", "call", "teardown")
        }
        self.gc_stats = {
            str(g): {"count": 0, "seconds": 0.0, "max_seconds": 0.0, "collected": 0, "uncollectable": 0}
            for g in range(3)
        }
        self.gc_started: tuple[int, int, dict[str, Any] | None] | None = None
        self.callback = self._gc_callback
        self.attached = False
        self.closed = False
        self.summary_written = False
        self.exitstatus: int | None = None
        self.observer_errors = 0
        self.io_disabled = False
        self.callback_removed = False
        self.report_count = 0
        self.last_report_offset: float | None = None
        self.checkpoint_count = 0
        self.last_checkpoint_reports = 0
        self.last_checkpoint_attempt_reports: int | None = None
        self.checkpoint_disabled = False
        gc.callbacks.append(self.callback)
        self.attached = True

    def _gc_callback(self, phase: str, info: dict[str, int]) -> None:
        # No object inspection, payloads, stack reads, file I/O or gc.collect.
        try:
            now = time.perf_counter_ns()
            generation = info.get("generation", -1)
            if generation not in (0, 1, 2):
                return
            if phase == "start":
                if self.gc_started is not None:
                    self.omissions["gc_unmatched_start"] += 1
                self.gc_started = (now, generation, self.active)
            elif phase == "stop":
                started = self.gc_started
                self.gc_started = None
                if started is None or started[1] != generation:
                    self.omissions["gc_unmatched_stop"] += 1
                    return
                elapsed = max(0, now - started[0]) / 1e9
                stats = self.gc_stats[str(generation)]
                stats["count"] += 1
                stats["seconds"] += elapsed
                stats["max_seconds"] = max(stats["max_seconds"], elapsed)
                stats["collected"] += max(0, info.get("collected", 0))
                stats["uncollectable"] += max(0, info.get("uncollectable", 0))
                context = started[2]
                if context is not None and context is self.active:
                    context["gc_count"] += 1
                    context["gc_seconds"] += elapsed
                    context["gc_max_seconds"] = max(context["gc_max_seconds"], elapsed)
                elif context is not self.active:
                    self.omissions["gc_cross_phase"] += 1
        except Exception:
            self.observer_errors += 1

    def begin(self, item: pytest.Item, phase: str) -> dict[str, Any] | None:
        try:
            self.sequence += 1
            try:
                filename = item.path.resolve().relative_to(self.root).as_posix()
            except ValueError:
                filename = "<outside-root>"
            # Parameters can contain user payloads. Keep only a hash of the full
            # node identifier; the ordinary test function name has no parameter ID.
            base_name = item.name.split("[", 1)[0]
            context = {
                "sequence": self.sequence,
                "node_sha256": hashlib.sha256(item.nodeid.encode()).hexdigest(),
                "test_file": filename[:512],
                "test_name": re.sub(r"[^A-Za-z0-9_.]", "_", base_name)[:160],
                "phase": phase,
                "start_offset_seconds": (time.perf_counter_ns() - self.started_ns) / 1e9,
                "gc_count": 0,
                "gc_seconds": 0.0,
                "gc_max_seconds": 0.0,
            }
            self.active = context
            return context
        except Exception:
            self.observer_errors += 1
            return None

    def end(self, item: pytest.Item, phase: str, context: dict[str, Any] | None) -> None:
        try:
            self.active = None
            if context is not None:
                self.pending[(item.nodeid, phase)] = context
        except Exception:
            self.observer_errors += 1

    def report(self, report: pytest.TestReport) -> None:
        try:
            if report.when not in self.phases:
                return
            self.report_count += 1
            self.last_report_offset = (time.perf_counter_ns() - self.started_ns) / 1e9
            stats = self.phases[report.when]
            stats["reports"] += 1
            if report.outcome in stats["outcomes"]:
                stats["outcomes"][report.outcome] += 1
            duration = max(0.0, float(report.duration))
            stats["duration_seconds"] += duration
            stats["max_duration_seconds"] = max(stats["max_duration_seconds"], duration)
            context = self.pending.pop((report.nodeid, report.when), None)
            if context is None:
                self.omissions["missing_phase_context"] += 1
                return
            stats["gc_count"] += context["gc_count"]
            stats["gc_seconds"] += context["gc_seconds"]
            if duration < self.slow and report.outcome == "passed":
                self.omissions["below_threshold"] += 1
                return
            self.eligible_events += 1
            self.emit({**context, "outcome": report.outcome, "duration_seconds": duration})
        except Exception:
            self.observer_errors += 1
        finally:
            if (
                self.report_count
                and self.report_count % self.checkpoint_interval == 0
                and self.last_checkpoint_attempt_reports != self.report_count
            ):
                self.checkpoint(final=False)

    def emit(self, event: dict[str, Any]) -> None:
        if self.io_disabled:
            if self.first_omitted_offset is None:
                self.first_omitted_offset = event["start_offset_seconds"]
            self.omissions["io_error"] += 1
            return
        if self.emitted >= self.max_events:
            if self.first_omitted_offset is None:
                self.first_omitted_offset = event["start_offset_seconds"]
            self.omissions["event_limit"] += 1
            return
        encoded = json.dumps(event, ensure_ascii=True, separators=(",", ":")) + "\n"
        size = len(encoded.encode("utf-8"))
        if self.event_bytes + size > self.max_bytes - _TOTAL_RESERVE:
            if self.first_omitted_offset is None:
                self.first_omitted_offset = event["start_offset_seconds"]
            self.omissions["byte_limit"] += 1
            return
        try:
            self.events.write(encoded)
            self.events.flush()
            self.event_bytes += size
            self.emitted += 1
            self.last_emitted_offset = event["start_offset_seconds"]
        except Exception:
            self.io_disabled = True
            if self.first_omitted_offset is None:
                self.first_omitted_offset = event["start_offset_seconds"]
            self.omissions["io_error"] += 1

    def checkpoint(self, final: bool) -> None:
        """Atomically retain a bounded observed prefix, without signal hooks."""
        if self.checkpoint_disabled:
            return
        self.last_checkpoint_attempt_reports = self.report_count
        temporary = self.directory / "checkpoint.tmp"
        snapshot = {
            "schema_version": _VERSION,
            "kind": "phase_gc_checkpoint",
            "final": final,
            "observed_prefix_reports": self.report_count,
            "observed_prefix_last_report_offset_seconds": self.last_report_offset,
            "snapshot_offset_seconds": (time.perf_counter_ns() - self.started_ns) / 1e9,
            "checkpoint_sequence": self.checkpoint_count + 1,
            "collected": self.collected,
            "phases": self.phases,
            "gc_by_generation": self.gc_stats,
            "eligible_phase_events": self.eligible_events,
            "events_emitted": self.emitted,
            "eligible_phase_events_omitted": self.eligible_events - self.emitted,
            "event_bytes": self.event_bytes,
            "omissions": self.omissions,
            "observer_errors": self.observer_errors,
            "limits": {
                "total_bytes": self.max_bytes,
                "event_budget_bytes": self.max_bytes - _TOTAL_RESERVE,
                "checkpoint_bytes": _CHECKPOINT_RESERVE,
                "temporary_checkpoint_bytes": _CHECKPOINT_RESERVE,
                "final_summary_bytes": _SUMMARY_RESERVE,
                "report_interval": self.checkpoint_interval,
            },
            "meaning": "Observed prefix only. final marks observer session finish, not full-suite completion or acceptance. Detail caps may omit later slow phases. No root-cause claim.",
        }
        try:
            encoded = json.dumps(snapshot, ensure_ascii=True, separators=(",", ":")) + "\n"
            if len(encoded.encode()) > _CHECKPOINT_RESERVE:
                raise ValueError("checkpoint reserve exceeded")
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.directory / "checkpoint.json")
            self.checkpoint_count += 1
            self.last_checkpoint_reports = self.report_count
        except Exception:
            # Leave the previous complete checkpoint intact. A partial temp file
            # remains bounded by its separate reserve. Do not retry unhealthy I/O.
            self.omissions["checkpoint_error"] += 1
            self.checkpoint_disabled = True
            _write_terminal(
                self.config,
                "checkpoint unavailable; previous complete prefix retained; further checkpoint writes disabled",
            )

    def detach(self) -> None:
        if not self.attached:
            return
        # Preserve unrelated GC callbacks, regardless of their registration order.
        gc.callbacks[:] = [callback for callback in gc.callbacks if callback is not self.callback]
        self.attached = False
        self.callback_removed = all(callback is not self.callback for callback in gc.callbacks)
        if self.gc_started is not None:
            self.omissions["gc_unmatched_start"] += 1
            self.gc_started = None

    def finish(self, exitstatus: int | None = None) -> None:
        if self.closed:
            return
        self.exitstatus = exitstatus
        self.detach()
        self.active = None
        self.omissions["unreported_phase_contexts"] += len(self.pending)
        self.pending.clear()
        try:
            self.events.close()
        except Exception:
            self.omissions["io_error"] += 1
        self.checkpoint(final=True)
        summary = {
            "schema_version": _VERSION,
            "pid": os.getpid(),
            "pytest_version": pytest.__version__,
            "final": True,
            "observed_prefix_reports": self.report_count,
            "observed_prefix_last_report_offset_seconds": self.last_report_offset,
            "collected": self.collected,
            "exitstatus": self.exitstatus,
            "phases": self.phases,
            "gc_by_generation": self.gc_stats,
            "events_emitted": self.emitted,
            "eligible_phase_events": self.eligible_events,
            "eligible_phase_events_omitted": self.eligible_events - self.emitted,
            "last_emitted_start_offset_seconds": self.last_emitted_offset,
            "first_omitted_eligible_start_offset_seconds": self.first_omitted_offset,
            "event_bytes": self.event_bytes,
            "limits": {
                "events": self.max_events,
                "total_bytes": self.max_bytes,
                "summary_reserved_bytes": _SUMMARY_RESERVE,
                "checkpoint_reserved_bytes": _CHECKPOINT_RESERVE,
                "checkpoint_temp_reserved_bytes": _CHECKPOINT_RESERVE,
                "event_budget_bytes": self.max_bytes - _TOTAL_RESERVE,
                "checkpoint_report_interval": self.checkpoint_interval,
                "slow_seconds": self.slow,
            },
            "checkpoints_written": self.checkpoint_count,
            "last_checkpoint_reports": self.last_checkpoint_reports,
            "omissions": self.omissions,
            "observer_errors": self.observer_errors,
            "gc_callback_removed": self.callback_removed,
            "passive_stacks": "not_collected",
            "detail_retention": "Initial eligible reports, not a sample or complete file inventory. Count/byte caps can omit later slow files. A file absent from events cannot be classified as fast, unvisited, skipped or passed; inspect omissions and aggregate counts first.",
            "limits_of_inference": "Observer perturbs timing. GC time is wall time between callback entries, including concurrent/background activity. Log events omit parameters, captured output, exceptions, locals and stacks. Aggregates retain all observed phase reports even when event limits omit detail. Native L4 execution is still required; this diagnostic establishes neither root cause nor acceptance.",
        }
        try:
            data = json.dumps(summary, ensure_ascii=True, separators=(",", ":")) + "\n"
            if len(data.encode()) > _SUMMARY_RESERVE:
                raise ValueError("summary reserve exceeded")
            with (self.directory / "summary.json").open("x", encoding="utf-8") as handle:
                handle.write(data)
            self.summary_written = True
        except Exception:
            self.omissions["io_error"] += 1
        self.closed = True


def pytest_configure(config: pytest.Config) -> None:
    config.stash[_KEY] = None
    if _OUTPUT_ENV not in os.environ:
        return
    try:
        config.stash[_KEY] = Observer(config)
    except Exception as error:
        # Observer failures never replace the test session's exit status.
        _write_terminal(
            config, "disabled; output unavailable or invalid (" + type(error).__name__ + "); no observations collected"
        )


def pytest_collection_finish(session: pytest.Session) -> None:
    observer = session.config.stash.get(_KEY, None)
    if observer is not None:
        observer.collected = len(session.items)


def _begin(item: pytest.Item, phase: str) -> tuple[Observer | None, dict[str, Any] | None]:
    observer = item.config.stash.get(_KEY, None)
    return observer, observer.begin(item, phase) if observer is not None else None


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_setup(item: pytest.Item):
    observer, context = _begin(item, "setup")
    try:
        yield
    finally:
        if observer is not None:
            observer.end(item, "setup", context)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_call(item: pytest.Item):
    observer, context = _begin(item, "call")
    try:
        yield
    finally:
        if observer is not None:
            observer.end(item, "call", context)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_teardown(item: pytest.Item):
    observer, context = _begin(item, "teardown")
    try:
        yield
    finally:
        if observer is not None:
            observer.end(item, "teardown", context)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    observer = item.config.stash.get(_KEY, None)
    if observer is not None and outcome.excinfo is None:
        observer.report(outcome.get_result())


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    observer = session.config.stash.get(_KEY, None)
    if observer is not None:
        observer.finish(int(exitstatus))


def pytest_terminal_summary(terminalreporter, exitstatus: int, config: pytest.Config) -> None:
    observer = config.stash.get(_KEY, None)
    if observer is not None:
        terminalreporter.write_line(
            "native-phase-probe: bounded phase/GC metadata; events="
            + str(observer.emitted)
            + "; omissions="
            + json.dumps(observer.omissions, sort_keys=True)
            + "; summary_written="
            + str(observer.summary_written)
        )


def pytest_unconfigure(config: pytest.Config) -> None:
    observer = config.stash.get(_KEY, None)
    if observer is not None:
        observer.finish()
