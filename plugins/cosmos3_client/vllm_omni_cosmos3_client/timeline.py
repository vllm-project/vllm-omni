"""Buffered host-clock intervals for plotting inference/control overlap."""

import atexit
import json
import os
import resource
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path


class Timeline:
    def __init__(self):
        self.rows = []
        self.lock = threading.Lock()
        self.clock_domain = Path("/proc/sys/kernel/random/boot_id").read_text().strip()

    def interval(self, stage, start_s, end_s, **metadata):
        if not os.environ.get("COSMOS3_CLIENT_TIMELINE"):
            return
        row = {
            "schema": "cosmos3_overlap_v1",
            "host": socket.gethostname(),
            "clock": "CLOCK_MONOTONIC perf_counter",
            "clock_domain": self.clock_domain,
            "pid": os.getpid(),
            "thread": threading.current_thread().name,
            "stage": stage,
            "start_s": start_s,
            "end_s": end_s,
            **metadata,
        }
        with self.lock:
            self.rows.append(row)

    @contextmanager
    def span(self, stage, **metadata):
        start = time.perf_counter()
        diagnose_cpu = os.environ.get("COSMOS3_DIAG_THREAD_CPU") == "1"
        if diagnose_cpu:
            cpu_start = time.thread_time()
            usage_start = resource.getrusage(resource.RUSAGE_THREAD)
        outcome = "complete"
        try:
            yield
        except BaseException as exc:
            outcome = type(exc).__name__
            raise
        finally:
            end = time.perf_counter()
            if diagnose_cpu:
                usage_end = resource.getrusage(resource.RUSAGE_THREAD)
                metadata.update(
                    thread_cpu_ms=(time.thread_time() - cpu_start) * 1000,
                    thread_voluntary_context_switches=usage_end.ru_nvcsw - usage_start.ru_nvcsw,
                    thread_involuntary_context_switches=usage_end.ru_nivcsw - usage_start.ru_nivcsw,
                )
            self.interval(stage, start, end, outcome=outcome, **metadata)

    def flush(self):
        destination = os.environ.get("COSMOS3_CLIENT_TIMELINE")
        if not destination:
            return
        with self.lock:
            rows, self.rows = self.rows, []
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.writelines(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)

    def read_events(self):
        self.flush()
        destination = os.environ.get("COSMOS3_CLIENT_TIMELINE")
        if destination:
            with Path(destination).open() as stream:
                for line in stream:
                    if line.strip():
                        yield json.loads(line)


TIMELINE = Timeline()
atexit.register(TIMELINE.flush)
