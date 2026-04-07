"""Shared reliability fault-injection helpers.

This module keeps fault injection callable from tests directly:
- abnormal input (raw HTTP malformed requests)
- GPU OOM (CUDA sidecar memory hog)
- process kill by pattern and signal
"""

from __future__ import annotations

import http.client
import json
import os
import select
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

FaultVariant = str


@dataclass(frozen=True)
class FaultPhaseResult:
    """Observed HTTP statuses during abnormal-input injection."""

    http_statuses: list[int]
    notes: str


@dataclass(frozen=True)
class RecoveryResult:
    """Structured outcome after fault injection and follow-up request."""

    recovered: bool
    recovery_time_sec: float | None
    health_check_ok: bool
    post_fault_success_count: int
    post_fault_error_count: int
    notes: str


@dataclass
class OomHandle:
    """Handle for a started CUDA memory hog subprocess."""

    proc: subprocess.Popen
    device: int
    target_mem_ratio: float
    start_ts: float


def post_chat_completions_raw(
    host: str,
    port: int,
    body: bytes | str,
    *,
    content_type: str = "application/json",
    timeout_sec: int = 120,
) -> tuple[int, bytes]:
    """POST /v1/chat/completions with raw bytes; returns (status, response_body)."""
    conn = http.client.HTTPConnection(host, port, timeout=timeout_sec)
    try:
        headers = {"Content-Type": content_type}
        payload = body.encode("utf-8") if isinstance(body, str) else body
        conn.request("POST", "/v1/chat/completions", body=payload, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, data
    finally:
        conn.close()


def inject_abnormal_input_faults(
    host: str,
    port: int,
    model: str,
    fault_params: dict[str, Any],
) -> FaultPhaseResult:
    """Run configured abnormal-input variants against the running server."""
    variants: list[FaultVariant] = fault_params.get(
        "variants",
        ["malformed_json", "invalid_messages_type"],
    )
    statuses: list[int] = []
    notes_parts: list[str] = []

    for key in variants:
        if key == "malformed_json":
            status, _ = post_chat_completions_raw(host, port, b"not json {{{")
            statuses.append(status)
            notes_parts.append(f"malformed_json:{status}")
        elif key == "invalid_messages_type":
            bad = json.dumps({"model": model, "messages": "must_be_a_list_not_string"})
            status, body = post_chat_completions_raw(host, port, bad)
            statuses.append(status)
            notes_parts.append(f"invalid_messages_type:{status}:{len(body)}B")
        else:
            raise ValueError(f"Unknown abnormal_input variant: {key!r}")

    return FaultPhaseResult(http_statuses=statuses, notes="; ".join(notes_parts))


def assert_fault_http_expectation(statuses: list[int], expect: dict[str, Any]) -> None:
    """Assert fault-phase HTTP statuses match ``expect.fault_http_status_class``."""
    cls = expect.get("fault_http_status_class", "4xx")
    if cls == "4xx":
        for status in statuses:
            assert 400 <= status < 500, f"expected 4xx for abnormal input, got {status}"
        return
    raise ValueError(f"Unsupported fault_http_status_class: {cls!r}")


def build_recovery_result(
    *,
    recovered: bool,
    recovery_time_sec: float | None,
    health_check_ok: bool,
    post_fault_success_count: int,
    post_fault_error_count: int,
    notes: str,
) -> RecoveryResult:
    """Factory for RecoveryResult."""
    return RecoveryResult(
        recovered=recovered,
        recovery_time_sec=recovery_time_sec,
        health_check_ok=health_check_ok,
        post_fault_success_count=post_fault_success_count,
        post_fault_error_count=post_fault_error_count,
        notes=notes,
    )


def _build_sidecar_cmd(device: int, target_mem_ratio: float, hold_seconds: int, strict: bool) -> list[str]:
    sidecar = r"""
import sys
import time
import torch

device = int(sys.argv[1])
target_ratio = float(sys.argv[2])
hold_seconds = int(sys.argv[3])
strict = sys.argv[4] == "1"

torch.cuda.init()
torch.cuda.set_device(device)
props = torch.cuda.get_device_properties(device)
free_before, total_bytes = torch.cuda.mem_get_info(device)
target_bytes = int(free_before * target_ratio)
chunk_bytes = 256 * 1024 * 1024
chunks = []
allocated = 0

while allocated < target_bytes:
    req_bytes = min(chunk_bytes, target_bytes - allocated)
    req_elems = max(1, req_bytes // 2)  # float16 -> 2 bytes
    try:
        chunk = torch.empty((req_elems,), dtype=torch.float16, device=f"cuda:{device}")
        chunks.append(chunk)
        allocated += chunk.numel() * 2
    except RuntimeError:
        break

achieved_ratio = allocated / max(1, props.total_memory)
achieved_free_ratio = allocated / max(1, int(free_before))
if strict and allocated < target_bytes:
    print(
        "ERROR:"
        f"achieved_free_ratio={achieved_free_ratio:.4f};"
        f"achieved_total_ratio={achieved_ratio:.4f};"
        f"free_before={int(free_before)};"
        f"target_bytes={target_bytes};"
        f"allocated={allocated}",
        flush=True,
    )
    sys.exit(2)

print(
    "READY:"
    f"achieved_free_ratio={achieved_free_ratio:.4f};"
    f"achieved_total_ratio={achieved_ratio:.4f};"
    f"free_before={int(free_before)};"
    f"target_bytes={target_bytes};"
    f"allocated={allocated}",
    flush=True,
)
time.sleep(hold_seconds)
print("DONE", flush=True)
"""
    return [
        sys.executable,
        "-u",
        "-c",
        sidecar,
        str(device),
        str(target_mem_ratio),
        str(hold_seconds),
        "1" if strict else "0",
    ]


def start_gpu_oom_hog(
    *,
    device: int = 0,
    target_mem_ratio: float = 0.95,
    hold_seconds: int = 60,
    startup_timeout_sec: int = 20,
    strict: bool = True,
    poll_interval_sec: float = 0.2,
) -> OomHandle:
    """Start a CUDA sidecar process that occupies GPU memory to trigger OOM.

    Note:
        ``target_mem_ratio`` is evaluated against free memory at injection start
        (not total GPU memory), i.e. success gate is ``allocated / free_before``.
    """
    if os.name == "nt":
        raise RuntimeError("CUDA OOM sidecar is intended for Linux CI/runtime.")
    if not (0.1 <= target_mem_ratio < 1.0):
        raise ValueError("target_mem_ratio should be in [0.1, 1.0).")

    cmd = _build_sidecar_cmd(device, target_mem_ratio, hold_seconds, strict)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None

    deadline = time.time() + startup_timeout_sec
    logs: list[str] = []
    while time.time() < deadline:
        ready, _, _ = select.select([proc.stdout], [], [], poll_interval_sec)
        if ready:
            line = proc.stdout.readline().strip()
            if line:
                logs.append(line)
                print(f"[oom-sidecar][gpu={device}] {line}", flush=True)
                if line.startswith("READY:"):
                    return OomHandle(
                        proc=proc,
                        device=device,
                        target_mem_ratio=target_mem_ratio,
                        start_ts=time.time(),
                    )
                if line.startswith("ERROR:"):
                    proc.terminate()
                    raise RuntimeError(f"OOM sidecar failed to reach target: {line}")
        if proc.poll() is not None:
            break

    proc.terminate()
    if logs:
        print(f"[oom-sidecar][gpu={device}] startup logs: {' | '.join(logs)}", flush=True)
    raise TimeoutError(f"OOM sidecar startup timeout. logs={logs}")


def stop_gpu_oom_hog(handle: OomHandle, *, timeout_sec: int = 5) -> None:
    """Stop and cleanup CUDA OOM sidecar."""
    proc = handle.proc
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout_sec)


def inject_gpu_oom(
    *,
    device: int | str | list[int] = 0,
    target_mem_ratio: float = 0.95,
    hold_seconds: int = 60,
    startup_timeout_sec: int = 20,
    strict: bool = True,
) -> OomHandle | list[OomHandle]:
    """Convenience wrapper to start CUDA OOM sidecar(s).

    Args:
        device: One device id (``0``), comma-separated string (``"0,1,2"``),
            or a list of device ids (``[0, 1, 2]``).
    """
    if isinstance(device, int):
        devices = [device]
    elif isinstance(device, str):
        devices = [int(x.strip()) for x in device.split(",") if x.strip()]
    else:
        devices = [int(x) for x in device]
    if not devices:
        raise ValueError("device must not be empty.")

    handles = [
        start_gpu_oom_hog(
            device=dev,
            target_mem_ratio=target_mem_ratio,
            hold_seconds=hold_seconds,
            startup_timeout_sec=startup_timeout_sec,
            strict=strict,
        )
        for dev in devices
    ]
    if len(handles) == 1:
        return handles[0]
    return handles


def stop_gpu_oom_hogs(handles: OomHandle | list[OomHandle], *, timeout_sec: int = 5) -> None:
    """Stop one or multiple OOM sidecars."""
    if isinstance(handles, OomHandle):
        stop_gpu_oom_hog(handles, timeout_sec=timeout_sec)
        return
    for handle in handles:
        stop_gpu_oom_hog(handle, timeout_sec=timeout_sec)


def inject_process_kill(
    *,
    grep_pattern: str,
    signal_name: str = "SIGTERM",
    limit: int | None = None,
    allow_zero_match: bool = False,
) -> list[int]:
    """Kill processes matching pattern with selected signal."""
    if os.name == "nt":
        raise RuntimeError("process-kill helper currently supports POSIX platforms only.")
    if not grep_pattern.strip():
        raise ValueError("grep_pattern must not be empty.")

    sig = getattr(signal, signal_name, None)
    if sig is None:
        raise ValueError(f"Unsupported signal_name: {signal_name}")

    out = subprocess.run(
        ["pgrep", "-f", grep_pattern],
        check=False,
        capture_output=True,
        text=True,
    )
    pids = [int(item) for item in out.stdout.split() if item.strip().isdigit()]
    if limit is not None:
        pids = pids[:limit]

    if not pids and not allow_zero_match:
        raise RuntimeError(f"No process matched pattern: {grep_pattern}")

    for pid in pids:
        os.kill(pid, sig)
    return pids
