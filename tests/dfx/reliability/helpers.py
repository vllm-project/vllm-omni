"""Shared reliability fault-injection helpers.

This module keeps fault injection callable from tests directly:
- GPU OOM (CUDA sidecar memory hog)
- process kill by pattern and signal
- post-ready hooks via ``fault_injector`` / ``omni_server_after_fault`` fixtures
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import select
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import psutil
import pytest

logger = logging.getLogger(__name__)


@dataclass
class OomHandle:
    """Handle for a started CUDA memory hog subprocess."""

    proc: subprocess.Popen | None
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


def get_health_raw(host: str, port: int, *, timeout_sec: int = 20) -> tuple[int, bytes]:
    """GET /health with stdlib HTTP client; returns (status, body)."""
    conn = http.client.HTTPConnection(host, port, timeout=timeout_sec)
    try:
        conn.request("GET", "/health")
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def post_json_raw_http_client(
    host: str,
    port: int,
    path: str,
    payload: dict[str, Any],
    *,
    timeout_sec: int = 30,
) -> tuple[int, bytes]:
    """POST JSON to one endpoint with stdlib HTTP client; returns (status, body)."""
    conn = http.client.HTTPConnection(host, port, timeout=timeout_sec)
    try:
        body = json.dumps(payload).encode("utf-8")
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def post_json_raw(
    host: str,
    port: int,
    path: str,
    payload: dict[str, Any],
    *,
    timeout_sec: int = 30,
) -> tuple[int, bytes]:
    """POST JSON to one endpoint; returns (status, body)."""
    return (
        post_chat_completions_raw(
            host,
            port,
            json.dumps(payload),
            content_type="application/json",
            timeout_sec=timeout_sec,
        )
        if path == "/v1/chat/completions"
        else post_json_raw_http_client(
            host,
            port,
            path,
            payload,
            timeout_sec=timeout_sec,
        )
    )


def extract_openai_error_contract_from_bytes(response_body: bytes) -> dict[str, Any] | None:
    """Best-effort parse OpenAI-style error object from raw response bytes."""
    try:
        payload = json.loads(response_body.decode("utf-8", errors="replace"))
    except Exception:  # noqa: BLE001
        return None
    return extract_openai_error_contract_from_payload(payload)


def extract_openai_error_contract_from_payload(payload: Any) -> dict[str, Any] | None:
    """Best-effort parse OpenAI-style error object from decoded JSON payload."""
    if not isinstance(payload, dict):
        return None
    error_obj = payload.get("error")
    if not isinstance(error_obj, dict):
        return None
    if not isinstance(error_obj.get("message"), str):
        return None
    return error_obj


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

# In strict mode, keep filling with smaller chunks until allocator rejects.
# This minimizes residual free memory and makes fault-path assertions steadier.
if strict:
    tail_chunk_bytes = [64 * 1024 * 1024, 16 * 1024 * 1024, 4 * 1024 * 1024, 1 * 1024 * 1024]
    for tail_bytes in tail_chunk_bytes:
        while True:
            req_elems = max(1, tail_bytes // 2)
            try:
                chunk = torch.empty((req_elems,), dtype=torch.float16, device=f"cuda:{device}")
                chunks.append(chunk)
                allocated += chunk.numel() * 2
            except RuntimeError:
                break

achieved_ratio = allocated / max(1, props.total_memory)
achieved_free_ratio = allocated / max(1, int(free_before))
free_after, _ = torch.cuda.mem_get_info(device)
if strict and allocated < target_bytes:
    print(
        "ERROR:"
        f"achieved_free_ratio={achieved_free_ratio:.4f};"
        f"achieved_total_ratio={achieved_ratio:.4f};"
        f"free_before={int(free_before)};"
        f"free_after={int(free_after)};"
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
    f"free_after={int(free_after)};"
    f"target_bytes={target_bytes};"
    f"allocated={allocated}",
    flush=True,
)
if hold_seconds <= 0:
    while True:
        time.sleep(3600)
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
        ``hold_seconds <= 0`` means keeping OOM pressure until the sidecar is
        explicitly stopped via ``stop_gpu_oom_hog(s)``.
    """
    if os.name == "nt":
        raise RuntimeError("CUDA OOM sidecar is intended for Linux CI/runtime.")
    if not (0.0 <= target_mem_ratio < 1.0):
        raise ValueError("target_mem_ratio should be in [0.0, 1.0).")

    # Explicit opt-out for debugging: keep API shape stable while disabling injection.
    if target_mem_ratio == 0.0:
        print(f"[oom-sidecar][gpu={device}] DISABLED: target_mem_ratio=0.0 (no OOM injection)", flush=True)
        return OomHandle(
            proc=None,
            device=device,
            target_mem_ratio=target_mem_ratio,
            start_ts=time.time(),
        )

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
    if proc is None:
        return
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
        hold_seconds: OOM hold time in seconds; ``<=0`` keeps pressure until
            ``stop_gpu_oom_hogs`` is called.
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


def _runtime_teardown_ssh_target() -> str:
    target = os.getenv("RUNTIME_TEARDOWN_SSH_TARGET", "").strip()
    # Default to root@127.0.0.1 for same-host SSH control path.
    return target or "root@127.0.0.1"


def _runtime_teardown_ssh_cmd(remote_cmd: str, *, step: str | None = None) -> subprocess.CompletedProcess[str]:
    ssh_target = _runtime_teardown_ssh_target()
    default_reuse_opts = "-o ControlMaster=auto -o ControlPersist=10m -o ControlPath=/tmp/vllm-rt-ssh-%r@%h:%p"
    raw_opts = os.getenv("RUNTIME_TEARDOWN_SSH_OPTS", "").strip()
    ssh_opts = shlex.split(raw_opts or default_reuse_opts)
    timeout_sec = int(os.getenv("RUNTIME_TEARDOWN_SSH_TIMEOUT_SEC", "600"))
    step_prefix = f"[runtime-teardown][ssh]{f'[{step}]' if step else ''}"
    print(f"{step_prefix} target={ssh_target} running remote command...", flush=True)
    # IMPORTANT: SSH joins remote argv into one shell command string. If we pass
    # ["bash", "-c", remote_cmd] as separate argv items, remote shell parsing can
    # make `-c` consume only the first word (e.g. "docker"), causing docker help.
    # Wrap the whole command as one quoted string for remote bash -c.
    remote_invocation = f"bash --noprofile --norc -c {shlex.quote(remote_cmd)}"
    try:
        out = subprocess.run(
            ["ssh", *ssh_opts, ssh_target, remote_invocation],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"{step_prefix} timed out after {timeout_sec}s. Increase RUNTIME_TEARDOWN_SSH_TIMEOUT_SEC if needed."
        ) from exc
    print(f"{step_prefix} exit_code={out.returncode}", flush=True)
    return out


def list_remote_process_pids_by_pattern(pattern: str) -> list[int]:
    """Return matched PIDs from remote host ``pgrep -f <pattern>`` via SSH."""
    cmd = f"pgrep -f {shlex.quote(pattern)} || true"
    out = _runtime_teardown_ssh_cmd(cmd, step="pgrep")
    if out.returncode not in (0, 1):
        raise RuntimeError(f"remote pgrep failed for pattern={pattern!r}: {out.stderr.strip()}")
    return [int(item) for item in out.stdout.split() if item.strip().isdigit()]


def inject_process_kill(
    *,
    grep_pattern: str,
    signal_name: str = "SIGTERM",
    limit: int | None = None,
    allow_zero_match: bool = False,
    execute_kill: bool = True,
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

    if execute_kill:
        for pid in pids:
            os.kill(pid, sig)
    return pids


def _safe_proc_info(pid: int) -> tuple[str, str]:
    """Best-effort process name/cmdline lookup for debug logging."""
    try:
        proc = psutil.Process(pid)
        name = proc.name()
        cmdline = " ".join(proc.cmdline()) or "<empty-cmdline>"
        return name, cmdline
    except Exception:  # noqa: BLE001
        return "<unknown>", "<unavailable>"


def _list_server_process_tree(server: Any) -> list[int]:
    """Return [root, descendants...] PIDs for the current test server instance."""
    root_proc = getattr(server, "proc", None)
    if root_proc is None or getattr(root_proc, "pid", None) is None:
        return []

    root_pid = int(root_proc.pid)
    try:
        root = psutil.Process(root_pid)
    except Exception:  # noqa: BLE001
        return [root_pid]

    descendants = [child.pid for child in root.children(recursive=True)]
    return [root_pid, *descendants]


def _log_server_process_tree(server: Any) -> None:
    """Print server process tree for debugging fault injection targets."""
    pids = _list_server_process_tree(server)
    if not pids:
        logger.warning("[reliability][process-kill] current server has no visible process tree")
        return
    for pid in pids:
        name, cmdline = _safe_proc_info(pid)
        print(
            f"[reliability][process-kill] current_server_proc pid={pid} name={name} cmdline={cmdline}",
            flush=True,
        )


FaultInjector = Callable[[Any], None]
"""Callable invoked with the live ``OmniServer`` after it is ready (see ``omni_server_after_fault``)."""


def make_process_kill_fault_injector(
    *,
    grep_patterns: str | Sequence[str],
    signal_name: str = "SIGKILL",
    limit: int = 1,
    post_kill_wait_seconds: float = 0.0,
) -> FaultInjector:
    """Build a post-ready injector that kills processes matched by ``pgrep -f``.

    Tries each pattern in order until at least one PID is killed. If none match,
    the returned callable issues ``pytest.skip`` (same behavior as the previous
    inline reliability test).

    Args:
        grep_patterns: One pattern or an ordered list of patterns.
        signal_name: Passed to :func:`inject_process_kill` (e.g. ``SIGKILL``).
        limit: Maximum PIDs to kill per pattern (default ``1``).
        post_kill_wait_seconds: Optional wait time after kill before test request starts.
    """
    patterns: tuple[str, ...] = (grep_patterns,) if isinstance(grep_patterns, str) else tuple(grep_patterns)

    def _inject(server: Any) -> None:
        _log_server_process_tree(server)
        server_tree = set(_list_server_process_tree(server))
        if not server_tree:
            logger.warning(
                "[reliability][process-kill] no server process tree found; fallback to global pgrep matching"
            )
        for pattern in patterns:
            print(
                f"[reliability][process-kill] trying pattern={pattern} signal={signal_name} limit={limit}",
                flush=True,
            )
            pids = inject_process_kill(
                grep_pattern=pattern,
                signal_name=signal_name,
                limit=limit,
                allow_zero_match=True,
                execute_kill=False,
            )
            filtered = [pid for pid in pids if not server_tree or pid in server_tree]
            if pids and not filtered:
                logger.warning(
                    "[reliability][process-kill] pattern=%s matched non-server pids=%s, skip them",
                    pattern,
                    pids,
                )
                continue
            if filtered:
                sig = getattr(signal, signal_name, None)
                if sig is None:
                    raise ValueError(f"Unsupported signal_name: {signal_name}")
                for pid in filtered:
                    name, cmdline = _safe_proc_info(pid)
                    print(
                        f"[reliability][process-kill] killing pid={pid} name={name} signal={signal_name} cmdline={cmdline}",
                        flush=True,
                    )
                    os.kill(pid, sig)
                print(
                    f"[reliability][process-kill] matched pattern={pattern} killed_pids={filtered} killed_count={len(filtered)}",
                    flush=True,
                )
                if post_kill_wait_seconds > 0:
                    print(
                        f"[reliability][process-kill] waiting {post_kill_wait_seconds:.2f}s after kill",
                        flush=True,
                    )
                    time.sleep(post_kill_wait_seconds)
                return
        logger.warning(
            "[reliability][process-kill] no process matched patterns=%s signal=%s limit=%s",
            patterns,
            signal_name,
            limit,
        )
        pytest.skip("no matching runtime process found for kill injection")

    return _inject
