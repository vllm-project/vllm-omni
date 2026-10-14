"""Shared reliability fault-injection helpers.

This module keeps fault injection callable from tests directly:
- GPU OOM (CUDA sidecar memory hog)
- process kill by pattern and signal
- post-ready hooks via ``fault_injector`` / ``omni_server_after_fault`` fixtures

Worker PID classification (``worker_pids`` in snapshots) is still used for logging /
markers; post-fault **process** cleanup assertions use the full captured
``tree_pids`` (serve root + descendants at fault time) — see
:func:`assert_no_server_tree_process_residual_and_gpu_release`.
"""

from __future__ import annotations

import concurrent.futures
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

# Substrings matched against ``psutil.Process.name`` + argv text for classifying GPU
# workers under the test server tree. Extend per-suite via
# ``server.reliability_worker_markers_extra`` or replace via
# ``server.reliability_worker_markers`` (see ``_resolve_runtime_worker_markers``).
DEFAULT_RUNTIME_WORKER_MARKERS: tuple[str, ...] = (
    "multiprocessing.spawn",
    # Covers ``VLLM::Worker``, ``VLLM::StageEngineCoreProc_*`` (Omni stage engines), etc.
    "VLLM::",
)


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


def _normalize_worker_marker_substrings(seq: Sequence[Any], *, context: str) -> tuple[str, ...]:
    """Return deduped non-empty marker substrings; raises if any entry is blank."""
    out: list[str] = []
    for item in seq:
        s = str(item).strip()
        if not s:
            raise ValueError(f"{context}: worker marker must be a non-empty string, got {item!r}")
        out.append(s)
    return tuple(dict.fromkeys(out))


def _resolve_runtime_worker_markers(
    server: Any,
    injector_markers_extra: Sequence[str] | None,
) -> tuple[str, ...]:
    """Resolve which substrings classify worker PIDs for fault snapshots.

    If ``server.reliability_worker_markers`` is set to a sequence, **only** those
    markers are used (no defaults). Otherwise defaults from
    :data:`DEFAULT_RUNTIME_WORKER_MARKERS` are merged with:

    - ``server.reliability_worker_markers_extra`` when present, and
    - ``injector_markers_extra`` from :func:`make_server_root_kill_fault_injector` /
      :func:`make_server_tree_kill_fault_injector`.
    """
    replace = getattr(server, "reliability_worker_markers", None)
    if replace is not None:
        return _normalize_worker_marker_substrings(replace, context="server.reliability_worker_markers")

    merged: list[str] = []
    merged.extend(DEFAULT_RUNTIME_WORKER_MARKERS)
    extra_attr = getattr(server, "reliability_worker_markers_extra", None)
    if extra_attr is not None:
        merged.extend(
            _normalize_worker_marker_substrings(extra_attr, context="server.reliability_worker_markers_extra")
        )
    if injector_markers_extra is not None:
        merged.extend(
            _normalize_worker_marker_substrings(
                injector_markers_extra,
                context="injector worker_markers_extra",
            )
        )
    return tuple(dict.fromkeys(merged))


def _pid_looks_like_runtime_worker(pid: int, markers: Sequence[str]) -> bool:
    """True when process name or argv contains any resolved worker marker substring."""
    if not markers:
        return False
    name, cmdline = _safe_proc_info(pid)
    text = f"{name} {cmdline}"
    return any(marker in text for marker in markers)


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


def _pids_in_server_tree_substring_match(server: Any, pattern: str, *, limit: int) -> list[int]:
    """Return up to ``limit`` PIDs in the server tree whose name+cmdline contains ``pattern`` literally.

    ``pgrep -f`` matches the process command line used by procps and does not consult the
    short process name; vLLM-Omni often exposes titles like ``VLLM::Worker`` or
    ``VLLM::StageEngineCoreProc_*`` via prctl/setproctitle in ``ps``/``name()`` only.
    This helper keeps kill injection aligned with :func:`_safe_proc_info` / fault snapshots.
    """
    if limit <= 0:
        return []
    tree = _list_server_process_tree(server)
    if not tree or not pattern:
        return []
    hits: list[int] = []
    for pid in tree:
        if len(hits) >= limit:
            break
        name, cmdline = _safe_proc_info(int(pid))
        if pattern in f"{name} {cmdline}":
            hits.append(int(pid))
    return hits


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


def run_fault_injection_with_rate_load(
    *,
    submit_request: Callable[[], Any],
    inject_fault: Callable[[], None],
    num_requests: int = 10,
    request_rate: float = 0.3,
    submit_interval_sec: float | None = None,
    completion_timeout_sec: float = 120.0,
) -> dict[str, Any]:
    """Submit requests with a fixed rate, then inject fault once an in-flight request is observed."""
    if num_requests <= 0:
        raise ValueError("num_requests must be > 0")
    if request_rate <= 0:
        raise ValueError("request_rate must be > 0")

    interval_sec = submit_interval_sec if submit_interval_sec is not None else (1.0 / request_rate)
    if interval_sec < 0:
        raise ValueError("submit_interval_sec must be >= 0")

    max_workers = min(max(2, num_requests), 8)
    injected = False
    futures: list[concurrent.futures.Future[Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx in range(num_requests):
            futures.append(executor.submit(submit_request))
            in_flight = any(not future.done() for future in futures)
            if in_flight and not injected:
                inject_fault()
                injected = True
            if idx != num_requests - 1 and interval_sec > 0:
                time.sleep(interval_sec)

        if not injected:
            in_flight = any(not future.done() for future in futures)
            if in_flight:
                inject_fault()
                injected = True

        if not injected:
            pytest.skip(
                "no in-flight request observed before fault injection; "
                "increase request cost or request rate for this environment"
            )

        done, pending = concurrent.futures.wait(
            futures,
            timeout=completion_timeout_sec,
            return_when=concurrent.futures.ALL_COMPLETED,
        )
        assert not pending, (
            "some in-flight-load requests did not converge after fault injection: "
            f"pending={len(pending)} done={len(done)}"
        )
    failure_observed = False
    completed = 0
    success = 0
    exceptions = 0
    for future in futures:
        completed += 1
        try:
            future.result()
            success += 1
        except Exception:  # noqa: BLE001
            exceptions += 1
            failure_observed = True
    return {
        "num_submitted": num_requests,
        "completed": completed,
        "success": success,
        "exceptions": exceptions,
        "failure_observed": failure_observed,
        "inflight_observed": True,
    }


def list_alive_pids(pids: Sequence[int]) -> list[int]:
    """Return PIDs from ``pids`` that still exist in the kernel and are **not** zombies.

    After SIGTERM/SIGKILL the serve child may exit before the test harness calls
    ``Popen.wait()``; until reaped, ``psutil.pid_exists`` stays true for the defunct
    slot. Those PIDs must not count as "residual server processes" for leak checks.
    """
    out: list[int] = []
    for pid in pids:
        pid_i = int(pid)
        if not psutil.pid_exists(pid_i):
            continue
        try:
            if psutil.Process(pid_i).status() == psutil.STATUS_ZOMBIE:
                continue
        except psutil.Error:
            continue
        out.append(pid_i)
    return out


def query_gpu_compute_pid_used_memory_mb() -> dict[int, int] | None:
    """Query NVIDIA compute-process memory map (pid -> used MB).

    Returns ``None`` when ``nvidia-smi`` is unavailable/unreadable on current host.
    """
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        logger.warning("[reliability][gpu] nvidia-smi query failed: %s", out.stderr.strip())
        return None

    pid_to_mem_mb: dict[int, int] = {}
    for line in out.stdout.splitlines():
        row = line.strip()
        if not row:
            continue
        pieces = [part.strip() for part in row.split(",", maxsplit=1)]
        if len(pieces) != 2:
            continue
        pid_raw, mem_raw = pieces
        if not pid_raw.isdigit():
            continue
        mem_digits = "".join(ch for ch in mem_raw if ch.isdigit())
        if not mem_digits:
            continue
        pid_to_mem_mb[int(pid_raw)] = int(mem_digits)
    return pid_to_mem_mb


def worker_residual_timeout_after_kill_signal(signal_name: str) -> float:
    """Wall-clock budget for fault-snapshot PIDs / GPU to clear after ``signal_name``."""
    if signal_name == "SIGKILL":
        return 30.0
    if signal_name == "SIGINT":
        return 120.0
    if signal_name == "SIGTERM":
        return 90.0
    return 30.0


def assert_no_server_tree_process_residual_and_gpu_release(
    server: Any,
    *,
    scenario: str,
    timeout_sec: float = 30.0,
    poll_interval_sec: float = 0.5,
) -> None:
    """Assert no live PIDs remain from the fault-time server tree and no GPU use by those PIDs.

    Uses ``reliability_fault_snapshot["tree_pids"]`` (root + all descendants captured at
    injection time), not the marker-filtered ``worker_pids`` subset — so helpers like
    ``multiprocessing.resource_tracker`` are included in the residual check.

    **Zombie** PIDs (exited children not yet reaped by the test ``Popen``) are excluded
    from the "alive" list; ``psutil.pid_exists`` alone would false-positive on them.

    Note: if a child is reparented to PID 1 while staying alive, its PID is unchanged
    and remains detected; if it **exits and a new unrelated process reuses the same
    numeric PID**, this check may false-positive (rare on short windows).
    """
    snapshot = getattr(server, "reliability_fault_snapshot", None)
    if not isinstance(snapshot, dict):
        pytest.fail(f"[{scenario}] missing reliability fault snapshot on server")
    tree_pids = [int(pid) for pid in snapshot.get("tree_pids", [])]
    if not tree_pids:
        pytest.skip(f"[{scenario}] no server process tree PIDs captured for this run")

    tree_pid_set = set(tree_pids)
    deadline = time.monotonic() + timeout_sec
    last_alive: list[int] = []
    last_gpu_leaks: dict[int, int] = {}
    while time.monotonic() < deadline:
        last_alive = list_alive_pids(tree_pids)
        gpu_map = query_gpu_compute_pid_used_memory_mb()
        if gpu_map is None:
            pytest.skip(f"[{scenario}] nvidia-smi unavailable; skip GPU release assertion")
        last_gpu_leaks = {pid: mem_mb for pid, mem_mb in gpu_map.items() if pid in tree_pid_set and mem_mb > 0}
        if not last_alive and not last_gpu_leaks:
            return
        time.sleep(poll_interval_sec)

    assert not last_alive, f"[{scenario}] residual server-tree processes remain alive: {last_alive}"
    assert not last_gpu_leaks, f"[{scenario}] server-tree PID GPU memory not released: {last_gpu_leaks}"


def assert_no_worker_residual_and_gpu_release(
    server: Any,
    *,
    scenario: str,
    timeout_sec: float = 30.0,
    poll_interval_sec: float = 0.5,
) -> None:
    """Deprecated: use :func:`assert_no_server_tree_process_residual_and_gpu_release`."""
    assert_no_server_tree_process_residual_and_gpu_release(
        server,
        scenario=scenario,
        timeout_sec=timeout_sec,
        poll_interval_sec=poll_interval_sec,
    )


def _capture_server_fault_snapshot(
    server: Any,
    *,
    worker_markers_extra: Sequence[str] | None = None,
) -> dict[str, list[int]]:
    """Capture current server process snapshot for post-fault assertions.

    ``tree_pids`` is ``[root, ...descendants]`` and is used by
    :func:`assert_no_server_tree_process_residual_and_gpu_release`.
    ``worker_pids`` is the marker-filtered subset for debugging only.
    """
    markers = _resolve_runtime_worker_markers(server, worker_markers_extra)
    tree_pids = _list_server_process_tree(server)
    root_pid = tree_pids[0] if tree_pids else None
    worker_pids: list[int] = []
    for pid in tree_pids:
        if root_pid is not None and int(pid) == int(root_pid):
            continue
        if _pid_looks_like_runtime_worker(pid, markers):
            worker_pids.append(pid)
    snapshot = {
        "root_pids": [root_pid] if root_pid is not None else [],
        "tree_pids": tree_pids,
        "worker_pids": worker_pids,
        "worker_markers": list(markers),
    }
    setattr(server, "reliability_fault_snapshot", snapshot)
    return snapshot


def make_server_root_kill_fault_injector(
    *,
    signal_name: str = "SIGKILL",
    post_kill_wait_seconds: float = 0.0,
    worker_markers_extra: Sequence[str] | None = None,
) -> FaultInjector:
    """Kill only the current server root process (simulate first strike on PID1-like process).

    ``worker_markers_extra`` is merged into worker PID detection for the fault snapshot
    (after :data:`DEFAULT_RUNTIME_WORKER_MARKERS` unless
    ``server.reliability_worker_markers`` replaces the set entirely).
    """

    def _inject(server: Any) -> None:
        _log_server_process_tree(server)
        snapshot = _capture_server_fault_snapshot(server, worker_markers_extra=worker_markers_extra)
        root_pids = snapshot["root_pids"]
        if not root_pids:
            pytest.skip("no server root process found for root-kill injection")
        root_pid = int(root_pids[0])
        sig = getattr(signal, signal_name, None)
        if sig is None:
            raise ValueError(f"Unsupported signal_name: {signal_name}")
        name, cmdline = _safe_proc_info(root_pid)
        print(
            f"[reliability][process-kill] root-kill pid={root_pid} name={name} signal={signal_name} cmdline={cmdline}",
            flush=True,
        )
        os.kill(root_pid, sig)
        if post_kill_wait_seconds > 0:
            time.sleep(post_kill_wait_seconds)

    return _inject


def make_server_tree_kill_fault_injector(
    *,
    signal_name: str = "SIGKILL",
    post_kill_wait_seconds: float = 0.0,
    inter_kill_wait_seconds: float = 0.2,
    worker_markers_extra: Sequence[str] | None = None,
) -> FaultInjector:
    """Kill current server root first, then kill all remaining descendants.

    See :func:`make_server_root_kill_fault_injector` for ``worker_markers_extra``.
    """

    def _inject(server: Any) -> None:
        _log_server_process_tree(server)
        snapshot = _capture_server_fault_snapshot(server, worker_markers_extra=worker_markers_extra)
        tree_pids = [int(pid) for pid in snapshot["tree_pids"]]
        if not tree_pids:
            pytest.skip("no server process tree found for tree-kill injection")

        sig = getattr(signal, signal_name, None)
        if sig is None:
            raise ValueError(f"Unsupported signal_name: {signal_name}")

        root_pid = tree_pids[0]
        kill_order = [root_pid, *[pid for pid in tree_pids if pid != root_pid]]
        for pid in kill_order:
            if not psutil.pid_exists(pid):
                continue
            name, cmdline = _safe_proc_info(pid)
            print(
                f"[reliability][process-kill] tree-kill pid={pid} name={name} signal={signal_name} cmdline={cmdline}",
                flush=True,
            )
            os.kill(pid, sig)
            if inter_kill_wait_seconds > 0:
                time.sleep(inter_kill_wait_seconds)

        if post_kill_wait_seconds > 0:
            time.sleep(post_kill_wait_seconds)

    return _inject


def make_process_kill_fault_injector(
    *,
    grep_patterns: str | Sequence[str],
    signal_name: str = "SIGKILL",
    limit: int = 1,
    post_kill_wait_seconds: float = 0.0,
) -> FaultInjector:
    """Build a post-ready injector that kills processes matched by ordered ``grep_patterns``.

    Matching is tried in two phases per pattern (first hit wins):

    1. **Server process tree** (``psutil``): literal substring match against
       ``Process.name()`` plus argv text. This matches how :func:`_safe_proc_info`
       logs processes and catches vLLM-style titles (``VLLM::...``) that often do not
       appear in ``pgrep -f``\'s command-line view.
    2. **``pgrep -f``** (legacy): scoped to the server PID tree when it is known;
       uses procps regular-expression rules for the pattern.

    If neither phase finds a target, the returned callable issues ``pytest.skip``.

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

        def _kill_and_wait(filtered: list[int], pattern: str, *, source: str) -> None:
            sig = getattr(signal, signal_name, None)
            if sig is None:
                raise ValueError(f"Unsupported signal_name: {signal_name}")
            for pid in filtered:
                name, cmdline = _safe_proc_info(pid)
                print(
                    f"[reliability][process-kill] killing pid={pid} name={name} signal={signal_name} "
                    f"source={source} cmdline={cmdline}",
                    flush=True,
                )
                os.kill(pid, sig)
            print(
                f"[reliability][process-kill] matched pattern={pattern!r} source={source} "
                f"killed_pids={filtered} killed_count={len(filtered)}",
                flush=True,
            )
            if post_kill_wait_seconds > 0:
                print(
                    f"[reliability][process-kill] waiting {post_kill_wait_seconds:.2f}s after kill",
                    flush=True,
                )
                time.sleep(post_kill_wait_seconds)

        for pattern in patterns:
            print(
                f"[reliability][process-kill] trying server_tree substring pattern={pattern!r} "
                f"signal={signal_name} limit={limit}",
                flush=True,
            )
            tree_hits = _pids_in_server_tree_substring_match(server, pattern, limit=limit)
            if tree_hits:
                _kill_and_wait(tree_hits, pattern, source="server_tree")
                return

        for pattern in patterns:
            print(
                f"[reliability][process-kill] trying pgrep -f pattern={pattern!r} signal={signal_name} limit={limit}",
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
                _kill_and_wait(filtered, pattern, source="pgrep")
                return
        logger.warning(
            "[reliability][process-kill] no process matched patterns=%s signal=%s limit=%s",
            patterns,
            signal_name,
            limit,
        )
        pytest.skip("no matching runtime process found for kill injection")

    return _inject
