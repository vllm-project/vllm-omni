"""Shared reliability fault-injection helpers.

This module keeps fault injection callable from tests directly:
- abnormal input (raw HTTP malformed requests)
- GPU OOM (CUDA sidecar memory hog)
- process kill by pattern and signal
- post-ready hooks via ``fault_injector`` / ``omni_server_after_fault`` fixtures
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import psutil
import select
import shlex
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
import yaml

from tests.conftest import (
    OmniServer,
    OmniServerParams,
    OmniServerStageCli,
    OpenAIClientHandler,
    _omni_server_lock,
    modify_stage_config,
)

FaultVariant = str
logger = logging.getLogger(__name__)


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

    proc: subprocess.Popen | None
    device: int
    target_mem_ratio: float
    start_ts: float


@dataclass
class RuntimeTeardownContainerHandle:
    """Handle for a dockerized vLLM-Omni server used by runtime teardown tests."""

    container_name: str
    host: str
    port: int
    model: str


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


def list_process_pids_by_pattern(pattern: str) -> list[int]:
    """Return matched PIDs from ``pgrep -f <pattern>``."""
    out = subprocess.run(
        ["pgrep", "-f", pattern],
        check=False,
        capture_output=True,
        text=True,
    )
    if out.returncode not in (0, 1):
        raise RuntimeError(f"pgrep failed for pattern={pattern!r}: {out.stderr.strip()}")
    return [int(item) for item in out.stdout.split() if item.strip().isdigit()]


def _runtime_teardown_ssh_target() -> str:
    target = os.getenv("RUNTIME_TEARDOWN_SSH_TARGET", "").strip()
    # Default to root@127.0.0.1 for same-host SSH control path.
    return target or "root@127.0.0.1"


def _runtime_teardown_ssh_cmd(remote_cmd: str, *, step: str | None = None) -> subprocess.CompletedProcess[str]:
    ssh_target = _runtime_teardown_ssh_target()
    default_reuse_opts = (
        "-o ControlMaster=auto "
        "-o ControlPersist=10m "
        "-o ControlPath=/tmp/vllm-rt-ssh-%r@%h:%p"
    )
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
            f"{step_prefix} timed out after {timeout_sec}s. "
            "Increase RUNTIME_TEARDOWN_SSH_TIMEOUT_SEC if needed."
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


def force_remove_container(container_name: str) -> None:
    """Force-remove a docker container on remote host via SSH."""
    _runtime_teardown_ssh_cmd(
        f"docker rm -f {shlex.quote(container_name)} >/dev/null 2>&1 || true",
        step="docker-rm-f",
    )


def _allocate_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_tcp_port_ready(host: str, port: int, timeout_sec: int) -> None:
    deadline = time.time() + timeout_sec
    start_ts = time.time()
    next_log_ts = start_ts
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                elapsed = int(time.time() - start_ts)
                print(
                    f"[runtime-teardown][wait-port] ready host={host} port={port} elapsed={elapsed}s",
                    flush=True,
                )
                return
        now = time.time()
        if now >= next_log_ts:
            elapsed = int(now - start_ts)
            print(
                f"[runtime-teardown][wait-port] waiting host={host} port={port} elapsed={elapsed}s",
                flush=True,
            )
            next_log_ts = now + 15
        time.sleep(2)
    raise TimeoutError(f"Server in container did not become ready within {timeout_sec}s: {host}:{port}")


def start_runtime_teardown_container_server(
    *,
    model: str,
    serve_args: list[str],
    image: str | None = None,
    docker_run_args: list[str] | None = None,
    bootstrap_cmd: str | None = None,
    startup_timeout_sec: int = 1200,
) -> RuntimeTeardownContainerHandle:
    """Start a dedicated container on remote host via SSH and wait until port is ready."""
    if os.name == "nt":
        raise RuntimeError("runtime teardown container helper currently supports POSIX platforms only.")
    if subprocess.run(["ssh", "-V"], check=False, capture_output=True, text=True).returncode != 0:
        pytest.skip("ssh client not available for runtime teardown test")

    container_name = f"omni_rt_teardown_{uuid4().hex[:8]}"
    host = os.getenv("RUNTIME_TEARDOWN_SERVER_HOST", "127.0.0.1")
    bind_host = os.getenv("RUNTIME_TEARDOWN_BIND_HOST", "0.0.0.0")
    port = _allocate_open_port()
    local_repo_root = Path(__file__).resolve().parents[3]
    remote_workdir = os.getenv("RUNTIME_TEARDOWN_REMOTE_WORKDIR", str(local_repo_root))
    resolved_image = image or os.getenv("RUNTIME_TEARDOWN_IMAGE", "nvcr.io/nvidia/pytorch:25.01-py3")
    resolved_bootstrap = bootstrap_cmd if bootstrap_cmd is not None else os.getenv("RUNTIME_TEARDOWN_BOOTSTRAP_CMD", "")
    if docker_run_args is not None:
        extra_run_args = docker_run_args
    else:
        extra_run_args = shlex.split(os.getenv("RUNTIME_TEARDOWN_DOCKER_ARGS", ""))

    run_cmd_args = [
        "docker",
        "run",
        "-d",
        "--shm-size=128g",
        "--privileged=true",
        "--restart=always",
        "--gpus",
        "all",
        "--name",
        container_name,
        "--net=host",
        "-v",
        "/home:/home",
        "-v",
        f"{remote_workdir}:{remote_workdir}",
        "--cap-add=SYS_PTRACE",
        "--security-opt",
        "seccomp=unconfined",
        *extra_run_args,
        resolved_image,
        "sleep",
        "infinity",
    ]
    run_cmd = " ".join(shlex.quote(arg) for arg in run_cmd_args)
    print(
        f"[runtime-teardown] container={container_name} host={host} port={port} image={resolved_image}",
        flush=True,
    )
    run_out = _runtime_teardown_ssh_cmd(run_cmd, step="docker-run")
    if run_out.returncode != 0:
        raise RuntimeError(f"failed to start runtime teardown container: {run_out.stderr.strip()}")
    run_stdout = (run_out.stdout or "").strip()
    run_stderr = (run_out.stderr or "").strip()
    if "Usage:  docker [OPTIONS] COMMAND" in run_stderr:
        raise RuntimeError(
            "docker-run rendered docker CLI help on remote host; command likely malformed. "
            f"run_cmd={run_cmd!r}, stderr={run_stderr!r}. "
            "Check RUNTIME_TEARDOWN_DOCKER_ARGS and avoid passing a full 'docker run ...' command there "
            "(only extra args are allowed)."
        )
    print(
        f"[runtime-teardown][docker-run] stdout={run_stdout!r} stderr={run_stderr!r}",
        flush=True,
    )
    # Verify container really exists right after docker run.
    ps_out = _runtime_teardown_ssh_cmd(
        f"docker ps -a --filter name={shlex.quote(container_name)} --format '{{{{.ID}}}} {{{{.Status}}}} {{{{.Names}}}}'",
        step="docker-ps-verify",
    )
    ps_text = (ps_out.stdout or "").strip()
    print(f"[runtime-teardown][docker-ps-verify] {ps_text!r}", flush=True)
    if not ps_text:
        raise RuntimeError(
            f"container not found right after docker run: name={container_name}, "
            f"docker_run_stdout={run_stdout!r}, docker_run_stderr={run_stderr!r}"
        )

    serve_cmd = [
        "python",
        "-m",
        "vllm_omni.entrypoints.cli.main",
        "serve",
        model,
        "--host",
        bind_host,
        "--port",
        str(port),
        "--omni",
        *serve_args,
    ]
    serve_cmd_str = " ".join(shlex.quote(arg) for arg in serve_cmd)
    bootstrap_prefix = f"{resolved_bootstrap} && " if resolved_bootstrap.strip() else ""
    serve_log_path = "/tmp/vllm_runtime_teardown_serve.log"
    exec_cmd = [
        "docker",
        "exec",
        "-d",
        container_name,
        "bash",
        "-lc",
        (
            f"cd {shlex.quote(remote_workdir)} && "
            f"{bootstrap_prefix}VLLM_WORKER_MULTIPROC_METHOD=spawn {serve_cmd_str} "
            f"> {shlex.quote(serve_log_path)} 2>&1"
        ),
    ]
    exec_cmd_str = " ".join(shlex.quote(arg) for arg in exec_cmd)
    exec_out = _runtime_teardown_ssh_cmd(exec_cmd_str, step="docker-exec-start-serve")
    if exec_out.returncode != 0:
        force_remove_container(container_name)
        raise RuntimeError(f"failed to start server in runtime teardown container: {exec_out.stderr.strip()}")

    try:
        print(
            f"[runtime-teardown] waiting server ready container={container_name} host={host} bind_host={bind_host} port={port} timeout={startup_timeout_sec}s",
            flush=True,
        )
        _wait_tcp_port_ready(host, port, timeout_sec=startup_timeout_sec)
    except Exception:
        logs = _runtime_teardown_ssh_cmd(
            (
                f"docker exec {shlex.quote(container_name)} bash -lc "
                f"\"tail -n 200 {shlex.quote(serve_log_path)} 2>/dev/null || "
                f"echo '[no-serve-log-file] {shlex.quote(serve_log_path)}'\""
            ),
            step="docker-exec-tail-serve-log",
        )
        keep_on_failure = os.getenv("RUNTIME_TEARDOWN_KEEP_CONTAINER_ON_FAILURE", "0").strip() == "1"
        if keep_on_failure:
            print(
                f"[runtime-teardown] keep container for debugging: {container_name}",
                flush=True,
            )
        else:
            force_remove_container(container_name)
        raise RuntimeError(
            "runtime teardown container server startup failed. "
            f"host={host} bind_host={bind_host} port={port}; "
            f"serve_log_tail={logs.stdout[-2000:]}"
        ) from None

    return RuntimeTeardownContainerHandle(
        container_name=container_name,
        host=host,
        port=port,
        model=model,
    )


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
        logger.info(
            "[reliability][process-kill] current_server_proc pid=%s name=%s cmdline=%s",
            pid,
            name,
            cmdline,
        )


FaultInjector = Callable[[Any], None]
"""Callable invoked with the live ``OmniServer`` after it is ready (see ``omni_server_after_fault``)."""


def make_process_kill_fault_injector(
    *,
    grep_patterns: str | Sequence[str],
    signal_name: str = "SIGKILL",
    limit: int = 1,
) -> FaultInjector:
    """Build a post-ready injector that kills processes matched by ``pgrep -f``.

    Tries each pattern in order until at least one PID is killed. If none match,
    the returned callable issues ``pytest.skip`` (same behavior as the previous
    inline reliability test).

    Args:
        grep_patterns: One pattern or an ordered list of patterns.
        signal_name: Passed to :func:`inject_process_kill` (e.g. ``SIGKILL``).
        limit: Maximum PIDs to kill per pattern (default ``1``).
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
            logger.info(
                "[reliability][process-kill] trying pattern=%s signal=%s limit=%s",
                pattern,
                signal_name,
                limit,
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
                    logger.info(
                        "[reliability][process-kill] killing pid=%s name=%s signal=%s cmdline=%s",
                        pid,
                        name,
                        signal_name,
                        cmdline,
                    )
                    os.kill(pid, sig)
                logger.info(
                    "[reliability][process-kill] matched pattern=%s killed_pids=%s killed_count=%d",
                    pattern,
                    filtered,
                    len(filtered),
                )
                return
        logger.warning(
            "[reliability][process-kill] no process matched patterns=%s signal=%s limit=%s",
            patterns,
            signal_name,
            limit,
        )
        pytest.skip("no matching runtime process found for kill injection")

    return _inject


@pytest.fixture
def fault_injector(request: pytest.FixtureRequest) -> FaultInjector:
    """Indirect only: ``request.param`` must be a ``FaultInjector`` callable."""
    return request.param


@pytest.fixture
def omni_server_after_fault(omni_server: Any, fault_injector: FaultInjector):
    """After ``omni_server`` is up, run ``fault_injector(omni_server)``, then yield the server."""
    fault_injector(omni_server)
    yield omni_server


@pytest.fixture(scope="function")
def omni_server_function(
    request: pytest.FixtureRequest,
    run_level: str,
    model_prefix: str,
):
    """Function-scoped Omni server fixture for reliability tests."""
    with _omni_server_lock:
        params: OmniServerParams = request.param
        model = model_prefix + params.model
        port = params.port
        stage_config_path = params.stage_config_path
        if run_level == "advanced_model" and stage_config_path is not None:
            with open(stage_config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            stage_ids = [stage["stage_id"] for stage in cfg.get("stage_args", []) if "stage_id" in stage]
            stage_config_path = modify_stage_config(
                stage_config_path,
                deletes={"stage_args": {stage_id: ["engine_args.load_format"] for stage_id in stage_ids}},
            )

        server_args = params.server_args or []
        if params.use_omni and params.stage_init_timeout is not None:
            server_args = [*server_args, "--stage-init-timeout", str(params.stage_init_timeout)]
        else:
            server_args = [*server_args, "--stage-init-timeout", "600"]
        if params.init_timeout is not None:
            server_args = [*server_args, "--init-timeout", str(params.init_timeout)]
        else:
            server_args = [*server_args, "--init-timeout", "900"]
        if params.use_stage_cli:
            if not params.use_omni:
                raise ValueError("omni_server with use_stage_cli=True requires use_omni=True")
            if stage_config_path is None:
                raise ValueError("omni_server with use_stage_cli=True requires a stage_config_path")

            with OmniServerStageCli(
                model,
                stage_config_path,
                server_args,
                port=port,
                env_dict=params.env_dict,
            ) as server:
                print("OmniServer started successfully")
                yield server
                print("OmniServer stopping...")
        else:
            if stage_config_path is not None:
                server_args += ["--stage-configs-path", stage_config_path]

            with (
                OmniServer(
                    model,
                    server_args,
                    port=port,
                    env_dict=params.env_dict,
                    use_omni=params.use_omni,
                )
                if port
                else OmniServer(
                    model,
                    server_args,
                    env_dict=params.env_dict,
                    use_omni=params.use_omni,
                )
            ) as server:
                print("OmniServer started successfully")
                yield server
                print("OmniServer stopping...")
        print("OmniServer stopped")


@pytest.fixture
def openai_client_function(omni_server_function: Any, run_level: str):
    """OpenAI client bound to function-scoped ``omni_server_function``."""
    return OpenAIClientHandler(
        host=omni_server_function.host,
        port=omni_server_function.port,
        api_key="EMPTY",
        run_level=run_level,
    )


@pytest.fixture
def omni_server_after_fault_function(omni_server_function: Any, fault_injector: FaultInjector):
    """Inject fault after function-scoped server startup, then yield server."""
    fault_injector(omni_server_function)
    yield omni_server_function
