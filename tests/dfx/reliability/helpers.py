# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

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
import copy
import errno
import http.client
import inspect
import json
import logging
import os
import re
import select
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import psutil
import pytest

if TYPE_CHECKING:
    from tests.helpers.runtime import OmniServerParams

logger = logging.getLogger(__name__)


class GpuOomInjectionConfig(TypedDict, total=False):
    """Fields read by :func:`inject_gpu_oom` and ``resolve_oom_device_spec``."""

    device: str
    target_mem_ratio: float
    hold_seconds: int
    startup_timeout_sec: int
    strict: bool


class ServerFaultSnapshot(TypedDict):
    root_pids: list[int]
    tree_pids: list[int]
    worker_pids: list[int]
    worker_markers: list[str]


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


_ALLOC_ATTEMPT_TIMEOUT_SEC = 2.0


def reliability_scenarios_for_eager_oom(
    scenarios: list[dict[str, Any]],
    *,
    is_hip: bool,
    stage_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Clone scenarios and force selected HIP stages to allocate eagerly.

    External sidecars only consume device-global free memory. A server can
    otherwise reuse captured graph pools or cached profile buffers, causing an
    OOM probe to succeed without making a new device allocation.
    """
    cloned = copy.deepcopy(scenarios)
    if not is_hip:
        return cloned
    for scenario in cloned:
        server_params = scenario["server_params"]
        updates = server_params.setdefault("update", {})
        stages = updates.setdefault("stages", {})
        for stage_id in stage_ids:
            stages.setdefault(int(stage_id), {})["enforce_eager"] = True
    return cloned


def allocate_device_chunk(
    req_elems: int,
    *,
    allocate: Callable[[int], Any],
    free_bytes: Callable[[], int],
    timeout_sec: float = _ALLOC_ATTEMPT_TIMEOUT_SEC,
) -> Any | None:
    """Return one chunk, or None when the allocator rejects or stalls.

    A ``RuntimeError``, a call that exceeds ``timeout_sec``, or a call that
    does not reduce reported free memory is rejection. HIP can log an OOM and
    retry inside ``allocate`` without raising; the timeout lets the sidecar
    print READY instead of blocking the startup wait.
    """
    import threading

    free_before = int(free_bytes())
    box: dict[str, Any] = {}

    def _run() -> None:
        try:
            box["chunk"] = allocate(req_elems)
        except RuntimeError as exc:
            box["error"] = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout_sec)
    if thread.is_alive() or "error" in box or "chunk" not in box:
        return None
    if int(free_bytes()) >= free_before:
        return None
    return box["chunk"]


def _build_sidecar_cmd(
    device: int,
    target_mem_ratio: float,
    hold_seconds: int,
    strict: bool,
    max_free_bytes: int | None,
) -> list[str]:
    sidecar = (
        "from __future__ import annotations\n"
        "from collections.abc import Callable\n"
        "from typing import Any\n"
        f"_ALLOC_ATTEMPT_TIMEOUT_SEC = {_ALLOC_ATTEMPT_TIMEOUT_SEC!r}\n\n"
        + inspect.getsource(allocate_device_chunk)
        + r"""
import sys
import time
import torch

device = int(sys.argv[1])
target_ratio = float(sys.argv[2])
hold_seconds = int(sys.argv[3])
strict = sys.argv[4] == "1"
max_free_bytes_arg = int(sys.argv[5])
max_free_bytes = None if max_free_bytes_arg < 0 else max_free_bytes_arg

torch.cuda.init()
torch.cuda.set_device(device)
props = torch.cuda.get_device_properties(device)
free_before, total_bytes = torch.cuda.mem_get_info(device)
ratio_target_bytes = int(free_before * target_ratio)
ceiling_target_bytes = 0 if max_free_bytes is None else max(0, int(free_before) - max_free_bytes)
target_bytes = max(ratio_target_bytes, ceiling_target_bytes)
chunk_bytes = 256 * 1024 * 1024
chunks = []
allocated = 0
rejected = False

def _take(req_elems):
    return allocate_device_chunk(
        req_elems,
        allocate=lambda n: torch.empty((n,), dtype=torch.float16, device=f"cuda:{device}"),
        free_bytes=lambda: torch.cuda.mem_get_info(device)[0],
    )

while allocated < target_bytes:
    req_bytes = min(chunk_bytes, target_bytes - allocated)
    req_elems = max(1, req_bytes // 2)  # float16 -> 2 bytes
    chunk = _take(req_elems)
    if chunk is None:
        rejected = True
        break
    chunks.append(chunk)
    allocated += chunk.numel() * 2

# Strict mode keeps filling with smaller chunks until the allocator rejects.
# Rejection is a raised error, a stalled call, or an alloc that does not
# reduce free memory. HIP can log OOM and retry without raising.
if strict and not rejected:
    tail_chunk_bytes = [64 * 1024 * 1024, 16 * 1024 * 1024, 4 * 1024 * 1024, 1 * 1024 * 1024]
    for tail_bytes in tail_chunk_bytes:
        while True:
            chunk = _take(max(1, tail_bytes // 2))
            if chunk is None:
                rejected = True
                break
            chunks.append(chunk)
            allocated += chunk.numel() * 2
        if rejected:
            break

achieved_ratio = allocated / max(1, props.total_memory)
achieved_free_ratio = allocated / max(1, int(free_before))
free_after, _ = torch.cuda.mem_get_info(device)
# A full device is readiness, even when the byte target was not met.
residual_floor = 64 * 1024 * 1024 if max_free_bytes is None else max_free_bytes
allocator_full = rejected and int(free_after) <= residual_floor
ratio_target_missed = strict and allocated < ratio_target_bytes and not allocator_full
free_ceiling_missed = max_free_bytes is not None and int(free_after) > max_free_bytes
if ratio_target_missed or free_ceiling_missed:
    print(
        "ERROR:"
        f"achieved_free_ratio={achieved_free_ratio:.4f};"
        f"achieved_total_ratio={achieved_ratio:.4f};"
        f"free_before={int(free_before)};"
        f"free_after={int(free_after)};"
        f"target_bytes={target_bytes};"
        f"max_free_bytes={max_free_bytes};"
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
    f"max_free_bytes={max_free_bytes};"
    f"allocated={allocated}",
    flush=True,
)
if hold_seconds <= 0:
    while True:
        time.sleep(3600)
time.sleep(hold_seconds)
print("DONE", flush=True)
"""
    )
    return [
        sys.executable,
        "-u",
        "-c",
        sidecar,
        str(device),
        str(target_mem_ratio),
        str(hold_seconds),
        "1" if strict else "0",
        str(max_free_bytes) if max_free_bytes is not None else "-1",
    ]


def start_gpu_oom_hog(
    *,
    device: int = 0,
    target_mem_ratio: float = 0.95,
    hold_seconds: int = 60,
    startup_timeout_sec: int = 20,
    strict: bool = True,
    max_free_bytes: int | None = None,
    poll_interval_sec: float = 0.2,
) -> OomHandle:
    """Start a CUDA sidecar process that occupies GPU memory to trigger OOM.

    Note:
        ``target_mem_ratio`` is evaluated against free memory at injection start
        (not total GPU memory), i.e. success gate is ``allocated / free_before``.
        When ``max_free_bytes`` is set, the sidecar also allocates until no more
        than that many bytes remain free and fails startup if it cannot reach the
        ceiling.
        ``hold_seconds <= 0`` means keeping OOM pressure until the sidecar is
        explicitly stopped via ``stop_gpu_oom_hog(s)``.
    """
    if os.name == "nt":
        raise RuntimeError("CUDA OOM sidecar is intended for Linux CI/runtime.")
    if not (0.0 <= target_mem_ratio < 1.0):
        raise ValueError("target_mem_ratio should be in [0.0, 1.0).")
    if max_free_bytes is not None and max_free_bytes <= 0:
        raise ValueError("max_free_bytes should be greater than 0 when set.")

    # Explicit opt-out for debugging: keep API shape stable while disabling injection.
    if target_mem_ratio == 0.0 and max_free_bytes is None:
        print(f"[oom-sidecar][gpu={device}] DISABLED: target_mem_ratio=0.0 (no OOM injection)", flush=True)
        return OomHandle(
            proc=None,
            device=device,
            target_mem_ratio=target_mem_ratio,
            start_ts=time.time(),
        )

    cmd = _build_sidecar_cmd(device, target_mem_ratio, hold_seconds, strict, max_free_bytes)
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


def _device_spec_from_deploy(deploy: Any, stage_config_path: str | Path) -> str:
    device_ids: set[int] = set()
    for stage in deploy.stages:
        devices = stage.devices
        if not isinstance(devices, str):
            continue
        for token in devices.split(","):
            token = token.strip()
            if token:
                device_ids.add(int(token))
    if not device_ids:
        raise ValueError(f"No runtime.devices found in stage config: {stage_config_path}")
    return ",".join(str(device_id) for device_id in sorted(device_ids))


def stage_config_device_spec(stage_config_path: str | Path, platform: str | None) -> str:
    """Devices the server keeps after ``base_config`` inheritance and one platform overlay.

    ``platform=None`` keeps the inherited base stages and does not apply
    ``platforms.*``. A named platform uses
    :func:`vllm_omni.config.stage_config._apply_platform_overrides` with that
    same ``device_name`` key. Other platform blocks, including an XPU 8-GPU
    list, stay out of the result.
    """
    from vllm_omni.config.stage_config import _apply_platform_overrides, load_deploy_config

    deploy = load_deploy_config(stage_config_path)
    if platform is not None:
        deploy = _apply_platform_overrides(deploy, platform)
    return _device_spec_from_deploy(deploy, stage_config_path)


def inject_gpu_oom(
    *,
    device: int | str | list[int] = 0,
    target_mem_ratio: float = 0.95,
    hold_seconds: int = 60,
    startup_timeout_sec: int = 20,
    strict: bool = True,
    max_free_bytes: int | None = None,
) -> OomHandle | list[OomHandle]:
    """Convenience wrapper to start CUDA OOM sidecar(s).

    Args:
        device: One device id (``0``), comma-separated string (``"0,1,2"``),
            or a list of device ids (``[0, 1, 2]``).
        hold_seconds: OOM hold time in seconds; ``<=0`` keeps pressure until
            ``stop_gpu_oom_hogs`` is called.
        max_free_bytes: Optional upper bound for free memory after injection.
            Sidecar startup fails if this ceiling cannot be reached.
    """
    if isinstance(device, int):
        devices = [device]
    elif isinstance(device, str):
        devices = [int(x.strip()) for x in device.split(",") if x.strip()]
    else:
        devices = [int(x) for x in device]
    if not devices:
        raise ValueError("device must not be empty.")

    handles: list[OomHandle] = []
    try:
        for dev in devices:
            handles.append(
                start_gpu_oom_hog(
                    device=dev,
                    target_mem_ratio=target_mem_ratio,
                    hold_seconds=hold_seconds,
                    startup_timeout_sec=startup_timeout_sec,
                    strict=strict,
                    max_free_bytes=max_free_bytes,
                )
            )
    except BaseException:
        stop_gpu_oom_hogs(handles)
        raise
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


def _pid_is_dead(pid: int) -> bool:
    """True when ``pid`` is gone or only a zombie remains."""
    if not psutil.pid_exists(pid):
        return True
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.Error:
        return True


def _wait_until_pid_dead(pid: int, timeout_sec: float) -> bool:
    """Poll until ``pid`` is dead or ``timeout_sec`` elapses."""
    deadline = time.monotonic() + max(0.0, timeout_sec)
    while True:
        if _pid_is_dead(pid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


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


def _parse_nvidia_compute_pid_used_memory_mb(output: str) -> dict[int, int]:
    pid_to_mem_mb: dict[int, int] = {}
    for line in output.splitlines():
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


def _parse_rocm_compute_pid_used_memory_mb(output: str) -> dict[int, int]:
    """Parse ``rocm-smi --showpids --json`` output into a PID -> used MiB map."""
    payload = json.loads(output)
    system = payload.get("system", {})
    if not isinstance(system, dict):
        return {}

    pid_to_mem_mb: dict[int, int] = {}
    for key, value in system.items():
        pid_match = re.fullmatch(r"pid\s*(\d+)", str(key), flags=re.IGNORECASE)
        if pid_match is None or not isinstance(value, str):
            continue

        # rocm-smi emits: process name, gpu count, VRAM bytes, SDMA bytes,
        # CU occupancy. Split from the right so a process name containing a
        # comma does not move the VRAM field.
        fields = [part.strip() for part in value.rsplit(",", maxsplit=4)]
        if len(fields) != 5:
            continue
        vram_raw = fields[-3]
        value_match = re.search(r"\d+", vram_raw.replace("_", ""))
        if value_match is None:
            continue

        value_int = int(value_match.group())
        normalized = vram_raw.lower()
        if "gib" in normalized or normalized.endswith("gb"):
            used_mb = value_int * 1024
        elif "mib" in normalized or normalized.endswith("mb"):
            used_mb = value_int
        elif "kib" in normalized or normalized.endswith("kb"):
            used_mb = (value_int + 1023) // 1024
        else:
            # The official rocm-smi JSON format reports unitless VRAM bytes.
            used_mb = (value_int + (1024**2 - 1)) // (1024**2)
        pid_to_mem_mb[int(pid_match.group(1))] = used_mb
    return pid_to_mem_mb


def _gpu_process_query_spec(
    platform: Any | None = None,
) -> tuple[list[str], Callable[[str], dict[int, int]]] | None:
    if platform is None:
        from vllm_omni.platforms import current_omni_platform

        platform = current_omni_platform
    if platform.is_rocm():
        return ["rocm-smi", "--showpids", "--json"], _parse_rocm_compute_pid_used_memory_mb
    if platform.is_cuda():
        return (
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            _parse_nvidia_compute_pid_used_memory_mb,
        )
    return None


def query_gpu_compute_pid_used_memory_mb() -> dict[int, int] | None:
    """Query platform compute-process memory (pid -> used MiB).

    Returns ``None`` when the platform SMI tool is unavailable, fails, times
    out, or returns unreadable output. Callers must still enforce process-tree
    cleanup when GPU telemetry is unavailable.
    """
    query_spec = _gpu_process_query_spec()
    if query_spec is None:
        return None
    command, parser = query_spec

    try:
        out = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("[reliability][gpu] %s query unavailable: %s", command[0], exc)
        return None
    if out.returncode != 0:
        logger.warning("[reliability][gpu] %s query failed: %s", command[0], out.stderr.strip())
        return None
    try:
        return parser(out.stdout)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("[reliability][gpu] %s output was unreadable: %s", command[0], exc)
        return None


_TERMINAL_HEALTH_ERRNOS = frozenset(
    {
        errno.ECONNREFUSED,
        errno.ECONNRESET,
        errno.EPIPE,
        errno.ETIMEDOUT,
        errno.EHOSTUNREACH,
        errno.ENETUNREACH,
    }
)


def post_fault_health_exception_is_terminal(exc: BaseException) -> bool:
    """Return whether a ``/health`` poll failed because the API is gone or the read timed out.

    An ``AssertionError`` from an unexpected HTTP status means something is still
    listening. Callers decide whether that is a failure or a reason to keep polling.
    """
    if isinstance(exc, AssertionError):
        return False
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if not isinstance(exc, OSError):
        return False
    errno_val = getattr(exc, "errno", None)
    if errno_val in _TERMINAL_HEALTH_ERRNOS:
        return True
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "timed out",
            "timeout",
            "connection refused",
            "actively refused",
            "connection aborted",
            "connection reset",
        )
    )


def assert_post_fault_health_503(online_client: Any, *, scenario: str) -> None:
    """Wait 20s, then require one ``/health`` response of HTTP 503.

    The API process must still be listening. A connection error propagates.
    """
    time.sleep(20.0)
    status = online_client.send_health_http_request({"timeout": 5}, err_code=(200, 503))[0].status_code
    assert status == 503, f"[{scenario} health] expected /health 503 after 20s, got status={status}"


def assert_post_fault_api_unreachable(online_client: Any, *, scenario: str) -> None:
    """Poll ``/health`` until it fails to connect, or until 20s elapse.

    HTTP 200 and HTTP 503 both mean a process is still listening, so polling
    continues. A connection failure means the API process is gone.
    """
    deadline = time.monotonic() + 20.0
    last_observation = "no response"
    while time.monotonic() < deadline:
        try:
            status = online_client.send_health_http_request({"timeout": 5}, err_code=(200, 503))[0].status_code
            last_observation = f"http={status}"
        except Exception as exc:  # noqa: BLE001
            if post_fault_health_exception_is_terminal(exc):
                return
            last_observation = f"exception={exc!r}"
        time.sleep(0.5)
    pytest.fail(
        f"[{scenario} health] /health stayed reachable for 20s after the API process was killed; "
        f"last_observation={last_observation}"
    )


SIGTERM_CLEANUP_GRACE_SEC = 60.0
PARTIAL_ISOLATION_HTTP_TIMEOUT_SEC = 30.0


def assert_api_still_listening(online_client: Any, *, scenario: str) -> None:
    """The API process is still bound. HTTP 200 and HTTP 503 both count."""
    status = online_client.send_health_http_request({"timeout": 5}, err_code=(200, 503))[0].status_code
    assert status in (200, 503), f"[{scenario}] expected /health 200 or 503 while the API is still up, got {status}"


def sigterm_serve_root_and_assert_cleanup(
    server: Any,
    *,
    scenario: str,
    grace_sec: float = SIGTERM_CLEANUP_GRACE_SEC,
) -> None:
    """Send one SIGTERM to the serve root and require the process tree and GPU to clear."""
    make_server_root_kill_fault_injector(signal_name="SIGTERM", post_kill_wait_seconds=0.0)(server)
    assert_no_server_tree_process_residual_and_gpu_release(server, scenario=scenario, timeout_sec=grace_sec)


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
    assert isinstance(snapshot, dict)
    tree_pids = [int(pid) for pid in snapshot.get("tree_pids", [])]
    if not tree_pids:
        pytest.skip(f"[{scenario}] no server process tree PIDs captured for this run")

    tree_pid_set = set(tree_pids)
    deadline = time.monotonic() + timeout_sec
    last_alive: list[int] = []
    last_gpu_leaks: dict[int, int] = {}
    gpu_telemetry_available: bool | None = None
    while time.monotonic() < deadline:
        last_alive = list_alive_pids(tree_pids)
        gpu_map = query_gpu_compute_pid_used_memory_mb() if gpu_telemetry_available is not False else None
        if gpu_map is None:
            if gpu_telemetry_available is None:
                logger.warning(
                    "[%s] GPU process telemetry unavailable; process-tree cleanup remains mandatory",
                    scenario,
                )
            gpu_telemetry_available = False
            last_gpu_leaks = {}
        else:
            gpu_telemetry_available = True
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
) -> ServerFaultSnapshot:
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
    snapshot: ServerFaultSnapshot = {
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


def _server_is_single_process_uniproc(server: Any) -> bool:
    """True when the server tree holds only the root process (UniProc inline worker).

    The UniProc executor runs the diffusion worker inside the API-server process,
    so there is no dedicated worker process to kill; killing the root would take
    the whole server down. Multi-process deployments (``mp``/``ray``) spawn worker
    subprocesses as descendants of the root.
    """
    return len(_list_server_process_tree(server)) <= 1


def make_process_kill_fault_injector(
    *,
    grep_patterns: str | Sequence[str],
    signal_name: str = "SIGKILL",
    limit: int = 1,
    post_kill_wait_seconds: float = 0.0,
) -> FaultInjector:
    """Build a post-ready injector that kills processes matched by ordered ``grep_patterns``.

    The worker-kill scenario needs a dedicated worker process: when the server
    tree holds only the root (UniProc / single process), the injector skips
    unconditionally — there is no worker to kill, and matching the serving
    process would tear the whole server down.

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
        if _server_is_single_process_uniproc(server):
            pytest.skip(
                "worker-kill scenario invalid: UniProc single-process deployment has no "
                "dedicated worker process to kill"
            )
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
                waited_from = time.monotonic()
                dead = _wait_until_pid_dead(pid, post_kill_wait_seconds)
                remaining = post_kill_wait_seconds - (time.monotonic() - waited_from)
                if dead and remaining > 0:
                    time.sleep(remaining)
                print(
                    f"[reliability][process-kill] killed pid={pid} stage={name} signal={signal_name} dead={int(dead)}",
                    flush=True,
                )
                if signal_name == "SIGKILL" and not dead:
                    raise RuntimeError(f"pid {pid} ({name}) still alive after SIGKILL")
            print(
                f"[reliability][process-kill] matched pattern={pattern!r} source={source} "
                f"killed_pids={filtered} killed_count={len(filtered)}",
                flush=True,
            )

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


# (signal name, skip reason). ``None`` means the signal runs.
SignalSpec = tuple[str, str | None]

# SIGTERM and SIGKILL, both executed. Serve-root adds SIGINT on top of this pair.
TERM_KILL_SIGNALS: tuple[SignalSpec, ...] = (("SIGTERM", None), ("SIGKILL", None))
SERVE_ROOT_SIGNALS: tuple[SignalSpec, ...] = (
    ("SIGTERM", None),
    ("SIGINT", None),
    ("SIGKILL", None),
)

INFLIGHT_INJECTION_REQUEST_RATE = 0.3
INFLIGHT_INJECTION_REQUEST_COUNT = 10


class FaultModel:
    """One model's deploy, hardware, and request probes for a fault scenario."""

    model_id: str
    res: dict[str, str]
    num_cards: int
    extra_marks: tuple[Any, ...] = ()
    scenarios: list[dict[str, Any]]
    deploy_dir: Any
    lifecycle_scenario: str | None = None
    eager_oom_stage_ids: tuple[int, ...] | None = None
    worker_pattern: str
    isolation_pattern: str
    hard_oom_config: dict[str, Any]
    recover_oom_config: dict[str, Any] | None = None
    fault_error_keywords: tuple[str, ...]
    process_kill_error_keywords: tuple[str, ...]
    with_load_timeout_sec: float = 120.0
    concurrent_workers: int = 3
    concurrent_timeout_sec: float = 40.0
    health_miss_skip: str | None = None
    worker_signals: tuple[SignalSpec, ...] = TERM_KILL_SIGNALS
    serve_root_signals: tuple[SignalSpec, ...] = SERVE_ROOT_SIGNALS
    serve_root_with_load_signals: tuple[SignalSpec, ...] = SERVE_ROOT_SIGNALS
    tree_signals: tuple[SignalSpec, ...] = TERM_KILL_SIGNALS
    tree_with_load_signals: tuple[SignalSpec, ...] = TERM_KILL_SIGNALS
    oom_large_fail_message: str = "expected request failure during GPU OOM injection"
    oom_concurrent_fail_message: str = "expected concurrent request failure under OOM injection"

    def scenarios_for_server_set(self, server_set: str) -> list[dict[str, Any]]:
        """Return deploy scenarios for one fault server set.

        The ``oom`` set forces ``enforce_eager`` on ``eager_oom_stage_ids``
        under HIP. Models that do not declare those stages keep their deploy.
        """
        scenarios = list(self.scenarios)
        if server_set == "lifecycle":
            if self.lifecycle_scenario is not None:
                scenarios = [item for item in scenarios if item["test_name"] == self.lifecycle_scenario]
        elif server_set == "oom" and self.eager_oom_stage_ids:
            import torch

            scenarios = reliability_scenarios_for_eager_oom(
                scenarios,
                is_hip=bool(getattr(torch.version, "hip", None)),
                stage_ids=self.eager_oom_stage_ids,
            )
        elif server_set not in ("default", "oom"):
            raise ValueError(f"unknown server set {server_set!r} for {self.model_id}")
        return scenarios

    def servers(self, server_set: str) -> list[tuple[str, OmniServerParams]]:
        scenarios = self.scenarios_for_server_set(server_set)
        from tests.dfx.conftest import create_reliability_omni_server_params

        params = create_reliability_omni_server_params(scenarios, self.deploy_dir)
        return [(str(item["test_name"]), param) for item, param in zip(scenarios, params, strict=True)]

    def warmup(self, client: Any, server: Any) -> None:
        raise NotImplementedError(self.model_id)

    def submit_worker_load(self, client: Any, server: Any) -> None:
        self.warmup(client, server)

    def submit_serve_load(self, client: Any, server: Any) -> None:
        self.warmup(client, server)

    def submit_concurrent(self, client: Any, server: Any) -> Any:
        self.warmup(client, server)
        return None

    def assert_fast_fail(self, client: Any, server: Any) -> None:
        raise NotImplementedError(self.model_id)

    def isolation_warmup(self, client: Any, server: Any) -> None:
        self.warmup(client, server)

    def isolation_after(self, client: Any, server: Any) -> None:
        self.assert_fast_fail(client, server)

    def oom_large_probe(self, client: Any, server: Any) -> None:
        raise NotImplementedError(self.model_id)

    def oom_concurrent_probe(self, client: Any, server: Any) -> None:
        raise NotImplementedError(self.model_id)

    def run_error_contract(self, server: Any, client: Any) -> None:
        raise NotImplementedError(self.model_id)

    def run_oom_recovery(self, server: Any, client: Any) -> None:
        raise NotImplementedError(self.model_id)


def _skip_marks(reason: str | None) -> list[Any]:
    if not reason:
        return []
    return [pytest.mark.skip(reason=reason)]


class FaultScenario:
    """Shared fault steps. A model test calls ``run`` with its ``FaultModel``."""

    server_set: str = "default"
    signals_attr: str | None = None

    def run(self, model: FaultModel, server: Any, client: Any, signal_name: str | None = None) -> None:
        raise NotImplementedError(type(self).__name__)


def server_params(
    model: FaultModel,
    *,
    server_set: str = "default",
    skip_reason: str | None = None,
) -> list[Any]:
    """One indirect server param per scenario of ``model``."""
    return [
        pytest.param(server, marks=_skip_marks(skip_reason), id=scenario_name)
        for scenario_name, server in model.servers(server_set)
    ]


def server_and_signal_params(
    model: FaultModel,
    signals_attr: str,
    *,
    server_set: str = "default",
    skip_reason: str | None = None,
) -> list[Any]:
    """Server param plus a direct signal name. A fault skip covers every signal."""
    items: list[Any] = []
    for scenario_name, server in model.servers(server_set):
        for signal_name, signal_skip in getattr(model, signals_attr):
            items.append(
                pytest.param(
                    server,
                    signal_name,
                    marks=_skip_marks(skip_reason or signal_skip),
                    id=f"{scenario_name}__{signal_name.lower()}",
                )
            )
    return items


def scenario_params(
    model: FaultModel,
    scenario: type[FaultScenario],
    *,
    skip_reason: str | None = None,
) -> Any:
    """Parametrize ``omni_server_function`` from the scenario's server set and signals."""
    if scenario.signals_attr is None:
        return pytest.mark.parametrize(
            "omni_server_function",
            server_params(model, server_set=scenario.server_set, skip_reason=skip_reason),
            indirect=True,
        )
    return pytest.mark.parametrize(
        ("omni_server_function", "signal_name"),
        server_and_signal_params(
            model,
            scenario.signals_attr,
            server_set=scenario.server_set,
            skip_reason=skip_reason,
        ),
        indirect=["omni_server_function"],
    )
