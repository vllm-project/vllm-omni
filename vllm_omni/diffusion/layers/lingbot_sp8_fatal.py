# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in SP8 failure receipts for the root-owned eight-worker supervisor.

Only standard-library imports. This module retains owners and reports; it never
signals processes, changes runtime policy, retries kernels, or rolls back caches.
The supervisor verifies nonce/PID/start time/group before reclaiming its workers.
"""

import json
import os
import sys
import threading
import time
from pathlib import Path

_DIRECTORY = os.environ.get("ROOT_LINGBOT_SP8_FATAL_DIR", "")
_NONCE = os.environ.get("ROOT_LINGBOT_SP8_FATAL_NONCE", "")
ENABLED = bool(_DIRECTORY or _NONCE)
_QUARANTINE = []
_LOCK = threading.Lock()
_COUNTER = 0


def ensure_protocol():
    """Validate the launch-owned channel before admitting custom writes."""
    if not ENABLED:
        return False
    if len(_NONCE) != 32 or any(value not in "0123456789abcdef" for value in _NONCE):
        raise RuntimeError("SP8 fatal channel requires a root-configured 32-hex nonce")
    directory = Path(_DIRECTORY)
    if not directory.is_absolute() or not directory.is_dir():
        raise RuntimeError("SP8 fatal channel must be an existing launch-owned absolute directory")
    return True


def _process_identity():
    identity: dict[str, int | str | None] = {}
    try:
        # After the final ')' the list starts at stat field3; starttime is22.
        identity["process_start_ticks"] = int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])
        identity["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except (OSError, ValueError, IndexError):
        identity["process_start_ticks"] = None
        identity["boot_id"] = None
    return identity


def report_failure(origin, reason, owners):
    """Hold all owners, atomically notify the supervisor, preserve original error.

    Reporting failure cannot enable a native retry. A broken reporting channel
    also emits a structured stderr record for the same supervisor to inspect.
    """
    global _COUNTER
    owner_tuple = tuple(owners)
    with _LOCK:
        _QUARANTINE.append(owner_tuple)
        _COUNTER += 1
        counter = _COUNTER
    if not ENABLED:
        return None
    pid = os.getpid()
    record = dict(
        version=1,
        kind="lingbot_sp8_fatal",
        nonce=_NONCE,
        pid=pid,
        ppid=os.getppid(),
        origin=str(origin)[:256],
        reason=str(reason)[:4096],
        owner_count=len(owner_tuple),
        after_write_may_have_occurred=True,
        restart_scope="all8",
        timestamp_ns=time.time_ns(),
        rank_environment=os.environ.get("RANK"),
        local_rank_environment=os.environ.get("LOCAL_RANK"),
        visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        **_process_identity(),
    )
    temporary = None
    try:
        ensure_protocol()
        directory = Path(_DIRECTORY)
        name = f"fatal-{_NONCE}-{pid}-{counter}.json"
        target = directory / name
        temporary = directory / (name + ".tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(record, handle, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        return str(target)
    except BaseException as error:
        record.update(kind="lingbot_sp8_fatal_report_failed", report_error=str(error)[:4096])
        try:
            sys.stderr.write("LINGBOT_SP8_FATAL_REPORT_FAILED " + json.dumps(record, allow_nan=False) + "\n")
            sys.stderr.flush()
        except BaseException:
            pass
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        return None
