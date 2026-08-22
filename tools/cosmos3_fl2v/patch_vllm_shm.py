#!/usr/bin/env python3
"""Fix an upstream vLLM shared-memory deadlock that hangs large FL2V requests.

WHY THIS EXISTS
  FL2V locks the *last* latent, so the server must decode a reference video of
  `max(index) * 4 + 1` frames — 189 frames for the demo defaults. Decoded to raw
  pixels on the server that is 189 * 720 * 1280 * 3 = ~500 MiB, which the engine
  process must hand to the diffusion worker process.

  Stock Cosmos3 video-to-video conditions on latents [0, 1], i.e. 5 frames
  (~13 MiB), so ordinary usage never approaches this size and never trips the bug.

THE BUG (vllm/distributed/device_communicators/shm_broadcast.py)
  Requests travel through a shared-memory ring buffer with 24 MB chunks. Anything
  larger takes an "overflow" path: a one-byte flag goes into the ring buffer and
  the real payload goes over a ZMQ socket. `MessageQueue.dequeue()` then does:

    1. acquire the ring slot, read the flag, release the slot  <- slot consumed
    2. seeing the overflow flag, recv() the payload from the socket

  The diffusion worker calls `dequeue(timeout=1.0)`, and that same 1 s timeout is
  passed down to the socket recv in step 2. A ~500 MiB payload does not arrive in
  one second, so recv raises TimeoutError and the worker's busy loop swallows it.

  At that point the request is silently destroyed: the ring slot is already
  marked read, so nothing records that a message is pending, and the payload sits
  in the socket with no reader that will ever return for it. The worker waits for
  a new message forever; the engine waits for a reply forever. No error, no
  timeout, no GPU activity — just a permanent hang.

THE FIX
  Block instead of timing out in step 2. Once the overflow flag has been
  consumed the writer is already committed to sending, so the only open question
  is how long the copy takes: waiting is always correct, timing out never is.

USAGE
  python patch_vllm_shm.py            # apply (idempotent)
  python patch_vllm_shm.py --check    # report status; exit 1 if unpatched
  python patch_vllm_shm.py --revert   # restore the original from the backup

  Run it inside the virtualenv that runs the *server*; it edits the installed
  vLLM in site-packages. Re-run it after any `pip install`/`uv pip install` that
  reinstalls vLLM. Verified against vllm 0.26.0.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

RELATIVE_TARGET = Path("distributed") / "device_communicators" / "shm_broadcast.py"
BACKUP_SUFFIX = ".fl2v-orig"

ORIGINAL = """            if overflow:
                obj = MessageQueue.recv(self.local_socket, timeout)
"""

PATCHED = """            if overflow:
                # FL2V patch (see vllm-omni/patch_vllm_shm.py): the ring slot was
                # already consumed above, so the writer is committed to sending
                # this payload and the only question is how long the copy takes.
                # Timing out here drops the slot and orphans the socket message,
                # deadlocking reader and writer. Large reference videos (FL2V
                # sends ~500 MiB of frames) never make the caller's 1 s deadline.
                obj = MessageQueue.recv(self.local_socket, None)
"""

MARKER = "FL2V patch (see vllm-omni/patch_vllm_shm.py)"


def find_target() -> Path:
    """Locate shm_broadcast.py in the installed vLLM without importing vLLM."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(
            "vllm is not installed in this environment.\n"
            "Activate the venv that runs the server, then re-run this script."
        )
    root = Path(list(spec.submodule_search_locations)[0])
    target = root / RELATIVE_TARGET
    if not target.is_file():
        raise SystemExit(
            f"expected file not found: {target}\nvLLM's layout changed; re-check the fix against your version."
        )
    return target


def is_patched(text: str) -> bool:
    return MARKER in text


def apply(target: Path) -> int:
    text = target.read_text()
    if is_patched(text):
        print(f"already patched: {target}")
        return 0
    if ORIGINAL not in text:
        raise SystemExit(
            f"could not find the code to patch in {target}\n"
            "This vLLM version differs from the one this fix targets (0.26.0).\n"
            "Check MessageQueue.dequeue(): the overflow branch must not pass the\n"
            "caller's timeout to MessageQueue.recv()."
        )
    backup = target.with_name(target.name + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"backed up original -> {backup}")
    target.write_text(text.replace(ORIGINAL, PATCHED, 1))
    print(f"patched: {target}")
    print("Restart the vLLM-Omni server for this to take effect.")
    return 0


def check(target: Path) -> int:
    if is_patched(target.read_text()):
        print(f"patched: {target}")
        return 0
    print(f"NOT patched: {target}", file=sys.stderr)
    print("Large FL2V requests will hang. Run: python patch_vllm_shm.py", file=sys.stderr)
    return 1


def revert(target: Path) -> int:
    backup = target.with_name(target.name + BACKUP_SUFFIX)
    if not backup.exists():
        raise SystemExit(f"no backup to restore: {backup}")
    shutil.copy2(backup, target)
    print(f"restored original from {backup}")
    print("Restart the vLLM-Omni server for this to take effect.")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Report status; exit 1 if unpatched")
    mode.add_argument("--revert", action="store_true", help="Restore the original file")
    args = p.parse_args()

    target = find_target()
    if args.check:
        raise SystemExit(check(target))
    if args.revert:
        raise SystemExit(revert(target))
    raise SystemExit(apply(target))


if __name__ == "__main__":
    main()
