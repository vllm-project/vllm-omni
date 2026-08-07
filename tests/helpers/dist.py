# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Distributed test helpers using file-based rendezvous.

Avoids TCP port conflicts in CI by using a temporary file for
``torch.distributed`` rendezvous instead of ``MASTER_ADDR``/``MASTER_PORT``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch


@contextmanager
def file_rendezvous(prefix: str):
    """Context manager that yields a file:// init method string.

    This creates a temporary directory with a rendezvous file and cleans it
    up on exit.  You can pass the yielded string as ``distributed_init_method``
    when calling ``init_distributed_environment()``.

    Example:

        with file_rendezvous(prefix="example_") as init_method:
            torch.multiprocessing.spawn(
                worker_fn,
                args=(world_size, init_method),
                nprocs=world_size,
            )
    """
    tmpdir = tempfile.mkdtemp(prefix=prefix)
    try:
        yield f"file://{os.path.join(tmpdir, 'rendezvous')}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def set_dist_env(*, rank: int, world_size: int) -> None:
    """Set minimal env vars for the distributed environment."""
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)


def clear_dist_env() -> None:
    """Remove distributed env vars potentially left over from other tests."""
    for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"]:
        os.environ.pop(key, None)


def spawn_with_file_rendezvous(
    worker_fn: Callable[..., None],
    *,
    world_size: int,
    args: tuple[Any, ...] = (),
    prefix: str,
) -> None:
    """Spawn workers with file-based rendezvous.

    The first two arguments to the worker_fn should be `(world_size, init_method)`,
    then any other positional args."""
    with file_rendezvous(prefix=prefix) as init_method:
        torch.multiprocessing.spawn(
            worker_fn,
            args=(world_size, init_method, *args),
            nprocs=world_size,
        )
