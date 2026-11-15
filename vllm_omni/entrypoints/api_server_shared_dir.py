# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A scratch directory shared by the API processes of one multi-API server.

The multi-API launcher creates it before spawning the API processes, passes
its path to them in their arguments, and removes it at shutdown. Frontend
components that cooperate across API processes (for example the MOSS
reference encoder) keep their files under it, so the sharing is scoped to one
server's lifetime.
"""

from __future__ import annotations

import os
import shutil
import tempfile

# Short local parents first: Unix sockets kept in the directory must fit in
# sun_path (about 100 bytes), which a long TMPDIR can exceed.
_PARENTS = ("/dev/shm", "/tmp")


def create_api_server_shared_dir() -> str:
    """Create the directory; the launcher passes its path to the API processes."""
    for parent in _PARENTS:
        if os.access(parent, os.W_OK | os.X_OK):
            return tempfile.mkdtemp(prefix="vllm-omni-api-", dir=parent)
    return tempfile.mkdtemp(prefix="vllm-omni-api-")


def remove_api_server_shared_dir(path: str) -> None:
    """Remove the directory once every API process has exited."""
    shutil.rmtree(path, ignore_errors=True)
