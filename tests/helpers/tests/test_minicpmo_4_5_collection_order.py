# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Lock MiniCPM-o 4.5 merge collection so one async_chunk server is reused."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ONLINE = "tests/e2e/online_serving/test_minicpmo_4_5.py"


def _param_id(nodeid: str) -> str | None:
    if not nodeid.endswith("]"):
        return None
    opening = nodeid.rfind("[")
    if opening < 0:
        return None
    return nodeid[opening + 1 : -1]


def test_async_chunk_cases_stay_one_contiguous_block() -> None:
    """Merge ``-m 'advanced_model and cuda'`` keeps id ``async_chunk`` adjacent.

    This does not lock Ready (``core_model``) or nightly (``full_model``).
    ``omni_server`` is module-scoped. pytest 9.1.1 drops it when the next
    item's param value differs and does not revive it, so a different deploy
    between two ``async_chunk`` cases cold-starts the same server again.
    The 300s limit is the abort ceiling for importing this module, not a budget.
    """
    env = os.environ.copy()
    # A worker-scoped outer run must not make this collection look distributed.
    env.pop("PYTEST_XDIST_WORKER", None)
    env.pop("PYTEST_XDIST_WORKER_COUNT", None)
    env.pop("PYTEST_CURRENT_TEST", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            _ONLINE,
            "-m",
            "advanced_model and cuda",
            "--run-level",
            "advanced_model",
        ],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    nodeids = [line.strip() for line in completed.stdout.splitlines() if "::" in line]
    param_ids = [_param_id(nodeid) for nodeid in nodeids]
    positions = [index for index, param_id in enumerate(param_ids) if param_id == "async_chunk"]
    assert len(positions) >= 2, param_ids
    start = positions[0]
    assert positions == list(range(start, start + len(positions))), param_ids
