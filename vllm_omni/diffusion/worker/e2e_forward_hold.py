# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Optional gate that blocks COMPUTE_DONE until an e2e client releases it.

Used by request-streaming e2e to prove media arrived while the forward is still
in flight: the worker writes ``{gate}.holding`` then waits for ``{gate}``.
Unset ``VLLM_OMNI_E2E_HOLD_FORWARD_GATE`` keeps production paths unchanged.
"""

from __future__ import annotations

import os
import time
from pathlib import Path


def maybe_hold_before_compute_done() -> None:
    """Block until the e2e release file appears; no-op when the gate is unset."""
    gate = os.environ.get("VLLM_OMNI_E2E_HOLD_FORWARD_GATE")
    if not gate:
        return
    path = Path(gate)
    holding = Path(f"{gate}.holding")
    holding.parent.mkdir(parents=True, exist_ok=True)
    holding.write_text("holding\n", encoding="utf-8")
    timeout_s = float(os.environ.get("VLLM_OMNI_E2E_HOLD_FORWARD_TIMEOUT_S", "180"))
    deadline = time.monotonic() + timeout_s
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"e2e forward hold timed out after {timeout_s:.1f}s waiting for release file {gate!r}"
            )
        time.sleep(0.05)
