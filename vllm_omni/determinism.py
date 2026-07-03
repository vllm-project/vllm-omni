# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import os
from typing import Any

_TRUE_VALUES = {"1", "true", "yes", "on"}


def is_batch_invariant_enabled() -> bool:
    """Return whether rollout scheduling should avoid arrival-order effects."""
    return os.environ.get("VLLM_BATCH_INVARIANT", "").strip().lower() in _TRUE_VALUES


def deterministic_request_key(request: Any) -> tuple[int, str]:
    """Stable ordering key for deterministic scheduling."""
    request_id = getattr(request, "request_id", None)
    if request_id is None:
        request_ids = getattr(request, "request_ids", None)
        if request_ids:
            request_id = request_ids[0]
    return (int(getattr(request, "priority", 0) or 0), str(request_id or ""))


def deterministic_sample_seed(base_seed: int, sample_id: str) -> int:
    """Derive a stable per-sample seed from a request seed and sample id."""
    payload = f"{int(base_seed)}:{sample_id}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % (2**63)
