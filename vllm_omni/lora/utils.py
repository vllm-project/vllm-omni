# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib


def stable_lora_int_id(lora_path: str) -> int:
    """Return a deterministic positive integer ID for a LoRA adapter.

    vLLM uses `lora_int_id` as the adapter's cache key. Python's built-in
    `hash()` is intentionally randomized per process (PYTHONHASHSEED), which
    makes it unsuitable for persistent IDs. This helper derives a stable
    63-bit positive integer from the adapter path.
    """
    digest = hashlib.sha256(lora_path.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False) & ((1 << 63) - 1)
    return value or 1


__all__ = ["stable_lora_int_id"]
