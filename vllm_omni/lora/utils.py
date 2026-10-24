# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_omni.lora.request import LoRARequest


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


LORA_SCALE_ROUND_DECIMALS = 4


def round_lora_scale(scale: float) -> float:
    """Canonical scale rounding for LoRA identity.

    The scheduler batch key, the KV-cache namespace, and the LoRA manager's
    adapter/composed-adapter keys must all quantize scales identically;
    otherwise two requests can be scheduled apart yet share an adapter composed
    with a different scale (or vice versa).
    """
    return round(float(scale), LORA_SCALE_ROUND_DECIMALS)


def build_lora_batch_key(
    lora_request: LoRARequest | list[LoRARequest] | None,
    lora_scale: float | list[float] | None,
) -> tuple[tuple[int, float], ...] | None:
    """Normalize per-request LoRA identity into a comparable batch key.

    Handles a single adapter or a stacked list of adapters. The result is a
    sorted tuple of ``(lora_int_id, scale)`` pairs with scales quantized by
    ``round_lora_scale``, so the same adapter combination always batches
    together regardless of list order; ``None`` (or an empty list) means no
    LoRA.
    """
    if lora_request is None:
        return None
    requests = list(lora_request) if isinstance(lora_request, (list, tuple)) else [lora_request]
    if not requests:
        return None
    if isinstance(lora_scale, (list, tuple)):
        scales = [float(scale) for scale in lora_scale]
    else:
        scales = [float(lora_scale) if lora_scale is not None else 1.0] * len(requests)
    return tuple(
        sorted(
            (int(request.lora_int_id), round_lora_scale(scales[i] if i < len(scales) else 1.0))
            for i, request in enumerate(requests)
        )
    )


__all__ = ["LORA_SCALE_ROUND_DECIMALS", "build_lora_batch_key", "round_lora_scale", "stable_lora_int_id"]
