"""Shared helpers for multimodal output handling."""

from __future__ import annotations

from typing import Any

import torch


def _to_cpu(x: Any) -> Any:
    """Move a tensor to CPU; pass through non-tensors unchanged."""
    if isinstance(x, torch.Tensor):
        try:
            return x.detach().to("cpu", non_blocking=True).contiguous()
        except (RuntimeError, AttributeError):
            return x
    if isinstance(x, dict):
        return {str(sub_key): _to_cpu(sub_value) for sub_key, sub_value in x.items()}
    return x


def _is_tensor_list(value: Any) -> bool:
    """Return True if *value* is a non-empty list of tensors (deferred concat)."""
    return isinstance(value, list) and bool(value) and isinstance(value[0], torch.Tensor)
