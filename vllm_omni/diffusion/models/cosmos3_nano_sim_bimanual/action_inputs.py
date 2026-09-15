# SPDX-License-Identifier: Apache-2.0
"""Validation of raw and model-space action inputs, before session mutation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def validate_action_values(value: Any, *, width: int) -> torch.Tensor:
    """Reject coercible strings, booleans, complex values and non-finite actions."""

    def check_scalars(item: Any) -> None:
        if isinstance(item, list | tuple):
            for child in item:
                check_scalars(child)
        elif isinstance(item, bool | np.bool_) or not isinstance(item, int | float | np.integer | np.floating):
            raise ValueError("Actions must contain numeric values, excluding booleans.")

    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bool or value.is_complex():
            raise ValueError("Actions must contain real numeric values, excluding booleans.")
        action = value.detach().to(dtype=torch.float32)
    else:
        if isinstance(value, list | tuple):
            check_scalars(value)
        array = np.asarray(value)
        if array.dtype.kind not in "iuf":
            raise ValueError("Actions must contain real numeric values, excluding booleans.")
        action = torch.as_tensor(array, dtype=torch.float32)
    if action.ndim == 3 and action.shape[0] == 1:
        action = action.squeeze(0)
    if action.ndim != 2 or action.shape[1] != width or action.shape[0] == 0:
        raise ValueError(f"Expected non-empty actions of shape [N, {width}], got {tuple(action.shape)}.")
    if not torch.isfinite(action).all():
        raise ValueError("Actions must contain only finite values.")
    return action


def prepare_action_values(
    value: Any, *, width: int, model_width: int, action_space: str, normalizer: Any
) -> torch.Tensor:
    """Normalize raw actions once; preserve explicitly model-space values."""
    if action_space not in ("raw", "model"):
        raise ValueError("action_space must be 'raw' or 'model'.")
    action = validate_action_values(value, width=width)
    if action_space == "raw":
        action = normalizer.normalize(action)
    if model_width < width:
        raise ValueError("Model action width must cover the embodiment width.")
    return torch.nn.functional.pad(action, (0, model_width - width))
