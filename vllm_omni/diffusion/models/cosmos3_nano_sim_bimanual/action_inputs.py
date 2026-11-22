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


def prepare_domain_ids(value: Any, *, rows: int, allowed: set[int]) -> torch.Tensor:
    """Validate scalar or per-action-row routing before session mutation."""
    if isinstance(value, int) and not isinstance(value, bool):
        value = [value]
    if isinstance(value, torch.Tensor):
        if value.dtype not in (torch.int32, torch.int64):
            raise ValueError("Domain IDs must be integers")
        value = value.detach().cpu().reshape(-1).tolist()
    if not isinstance(value, (list, tuple)) or len(value) not in (1, rows):
        raise ValueError("Domain IDs must contain one ID or one per action row")
    if any(type(v) is not int or v not in allowed for v in value):
        raise ValueError("Domain IDs must name trained domains")
    return torch.tensor(value, dtype=torch.long)


def domains_for_frames(
    domains: torch.Tensor,
    *,
    layout: str | None,
    request_start_frame: int,
    frame_start: int,
    frame_end: int,
    action_count: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Slice routing with exactly the same prefix and row offsets as actions."""
    if domains.numel() == 1:
        return domains.to(device)
    blocks = []
    for frame in range(frame_start, frame_end):
        if frame == 0:
            blocks.append(domains[:1].expand(action_count))
        else:
            start = (frame - (1 if layout == "global" else max(request_start_frame, 1))) * action_count
            blocks.append(domains[start : start + action_count])
    selected = torch.cat(blocks)
    # Most chunks remain single-domain even in mixed rollouts. Preserve the
    # existing batch projection path and avoid repeated weight gathers there.
    if bool((selected == selected[0]).all()):
        return selected[:1].to(device)
    return selected.unsqueeze(0).to(device)
