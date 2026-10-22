# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Optional action-shape contract at the OpenPI serving boundary."""

from collections.abc import Mapping
from typing import Any, TypeAlias

import numpy as np

ActionOutput: TypeAlias = np.ndarray | dict[str, np.ndarray]


def _optional_int(values: Mapping[str, Any], key: str, *, minimum: int = 1) -> int | None:
    value = values.get(key)
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return int(value)


def validate_action_output(
    actions: ActionOutput,
    action_metadata: Mapping[str, Any] | None,
    policy_server_config: Mapping[str, Any],
) -> None:
    """Validate advertised constraints without changing the wire representation.

    Dense chunks use [H, D] or [B, H, D]. Named chunks share the same
    leading dimensions, with independently sized final dimensions. Scalar
    action_dim also applies to the sole chunk in a single-key mapping. Legacy
    vectors remain accepted when no explicit shape contract is advertised.
    Unknown metadata, including raw model dimensions, is left untouched.
    """
    if action_metadata is not None and not isinstance(action_metadata, Mapping):
        raise ValueError("metadata.actions must be a mapping")
    metadata = action_metadata if action_metadata is not None else {}
    fixed_horizon = _optional_int(policy_server_config, "action_horizon")
    max_horizon = _optional_int(policy_server_config, "max_action_horizon")
    configured_dim = _optional_int(policy_server_config, "action_dim")
    horizon = _optional_int(metadata, "horizon")
    action_dim = _optional_int(metadata, "action_dim")
    valid_steps = _optional_int(metadata, "valid_steps", minimum=0)
    if fixed_horizon is not None and max_horizon is not None and fixed_horizon > max_horizon:
        raise ValueError("action_horizon exceeds max_action_horizon")

    chunks: Mapping[str, np.ndarray]
    if isinstance(actions, Mapping):
        named = True
        chunks = actions
    else:
        named = False
        chunks = {"actions": actions}
    if not chunks:
        raise ValueError("Action output must not be empty")
    keys = policy_server_config.get("action_keys")
    if keys is not None:
        if not isinstance(keys, (list, tuple)) or not all(isinstance(key, str) for key in keys):
            raise ValueError("action_keys must be a list of strings")
        if not named or len(set(keys)) != len(keys) or set(chunks) != set(keys):
            raise ValueError("Action output keys do not match policy_server_config.action_keys")
    if named and len(chunks) > 1 and (configured_dim is not None or action_dim is not None):
        raise ValueError("Scalar action_dim applies only to dense or single-key actions; ambiguous for multiple keys")

    shape_contract = any(
        value is not None
        for value in (fixed_horizon, max_horizon, configured_dim, horizon, action_dim, valid_steps, keys)
    )
    leading_shape = None
    for key, chunk in chunks.items():
        if not isinstance(chunk, np.ndarray) or chunk.size == 0:
            raise ValueError(f"Action chunk {key!r} must be a non-empty array")
        if not np.isfinite(chunk).all():
            raise ValueError(f"Action chunk {key!r} contains non-finite values")
        if chunk.ndim == 1 and not named and not shape_contract:
            continue
        if chunk.ndim not in (2, 3):
            raise ValueError(f"Action chunk {key!r} must have shape [H, D] or [B, H, D]")
        if leading_shape is not None and chunk.shape[:-1] != leading_shape:
            raise ValueError("Named action chunks must share batch and horizon dimensions")
        leading_shape = chunk.shape[:-1]
        actual_horizon, actual_dim = chunk.shape[-2:]
        for expected in (fixed_horizon, horizon):
            if expected is not None and actual_horizon != expected:
                raise ValueError(f"Action horizon {actual_horizon} does not match declared horizon {expected}")
        if max_horizon is not None and actual_horizon > max_horizon:
            raise ValueError(f"Action horizon {actual_horizon} exceeds max_action_horizon {max_horizon}")
        if valid_steps is not None and valid_steps > actual_horizon:
            raise ValueError("valid_steps exceeds action horizon")
        for expected in (configured_dim, action_dim):
            if expected is not None and actual_dim != expected:
                raise ValueError(f"Action dimension {actual_dim} does not match declared action_dim {expected}")

    for source in (policy_server_config, metadata):
        space = source.get("action_space")
        if space is not None and (not isinstance(space, str) or not space.strip()):
            raise ValueError("action_space must be a non-empty string")
    configured_space = policy_server_config.get("action_space")
    output_space = metadata.get("action_space")
    if configured_space is not None and output_space is not None and configured_space != output_space:
        raise ValueError("Output action_space does not match policy_server_config.action_space")
