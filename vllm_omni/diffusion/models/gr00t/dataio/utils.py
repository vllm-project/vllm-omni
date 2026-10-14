# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

import numpy as np

from vllm_omni.diffusion.models.gr00t.configs.embodiment_configs import ModalityConfig


def apply_sin_cos_encoding(values: np.ndarray) -> np.ndarray:
    """Apply sin/cos encoding to values.

    Args:
        values: Array of shape (..., D) containing values to encode

    Returns:
        Array of shape (..., 2*D) with [sin, cos] concatenated

    Note: This DOUBLES the dimension. For example:
        Input:  [v₁, v₂, v₃] with shape (..., 3)
        Output: [sin(v₁), sin(v₂), sin(v₃), cos(v₁), cos(v₂), cos(v₃)] with shape (..., 6)
    """
    sin_values = np.sin(values)
    cos_values = np.cos(values)
    # Concatenate sin and cos: [sin(v1), sin(v2), ..., cos(v1), cos(v2), ...]
    return np.concatenate([sin_values, cos_values], axis=-1)


def nested_dict_to_numpy(data: Any) -> Any:
    """Recursively convert bottom-level list-of-lists to NumPy arrays.

    Args:
        data: A nested dictionary whose leaves are lists (or lists of lists).

    Returns:
        The same dictionary structure with leaf lists converted to ``np.ndarray``.
    """
    if isinstance(data, dict):
        return {key: nested_dict_to_numpy(value) for key, value in data.items()}
    elif isinstance(data, list):
        # Convert lists to numpy arrays
        # NumPy will handle both 1D and 2D cases appropriately
        return np.array(data)
    else:
        return data


def normalize_values_minmax(values: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    """Min-max normalize ``values`` to ``[-1, 1]`` using ``params['min']`` and ``params['max']``.

    Accepts 2D ``(T, D)`` or 3D ``(B, T, D)`` arrays. Features with ``min == max``
    are emitted as 0.
    """
    min_vals = params["min"]
    max_vals = params["max"]
    normalized = np.zeros_like(values)

    mask = ~np.isclose(max_vals, min_vals)

    normalized[..., mask] = (values[..., mask] - min_vals[..., mask]) / (max_vals[..., mask] - min_vals[..., mask])
    normalized[..., mask] = 2 * normalized[..., mask] - 1

    return normalized


def unnormalize_values_minmax(normalized_values: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    """Inverse of :func:`normalize_values_minmax`.

    Input values are clipped to ``[-1, 1]`` before being mapped back to
    ``[params['min'], params['max']]``.
    """

    min_vals = params["min"]
    max_vals = params["max"]
    range_vals = max_vals - min_vals

    # Unnormalize from [-1, 1]
    unnormalized = (np.clip(normalized_values, -1.0, 1.0) + 1.0) / 2.0 * range_vals + min_vals
    return unnormalized


def normalize_values_meanstd(values: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    """Z-score normalize ``values`` using ``params['mean']`` and ``params['std']``.

    Features whose ``std == 0`` are passed through unchanged.
    """
    mean_vals = params["mean"]
    std_vals = params["std"]

    # Create mask for non-zero standard deviations
    mask = std_vals != 0

    # Initialize normalized array
    normalized = np.zeros_like(values)

    # Normalize only features with non-zero std
    normalized[..., mask] = (values[..., mask] - mean_vals[..., mask]) / std_vals[..., mask]

    # Keep original values for zero-std features
    normalized[..., ~mask] = values[..., ~mask]

    return normalized


def unnormalize_values_meanstd(normalized_values: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    """Inverse of :func:`normalize_values_meanstd` (``x * std + mean``).

    Features whose ``std == 0`` are passed through unchanged.
    """
    mean_vals = params["mean"]
    std_vals = params["std"]

    # Create mask for non-zero standard deviations
    mask = std_vals != 0

    # Initialize unnormalized array
    unnormalized = np.zeros_like(normalized_values)

    # Unnormalize only features with non-zero std
    unnormalized[..., mask] = normalized_values[..., mask] * std_vals[..., mask] + mean_vals[..., mask]

    # Keep normalized values for zero-std features
    unnormalized[..., ~mask] = normalized_values[..., ~mask]

    return unnormalized


def to_json_serializable(obj: Any) -> Any:
    """
    Recursively convert dataclasses and numpy arrays to JSON-serializable format.

    Args:
        obj: Object to convert (can be dataclass, numpy array, dict, list, etc.)

    Returns:
        JSON-serializable representation of the object
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        # Convert dataclass to dict, then recursively process the dict
        return to_json_serializable(asdict(obj))
    elif isinstance(obj, np.ndarray):
        # Convert numpy array to list
        return obj.tolist()
    elif isinstance(obj, np.integer):
        # Convert numpy integers to Python int
        return int(obj)
    elif isinstance(obj, np.floating):
        # Convert numpy floats to Python float
        return float(obj)
    elif isinstance(obj, np.bool_):
        # Convert numpy bool to Python bool
        return bool(obj)
    elif isinstance(obj, dict):
        # Recursively process dictionary values
        return {key: to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Recursively process list/tuple elements
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, set):
        # Convert set to list
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        # Already JSON-serializable
        return obj
    elif isinstance(obj, Enum):
        return obj.name
    else:
        # For other types, try to convert to string as fallback
        # You might want to handle specific types differently
        return str(obj)


def parse_modality_configs(
    modality_configs: dict[str, dict[str, ModalityConfig]],
) -> dict[str, dict[str, ModalityConfig]]:
    parsed_modality_configs = {}
    for embodiment_tag, modality_config in modality_configs.items():
        parsed_modality_configs[embodiment_tag] = {}
        for modality, config in modality_config.items():
            if isinstance(config, dict):
                parsed_modality_configs[embodiment_tag][modality] = ModalityConfig(**config)
            else:
                parsed_modality_configs[embodiment_tag][modality] = config
    return parsed_modality_configs
