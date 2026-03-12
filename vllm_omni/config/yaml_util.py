# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Centralized OmegaConf wrapper for vLLM-Omni.

All OmegaConf usage in the project MUST go through this module.
Other modules should import these wrapper functions instead of
using OmegaConf directly.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_yaml_config(path: str | Any) -> DictConfig:
    """Load a YAML file and return it as a DictConfig.

    Args:
        path: Path to the YAML file.

    Returns:
        OmegaConf DictConfig with attribute-style access.
    """
    return OmegaConf.load(path)  # type: ignore[return-value]


def create_config(data: Any) -> DictConfig:
    """Wrap a dict (or list) into a DictConfig.

    Args:
        data: Dict, list, or other structure to wrap.

    Returns:
        OmegaConf DictConfig / ListConfig.
    """
    return OmegaConf.create(data)


def merge_configs(*cfgs: Any) -> dict:
    """Deep-merge multiple configs and return a plain dict.

    Args:
        *cfgs: DictConfig or dict objects to merge (left to right).

    Returns:
        Plain dict with merged, resolved values.
    """
    merged = OmegaConf.merge(*cfgs)
    return OmegaConf.to_container(merged, resolve=True)  # type: ignore[return-value]


def to_dict(obj: Any, *, resolve: bool = True) -> Any:
    """Convert a DictConfig (or similar) to a plain dict.

    Args:
        obj: OmegaConf container to convert.
        resolve: Whether to resolve interpolations (default True).

    Returns:
        Plain dict.
    """
    return OmegaConf.to_container(obj, resolve=resolve)  # type: ignore[return-value]
