# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import argparse
import json
from typing import Any


def parse_json_object(value: str, flag_name: str = "argument") -> dict[str, Any]:
    """Parse a CLI value as a JSON object, attributing errors to ``flag_name``.

    Args:
        value: JSON-encoded command-line value.
        flag_name: Option name to include in validation errors.

    Returns:
        The decoded dictionary. Field validation belongs to the engine config.

    Raises:
        argparse.ArgumentTypeError: If the value is invalid JSON or is not an
            object (for example, an array, scalar, or null).
    """
    try:
        config = json.loads(value)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"{flag_name} must be valid JSON: {e}") from e
    if not isinstance(config, dict):
        raise argparse.ArgumentTypeError(f"{flag_name} must be a JSON object")
    return config


def parse_profiler_config(value: str) -> dict[str, Any]:
    """Parse ``--profiler-config`` without validating backend-specific fields.

    Raises:
        argparse.ArgumentTypeError: If ``value`` is not a JSON object.
    """
    return parse_json_object(value, flag_name="--profiler-config")


def add_profiler_config_arg(parser: argparse.ArgumentParser) -> None:
    """Register ``--profiler-config`` on an offline example's argument parser.

    The parsed ``profiler_config`` attribute is a dictionary when supplied and
    ``None`` when omitted, independently of ``VLLM_TORCH_PROFILER_DIR``. Forward
    it to ``Omni(profiler_config=...)`` and bracket generation with
    ``start_profile()`` / ``stop_profile()`` when it is not ``None``.

    Args:
        parser: Argument parser on which to register the option once.
    """
    parser.add_argument(
        "--profiler-config",
        type=parse_profiler_config,
        default=None,
        help=(
            "JSON profiler config for torch/cuda profiling, e.g. "
            '\'{"profiler":"torch","torch_profiler_dir":"./perf"}\'.'
        ),
    )
