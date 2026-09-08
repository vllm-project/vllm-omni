# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import argparse
import json
from typing import Any


def parse_json_object(value: str, flag_name: str = "argument") -> dict[str, Any]:
    """Parse a CLI value as a JSON object, attributing errors to ``flag_name``."""
    try:
        config = json.loads(value)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"{flag_name} must be valid JSON: {e}") from e
    if not isinstance(config, dict):
        raise argparse.ArgumentTypeError(f"{flag_name} must be a JSON object")
    return config


def parse_profiler_config(value: str) -> dict[str, Any]:
    """Parse the JSON object passed to ``--profiler-config``."""
    return parse_json_object(value, flag_name="--profiler-config")


def add_profiler_config_arg(parser: argparse.ArgumentParser) -> None:
    """Add the shared JSON profiler configuration option to ``parser``."""
    parser.add_argument(
        "--profiler-config",
        type=parse_profiler_config,
        default=None,
        help=(
            "JSON profiler config for torch/cuda profiling, e.g. "
            '\'{"profiler":"torch","torch_profiler_dir":"./perf"}\'.'
        ),
    )
