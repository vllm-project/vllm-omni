#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run the Omni serving benchmark through a dedicated Python environment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _benchmark_python() -> str:
    configured = os.environ.get("VLLM_OMNI_PYTHON")
    if configured:
        path = Path(configured)
        if not path.is_file():
            raise FileNotFoundError(f"VLLM_OMNI_PYTHON does not exist: {path}")
        return str(path)
    return sys.executable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--request-backend", choices=("sglang_omni",), required=True)
    args, benchmark_args = parser.parse_known_args()

    output_file = Path(args.output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _benchmark_python(),
        "-m",
        "vllm_omni.entrypoints.cli.main",
        "bench",
        "serve",
        "--omni",
        *benchmark_args,
        "--num-warmups",
        "2",
        "--save-result",
        "--result-dir",
        str(output_file.parent),
        "--result-filename",
        output_file.name,
    ]
    subprocess.run(command, cwd=_REPO_ROOT, check=True)
    if not output_file.is_file():
        raise FileNotFoundError(f"Benchmark did not create result JSON: {output_file}")


if __name__ == "__main__":
    main()
