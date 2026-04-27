# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Subprocess coverage for the `vllm generate` CLI."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from vllm_omni.platforms import current_omni_platform

MODEL = "riverclouds/qwen_image_random"
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cuda
def test_vllm_generate_cli_writes_image(tmp_path):
    if not current_omni_platform.is_cuda():
        pytest.skip("vllm generate CLI subprocess test requires CUDA.")
    if current_omni_platform.get_device_count() < 1:
        pytest.skip("vllm generate CLI subprocess test requires at least one CUDA GPU.")
    if importlib.util.find_spec("vllm.inputs") is None:
        pytest.skip("vllm generate CLI subprocess test requires a compatible vLLM installation.")

    output = tmp_path / "generated"
    env = os.environ.copy()
    env["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm_omni.entrypoints.cli.main",
            "generate",
            "--model",
            MODEL,
            "--prompt",
            "a photo of a cat sitting on a laptop keyboard",
            "--output",
            str(output),
            "--height",
            "256",
            "--width",
            "256",
            "--num-inference-steps",
            "2",
            "--guidance-scale",
            "0.0",
            "--seed",
            "42",
        ],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    output_path = output.with_suffix(".png")
    assert output_path.exists(), result.stderr
    with Image.open(output_path) as image:
        assert image.size == (256, 256)
