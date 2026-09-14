# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""WF-07 E2E against running ComfyUI and vLLM-Omni H3 services."""

import json
import os
import secrets
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.gpu]


def test_minimax_h3_upscale_e2e(tmp_path: Path) -> None:
    comfy_url = os.environ.get("COMFYUI_URL", "")
    server_url = os.environ.get("VLLM_OMNI_URL", "")
    if not comfy_url or not server_url:
        pytest.skip("Set COMFYUI_URL and VLLM_OMNI_URL to run the live WF-07 E2E test.")

    root = Path(__file__).resolve().parents[4]
    output_dir = Path(os.environ.get("WF07_OUTPUT_DIR", str(tmp_path)))
    # Record the seed in the submitted graph; a new seed avoids ComfyUI's output cache.
    seed = os.environ.get("WF07_SEED", str(secrets.randbits(32)))
    command = [
        sys.executable,
        str(root / "apps/ComfyUI-vLLM-Omni/scripts/validate_h3_upscale.py"),
        "--comfy-url",
        comfy_url,
        "--server-url",
        server_url,
        "--seed",
        seed,
        "--output-dir",
        str(output_dir),
    ]
    if lora_path := os.environ.get("WF07_LORA_PATH"):
        command.extend(["--lora-path", lora_path])

    # The runner submits the graph, waits for real inference/upscale, and downloads both MP4s.
    subprocess.run(command, check=True, timeout=7300)
    report = json.loads((output_dir / "validation.json").read_text())
    assert report["passed"], report
    assert report["checks"]["remote_generation_executed"], report
