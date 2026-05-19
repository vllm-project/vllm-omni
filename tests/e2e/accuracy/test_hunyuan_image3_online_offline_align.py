# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import base64
import io
import shutil
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from tests.e2e.accuracy.helpers import assert_similarity, model_output_dir

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]

MODEL_NAME = "/data/HunyuanImage-3.0-Instruct"
SEED = 42
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 5.0
HEIGHT, WIDTH = 1024, 1024
PSNR_THRESHOLD = 40.0
SSIM_THRESHOLD = 0.99
PROMPT = "A brown and white dog is running on the grass."


async def run_offline_inference(model_path: str, output_path: Path) -> tuple[Image.Image, str]:
    """Run offline inference using the end2end.py script."""
    import subprocess
    import sys

    script_path = Path(__file__).resolve().parents[3] / "examples" / "offline_inference" / "hunyuan_image3" / "end2end.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--modality", "text2img",
        "--model", model_path,
        "--prompts", PROMPT,
        "--bot-task", "think",
        "--sys-type", "en_unified",
        "--seed", str(SEED),
        "--steps", str(NUM_INFERENCE_STEPS),
        "--output", str(output_path),
        "--deploy-config", "vllm_omni/deploy/hunyuan_image3_dit.yaml",
        "--height", str(HEIGHT),
        "--width", str(WIDTH),
        "--guidance-scale", str(GUIDANCE_SCALE),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Offline inference failed: {result.stderr}"

    image_files = list(output_path.glob("output_0_0.png"))
    assert len(image_files) == 1, f"Expected 1 image, found {len(image_files)}"

    image = Image.open(image_files[0]).convert("RGB")

    cot_output = ""
    for line in result.stdout.split("\n"):
        if "[Output] Text:" in line:
            cot_output = "\n".join(result.stdout.split("[Output] Text:\n")[1].split("[Output] Saved image")[0].strip().split("\n"))
            break

    return image, cot_output


async def run_online_inference(model_path: str) -> tuple[Image.Image, str]:
    """Run online inference using the OpenAI-compatible API."""
    import httpx
    import subprocess
    import sys
    import time

    server_cmd = [
        "vllm", "serve",
        model_path,
        "--omni",
        "--host", "localhost",
        "--port", "8091",
        "--deploy-config", "vllm_omni/deploy/hunyuan_image3_dit.yaml",
        "--enforce-eager",
    ]

    server_process = subprocess.Popen(
        server_cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    health_url = "http://localhost:8091/health"
    start_time = time.time()
    while time.time() - start_time < 300:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(health_url, timeout=2)
            if resp.status_code == 200:
                print("Online server ready!")
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        server_process.terminate()
        raise RuntimeError("Online server failed to start within 5 minutes")

    try:
        payload = {
            "prompt": PROMPT,
            "use_system_prompt": "en_unified",
            "bot_task": "think",
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "n": 1,
            "seed": SEED,
            "size": f"{WIDTH}x{HEIGHT}",
            "guidance_scale": GUIDANCE_SCALE,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8091/v1/images/generations",
                json=payload,
                timeout=300,
            )

        assert response.status_code == 200, f"Online inference failed: {response.text}"
        data = response.json()
        assert "data" in data and len(data["data"]) > 0, "No images in response"

        b64_json = data["data"][0]["b64_json"]
        assert b64_json, "No b64_json in response"

        img_bytes = base64.b64decode(b64_json)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        cot_output = data.get("cot_output", "")

        return image, cot_output
    finally:
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_process.kill()


async def test_online_offline_align(accuracy_artifact_root: Path) -> None:
    """Test alignment between online and offline inference for HunyuanImage-3."""
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_NAME + "-online-offline-align")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Run offline inference
            print("Running offline inference...")
            offline_image, _ = await run_offline_inference(MODEL_NAME, tmp)
            offline_image.save(output_dir / "offline_image.png")

            # Run online inference
            print("Running online inference...")
            online_image, _ = await run_online_inference(MODEL_NAME)
            online_image.save(output_dir / "online_image.png")

            # Compare images
            print("\n--- Alignment ---")
            assert_similarity(
                model_name=f"{MODEL_NAME} online vs offline",
                vllm_image=online_image,
                diffusers_image=offline_image,
                ssim_threshold=SSIM_THRESHOLD,
                psnr_threshold=PSNR_THRESHOLD,
                width=WIDTH,
                height=HEIGHT,
            )

    finally:
        print(f"\nCleaning up {output_dir}")
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        artifacts_dir = accuracy_artifact_root
        if artifacts_dir.exists() and not any(artifacts_dir.iterdir()):
            artifacts_dir.rmdir()


if __name__ == "__main__":
    asyncio.run(test_online_offline_align(Path("./output")))