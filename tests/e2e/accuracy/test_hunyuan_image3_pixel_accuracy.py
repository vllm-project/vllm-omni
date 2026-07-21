from __future__ import annotations

import base64
import copy
import io
import os
import tempfile
from pathlib import Path

import pytest
import requests
import yaml
from PIL import Image

from tests.e2e.accuracy.helpers import assert_images_pixel_close, assert_similarity, model_output_dir
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]

MODEL_NAME = "tencent/HunyuanImage-3.0-Instruct"
SEED = 42
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 2.5
HEIGHT = 1024
WIDTH = 1024
PROMPT = "A brown and white dog is running on the grass."
MEAN_THRESHOLD = 3e-2
P99_THRESHOLD = 3e-1
SSIM_THRESHOLD = 0.97
PSNR_THRESHOLD = 30.0

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BASELINE_PATH = _REPO_ROOT / "tests" / "assets" / "hunyuan" / "hunyuan_baseline.png"
_OFFLINE_SCRIPT = _REPO_ROOT / "examples" / "offline_inference" / "hunyuan_image3" / "end2end.py"

# DiT-only deploy config with trust_remote_code (based on hunyuan_image3_dit.yaml).
_DEPLOY_CONFIG = {
    "pipeline": "hunyuan_image3_dit",
    "async_chunk": False,
    "trust_remote_code": True,
    "stages": [
        {
            "stage_id": 0,
            "max_num_seqs": 1,
            "gpu_memory_utilization": 0.9,
            "enforce_eager": True,
            "trust_remote_code": True,
            "devices": "0,1,2,3",  # set dynamically by _write_deploy_config
            "vae_use_slicing": False,
            "moe_backend": "flashinfer_cutlass",
            "vae_use_tiling": False,
            "parallel_config": {
                "pipeline_parallel_size": 1,
                "data_parallel_size": 1,
                "tensor_parallel_size": 4,
                "enable_expert_parallel": True,
                "sequence_parallel_size": 1,
                "ulysses_degree": 1,
                "ring_degree": 1,
                "cfg_parallel_size": 1,
                "vae_patch_parallel_size": 1,
                "use_hsdp": False,
                "hsdp_shard_size": -1,
                "hsdp_replicate_size": 1,
            },
            "default_sampling_params": {
                "seed": SEED,
            },
        },
    ],
}


def _model_name() -> str:
    return os.environ.get("HUNYUAN_IMAGE3_MODEL", MODEL_NAME)


def _devices() -> str:
    return os.environ.get("HUNYUAN_IMAGE3_DEVICES", "0,1,2,3")


def _write_deploy_config(path: Path) -> None:
    config = copy.deepcopy(_DEPLOY_CONFIG)
    devices = _devices()
    config["stages"][0]["devices"] = devices
    config["stages"][0]["parallel_config"]["tensor_parallel_size"] = len(devices.split(","))
    path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))


def _run_vllm_omni_hunyuan_image3_online(*, model: str, deploy_config: str, output_path: Path) -> Image.Image:
    server_args = [
        "--deploy-config",
        deploy_config,
        "--stage-init-timeout",
        "300",
        "--init-timeout",
        "900",
        "--enforce-eager",
        "--trust-remote-code",
    ]
    with OmniServer(model, server_args, use_omni=True) as omni_server:
        response = requests.post(
            f"http://{omni_server.host}:{omni_server.port}/v1/images/generations",
            json={
                "model": omni_server.model,
                "prompt": PROMPT,
                "size": f"{WIDTH}x{HEIGHT}",
                "n": 1,
                "response_format": "b64_json",
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "guidance_scale": GUIDANCE_SCALE,
                "seed": SEED,
                "bot_task": "none",
                "use_system_prompt": "en_unified",
            },
            timeout=600,
        )
        response.raise_for_status()
        payload = response.json()
        assert len(payload["data"]) == 1
        image_bytes = base64.b64decode(payload["data"][0]["b64_json"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.load()
        image.save(output_path)
        return image


def _run_vllm_omni_hunyuan_image3_offline(*, model: str, deploy_config: str, output_path: Path) -> Image.Image:
    import subprocess
    import sys

    output_dir = str(output_path.parent)
    subprocess.run(
        [
            sys.executable,
            str(_OFFLINE_SCRIPT),
            "--modality",
            "text2img",
            "--deploy-config",
            deploy_config,
            "--prompts",
            PROMPT,
            "--output",
            output_dir,
            "--steps",
            str(NUM_INFERENCE_STEPS),
            "--guidance-scale",
            str(GUIDANCE_SCALE),
            "--seed",
            str(SEED),
            "--height",
            str(HEIGHT),
            "--width",
            str(WIDTH),
            "--bot-task",
            "none",
            "--sys-type",
            "en_unified",
            "--model",
            model,
            "--enforce-eager",
        ],
        check=True,
    )
    images = sorted(Path(output_dir).glob("output_*.png"))
    assert images, f"No output image found in {output_dir}"
    image = Image.open(images[0]).convert("RGB")
    image.load()
    image.save(output_path)
    return image


def _assert_against_baseline(image: Image.Image, label: str) -> None:
    assert BASELINE_PATH.exists(), f"Baseline image not found at {BASELINE_PATH}"
    baseline_image = Image.open(BASELINE_PATH).convert("RGB")

    assert_images_pixel_close(
        model_name=f"{MODEL_NAME} ({label} vs baseline)",
        vllm_image=image,
        diffusers_image=baseline_image,
        mean_threshold=MEAN_THRESHOLD,
        p99_threshold=P99_THRESHOLD,
    )
    assert_similarity(
        model_name=f"{MODEL_NAME} ({label} vs baseline)",
        vllm_image=image,
        diffusers_image=baseline_image,
        ssim_threshold=SSIM_THRESHOLD,
        psnr_threshold=PSNR_THRESHOLD,
    )


@hardware_test(res={"cuda": "H100"}, num_cards=4)
def test_hunyuan_image3_pixel_accuracy_online(accuracy_artifact_root: Path) -> None:
    model = _model_name()
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_NAME)

    with tempfile.TemporaryDirectory() as tmpdir:
        deploy_config_path = Path(tmpdir) / "deploy.yaml"
        _write_deploy_config(deploy_config_path)
        image = _run_vllm_omni_hunyuan_image3_online(
            model=model, deploy_config=str(deploy_config_path), output_path=output_dir / "vllm_omni_online.png"
        )
    _assert_against_baseline(image, "online")


@hardware_test(res={"cuda": "H100"}, num_cards=4)
def test_hunyuan_image3_pixel_accuracy_offline(accuracy_artifact_root: Path) -> None:
    model = _model_name()
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_NAME)

    with tempfile.TemporaryDirectory() as tmpdir:
        deploy_config_path = Path(tmpdir) / "deploy.yaml"
        _write_deploy_config(deploy_config_path)
        image = _run_vllm_omni_hunyuan_image3_offline(
            model=model, deploy_config=str(deploy_config_path), output_path=output_dir / "vllm_omni_offline.png"
        )
    _assert_against_baseline(image, "offline")
