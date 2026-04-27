# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
End-to-end test for Bagel text2img generation.

This test validates that the Bagel model generates images that match
expected reference pixel values within a ±5 tolerance.

Equivalent to running:
    python3 examples/offline_inference/bagel/end2end.py \
        --prompts "A futuristic city skyline at twilight, cyberpunk style" \
        --modality text2img --step 15
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import signal
import socket
import subprocess
import tempfile
import time
from typing import Any

import pytest
from PIL import Image

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.platforms import current_omni_platform

BAGEL_CI_DEPLOY = get_deploy_config_path("ci/bagel.yaml")
BAGEL_MOONCAKE_CI_DEPLOY = get_deploy_config_path("ci/bagel_mooncake.yaml")

# Reference pixel data extracted from the known-good output image
# Each entry contains (x, y) position and expected (R, G, B) values
# "Generated with seed=52, num_inference_steps=15,
# prompt='A futuristic city skyline at twilight, cyberpunk style'"
REFERENCE_PIXELS = [
    {"position": (100, 100), "rgb": (115, 113, 94)},
    {"position": (400, 50), "rgb": (159, 160, 144)},
    {"position": (700, 100), "rgb": (164, 151, 123)},
    {"position": (150, 400), "rgb": (120, 121, 107)},
    {"position": (512, 512), "rgb": (165, 133, 127)},
    {"position": (700, 400), "rgb": (217, 130, 66)},
    {"position": (100, 700), "rgb": (191, 168, 152)},
    {"position": (400, 700), "rgb": (130, 96, 77)},
    {"position": (700, 700), "rgb": (247, 203, 140)},
    {"position": (256, 256), "rgb": (167, 156, 150)},
]

if current_omni_platform.is_rocm():
    REFERENCE_PIXELS = [
        {"position": (100, 100), "rgb": (115, 113, 94)},
        {"position": (400, 50), "rgb": (159, 160, 144)},
        {"position": (700, 100), "rgb": (164, 151, 123)},
        {"position": (150, 400), "rgb": (120, 121, 107)},
        {"position": (512, 512), "rgb": (165, 133, 127)},
        {"position": (700, 400), "rgb": (217, 130, 66)},
        {"position": (100, 700), "rgb": (191, 168, 152)},
        {"position": (400, 700), "rgb": (130, 96, 77)},
        {"position": (700, 700), "rgb": (247, 203, 140)},
        {"position": (256, 256), "rgb": (167, 156, 150)},
    ]

# Maximum allowed difference per color channel
PIXEL_TOLERANCE = 5

# Default test prompt
DEFAULT_PROMPT = "<|im_start|>A cute cat<|im_end|>"


def _find_free_port() -> int:
    """Find and return a free ephemeral port by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def _configure_sampling_params(omni: Omni, num_inference_steps: int = 15) -> list:
    """Configure sampling parameters for Bagel text2img generation.

    Args:
        omni: The Omni instance to get default params from.
        num_inference_steps: Number of inference steps for the diffusion stage.

    Returns:
        Configured sampling params list.
    """
    params_list = omni.default_sampling_params_list
    if len(params_list) > 1:
        params_list[1].num_inference_steps = num_inference_steps  # type: ignore
        params_list[1].extra_args = {  # type: ignore
            "cfg_text_scale": 4.0,
            "cfg_img_scale": 1.5,
        }
    return params_list


def _extract_generated_image(omni_outputs: list) -> Image.Image | None:
    """Extract the generated image from Omni outputs.

    Args:
        omni_outputs: List of outputs from omni.generate().

    Returns:
        The first generated PIL Image, or None if no image found.
    """
    for req_output in omni_outputs:
        if images := getattr(req_output, "images", None):
            return images[0]
        if hasattr(req_output, "request_output") and req_output.request_output:
            stage_out = req_output.request_output
            if hasattr(stage_out, "images") and stage_out.images:
                return stage_out.images[0]
    return None


def _validate_pixels(
    image: Image.Image,
    reference_pixels: list[dict[str, Any]] = REFERENCE_PIXELS,
    tolerance: int = PIXEL_TOLERANCE,
) -> None:
    """Validate that image pixels match expected reference values.

    Args:
        image: The PIL Image to validate.
        reference_pixels: List of dicts with 'position' (x, y) and 'rgb' (R, G, B).
        tolerance: Maximum allowed difference per color channel.

    Raises:
        AssertionError: If any pixel differs beyond tolerance.
    """
    for ref in reference_pixels:
        x, y = ref["position"]
        expected = ref["rgb"]
        actual = image.getpixel((x, y))[:3]
        assert all(abs(a - e) <= tolerance for a, e in zip(actual, expected)), (
            f"Pixel mismatch at ({x}, {y}): expected {expected}, got {actual}"
        )


def _generate_bagel_image(omni: Omni, prompt: str = DEFAULT_PROMPT) -> Image.Image:
    """Generate an image using Bagel model with configured parameters.

    Args:
        omni: The Omni instance to use for generation.
        prompt: The text prompt for image generation.

    Returns:
        The generated PIL Image.

    Raises:
        AssertionError: If no image is generated or size is incorrect.
    """
    params_list = _configure_sampling_params(omni)

    omni_outputs = list(
        omni.generate(
            prompts=[{"prompt": prompt, "modalities": ["image"]}],
            sampling_params_list=params_list,
        )
    )

    generated_image = _extract_generated_image(omni_outputs)
    assert generated_image is not None, "No images generated"
    assert generated_image.size == (1024, 1024), f"Expected 1024x1024, got {generated_image.size}"

    return generated_image


def _resolve_deploy_config(config_path: str, run_level: str) -> str:
    """Resolve deploy config based on run level.

    For advanced_model (real weights), strip load_format: dummy so the model
    falls back to loading real weights from HuggingFace.
    """
    if run_level == "advanced_model":
        return modify_stage_config(
            config_path,
            deletes={
                "stages": {
                    0: ["load_format"],
                    1: ["load_format"],
                }
            },
        )
    return config_path


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
def test_bagel_text2img_shared_memory_connector(run_level):
    """Test Bagel text2img with shared memory connector."""
    config_path = _resolve_deploy_config(BAGEL_CI_DEPLOY, run_level)
    with OmniRunner(
        "ByteDance-Seed/BAGEL-7B-MoT",
        stage_configs_path=config_path,
    ) as runner:
        generated_image = _generate_bagel_image(runner.omni)
        if run_level == "advanced_model":
            _validate_pixels(generated_image)


def _wait_for_port(host: str, port: int, timeout: int = 30) -> bool:
    """Wait for a port to become available.

    Args:
        host: The host address.
        port: The port number.
        timeout: Maximum seconds to wait.

    Returns:
        True if port becomes available, False otherwise.
    """
    for _ in range(timeout):
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (TimeoutError, ConnectionRefusedError):
            time.sleep(1)
    return False


def _is_mooncake_master_available() -> bool:
    """Check if mooncake_master binary is present and can actually execute."""
    import shutil

    binary = shutil.which("mooncake_master")
    if binary is None:
        return False
    try:
        result = subprocess.run(
            [binary, "--help"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode != 127
    except (subprocess.TimeoutExpired, OSError):
        return True


def _cleanup_mooncake_processes(timeout_secs: int = 5) -> None:
    """Clean up any existing mooncake_master processes.

    Args:
        timeout_secs: Maximum seconds to wait for graceful termination.
    """
    subprocess.run(
        ["pkill", "-f", "mooncake_master"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    start_time = time.time()
    while time.time() - start_time < timeout_secs:
        result = subprocess.run(
            ["pgrep", "-f", "mooncake_master"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            break
        time.sleep(0.5)
    else:
        subprocess.run(
            ["pkill", "-9", "-f", "mooncake_master"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    time.sleep(1)


def _load_mooncake_config(host: str, rpc_port: int, http_port: int) -> str:
    """Load Mooncake config from CI overlay and substitute placeholders.

    Args:
        host: Mooncake host address.
        rpc_port: RPC port for Mooncake master.
        http_port: HTTP metadata server port.

    Returns:
        Path to the temporary config file with substituted values.
    """
    with open(BAGEL_MOONCAKE_CI_DEPLOY) as f:
        config_content = f.read()

    config_content = config_content.replace("${MOONCAKE_HOST}", host)
    config_content = config_content.replace("${MOONCAKE_RPC_PORT}", str(rpc_port))
    config_content = config_content.replace("${MOONCAKE_HTTP_PORT}", str(http_port))

    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    temp_file.write(config_content)
    temp_file.close()
    return temp_file.name


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_bagel_text2img_mooncake_connector(run_level):
    """Test Bagel text2img with Mooncake connector for inter-stage communication."""
    if not _is_mooncake_master_available():
        raise RuntimeError(
            "mooncake_master is not available or cannot execute (missing shared libraries like libibverbs)"
        )
    MOONCAKE_HOST = "127.0.0.1"
    MOONCAKE_RPC_PORT = _find_free_port()
    MOONCAKE_HTTP_PORT = _find_free_port()
    MOONCAKE_METRICS_PORT = _find_free_port()

    mooncake_master_proc = None
    temp_config_file = None

    try:
        _cleanup_mooncake_processes()

        # Start mooncake_master
        mooncake_master_proc = subprocess.Popen(
            [
                "mooncake_master",
                f"--rpc_port={MOONCAKE_RPC_PORT}",
                "--enable_http_metadata_server=true",
                "--http_metadata_server_host=0.0.0.0",
                f"--http_metadata_server_port={MOONCAKE_HTTP_PORT}",
                f"--metrics_port={MOONCAKE_METRICS_PORT}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )

        assert _wait_for_port(MOONCAKE_HOST, MOONCAKE_RPC_PORT), "mooncake_master failed to start"

        # Create temp config and initialize Omni
        temp_config_file = _load_mooncake_config(
            host=MOONCAKE_HOST,
            rpc_port=MOONCAKE_RPC_PORT,
            http_port=MOONCAKE_HTTP_PORT,
        )

        temp_config_file = _resolve_deploy_config(temp_config_file, run_level)
        with OmniRunner(
            "ByteDance-Seed/BAGEL-7B-MoT",
            stage_configs_path=temp_config_file,
            stage_init_timeout=300,
        ) as runner:
            generated_image = _generate_bagel_image(runner.omni)
            if run_level == "advanced_model":
                _validate_pixels(generated_image)

    finally:
        if temp_config_file:
            try:
                os.unlink(temp_config_file)
            except OSError:
                pass
        if mooncake_master_proc:
            try:
                os.killpg(os.getpgid(mooncake_master_proc.pid), signal.SIGKILL)
            except OSError:
                pass
