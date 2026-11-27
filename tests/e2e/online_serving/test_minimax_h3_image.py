# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H3-Image serving: 4K output transfer on Rubin, and T2I smoke on Ascend NPUs."""

import base64
import hashlib
import io
import json
import os
from urllib.parse import urlsplit

import pytest

# === BEGIN temporary diagnostic probe (A5 torch/triton double registration) ===
# Reproduce/inspect the "Only a single TORCH_LIBRARY can be used to register the
# namespace triton" failure seen only on the A5 NPU agent. Remove before merge.
import sys
import traceback


def _a5_torch_probe() -> None:
    import os as _os
    import pathlib as _pathlib
    import subprocess as _sp

    import torch as _torch

    print("PROBE torch file:", _torch.__file__)
    print("PROBE torch version:", _torch.__version__)
    print(
        "PROBE plugin autoload:",
        _os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "enabled(default)"),
    )
    try:
        _pkgs = _sp.check_output(
            [_sp.sys.executable, "-m", "pip", "list", "--format=freeze"]
        ).decode().splitlines()
        print(
            "PROBE torch/triton pkgs:",
            [p for p in _pkgs if any(k in p.lower() for k in ("torch", "triton"))],
        )
    except Exception as _e:  # noqa: BLE001
        print("PROBE pip list failed:", _e)
    print(
        "PROBE sys.meta_path:",
        [m.__name__ if hasattr(m, "__name__") else str(m) for m in sys.meta_path],
    )
    print("PROBE sys.path:", sys.path)
    _torch_dir = _pathlib.Path(_torch.__file__).parent.parent
    for _pth in _torch_dir.glob("*.pth"):
        print("PROBE .pth file:", _pth, "->", _pth.read_text().strip())

    # Deliberately force a second execution of torch/__init__.py to capture the
    # full traceback of where the duplicate "triton" TORCH_LIBRARY happens.
    sys.modules.pop("torch", None)
    try:
        import torch  # noqa: F401
    except RuntimeError:
        print("PROBE reimport traceback:")
        for _fr in traceback.extract_tb(sys.exc_info()[2]):
            print(f"PROBE   {_fr.filename}:{_fr.lineno} in {_fr.name}: {_fr.line}")
        raise


_a5_torch_probe()
# === END temporary diagnostic probe ===

from PIL import Image

from tests.helpers.client import OnlineOmniClient
from tests.helpers.mark import hardware_marks, hardware_test
from tests.helpers.runtime import OmniServer, OmniServerParams

MODEL = os.environ.get("VLLM_TEST_MINIMAX_H3_IMAGE_MODEL", "MiniMaxAI/MiniMax-H3-Image")


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parallel
@hardware_test(res={"cuda": "B200"}, num_cards=4)
def test_repeated_native_4k_outputs_complete_with_identical_pixels() -> None:
    """A 2K smoke test misses the repeated 16MP transfer stall on Rubin."""
    base_url = os.environ.get("H3_IMAGE_TEST_BASE_URL")
    if not base_url:
        pytest.skip("Set H3_IMAGE_TEST_BASE_URL to a resident four-GPU SP4 H3-Image server")
    payload = {
        "model": "MiniMaxAI/MiniMax-H3-Image",
        "prompt": "A red bicycle leaning against a pale stone wall in warm afternoon sunlight.",
        "size": "4096x4096",
        "num_inference_steps": 6,
        "seed": 42,
        "generator_device": "cpu",
        "n": 1,
        "response_format": "b64_json",
        "output_format": "png",
    }
    address = urlsplit(base_url)
    assert address.scheme == "http" and address.hostname and address.port
    client = OnlineOmniClient(host=address.hostname, port=address.port, run_level="core_model")
    pixel_hashes = []
    try:
        for _ in range(3):
            response = client.send_images_generations_http_request({"json": payload, "timeout": 120})[0]
            images = response.json_body["data"]
            png = base64.b64decode(images[0]["b64_json"], validate=True)
            with Image.open(io.BytesIO(png)) as image:
                image.load()
                assert image.format == "PNG"
                assert image.size == (4096, 4096)
                rgb = image.convert("RGB")
                assert any(low != high for low, high in rgb.getextrema())
                pixel_hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
    finally:
        client.client.close()
    assert len(set(pixel_hashes)) == 1


# --- Ascend NPU A3/A5 T2I smoke ---

NPU_SERVER_ARGS = [
    "--enforce-eager",
    "--num-gpus",
    "2",
    "--usp",
    "2",
    "--text-encoder-tp-size",
    "2",
    "--diffusion-offload-config",
    json.dumps(
        {
            "mode": "layer",
            "components": ["dit"],
            "layer_options": {"dit": {"weight_transfer": "allgather"}},
        },
        separators=(",", ":"),
    ),
]
A3_TWO_CARD_MARKS = hardware_marks(res={"npu": "A3"}, num_cards=2)
A5_TWO_CARD_MARKS = hardware_marks(res={"npu": "A5"}, num_cards=2)


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parallel
@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=NPU_SERVER_ARGS,
                stage_init_timeout=1800,
                init_timeout=1800,
            ),
            id="minimax_h3_image_a3_dlo_sp2_t2i",
            marks=A3_TWO_CARD_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=NPU_SERVER_ARGS,
                stage_init_timeout=1800,
                init_timeout=1800,
            ),
            id="minimax_h3_image_a5_dlo_sp2_t2i",
            marks=A5_TWO_CARD_MARKS,
        ),
    ],
    indirect=True,
)
def test_npu_text_to_image(omni_server: OmniServer, online_client: OnlineOmniClient) -> None:
    """Validate T2I with SP2, text-encoder TP2, and DiT AllGather layer offload."""
    online_client.send_images_generations_http_request(
        {
            "json": {
                "model": omni_server.model,
                "prompt": "A red bicycle leaning against a pale stone wall in warm afternoon sunlight.",
                "size": "1024x1024",
                "num_inference_steps": 6,
                "seed": 42,
                "n": 1,
                "response_format": "b64_json",
                "output_format": "png",
            },
            "timeout": 300,
        }
    )
