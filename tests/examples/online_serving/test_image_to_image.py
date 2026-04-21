"""
Online serving tests: image-to-image.
See examples/online_serving/image_to_image/README.md
"""

import base64
import json
import sys
from pathlib import Path

import pytest
import requests

from tests.conftest import OmniServer, OmniServerParams, assert_image_valid
from tests.examples.conftest import EXAMPLES, OUTPUT_DIR, run_cmd
from tests.utils import hardware_marks

pytestmark = [pytest.mark.advanced_model, pytest.mark.example, *hardware_marks(res={"cuda": "H100"})]

I2I_ONLINE_CLIENT = EXAMPLES / "online_serving" / "image_to_image" / "openai_chat_client.py"
EXAMPLE_OUTPUT_SUBFOLDER = "example_online_i2i"

# ---------------------------------------------------------------------------
# Server parameter sets.
# Tests sharing the same param list reuse one module-scoped server instance,
# so keep tests that use the same server adjacent to each other.
# ---------------------------------------------------------------------------

qwen_image_edit_server_params = [OmniServerParams(model="Qwen/Qwen-Image-Edit")]
qwen_image_layered_server_params = [OmniServerParams(model="Qwen/Qwen-Image-Layered")]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def example_output_dir() -> Path:
    d = OUTPUT_DIR / EXAMPLE_OUTPUT_SUBFOLDER
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="module")
def input_image() -> Path:
    """Return a locally generated synthetic PNG (514×556 RGB) as the I2I input image.

    Uses PIL instead of a runtime network fetch so CI stays hermetic even on
    runners without outbound internet access or during transient S3 issues.
    """
    from PIL import Image

    image_path = OUTPUT_DIR / "qwen-bear.png"
    if not image_path.exists():
        Image.new("RGB", (514, 556), color=(128, 160, 200)).save(image_path)
    return image_path


# ---------------------------------------------------------------------------
# Method 1: Using curl
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("omni_server", qwen_image_edit_server_params, indirect=True)
def test_api_calls_001(omni_server: OmniServer, example_output_dir: Path, input_image: Path):
    """curl: single-image edit via /v1/chat/completions."""
    case_dir = example_output_dir / "api_calls-001"
    case_dir.mkdir(parents=True, exist_ok=True)
    out = case_dir / "api_calls_001.png"

    img_b64 = base64.b64encode(input_image.read_bytes()).decode()
    url = f"http://{omni_server.host}:{omni_server.port}/v1/chat/completions"

    payload = json.dumps(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Convert this image to watercolor style"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    ],
                }
            ],
            "extra_body": {
                "num_inference_steps": 50,
                "guidance_scale": 1.0,
                "seed": 42,
            },
        }
    )

    req_file = case_dir / "request.json"
    req_file.write_text(payload, encoding="utf-8")

    run_cmd(
        f"curl -s '{url}'"
        " -H 'Content-Type: application/json'"
        f" -d @'{req_file}'"
        " | jq -r '.choices[0].message.content[0].image_url.url'"
        f" | cut -d',' -f2- | base64 -d > '{out}'",
        shell=True,
    )
    assert_image_valid(out)


# ---------------------------------------------------------------------------
# Method 2: Using Python Client Script
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("omni_server", qwen_image_edit_server_params, indirect=True)
def test_api_calls_002(omni_server: OmniServer, example_output_dir: Path, input_image: Path):
    """openai_chat_client.py: single-image edit."""
    case_dir = example_output_dir / "api_calls-002"
    case_dir.mkdir(parents=True, exist_ok=True)
    out = case_dir / "api_calls_002.png"

    run_cmd(
        [
            sys.executable,
            str(I2I_ONLINE_CLIENT),
            "--input",
            str(input_image),
            "--prompt",
            "Convert to oil painting style",
            "--output",
            str(out),
            "--server",
            f"http://{omni_server.host}:{omni_server.port}",
            "--steps",
            "50",
            "--seed",
            "42",
        ]
    )
    assert_image_valid(out)


# ---------------------------------------------------------------------------
# Method 3: Gradio Demo — skipped (interactive UI, not automatable)
# ---------------------------------------------------------------------------


@pytest.mark.skip("README section 'Method 4: Using Gradio Demo' is intentionally excluded for example tests")
def test_api_calls_003(): ...


# ---------------------------------------------------------------------------
# Layered Image Generation (Qwen-Image-Layered)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("omni_server", qwen_image_layered_server_params, indirect=True)
def test_layered_001(omni_server: OmniServer, example_output_dir: Path, input_image: Path):
    """API call: layered image generation (Qwen-Image-Layered).

    Qwen-Image-Layered returns all generated RGBA layer images as separate
    content items inside ``choices[0].message.content``.  This test calls the
    API directly (rather than through the example client, which only saves the
    first layer) so that every returned layer is decoded, written to disk, and
    validated as a well-formed image.
    """
    case_dir = example_output_dir / "layered-001"
    case_dir.mkdir(parents=True, exist_ok=True)

    img_b64 = base64.b64encode(input_image.read_bytes()).decode()
    url = f"http://{omni_server.host}:{omni_server.port}/v1/chat/completions"
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a rabbit"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            }
        ],
        "extra_body": {
            "num_inference_steps": 50,
            "seed": 0,
        },
    }

    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]

    assert isinstance(content, list) and len(content) > 0, (
        f"Expected at least one image in response content, got: {content!r}"
    )

    for idx, item in enumerate(content):
        image_url = item.get("image_url", {}).get("url", "")
        assert image_url.startswith("data:image"), f"Layer {idx}: expected a data-URI image, got: {image_url!r}"
        _, b64_data = image_url.split(",", 1)
        layer_path = case_dir / f"layered_001_layer_{idx}.png"
        layer_path.write_bytes(base64.b64decode(b64_data))
        assert_image_valid(layer_path)
