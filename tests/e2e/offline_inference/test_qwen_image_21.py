# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path

import pytest
from PIL import Image

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OfflineOmniClient
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "Qwen/Qwen-Image-2.1"
RUNNER = (MODEL, None, {"enforce_eager": True})
INPUT_IMAGE = Path(__file__).resolve().parents[2] / "assets/qwen_image_edit/qwen_image_edit_2511_test1.png"


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "npu": "A5"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", [RUNNER], indirect=True)
@pytest.mark.parametrize("scenario", ["t2i", "i2i"])
def test_generate_image(offline_client: OfflineOmniClient, scenario: str):
    """Exercise prefill/decode through the public model API on either platform."""
    sampling = OmniDiffusionSamplingParams(
        height=512,
        width=512,
        num_inference_steps=4,
        seed=42,
        true_cfg_scale=1.0,
        num_outputs_per_prompt=1,
    )
    request: dict[str, object] = {
        "model": MODEL,
        "prompt": "A ceramic teapot on a wooden table",
        "sampling_params": sampling,
    }
    if scenario == "i2i":
        request["prompt"] = "Turn this image into a warm oil painting, preserving its subject and pose."
        with Image.open(INPUT_IMAGE) as image:
            request["images"] = image.convert("RGB")

    response = offline_client.send_diffusion_request(request)
    assert response.success and response.images is not None
    assert len(response.images) == 1
    image = response.images[0]
    assert image.size == (512, 512)
    assert any(low < high for low, high in image.convert("RGB").getextrema())
