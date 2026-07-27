# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""L4 online-serving tests for the Krea 2 text-to-image diffusion pipeline.

Exercises the OpenAI-compatible ``/v1/images/generations`` route (served via
``vllm serve <model> --omni --enforce-eager``) rather than the offline engine path
covered by ``tests/e2e/offline_inference/test_krea2_expansion.py``.

Runs against the public few-step distilled checkpoint ``krea/Krea-2-Turbo`` by default
(override with the ``KREA2_MODEL`` environment variable, e.g. ``krea/Krea-2-Raw`` for the
Raw checkpoint or a local diffusers directory). Krea 2 shares the Qwen-Image VAE and a
similar single-stream MMDiT, so this test mirrors the structure/fixtures/markers of
``test_qwen_image_expansion.py``.

Coverage mirrors the offline suite: a basic functional smoke plus the layerwise-CPU-offload
path that the pipeline declares via ``SupportsComponentDiscovery`` /
``_layerwise_offload_blocks_attrs``. Sizing is kept light (512x512, few steps) so the case
fits single-GPU H100 CI rather than the full 2048x2048 distilled resolution.
"""

import base64
import os
from io import BytesIO

import numpy as np
import pytest
import requests
from PIL import Image

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler, dummy_messages_from_mix_data

pytestmark = [pytest.mark.diffusion, pytest.mark.slow]

MODEL = os.environ.get("KREA2_MODEL", "krea/Krea-2-Turbo")
# vLLM-Omni-compatible PEFT repackaging of krea/Krea-2-LoRA-darkbrush (264 modules, r=alpha=32).
LORA = os.environ.get("KREA2_LORA", "NagaSaiAbhinay/Krea-2-vllm-darkbrush-LoRA")
PROMPT = "a fox in the snow, photorealistic"
# darkbrush is a trigger-word LoRA; include its trigger phrase so the LoRA case has a strong signal.
LORA_PROMPT = "a fox in the snow, monochrome ink wash style"

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})


def _get_diffusion_feature_cases(model: str):
    """Return single-card L4 online-serving cases for Krea 2.

    Mirrors the offline suite: a basic functional smoke plus layerwise CPU offload.
    """
    return [
        # Basic functional smoke (single-card).
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=["--enforce-eager"],
            ),
            id="single_card_basic",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
        # Layerwise CPU offload (single-card): SupportsComponentDiscovery +
        # _layerwise_offload_blocks_attrs.
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=["--enforce-eager", "--enable-layerwise-offload"],
            ),
            id="single_card_layerwise_offload",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
    ]


@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases(MODEL),
    indirect=True,
)
def test_krea2(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Serve Krea 2 and validate the OpenAI-compatible image-generation route.

    guidance_scale resolution is checkpoint-aware inside the pipeline (distilled -> no-CFG,
    Raw -> CFG), so guidance_scale=0.0 stays agnostic to which checkpoint KREA2_MODEL points at.
    Validation is delegated to assert_diffusion_response, which checks output dimensions and
    basic correctness.
    """
    messages = dummy_messages_from_mix_data(content_text=PROMPT)
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 8,
            "guidance_scale": 0.0,
            "seed": 42,
        },
    }
    openai_client.send_diffusion_request(request_config)


def _basic_lora_payload(lora: dict | None = None) -> dict:
    payload = {
        "prompt": LORA_PROMPT,
        "n": 1,
        "size": "512x512",
        "num_inference_steps": 8,
        "guidance_scale": 0.0,
        "seed": 42,
    }
    if lora is not None:
        payload["lora"] = lora
    return payload


def _post_image(server: OmniServer, payload: dict) -> np.ndarray:
    url = f"http://{server.host}:{server.port}/v1/images/generations"
    resp = requests.post(url, json=payload, headers={"Authorization": "Bearer EMPTY"}, timeout=900)
    resp.raise_for_status()
    b64 = resp.json()["data"][0]["b64_json"]
    img = Image.open(BytesIO(base64.b64decode(b64)))
    img.load()
    return np.asarray(img.convert("RGB"), dtype=np.int16)


@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            OmniServerParams(model=MODEL, server_args=["--enforce-eager"]),
            id="single_card_lora",
            marks=SINGLE_CARD_FEATURE_MARKS,
        )
    ],
    indirect=True,
)
def test_krea2_lora(omni_server: OmniServer):
    """Serve Krea 2 and validate per-request LoRA over ``/v1/images/generations``.

    Passes ``LORA`` via the top-level ``lora`` object (``{name, path, scale}``) the diffusion
    image route parses into a ``LoRARequest`` — no server-side LoRA flag needed. Confirms the
    adapter has a visible effect and that omitting ``lora`` cleanly restores the base output.
    """
    baseline = _post_image(omni_server, _basic_lora_payload())
    with_lora = _post_image(omni_server, _basic_lora_payload({"name": "darkbrush", "path": LORA, "scale": 1.0}))
    restored = _post_image(omni_server, _basic_lora_payload())

    diff_lora = np.abs(baseline - with_lora).mean()
    diff_restored = np.abs(baseline - restored).mean()

    assert diff_lora > 0.5, f"LoRA had no visible effect over the serving path: diff={diff_lora}"
    assert diff_restored < 5.0, f"LoRA did not deactivate cleanly: diff_restored={diff_restored:.2f}"
