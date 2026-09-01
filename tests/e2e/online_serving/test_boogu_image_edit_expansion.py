# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
L4 expansion coverage for Boogu-Image Edit and Edit-Turbo (image editing / TI2I).

The nightly smoke module (``test_boogu_image_edit.py``) already covers single-image
text-guided editing and the text+image double-guidance path. This expansion
module adds the two edit paths smoke does not exercise:

- ``test_image_only_guidance`` — ``guidance_scale=1.0`` (text CFG off) with
  ``guidance_scale_2=2.0`` (image CFG on): the image-only branch of upstream's
  CFG-branch priority (double > text-only > image-only > t2i), which no existing
  e2e test reaches.
- ``test_align_res_output_size`` — a non-square reference with no requested
  ``height`` / ``width``: asserts the output resolution follows the reference
  (upstream ``align_res``, on by default for a single reference).

Both Edit checkpoints share ``BooguImagePipeline`` with the Base checkpoint;
the TI2I path activates when the request carries a reference image. Only a
single reference image is supported. The regular Edit cases stay smoke-depth
(``num_inference_steps=2``), while Edit-Turbo uses its tuned four-step
configuration. CPU offload, Cache-DiT, and multi-GPU parallelism are not yet
supported for these models and are intentionally omitted.

From ``tests/``::

    pytest -s -v e2e/online_serving/test_boogu_image_edit_expansion.py \
        -m "slow and diffusion and H100"
"""

import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler, dummy_messages_from_mix_data

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

pytestmark = [pytest.mark.diffusion, pytest.mark.slow]

MODEL = "Boogu/Boogu-Image-0.1-Edit"
EDIT_TURBO_MODEL = "Boogu/Boogu-Image-0.1-Edit-Turbo"
EDIT_TURBO_REVISION = "hotfix-1k-20260708"
EDIT_PROMPT = "Change the style to a colored pencil drawing."

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})


def _get_diffusion_feature_cases(model: str):
    """Return the single-card server configuration supported for Boogu-Image-Edit today."""
    return [
        pytest.param(
            OmniServerParams(model=model),
            id="default",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
    ]


def _get_align_res_cases():
    """Return regular and Turbo Edit configurations for align-res coverage."""
    return [
        pytest.param(
            OmniServerParams(model=MODEL),
            {
                "num_inference_steps": 2,
                "guidance_scale": 5.0,
            },
            id="edit",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=EDIT_TURBO_MODEL,
                server_args=["--revision", EDIT_TURBO_REVISION],
            ),
            {
                "num_inference_steps": 4,
                "guidance_scale": 1.0,
            },
            id="edit-turbo-hotfix-1k",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
    ]


@pytest.mark.parametrize("omni_server", _get_diffusion_feature_cases(MODEL), indirect=True)
def test_image_only_guidance(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """Image-only guidance: text CFG off (1.0), image CFG on (2.0) -> image-only CFG branch."""
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"
    messages = dummy_messages_from_mix_data(image_data_url=image_data_url, content_text=EDIT_PROMPT)
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "num_inference_steps": 2,
            # Text guidance off, image guidance on -> upstream's image-only branch.
            "guidance_scale": 1.0,
            "guidance_scale_2": 2.0,
            "seed": 42,
        },
    }
    openai_client.send_diffusion_request(request_config)


@pytest.mark.parametrize(
    ("omni_server", "sampling_params"),
    _get_align_res_cases(),
    indirect=["omni_server"],
)
def test_align_res_output_size(
    omni_server: OmniServer,
    sampling_params: dict,
    openai_client: OpenAIClientHandler,
) -> None:
    """Output resolution follows the reference image (upstream ``align_res``).

    No ``height`` / ``width`` is requested. The reference is 512x384 — both
    already multiples of 32 (>= the ``vae_scale_factor * 2`` alignment) and well
    under the VAE ``max_pixels`` / ``max_side`` limits — so the align-res target
    equals the reference dims regardless of the exact VAE scale factor.
    """
    ref_width, ref_height = 512, 384
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(ref_width, ref_height)['base64']}"
    messages = dummy_messages_from_mix_data(image_data_url=image_data_url, content_text=EDIT_PROMPT)
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            **sampling_params,
            "seed": 42,
        },
    }
    responses = openai_client.send_diffusion_request(request_config)

    assert responses and responses[0].images, "No image returned for the edit request"
    out_image = responses[0].images[0]
    assert (out_image.width, out_image.height) == (ref_width, ref_height), (
        f"align_res should size the output to the reference dims {(ref_width, ref_height)}; "
        f"got {(out_image.width, out_image.height)}"
    )
