# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Online serving smokes for all four canonical LTX-2.5 pipelines."""

import os

import pytest

from tests.helpers import skip_if_gated_repo_inaccessible
from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

DEFAULT_MODEL = "Lightricks/LTX-2.5-Diffusers"
DEFAULT_REVISION = "a6de4b5354f078db24d9cf4778c14846788aea3d"
MODEL = os.getenv("VLLM_TEST_LTX25_MODEL", DEFAULT_MODEL)
MODEL_REVISION = os.getenv("VLLM_TEST_LTX25_MODEL_REVISION", DEFAULT_REVISION if MODEL == DEFAULT_MODEL else "")
PROMPT = "A red fox walks through a snowy forest while the camera remains fixed."

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]
SINGLE_CARD_MARKS = hardware_marks(res={"cuda": "H100"})


@pytest.fixture(scope="module", autouse=True)
def require_ltx25_model_access() -> None:
    if not os.path.isdir(MODEL):
        skip_if_gated_repo_inaccessible(
            MODEL,
            revision=MODEL_REVISION or None,
            filename="model_index.json",
        )


def _server(model_class_name: str) -> OmniServerParams:
    return OmniServerParams(
        model=MODEL,
        server_args=[
            *(["--revision", MODEL_REVISION] if MODEL_REVISION else []),
            "--model-class-name",
            model_class_name,
            "--enforce-eager",
            "--enable-layerwise-offload",
            "--diffusion-attention-backend",
            "CUDNN_ATTN",
        ],
    )


def _cases():
    return [
        pytest.param(_server("LTX2Pipeline"), 30, id="full_one_stage", marks=SINGLE_CARD_MARKS),
        pytest.param(_server("LTX2TwoStagePipeline"), 30, id="full_two_stage", marks=SINGLE_CARD_MARKS),
        pytest.param(_server("LTX2DistilledOneStagePipeline"), 8, id="distilled_one_stage", marks=SINGLE_CARD_MARKS),
        pytest.param(_server("LTX2DistilledTwoStagePipeline"), 8, id="distilled_two_stage", marks=SINGLE_CARD_MARKS),
    ]


@pytest.mark.parametrize(("omni_server", "num_inference_steps"), _cases(), indirect=["omni_server"])
def test_ltx25_pipeline_entries(
    omni_server: OmniServer,
    num_inference_steps: int,
    openai_client: OpenAIClientHandler,
    subtests,
) -> None:
    """Generate T2V and first-frame I2V through the synchronous video API."""
    for task in ("t2v", "i2v"):
        with subtests.test(task=task):
            request_config = {
                "model": omni_server.model,
                "form_data": {
                    "model": omni_server.model,
                    "prompt": PROMPT,
                    "height": 128,
                    "width": 128,
                    "num_frames": 9,
                    "fps": 24,
                    "num_inference_steps": num_inference_steps,
                    "seed": 42,
                },
            }
            if task == "i2v":
                request_config["image_reference"] = (
                    f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"
                )

            openai_client.send_video_diffusion_request(request_config)
