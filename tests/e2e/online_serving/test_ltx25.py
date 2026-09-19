# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

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

pytestmark = [pytest.mark.diffusion, pytest.mark.slow]
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
                    "height": 256,
                    "width": 256,
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


# Distributed VAE patch parallelism needs a multi-rank DiT group to fan out over, so this
# case runs on two cards instead of the single-card smokes above.
VAE_PARALLEL_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)


def _vae_parallel_server() -> OmniServerParams:
    return OmniServerParams(
        model=MODEL,
        server_args=[
            *(["--revision", MODEL_REVISION] if MODEL_REVISION else []),
            "--model-class-name",
            "LTX2Pipeline",
            "--enforce-eager",
            "--diffusion-attention-backend",
            "CUDNN_ATTN",
            # HSDP shards the transformer to give a 2-rank DiT group; the VAE patch-parallel
            # executor fans its tiles out over that same group.
            "--use-hsdp",
            "--hsdp-shard-size",
            "2",
            "--vae-patch-parallel-size",
            "2",
            "--vae-use-tiling",
        ],
    )


@pytest.mark.parametrize(
    "omni_server",
    [pytest.param(_vae_parallel_server(), id="i2v_vae_patch_parallel", marks=VAE_PARALLEL_MARKS)],
    indirect=["omni_server"],
)
def test_ltx25_i2v_vae_patch_parallel(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
) -> None:
    """I2V through the distributed VAE tiling parallel encode and decode path.

    The full one-stage pipeline encodes the reference image at the full output resolution, so
    576x576 (above the VAE's 512 tile threshold) splits into a 2x2 tile grid and drives
    tiled_encode/tiled_decode across the DiT group. The single-card smokes above run at
    256x256, which stays under the threshold and never reaches the tiled path.
    """
    request_config = {
        "model": omni_server.model,
        "form_data": {
            "model": omni_server.model,
            "prompt": PROMPT,
            "height": 576,
            "width": 576,
            "num_frames": 9,
            "fps": 24,
            "num_inference_steps": 8,
            "seed": 42,
        },
        "image_reference": f"data:image/jpeg;base64,{generate_synthetic_image(576, 576)['base64']}",
    }

    openai_client.send_video_diffusion_request(request_config)
