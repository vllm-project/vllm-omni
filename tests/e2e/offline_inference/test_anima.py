# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline e2e smoke test for native Anima single-file checkpoints."""

import os

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner, OmniRunnerHandler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

CHECKPOINT_ENV = "ANIMA_CHECKPOINT_PATH"
COMPONENTS_ENV = "ANIMA_COMPONENTS_PATH"
CHECKPOINT_REPO = "circlestone-labs/Anima"
CHECKPOINT_FILENAME = "split_files/diffusion_models/anima-base-v1.0.safetensors"
COMPONENTS_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"
PROMPT = "A cinematic close-up of a glass teapot on a wooden table."


def _resolve_anima_assets() -> tuple[str, str]:
    checkpoint = os.environ.get(CHECKPOINT_ENV)
    components = os.environ.get(COMPONENTS_ENV)
    if checkpoint and components:
        return checkpoint, components

    from huggingface_hub import hf_hub_download, snapshot_download

    if checkpoint is None:
        checkpoint = hf_hub_download(
            repo_id=CHECKPOINT_REPO,
            filename=CHECKPOINT_FILENAME,
        )
    if components is None:
        components = snapshot_download(repo_id=COMPONENTS_REPO)
    return checkpoint, components


@pytest.mark.full_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("guidance_scale", [1.0, 4.0])
def test_anima_text_to_image(guidance_scale):
    checkpoint, components = _resolve_anima_assets()
    request_config = {
        "model": checkpoint,
        "prompt": PROMPT,
        "sampling_params": OmniDiffusionSamplingParams(
            height=256,
            width=256,
            num_inference_steps=1,
            guidance_scale=guidance_scale,
            seed=42,
        ),
    }
    with OmniRunner(
        checkpoint,
        seed=42,
        model_class_name="AnimaPipeline",
        custom_pipeline_args={"components_path": components},
    ) as runner:
        OmniRunnerHandler(runner).send_diffusion_request(request_config)
