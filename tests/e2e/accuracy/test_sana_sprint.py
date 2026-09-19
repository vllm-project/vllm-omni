# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from diffusers import SanaSprintPipeline as ReferencePipeline
from diffusers import SanaTransformer2DModel, SCMScheduler

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.sana_sprint.pipeline_sana_sprint import SanaSprintPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.advanced_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": "L4"})]
MODEL = "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers"
REVISION = "aa76e7f4f4928f378716b6716a2130fba3caf5b1"
PROMPT = "A small red panda sitting on a mossy rock in a misty bamboo forest, soft morning light."


@pytest.fixture(scope="module")
def pipelines():
    native = SanaSprintPipeline(
        od_config=OmniDiffusionConfig(
            model=MODEL,
            revision=REVISION,
            dtype=torch.bfloat16,
            output_type="latent",
        )
    )
    transformer = SanaTransformer2DModel.from_pretrained(
        MODEL,
        revision=REVISION,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    ).to(native.device)
    native.transformer.load_weights(iter(transformer.state_dict().items()))
    native.transformer.to(device=native.device, dtype=torch.bfloat16)
    native.eval()
    reference = ReferencePipeline(
        tokenizer=native.tokenizer,
        text_encoder=native.text_encoder,
        vae=native.vae,
        transformer=transformer,
        scheduler=SCMScheduler.from_config(native.scheduler.config),
    )
    return native, reference


@pytest.mark.parametrize("steps", [1, 2, 4])
@torch.inference_mode()
def test_full_checkpoint_latent_parity(pipelines, steps):
    native, reference = pipelines
    expected = reference(
        PROMPT,
        num_inference_steps=steps,
        intermediate_timesteps=1.3 if steps == 2 else None,
        generator=torch.Generator("cpu").manual_seed(42),
        output_type="latent",
    ).images
    request = OmniDiffusionRequest(
        prompt=PROMPT,
        request_id=f"sana-sprint-parity-{steps}",
        sampling_params=OmniDiffusionSamplingParams(
            num_inference_steps=steps,
            generator=torch.Generator("cpu").manual_seed(42),
        ),
    )
    actual = native(DiffusionRequestBatch([request])).output
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
