# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from pytest_mock import MockerFixture

from vllm_omni.diffusion.models.bagel.pipeline_bagel import BagelPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _NegativePromptPreparedError(Exception):
    pass


@pytest.mark.parametrize(
    ("prompt_negative", "extra_negative", "expected"),
    [
        ("prompt negative", "legacy negative", "prompt negative"),
        (None, "legacy negative", "legacy negative"),
    ],
    ids=["prompt-owner", "legacy-extra-args"],
)
def test_forward_reads_negative_prompt_from_prompt_with_legacy_fallback(
    mocker: MockerFixture,
    prompt_negative: str | None,
    extra_negative: str,
    expected: str,
) -> None:
    pipeline = object.__new__(BagelPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.od_config = mocker.Mock(dtype=torch.float32)
    pipeline.tokenizer = mocker.sentinel.tokenizer
    pipeline.new_token_ids = {}
    pipeline.language_model = mocker.Mock(vocab_size=10)
    pipeline.image_processor = None
    pipeline.vae = None

    bagel = mocker.MagicMock()
    bagel.max_latent_size = 32
    bagel.latent_downsample = 8
    bagel.config.llm_config.num_hidden_layers = 1
    prepared_prompts: list[str] = []

    def prepare_prompts(*, prompts, **_):
        prepared_prompts.append(prompts[0])
        if len(prepared_prompts) == 1:
            return {"packed_text_ids": torch.tensor([1])}, [1], [1]
        raise _NegativePromptPreparedError

    bagel.prepare_prompts.side_effect = prepare_prompts
    pipeline.bagel = bagel

    prompt = {
        "prompt": "positive",
        "negative_prompt": prompt_negative,
    }
    request = DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                prompt=prompt,
                sampling_params=OmniDiffusionSamplingParams(
                    extra_args={"negative_prompt": extra_negative},
                ),
                request_id="test",
            )
        ]
    )

    with pytest.raises(_NegativePromptPreparedError):
        pipeline.forward(request)

    assert prepared_prompts == ["positive", expected]
