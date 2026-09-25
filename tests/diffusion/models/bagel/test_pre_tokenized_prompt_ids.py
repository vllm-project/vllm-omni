# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from pytest_mock import MockerFixture

from vllm_omni.diffusion.models.bagel.bagel_transformer import Bagel
from vllm_omni.diffusion.models.bagel.pipeline_bagel import BagelPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

POSITIVE_IDS = [1049, 4702, 99]
NEGATIVE_IDS = [7, 8, 9]


class _PreparedPromptsCapturedError(Exception):
    pass


def _build_pipeline(mocker: MockerFixture, prepared_prompts: list, stop_after: int) -> BagelPipeline:
    pipeline = object.__new__(BagelPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.od_config = mocker.Mock(dtype=torch.float32)
    pipeline.tokenizer = mocker.sentinel.tokenizer
    pipeline.new_token_ids = {"bos_token_id": 1, "eos_token_id": 2}
    pipeline.language_model = mocker.Mock(vocab_size=1000)
    pipeline.image_processor = None
    pipeline.vae = None

    bagel = mocker.MagicMock()
    bagel.max_latent_size = 32
    bagel.latent_downsample = 8
    bagel.config.llm_config.num_hidden_layers = 1

    def prepare_prompts(*, prompts, **_):
        prepared_prompts.append(prompts[0])
        if len(prepared_prompts) == stop_after:
            raise _PreparedPromptsCapturedError
        return {"packed_text_ids": torch.tensor([1])}, [1], [1]

    bagel.prepare_prompts.side_effect = prepare_prompts
    pipeline.bagel = bagel
    return pipeline


def _forward_text_to_image(pipeline: BagelPipeline, prompt: dict) -> None:
    request = DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                prompt=prompt,
                sampling_params=OmniDiffusionSamplingParams(),
                request_id="test",
            )
        ]
    )
    with pytest.raises(_PreparedPromptsCapturedError):
        pipeline.forward(request)


def test_forward_uses_prompt_ids_instead_of_text(mocker: MockerFixture):
    prepared_prompts: list = []
    pipeline = _build_pipeline(mocker, prepared_prompts, stop_after=2)

    _forward_text_to_image(pipeline, {"prompt_ids": POSITIVE_IDS})

    # Both the generation context and the cfg_img context are text-conditioned.
    assert prepared_prompts == [POSITIVE_IDS, POSITIVE_IDS]


def test_forward_prefers_prompt_ids_over_the_text(mocker: MockerFixture):
    prepared_prompts: list = []
    pipeline = _build_pipeline(mocker, prepared_prompts, stop_after=2)

    _forward_text_to_image(pipeline, {"prompt": "a cat", "prompt_ids": POSITIVE_IDS})

    assert prepared_prompts[0] == POSITIVE_IDS


def test_forward_uses_negative_prompt_ids(mocker: MockerFixture):
    prepared_prompts: list = []
    pipeline = _build_pipeline(mocker, prepared_prompts, stop_after=3)

    _forward_text_to_image(
        pipeline,
        {"prompt_ids": POSITIVE_IDS, "negative_prompt_ids": NEGATIVE_IDS},
    )

    assert prepared_prompts == [POSITIVE_IDS, NEGATIVE_IDS, POSITIVE_IDS]


def test_forward_still_uses_text_without_ids(mocker: MockerFixture):
    prepared_prompts: list = []
    pipeline = _build_pipeline(mocker, prepared_prompts, stop_after=2)

    _forward_text_to_image(pipeline, {"prompt": "a cat"})

    assert prepared_prompts == ["a cat", "a cat"]


def test_prepare_prompts_packs_ids_verbatim(mocker: MockerFixture):
    bagel = object.__new__(Bagel)
    torch.nn.Module.__init__(bagel)
    tokenizer = mocker.Mock()
    tokenizer.encode.return_value = [10, 11]
    new_token_ids = {"bos_token_id": 1, "eos_token_id": 2}

    packaged = bagel.prepare_prompts(
        curr_kvlens=[0],
        curr_rope=[0],
        prompts=[[1, 10, 11, 2]],
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )

    from_ids, newlens, new_rope = packaged
    # The caller owns the sequence: no bos/eos is wrapped around it.
    assert from_ids["packed_text_ids"].tolist() == [1, 10, 11, 2]
    assert from_ids["text_token_lens"].tolist() == [4]
    assert from_ids["packed_text_position_ids"].tolist() == [0, 1, 2, 3]
    assert newlens == [4]
    assert new_rope == [4]


def test_prepare_prompts_ids_match_the_equivalent_text(mocker: MockerFixture):
    bagel = object.__new__(Bagel)
    torch.nn.Module.__init__(bagel)
    tokenizer = mocker.Mock()
    tokenizer.encode.return_value = [10, 11]
    new_token_ids = {"bos_token_id": 1, "eos_token_id": 2}

    from_text = bagel.prepare_prompts(
        curr_kvlens=[0],
        curr_rope=[0],
        prompts=["hi"],
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    # ``hi`` tokenizes to [10, 11], so the equivalent pre-tokenized prompt is
    # bos + those ids + eos, which is what the text path builds internally.
    from_ids = bagel.prepare_prompts(
        curr_kvlens=[0],
        curr_rope=[0],
        prompts=[[1, 10, 11, 2]],
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )

    assert from_ids[0]["packed_text_ids"].tolist() == from_text[0]["packed_text_ids"].tolist()
    assert from_ids[0]["text_token_lens"].tolist() == from_text[0]["text_token_lens"].tolist()
    assert from_ids[0]["packed_text_position_ids"].tolist() == from_text[0]["packed_text_position_ids"].tolist()
    assert from_ids[1] == from_text[1]
    assert from_ids[2] == from_text[2]
    # The text path is the one that never tokenizes twice.
    tokenizer.encode.assert_called_once_with("hi", add_special_tokens=False)
