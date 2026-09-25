# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.lance.lance_transformer import LanceBagel
from vllm_omni.diffusion.models.lance.pipeline_lance import LancePipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PROMPT_IDS = [1049, 4702, 99]
NEGATIVE_PROMPT_IDS = [7, 8, 9]


class StopAfterTextPreprocessError(Exception):
    pass


def test_prepare_prompts_keeps_ids_verbatim_and_expands_mrope_positions():
    bagel = object.__new__(LanceBagel)
    torch.nn.Module.__init__(bagel)

    gen_input, newlens, new_rope = bagel.prepare_prompts(
        curr_kvlens=[0],
        curr_rope=[0],
        prompts=[PROMPT_IDS],
        tokenizer=None,
        new_token_ids={"bos_token_id": 1, "eos_token_id": 2},
    )

    # ``LanceBagel`` wraps ``Bagel.prepare_prompts`` to expand scalar positions
    # into 3-axis mRoPE, and the pre-tokenized path has to go through it as well.
    assert gen_input["packed_text_ids"].tolist() == PROMPT_IDS
    assert gen_input["packed_text_position_ids"].shape == (3, len(PROMPT_IDS))
    assert newlens == [len(PROMPT_IDS)]
    assert new_rope == [len(PROMPT_IDS)]


def _t2v_pipeline(captured: list, stop_after: int) -> LancePipeline:
    def prepare_prompts(*, prompts, **_):
        captured.append(prompts[0])
        if len(captured) == stop_after:
            raise StopAfterTextPreprocessError
        return {"packed_text_ids": torch.tensor([1])}, [1], [1]

    pipeline = object.__new__(LancePipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.od_config = SimpleNamespace(dtype=torch.float32)
    # ``_forward_t2v`` builds ``tokenizer=self.tokenizer`` and
    # ``new_token_ids=self.new_token_ids`` before it reaches the stub below, so
    # both attributes are read even though the stub ignores them.
    pipeline.tokenizer = SimpleNamespace()
    pipeline.new_token_ids = {"bos_token_id": 1, "eos_token_id": 2}
    pipeline.bagel = SimpleNamespace(
        max_latent_size=64,
        latent_downsample=16,
        prepare_prompts=prepare_prompts,
        forward_cache_update_text=lambda *args, **kwargs: None,
        config=SimpleNamespace(
            vae_config=SimpleNamespace(downsample_temporal=4),
            llm_config=SimpleNamespace(num_hidden_layers=1),
        ),
    )
    return pipeline


def _t2v_request(prompt: dict) -> SimpleNamespace:
    return SimpleNamespace(
        prompts=[{"modalities": ["video"], **prompt}],
        sampling_params=SimpleNamespace(
            extra_args={"video_height": 64, "video_width": 64, "num_frames": 5},
            num_inference_steps=1,
            seed=None,
            height=None,
            width=None,
        ),
    )


def test_t2v_uses_pre_tokenized_prompt_ids():
    captured: list = []
    pipeline = _t2v_pipeline(captured, stop_after=1)

    with pytest.raises(StopAfterTextPreprocessError):
        pipeline._forward_t2v(_t2v_request({"prompt_ids": PROMPT_IDS}))

    assert captured == [PROMPT_IDS]


def test_t2v_prefers_prompt_ids_over_the_text():
    captured: list = []
    pipeline = _t2v_pipeline(captured, stop_after=1)

    with pytest.raises(StopAfterTextPreprocessError):
        pipeline._forward_t2v(_t2v_request({"prompt": "a running fox", "prompt_ids": PROMPT_IDS}))

    assert captured == [PROMPT_IDS]


def test_t2v_uses_negative_pre_tokenized_prompt_ids():
    captured: list = []
    pipeline = _t2v_pipeline(captured, stop_after=2)

    with pytest.raises(StopAfterTextPreprocessError):
        pipeline._forward_t2v(_t2v_request({"prompt_ids": PROMPT_IDS, "negative_prompt_ids": NEGATIVE_PROMPT_IDS}))

    assert captured == [PROMPT_IDS, NEGATIVE_PROMPT_IDS]


def test_t2v_still_uses_the_text_without_ids():
    captured: list = []
    pipeline = _t2v_pipeline(captured, stop_after=2)

    with pytest.raises(StopAfterTextPreprocessError):
        pipeline._forward_t2v(_t2v_request({"prompt": "a running fox"}))

    assert captured == ["a running fox", ""]
