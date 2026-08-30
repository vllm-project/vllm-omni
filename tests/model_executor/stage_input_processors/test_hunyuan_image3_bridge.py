# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HunyuanImage3 stage input processor."""

import builtins
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
    HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS,
)
from vllm_omni.model_executor.stage_input_processors.hunyuan_image3 import (
    _extract_ratio_index,
    _truncate_at_cot_end,
    ar2diffusion,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _source_output(
    token_ids: list[int],
    text: str = "",
    *,
    logprobs: list | None = None,
    prompt_token_ids: list[int] | None = None,
):
    return SimpleNamespace(
        outputs=[
            SimpleNamespace(
                token_ids=token_ids,
                cumulative_token_ids=token_ids,
                text=text,
                logprobs=logprobs,
            )
        ],
        multimodal_output=None,
        prompt_token_ids=prompt_token_ids,
    )


def test_extract_ratio_index_uses_fixed_special_token_ids():
    ratio_33 = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_33>"]
    ratio_36 = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_36>"]

    assert _extract_ratio_index([1, ratio_33, 2]) == 33
    assert _extract_ratio_index([1, ratio_33, 2, ratio_36]) == 36


def test_truncate_at_cot_end_strips_tail_after_recaption_marker():
    text = _truncate_at_cot_end("body text</recaption><answer><boi><img_size_1024><img_ratio_0>")
    assert text == "body text</recaption>"


def test_ar2diffusion_returns_one_request_payload_for_request_level_batching():
    result = ar2diffusion(
        [_source_output([100], text="thought")],
        prompt={"prompt": "edit"},
    )

    assert isinstance(result, dict)
    assert result["prompt"] == "edit"


def test_ar2diffusion_uses_parent_output_when_companions_are_present():
    result = ar2diffusion(
        [
            _source_output([100], text="parent thought"),
            _source_output([200], text="companion thought"),
        ],
        prompt={"prompt": "edit"},
    )

    assert result is not None
    assert result["extra"]["ar_generated_text"] == "parent thought"


def test_ar2diffusion_exposes_ar_rollout_metadata():
    """#6768: sampled token ids / per-token log-probs / prompt ids must be
    carried alongside ar_generated_text instead of being dropped."""
    logprobs = [
        {101: SimpleNamespace(logprob=-0.5), 999: SimpleNamespace(logprob=-8.0)},
        {102: SimpleNamespace(logprob=-1.25)},
    ]
    result = ar2diffusion(
        [_source_output([101, 102], text="thought", logprobs=logprobs, prompt_token_ids=[7, 8, 9])],
        prompt={"prompt": "edit"},
    )

    assert result is not None
    extra = result["extra"]
    assert extra["ar_generated_text"] == "thought"
    assert extra["ar_generated_token_ids"] == [101, 102]
    assert extra["ar_generated_logprobs"] == [-0.5, -1.25]
    assert extra["ar_prompt_token_ids"] == [7, 8, 9]


def test_ar2diffusion_omits_logprobs_when_not_requested():
    result = ar2diffusion(
        [_source_output([100], text="thought", prompt_token_ids=[7])],
        prompt={"prompt": "edit"},
    )

    assert result is not None
    assert "ar_generated_logprobs" not in result["extra"]
    assert result["extra"]["ar_generated_token_ids"] == [100]
    assert result["extra"]["ar_prompt_token_ids"] == [7]


def test_ar2diffusion_returns_none_without_parent_output():
    assert ar2diffusion([], prompt={"prompt": "edit"}) is None


def test_ar2diffusion_applies_ratio_and_truncates_tail_without_tokenizer(monkeypatch: pytest.MonkeyPatch):
    real_import = builtins.__import__

    def _block_transformers_import(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise AssertionError("ar2diffusion must not import transformers on the bridge path")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_transformers_import)

    end_recaption = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["</recaption>"]
    answer = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<answer>"]
    boi = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<boi>"]
    size = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_size_1024>"]
    ratio_0 = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_0>"]
    token_ids = [100, 101, end_recaption, answer, boi, size, ratio_0]

    result = ar2diffusion(
        [_source_output(token_ids, text="decoded without special tokens")],
        prompt=[{"prompt": "edit", "height": 64, "width": 64}],
    )

    assert result is not None
    assert (result["height"], result["width"]) == (512, 2048)
    assert result["extra"]["ar_generated_text"] == "decoded without special tokens"
    assert "ar_token_ids" not in result["extra"]


def test_ar2diffusion_forwards_custom_system_prompt_body():
    end_think = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["</think>"]
    marker = "CUSTOM_SYSTEM_BODY"

    result = ar2diffusion(
        [_source_output([100, end_think], text="thought</think>")],
        prompt=[
            {
                "prompt": "edit",
                "use_system_prompt": "custom",
                "system_prompt": marker,
            }
        ],
    )

    assert result is not None
    assert result["use_system_prompt"] == "custom"
    assert result["system_prompt"] == marker
