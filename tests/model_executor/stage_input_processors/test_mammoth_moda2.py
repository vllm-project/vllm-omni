# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.mammoth_moda2 import ar2diffusion

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _source_output(*, include_latent: bool = True):
    multimodal_output = (
        {"latent": torch.arange(32, dtype=torch.float32).reshape(4, 8)}
        if include_latent
        else {}
    )
    completion = SimpleNamespace(
        cumulative_token_ids=[100, 101, 102],
        multimodal_output=multimodal_output,
    )
    return SimpleNamespace(
        request_id="req-7",
        prompt_token_ids=[10, 11],
        outputs=[completion],
    )


def test_ar2diffusion_builds_one_prompt_with_raw_ar_conditions() -> None:
    result = ar2diffusion(
        [_source_output()],
        {"prompt": "a cat", "mm_processor_kwargs": {"target_h": 512, "target_w": 768}},
    )
    assert not isinstance(result, list)
    assert result["prompt"] == ""
    assert result["height"] == 512
    assert result["width"] == 768
    info = result["additional_information"]
    assert info["full_token_ids"] == [10, 11, 100, 101]
    assert info["answer_start_index"] == 2
    torch.testing.assert_close(
        info["full_hidden_states"],
        torch.arange(32, dtype=torch.float32).reshape(4, 8),
    )
    assert info["full_hidden_states"].is_contiguous()


def test_ar2diffusion_uses_prompt_dimension_fallbacks() -> None:
    result = ar2diffusion(
        [_source_output()],
        {"additional_information": {"image_height": [256], "image_width": [384]}},
    )
    assert (result["height"], result["width"]) == (256, 384)


def test_ar2diffusion_unwraps_the_orchestrator_prompt_list() -> None:
    result = ar2diffusion(
        [_source_output()],
        [{"mm_processor_kwargs": {"target_h": 640, "target_w": 960}}],
    )
    assert (result["height"], result["width"]) == (640, 960)


def test_ar2diffusion_rejects_multiple_source_requests() -> None:
    with pytest.raises(ValueError, match="exactly one AR output"):
        ar2diffusion([_source_output(), _source_output()], {})


def test_ar2diffusion_reports_missing_latent_with_request_id() -> None:
    with pytest.raises(ValueError, match="req-7"):
        ar2diffusion([_source_output(include_latent=False)], {})


def test_ar2diffusion_rejects_hidden_state_length_mismatch() -> None:
    source = _source_output()
    source.outputs[0].multimodal_output["latent"] = torch.zeros(3, 8)
    with pytest.raises(ValueError, match="Hidden states length mismatch"):
        ar2diffusion([source], {})
