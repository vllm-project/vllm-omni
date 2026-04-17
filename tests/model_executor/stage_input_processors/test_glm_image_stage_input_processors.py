# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.glm_image import (
    _parse_generated_tokens,
    _upsample_token_ids,
    ar2diffusion,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _ar_output(
    token_ids: list[int],
    *,
    multimodal_output: dict | None = None,
):
    return SimpleNamespace(
        outputs=[SimpleNamespace(token_ids=token_ids)],
        multimodal_output=multimodal_output,
    )


def test_upsample_token_ids_matches_nearest_neighbor_layout():
    token_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)

    upsampled = _upsample_token_ids(token_ids, token_h=2, token_w=2)

    expected = torch.tensor(
        [
            1,
            1,
            2,
            2,
            1,
            1,
            2,
            2,
            3,
            3,
            4,
            4,
            3,
            3,
            4,
            4,
        ],
        dtype=torch.long,
    )
    torch.testing.assert_close(upsampled, expected)


def test_ar2diffusion_builds_upsampled_prior_tokens_for_t2i():
    stage_list = [
        SimpleNamespace(
            engine_outputs=[
                _ar_output([10, 11, 12, 13, 14], multimodal_output=None),
            ]
        )
    ]

    outputs = ar2diffusion(
        stage_list=stage_list,
        engine_input_source=[0],
        prompt=[{"prompt": "hello", "height": 64, "width": 64}],
    )

    assert len(outputs) == 1
    assert outputs[0]["prompt"] == "hello"
    assert outputs[0]["height"] == 64
    assert outputs[0]["width"] == 64
    torch.testing.assert_close(
        outputs[0]["extra"]["prior_token_ids"],
        torch.tensor([11, 11, 12, 12, 11, 11, 12, 12, 13, 13, 14, 14, 13, 13, 14, 14], dtype=torch.long),
    )


def test_ar2diffusion_normalizes_serialized_prior_token_image_ids():
    stage_list = [
        SimpleNamespace(
            engine_outputs=[
                _ar_output(
                    [21, 22, 23, 24],
                    multimodal_output={"prior_token_image_ids": [[101, 102, 103, 104]]},
                ),
            ]
        )
    ]

    outputs = ar2diffusion(
        stage_list=stage_list,
        engine_input_source=[0],
        prompt=[
            {
                "prompt": "edit",
                "height": 64,
                "width": 64,
                "multi_modal_data": {"image": object()},
            }
        ],
        requires_multimodal_data=True,
    )

    prior_image_ids = outputs[0]["extra"]["prior_token_image_ids"]
    assert isinstance(prior_image_ids, list)
    assert len(prior_image_ids) == 1
    assert isinstance(prior_image_ids[0], torch.Tensor)
    torch.testing.assert_close(
        prior_image_ids[0],
        torch.tensor([101, 102, 103, 104], dtype=torch.long),
    )
    assert "pil_image" in outputs[0]


def test_ar2diffusion_uses_i2i_large_tokens_without_preview_prefix():
    stage_list = [
        SimpleNamespace(
            engine_outputs=[
                _ar_output(
                    [31, 32, 33, 34, 16385],
                    multimodal_output={"prior_token_image_ids": [torch.tensor([201, 202, 203, 204], dtype=torch.long)]},
                ),
            ]
        )
    ]

    outputs = ar2diffusion(
        stage_list=stage_list,
        engine_input_source=[0],
        prompt=[{"prompt": "edit", "height": 64, "width": 64}],
    )

    torch.testing.assert_close(
        outputs[0]["extra"]["prior_token_ids"],
        torch.tensor([31, 31, 32, 32, 31, 31, 32, 32, 33, 33, 34, 34, 33, 33, 34, 34], dtype=torch.long),
    )
    torch.testing.assert_close(
        outputs[0]["extra"]["prior_token_image_ids"][0],
        torch.tensor([201, 202, 203, 204], dtype=torch.long),
    )


def test_ar2diffusion_reads_prior_token_image_ids_from_completion_output_fallback():
    output = SimpleNamespace(
        token_ids=[41, 42, 43, 44],
        multimodal_output={"prior_token_image_ids": [torch.tensor([301, 302, 303, 304], dtype=torch.long)]},
    )
    stage_list = [SimpleNamespace(engine_outputs=[SimpleNamespace(outputs=[output], multimodal_output=None)])]

    outputs = ar2diffusion(
        stage_list=stage_list,
        engine_input_source=[0],
        prompt=[{"prompt": "fallback", "height": 64, "width": 64}],
    )

    torch.testing.assert_close(
        outputs[0]["extra"]["prior_token_image_ids"][0],
        torch.tensor([301, 302, 303, 304], dtype=torch.long),
    )


def test_parse_generated_tokens_adjusts_grid_for_truncated_output():
    prior_token_ids, pixel_h, pixel_w = _parse_generated_tokens(
        [51, 52, 53, 54],
        height=128,
        width=128,
    )

    assert pixel_h == 64
    assert pixel_w == 64
    torch.testing.assert_close(
        prior_token_ids,
        torch.tensor([51, 51, 52, 52, 51, 51, 52, 52, 53, 53, 54, 54, 53, 53, 54, 54], dtype=torch.long),
    )
