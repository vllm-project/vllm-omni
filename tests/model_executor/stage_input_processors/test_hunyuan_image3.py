# SPDX-License-Identifier: Apache-2.0
"""Regression tests for HunyuanImage3 stage input bridging."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from vllm_omni.model_executor.stage_input_processors.hunyuan_image3 import ar2diffusion

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_ar2diffusion_forwards_img2img_source_image():
    image = Image.new("RGB", (24, 24), color="green")
    ar_output = SimpleNamespace(
        outputs=[
            SimpleNamespace(
                cumulative_token_ids=[1, 2, 3],
                text="latent",
            )
        ],
    )
    diffusion_input = ar2diffusion(
        source_outputs=[ar_output],
        prompt={
            "prompt": "将背景更改为森林",
            "multi_modal_data": {"img2img": image},
            "seed": 42,
            "use_system_prompt": "en_unified",
        },
        requires_multimodal_data=True,
    )

    assert diffusion_input["prompt"] == "将背景更改为森林"
    assert diffusion_input["multi_modal_data"]["image"] is image
    assert diffusion_input["seed"] == 42
    assert diffusion_input["use_system_prompt"] == "en_unified"
    assert diffusion_input["extra"]["ar_generated_text"] == "latent"
