# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MiniMax-H3 FastH3 four-step LoRA L3 online-serving case."""

from __future__ import annotations

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OpenAIClientHandler
from vllm_omni.platforms import current_omni_platform

from ._common import (
    FASTH3_HEIGHT,
    FASTH3_LORA,
    FASTH3_SUPPORTED,
    FASTH3_WIDTH,
    assert_h3_video,
    fasth3_form,
    fasth3_params,
    post_sync,
)

pytestmark = [pytest.mark.advanced_model, pytest.mark.diffusion]
H100_TWO_CARD_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)


@pytest.mark.skipif(not FASTH3_SUPPORTED, reason="requires FastH3 support from MiniMax-H3 PR #6714")
@pytest.mark.skipif(
    FASTH3_LORA is None,
    reason="set VLLM_TEST_MINIMAX_H3_FASTH3_LORA or populate the local HF cache",
)
@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            fasth3_params(FASTH3_LORA),
            id="minimax_h3_fasth3_lora_hsdp2_usp2_vpp2",
            marks=[
                *H100_TWO_CARD_MARKS,
                pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="requires CUDA"),
            ],
        ),
        pytest.param(
            fasth3_params(FASTH3_LORA, num_cards=4),
            id="minimax_h3_fasth3_lora_hsdp4_usp4_vpp4",
            marks=[
                *hardware_marks(res={"npu": "A3"}, num_cards=4),
                pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires NPU"),
            ],
        ),
    ],
    indirect=True,
)
def test_minimax_h3_fasth3_lora(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
) -> None:
    """Run the dense/data-free FastH3 adapter on two H100s or four A3 dies.

    The preview adapter is T2VA-only and must be fused through ``--lora-path``;
    it is not a request-switchable PEFT adapter. 1024x576 is intentionally
    below the 1344x768 H3 reference shape to stay inside an 80 GiB H100 after
    regional compile cache growth. A3 uses four dies because the resident
    two-die layout exhausts device memory while loading the text encoder.
    """
    video = post_sync(openai_client, fasth3_form(seed=3101))
    assert_h3_video(video, width=FASTH3_WIDTH, height=FASTH3_HEIGHT)
