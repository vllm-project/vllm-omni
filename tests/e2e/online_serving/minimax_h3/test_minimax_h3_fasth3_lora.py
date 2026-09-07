# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MiniMax-H3 FastH3 four-step LoRA L3 online-serving case."""

from __future__ import annotations

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OpenAIClientHandler

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

pytestmark = [pytest.mark.core_model, pytest.mark.advanced_model, pytest.mark.diffusion, pytest.mark.slow]
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
            marks=H100_TWO_CARD_MARKS,
        )
    ],
    indirect=True,
)
def test_minimax_h3_fasth3_lora_hsdp2_usp2_vpp2(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
) -> None:
    """Run the dense/data-free FastH3 adapter on two H100-class cards.

    The preview adapter is T2VA-only and must be fused through ``--lora-path``;
    it is not a request-switchable PEFT adapter. 1024x576 is intentionally
    below the 1344x768 H3 reference shape to stay inside an 80 GiB H100 after
    regional compile cache growth.
    """
    video = post_sync(openai_client, fasth3_form(seed=3101))
    assert_h3_video(video, width=FASTH3_WIDTH, height=FASTH3_HEIGHT)
