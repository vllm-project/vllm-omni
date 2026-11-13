# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Router replay preserves exact values and invalidates moved parameter storage."""

import pytest
import torch
from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode, VllmConfig, set_current_vllm_config

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.zonos2.configuration_zonos2 import Zonos2Config
from vllm_omni.model_executor.models.zonos2.zonos2_talker import Zonos2SonicRouter

pytestmark = [pytest.mark.tts, pytest.mark.core_model]


@hardware_test(res={"cuda": ["L4", "H100", "B200"]}, num_cards=1)
def test_single_row_router_replay_exact_values_batch_fallback_and_parameter_move(monkeypatch):
    monkeypatch.setenv("VLLM_ZONOS2_ROUTER_REPLAY", "1")
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.NONE, custom_ops=["all"]
        )
    )
    generator = torch.Generator(device="cpu").manual_seed(42)
    with torch.inference_mode(), set_current_vllm_config(config):
        router = Zonos2SonicRouter(Zonos2Config(), has_prev_state=True).to("cuda:0", torch.bfloat16).eval()
        for step in range(12):
            x = torch.randn((1, 2048), generator=generator).to("cuda:0", torch.bfloat16)
            previous = torch.randn((1, 128), generator=generator).to("cuda:0", torch.bfloat16) if step % 2 else None
            expected = tuple(value.clone() for value in router._forward_eager(x, previous))
            actual = router(x, previous)
            for value, reference in zip(actual, expected):
                torch.testing.assert_close(value, reference, rtol=0, atol=0)
        assert len(router._captures) == 2
        batch = x.expand(4, -1).contiguous()
        expected = router._forward_eager(batch, None)
        actual = router(batch, None)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        # A new parameter allocation must never replay the old captured addresses.
        router.to("cpu")
        router.down_proj.bias.add_(0.25)
        router.to("cuda:0")
        expected = tuple(value.clone() for value in router._forward_eager(x, None))
        actual = router(x, None)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
