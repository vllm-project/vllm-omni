# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.diffusion.lora.test_merged_weights import _layer
from tests.helpers.mark import hardware_marks
from vllm_omni.lora.merged_weights import MergedLinearWeights
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": ["H100", "B200"], "rocm": "MI325"}, num_cards=1),
]


@torch.inference_mode()
def test_captured_dense_forward_observes_repeated_merged_updates():
    device = current_omni_platform.get_torch_device(0)
    layer = _layer()
    layer.base_layer.to(device)
    layer.lora_a_stacked = tuple(t.to(device) for t in layer.lora_a_stacked)
    layer.lora_b_stacked = tuple(t.to(device) for t in layer.lora_b_stacked)
    x = torch.randn(8, 4, dtype=torch.bfloat16, device=device)
    output = torch.empty(8, 5, dtype=torch.bfloat16, device=device)

    def forward():
        output.copy_(layer.base_layer(x))

    for _ in range(3):
        forward()
    current_omni_platform.synchronize()
    graph = current_omni_platform.capture_graph(forward)
    assert graph is not None
    bank = MergedLinearWeights()
    pointer = layer.base_layer.weight.data_ptr()
    for scale in (0.5, 1.0, 2.0):
        layer._merged_lora_scales = (scale, scale)
        bank.apply(bank.prepare({"proj": layer}))
        current_omni_platform.replay_graph(graph, forward)
        torch.testing.assert_close(output, layer.base_layer(x), rtol=0, atol=0)
        assert layer.base_layer.weight.data_ptr() == pointer
    bank.close()
    current_omni_platform.replay_graph(graph, forward)
    torch.testing.assert_close(output, layer.base_layer(x), rtol=0, atol=0)
