# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Downstream GDN op: FULL replay must retain recurrent state across windows."""

import pytest
import torch
from vllm.forward_context import create_forward_context, override_forward_context

from tests.helpers.mark import hardware_marks
from tests.model_executor.models.test_zonos3_modeling import valid_runtime
from vllm_omni.model_executor.models.zonos3.modeling_zonos3_mini import (
    Zonos3BackboneBatch,
    Zonos3ForCausalLM,
    Zonos3GDN,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}),
    pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="Requires FLA CUDA graphs"),
]


@pytest.mark.parametrize("batch_size", [1, 4])
@torch.inference_mode()
def test_gdn_custom_op_full_replay_keeps_state(batch_size, monkeypatch):
    monkeypatch.setenv("FLA_TILELANG", "0")
    torch.manual_seed(106)
    layer = Zonos3GDN(16).cuda().to(torch.bfloat16).eval()
    layer.prepare_cuda([batch_size])
    captured, reference = layer.allocate_state(batch_size), layer.allocate_state(batch_size)
    slots = torch.arange(batch_size, device="cuda")
    batch = Zonos3BackboneBatch({}, None, slots, slots, dummy_slots=tuple(range(batch_size)))
    layer._runtime_batch, layer._runtime_state = batch, captured
    config = valid_runtime()
    config.compilation_config.static_forward_context["test_zonos3_gdn"] = layer
    x = torch.randn(batch_size, 16, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)

    def execute():
        torch.ops.vllm.zonos3_gdn_core(
            x,
            captured.conv,
            captured.recurrent,
            captured.window,
            captured.window_length,
            captured.total_tokens,
            out,
            "test_zonos3_gdn",
        )

    with override_forward_context(create_forward_context(None, config)):
        for _ in range(3):
            execute()
        current_omni_platform.synchronize()
        graph = current_omni_platform.capture_graph(execute)
    assert graph is not None
    captured.reset(list(range(batch_size)))
    for step in range(66):
        x.normal_()
        slots.copy_(
            torch.arange(batch_size, device="cuda").flip(0) if step % 2 else torch.arange(batch_size, device="cuda")
        )
        expected = Zonos3ForCausalLM._gdn_forward(layer, x, batch, reference)
        graph.replay()
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
    for name in ("conv", "recurrent", "window", "window_length", "total_tokens"):
        torch.testing.assert_close(getattr(captured, name), getattr(reference, name), atol=0, rtol=0)
    assert captured.total_tokens.tolist() == [66] * batch_size
    # Compilation may functionalize mutable operands. The custom op must
    # update the tensors it receives, not a separately bound original pool.
    redirected = layer.allocate_state(batch_size)
    batch.dummy_slots = ()
    with override_forward_context(create_forward_context(None, config)):
        torch.ops.vllm.zonos3_gdn_core(
            x,
            redirected.conv,
            redirected.recurrent,
            redirected.window,
            redirected.window_length,
            redirected.total_tokens,
            out,
            "test_zonos3_gdn",
        )
    assert redirected.total_tokens.tolist() == [1] * batch_size
    assert captured.total_tokens.tolist() == [66] * batch_size
