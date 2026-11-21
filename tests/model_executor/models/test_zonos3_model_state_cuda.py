# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Batched conditioning and decode feedback must not synchronize the CUDA stream."""

from typing import Any

import pytest
import torch

from tests.helpers.mark import hardware_marks
from tests.model_executor.models.test_zonos3_model_state import add, batch, make_owner
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}),
    pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="Requires pinned CUDA transfers"),
]


@torch.inference_mode()
def test_batched_prefill_and_decode_have_no_stream_sync():
    owner = make_owner(4, "cuda", torch.bfloat16)
    state, _, _ = owner
    add(owner, "a", 0, 6)
    add(owner, "b", 1, 5)
    state.apply_staged_writes()
    expected = [
        state.build_prompt(name, state.intermediate_buffer.buffers[i]).embeddings for i, name in enumerate(("a", "b"))
    ]

    def preprocess(data):
        data.idx_mapping = data.idx_mapping.cuda()
        inputs: dict[str, Any] = {}
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as trace:
            with torch.profiler.record_function("zonos3_test_preprocess"):
                state.run_preprocess(data, inputs)
        for event in trace.events():
            if "Synchronize" not in event.name:
                continue
            parent = event.cpu_parent
            while parent is not None:
                assert parent.name != "zonos3_test_preprocess", event.name
                parent = parent.cpu_parent
        return inputs

    first = preprocess(batch(owner, [("a", 0, 2, 0, True), ("b", 1, 5, 0, True)]))
    torch.testing.assert_close(first["inputs_embeds"][:2], expected[0][:2], atol=0.01, rtol=0.01)
    torch.testing.assert_close(first["inputs_embeds"][2:7], expected[1], atol=0.01, rtol=0.01)
    assert state.sampling_state.has_audio_prefix[:2].tolist() == [True, True]
    state.next_embeddings.normal_()
    feedback = state.next_embeddings[[1, 0]].clone()
    second = preprocess(batch(owner, [("b", 1, 1, 5, False), ("a", 0, 1, 6, False)], padded=4))
    torch.testing.assert_close(second["inputs_embeds"][:2], feedback, atol=0, rtol=0)
    assert not second["inputs_embeds"][2:].any()
    assert second["zonos3_batch"].decode_slots.tolist() == [1, 0, 4, 5]
