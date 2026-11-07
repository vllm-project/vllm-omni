# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project


import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.phonon.cuda_graph_wrapper import FlowSampleRunner

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("frames", [1, 16, 51, 52, 270])
def test_direct_ring_attention_matches_cpu_reference(frames):
    from vllm_omni.model_executor.models.phonon.mimi_attention import ring_attention_chunk

    torch.manual_seed(89)
    slots = torch.tensor([2, 0])
    positions = torch.tensor([247, 500])
    keys = torch.randn(3, 300, 2, 64)
    values = torch.randn_like(keys)
    gpu_keys, gpu_values = keys.cuda(), values.cuda()
    for _ in range(2):
        query, key, value = [torch.randn(2, frames, 2, 64) for _ in range(3)]
        expected = ring_attention_chunk(query, key, value, keys, values, slots, positions, window=250)
        actual = ring_attention_chunk(
            query.cuda(), key.cuda(), value.cuda(), gpu_keys, gpu_values, slots.cuda(), positions.cuda(), window=250
        )
        torch.testing.assert_close(actual.cpu(), expected, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(gpu_keys.cpu(), keys)
        torch.testing.assert_close(gpu_values.cpu(), values)
        positions += frames


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flow_postprocess_graph_restores_capture_state_and_reuses_slots():
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.phonon.model_state import PhononModelState

    class TestFlow(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_linear = torch.nn.Linear(4, 8, bias=False)
            self.register_buffer("emb_std", torch.arange(1, 5, dtype=torch.float32))
            self.register_buffer("emb_mean", torch.ones(4))

        def sample(self, hidden, noise):
            return hidden[:, :4] + noise, hidden[:, -1:]

    model = SimpleNamespace(flow_lm=TestFlow().cuda(), phonon_config={"eos_threshold": 0.5})
    condition = torch.randn(8, device="cuda")

    def state():
        owner = PhononModelState.__new__(PhononModelState)
        owner.model = model
        owner.condition = condition
        owner.remaining_tail = torch.tensor([7, -1], device="cuda")
        owner.tail_frames = torch.tensor([3, 1], device="cuda")
        owner.active = torch.ones(2, dtype=torch.bool, device="cuda")
        owner.next_embeddings = torch.zeros(2, 8, device="cuda")
        return owner

    eager, captured = state(), state()
    runner = FlowSampleRunner(
        model.flow_lm,
        postprocess=captured._flow_postprocess,
        state_buffers=(captured.remaining_tail, captured.active, captured.next_embeddings),
    )
    with torch.inference_mode():
        snapshot = None
        for step, slot in enumerate((1, 1, 1, 0, 1)):
            if step == 4:
                for owner in (eager, captured):
                    owner.remaining_tail[1].fill_(-1)
                    owner.active[1].fill_(True)
                    owner.next_embeddings[1].zero_()
            hidden = torch.ones(1, 8, device="cuda")
            noise = torch.randn(1, 4, device="cuda")
            slots = torch.tensor([slot], device="cuda")
            expected = eager._flow_postprocess(hidden, noise, slots)
            actual = runner.sample(hidden, noise, slots)
            for result, reference in zip(actual, expected, strict=True):
                torch.testing.assert_close(result, reference)
            for name in ("remaining_tail", "active", "next_embeddings"):
                torch.testing.assert_close(getattr(captured, name), getattr(eager, name))
            if step == 0:
                snapshot = actual[0].clone()
            if step == 2:
                assert snapshot.item()
                assert not actual[0].item()
        assert snapshot.item()
