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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compiled_mimi_replay_slot_reuse_and_pcm_ownership():
    import copy

    from vllm_omni.model_executor.models.phonon.cuda_graph_wrapper import MimiDecodeRunner
    from vllm_omni.model_executor.models.phonon.mimi_decoder import MimiStreamingDecoder

    config = dict(
        dimension=8,
        quantizer_dimension=4,
        quantizer_output_dimension=8,
        transformer_d_model=8,
        transformer_num_heads=2,
        transformer_num_layers=1,
        transformer_context=3,
        transformer_max_period=10000.0,
        transformer_dim_feedforward=16,
        transformer_layer_scale=0.01,
        ratios=[2],
        sample_rate=8,
        frame_rate=2,
        n_filters=4,
        kernel_size=3,
        last_kernel_size=3,
        channels=1,
        n_residual_layers=1,
        residual_kernel_size=3,
        dilation_base=2,
        compress=2,
    )
    torch.manual_seed(42)
    eager = MimiStreamingDecoder(config, 2).cuda().eval()
    captured = copy.deepcopy(eager)
    runner = MimiDecodeRunner(captured)
    slot = torch.tensor([1], device="cuda")
    saved = []
    with torch.inference_mode():
        for step in range(6):
            if step == 4:
                for decoder in (eager, captured):
                    for name, buffer in decoder.named_buffers():
                        if name.endswith(("history", "key_pool", "value_pool")):
                            buffer.fill_(float("nan"))
            position = torch.tensor([((step if step < 4 else step - 4) * eager.upsample_ratio)], device="cuda")
            latent = torch.randn(1, 4, 1, device="cuda")
            expected = eager(latent, slot, position)
            actual, event = runner.decode(latent, slot, position)
            torch.cuda.current_stream().wait_event(event)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
            for returned, snapshot in saved:
                torch.testing.assert_close(returned, snapshot, atol=0, rtol=0)
            saved.append((actual, actual.clone()))
