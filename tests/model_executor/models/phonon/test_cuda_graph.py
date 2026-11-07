# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import os
from pathlib import Path

import pytest
import torch

from vllm_omni.model_executor.models.phonon.cuda_graph_wrapper import (
    FlowSampleRunner,
    MimiDecodeRunner,
)
from vllm_omni.model_executor.models.phonon.mimi_decoder import MimiStreamingDecoder
from vllm_omni.model_executor.models.phonon.phonon_talker import PhononFlowLM

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_stream_handoff_and_slot_reset():
    checkpoint = os.environ.get("PHONON_CHECKPOINT_DIR") or "gradium/phonon-31m-en"
    from safetensors.torch import load_file
    from transformers.utils.hub import cached_file

    config = json.loads(Path(cached_file(checkpoint, "config.json")).read_text())
    weights = load_file(cached_file(checkpoint, "model.safetensors"))
    eager = MimiStreamingDecoder(config["mimi"], 2).cuda().eval()
    captured = MimiStreamingDecoder(config["mimi"], 2).cuda().eval()
    eager.load_official_weights(weights)
    captured.load_official_weights(weights)
    lm = PhononFlowLM(config).cuda().eval()
    lm.load_official_weights(weights)
    flow = FlowSampleRunner(lm)
    runner = MimiDecodeRunner(captured)
    slot = torch.tensor([1], device="cuda")
    first = None
    with torch.inference_mode():
        for step in range(4):
            hidden, noise = (
                torch.randn(1, 512, device="cuda"),
                torch.randn(1, 32, device="cuda"),
            )
            latent, eos = flow.sample(hidden, noise)
            reference, expected_eos = lm.sample(hidden, noise)
            torch.testing.assert_close(latent, reference)
            torch.testing.assert_close(eos, expected_eos)
            codec_latent = (latent * lm.emb_std + lm.emb_mean).unsqueeze(-1)
            positions = torch.tensor([step * eager.upsample_ratio], device="cuda")
            expected = eager(codec_latent, slot, positions)
            actual, event = runner.decode(codec_latent, slot, positions)
            torch.cuda.current_stream().wait_event(event)
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
            if first is None:
                first = actual
                saved = first.clone()
        torch.testing.assert_close(first, saved)
        tail_valid = torch.ones(1, dtype=torch.bool, device="cuda")
        tail_positions = positions + eager.upsample_ratio
        expected_tail = eager(codec_latent, slot, tail_positions)
        old_tail, tail_event = runner.decode(codec_latent, slot, tail_positions)
        with torch.cuda.stream(runner.stream):
            for name, buffer in captured.named_buffers():
                if name.endswith(("history", "key_pool", "value_pool")):
                    buffer.fill_(float("nan"))
        eager.reset_slots(slot)
        positions.zero_()
        actual, event = runner.decode(codec_latent, slot, positions)
        torch.cuda.current_stream().wait_event(event)
        torch.testing.assert_close(actual, eager(codec_latent, slot, positions), atol=2e-5, rtol=2e-4)

        # Deliver the old request's final PCM after slot reset and a new decode.
        from vllm_omni.model_executor.models.phonon.model_state import PhononAudioOutput

        copy_stream = torch.cuda.Stream()
        snapshot = PhononAudioOutput(old_tail, tail_valid, [0], 1, tail_event)

        def copy_tensor(tensor):
            host = torch.empty_like(tensor, device="cpu", pin_memory=True)
            host.copy_(tensor, non_blocking=True)
            tensor.record_stream(copy_stream)
            return host

        with torch.cuda.stream(copy_stream):
            host_snapshot = snapshot.to_cpu(copy_stream, copy_tensor)
            copied = torch.cuda.Event()
            copied.record(copy_stream)
        copied.synchronize()
        delivered = host_snapshot.get_output()[0]["model_outputs"]
        torch.testing.assert_close(delivered, expected_tail.cpu().flatten(), atol=2e-5, rtol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("frames", [1, 16, 270])
def test_direct_ring_attention_matches_cpu_reference(frames):
    from vllm_omni.model_executor.models.phonon.mimi_kernels import ring_attention_chunk

    torch.manual_seed(89)
    slots = torch.tensor([2, 0])
    positions = torch.tensor([247, 500])
    keys = torch.randn(3, 250, 2, 64)
    values = torch.randn_like(keys)
    gpu_keys, gpu_values = keys.cuda(), values.cuda()
    for _ in range(2):
        query, key, value = [torch.randn(2, frames, 2, 64) for _ in range(3)]
        expected = ring_attention_chunk(query, key, value, keys, values, slots, positions)
        actual = ring_attention_chunk(
            query.cuda(), key.cuda(), value.cuda(), gpu_keys, gpu_values, slots.cuda(), positions.cuda()
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
