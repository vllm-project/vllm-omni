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
        runner.reset_slots(1)
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
