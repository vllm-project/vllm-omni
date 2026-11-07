# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Full-weight Phonon generation and compiled CUDA codec validation."""

import json
import os
from pathlib import Path

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.phonon.cuda_graph_wrapper import MimiDecodeRunner
from vllm_omni.model_executor.models.phonon.mimi_decoder import MimiStreamingDecoder

pytestmark = [pytest.mark.advanced_model, pytest.mark.tts, pytest.mark.slow]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_stream_handoff_and_slot_reset():
    torch.manual_seed(53)
    # Allow one PCM16 quantization step for compiled/eager waveform rounding.
    waveform_atol = 1.0 / 32768
    checkpoint = os.environ.get("PHONON_CHECKPOINT_DIR") or "gradium/phonon-31m-en"
    from safetensors.torch import load_file
    from transformers.utils.hub import cached_file

    config = json.loads(Path(cached_file(checkpoint, "config.json")).read_text())
    weights = load_file(cached_file(checkpoint, "model.safetensors"))
    eager = MimiStreamingDecoder(config["mimi"], 2).cuda().eval()
    captured = MimiStreamingDecoder(config["mimi"], 2).cuda().eval()
    eager.load_official_weights(weights)
    captured.load_official_weights(weights)
    runner = MimiDecodeRunner(captured)
    slot = torch.tensor([1], device="cuda")
    emb_std = weights["flow_lm.emb_std"].cuda()
    emb_mean = weights["flow_lm.emb_mean"].cuda()
    first = None
    with torch.inference_mode():
        for step in range(4):
            latent = torch.randn(1, 32, device="cuda")
            codec_latent = (latent * emb_std + emb_mean).unsqueeze(-1)
            positions = torch.tensor([step * eager.upsample_ratio], device="cuda")
            expected = eager(codec_latent, slot, positions)
            actual, event = runner.decode(codec_latent, slot, positions)
            torch.cuda.current_stream().wait_event(event)
            torch.testing.assert_close(actual, expected, atol=waveform_atol, rtol=2e-4)
            if first is None:
                first = actual
                saved = first.clone()
        torch.testing.assert_close(first, saved)
        tail_valid = torch.ones(1, dtype=torch.bool, device="cuda")
        tail_positions = positions + eager.upsample_ratio
        old_tail, tail_event = runner.decode(codec_latent, slot, tail_positions)
        torch.cuda.current_stream().wait_event(tail_event)
        saved_tail = old_tail.clone()

        # Compare reuse with the same compiled graph to isolate stale-state handling.
        runner.reset_slots(1)
        positions.zero_()
        cleared_outputs = []
        for step in range(2):
            reference, event = runner.decode(codec_latent, slot, positions)
            torch.cuda.current_stream().wait_event(event)
            cleared_outputs.append(reference)
            positions.add_(eager.upsample_ratio)
        with torch.cuda.stream(runner.stream):
            for name, buffer in captured.named_buffers():
                if name.endswith(("history", "key_pool", "value_pool")):
                    buffer.fill_(float("nan"))
        positions.zero_()
        for reference in cleared_outputs:
            actual, event = runner.decode(codec_latent, slot, positions)
            torch.cuda.current_stream().wait_event(event)
            torch.testing.assert_close(actual, reference, atol=waveform_atol, rtol=2e-4)
            positions.add_(eager.upsample_ratio)

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
        torch.testing.assert_close(delivered, saved_tail.cpu().flatten(), atol=0, rtol=0)
