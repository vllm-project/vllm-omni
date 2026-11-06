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
        runner.reset_slots(slot)
        eager.reset_slots(slot)
        positions.zero_()
        actual, event = runner.decode(codec_latent, slot, positions)
        torch.cuda.current_stream().wait_event(event)
        torch.testing.assert_close(
            actual, eager(codec_latent, slot, positions), atol=2e-5, rtol=2e-4
        )
