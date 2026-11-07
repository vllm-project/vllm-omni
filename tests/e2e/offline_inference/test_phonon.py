# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Full-weight compiled Mimi streaming and request-reuse validation."""

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
def test_compiled_mimi_streaming_and_request_reuse():
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
    inputs = []
    outputs = []
    saved_outputs = []
    with torch.inference_mode():
        for step in range(4):
            latent = torch.randn(1, 32, device="cuda")
            codec_latent = (latent * emb_std + emb_mean).unsqueeze(-1)
            positions = torch.tensor([step * eager.upsample_ratio], device="cuda")
            inputs.append(codec_latent)
            expected = eager(codec_latent, slot, positions)
            actual, event = runner.decode(codec_latent, slot, positions)
            torch.cuda.current_stream().wait_event(event)
            torch.testing.assert_close(actual, expected, atol=waveform_atol, rtol=2e-4)
            outputs.append(actual)
            saved_outputs.append(actual.clone())
        positions.zero_()
        for latent, expected in zip(inputs[:2], saved_outputs[:2], strict=True):
            actual, event = runner.decode(latent, slot, positions)
            torch.cuda.current_stream().wait_event(event)
            torch.testing.assert_close(actual, expected, atol=waveform_atol, rtol=2e-4)
            positions.add_(eager.upsample_ratio)
        for output, expected in zip(outputs, saved_outputs, strict=True):
            torch.testing.assert_close(output, expected, atol=0, rtol=0)
