# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Numerical oracle from pinned, unmodified GLM-4-Voice on small CPU weights.

Fixture metadata records the source, revision, and reduced model config.
It covers block boundaries and partial final pooling windows in padded batches.
This is an encoder equivalence check, not pretrained-model validation.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import WhisperConfig

from vllm_omni.model_executor.models.common import whisper_vq

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.omni]
REFERENCE = Path(__file__).parent / "fixtures/glm_encoder_reference.safetensors"


@pytest.mark.parametrize("case", ["case0", "case1", "case2"])
def test_glm_encoder_matches_official_features_and_codes(case, monkeypatch):
    with safe_open(REFERENCE, framework="pt") as fixture:
        config = WhisperConfig(**json.loads(fixture.metadata()["config"]))
    config._attn_implementation = "eager"
    tensors = load_file(REFERENCE)
    encoder = whisper_vq.WhisperVQEncoder(
        config, causal_block_size=config.quantize_causal_block_size, preserve_padding=True
    ).eval()
    weights = {name.removeprefix("weights."): value for name, value in tensors.items() if name.startswith("weights.")}
    for name in ("ema_count", "ema_weight"):
        weights.pop(name)
    encoder.load_state_dict(weights, strict=True)

    quantize = whisper_vq.vector_quantize
    captured = []

    def capture_features(hidden_states, codebook):
        captured.append(hidden_states)
        return quantize(hidden_states, codebook)

    monkeypatch.setattr(whisper_vq, "vector_quantize", capture_features)
    with torch.inference_mode():
        output = encoder(tensors[f"{case}.input_features"], attention_mask=tensors[f"{case}.attention_mask"])
    torch.testing.assert_close(captured[0], tensors[f"{case}.pre_vq"], rtol=1e-5, atol=1e-6)
    assert torch.equal(output.quantized_token_ids, tensors[f"{case}.codes"])
