# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""IndexTTS2 S2Mel decoder emits the canonical sample-rate payload shape."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.indextts2.configuration_indextts2 import IndexTTS2Config
from vllm_omni.model_executor.models.indextts2.indextts2_s2mel_decoder import IndexTTS2S2MelDecoder
from vllm_omni.sample_rate import SAMPLE_RATE_KEY, get_sampling_rate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

UNUSUAL_RATE = 22050


def _bare_decoder() -> IndexTTS2S2MelDecoder:
    model = object.__new__(IndexTTS2S2MelDecoder)
    nn.Module.__init__(model)
    model.use_gpt_latent = True
    model._sample_rate = UNUSUAL_RATE
    # Only touched for dtype resolution before the warmup return.
    model.s2mel = SimpleNamespace(models={"gpt_layer": [SimpleNamespace(weight=SimpleNamespace(dtype=torch.float32))]})
    return model


def test_config_output_sample_rate_follows_s2mel_sr() -> None:
    # The config property is the single source both decoder emission sites
    # now read; an unusual s2mel preprocess sr must flow through it.
    config = IndexTTS2Config(s2mel={"preprocess_params": {"sr": UNUSUAL_RATE}})
    assert config.output_sample_rate == UNUSUAL_RATE


def test_warmup_path_emits_canonical_shape() -> None:
    # vLLM's profiling forward: synthetic token inputs, no upstream payload.
    model = _bare_decoder()
    output = model.forward(torch.zeros(2, 4, dtype=torch.long))

    multimodal_outputs = output.multimodal_outputs
    assert "audio" in multimodal_outputs
    # The warmup path used to emit a plain int (``"sr": 22050``) while the
    # main path emitted a tensor; both now share the canonical list shape.
    srs = multimodal_outputs[SAMPLE_RATE_KEY]
    assert isinstance(srs, list) and len(srs) == 1
    assert srs[0].dim() == 0 and srs[0].dtype == torch.int32
    assert get_sampling_rate(multimodal_outputs) == UNUSUAL_RATE
