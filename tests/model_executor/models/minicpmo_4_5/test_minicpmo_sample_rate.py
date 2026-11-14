# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniCPM-o 4.5 code2wav emits the canonical sample-rate payload shape."""

import contextlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav as code2wav_module
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import MiniCPMO45Code2Wav
from vllm_omni.sample_rate import SAMPLE_RATE_KEY, get_sampling_rate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

UNUSUAL_RATE = 22050


def _bare_model() -> MiniCPMO45Code2Wav:
    model = object.__new__(MiniCPMO45Code2Wav)
    nn.Module.__init__(model)
    model._watchdog = SimpleNamespace(
        phase=lambda *args, **kwargs: None,
        watch=lambda *args, **kwargs: contextlib.nullcontext(),
    )
    model._tf32_mode = "off"
    return model


def test_warmup_path_rate_follows_stage_operating_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    # The emission is derived from the module's stage operating rate (the
    # same constant the reference-audio resampler targets), never a fresh
    # literal: pin it to an unusual value and the emission must follow.
    monkeypatch.setattr(code2wav_module, "_REF_TARGET_SAMPLE_RATE", UNUSUAL_RATE)
    model = _bare_model()

    output = model.forward(input_ids=torch.zeros(3, dtype=torch.long))

    multimodal_outputs = output.multimodal_outputs
    srs = multimodal_outputs[SAMPLE_RATE_KEY]
    assert isinstance(srs, list) and len(srs) == 1
    assert srs[0].dim() == 0 and srs[0].dtype == torch.int32
    assert get_sampling_rate(multimodal_outputs) == UNUSUAL_RATE


def test_warmup_path_rate_matches_reference_target_by_default() -> None:
    # Un-stubbed, the stage advertises the token2wav operating rate.
    model = _bare_model()

    output = model.forward(input_ids=torch.zeros(3, dtype=torch.long))

    assert get_sampling_rate(output.multimodal_outputs) == code2wav_module._REF_TARGET_SAMPLE_RATE
