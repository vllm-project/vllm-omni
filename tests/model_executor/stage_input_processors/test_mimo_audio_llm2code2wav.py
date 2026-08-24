# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import logging
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors import mimo_audio as sip
from vllm_omni.model_executor.stage_input_processors.mimo_audio import (
    MAX_CODE2WAV_TOKENS,
    llm2code2wav_full_payload,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_llm2code2wav_full_payload_truncates_when_flat_exceeds_max(caplog):
    """Flat codec sequences longer than MAX_CODE2WAV_TOKENS must be truncated."""
    frames = (MAX_CODE2WAV_TOKENS // 36) + 100
    codec_codes = torch.ones(frames, 1, 8, 4, dtype=torch.long)
    request = SimpleNamespace(request_id="req-long")

    target_logger = logging.getLogger("vllm_omni.model_executor.stage_input_processors.mimo_audio")
    target_logger.addHandler(caplog.handler)
    prev_level = target_logger.level
    target_logger.setLevel(logging.WARNING)
    try:
        payload = llm2code2wav_full_payload(None, {"codes.audio": codec_codes}, request)
    finally:
        target_logger.removeHandler(caplog.handler)
        target_logger.setLevel(prev_level)

    assert payload is not None
    assert len(payload["codes"]["audio"]) == MAX_CODE2WAV_TOKENS


def test_llm2code2wav_full_payload_short_sequence_unchanged():
    codec_codes = torch.ones(4, 1, 8, 4, dtype=torch.long)
    request = SimpleNamespace(request_id="req-short")

    payload = llm2code2wav_full_payload(None, {"codes.audio": codec_codes}, request)

    assert payload is not None
    assert 0 < len(payload["codes"]["audio"]) <= MAX_CODE2WAV_TOKENS


def test_llm2code2wav_truncation_boundary_constant_matches_yaml():
    """MAX_CODE2WAV_TOKENS must match the stage-1 max_model_len in mimo_audio.yaml and end2end.py."""
    assert sip.MAX_CODE2WAV_TOKENS == 18192
