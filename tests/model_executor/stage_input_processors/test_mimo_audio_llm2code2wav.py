# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import logging
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors import mimo_audio as sip
from vllm_omni.model_executor.stage_input_processors.mimo_audio import (
    MAX_CODE2WAV_TOKENS,
    llm2code2wav,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_stage_list(codec_codes: torch.Tensor, request_id: str = "req-0"):
    """Build a minimal stage_list[0] with a single talker output carrying codec_codes."""
    output = SimpleNamespace(multimodal_output={"codes": {"audio": codec_codes}})
    talker_output = SimpleNamespace(outputs=[output], request_id=request_id)
    stage0 = SimpleNamespace(engine_outputs=[talker_output])
    return [stage0]


def test_llm2code2wav_truncates_when_flat_exceeds_max(caplog):
    """Flat codec sequences longer than MAX_CODE2WAV_TOKENS must be truncated, not passed through."""
    # prepend_and_flatten_colmajor produces 36 ids per (8, 4) codec frame:
    # pad adds one row -> (9, 4) per frame, permuted and flattened.
    # Pick enough frames to comfortably exceed the cap.
    frames = (MAX_CODE2WAV_TOKENS // 36) + 100
    codec_codes = torch.ones(frames, 1, 8, 4, dtype=torch.long)

    stage_list = _make_stage_list(codec_codes, request_id="req-long")

    # Attach caplog's handler directly to the module logger so the warning is
    # captured regardless of propagation (vllm's logger configuration can
    # interact badly with caplog.at_level's default root-handler path).
    target_logger = logging.getLogger("vllm_omni.model_executor.stage_input_processors.mimo_audio")
    target_logger.addHandler(caplog.handler)
    prev_level = target_logger.level
    target_logger.setLevel(logging.WARNING)
    try:
        prompts = llm2code2wav(stage_list, engine_input_source=[0])
    finally:
        target_logger.removeHandler(caplog.handler)
        target_logger.setLevel(prev_level)

    assert len(prompts) == 1
    assert len(prompts[0]["prompt_token_ids"]) == MAX_CODE2WAV_TOKENS
    assert any("truncating" in rec.getMessage() for rec in caplog.records), (
        f"Expected a 'truncating' warning; captured records: {[r.getMessage() for r in caplog.records]}"
    )


def test_llm2code2wav_short_sequence_unchanged():
    """Short codec sequences are returned without truncation."""
    codec_codes = torch.ones(4, 1, 8, 4, dtype=torch.long)
    stage_list = _make_stage_list(codec_codes, request_id="req-short")

    prompts = llm2code2wav(stage_list, engine_input_source=[0])

    assert len(prompts) == 1
    # 4 frames + 1 pad row, flattened col-major → well below the cap
    assert 0 < len(prompts[0]["prompt_token_ids"]) <= MAX_CODE2WAV_TOKENS


def test_llm2code2wav_truncation_boundary_constant_matches_yaml():
    """MAX_CODE2WAV_TOKENS must match the stage-1 max_model_len in mimo_audio.yaml and end2end.py."""
    assert sip.MAX_CODE2WAV_TOKENS == 18192
