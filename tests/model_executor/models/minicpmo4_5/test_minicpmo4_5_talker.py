# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo4_5.minicpmo4_5_talker import (
    MiniCPMO4_5TalkerForConditionalGeneration,
    _MiniCPMOAsyncTalkerState,
)
from vllm_omni.model_executor.models.minicpmo4_5.minicpmo4_5_tts_generator import (
    MiniCPMO4_5RepeatPenaltyLogitsProcessor,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _minimal_talker():
    model = object.__new__(MiniCPMO4_5TalkerForConditionalGeneration)
    model._async_codec_chunk_frames = 4
    model._async_eos_token_id = 99
    return model


def test_repeat_penalty_counts_recent_minicpm_codec_tokens():
    processor = MiniCPMO4_5RepeatPenaltyLogitsProcessor(
        penalty=2.0,
        max_input_ids=100,
        past_window=4,
    )

    scores = torch.tensor([[8.0, 8.0, -8.0]], dtype=torch.float32)
    input_ids = torch.tensor([[2, 1, 1, 1, 0]], dtype=torch.long)

    adjusted = processor(input_ids, scores)

    assert adjusted.tolist() == [[4.0, 1.0, -8.0]]


def test_async_sampling_controls_keep_multiple_tts_candidates():
    model = _minimal_talker()
    model._async_chunk_enabled = True
    model._async_sampling_repetition_penalty = 1.02
    model._async_sampling_do_sample = True
    model._async_sampling_top_p = 0.1
    model._async_sampling_top_k = 2
    model._async_sampling_min_p = 0.0
    model.config = SimpleNamespace(num_audio_tokens=10)

    model._rebuild_async_sampling_controls()

    assert isinstance(model._async_logits_processors[0], MiniCPMO4_5RepeatPenaltyLogitsProcessor)

    input_ids = torch.tensor([[0]], dtype=torch.long)
    logits = torch.tensor([[10.0, 9.0, 8.0, 1.0, 0.0]], dtype=torch.float32)
    for warper in model._async_logits_warpers:
        logits = warper(input_ids, logits)

    assert int(torch.isfinite(logits).sum().item()) == 3


def test_record_async_emitted_audio_token_writes_single_condition_chunk(tmp_path, monkeypatch):
    monkeypatch.setenv("MINICPMO45_E2E_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("MINICPMO45_E2E_DEBUG_ARTIFACTS", "1")
    model = _minimal_talker()
    state = _MiniCPMOAsyncTalkerState(request_id="r")

    for token_id in (10, 11, 12, 13):
        model._record_async_emitted_audio_token(
            state,
            token_id=token_id,
            condition_index=0,
            condition_shape=[1, 10, 768],
            text_finished=False,
        )

    chunk_path = tmp_path / "debug" / "minicpmo4_5_async_chunk" / "r" / "talker_codec_condition_chunks.jsonl"
    rows = [json.loads(line) for line in chunk_path.read_text(encoding="utf-8").splitlines()]

    assert rows == [
        {
            "audio_token_count": 4,
            "audio_token_ids": [10, 11, 12, 13],
            "chunk_index": 0,
            "condition_index": 0,
            "condition_index_end": 0,
            "condition_index_start": 0,
            "condition_indices": [0],
            "condition_shape": [1, 10, 768],
            "condition_shape_end": [1, 10, 768],
            "condition_shape_start": [1, 10, 768],
            "condition_shapes": [[1, 10, 768]],
            "is_last_audio_chunk": False,
            "text_finished": False,
            "text_finished_flags": [False],
        }
    ]


def test_record_async_emitted_audio_token_marks_mixed_condition_final_chunk(tmp_path, monkeypatch):
    monkeypatch.setenv("MINICPMO45_E2E_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("MINICPMO45_E2E_DEBUG_ARTIFACTS", "1")
    model = _minimal_talker()
    state = _MiniCPMOAsyncTalkerState(request_id="r")

    model._record_async_emitted_audio_token(
        state,
        token_id=20,
        condition_index=0,
        condition_shape=[1, 10, 768],
        text_finished=False,
    )
    model._record_async_emitted_audio_token(
        state,
        token_id=21,
        condition_index=1,
        condition_shape=[1, 7, 768],
        text_finished=True,
    )
    model._record_async_emitted_audio_token(
        state,
        token_id=22,
        condition_index=1,
        condition_shape=[1, 7, 768],
        text_finished=True,
    )
    model._record_async_emitted_audio_token(
        state,
        token_id=99,
        condition_index=1,
        condition_shape=[1, 7, 768],
        text_finished=True,
    )

    chunk_path = tmp_path / "debug" / "minicpmo4_5_async_chunk" / "r" / "talker_codec_condition_chunks.jsonl"
    rows = [json.loads(line) for line in chunk_path.read_text(encoding="utf-8").splitlines()]

    assert rows == [
        {
            "audio_token_count": 4,
            "audio_token_ids": [20, 21, 22, 99],
            "chunk_index": 0,
            "condition_index": None,
            "condition_index_end": 1,
            "condition_index_start": 0,
            "condition_indices": [0, 1],
            "condition_shape": None,
            "condition_shape_end": [1, 7, 768],
            "condition_shape_start": [1, 10, 768],
            "condition_shapes": [[1, 10, 768], [1, 7, 768]],
            "is_last_audio_chunk": True,
            "text_finished": True,
            "text_finished_flags": [False, True],
        }
    ]
