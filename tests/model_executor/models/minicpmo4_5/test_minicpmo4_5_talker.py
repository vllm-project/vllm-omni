# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm_omni.model_executor.models.minicpmo4_5.minicpmo4_5_talker import (
    MiniCPMO4_5TalkerForConditionalGeneration,
    _MiniCPMOAsyncTalkerState,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _minimal_talker():
    model = object.__new__(MiniCPMO4_5TalkerForConditionalGeneration)
    model._async_codec_chunk_frames = 4
    model._async_eos_token_id = 99
    return model


def test_record_async_emitted_audio_token_writes_single_condition_chunk(tmp_path, monkeypatch):
    monkeypatch.setenv("MINICPMO45_E2E_OUTPUT_DIR", str(tmp_path))
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
