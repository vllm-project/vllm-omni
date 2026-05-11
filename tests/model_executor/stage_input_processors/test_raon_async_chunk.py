# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Raon stage0_to_stage1_async_chunk (async-chunk emission and finish-tick cleanup)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.raon import (
    _DEFAULT_CHUNK_FRAMES,
    stage0_to_stage1_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_G = 8
_CHUNK_SIZE = _DEFAULT_CHUNK_FRAMES


def _req(
    external_req_id: str,
    *,
    finished: bool = False,
    request_id: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        external_req_id=external_req_id,
        request_id=request_id or external_req_id,
        is_finished=lambda: finished,
    )


def _tm() -> SimpleNamespace:
    return SimpleNamespace()


def _tm_with_config(chunk_frames: int = 10, left_context_frames: int = 10) -> SimpleNamespace:
    return SimpleNamespace(
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": chunk_frames,
                    "codec_left_context_frames": left_context_frames,
                }
            }
        )
    )


def _codes(t: int, g: int = _G) -> torch.Tensor:
    return torch.arange(t * g, dtype=torch.long).reshape(t, g)


def _pooling(*, chunk: torch.Tensor | None = None, full: torch.Tensor | None = None) -> dict:
    out: dict = {}
    if chunk is not None:
        out["codec_codes_chunk"] = chunk
    if full is not None:
        out["codec_codes"] = full
    return out


# ===================================================================
# Skip path (non-audio)
# ===================================================================


@pytest.mark.parametrize(
    "pooling_output",
    [None, {"some_other_key": 123}],
    ids=["none-pooling", "no-codec-keys"],
)
def test_skip_path_returns_none(pooling_output):
    req = _req("skip-test")
    assert stage0_to_stage1_async_chunk(_tm(), pooling_output, req) is None


# ===================================================================
# Windowed chunk emission
# ===================================================================


class TestAsyncChunkWindowedEmission:
    def test_first_chunk_waits_then_emits_at_chunk_size(self):
        rid = "windowed"
        tm = _tm()
        req = _req(rid)
        assert stage0_to_stage1_async_chunk(tm, _pooling(chunk=_codes(1)), req) is None

        payload = None
        for _ in range(_CHUNK_SIZE - 1):
            payload = stage0_to_stage1_async_chunk(tm, _pooling(chunk=_codes(1)), req)
        assert payload is not None
        assert "codec_codes" in payload
        assert payload["left_context_size"] == 0
        assert tm._raon_chunk_state[rid].emitted_frames == _CHUNK_SIZE

    def test_second_chunk_includes_left_context(self):
        rid = "windowed-ctx"
        tm = _tm_with_config(chunk_frames=5, left_context_frames=3)
        req = _req(rid)
        for _ in range(5):
            stage0_to_stage1_async_chunk(tm, _pooling(chunk=_codes(1)), req)
        for _ in range(4):
            stage0_to_stage1_async_chunk(tm, _pooling(chunk=_codes(1)), req)
        payload = stage0_to_stage1_async_chunk(tm, _pooling(chunk=_codes(1)), req)
        assert payload is not None
        assert payload["left_context_size"] == 3
        assert len(payload["codec_codes"]) == 8 * _G

    def test_history_retained_after_emission(self):
        rid = "windowed-hist"
        tm = _tm()
        req = _req(rid)
        for _ in range(_CHUNK_SIZE):
            stage0_to_stage1_async_chunk(tm, _pooling(chunk=_codes(1)), req)
        all_codes = tm._raon_chunk_state[rid].all_codes
        total = sum(int(c.shape[0]) for c in all_codes)
        assert total == _CHUNK_SIZE


# ===================================================================
# Finish / flush path
# ===================================================================


class TestAsyncChunkFinishFlush:
    def test_finish_emits_accumulated_data_and_cleans_state(self):
        rid = "finish-flush"
        tm = _tm()
        req_running = _req(rid, finished=False)
        req_done = _req(rid, finished=True)
        stage0_to_stage1_async_chunk(tm, _pooling(chunk=_codes(1)), req_running)
        payload = stage0_to_stage1_async_chunk(tm, _pooling(chunk=_codes(2)), req_done, is_finished=True)
        assert payload is not None
        assert payload["finished"].item() is True
        assert payload["flush_only"] is False
        assert rid not in tm._raon_chunk_state

    def test_finish_with_no_accumulated_data(self):
        rid = "finish-no-data"
        tm = _tm()
        req = _req(rid, finished=True)
        payload = stage0_to_stage1_async_chunk(tm, _pooling(chunk=_codes(1)), req, is_finished=True)
        assert payload is not None

    def test_finish_with_none_pooling_and_no_prior_data(self):
        rid = "finish-none"
        tm = _tm()
        req = _req(rid, finished=True)
        payload = stage0_to_stage1_async_chunk(tm, None, req, is_finished=True)
        if payload is not None:
            assert payload.get("flush_only") is True


# ===================================================================
# Full payload fallback
# ===================================================================


def test_full_payload_key_used_when_chunk_absent():
    rid = "full-payload"
    req = _req(rid)
    payload = stage0_to_stage1_async_chunk(_tm(), _pooling(full=_codes(_CHUNK_SIZE)), req)
    assert payload is not None
    assert "codec_codes" in payload


# ===================================================================
# Request ID mapping
# ===================================================================


class TestAsyncChunkReqIdMapping:
    def test_internal_id_stored_on_state_and_cleared_on_finish(self):
        internal_id = "int-id-map"
        external_id = "ext-id-map"
        tm = _tm()
        req_running = SimpleNamespace(
            external_req_id=external_id,
            request_id=internal_id,
            is_finished=lambda: False,
        )
        stage0_to_stage1_async_chunk(tm, _pooling(chunk=_codes(1)), req_running)
        assert external_id in tm._raon_chunk_state
        assert tm._raon_chunk_state[external_id].internal_id == internal_id

        req_done = SimpleNamespace(
            external_req_id=external_id,
            request_id=internal_id,
            is_finished=lambda: True,
        )
        stage0_to_stage1_async_chunk(tm, _pooling(chunk=_codes(1)), req_done, is_finished=True)
        assert external_id not in tm._raon_chunk_state


# ===================================================================
# Payload structure validation (comprehensive)
# ===================================================================


def test_payload_structure_comprehensive():
    rid = "payload-struct"
    req = _req(rid, finished=False)
    payload = stage0_to_stage1_async_chunk(_tm(), _pooling(chunk=_codes(_CHUNK_SIZE)), req)
    assert payload is not None

    for key in ("codec_codes", "codec_codes_flat", "global_request_id", "finished", "flush_only", "left_context_size"):
        assert key in payload, f"missing key: {key}"

    assert isinstance(payload["finished"], torch.Tensor)
    assert payload["finished"].dtype == torch.bool
    assert payload["finished"].item() is False

    codes = payload["codec_codes"]
    assert isinstance(codes, list)
    assert all(isinstance(v, int) for v in codes)
    assert len(codes) == _CHUNK_SIZE * _G

    assert isinstance(payload["left_context_size"], int)


# ===================================================================
# _trim_left_context_audio
# ===================================================================


class TestTrimLeftContextAudio:
    @pytest.mark.parametrize(
        "audio_len,left_ctx,total_frames,expected_len",
        [
            (100, 0, 10, 100),
            (1000, 2, 10, 800),
            (100, 10, 10, 0),
            (100, 5, 0, 100),
        ],
        ids=["no-trim", "proportional", "exceeds-length", "zero-total-frames"],
    )
    def test_trim(self, audio_len, left_ctx, total_frames, expected_len):
        from vllm_omni.model_executor.models.raon.raon_code2wav import RaonCode2WavModel

        audio = torch.randn(audio_len)
        result = RaonCode2WavModel._trim_left_context_audio(audio, left_ctx, total_frames)
        assert result.shape[-1] == expected_len
