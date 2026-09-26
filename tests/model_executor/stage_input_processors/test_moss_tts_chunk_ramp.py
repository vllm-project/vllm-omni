# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""UT for Moss-TTS static chunk ramp in talker2codec_raw_async_chunk.

Tests the Moss-TTS stage input processor's chunk-size mode selection:
  1. Static ramp (codec_chunk_ramp configured)
  2. Backward compat (not configured -> original IC/steady behavior)

Framework utilities (parse_chunk_ramp, ramp_chunk_size) are already tested in
test_qwen3_tts_async_chunk.py -- this file only tests Moss-TTS adapter wiring.
"""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.moss_tts import (
    talker2codec_raw_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_NQ = 12  # Moss-TTS Local v1.5 n_vq


def _frame():
    """One valid codec frame (n_vq codes, none equal to pad code 1024)."""
    return torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=torch.long)


def _req(rid):
    return SimpleNamespace(external_req_id=rid)


def _tm(*, chunk_frames=15, initial_chunk_frames=1, chunk_ramp=None):
    extra = {
        "codec_chunk_frames": chunk_frames,
        "initial_codec_chunk_frames": initial_chunk_frames,
        "codec_left_context_frames": 0,
    }
    if chunk_ramp is not None:
        extra["codec_chunk_ramp"] = chunk_ramp
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        put_req_chunk=defaultdict(int),
        ramp_chunk_count=defaultdict(int),
        request_payload={},
        connector=SimpleNamespace(config={"extra": extra}),
    )


def _feed(tm, rid, n_frames, *, finished=False):
    """Call the processor with *n_frames* new frames appended."""
    if n_frames <= 0:
        mm = None
    else:
        mm = {"codes": {"audio": _frame().reshape(1, -1).repeat(n_frames, 1)}}
    return talker2codec_raw_async_chunk(
        transfer_manager=tm,
        multimodal_output=mm,
        request=_req(rid),
        is_finished=finished,
    )


class TestStaticRampEmission:
    """Verify chunk sizes follow the ramp table [2, 4, 8, 15]."""

    def test_ramp_sequence(self):
        tm = _tm(chunk_ramp=[2, 4, 8, 15])
        rid = "ramp-seq"

        # Chunk 0: need 2 frames
        assert _feed(tm, rid, 1) is None  # 1 < 2 -> hold
        p0 = _feed(tm, rid, 1)  # 2 >= 2 -> emit
        assert p0 is not None
        assert len(p0.codes.audio) == _NQ * 2
        tm.ramp_chunk_count[rid] = 1

        # Chunk 1: need 4 frames
        assert _feed(tm, rid, 3) is None
        p1 = _feed(tm, rid, 1)
        assert p1 is not None
        assert len(p1.codes.audio) == _NQ * 4
        tm.ramp_chunk_count[rid] = 2

        # Chunk 2: need 8 frames
        assert _feed(tm, rid, 7) is None
        p2 = _feed(tm, rid, 1)
        assert p2 is not None
        assert len(p2.codes.audio) == _NQ * 8
        tm.ramp_chunk_count[rid] = 3

        # Chunk 3+: steady 15
        assert _feed(tm, rid, 14) is None
        p3 = _feed(tm, rid, 1)
        assert p3 is not None
        assert len(p3.codes.audio) == _NQ * 15
        tm.ramp_chunk_count[rid] = 4

        # Chunk 4: still 15
        assert _feed(tm, rid, 14) is None
        p4 = _feed(tm, rid, 1)
        assert p4 is not None
        assert len(p4.codes.audio) == _NQ * 15

    def test_ramp_finished_flush(self):
        tm = _tm(chunk_ramp=[2, 4, 8, 15])
        rid = "ramp-flush"

        p0 = _feed(tm, rid, 2)
        assert p0 is not None
        tm.ramp_chunk_count[rid] = 1

        # 3 pending, finished -> flush all 3
        p_fin = _feed(tm, rid, 3, finished=True)
        assert p_fin is not None
        assert p_fin.meta.finished.item() is True
        assert len(p_fin.codes.audio) == _NQ * 3

    def test_ramp_steady_after_exhausted(self):
        """Short ramp [2, 4] -> steady 15 after ramp exhausted."""
        tm = _tm(chunk_ramp=[2, 4], chunk_frames=15)
        rid = "ramp-short"

        p0 = _feed(tm, rid, 2)
        assert len(p0.codes.audio) == _NQ * 2
        tm.ramp_chunk_count[rid] = 1

        p1 = _feed(tm, rid, 4)
        assert len(p1.codes.audio) == _NQ * 4
        tm.ramp_chunk_count[rid] = 2

        # Past ramp table -> steady
        assert _feed(tm, rid, 14) is None
        p2 = _feed(tm, rid, 1)
        assert len(p2.codes.audio) == _NQ * 15

    def test_ramp_string_config(self):
        """YAML-style string value "2,4,8,15" parses the same as a list."""
        tm = _tm(chunk_ramp="2,4,8,15")
        rid = "ramp-str"

        p0 = _feed(tm, rid, 2)
        assert p0 is not None
        assert len(p0.codes.audio) == _NQ * 2


class TestBackwardCompat:
    """Without ramp config, original IC/steady behavior is unchanged."""

    def test_ic_then_steady(self):
        tm = _tm(chunk_ramp=None, initial_chunk_frames=1, chunk_frames=15)
        rid = "no-ramp"

        # IC=1: first chunk emits 1 frame
        p0 = _feed(tm, rid, 1)
        assert p0 is not None
        assert len(p0.codes.audio) == _NQ * 1
        tm.put_req_chunk[rid] = 1

        # Steady=15
        assert _feed(tm, rid, 14) is None
        p1 = _feed(tm, rid, 1)
        assert p1 is not None
        assert len(p1.codes.audio) == _NQ * 15

    def test_no_frames_returns_none(self):
        tm = _tm(chunk_ramp=None, initial_chunk_frames=1)
        rid = "empty"
        assert _feed(tm, rid, 0) is None
