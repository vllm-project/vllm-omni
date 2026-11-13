# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.moss_tts import talker2codec_raw_async_chunk


def manager(**overrides):
    return SimpleNamespace(
        connector=SimpleNamespace(config=dict(initial_codec_chunk_frames=1, codec_chunk_frames=15, **overrides))
    )


def output(i, first=None):
    result = {"codes": {"audio": torch.tensor([[i, i + 1]])}}
    if first is not None:
        result["meta"] = {"first_audio": torch.tensor(first)}
    return result


def test_defer_includes_first_code_and_emits_at_fifteen():
    m = manager()
    r = SimpleNamespace(request_id="a")
    assert talker2codec_raw_async_chunk(m, output(0, True), r) is None
    for i in range(1, 14):
        assert talker2codec_raw_async_chunk(m, output(i), r) is None
    p = talker2codec_raw_async_chunk(m, output(14), r)
    assert p.meta.codec_chunk_frames == 15 and bool(p.meta.first_audio)
    assert p.codes.audio.reshape(2, 15)[0].tolist() == list(range(15))
    m.put_req_chunk["a"] = 1
    for i in range(15, 29):
        assert talker2codec_raw_async_chunk(m, output(i), r) is None
    p = talker2codec_raw_async_chunk(m, output(29), r)
    assert p.meta.codec_chunk_frames == 15
    assert p.codes.audio.reshape(2, 15)[0].tolist() == list(range(15, 30))


def test_short_terminal_flushes_and_clears_promise():
    m = manager()
    r = SimpleNamespace(request_id="a")
    assert talker2codec_raw_async_chunk(m, output(0, True), r) is None
    p = talker2codec_raw_async_chunk(m, None, r, True)
    assert p.meta.codec_chunk_frames == 1 and bool(p.meta.first_audio) and bool(p.meta.finished)
    assert not m.request_payload and not m.code_prompt_token_ids


def test_unaccepted_direct_path_keeps_one_frame_latency():
    m = manager()
    r = SimpleNamespace(request_id="a")
    p = talker2codec_raw_async_chunk(m, output(0, False), r)
    assert p.meta.codec_chunk_frames == 1 and p.meta.first_audio is None


def test_explicit_ramp_takes_precedence_over_deferral():
    m = manager(codec_chunk_ramp=[2, 4, 15])
    r = SimpleNamespace(request_id="a")
    assert talker2codec_raw_async_chunk(m, output(0, True), r) is None
    p = talker2codec_raw_async_chunk(m, output(1), r)
    assert p.meta.codec_chunk_frames == 2 and bool(p.meta.first_audio)
    assert p.codes.audio.reshape(2, 2)[0].tolist() == [0, 1]


pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
