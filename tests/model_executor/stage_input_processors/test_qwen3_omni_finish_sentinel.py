"""code2wav async finish-sentinel terminal flush.

The producer runner sends every in-step codec chunk with ``finished=False`` and
emits a separate finish sentinel next cycle (empty payload + the
``ASYNC_FINISH_SENTINEL_KEY`` marker the legacy adapter never sets).  On that
marker, ``talker2code2wav_async_chunk`` must flush the trailing partial codec
chunk that the live ``is_finished`` branch would otherwise have flushed, reusing
the same context math, without re-appending.
"""

from types import SimpleNamespace

import torch

from vllm_omni.data_entry_keys import ASYNC_FINISH_SENTINEL_KEY
from vllm_omni.model_executor.stage_input_processors.qwen3_omni import talker2code2wav_async_chunk


def _tm(accumulated, chunk_frames=4, left_frames=25, initial_frames=None):
    cfg = {"codec_chunk_frames": chunk_frames, "codec_left_context_frames": left_frames}
    if initial_frames is not None:
        cfg["initial_codec_chunk_frames"] = initial_frames
    return SimpleNamespace(
        code_prompt_token_ids=dict(accumulated),
        connector=SimpleNamespace(config={"extra": cfg}),
    )


def _sentinel_payload():
    return {ASYNC_FINISH_SENTINEL_KEY: True}


def test_codec_config_can_come_from_runner_model_config():
    cfg = {
        "codec_chunk_frames": 25,
        "codec_left_context_frames": 25,
        "initial_codec_chunk_frames": 4,
    }
    model_config = SimpleNamespace(stage_connector_config={"extra": cfg})
    tm = SimpleNamespace(
        code_prompt_token_ids={"r": []},
        put_req_chunk={"r": 0},
    )
    tm._get_model_config = lambda: model_config
    req = SimpleNamespace(external_req_id="r", sampling_params=None)
    frame = torch.ones((1, 16), dtype=torch.long)

    for _ in range(3):
        assert talker2code2wav_async_chunk(tm, {"codes": {"audio": frame}}, req) is None
    out = talker2code2wav_async_chunk(tm, {"codes": {"audio": frame}}, req)

    assert out is not None
    assert out.codes.audio.numel() == 64
    assert out.meta.left_context_size == 0


def test_talker2code2wav_async_chunk_accepts_flat_payload():
    tm = _tm({"r": []}, chunk_frames=1, left_frames=0)
    tm.put_req_chunk = {"r": 0}
    req = SimpleNamespace(external_req_id="r", sampling_params=None)
    frame = torch.arange(1, 17, dtype=torch.long).reshape(1, 16)

    out = talker2code2wav_async_chunk(tm, {"codes.audio": frame}, req)

    assert out is not None
    assert torch.equal(out.codes.audio, frame.transpose(0, 1).reshape(-1))


def test_finish_sentinel_flushes_partial_tail():
    # 6 frames accumulated, chunk size 4 -> a 2-frame partial tail is still held.
    tm = _tm({"r": [torch.tensor([[i]]) for i in range(1, 7)]}, chunk_frames=4, left_frames=25)
    req = SimpleNamespace(external_req_id="r")

    out = talker2code2wav_async_chunk(tm, _sentinel_payload(), req, is_finished=True)

    assert out is not None
    assert bool(out.meta.finished) is True
    # context_length = 6 % 4 = 2; left = min(6-2, 25) = 4; end_index = min(6, 4+2) = 6.
    assert out.meta.left_context_size == 4
    assert isinstance(out.codes.audio, torch.Tensor)
    # 6 single-codebook frames -> flattened length 6.
    assert out.codes.audio.numel() == 6


def test_finish_sentinel_on_chunk_boundary_emits_flag_only():
    # 4 frames, chunk size 4 -> the last full chunk was already sent in-step;
    # no unsent tail, so the sentinel must NOT re-send codec (flag only).
    tm = _tm({"r": [torch.tensor([[i]]) for i in range(1, 5)]}, chunk_frames=4)
    req = SimpleNamespace(external_req_id="r")

    out = talker2code2wav_async_chunk(tm, _sentinel_payload(), req, is_finished=True)

    assert out is not None
    assert bool(out.meta.finished) is True
    assert out.codes is None, "boundary finish must not re-send the last full chunk"


def test_finish_sentinel_with_no_sent_chunks_emits_flag_only():
    tm = _tm({}, chunk_frames=4)
    req = SimpleNamespace(external_req_id="missing")

    out = talker2code2wav_async_chunk(tm, _sentinel_payload(), req, is_finished=True)

    assert out is not None
    assert bool(out.meta.finished) is True
    assert out.codes is None


def test_non_sentinel_empty_call_is_unchanged():
    # Without the marker, an empty/codeless call returns None as before -> the
    # adapter path (which never sets the marker) is byte-identical.
    tm = _tm({"r": [torch.tensor([[1]]), torch.tensor([[2]])]}, chunk_frames=4)
    req = SimpleNamespace(external_req_id="r")

    assert talker2code2wav_async_chunk(tm, {"codes": {}}, req, is_finished=True) is None
    assert talker2code2wav_async_chunk(tm, {}, req, is_finished=True) is None


def test_finish_sentinel_on_initial_chunk_boundary_emits_flag_only():
    tm = _tm({"r": [torch.tensor([[i]]) for i in range(1, 5)]}, chunk_frames=25, initial_frames=4)
    req = SimpleNamespace(external_req_id="r")
    out = talker2code2wav_async_chunk(tm, _sentinel_payload(), req, is_finished=True)

    assert out is not None
    assert bool(out.meta.finished) is True
    assert out.codes is None, "initial-boundary finish must not re-send the initial chunk"


def test_finish_sentinel_flushes_partial_tail_after_initial_chunk():
    tm = _tm({"r": [torch.tensor([[i]]) for i in range(1, 7)]}, chunk_frames=25, initial_frames=4)
    req = SimpleNamespace(external_req_id="r")
    out = talker2code2wav_async_chunk(tm, _sentinel_payload(), req, is_finished=True)

    assert out is not None
    assert bool(out.meta.finished) is True
    assert out.meta.left_context_size == 4
    assert isinstance(out.codes.audio, torch.Tensor)
    assert out.codes.audio.numel() == 6
