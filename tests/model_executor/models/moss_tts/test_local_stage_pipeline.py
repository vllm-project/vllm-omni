# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact MOSS-TTS Local Stage-0 regression coverage."""

from __future__ import annotations

import functools
from collections import defaultdict
from types import MethodType, SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import (  # noqa: E402
    MossTTSCodecDecoder,
    _moss_codec_codes_from_payload_or_input,
)
from vllm_omni.model_executor.stage_input_processors.moss_tts import (  # noqa: E402
    _MOSS_AUDIO_PAD_CODE,
    talker2codec_raw_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

N_VQ = 4
HIDDEN = 8


class _FakeCodec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(codebook_size=1024)
        self.downsample_rate = 2
        self.decoded_codes: list[torch.Tensor] = []

    def batch_decode(self, codes_list: list[torch.Tensor], num_quantizers: int | None = None) -> SimpleNamespace:
        del num_quantizers
        self.decoded_codes.append(codes_list[0].detach().cpu().clone())
        t = int(codes_list[0].shape[1])
        audio = torch.arange(t * self.downsample_rate, dtype=torch.float32).reshape(1, 1, -1)
        lengths = torch.tensor([audio.shape[-1]], dtype=torch.long)
        return SimpleNamespace(audio=audio, audio_lengths=lengths)


def _decoder() -> MossTTSCodecDecoder:
    decoder = MossTTSCodecDecoder.__new__(MossTTSCodecDecoder)
    nn.Module.__init__(decoder)
    decoder.vllm_config = SimpleNamespace()
    decoder._n_vq = N_VQ
    decoder._codec = _FakeCodec()
    decoder._cuda_graph_wrapper = None
    decoder._n_channels = 1
    decoder._sr_tensor = torch.tensor(24_000, dtype=torch.int32)
    decoder._stream_session = None
    decoder._stream_slots = 0
    decoder._stream_max_step_frames = 100
    decoder._stream_req_slots = {}
    decoder._stream_pending_codes = {}
    decoder._stream_starved_reqs = set()
    return decoder


def _codec_info(
    audio: Any,
    *,
    streaming: bool = False,
    finished: bool = False,
    code_flat_numel: int | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "codec_streaming": streaming,
        "stream_finished": torch.tensor(finished, dtype=torch.bool),
        "finished": torch.tensor(finished, dtype=torch.bool),
        "req_id": ["r"],
    }
    if code_flat_numel is not None:
        meta["code_flat_numel"] = code_flat_numel
    return {"codes": {"audio": audio}, "meta": meta}


@functools.lru_cache(maxsize=1)
def _local_talker_symbols():
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import (
        MossTTSLocalTalkerForGeneration,
        _moss_local_materialize_history,
    )

    return MossTTSLocalTalkerForGeneration, _moss_local_materialize_history


class _FakeLocalTransformer:
    def __init__(self, frames: list[torch.Tensor], continues: list[bool] | None = None) -> None:
        self.frames = frames
        self.continues = continues if continues is not None else [True] * len(frames)
        self.histories: list[list[list[int]]] = []

    def generate_frame(
        self,
        last_talker_hidden: torch.Tensor,
        audio_lm_heads: Any,
        audio_embeddings: Any,
        local_text_lm_head: Any,
        *,
        n_vq: int,
        history_per_codebook: list[list[int]],
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del audio_lm_heads, audio_embeddings, local_text_lm_head, n_vq
        self.histories.append([list(values) for values in history_per_codebook])
        index = len(self.histories) - 1
        frame = self.frames[index].to(device=last_talker_hidden.device, dtype=torch.long).reshape(1, -1)
        should_continue = torch.tensor([self.continues[index]], device=last_talker_hidden.device)
        return should_continue, frame


def _make_bare_local_talker(frames: list[torch.Tensor], continues: list[bool] | None = None):
    cls, _ = _local_talker_symbols()
    talker = cls.__new__(cls)
    nn.Module.__init__(talker)
    talker.n_vq = N_VQ
    talker.hidden_size = HIDDEN
    talker.local_transformer = _FakeLocalTransformer(frames, continues)
    talker.audio_lm_heads = nn.ModuleList()
    talker.audio_embeddings = nn.ModuleList()
    talker.local_text_lm_head = nn.Linear(HIDDEN, 2, bias=False)
    talker._batch_state = None
    talker._batch_state_spans = None

    def _audio_embed(codes: torch.Tensor) -> torch.Tensor:
        return torch.zeros((codes.shape[0], HIDDEN), device=codes.device, dtype=torch.float32)

    talker._audio_embed = _audio_embed
    return talker


def _run_talker_step(talker: Any, info: dict[str, Any]) -> None:
    input_embeds = torch.zeros((1, HIDDEN), dtype=torch.float32)
    last_hidden = torch.zeros((1, HIDDEN), dtype=torch.float32)
    input_ids = torch.zeros((1,), dtype=torch.long)
    text_step = torch.zeros((1,), dtype=torch.long)
    talker.talker_mtp(input_ids, input_embeds, last_hidden, text_step, req_infos=[info])


def _history_tensor(info: dict[str, Any]) -> torch.Tensor:
    _, materialize = _local_talker_symbols()
    return materialize(info["audio_codes"]["accumulated"])


def _tm(*, chunk_frames: int = 3, initial_chunk_frames: int = 0):
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        put_req_chunk=defaultdict(int),
        request_payload={},
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": chunk_frames,
                    "initial_codec_chunk_frames": initial_chunk_frames,
                }
            }
        ),
    )


def _req(rid: str):
    return SimpleNamespace(external_req_id=rid, request_id=rid)


def _payload(frames: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
    return {"codes": {"audio": frames}}


def _assert_native_chunk(payload: Any, expected_frames: torch.Tensor, *, finished: bool = False) -> None:
    assert payload is not None
    assert isinstance(payload.codes.audio, torch.Tensor)
    assert payload.codes.audio.shape == (N_VQ, expected_frames.shape[0])
    assert payload.meta.code_flat_numel == N_VQ * expected_frames.shape[0]
    assert payload.meta.stream_finished.item() is finished
    assert torch.equal(payload.codes.audio, expected_frames.transpose(0, 1).contiguous())


def _assert_control_sentinel(payload: Any) -> None:
    assert payload is not None
    assert torch.equal(payload.codes.audio, torch.tensor([0], dtype=torch.long))
    assert payload.codes.audio.shape == (1,)
    assert payload.meta.codec_chunk_frames == 0
    assert payload.meta.code_flat_numel == 0
    assert payload.meta.stream_finished.item() is True


def test_codec_decoder_accepts_connector_payload_layouts_and_control_sentinel() -> None:
    native = torch.arange(N_VQ * 3, dtype=torch.long).reshape(N_VQ, 3)
    frame_major = torch.tensor([[0, 1, 2, 3], [10, 11, 12, 13]], dtype=torch.long)
    flat = torch.arange(N_VQ * 2, dtype=torch.long)

    cases = [
        (_codec_info(native), native),
        (_codec_info(flat), flat.reshape(N_VQ, 2)),
        (_codec_info(list(range(N_VQ * 2))), flat.reshape(N_VQ, 2)),
        ({"meta": {}}, flat.reshape(N_VQ, 2)),
    ]
    for info, expected in cases:
        parsed = _moss_codec_codes_from_payload_or_input(flat, info, N_VQ, "cpu")
        assert parsed is not None
        assert parsed.dtype == torch.long
        assert parsed.device.type == "cpu"
        assert torch.equal(parsed, expected)

    assert _moss_codec_codes_from_payload_or_input(flat, _codec_info(frame_major), N_VQ, "cpu") is None

    decoder = _decoder()
    out = decoder.forward(
        input_ids=torch.tensor([0], dtype=torch.long),
        runtime_additional_information=[_codec_info(native)],
        seq_token_counts=[1],
    )
    assert torch.equal(decoder._codec.decoded_codes[0], native)
    assert out.multimodal_outputs["model_outputs"][0].shape == (6,)
    assert int(out.multimodal_outputs["sr"][0].item()) == 24_000

    streaming_decoder = _decoder()
    captured: dict[str, Any] = {}

    def _fake_decode_streaming_batch(self: MossTTSCodecDecoder, items: list[tuple[int, str, torch.Tensor, bool]]):
        del self
        captured["items"] = items
        return {0: torch.tensor([1.0, 2.0])}

    streaming_decoder._decode_streaming_batch = MethodType(_fake_decode_streaming_batch, streaming_decoder)
    streaming_out = streaming_decoder.forward(
        input_ids=torch.tensor([0], dtype=torch.long),
        runtime_additional_information=[_codec_info(native[:, :2], streaming=True)],
        seq_token_counts=[1],
    )
    [(index, req_key, passed_codes, finished)] = captured["items"]
    assert index == 0
    assert req_key == "r"
    assert finished is False
    assert torch.equal(passed_codes.cpu(), native[:, :2])
    assert torch.equal(streaming_out.multimodal_outputs["model_outputs"][0], torch.tensor([1.0, 2.0]))
    assert streaming_decoder._codec.decoded_codes == []

    sentinel_decoder = _decoder()
    sentinel_out = sentinel_decoder.forward(
        input_ids=torch.tensor([0], dtype=torch.long),
        runtime_additional_information=[
            _codec_info(torch.tensor([0], dtype=torch.long), streaming=True, finished=True, code_flat_numel=0)
        ],
        seq_token_counts=[1],
    )
    assert sentinel_decoder._codec.decoded_codes == []
    assert sentinel_out.multimodal_outputs["model_outputs"][0].numel() == 0


def test_local_talker_uses_cpu_history_buffer_without_hot_path_cat(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = [torch.arange(N_VQ) + offset for offset in (0, 10, 20)]
    talker = _make_bare_local_talker(frames)
    info: dict[str, Any] = {"audio_state": {"step": 0, "max_new_frames": -1}}

    def _fail_cat(*args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        raise AssertionError("torch.cat must not be used for Local Stage-0 audio history")

    monkeypatch.setattr(torch, "cat", _fail_cat)
    for _ in frames:
        _run_talker_step(talker, info)

    accumulated = info["audio_codes"]["accumulated"]
    assert isinstance(accumulated, dict)
    assert accumulated["length"] == 3
    assert accumulated["buffer"].device.type == "cpu"
    assert accumulated["buffer"].shape == (64, N_VQ)
    assert torch.equal(_history_tensor(info), torch.stack(frames))

    output_talker = _make_bare_local_talker([])
    hidden = torch.zeros((1, HIDDEN), dtype=torch.float32)
    prior_history = {"buffer": torch.arange(3 * N_VQ, dtype=torch.long).reshape(3, N_VQ), "length": 3}
    current = torch.tensor([[40, 41, 42, 43]], dtype=torch.long)
    result = output_talker.make_omni_output(
        hidden,
        runtime_additional_information=[
            {"audio_codes": {"accumulated": prior_history, "current": current, "emit": True}}
        ],
        request_token_spans=[(0, 1)],
    )
    assert torch.equal(result.multimodal_outputs["codes"]["audio"][0], current)


def test_local_talker_converts_legacy_history_and_passes_last_50_frames() -> None:
    legacy = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
    frame = torch.tensor([9, 10, 11, 12], dtype=torch.long)
    talker = _make_bare_local_talker([frame])
    info: dict[str, Any] = {
        "audio_state": {"step": 0, "max_new_frames": -1},
        "audio_codes": {"accumulated": legacy},
    }

    _run_talker_step(talker, info)

    assert isinstance(info["audio_codes"]["accumulated"], dict)
    assert torch.equal(_history_tensor(info), torch.cat([legacy, frame.reshape(1, -1)], dim=0))

    long_history = torch.arange(55 * N_VQ, dtype=torch.long).reshape(55, N_VQ)
    stop_frame = torch.full((N_VQ,), 999, dtype=torch.long)
    stop_talker = _make_bare_local_talker([stop_frame], continues=[False])
    stop_info: dict[str, Any] = {
        "audio_state": {"step": 0, "max_new_frames": -1},
        "audio_codes": {"accumulated": long_history},
    }

    _run_talker_step(stop_talker, stop_info)

    expected_tail = long_history[-50:]
    expected_history = [[int(row[cb].item()) for row in expected_tail] for cb in range(N_VQ)]
    assert stop_talker.local_transformer.histories == [expected_history]
    assert torch.equal(_history_tensor(stop_info), long_history)


def test_local_talker_stop_conditions_emit_empty_current_without_appending() -> None:
    for should_continue, audio_state in [
        (False, {"step": 0, "max_new_frames": -1}),
        (True, {"step": 1, "max_new_frames": 1}),
    ]:
        legacy = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        next_frame = torch.tensor([9, 9, 9, 9], dtype=torch.long)
        talker = _make_bare_local_talker([next_frame], continues=[should_continue])
        info: dict[str, Any] = {
            "audio_state": dict(audio_state),
            "audio_codes": {"accumulated": legacy},
        }

        _run_talker_step(talker, info)

        assert torch.equal(_history_tensor(info), legacy)
        assert info["audio_codes"]["emit"] is False
        assert info["audio_codes"]["current"].shape == (0, N_VQ)
        assert info["audio_state"]["is_stopping"] is True


def test_raw_chunking_emits_native_layout_and_uses_initial_then_steady_threshold() -> None:
    initial_tm = _tm(chunk_frames=3, initial_chunk_frames=1)
    req = _req("r")
    first_frame = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    first_payload = talker2codec_raw_async_chunk(initial_tm, _payload(first_frame), req)
    _assert_native_chunk(first_payload, first_frame)
    assert first_payload.meta.codec_chunk_frames == 1

    assert (
        talker2codec_raw_async_chunk(initial_tm, _payload(torch.tensor([[10, 11, 12, 13]], dtype=torch.long)), req)
        is None
    )
    assert (
        talker2codec_raw_async_chunk(initial_tm, _payload(torch.tensor([[20, 21, 22, 23]], dtype=torch.long)), req)
        is None
    )
    fourth_frame = torch.tensor([[30, 31, 32, 33]], dtype=torch.long)
    steady_after_initial = talker2codec_raw_async_chunk(initial_tm, _payload(fourth_frame), req)
    _assert_native_chunk(
        steady_after_initial,
        torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]], dtype=torch.long),
    )
    assert initial_tm._moss_tts_raw_chunk_states["r"].emitted_frame_count == 4

    steady_tm = _tm(chunk_frames=3)
    steady_frames = torch.tensor([[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.long)
    steady_payload = talker2codec_raw_async_chunk(steady_tm, _payload(steady_frames), _req("steady"))
    _assert_native_chunk(steady_payload, steady_frames)
    assert steady_payload.meta.codec_chunk_frames == 3

    pending_tm = _tm(chunk_frames=3)
    pending_req = _req("pending")
    assert (
        talker2codec_raw_async_chunk(pending_tm, _payload(torch.tensor([[0, 1, 2, 3]], dtype=torch.long)), pending_req)
        is None
    )
    assert (
        talker2codec_raw_async_chunk(
            pending_tm, _payload(torch.tensor([[10, 11, 12, 13]], dtype=torch.long)), pending_req
        )
        is None
    )
    state = pending_tm._moss_tts_raw_chunk_states["pending"]
    assert state.emitted_frame_count == 0
    assert len(state.pending_frames) == 2
    pending_payload = talker2codec_raw_async_chunk(
        pending_tm,
        _payload(torch.tensor([[20, 21, 22, 23]], dtype=torch.long)),
        pending_req,
    )
    _assert_native_chunk(
        pending_payload,
        torch.tensor([[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.long),
    )
    assert state.pending_frames == []


def test_raw_chunking_finish_flushes_pending_or_emits_control_sentinel_once() -> None:
    immediate_tm = _tm(chunk_frames=3)
    immediate_frames = torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]], dtype=torch.long)
    immediate_payload = talker2codec_raw_async_chunk(
        immediate_tm, _payload(immediate_frames), _req("immediate"), is_finished=True
    )
    _assert_native_chunk(immediate_payload, immediate_frames, finished=True)
    assert immediate_payload.meta.finished.item() is True
    assert "immediate" not in immediate_tm.code_prompt_token_ids

    flush_tm = _tm(chunk_frames=3)
    flush_req = _req("flush")
    assert talker2codec_raw_async_chunk(flush_tm, _payload(immediate_frames), flush_req) is None
    flush_payload = talker2codec_raw_async_chunk(flush_tm, None, flush_req, is_finished=True)
    repeat_flush = talker2codec_raw_async_chunk(flush_tm, None, flush_req, is_finished=True)
    _assert_native_chunk(flush_payload, immediate_frames, finished=True)
    assert repeat_flush is None
    flush_state = flush_tm._moss_tts_raw_chunk_states["flush"]
    assert flush_state.final_flushed is True
    assert flush_state.sent_control_sentinel is False

    sentinel_tm = _tm(chunk_frames=3)
    sentinel_req = _req("sentinel")
    sentinel = talker2codec_raw_async_chunk(sentinel_tm, None, sentinel_req, is_finished=True)
    repeat_sentinel = talker2codec_raw_async_chunk(sentinel_tm, None, sentinel_req, is_finished=True)
    _assert_control_sentinel(sentinel)
    assert repeat_sentinel is None
    sentinel_state = sentinel_tm._moss_tts_raw_chunk_states["sentinel"]
    assert sentinel_state.final_flushed is True
    assert sentinel_state.sent_control_sentinel is True

    completed_tm = _tm(chunk_frames=3)
    completed_req = _req("completed")
    complete_frames = torch.tensor([[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.long)
    chunk = talker2codec_raw_async_chunk(completed_tm, _payload(complete_frames), completed_req)
    finish = talker2codec_raw_async_chunk(completed_tm, None, completed_req, is_finished=True)
    repeat_finish = talker2codec_raw_async_chunk(completed_tm, None, completed_req, is_finished=True)
    _assert_native_chunk(chunk, complete_frames)
    _assert_control_sentinel(finish)
    assert repeat_finish is None


def test_raw_chunking_filters_pad_rows_reuses_finalized_state_and_prunes_tombstones() -> None:
    pad_tm = _tm(chunk_frames=3)
    pad_req = _req("pad")
    pad_rows = torch.full((3, N_VQ), _MOSS_AUDIO_PAD_CODE, dtype=torch.long)
    assert talker2codec_raw_async_chunk(pad_tm, _payload(pad_rows), pad_req) is None
    assert pad_tm._moss_tts_raw_chunk_states["pad"].pending_frames == []
    _assert_control_sentinel(talker2codec_raw_async_chunk(pad_tm, None, pad_req, is_finished=True))

    reuse_tm = _tm(chunk_frames=3)
    reuse_req = _req("reuse")
    sentinel = talker2codec_raw_async_chunk(reuse_tm, None, reuse_req, is_finished=True)
    _assert_control_sentinel(sentinel)
    old_state = reuse_tm._moss_tts_raw_chunk_states["reuse"]

    first_reused_frame = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    assert talker2codec_raw_async_chunk(reuse_tm, _payload(first_reused_frame), reuse_req) is None
    new_state = reuse_tm._moss_tts_raw_chunk_states["reuse"]
    assert new_state is not old_state
    assert new_state.final_flushed is False
    assert new_state.sent_control_sentinel is False
    assert new_state.emitted_frame_count == 0
    assert len(new_state.pending_frames) == 1

    reused_tail = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.long)
    reused_chunk = talker2codec_raw_async_chunk(reuse_tm, _payload(reused_tail), reuse_req)
    _assert_native_chunk(
        reused_chunk,
        torch.tensor([[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.long),
    )
    assert reused_chunk.meta.stream_finished.item() is False
    assert new_state.emitted_frame_count == 3
    assert new_state.pending_frames == []

    prune_tm = _tm(chunk_frames=3)
    for i in range(4097):
        assert talker2codec_raw_async_chunk(prune_tm, None, _req(f"r{i}"), is_finished=True) is not None

    states = prune_tm._moss_tts_raw_chunk_states
    assert len(states) == 4096
    assert "r0" not in states
    assert states["r1"].final_flushed is True
