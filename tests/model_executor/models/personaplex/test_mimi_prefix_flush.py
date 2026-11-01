# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Opt-in weighted codec integration; no talker, EngineCore or live serving.

Set PERSONAPLEX_MIMI_CHECKPOINT to a locally cached reference Mimi checkpoint.
CUDA cases also require PERSONAPLEX_MIMI_CUDA=1; select the cpu or cuda marker.
This module never downloads weights. Depformer rows are deterministic valid
synthetic tokens, not output from the PersonaPlex 7B talker. The code2wav weight
loader, streaming neural codec and prefix processor are production instances.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest
import pytest_asyncio
import soundfile
import torch
from transformers import PretrainedConfig
from vllm.config import DeviceConfig, VllmConfig

from tests.engine.duplex.test_audio_drain import _generated, _until
from tests.engine.duplex.test_audio_drain_transport import transport as _transport_fixture
from tests.engine.duplex.test_session_runner_personaplex import code2wav_output, frame
from tests.engine.test_codec_prefix_flush import _CodecTransferState, _manager, _request
from tests.entrypoints.duplex.test_audio_send_completion import _next_audio
from vllm_omni.config.model import OmniModelConfig
from vllm_omni.data_entry_keys import OmniPayloadStruct
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.messages import DuplexSessionError
from vllm_omni.entrypoints.duplex.audio_encoding import encode_audio
from vllm_omni.model_executor.models.personaplex.personaplex_code2wav import PersonaPlexCode2Wav
from vllm_omni.model_executor.models.personaplex.personaplex_mimi import FRAME_SIZE, PersonaPlexMimiCodec
from vllm_omni.model_executor.stage_input_processors.personaplex import talker2code2wav_async_chunk
from vllm_omni.request import OmniRequest

pytestmark = [pytest.mark.advanced_model, pytest.mark.omni]
transport = _transport_fixture

_CHECKPOINT_SHA256 = "09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50"


@contextmanager
def _mimi_reference_precision(device: str) -> Generator[None, None, None]:
    """Use FP32 cuDNN convolution for the chunk-vs-frame numerical oracle.

    TF32 may choose differently rounded convolution paths for different chunk
    shapes. This is a test-only reference policy, not a serving configuration.
    Restore just the flag we change, including failed setup or test teardown.
    """
    if device != "cuda":
        yield
        return
    previous_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = previous_tf32


@pytest.fixture(
    scope="module",
    params=[pytest.param("cpu", marks=pytest.mark.cpu), pytest.param("cuda", marks=pytest.mark.cuda)],
)
def weighted_decoder(
    request: pytest.FixtureRequest,
    record_testsuite_property: Callable[[str, object], None],
) -> Generator[PersonaPlexCode2Wav, None, None]:
    device = str(request.param)
    if device == "cuda" and os.environ.get("PERSONAPLEX_MIMI_CUDA") != "1":
        pytest.skip("Set PERSONAPLEX_MIMI_CUDA=1 to opt into CUDA codec integration")
    checkpoint_name = os.environ.get("PERSONAPLEX_MIMI_CHECKPOINT")
    if not checkpoint_name:
        pytest.skip("Set PERSONAPLEX_MIMI_CHECKPOINT to opt into weighted codec integration")
    assert checkpoint_name is not None
    checkpoint = Path(checkpoint_name).resolve(strict=True)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    assert digest.hexdigest() == _CHECKPOINT_SHA256, "Not the pinned reference Mimi checkpoint"
    # Initialize only the local codec-read fields on the real config classes.
    # This is not a validated talker/serving config and never resolves a hub id.
    config = VllmConfig.__new__(VllmConfig)
    model_config = OmniModelConfig.__new__(OmniModelConfig)
    model_config.model = str(checkpoint.parent)
    model_config.hf_config = PretrainedConfig(mimi_name=checkpoint.name)
    model_config.duplex_max_sessions = 1
    # Match the PersonaPlex deployment: this suite feeds new delta frames,
    # not a persistent synchronous full-sequence payload.
    model_config.async_chunk = True
    config.model_config = model_config
    config.device_config = DeviceConfig(device=device)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with _mimi_reference_precision(device):
            model = PersonaPlexCode2Wav(vllm_config=config)
            loaded = model.load_weights([])
            assert loaded and model.mimi is not None
            assert isinstance(model.mimi, PersonaPlexMimiCodec)
            assert model.mimi.device.type == device
            assert all(parameter.device.type == device for parameter in model.mimi.parameters())
            assert model._output_sample_rate == 24000
            assert model._num_codebooks == 8
            # Loading the WAV encoder's lazy serving dependencies is model setup,
            # not part of the short control-harness watchdog. Exercise the real
            # encoder here, before any input is accepted or a drain is admitted.
            setup_started = time.perf_counter()
            model.mimi.streaming_init(1)
            setup_pcm = model.mimi.decode_frame(torch.zeros((1, 8), dtype=torch.long))
            assert setup_pcm.device.type == device
            encoded_setup = encode_audio(setup_pcm.reshape(-1), 24000, "wav", 1.0)
            assert isinstance(encoded_setup, str) and encoded_setup
            model.mimi.reset_streaming()
            record_testsuite_property("weighted_encoder_setup_seconds", time.perf_counter() - setup_started)
            record_testsuite_property("weighted_setup", "Real codec/WAV encoder initialized before accepting inputs")
            record_testsuite_property("mimi_checkpoint_sha256", _CHECKPOINT_SHA256)
            record_testsuite_property("mimi_parameter_count", sum(p.numel() for p in model.mimi.parameters()))
            record_testsuite_property("mimi_device", device)
            record_testsuite_property("mimi_reference_cudnn_allow_tf32", torch.backends.cudnn.allow_tf32)
            record_testsuite_property(
                "weighted_scope",
                f"{device} codec and native queued controls; synthetic talker/StagePort; recording socket",
            )
            yield model.eval()
    finally:
        torch.set_num_threads(previous_threads)


@pytest.fixture
def decoder(weighted_decoder: PersonaPlexCode2Wav) -> Generator[PersonaPlexCode2Wav, None, None]:
    weighted_decoder.on_requests_finished(list(weighted_decoder._request_codec_slots))
    codec = cast(PersonaPlexMimiCodec, weighted_decoder.mimi)
    codec.streaming_init(1)
    try:
        yield weighted_decoder
    finally:
        weighted_decoder.on_requests_finished(list(weighted_decoder._request_codec_slots))
        codec.streaming_init(1)


def _stream(chunk_frames: int = 5) -> tuple[_CodecTransferState, OmniRequest]:
    manager = _manager(chunk_frames)
    request = _request("weighted")
    request.request_id = "weighted"
    return manager, request


def _rows(count: int) -> torch.Tensor:
    return torch.randint(0, 2048, (count, 16), generator=torch.Generator().manual_seed(7530))


def _feed(manager: _CodecTransferState, request: OmniRequest, rows: torch.Tensor) -> list[OmniPayloadStruct]:
    payloads = []
    for row in rows:
        request.additional_information = {"codes": {"audio": row.unsqueeze(0)}}
        payload = talker2code2wav_async_chunk(manager, None, request, is_finished=True)
        if payload is not None:
            payloads.append(payload)
    return payloads


def _decode(
    model: PersonaPlexCode2Wav, payloads: list[OmniPayloadStruct], request_id: str = "weighted"
) -> tuple[torch.Tensor, torch.Tensor]:
    codec = cast(PersonaPlexMimiCodec, model.mimi)
    pcm_chunks, code_chunks = [], []
    for payload in payloads:
        flat = payload.codes.audio if payload.codes is not None else None
        if flat is None or flat.numel() == 0:
            continue
        code_chunks.append(flat.reshape(8, -1))
        output = model(
            input_ids=torch.zeros_like(flat, device=codec.device),
            runtime_additional_information=[{"codes": {"audio": flat}}],
            request_ids=[request_id],
        )
        pcm = output.multimodal_outputs["model_outputs"][0]
        assert pcm.device.type == codec.device.type
        pcm_chunks.append(pcm.detach().clone())
    pcm = torch.cat(pcm_chunks) if pcm_chunks else torch.empty(0, device=codec.device)
    codes = torch.cat(code_chunks, dim=1) if code_chunks else torch.empty((8, 0), dtype=torch.long)
    return pcm, codes


def _aligned_codes(rows: torch.Tensor) -> torch.Tensor:
    # Independent oracle for cb0(t) + cb1..7(t+1), not the processor's helper.
    return torch.cat((rows[:-1, :1], rows[1:, 1:8]), dim=1).T.contiguous()


def _assert_decoded(
    model: PersonaPlexCode2Wav,
    pcm: torch.Tensor,
    codes: torch.Tensor,
    expected_codes: torch.Tensor,
    record_property: Callable[[str, object], None],
) -> None:
    assert torch.equal(codes, expected_codes)
    assert pcm.numel() == expected_codes.shape[1] * FRAME_SIZE
    assert torch.isfinite(pcm).all()
    if pcm.numel() == 0:
        return
    assert pcm.abs().max() > 1e-5  # Real checkpoint output, not zero placeholder PCM.
    codec = cast(PersonaPlexMimiCodec, model.mimi)
    codec.streaming_init(1)
    reference = torch.cat([codec.decode_frame(row.unsqueeze(0)).reshape(-1) for row in expected_codes.T])
    record_property("maximum_pcm_error_vs_native_per_frame", float((pcm - reference).abs().max()))
    torch.testing.assert_close(pcm, reference, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("accepted_prefix", [1, 2, 4, 7, 19])
def test_weighted_prefix_flush_preserves_pcm_and_resumes_tail(
    decoder: PersonaPlexCode2Wav,
    accepted_prefix: int,
    record_property: Callable[[str, object], None],
) -> None:
    manager, request = _stream()
    rows = _rows(accepted_prefix + 3)
    payloads = _feed(manager, request, rows[:accepted_prefix])
    prefix = talker2code2wav_async_chunk(manager, None, request, flush_prefix=accepted_prefix)
    if prefix is not None:
        assert not prefix.meta.finished.item()
        payloads.append(prefix)
    first_pcm, first_codes = _decode(decoder, payloads)
    assert first_pcm.numel() == max(accepted_prefix - 1, 0) * FRAME_SIZE
    assert len(manager.request_payload[request.external_req_id]["personaplex_frames"]) == 1
    assert talker2code2wav_async_chunk(manager, None, request, flush_prefix=accepted_prefix) is None

    continuation = _feed(manager, request, rows[accepted_prefix:])
    final = talker2code2wav_async_chunk(manager, None, request, flush_prefix=len(rows))
    if final is not None:
        continuation.append(final)
    next_pcm, next_codes = _decode(decoder, continuation)
    assert next_pcm.numel() == 3 * FRAME_SIZE
    _assert_decoded(
        decoder,
        torch.cat((first_pcm, next_pcm)),
        torch.cat((first_codes, next_codes), dim=1),
        _aligned_codes(rows),
        record_property,
    )


def test_weighted_frozen_prefix_does_not_decode_later_buffered_rows(
    decoder: PersonaPlexCode2Wav, record_property: Callable[[str, object], None]
) -> None:
    manager, request = _stream(chunk_frames=25)
    rows = _rows(5)
    payloads = _feed(manager, request, rows)
    early = talker2code2wav_async_chunk(manager, None, request, flush_prefix=3)
    assert early is not None
    early_pcm, early_codes = _decode(decoder, [*payloads, early])
    assert early_pcm.numel() == 2 * FRAME_SIZE
    assert len(manager.request_payload[request.external_req_id]["personaplex_frames"]) == 3
    later = talker2code2wav_async_chunk(manager, None, request, flush_prefix=5)
    assert later is not None
    later_pcm, later_codes = _decode(decoder, [later])
    assert later_pcm.numel() == 2 * FRAME_SIZE
    _assert_decoded(
        decoder,
        torch.cat((early_pcm, later_pcm)),
        torch.cat((early_codes, later_codes), dim=1),
        _aligned_codes(rows),
        record_property,
    )


@pytest.mark.parametrize("accepted", [1, 4, 9])
def test_weighted_terminal_eof_drops_only_missing_successor(
    decoder: PersonaPlexCode2Wav,
    accepted: int,
    record_property: Callable[[str, object], None],
) -> None:
    manager, request = _stream()
    rows = _rows(accepted)
    payloads = _feed(manager, request, rows)
    request.resumable = False
    request.additional_information = None
    terminal = talker2code2wav_async_chunk(manager, None, request, is_finished=True)
    assert terminal is not None and terminal.meta.finished.item()
    pcm, codes = _decode(decoder, [*payloads, terminal])
    assert request.external_req_id not in manager.request_payload
    _assert_decoded(decoder, pcm, codes, _aligned_codes(rows), record_property)


def test_weighted_finished_request_reuses_clean_codec_state(
    decoder: PersonaPlexCode2Wav, record_property: Callable[[str, object], None]
) -> None:
    manager, request = _stream()
    rows = _rows(4)
    payloads = _feed(manager, request, rows)
    flushed = talker2code2wav_async_chunk(manager, None, request, flush_prefix=4)
    assert flushed is not None
    payloads.append(flushed)
    first, codes = _decode(decoder, payloads, "old-request")
    decoder.on_requests_finished(["old-request"])
    assert "old-request" not in decoder._consumed_full_payload_requests
    assert "old-request" not in decoder._request_codec_slots
    second, next_codes = _decode(decoder, payloads, "new-request")
    assert torch.equal(codes, next_codes)
    assert torch.equal(first, second)
    _assert_decoded(decoder, second, next_codes, _aligned_codes(rows), record_property)


def test_weighted_slot_reset_does_not_corrupt_another_active_stream(decoder: PersonaPlexCode2Wav) -> None:
    codec = cast(PersonaPlexMimiCodec, decoder.mimi)
    codes = _rows(5)[:, :8]
    active = torch.tensor([False, True])
    codec.streaming_init(2)
    reference = [codec.decode_frame(torch.stack((row, row)), active=active)[1].clone() for row in codes]
    codec.streaming_init(2)
    for index, row in enumerate(codes):
        if index == 2:
            codec.reset_slot(0)
        batch = torch.stack((codes[-1 - index], row))
        actual = codec.decode_frame(batch, active=torch.tensor([True, True]))[1]
        torch.testing.assert_close(actual, reference[index], rtol=2e-4, atol=2e-4)


@pytest_asyncio.fixture
async def weighted_transport(transport, decoder, mocker, monkeypatch):
    """Real queued controls, codec and WAV encoder; StagePort remains a bridge.

    The recorded StagePort flush invokes the real prefix processor and decoder,
    then hands their actual PCM to the native runner. No EngineCore process or
    GPU worker stage is created, and no socket bytes leave the recording attachment.
    CUDA codec output is copied to the CPU at the same PCM handoff boundary.
    """
    native = transport
    decoded: list[torch.Tensor] = []

    async def prepare() -> None:
        monkeypatch.setattr(native.h.runner.plugin.data_plane, "_encode_audio", encode_audio)
        manager, request = _stream()
        request.request_id = request.external_req_id = native.h.stage0_request_id()
        pending = _feed(manager, request, _rows(4))

        async def flush(request_id: str, *, epoch: int, sequence: int) -> int:
            assert request_id == request.external_req_id and (epoch, sequence) == (0, 4)
            prefix = talker2code2wav_async_chunk(manager, None, request, flush_prefix=sequence)
            assert prefix is not None
            for payload in [*pending, prefix]:
                pcm, _ = await asyncio.to_thread(_decode, decoder, [payload], request_id)
                if pcm.numel() == 0:
                    continue
                pcm = pcm.cpu()
                decoded.append(pcm)
                output = code2wav_output(request_id, samples=pcm.numel(), text="")
                output.multimodal_output["audio"] = pcm.numpy()
                native.h.deliver(output)
            return sequence

        mocker.patch.object(native.h.port, "flush_audio_prefix", new=flush)
        for _ in range(4):
            await native.h.run(frame())

    await native.backend(prepare())
    yield native, decoded


async def _weighted_generation(native) -> None:
    for sequence in range(1, 5):
        _generated(native.h, sequence)
    await _until(lambda: native.h.session.audio_delivery.projected_samples == 3 * FRAME_SIZE)


@pytest.mark.asyncio
async def test_weighted_native_drain_requires_every_successful_encoded_send(weighted_transport, mocker) -> None:
    native, decoded = weighted_transport
    drain = asyncio.create_task(native.handle.drain_audio(timeout=10))
    started, release = asyncio.Event(), asyncio.Event()
    sent: list[object] = []
    sending = None

    async def send(payload) -> None:
        if sent:  # Hold the final audio chunk after the first send completed.
            started.set()
            await release.wait()
        sent.append(payload)

    try:
        await native.pending()
        await native.backend(_weighted_generation(native))
        await native.handler._attachment_registry.create(native.handle.session_id, send=send, close=mocker.AsyncMock())
        first = await _next_audio(native.h.output_buffer)
        assert [pcm.numel() for pcm in decoded] == [FRAME_SIZE, 2 * FRAME_SIZE]
        audio_bytes = first.audio
        assert isinstance(audio_bytes, bytes)
        pcm, rate = soundfile.read(io.BytesIO(audio_bytes), dtype="float32")
        assert rate == 24000 and len(pcm) == decoded[0].numel()
        torch.testing.assert_close(torch.from_numpy(pcm), decoded[0].clamp(-1, 1), rtol=0, atol=4e-5)
        await native.handler._send_event(native.handle.session_id, first, handle=native.handle)

        async def first_sent() -> None:
            await _until(lambda: native.h.session.audio_delivery.completed_samples == FRAME_SIZE)

        await native.backend(first_sent())
        assert not drain.done()
        # The shared buffer owns one consumer-held event; do not dequeue the
        # next event until the previous event's send receipt has been reported.
        second = await _next_audio(native.h.output_buffer)
        audio_bytes = second.audio
        assert isinstance(audio_bytes, bytes)
        pcm, rate = soundfile.read(io.BytesIO(audio_bytes), dtype="float32")
        assert rate == 24000 and len(pcm) == decoded[1].numel()
        torch.testing.assert_close(torch.from_numpy(pcm), decoded[1].clamp(-1, 1), rtol=0, atol=4e-5)
        sending = asyncio.create_task(
            native.handler._send_event(native.handle.session_id, second, handle=native.handle)
        )
        await asyncio.wait_for(started.wait(), 3)
        assert not drain.done()
        release.set()
        await sending
        target = await drain
        assert (target.accepted_seq, target.expected_samples) == (4, 3 * FRAME_SIZE)
        assert native.h.session.audio_delivery.completed_samples == 3 * FRAME_SIZE
        assert native.h.session.playback.played_ms == 0
        assert not native.engine.rpc_client._router._pending
    finally:
        release.set()
        if sending is not None:
            await asyncio.gather(sending, return_exceptions=True)
        drain.cancel()
        await asyncio.gather(drain, return_exceptions=True)


@pytest.mark.asyncio
async def test_weighted_native_invalidated_send_cannot_complete_drain(weighted_transport, mocker) -> None:
    native, decoded = weighted_transport
    drain = asyncio.create_task(native.handle.drain_audio(timeout=10))
    started, release = asyncio.Event(), asyncio.Event()
    sending = None

    async def send(_payload) -> None:
        started.set()
        await release.wait()

    try:
        await native.pending()
        await native.backend(_weighted_generation(native))
        assert sum(pcm.numel() for pcm in decoded) == 3 * FRAME_SIZE
        old_state = native.h.session.audio_delivery
        event = await _next_audio(native.h.output_buffer)
        await native.handler._attachment_registry.create(native.handle.session_id, send=send, close=mocker.AsyncMock())
        sending = asyncio.create_task(native.handler._send_event(native.handle.session_id, event, handle=native.handle))
        await asyncio.wait_for(started.wait(), 3)
        assert not drain.done()
        await native.backend(native.h.run(commands.CancelResponse()))
        release.set()
        await sending
        with pytest.raises(DuplexSessionError, match="invalidated"):
            await drain
        assert old_state.completed_samples == 0
        assert native.h.session.epoch == 1
        assert not native.engine.rpc_client._router._pending
    finally:
        release.set()
        if sending is not None:
            await asyncio.gather(sending, return_exceptions=True)
        drain.cancel()
        await asyncio.gather(drain, return_exceptions=True)
