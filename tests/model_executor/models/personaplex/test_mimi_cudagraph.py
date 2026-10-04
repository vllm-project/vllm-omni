# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fixed-buffer streaming state and CUDA graph replay of the PersonaPlex Mimi codec."""

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.personaplex.personaplex_code2wav import _MIMI_DECODE_BATCH_FRAMES
from vllm_omni.model_executor.models.personaplex.personaplex_mimi import (
    CODEBOOKS,
    FRAME_SIZE,
    PersonaPlexMimiCodec,
    _MimiStreamingTransformer,
    _StreamConv1d,
    _StreamConvTr1d,
)

pytestmark = pytest.mark.core_model

SEED = 4321


def _mask(*rows: bool, device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.bool, device=device)


def _random_codec(device: torch.device, batch_size: int) -> PersonaPlexMimiCodec:
    """The real Mimi graph with random weights (no checkpoint download)."""
    from transformers import MimiConfig, MimiModel

    torch.manual_seed(SEED)
    codec = PersonaPlexMimiCodec.__new__(PersonaPlexMimiCodec)
    nn.Module.__init__(codec)
    codec.device = device
    model = MimiModel(MimiConfig())
    del model.encoder_transformer, model.decoder_transformer
    with torch.no_grad():
        for name, buffer in model.quantizer.named_buffers():
            if name.endswith("embed_sum"):
                buffer.normal_()
    codec.model = model.to(device).eval()
    codec.dtype = torch.float32
    codec.encoder_transformer = _MimiStreamingTransformer().to(device)
    codec.decoder_transformer = _MimiStreamingTransformer().to(device)
    for transformer in (codec.encoder_transformer, codec.decoder_transformer):
        for parameter in transformer.parameters():
            nn.init.normal_(parameter, std=0.02)
    codec._init_streaming_stages()
    codec.streaming_init(batch_size)
    return codec


def _state_tensors(codec: PersonaPlexMimiCodec) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    for state in codec._conv_states():
        tensors.append(state.prev if isinstance(state, _StreamConv1d) else state.partial)
        tensors.append(state._fresh)
    for transformer in (codec.encoder_transformer, codec.decoder_transformer):
        tensors.append(transformer._offset)
        for kv in transformer._kv:
            tensors.extend((kv.cache, kv.end_offset, kv.start_offset))
    return tensors


def _snapshot(codec: PersonaPlexMimiCodec) -> list[torch.Tensor]:
    return [tensor.clone() for tensor in _state_tensors(codec)]


@pytest.mark.cpu
@pytest.mark.parametrize("kind", ["conv", "conv_replicate", "convtr"])
def test_stream_carries_update_in_place(kind: str) -> None:
    torch.manual_seed(SEED)
    if kind == "convtr":
        stream = _StreamConvTr1d(nn.ConvTranspose1d(2, 3, kernel_size=4, stride=2))
        samples = 3
    else:
        pad_mode = "replicate" if kind == "conv_replicate" else "constant"
        stream = _StreamConv1d(nn.Conv1d(2, 3, kernel_size=5, stride=2), pad_mode=pad_mode)
        samples = 6
    stream.reset(3, torch.device("cpu"), torch.float32)
    carry = stream.prev if isinstance(stream, _StreamConv1d) else stream.partial
    fresh = stream._fresh
    pointers = (carry.data_ptr(), fresh.data_ptr())

    with torch.no_grad():
        for active in (_mask(True, False, True), _mask(False, True, True), _mask(True, True, True)):
            stream(torch.randn(3, 2, samples), active)
            current = stream.prev if isinstance(stream, _StreamConv1d) else stream.partial
            assert current is carry and stream._fresh is fresh
            assert (current.data_ptr(), stream._fresh.data_ptr()) == pointers
    assert carry.abs().sum() > 0
    assert not fresh.any()

    stream.reset_slot(1)
    assert not carry[1].any() and fresh[1]
    assert carry[0].abs().sum() > 0 and not fresh[0]

    stream.reset_all()
    assert not carry.any() and fresh.all()
    assert (carry.data_ptr(), fresh.data_ptr()) == pointers


@pytest.mark.cpu
def test_streaming_init_at_same_batch_size_resets_in_place() -> None:
    codec = _random_codec(torch.device("cpu"), batch_size=2)
    pcm = torch.randn(2, FRAME_SIZE)
    first = codec.encode_frame(pcm)
    codec.decode_frame(first)
    pointers = [tensor.data_ptr() for tensor in _state_tensors(codec)]

    codec.streaming_init(2)

    assert pointers == [tensor.data_ptr() for tensor in _state_tensors(codec)]
    assert torch.equal(codec.encode_frame(pcm), first)

    codec.streaming_init(3)
    assert codec._batch_size == 3
    assert codec.encode_frame(torch.randn(3, FRAME_SIZE)).shape == (3, CODEBOOKS)


@pytest.mark.cpu
def test_capture_off_cuda_stays_eager() -> None:
    codec = _random_codec(torch.device("cpu"), batch_size=2)
    assert codec.capture_cuda_graphs() == []
    assert codec._cuda_graphs == {}
    assert codec.encode_frame(torch.randn(2, FRAME_SIZE)).shape == (2, CODEBOOKS)


@pytest.mark.cpu
def test_quantizer_decode_matches_transformers() -> None:
    codec = _random_codec(torch.device("cpu"), batch_size=2)
    codes = torch.randint(0, 2048, (2, CODEBOOKS, 3))
    with torch.no_grad():
        expected = codec.model.quantizer.decode(codes)
        actual = codec._quantizer_decode(codes)
    assert torch.equal(actual, expected)


def _drive(codec: PersonaPlexMimiCodec, frames: int, recycle_at: int, device: torch.device):
    """Encode/decode ``frames`` frames under a mixed active schedule with a row recycle."""
    batch_size = codec._batch_size
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    schedule = [_mask(*[(step + row) % 3 != 0 for row in range(batch_size)], device=device) for step in range(frames)]
    codes_out, pcm_out, inactive_ok = [], [], True
    for step, active in enumerate(schedule):
        if step == recycle_at:
            codec.reset_slot(1)
        pcm = 0.1 * torch.randn(batch_size, FRAME_SIZE, generator=generator)
        before = _snapshot(codec)
        codes = codec.encode_frame(pcm, active)
        wav = codec.decode_frame(codes, active)
        after = _state_tensors(codec)
        for row, is_active in enumerate(active.tolist()):
            if is_active:
                continue
            for old, new in zip(before, after):
                if new.dim() > 0 and new.shape[0] == batch_size:
                    inactive_ok &= torch.equal(old[row], new[row])
                elif new.dim() > 1 and new.shape[1] == batch_size:
                    inactive_ok &= torch.equal(old[:, row], new[:, row])
        codes_out.append(codes)
        pcm_out.append(wav)
    return torch.stack(codes_out), torch.stack(pcm_out), inactive_ok


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_replay_is_bitwise_equal_to_eager_across_recycle() -> None:
    device = torch.device("cuda")
    codec = _random_codec(device, batch_size=3)
    eager_codes, eager_pcm, eager_inactive_ok = _drive(codec, frames=12, recycle_at=7, device=device)
    assert eager_inactive_ok
    assert torch.isfinite(eager_pcm).all()

    codec.streaming_init(3)
    pointers = [tensor.data_ptr() for tensor in _state_tensors(codec)]
    assert codec.capture_cuda_graphs(decode_frame_counts=(1, 3)) == ["decode_f1", "decode_f3", "encode"]
    assert pointers == [tensor.data_ptr() for tensor in _state_tensors(codec)]

    graph_codes, graph_pcm, graph_inactive_ok = _drive(codec, frames=12, recycle_at=7, device=device)

    assert codec._cuda_graphs["encode"].replays == 12
    assert codec._cuda_graphs["decode_f1"].replays == 12
    assert graph_inactive_ok
    assert torch.equal(graph_codes, eager_codes)
    assert torch.equal(graph_pcm, eager_pcm)

    # A full reset in place (a new stream on the same codec) replays the same
    # frames from a fresh state.
    codec.streaming_init(3)
    again_codes, again_pcm, _ = _drive(codec, frames=12, recycle_at=7, device=device)
    assert torch.equal(again_codes, eager_codes)
    assert torch.equal(again_pcm, eager_pcm)
    assert pointers == [tensor.data_ptr() for tensor in _state_tensors(codec)]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch_size", [1, 3])
def test_multi_frame_decode_graph_matches_eager_chunks(batch_size: int) -> None:
    # Code2Wav records the first single-frame delta and the full chunk, on one
    # B=1 decoder per session. B=3 runs the same graphs on shared rows, where
    # every call carries all rows and only some are active.
    chunk = _MIMI_DECODE_BATCH_FRAMES
    partial = chunk - 2
    assert partial not in (1, chunk)
    device = torch.device("cuda")
    codec = _random_codec(device, batch_size=batch_size)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    # First frame alone, full chunks, a partial chunk with no graph, then a
    # single frame and a full chunk again on the state the eager step left.
    frame_counts = [1, chunk, chunk, partial, 1, chunk]
    row_masks = [(True, True, False), (True, False, True), (True, True, True), (False, True, True)]
    steps = [
        (frames, None if batch_size == 1 else _mask(*row_masks[step % len(row_masks)], device=device))
        for step, frames in enumerate(frame_counts)
    ]
    chunks = [torch.randint(0, 2048, (batch_size, CODEBOOKS, frames), generator=generator) for frames, _ in steps]

    eager = [codec.decode_frames(chunk_codes, active) for chunk_codes, (_, active) in zip(chunks, steps)]
    codec.streaming_init(batch_size)
    assert codec.capture_cuda_graphs(encode=False, decode_frame_counts=(1, chunk)) == ["decode_f1", f"decode_f{chunk}"]
    graphed = [codec.decode_frames(chunk_codes, active) for chunk_codes, (_, active) in zip(chunks, steps)]

    # The partial chunk has no graph, so it is the one step that ran eagerly.
    assert codec._cuda_graphs["decode_f1"].replays == 2
    assert codec._cuda_graphs[f"decode_f{chunk}"].replays == 3
    assert f"decode_f{partial}" not in codec._cuda_graphs
    for (frames, active), expected, actual in zip(steps, eager, graphed):
        assert actual.shape == (batch_size, frames * FRAME_SIZE)
        rows = slice(None) if active is None else active
        assert torch.equal(actual[rows], expected[rows])


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_failed_capture_warns_with_the_error_and_stays_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.model_executor.models.personaplex import personaplex_mimi_cudagraph

    device = torch.device("cuda")
    codec = _random_codec(device, batch_size=2)
    pcm = 0.1 * torch.randn(2, FRAME_SIZE, generator=torch.Generator(device="cpu").manual_seed(SEED))
    expected_codes = codec.encode_frame(pcm)
    expected_pcm = codec.decode_frame(expected_codes)
    codec.streaming_init(2)

    class _FailingGraph:
        def __init__(self) -> None:
            raise RuntimeError("CUDA error: out of memory")

    warnings: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(torch.cuda, "CUDAGraph", _FailingGraph)
    monkeypatch.setattr(
        personaplex_mimi_cudagraph.logger,
        "warning",
        lambda *args, **kwargs: warnings.append((args, kwargs)),
    )

    assert codec.capture_cuda_graphs(encode=True, decode_frame_counts=(1,)) == []

    assert codec._cuda_graphs == {}
    assert len(warnings) == 1
    args, kwargs = warnings[0]
    assert "requested" in args[0] and "capture failed" in args[0] and "eagerly" in args[0]
    assert args[1:] == ("encode/decode_f1", 2)
    assert kwargs == {"exc_info": True}
    # The warmup frames are rolled back, so the eager codec starts a fresh stream.
    assert torch.equal(codec.encode_frame(pcm), expected_codes)
    assert torch.equal(codec.decode_frame(expected_codes), expected_pcm)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_capture_does_not_hide_errors_that_are_not_cuda_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    codec = _random_codec(torch.device("cuda"), batch_size=2)

    def _broken(*_args, **_kwargs):
        raise TypeError("not a capture failure")

    monkeypatch.setattr(codec, "_encode_frame_eager", _broken)
    with pytest.raises(TypeError, match="not a capture failure"):
        codec.capture_cuda_graphs(encode=True, decode_frame_counts=())
    assert codec._cuda_graphs == {}


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_capture_leaves_a_fresh_stream_and_resize_drops_graphs() -> None:
    device = torch.device("cuda")
    codec = _random_codec(device, batch_size=2)
    codec.capture_cuda_graphs(encode=True, decode_frame_counts=())
    assert list(codec._cuda_graphs) == ["encode"]
    for state in codec._conv_states():
        carry = state.prev if isinstance(state, _StreamConv1d) else state.partial
        assert not carry.any() and state._fresh.all()
    for transformer in (codec.encoder_transformer, codec.decoder_transformer):
        assert not transformer._offset.any()
        assert all(not kv.end_offset.any() for kv in transformer._kv)

    # Resizing reallocates the state, so graphs recorded at the old size are dropped.
    codec.streaming_init(4)
    assert codec._cuda_graphs == {}
    assert codec.encode_frame(torch.zeros(4, FRAME_SIZE)).shape == (4, CODEBOOKS)


class _RecordingCodec:
    """Stage 0 shared-encoder stand-in that records the graph capture request."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def streaming_init(self, batch_size: int) -> None:
        self.calls.append(("streaming_init", batch_size))

    def capture_cuda_graphs(self, *, encode: bool, decode_frame_counts: tuple[int, ...]) -> list[str]:
        self.calls.append(("capture", {"encode": encode, "decode_frame_counts": decode_frame_counts}))
        return ["encode"]


@pytest.mark.cpu
@pytest.mark.parametrize("enabled", [True, False])
def test_stage0_captures_the_shared_encoder_once_when_enabled(enabled: bool) -> None:
    from vllm_omni.model_executor.models.personaplex.duplex.stage0 import PersonaPlexStage0DuplexRuntime

    codec = _RecordingCodec()
    runtime = PersonaPlexStage0DuplexRuntime(
        object(),
        model_path="/unused",
        device="cpu",
        codec_factory=lambda: codec,
        max_sessions=4,
        codec_cuda_graphs=enabled,
    )

    assert runtime._shared_codec() is codec
    assert runtime._shared_codec() is codec
    expected: list[tuple[str, object]] = [("streaming_init", 4)]
    if enabled:
        expected.append(("capture", {"encode": True, "decode_frame_counts": ()}))
    assert codec.calls == expected


@pytest.mark.cpu
def test_resize_after_capture_warns_that_graphs_are_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.model_executor.models.personaplex import personaplex_mimi

    warnings: list[tuple] = []
    monkeypatch.setattr(personaplex_mimi.logger, "warning", lambda *args: warnings.append(args))
    codec = _random_codec(torch.device("cpu"), batch_size=2)
    codec._cuda_graphs = {"encode": object()}

    codec.streaming_init(2)
    assert "encode" in codec._cuda_graphs and not warnings

    codec.streaming_init(3)
    assert codec._cuda_graphs == {}
    assert len(warnings) == 1 and warnings[0][1:] == (2, 3, "encode")
