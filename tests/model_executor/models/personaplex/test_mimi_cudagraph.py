# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fixed-buffer streaming state and CUDA graph replay of the PersonaPlex Mimi codec."""

import gc
import os
import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from vllm.platforms import current_platform

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


def _assert_waveform_matches(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not current_platform.is_rocm():
        assert torch.equal(actual, expected)
        return

    # ROCm convolution kernels can accumulate low-order FP32 differences
    # between eager execution and CUDA graph replay. Keep token outputs exact,
    # and bound both the peak and aggregate waveform error here.
    error = (actual - expected).abs()
    rms = error.square().mean().sqrt().item()
    print(f"PersonaPlex Mimi ROCm graph parity: max_abs={error.max().item():.9g}, rms={rms:.9g}")
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-4)
    assert rms <= 1e-4, rms


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
    _assert_waveform_matches(graph_pcm, eager_pcm)

    # A full reset in place (a new stream on the same codec) replays the same
    # frames from a fresh state.
    codec.streaming_init(3)
    again_codes, again_pcm, _ = _drive(codec, frames=12, recycle_at=7, device=device)
    assert torch.equal(again_codes, eager_codes)
    _assert_waveform_matches(again_pcm, eager_pcm)
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
        _assert_waveform_matches(actual[rows], expected[rows])


@pytest.mark.cpu
def test_frame_graph_counts_eager_fallbacks() -> None:
    from vllm_omni.model_executor.models.personaplex.personaplex_mimi_cudagraph import MimiFrameGraph

    def eager(frame: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        del active
        return frame

    graph = MimiFrameGraph(
        graph=SimpleNamespace(replay=lambda: None),
        static_input=torch.zeros(2, CODEBOOKS, 1),
        static_active=torch.ones(2, dtype=torch.bool),
        static_output=torch.zeros(2, 1),
        eager=eager,
    )

    # A batch width the graph was not recorded for must run eagerly, be
    # counted, and never touch the recorded graph.
    for _ in range(2):
        out = graph.replay(torch.zeros(3, CODEBOOKS, 1), torch.ones(3, dtype=torch.bool))
    assert out.shape == (3, CODEBOOKS, 1)
    assert graph.replays == 0
    assert graph.eager_fallbacks == 2


@pytest.mark.cpu
def test_graph_replay_stats_count_dispatch_and_survive_resets() -> None:
    codec = _random_codec(torch.device("cpu"), batch_size=2)
    assert codec.graph_replay_stats() == {
        "encode_graph": 0,
        "encode_eager": 0,
        "decode_graph": 0,
        "decode_eager": 0,
    }

    pcm = torch.randn(2, FRAME_SIZE)
    codes = codec.encode_frame(pcm, _mask(True, False))
    codec.decode_frame(codes, _mask(True, False))
    assert codec.graph_replay_stats() == {
        "encode_graph": 0,
        "encode_eager": 1,
        "decode_graph": 0,
        "decode_eager": 1,
    }

    # An in-place stream reset (a recycled codec) keeps the counters.
    codec.streaming_init(2)
    codec.encode_frame(pcm)
    assert codec.graph_replay_stats() == {
        "encode_graph": 0,
        "encode_eager": 2,
        "decode_graph": 0,
        "decode_eager": 1,
    }


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_replay_stats_report_graph_and_eager_decode_calls() -> None:
    device = torch.device("cuda")
    codec = _random_codec(device, batch_size=2)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    chunk = _MIMI_DECODE_BATCH_FRAMES
    frames_list = [1, chunk, chunk, chunk - 2]  # the partial chunk has no graph
    codec.streaming_init(2)
    assert codec.capture_cuda_graphs(encode=True, decode_frame_counts=(1, chunk)) == [
        "decode_f1",
        f"decode_f{chunk}",
        "encode",
    ]

    for frames in frames_list:
        codec.decode_frames(torch.randint(0, 2048, (2, CODEBOOKS, frames), generator=generator))
    codec.encode_frame(torch.zeros(2, FRAME_SIZE))

    assert codec.graph_replay_stats() == {
        "encode_graph": 1,
        "encode_eager": 0,
        "decode_graph": 3,
        "decode_eager": 1,
    }


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    not os.environ.get("PERSONAPLEX_MIMI_CHECKPOINT"),
    reason="PERSONAPLEX_MIMI_CHECKPOINT must point at the bundled tokenizer safetensors",
)
def test_real_checkpoint_graph_replay_is_bitwise_equal_to_eager() -> None:
    codec = PersonaPlexMimiCodec(
        checkpoint=os.environ["PERSONAPLEX_MIMI_CHECKPOINT"],
        device=torch.device("cuda"),
    )
    device = torch.device("cuda")
    codec.streaming_init(2)
    eager_codes, eager_pcm, _ = _drive(codec, frames=48, recycle_at=31, device=device)
    after_eager = codec.graph_replay_stats()
    assert after_eager["encode_eager"] == 48 and after_eager["encode_graph"] == 0
    assert after_eager["decode_eager"] == 48 and after_eager["decode_graph"] == 0

    codec.streaming_init(2)
    assert codec.capture_cuda_graphs(decode_frame_counts=(1, 3)) == ["decode_f1", "decode_f3", "encode"]
    graph_codes, graph_pcm, _ = _drive(codec, frames=48, recycle_at=31, device=device)

    assert torch.equal(graph_codes, eager_codes)
    _assert_waveform_matches(graph_pcm, eager_pcm)
    # Counters accumulate since construction: the eager reference phase stays
    # counted, and every graphed-drive dispatch replays.
    stats = codec.graph_replay_stats()
    assert stats["encode_graph"] == 48 and stats["encode_eager"] == 48
    assert stats["decode_graph"] == 48 and stats["decode_eager"] == 48


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

    class _FailingCapture:
        def __enter__(self) -> None:
            pass

        def __exit__(self, *_args) -> None:
            raise RuntimeError("CUDA error: out of memory")

    warnings: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(torch.cuda, "graph", lambda *_args, **_kwargs: _FailingCapture())
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
    assert args[1:3] == ("encode/decode_f1", 2)
    assert "Traceback (most recent call last):" in args[3]
    assert "RuntimeError: CUDA error: out of memory" in args[3]
    assert not kwargs
    # The warmup frames are rolled back, so the eager codec starts a fresh stream.
    assert torch.equal(codec.encode_frame(pcm), expected_codes)
    _assert_waveform_matches(codec.decode_frame(expected_codes), expected_pcm)


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


@pytest.fixture
def capture_faults(monkeypatch: pytest.MonkeyPatch):
    """Exercise capture orchestration and error boundaries with CPU tensors."""
    from vllm_omni.model_executor.models.personaplex import personaplex_mimi_cudagraph as graphs

    state = SimpleNamespace(
        errors={},
        counts={},
        events=[],
        warnings=[],
        phase="setup",
        capturing=False,
        graph_refs=[],
        capture_refs=[],
        output_refs=[],
        check_released=False,
    )
    state.original_stream = state.stream = object()

    def record(event):
        state.events.append(event)
        state.counts[event] = state.counts.get(event, 0) + 1
        error = state.errors.get((event, state.counts[event]), state.errors.get(event))
        if error is not None:
            raise error

    def eager(name):
        def step(frame, active):
            state.phase = "capture" if state.capturing else "warmup"
            record(f"{name}_{state.phase}")
            if state.capturing:
                output = frame.clone()
                state.output_refs.append(weakref.ref(output))
                return output
            return frame

        return step

    def reset():
        assert state.stream is state.original_stream
        if state.check_released:
            assert all(ref() is None for ref in state.graph_refs + state.capture_refs + state.output_refs)
        state.phase = "reset"
        record("reset")

    codec = SimpleNamespace(
        device="cuda",
        dtype=torch.float32,
        _batch_size=2,
        _encode_frame_eager=eager("encode"),
        _decode_frames_eager=eager("decode"),
        reset_streaming=reset,
    )

    class Graph:
        def __init__(self):
            record("construct")
            state.graph_refs.append(weakref.ref(self))

    class Capture:
        def __init__(self, graph):
            self.graph = graph
            state.capture_refs.append(weakref.ref(self))

        def __enter__(self):
            record("capture_setup_sync")
            record("capture_setup_empty_cache")
            state.stream = object()
            state.capturing = True
            record("capture_begin")

        def __exit__(self, exc_type, exc, traceback):
            state.capturing = False
            # Match torch.cuda.graph: capture_end can raise before restoration,
            # and can replace a body exception with capture-invalidated.
            record("capture_end")
            state.stream = state.original_stream

    @contextmanager
    def stream_context(stream):
        previous = state.stream
        state.stream = stream
        try:
            yield
        finally:
            state.stream = previous

    zeros, ones = torch.zeros, torch.ones
    monkeypatch.setattr(torch, "zeros", lambda *args, **kwargs: zeros(*args, **(kwargs | {"device": "cpu"})))
    monkeypatch.setattr(torch, "ones", lambda *args, **kwargs: ones(*args, **(kwargs | {"device": "cpu"})))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: (0, 0))
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "graph", lambda graph, **kwargs: Capture(graph))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: state.stream)
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda device: record(f"sync_{state.phase}"))
    monkeypatch.setattr(graphs.logger, "warning", lambda *args, **kwargs: state.warnings.append((args, kwargs)))
    state.capture = lambda: graphs.capture_mimi_frame_graphs(
        codec,
        encode=True,
        decode_frame_counts=(1,),
        warmup_iters=1,
    )
    return state


@pytest.mark.cpu
@pytest.mark.parametrize("step", ["encode_warmup", "decode_warmup"])
@pytest.mark.parametrize("error_type", [RuntimeError, torch.cuda.OutOfMemoryError])
def test_eager_warmup_failures_propagate_before_capture(capture_faults, step, error_type):
    error = error_type("injected eager tensor failure")
    capture_faults.errors[step] = error
    with pytest.raises(error_type) as raised:
        capture_faults.capture()
    assert raised.value is error
    assert capture_faults.counts.get("construct", 0) == 0
    assert capture_faults.counts["reset"] == 1
    assert not capture_faults.warnings


@pytest.mark.cpu
@pytest.mark.parametrize(
    "point",
    [
        "sync_warmup",
        "construct",
        "capture_setup_sync",
        "capture_setup_empty_cache",
        "capture_begin",
        "sync_capture",
        "sync_reset",
        "reset",
    ],
)
def test_device_and_reset_failures_are_never_capture_fallbacks(capture_faults, point):
    # Even an otherwise recoverable error class must propagate at these boundaries.
    error = torch.cuda.OutOfMemoryError("injected device or reset failure")
    capture_faults.errors[point] = error
    with pytest.raises(torch.cuda.OutOfMemoryError) as raised:
        capture_faults.capture()
    assert raised.value is error
    assert not capture_faults.warnings


@pytest.mark.cpu
@pytest.mark.parametrize(
    "message",
    [
        "tensor size mismatch",
        "CUDA error: an illegal memory access was encountered",
        "CUDA error: device-side assert triggered",
        "CUDA error: operation failed due to a previous error during capture",
    ],
)
def test_unknown_or_fatal_capture_errors_propagate(capture_faults, message):
    error = RuntimeError(message)
    capture_faults.errors["encode_capture"] = error
    with pytest.raises(RuntimeError) as raised:
        capture_faults.capture()
    assert raised.value is error
    assert capture_faults.stream is capture_faults.original_stream
    assert capture_faults.counts["reset"] == 1
    assert not capture_faults.warnings


@pytest.mark.cpu
def test_recoverable_capture_failure_discards_partial_graphs_after_all_warmups(capture_faults):
    error = torch.cuda.OutOfMemoryError("capture allocation failed")
    capture_faults.errors[("capture_end", 2)] = error
    capture_faults.check_released = True
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        assert capture_faults.capture() == {}
    finally:
        if was_enabled:
            gc.enable()
    assert capture_faults.events.index("decode_warmup") < capture_faults.events.index("construct")
    assert capture_faults.counts["capture_end"] == 2  # an earlier graph completed
    assert capture_faults.counts["sync_reset"] == 1
    assert capture_faults.stream is capture_faults.original_stream
    assert len(capture_faults.warnings) == 1
    assert "capture allocation failed" in capture_faults.warnings[0][0][-1]
    assert "Traceback (most recent call last):" in capture_faults.warnings[0][0][-1]
    assert not capture_faults.warnings[0][1]
    assert error.__traceback__ is None


@pytest.mark.cpu
@pytest.mark.parametrize("recoverable", [True, False])
def test_capture_end_does_not_hide_the_original_failure(capture_faults, recoverable):
    message = "CUDA error: operation not permitted when stream is capturing" if recoverable else "tensor size mismatch"
    original = RuntimeError(message)
    capture_faults.errors["encode_capture"] = original
    invalidated = RuntimeError("CUDA error: operation failed due to a previous error during capture")
    capture_faults.errors["capture_end"] = invalidated
    if recoverable:
        capture_faults.check_released = True
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            assert capture_faults.capture() == {}
        finally:
            if was_enabled:
                gc.enable()
        assert len(capture_faults.warnings) == 1
        assert original.__traceback__ is None and invalidated.__traceback__ is None
    else:
        with pytest.raises(RuntimeError) as raised:
            capture_faults.capture()
        assert raised.value is invalidated and raised.value.__context__ is original
        assert not capture_faults.warnings
    assert capture_faults.stream is capture_faults.original_stream
    assert capture_faults.counts["reset"] == 1


@pytest.mark.cpu
@pytest.mark.parametrize("point", ["reset", "sync_reset"])
def test_capture_fallback_requires_successful_reset(capture_faults, point):
    capture_faults.errors["capture_end"] = torch.cuda.OutOfMemoryError("capture allocation failed")
    error = RuntimeError("codec could not reset")
    capture_faults.errors[point] = error
    with pytest.raises(RuntimeError) as raised:
        capture_faults.capture()
    assert raised.value is error
    assert not capture_faults.warnings


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_shared_width_codec_rows_match_dedicated_codecs_and_graphs_stay_bitwise() -> None:
    """One codec at N rows is the primitive of the shared Stage 1 decoder.

    Row r of the width-N codec must track a dedicated B=1 codec over a mixed
    frame-count schedule. Batched kernels reduce in different orders across
    batch widths, so cross-width parity is a tolerance comparison (as the RFC
    states for the encoder); graphed-versus-eager on the SAME width stays
    bitwise, exactly like the per-codec B=1 graphs.
    """
    device = torch.device("cuda")
    rows = 3
    frame_counts = [1, 5, 5, 2, 1]  # first frame, full chunks, an uncaptured count
    generator = torch.Generator(device="cpu").manual_seed(99)
    codes = [
        [torch.randint(0, 2048, (CODEBOOKS, frames), generator=generator).to(device) for frames in frame_counts]
        for _ in range(rows)
    ]
    active = torch.ones(rows, dtype=torch.bool, device=device)
    batches = [torch.stack([codes[row][step] for row in range(rows)]) for step in range(len(frame_counts))]

    shared = _random_codec(device, rows)
    per_request = [_random_codec(device, 1) for _ in range(rows)]  # same seed: same weights

    # Cross-width: row r vs codec r, tolerance-bounded (fp32 reduction-order noise).
    peak = rms = scale = 0.0
    for step in range(len(frame_counts)):
        expected = torch.cat(
            [per_request[row].decode_frames(codes[row][step].unsqueeze(0)) for row in range(rows)],
            dim=0,
        )
        actual = shared.decode_frames(batches[step], active)
        error = (actual - expected).abs()
        peak = max(peak, error.max().item())
        rms = max(rms, error.square().mean().sqrt().item())
        scale = max(scale, expected.abs().max().item())
    # Measured on the random-weight codec: rel_peak ~1e-3, rel_rms ~2e-4,
    # the expected fp32 batched-reduction noise amplified by the deep stack and
    # the streaming state. Bounds keep a 2x margin over the measurement.
    assert peak <= 2e-3 * scale, f"cross-width peak error {peak:.3e} on scale {scale:.3e}"
    assert rms <= 5e-4 * scale, f"cross-width rms error {rms:.3e} on scale {scale:.3e}"

    # Within-width: graphed vs eager on the same width-N codec is bitwise.
    shared.streaming_init(rows)  # in-place reset; fresh stream
    eager_audio = [shared.decode_frames(batch, active) for batch in batches]
    shared.streaming_init(rows)
    assert shared.capture_cuda_graphs(encode=False, decode_frame_counts=(1, 5)) == ["decode_f1", "decode_f5"]
    before = shared.graph_replay_stats()
    graph_audio = [shared.decode_frames(batch, active) for batch in batches]
    for step, (eager, graphed) in enumerate(zip(eager_audio, graph_audio)):
        assert torch.equal(eager, graphed), f"graphed step {step} diverged from eager at width {rows}"

    after = shared.graph_replay_stats()
    assert after["decode_graph"] - before["decode_graph"] == 4  # F in {1, 5}
    assert after["decode_eager"] - before["decode_eager"] == 1  # F == 2 has no graph
