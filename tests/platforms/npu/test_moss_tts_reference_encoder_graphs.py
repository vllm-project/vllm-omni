# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tensor tests of reference ACLGraph planning and runtime boundaries."""

from contextlib import contextmanager, nullcontext

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vllm_omni.model_executor.models.moss_tts.reference_encoder_graphs import MossReferenceEncoderGraphs, _Graph
from vllm_omni.platforms.npu.models import moss_tts_reference_encoder_graphs as backend

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]

FRAME, N_VQ = 8, 2


class _Tokenizer(nn.Module):
    number_channels = 2
    sampling_rate = 64
    downsample_rate = FRAME
    compute_dtype = None

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.encoder = nn.ModuleList()

    def _flatten_channels_for_codec(self, audio, lengths):
        batch, channels, samples = audio.shape
        hidden = audio.reshape(batch, channels, samples // FRAME, FRAME).mean((1, 3))
        return hidden, lengths // FRAME

    def _codec_inference_autocast(self):
        return nullcontext()

    def quantizer(self, hidden, lengths, n_vq):
        codes = torch.stack([hidden.round().long() + i for i in range(n_vq)])
        return hidden, codes, lengths

    def eager(self, wav):
        audio = F.pad(wav, (0, -wav.shape[-1] % FRAME))[None]
        hidden, lengths = self._flatten_channels_for_codec(audio, torch.tensor([wav.shape[-1]]))
        _, codes, lengths = self.quantizer(hidden, lengths, N_VQ)
        return codes[:, 0, : lengths[0]].T.contiguous()


class _Stream:
    def __init__(self, **kwargs):
        self.waits = []

    def wait_stream(self, stream):
        self.waits.append(stream)


class _Runtime:
    Stream = _Stream

    def __init__(self):
        self.config = type("Config", (), {"allow_internal_format": True})()
        self.compile_mode = None
        self.device_active = False
        self.stream_active = False
        self.selected = []
        self.current = _Stream()

    @contextmanager
    def device(self, device):
        self.selected.append(device)
        self.device_active = True
        try:
            yield
        finally:
            self.device_active = False

    @contextmanager
    def stream(self, stream):
        self.stream_active = True
        try:
            yield
        finally:
            self.stream_active = False

    def current_stream(self, device):
        assert self.device_active
        return self.current

    def set_compile_mode(self, *, jit_compile):
        self.compile_mode = jit_compile


class _Runner:
    """Model the existing runner's owned outputs and bounded signature cache."""

    def __init__(self, *, max_graphs, **kwargs):
        self.max_graphs = max_graphs
        self.outputs = {}
        self.calls = []
        self.failure = None
        self.hits = 0

    @staticmethod
    def is_supported():
        return True

    @property
    def stats(self):
        return {"captures": len(self.outputs), "hits": self.hits, "failed": int(self.failure is not None)}

    def run(self, operation, inputs, shape, compute):
        if self.failure is not None:
            raise RuntimeError(self.failure)
        assert not torch.is_grad_enabled()
        self.calls.append((shape, inputs[1].clone()))
        result = compute(*inputs)
        if shape not in self.outputs:
            assert len(self.outputs) < self.max_graphs
            self.outputs[shape] = tuple(t.clone() for t in result)
        else:
            self.hits += 1
            for dst, src in zip(self.outputs[shape], result):
                dst.copy_(src)
        return tuple(t.clone() for t in self.outputs[shape])


@pytest.fixture
def graph_runtime(monkeypatch):
    runtime = _Runtime()
    monkeypatch.setattr(torch, "get_device_module", lambda name: runtime)
    monkeypatch.setattr(backend, "NPUExactGraphRunner", _Runner)
    monkeypatch.delenv("ASCEND_LAUNCH_BLOCKING", raising=False)
    return runtime


def test_reference_aclgraph_buckets_lengths_order_and_owned_outputs(graph_runtime):
    tokenizer = _Tokenizer()
    graphs = backend.MossNPUReferenceEncoderGraphs(tokenizer, n_vq=N_VQ)
    graphs.capture()
    assert graphs.captured == [(1, 256), (1, 512), (2, 256)]
    assert graphs.stats["captures"] == 3
    graphs.capture()
    assert len(graphs._runner.calls) == 3
    # Includes non-frame lengths, >2 rows, different buckets and eager misses.
    wavs = [torch.full((2, length), float(i + 1)) for i, length in enumerate([17, 300, 8, 255, 700, 48])]
    codes = graphs.encode(wavs)
    for i, (got, wav) in enumerate(zip(codes, wavs)):
        if i == 4:
            assert got is None
        else:
            torch.testing.assert_close(got, tokenizer.eager(wav), rtol=0, atol=0)
            assert got.device.type == "cpu" and got.dtype == torch.long
    assert graphs.stats["captures"] == 3
    assert graphs.stats["hits"] > 0
    # Check actual input lengths, including the unused full-length silence row.
    graphs.encode([torch.ones(2, 9), torch.ones(2, 15)])
    assert graphs._runner.calls[-1][1].tolist() == [15, 9]
    old = graphs.encode([torch.ones(2, FRAME)])[0]
    saved = old.clone()
    graphs.encode([torch.full((2, FRAME), 99.0)])
    torch.testing.assert_close(old, saved, rtol=0, atol=0)
    assert not graph_runtime.device_active and not graph_runtime.stream_active
    assert graph_runtime.current.waits
    assert graph_runtime.config.allow_internal_format is False
    assert graph_runtime.compile_mode is False


def test_reference_aclgraph_invalid_shapes_stay_eager(graph_runtime):
    graphs = backend.MossNPUReferenceEncoderGraphs(_Tokenizer(), n_vq=N_VQ)
    graphs.capture()
    before = len(graphs._runner.calls)
    assert graphs.encode([torch.ones(3, 10), torch.ones(10), torch.ones(2, 0), torch.ones(2, 600)]) == [None] * 4
    assert len(graphs._runner.calls) == before


@pytest.mark.parametrize("during_capture", [False, True])
def test_reference_aclgraph_failure_requires_restart(graph_runtime, during_capture):
    graphs = backend.MossNPUReferenceEncoderGraphs(_Tokenizer(), n_vq=N_VQ)
    if not during_capture:
        graphs.capture()
    graphs._runner.failure = "simulated capture/replay failure"
    with pytest.raises(RuntimeError, match="simulated"):
        if during_capture:
            graphs.capture()
        else:
            graphs.encode([torch.ones(2, 10)])
    graphs._runner.failure = None
    before = len(graphs._runner.calls)
    with pytest.raises(RuntimeError, match="restart the API process"):
        graphs.encode([torch.ones(2, 600)])  # even eager misses cannot use a poisoned process
    with pytest.raises(RuntimeError, match="restart the API process"):
        graphs.capture()
    assert len(graphs._runner.calls) == before
    assert not graph_runtime.device_active and not graph_runtime.stream_active


@pytest.mark.parametrize("dtype", [None, torch.float32, torch.float16, torch.bfloat16])
def test_reference_aclgraph_core_matches_eager_autocast_scope(graph_runtime, monkeypatch, dtype):
    tokenizer = _Tokenizer()
    tokenizer.compute_dtype = dtype
    active = []
    seen = []

    @contextmanager
    def autocast(*, device_type, dtype):
        assert device_type == "npu"
        active.append(dtype)
        try:
            yield
        finally:
            active.pop()

    original = tokenizer.quantizer

    def quantizer(*args):
        seen.append(list(active))
        return original(*args)

    monkeypatch.setattr(torch, "autocast", autocast)
    monkeypatch.setattr(tokenizer, "quantizer", quantizer)
    graphs = backend.MossNPUReferenceEncoderGraphs(tokenizer, n_vq=N_VQ)
    graphs.capture()
    graphs.encode([torch.ones(2, 10)])
    assert all(s == ([dtype] if dtype in (torch.float16, torch.bfloat16) else []) for s in seen)
    assert active == []


def test_reference_aclgraph_blocking_mode_rejected_before_runtime_changes(graph_runtime, monkeypatch):
    monkeypatch.setenv("ASCEND_LAUNCH_BLOCKING", "1")
    graphs = backend.MossNPUReferenceEncoderGraphs(_Tokenizer(), n_vq=N_VQ)
    with pytest.raises(ValueError, match="ASCEND_LAUNCH_BLOCKING"):
        graphs.capture()
    assert graph_runtime.compile_mode is None and not graphs.captured


def test_reference_aclgraph_unsupported_runtime_does_not_capture(graph_runtime, monkeypatch):
    monkeypatch.setattr(_Runner, "is_supported", staticmethod(lambda: False))
    graphs = backend.MossNPUReferenceEncoderGraphs(_Tokenizer(), n_vq=N_VQ)
    graphs.capture()
    assert graphs.captured == [] and graphs.stats["captures"] == 0
    assert graphs.encode([torch.ones(2, 10)]) == [None]


@pytest.mark.parametrize("platform", ["cuda", "npu"])
def test_reference_aclgraph_shared_wrapper_selection_and_replay(graph_runtime, platform):
    """Exercise the common selector through both backend replay implementations."""
    tokenizer = _Tokenizer()
    calls = []
    if platform == "npu":
        graphs = backend.MossNPUReferenceEncoderGraphs(tokenizer, n_vq=N_VQ)
        graphs.capture()
        original = graphs._runner.run

        def run(operation, inputs, shape, compute):
            calls.append(shape)
            return original(operation, inputs, shape, compute)

        graphs._runner.run = run
    else:
        graphs = MossReferenceEncoderGraphs(
            tokenizer, n_vq=N_VQ, batch_sizes=(1, 2), bucket_seconds=(4, 8), max_batch_seconds=8
        )

        class Replay:
            def __init__(self, audio, lengths, codes, code_lengths, shape):
                self.audio, self.lengths = audio, lengths
                self.codes, self.code_lengths, self.shape = codes, code_lengths, shape

            def replay(self):
                calls.append(self.shape)
                codes, lengths = graphs._core(self.audio, self.lengths)
                self.codes.copy_(codes)
                self.code_lengths.copy_(lengths)

        # Real CPU buffers simulate CUDA capture outputs; no CUDA device is required.
        for batch, samples in [(1, 256), (1, 512), (2, 256)]:
            audio = torch.zeros(batch, 2, samples)
            lengths = torch.full((batch,), samples, dtype=torch.long)
            codes, code_lengths = graphs._core(audio, lengths)
            replay = Replay(audio, lengths, codes, code_lengths, (batch, samples))
            graphs._graphs[(batch, samples)] = _Graph(replay, audio, lengths, codes, code_lengths)

    wavs = [torch.full((2, length), float(i + 1)) for i, length in enumerate([17, 300, 8, 255, 700, 48])]
    results = graphs.encode(wavs)
    assert calls == [(1, 512), (2, 256), (2, 256)]
    for index, (result, wav) in enumerate(zip(results, wavs)):
        if index == 4:
            assert result is None
        else:
            torch.testing.assert_close(result, tokenizer.eager(wav), rtol=0, atol=0)
    calls.clear()
    graphs.encode([torch.ones(2, 17)])
    assert calls == [(1, 256)]
    calls.clear()
    assert graphs.encode([]) == [] and not calls


def test_reference_aclgraph_frame_rounded_capture_limit(graph_runtime):
    tokenizer = _Tokenizer()
    tokenizer.sampling_rate = 67
    tokenizer.downsample_rate = 10
    graphs = backend.MossNPUReferenceEncoderGraphs(tokenizer, n_vq=N_VQ)
    # Neither four nor eight seconds is frame-aligned for this geometry.
    assert graphs._shapes == [(1, 270), (1, 540), (2, 270)]
