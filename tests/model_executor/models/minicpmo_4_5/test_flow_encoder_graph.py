# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FlowEncoderGraphs: shared startup arena, dynamic CUDA capture and NPU fallback."""

import functools
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.model_executor.models.minicpmo_4_5.flow_encoder_graph as graph_module
from vllm_omni.model_executor.models.minicpmo_4_5.flow_encoder_graph import (
    FlowEncoderGraphs,
    reachable_conformer_cache_frames,
)

pytestmark = [pytest.mark.core_model]

_DEPTH, _HEADS, _WIDTH, _HIDDEN = 2, 2, 4, 8
_CNN = (3, 2)
_LOOKAHEAD = 1


def _encode(tokens, *, cnn_cache, att_cache, weight):
    """Row-independent stand-in for embedding + ``forward_chunk`` + projection: 1-token lookahead, 2x upsample."""
    rows = int(tokens.shape[0])
    x = (tokens.to(weight.dtype).unsqueeze(-1) @ weight[:1])[:, :-_LOOKAHEAD]
    x = x.unsqueeze(2).expand(-1, -1, 2, -1).reshape(rows, -1, _HIDDEN)
    x = x + cnn_cache.sum(dim=(1, 2))[:, None, None] + att_cache.mean(dim=(0, 2, 3, 4))[:, None, None]
    hidden = x @ weight
    new_cnn = hidden[:, -_CNN[1] :, : _CNN[0]].transpose(1, 2).contiguous()
    added = hidden[:, :, :_WIDTH][None, :, None].expand(_DEPTH, rows, _HEADS, -1, _WIDTH)
    return hidden, new_cnn, torch.cat((att_cache, added), dim=3)


@pytest.fixture
def fake_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    # A "graph" whose replay runs the recorded body on the static views.
    record = lambda self, views: SimpleNamespace(replay=functools.partial(self._body, views))  # noqa: E731
    monkeypatch.setattr(FlowEncoderGraphs, "_record", record)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


def _graphs(device: str, rows=(1, 2, 3), frames=(8, 18)):
    weight = torch.randn(_HIDDEN, _HIDDEN, generator=torch.Generator().manual_seed(0)).to(device)
    encode = functools.partial(_encode, weight=weight)
    graphs = FlowEncoderGraphs(encode, rows=rows, token_widths=[6], lookahead=_LOOKAHEAD, upsample=2)
    keys = [(r, 6, f) for r in rows for f in frames]
    layout = {"cnn_shape": _CNN, "att_layout": (_DEPTH, _HEADS, _WIDTH), "hidden_dim": _HIDDEN}
    assert graphs.capture(keys, **layout, dtype=torch.float32, device=torch.device(device)) == len(keys)
    return graphs, encode


def _inputs(rows: int, width: int, frames: int, device: str, seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, 50, (rows, width), generator=generator)
    cnn = [torch.randn(1, *_CNN, generator=generator) for _ in range(rows)]
    att = [torch.randn(_DEPTH, 1, _HEADS, frames, _WIDTH, generator=generator) for _ in range(rows)]
    return tokens.to(device), [c.to(device) for c in cnn], [a.to(device) for a in att]


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("growths", "expected"),
    [([50], [300, 350, 400]), ([150], [300, 400]), ([30], [300, 330, 360, 390, 400]), ([], [300])],
)
def test_reachable_cache_frames_follow_the_streaming_trim(growths, expected) -> None:
    # 6 s default voice (300 frames); 28-token duplex units grow the cache by 50 frames.
    assert reachable_conformer_cache_frames(start=300, prompt_len=300, suffix=100, growths=growths) == expected


@pytest.mark.cpu
def test_keys_cover_every_row_count_width_and_cache_length() -> None:
    graphs = FlowEncoderGraphs(lambda *a, **k: a, rows=[2, 1, 2], token_widths=[3, 28], lookahead=3, upsample=2)
    assert graphs.token_widths == (28,)  # a width with no new frames is dropped
    assert graphs.keys_for(start=300, prompt_len=300, suffix=100) == [
        (rows, 28, frames) for rows in (1, 2) for frames in (300, 350, 400)
    ]


def _check_replay(graphs, encode, rows: int, frames: int, device: str) -> None:
    tokens, cnn, att = _inputs(rows, 6, frames, device, seed=rows * 100 + frames)
    graphed = graphs.run(tokens, cnn, att)
    expected = encode(tokens, cnn_cache=torch.cat(cnn, dim=0), att_cache=torch.cat(att, dim=1))
    assert graphed is not None
    for actual, reference in zip(graphed, expected, strict=True):
        assert torch.equal(actual, reference)


@pytest.mark.cpu
@pytest.mark.parametrize(("rows", "frames"), [(1, 8), (3, 18), (3, 8)])
def test_replay_matches_the_eager_encoder_exactly(fake_capture, rows, frames) -> None:
    graphs, encode = _graphs("cpu")
    _check_replay(graphs, encode, rows, frames, "cpu")
    assert graphs.replays == 1


@pytest.mark.cpu
def test_other_shapes_and_layouts_run_eager(fake_capture, monkeypatch: pytest.MonkeyPatch) -> None:
    graphs, _ = _graphs("cpu")
    for rows, width, frames in ((4, 6, 8), (1, 5, 8), (1, 6, 10)):  # never padded, never another shape
        assert graphs.run(*_inputs(rows, width, frames, "cpu")) is None
    tokens, cnn, att = _inputs(2, 6, 8, "cpu")
    assert graphs.run(tokens, cnn, [att[0], att[1].double()]) is None
    assert graphs.run(tokens.int(), cnn, att) is None
    assert graphs.run(tokens[:1], cnn, att) is None
    with monkeypatch.context() as patch:
        patch.setattr(graph_module, "_precision_state", lambda: ("other precision flags",))
        assert graphs.run(tokens, cnn, att) is None
    assert graphs.run(tokens, cnn, att) is not None
    assert graphs.replays == 1


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_captured_graphs_match_eager_bit_exactly_on_cuda() -> None:
    graphs, encode = _graphs("cuda")
    for rows, frames in ((1, 8), (3, 18), (1, 18), (3, 8)):
        _check_replay(graphs, encode, rows, frames, "cuda")
    assert graphs.replays == 4


@pytest.mark.cpu
@torch.inference_mode()
def test_unified_call_preserves_cache_across_shared_replays(fake_capture):
    graphs, encode = _graphs("cpu")
    graphs.capture_on_request = False
    tokens, cnn, att = _inputs(1, 6, 8, "cpu")
    saved = graphs(tokens, last_chunk=False, cnn_cache=cnn[0], att_cache=att[0])
    expected = encode(tokens, cnn_cache=cnn[0], att_cache=att[0])
    graphs(tokens + 1, last_chunk=False, cnn_cache=cnn[0] + 1, att_cache=att[0] + 1)
    for actual, reference in zip(saved, expected, strict=True):
        torch.testing.assert_close(actual, reference)
    assert graphs.replays == 2
    assert not graphs.exact_graphs


@pytest.mark.cpu
def test_shared_capture_failure_blocks_subsequent_calls(fake_capture, monkeypatch):
    graphs = FlowEncoderGraphs(lambda x, **kwargs: (x, x, x))

    def fail(views):
        raise RuntimeError("capture failed")

    monkeypatch.setattr(graphs, "_record", fail)
    with pytest.raises(RuntimeError, match="capture failed"):
        graphs.capture(
            [(1, 6, 8)],
            cnn_shape=_CNN,
            att_layout=(_DEPTH, _HEADS, _WIDTH),
            hidden_dim=_HIDDEN,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
    tokens, cnn, att = _inputs(1, 6, 8, "cpu")
    with pytest.raises(RuntimeError, match="restart"):
        graphs(tokens, last_chunk=False, cnn_cache=cnn[0], att_cache=att[0])


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_shared_arena_rejects_other_stream_and_autocast():
    graphs, _ = _graphs("cuda")
    inputs = _inputs(1, 6, 8, "cuda")
    other = torch.cuda.Stream()
    other.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(other):
        assert graphs.run(*inputs) is None
    with torch.autocast("cuda", dtype=torch.float16):
        assert graphs.run(*inputs) is None
    assert graphs.run(*inputs) is not None


@pytest.mark.cpu
@pytest.mark.parametrize("rows", [8, 16])
def test_explicit_shared_batches_bypass_opportunistic_capture_cutoff(fake_capture, rows):
    graphs, encode = _graphs("cpu", rows=(rows,))
    graphs.eager_min_batch = 8
    _check_replay(graphs, encode, rows, 8, "cpu")


@pytest.mark.cpu
def test_startup_precapture_uses_shared_arena_and_freezes_admission(fake_capture):
    from contextlib import nullcontext

    from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import BatchedToken2Wav

    weight = torch.randn(_HIDDEN, _HIDDEN)
    graphs = FlowEncoderGraphs(functools.partial(_encode, weight=weight), max_graphs=2, rows=[1])
    _, cnn, att = _inputs(1, 6, 8, "cpu")
    state = SimpleNamespace(flow_cache={"conformer_cnn_cache": cnn[0], "conformer_att_cache": att[0]})
    features = SimpleNamespace(mels=torch.zeros(1, 8, _HIDDEN))
    probes = []
    backend = SimpleNamespace(
        _chunk_encoder_graph=graphs,
        _chunk_encoder_token_widths=(6,),
        _flow_on_cuda=lambda: True,
        _pre_lookahead_len=lambda: _LOOKAHEAD,
        _upsample_stride=lambda: 2,
        _encoder_position_tables=lambda: (),
        setup_batch=lambda features, rows: [state],
        _ensure_relpos_pe=lambda tokens, cache: probes.append(cache.shape[3]),
        _autocast=lambda device: nullcontext(),
        flow=SimpleNamespace(encoder_proj=SimpleNamespace(out_features=_HIDDEN)),
    )
    assert BatchedToken2Wav.precapture_chunk_encoder(backend, features) == 2
    assert len(graphs.graphs) == 2
    assert backend._encoder_graphs is graphs
    assert not graphs.exact_graphs
    assert not graphs.capture_on_request
    assert probes == [108]
    assert BatchedToken2Wav.precapture_chunk_encoder(backend, features) == 0


@pytest.mark.cpu
def test_legacy_graph_options_share_one_owner():
    from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import BatchedToken2Wav

    graph = FlowEncoderGraphs(lambda *args, **kwargs: args, max_graphs=16)
    encoder = SimpleNamespace(embed=SimpleNamespace(pos_enc=SimpleNamespace(pe=torch.zeros(1))), up_layer=object())
    backend = SimpleNamespace(
        _chunk_encoder_graph=graph,
        _flow_on_cuda=lambda: True,
        _pre_lookahead_len=lambda: 1,
        _upsample_stride=lambda: 2,
        flow=SimpleNamespace(encoder=encoder),
    )
    result = BatchedToken2Wav._build_encoder_graphs(backend, {"enabled": True, "rows": [1, 2], "token_widths": [6]}, {})
    assert result is graph
    assert result.rows == (1, 2)
    assert result.token_widths == (6,)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_exact_precapture_failure_blocks_retry(monkeypatch):
    graph = FlowEncoderGraphs(lambda x, **kwargs: (x, x, x))

    def fail(*args):
        raise RuntimeError("capture failed")

    monkeypatch.setattr(graph, "_capture", fail)
    tokens = torch.zeros(1, 6, device="cuda")
    with pytest.raises(RuntimeError, match="capture failed"):
        graph.capture_now(tokens, last_chunk=False, cnn_cache=None, att_cache=None)
    with pytest.raises(RuntimeError, match="restart"):
        graph.capture_now(tokens, last_chunk=False, cnn_cache=None, att_cache=None)
    with pytest.raises(RuntimeError, match="restart"):
        graph(tokens, last_chunk=False, cnn_cache=None, att_cache=None)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("amp", [False, True])
@torch.inference_mode()
def test_shared_real_conformer_preserves_mixed_cache_dtypes(amp):
    from cosyvoice2.transformer.upsample_encoder_v2 import UpsampleConformerEncoderV2

    from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import BatchedToken2Wav, _undecorate_dynamo

    backend = BatchedToken2Wav.__new__(BatchedToken2Wav)
    torch.nn.Module.__init__(backend)
    backend.flow = torch.nn.Module()
    backend.flow.encoder = UpsampleConformerEncoderV2(
        input_size=32,
        output_size=32,
        num_blocks=1,
        num_up_blocks=1,
        attention_heads=4,
        linear_units=64,
        dropout_rate=0.0,
        positional_dropout_rate=0.0,
        attention_dropout_rate=0.0,
    ).cuda()
    _undecorate_dynamo(backend.flow.encoder, "forward_chunk")
    backend.flow.input_embedding = torch.nn.Embedding(64, 32).cuda()
    backend.flow.encoder_proj = torch.nn.Linear(32, 16).cuda()
    backend.flow.eval()
    tokens = torch.ones(1, 8, dtype=torch.long, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        backend._ensure_relpos_pe(tokens, None)
        _, cnn, att = backend._encode_chunk_eager(tokens, last_chunk=False, cnn_cache=None, att_cache=None)
        backend._ensure_relpos_pe(tokens, att)
        graph = FlowEncoderGraphs(
            functools.partial(backend._encode_chunk_eager, last_chunk=False),
            lookahead=backend._pre_lookahead_len(),
            upsample=backend._upsample_stride(),
            held_tensors=backend._encoder_position_tables,
        )
        graph.capture(
            [(1, 8, att.shape[3])],
            cnn_shape=tuple(cnn.shape[1:]),
            att_layout=(att.shape[0], att.shape[2], att.shape[4]),
            hidden_dim=16,
            dtype=att.dtype,
            cnn_dtype=cnn.dtype,
            device=att.device,
        )
    # Exit the capture autocast context so its temporary weight cache is gone.
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        for code in (2, 3):
            tokens.fill_(code)
            actual = graph.run(tokens, [cnn], [att])
            expected = backend._encode_chunk_eager(tokens, last_chunk=False, cnn_cache=cnn, att_cache=att)
            assert actual is not None
            for value, reference in zip(actual, expected, strict=True):
                torch.testing.assert_close(value, reference)


@pytest.mark.cpu
def test_cpu_fallback_and_capacity():
    def forward(x, **kwargs):
        return x.sin(), x + 1, x + 2

    wrapper = FlowEncoderGraphs(forward)
    x = torch.randn(4, requires_grad=True)
    wrapper(x, last_chunk=False, cnn_cache=None, att_cache=None)[0].sum().backward()
    torch.testing.assert_close(x.grad, x.detach().cos())
    assert not wrapper.exact_graphs
    with pytest.raises(ValueError):
        FlowEncoderGraphs(forward, max_graphs=-1)
    with pytest.raises(ValueError, match="capture_after"):
        FlowEncoderGraphs(forward, capture_after=1)


@pytest.mark.cpu
@torch.inference_mode()
def test_rocm_cuda_devices_fall_back_before_nvidia_stream_creation(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "test-rocm")
    token = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    wrapper = FlowEncoderGraphs(lambda x, **kwargs: (x, x, x))
    assert wrapper(token, last_chunk=False, cnn_cache=None, att_cache=None) == (token, token, token)
    assert wrapper.stats["ineligible"] == 1
    assert not wrapper.exact_graphs


@pytest.mark.cpu
@torch.inference_mode()
def test_npu_admission_then_replays_through_npu_runner(monkeypatch):
    from vllm_omni.platforms.npu import graph_tools

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(current_stream=lambda device: SimpleNamespace(npu_stream=1), graph_pool_handle=object),
        raising=False,
    )
    calls = []

    class Runner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._captures = 0

        @staticmethod
        def is_supported():
            return True

        @staticmethod
        def _stream_is_capturing():
            return False

        def run(self, operation, inputs, constants, compute):
            calls.append((operation, constants, inputs[0]))
            self._captures = 1
            return compute(*inputs)

        @property
        def stats(self):
            return {"captures": self._captures, "failed": 0, "hits": 0}

    monkeypatch.setattr(graph_tools, "NPUExactGraphRunner", Runner)

    class Device:
        type = "npu"

        def __eq__(self, other):
            return isinstance(other, Device)

        def __hash__(self):
            return hash("npu:0")

        def __str__(self):
            return "npu:0"

    token = type("Tok", (), {"device": Device(), "shape": (4,), "dtype": torch.float32})()
    wrapper = FlowEncoderGraphs(lambda x, **kwargs: (x, x, x), capture_after=2, max_graphs=2)
    first = wrapper(token, last_chunk=False, cnn_cache=None, att_cache=None)
    assert not calls
    assert wrapper.stats["admission"] == 1
    assert first[0] is token
    second = wrapper(token, last_chunk=False, cnn_cache=None, att_cache=None)
    assert calls and calls[0][0] == "conformer_chunk"
    assert second[0] is token
    assert wrapper.stats["captures"] == 1
    wrapper(token, last_chunk=False, cnn_cache=None, att_cache=None)
    assert len(calls) == 2
    assert wrapper.stats["hits"] == 1


@pytest.fixture
def simulated_npu(monkeypatch):
    """Keep real runner admission/replay logic; simulate only NPU capture."""
    from vllm_omni.platforms.npu.graph_tools import NPUExactGraphRunner

    stream = SimpleNamespace(npu_stream=1)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_stream=lambda device: stream,
            graph_pool_handle=object,
            is_current_stream_capturing=lambda: False,
        ),
        raising=False,
    )
    monkeypatch.setattr(NPUExactGraphRunner, "is_supported", staticmethod(lambda: True))

    def capture(self, inputs, compute):
        return SimpleNamespace(replay=lambda values: compute(*values))

    monkeypatch.setattr(NPUExactGraphRunner, "capture", capture)

    class Device:
        type = "npu"

        def __str__(self):
            return "npu:0"

    class Token:
        device = Device()
        dtype = torch.int64

        def __init__(self, frames=4):
            self.shape = (1, frames)

    return SimpleNamespace(stream=stream, token=Token)


@pytest.mark.cpu
@torch.inference_mode()
def test_npu_streams_have_independent_runners_and_pools(simulated_npu):
    wrapper = FlowEncoderGraphs(lambda x, **kw: (x, x, x), max_graphs=2)
    token = simulated_npu.token()
    for stream_id in (1, 2):
        simulated_npu.stream.npu_stream = stream_id
        for _ in range(3):
            wrapper(token, last_chunk=False, cnn_cache=None, att_cache=None)
    runners = list(wrapper._npu_runners.runners.values())
    assert len(runners) == 2
    assert runners[0]._graph_pool is not runners[1]._graph_pool
    assert [runner.stats["captures"] for runner in runners] == [1, 1]
    assert [runner.stats["hits"] for runner in runners] == [1, 1]
    assert wrapper.stats["captures"] == 2
    assert wrapper.stats["hits"] == 2
    assert wrapper.stats["eager"] == 4  # Admission and capture warmup per stream.
    assert len(wrapper._npu_pe) == 2

    # A third stream must fall back without creating a runner or retaining PE.
    simulated_npu.stream.npu_stream = 3
    wrapper(token, last_chunk=False, cnn_cache=None, att_cache=None, position_tables=(torch.zeros(8),))
    assert len(wrapper._npu_runners.runners) == 2
    assert len(wrapper._npu_pe) == 2
    assert wrapper.stats["capacity"] == 1

    # Existing streams must still replay after the shared cap is exhausted.
    simulated_npu.stream.npu_stream = 1
    wrapper(token, last_chunk=False, cnn_cache=None, att_cache=None)
    assert wrapper.stats["hits"] == 3


@pytest.mark.cpu
@torch.inference_mode()
def test_npu_capacity_does_not_retain_uncaptured_position_tables(simulated_npu):
    import weakref

    wrapper = FlowEncoderGraphs(lambda x, **kw: (x, x, x), max_graphs=1)
    token = simulated_npu.token()
    table = torch.zeros(4, 8)
    retained = weakref.ref(table)
    for _ in range(2):
        wrapper(token, last_chunk=False, cnn_cache=None, att_cache=None, position_tables=(table,))
    del table
    assert retained() is not None
    for frames in range(5, 25):
        table = torch.zeros(frames, 8)
        uncaptured = weakref.ref(table)
        for _ in range(2):
            wrapper(
                simulated_npu.token(frames),
                last_chunk=False,
                cnn_cache=None,
                att_cache=None,
                position_tables=(table,),
            )
        del table
        assert uncaptured() is None
    assert wrapper.stats["captures"] == 1
    assert wrapper.stats["capacity"] == 40
    assert wrapper.stats["eager"] == 42
    assert len(wrapper._npu_pe) == 1
    assert not wrapper.seen


@pytest.mark.cpu
@torch.inference_mode()
def test_npu_position_table_replacement_captures_new_graph(simulated_npu):
    wrapper = FlowEncoderGraphs(lambda x, **kw: (x, x, x), max_graphs=2)
    token = simulated_npu.token()
    tables = (torch.zeros(4, 8), torch.ones(4, 8))
    for table in tables:
        for _ in range(3):
            wrapper(token, last_chunk=False, cnn_cache=None, att_cache=None, position_tables=(table,))
    assert wrapper.stats["captures"] == 2
    assert wrapper.stats["hits"] == 2
    assert len(wrapper._npu_pe) == 2


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_configurable_admission_avoids_short_lived_shape_capture():
    wrapper = FlowEncoderGraphs(lambda x, **kw: (x + 1, x + 2, x + 3), capture_after=4)
    x = torch.ones(8, device="cuda")
    for _ in range(3):
        torch.testing.assert_close(wrapper(x, last_chunk=False, cnn_cache=None, att_cache=None)[0], x + 1)
    assert not wrapper.exact_graphs
    assert wrapper.stats["admission"] == 3
    for value in (2, 3):
        x.fill_(value)
        result = wrapper(x, last_chunk=False, cnn_cache=None, att_cache=None)
        torch.testing.assert_close(result[0], x + 1)
    assert wrapper.stats["captures"] == 1
    assert wrapper.stats["hits"] == 2


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("amp", [False, True])
@torch.inference_mode()
def test_real_conformer_chunks_replay_and_preserve_state(amp):
    from cosyvoice2.transformer.upsample_encoder_v2 import UpsampleConformerEncoderV2

    from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import BatchedToken2Wav, _undecorate_dynamo

    encoder = (
        UpsampleConformerEncoderV2(
            input_size=32,
            output_size=32,
            num_blocks=1,
            num_up_blocks=1,
            attention_heads=4,
            linear_units=64,
            dropout_rate=0.0,
            positional_dropout_rate=0.0,
            attention_dropout_rate=0.0,
        )
        .eval()
        .cuda()
    )
    _undecorate_dynamo(encoder, "forward_chunk")
    backend = BatchedToken2Wav.__new__(BatchedToken2Wav)
    torch.nn.Module.__init__(backend)
    backend.flow = torch.nn.Module()
    backend.flow.input_embedding = torch.nn.Embedding(64, 32).cuda()
    backend.flow.encoder = encoder
    backend.flow.encoder_proj = torch.nn.Linear(32, 16).cuda()
    backend.flow.eval()
    graph = FlowEncoderGraphs(backend._encode_chunk_eager, max_graphs=4)
    backend._chunk_encoder_graph = graph
    cnn = att = None
    retained = []
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        for last in [False, False, True]:
            tokens = torch.randint(0, 64, (2, 8), device="cuda")
            # Repeated input shape with distinct values must update graph inputs.
            for _ in range(3):
                tokens = (tokens + 1) % 64
                expected = backend._encode_chunk_eager(tokens, last_chunk=last, cnn_cache=cnn, att_cache=att)
                actual = backend._encode_chunk(tokens, last_chunk=last, cnn_cache=cnn, att_cache=att)
                for a, b in zip(actual, expected, strict=True):
                    torch.testing.assert_close(a, b, atol=3e-3 if amp else 1e-5, rtol=3e-3 if amp else 1e-5)
                retained.append((actual, tuple(x.clone() for x in actual)))
            if not last:
                _, cnn, att = actual
        # A replacement positional table must select a new graph entry.
        before = graph.stats["captures"]
        encoder.embed.pos_enc.pe = encoder.embed.pos_enc.pe.clone()
        for _ in range(2):
            backend._encode_chunk(tokens, last_chunk=True, cnn_cache=cnn, att_cache=att)
        assert graph.stats["captures"] == before + 1
    assert graph.stats["hits"] >= 6
    for actual, expected in retained:
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_stream_isolation_cache_refresh_and_capacity():
    def forward(x, *, last_chunk, cnn_cache, att_cache):
        return x + cnn_cache + att_cache + int(last_chunk), cnn_cache + x, att_cache - x

    wrapper = FlowEncoderGraphs(forward, max_graphs=2)
    streams = [torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()]
    retained = []
    for stream in streams:
        with torch.cuda.stream(stream):
            x = torch.ones(3, device="cuda")
            cnn = torch.zeros_like(x)
            att = torch.zeros_like(x)
            for i in range(3):
                cnn.fill_(i)
                att.fill_(2 * i)
                out = wrapper(x, last_chunk=False, cnn_cache=cnn, att_cache=att)
                retained.append((out[0], torch.full_like(x, 1 + 3 * i)))
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    assert len(wrapper.exact_graphs) == 2
    assert wrapper.stats["hits"] == 4
    assert wrapper.stats["capacity"] == 3
    for actual, expected in retained:
        torch.testing.assert_close(actual, expected)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_overlapping_replays_use_independent_scratch_pools():
    def forward(x, **kwargs):
        intermediate = x.sin() + x.cos()
        for _ in range(8):
            intermediate = intermediate.sin() + 0.1 * x
        return intermediate, intermediate * 2, intermediate * 3

    wrapper = FlowEncoderGraphs(forward, max_graphs=2)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    # Complete captures before testing concurrent replay (warmup synchronizes).
    for i, stream in enumerate(streams):
        with torch.cuda.stream(stream):
            x = torch.full((512, 512), float(i), device="cuda")
            for _ in range(2):
                wrapper(x, last_chunk=False, cnn_cache=None, att_cache=None)
    torch.accelerator.synchronize()
    assert wrapper._slots[0][1] != wrapper._slots[1][1]
    retained = []
    for iteration in range(6):
        for i, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                x = torch.full((512, 512), float(iteration + i), device="cuda")
                result = wrapper(x, last_chunk=False, cnn_cache=None, att_cache=None)
                retained.append((result, x))
    torch.accelerator.synchronize()
    for result, x in retained:
        for a, b in zip(result, forward(x), strict=True):
            torch.testing.assert_close(a, b)


@pytest.mark.cpu
@torch.inference_mode()
def test_failed_capture_blocks_even_eager_retry(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=1))
    wrapper = FlowEncoderGraphs(lambda x, **kw: (x, x, x))
    x = SimpleNamespace(device=torch.device("cuda:0"), shape=(3,), dtype=torch.float32)
    wrapper(x, last_chunk=False, cnn_cache=None, att_cache=None)

    def fail(*args):
        raise RuntimeError("simulated capture failure")

    monkeypatch.setattr(wrapper, "_capture", fail)
    with pytest.raises(RuntimeError, match="simulated capture failure"):
        wrapper(x, last_chunk=False, cnn_cache=None, att_cache=None)
    with pytest.raises(RuntimeError, match="restart the stage"):
        wrapper(x, last_chunk=False, cnn_cache=None, att_cache=None)
    with pytest.raises(RuntimeError, match="restart the stage"):
        wrapper(torch.ones(3), last_chunk=False, cnn_cache=None, att_cache=None)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_full_cache_preserves_graphs_outputs_and_bounded_metadata():
    weight = torch.randn(32, 32, device="cuda")

    def forward(x, **kwargs):
        y = (x @ weight).sin()
        return y, y + 1, y + 2

    wrapper = FlowEncoderGraphs(forward, max_graphs=2)
    retained = []

    def call(size, repeats=1):
        x = torch.randn(size, 32, device="cuda")
        for _ in range(repeats):
            result = wrapper(x, last_chunk=False, cnn_cache=None, att_cache=None)
        for a, b in zip(result, forward(x), strict=True):
            torch.testing.assert_close(a, b)
        retained.append((result, tuple(value.clone() for value in result)))
        return next(reversed(wrapper.exact_graphs))

    key_a = call(1, 2)
    key_b = call(2, 2)
    call(1)
    call(3, 2)
    assert key_a in wrapper.exact_graphs
    assert key_b in wrapper.exact_graphs
    for size in range(4, 36):
        call(size, 2)
        assert len(wrapper.exact_graphs) == len(wrapper._slots) == 2
        assert len(wrapper.seen) <= 8
    assert wrapper.stats["evictions"] == 0
    assert wrapper.stats["captures"] == 2
    assert key_a in wrapper.exact_graphs and key_b in wrapper.exact_graphs
    hits = wrapper.stats["hits"]
    call(1)
    assert wrapper.stats["hits"] == hits + 1
    for result, expected in retained:
        for actual, reference in zip(result, expected, strict=True):
            torch.testing.assert_close(actual, reference)


@pytest.mark.cpu
@torch.inference_mode()
def test_large_batch_and_frozen_wrapper_stay_eager(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=1))
    calls = []

    def forward(x, **kwargs):
        calls.append(x)
        return x, x, x

    wrapper = FlowEncoderGraphs(forward, max_graphs=4, eager_min_batch=8)
    large = SimpleNamespace(device=torch.device("cuda:0"), shape=(8, 25), dtype=torch.float32)
    assert wrapper(large, last_chunk=False, cnn_cache=None, att_cache=None)[0] is large
    assert wrapper.stats["batch"] == 1
    assert not wrapper.exact_graphs

    small = SimpleNamespace(device=torch.device("cuda:0"), shape=(1, 25), dtype=torch.float32)
    wrapper.capture_on_request = False
    assert wrapper(small, last_chunk=False, cnn_cache=None, att_cache=None)[0] is small
    assert wrapper.stats["capacity"] == 1
    assert not wrapper.exact_graphs
    assert calls == [large, small]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_capture_streams_are_not_reused_by_torch_pool():
    # Retaining torch.cuda.Stream objects does not reserve their underlying
    # streams. Resetting a graph clears all cuBLAS workspaces on its capture
    # stream, so a pooled capture stream can invalidate a still-live peer.
    weight = torch.randn(512, 512, device="cuda")
    wrapper = FlowEncoderGraphs(lambda x, **kw: (x @ weight, x + 1, x + 2), max_graphs=2)
    for size in (28, 32):
        x = torch.randn(size, 512, device="cuda")
        for _ in range(2):
            wrapper(x, last_chunk=False, cnn_cache=None, att_cache=None)
    streams = {slot[5].cuda_stream for slot in wrapper._slots}
    assert len(streams) == 2
    pooled = {torch.cuda.Stream().cuda_stream for _ in range(128)}
    assert streams.isdisjoint(pooled)
    # A full cache stays eager and does not allocate more capture streams.
    for size in (36, 40, 44):
        x = torch.randn(size, 512, device="cuda")
        for _ in range(2):
            result = wrapper(x, last_chunk=False, cnn_cache=None, att_cache=None)
        torch.testing.assert_close(result[0], x @ weight)
    assert {slot[5].cuda_stream for slot in wrapper._slots} == streams
    assert wrapper.stats["evictions"] == 0
    assert wrapper.stats["captures"] == 2
    assert wrapper.stats["capacity"] == 6


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_precision_change_recaptures_instead_of_reusing_previous_gemm():
    x = torch.randn(128, 128, device="cuda")
    weight = torch.randn_like(x)
    wrapper = FlowEncoderGraphs(lambda value, **kw: (value @ weight, value + 1, value + 2), max_graphs=2)
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        retained = []
        for tf32 in (False, True):
            torch.backends.cuda.matmul.allow_tf32 = tf32
            expected = x @ weight
            for _ in range(2):
                result = wrapper(x, last_chunk=False, cnn_cache=None, att_cache=None)
                torch.testing.assert_close(result[0], expected)
                retained.append((result[0], result[0].clone()))
        assert wrapper.stats["captures"] == 2
        for actual, saved in retained:
            torch.testing.assert_close(actual, saved)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
