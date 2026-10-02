# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact-shape flow-encoder graphs (``cfm_encoder_cuda_graph``): keys, shared buffers, backend wiring.

CPU only: ``FlowEncoderGraphs._record`` is replaced by a fake graph whose
replay runs the recorded body, so these tests cover everything around the CUDA
capture itself -- which shapes get a graph, how inputs reach the shared static
buffers, that results match the eager encoder exactly, and that the backend
copies shared results it keeps past the next replay.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm_omni.model_executor.models.minicpmo_4_5.flow_encoder_graph as graph_module
from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import BatchedToken2Wav
from vllm_omni.model_executor.models.minicpmo_4_5.flow_encoder_graph import (
    FlowEncoderGraphs,
    reachable_conformer_cache_frames,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEPTH, _HEADS, _WIDTH = 2, 2, 4
_CNN = (3, 2)
_LOOKAHEAD, _UPSAMPLE = 1, 2


class _FakeGraph:
    def __init__(self, graphs: FlowEncoderGraphs, entry) -> None:
        self.graphs = graphs
        self.entry = entry
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1
        self.graphs._body(self.entry)


@pytest.fixture(autouse=True)
def _fake_capture(monkeypatch: pytest.MonkeyPatch):
    recorded: list[tuple[int, ...]] = []

    def record(self, entry):
        recorded.append((int(entry.tokens.shape[0]), int(entry.tokens.shape[1]), int(entry.att.shape[3])))
        return _FakeGraph(self, entry)

    monkeypatch.setattr(FlowEncoderGraphs, "_record", record)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return recorded


class _Encoder(nn.Module):
    """Row-independent stand-in for ``UpsampleConformerEncoderV2.forward_chunk`` with its cache layouts.

    ``cnn_cache`` ``(rows, 3, 2)``, ``att_cache`` ``(depth, rows, heads, frames, width)``,
    a 1-token lookahead and a 2x upsample; every output depends on the row's tokens and caches.
    """

    def __init__(self) -> None:
        super().__init__()
        self.pre_lookahead_layer = SimpleNamespace(pre_lookahead_len=_LOOKAHEAD)
        self.up_layer = SimpleNamespace(stride=_UPSAMPLE)
        self.calls: list[tuple[int, int, bool]] = []

    def forward_chunk(self, xs, last_chunk=False, cnn_cache=None, att_cache=None):
        rows, tokens, _ = xs.shape
        self.calls.append((rows, tokens, last_chunk))
        if last_chunk:
            xs = F.pad(xs, (0, 0, 0, _LOOKAHEAD))
        out = xs[:, :-_LOOKAHEAD].repeat_interleave(_UPSAMPLE, dim=1)
        if cnn_cache is not None:
            out = out + cnn_cache.sum(dim=(1, 2))[:, None, None]
        if att_cache is not None:
            out = out + 0.5 * att_cache.mean(dim=(0, 2, 3, 4))[:, None, None]
        new_cnn = out[:, -_CNN[1] :, :1].transpose(1, 2).expand(-1, _CNN[0], -1).contiguous()
        frames = out[:, :, 0].transpose(0, 1).reshape(1, -1, rows, 1, 1)
        added = (frames.permute(0, 2, 3, 1, 4) * torch.arange(1, _WIDTH + 1, dtype=out.dtype)).expand(
            _DEPTH, rows, _HEADS, -1, _WIDTH
        )
        new_att = added.contiguous() if att_cache is None else torch.cat((att_cache, added), dim=3)
        return out, new_cnn, new_att


class _Estimator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        conv1 = SimpleNamespace(causal_padding=(1, 0))
        block = SimpleNamespace(
            conv=SimpleNamespace(in_channels=1, out_channels=1, block=[None, conv1]),
            attn=SimpleNamespace(num_heads=1, head_dim=1),
        )
        self.blocks = [block]
        self.in_proj = nn.Identity()

    def t_embedder(self, time):
        return time[:, None]

    def blocks_forward_chunk(self, inputs, time, mask, cnn_cache, att_cache, cnn_out, att_out):
        del time, mask, cnn_cache, att_cache
        marker = inputs[:, 1, 0]
        cnn_out.copy_(marker.reshape(1, -1, 1, 1).expand_as(cnn_out))
        att_out.copy_(marker.reshape(1, -1, 1, 1, 1).expand_as(att_out))
        return inputs[:, 1:2] + 0.25 * inputs[:, 0:1]


class _Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.estimator = _Estimator()
        self.inference_cfg_rate = 0.7
        self.register_buffer("rand_noise", torch.linspace(-1, 1, 600).reshape(1, 1, 600), persistent=False)


class _Flow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()
        self.encoder_proj = nn.Linear(1, 1)
        with torch.no_grad():
            self.encoder_proj.weight.fill_(0.5)
            self.encoder_proj.bias.fill_(0.125)
        self.decoder = _Decoder()
        self.spk_embed_affine_layer = nn.Identity()

    def input_embedding(self, tokens):
        return tokens.to(torch.float32).unsqueeze(-1) / 16


class _HiFT(nn.Module):
    def inference(self, mel, source):
        del source
        speech = mel[:, 0].repeat_interleave(3, dim=1)
        return speech, speech[:, None]


class _Token2Wav:
    def __init__(self) -> None:
        self.flow = _Flow()
        self.hift = _HiFT()
        self.float16 = False
        self.n_timesteps = 2
        self.mel_cache_len = 1
        self.source_cache_len = 2
        self.speech_window = torch.hamming_window(4, periodic=False)

    def _prepare_prompt(self, prompt_wav):
        del prompt_wav
        return (
            torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
            torch.tensor([4], dtype=torch.int32),
            torch.ones(1, 1),
            torch.ones(1, 8, 1),
            torch.tensor([8], dtype=torch.int32),
        )


def _ragged_kernel(adapter: BatchedToken2Wav) -> None:
    def kernel(estimator, estimator_input, time_embedding, attn_mask, cnn_cache, att_cache, cnn_buf, att_buf, lengths):
        del lengths
        return estimator.blocks_forward_chunk(
            estimator_input, time_embedding, attn_mask, cnn_cache, att_cache, cnn_buf, att_buf
        )

    adapter._blocks_forward_chunk_ragged = kernel  # type: ignore[method-assign]


def _backend(*, rows=(1, 2), widths=(6,)) -> BatchedToken2Wav:
    adapter = BatchedToken2Wav(_Token2Wav())
    _ragged_kernel(adapter)
    if rows:
        adapter._encoder_graphs = FlowEncoderGraphs(
            adapter._encode_continuation,
            rows=rows,
            token_widths=widths,
            lookahead=_LOOKAHEAD,
            upsample=_UPSAMPLE,
        )
    return adapter


def _tokens(chunk: int, rows: int, width: int) -> torch.Tensor:
    return torch.stack([torch.arange(width) * (row + 1) + 7 * chunk + row for row in range(rows)])


def _assert_states_equal(actual, expected) -> None:
    for got, want in zip(actual, expected, strict=True):
        for name, value in want.flow_cache.items():
            assert torch.equal(got.flow_cache[name], value), name
        for name, value in want.hift_cache.items():
            assert torch.equal(got.hift_cache[name], value), name


# ---------------------------------------------------------------------------
# Keys


@pytest.mark.parametrize(
    ("growths", "expected"),
    [
        ([50], [300, 350, 400]),  # 6 s default voice, 28-token duplex units
        ([150], [300, 400]),  # 78-token turn chunks
        ([50, 150], [300, 350, 400]),
        ([30], [300, 330, 360, 390, 400]),
        ([], [300]),
    ],
)
def test_reachable_cache_frames_follow_the_streaming_trim(growths, expected) -> None:
    assert reachable_conformer_cache_frames(start=300, prompt_len=300, suffix=100, growths=growths) == expected


def test_keys_cover_every_row_count_width_and_cache_length() -> None:
    graphs = FlowEncoderGraphs(lambda *a: a, rows=[2, 1, 2], token_widths=[28], lookahead=3, upsample=2)

    assert graphs.rows == (1, 2)
    assert graphs.output_frames(28) == 50
    assert graphs.keys_for(start=300, prompt_len=300, suffix=100) == [
        (1, 28, 300),
        (1, 28, 350),
        (1, 28, 400),
        (2, 28, 300),
        (2, 28, 350),
        (2, 28, 400),
    ]


def test_widths_without_new_frames_are_dropped() -> None:
    graphs = FlowEncoderGraphs(lambda *a: a, rows=[1], token_widths=[3, 28], lookahead=3, upsample=2)
    assert graphs.token_widths == (28,)


# ---------------------------------------------------------------------------
# FlowEncoderGraphs on a stand-in encoder


def _graphs(rows=(1, 2, 3), widths=(6,), frames=(8, 18, 28)):
    encoder = _Encoder()
    proj = nn.Linear(1, 1)

    def encode(tokens, cnn, att):
        hidden, new_cnn, new_att = encoder.forward_chunk(
            tokens.to(torch.float32).unsqueeze(-1), last_chunk=False, cnn_cache=cnn, att_cache=att
        )
        return proj(hidden), new_cnn, new_att

    graphs = FlowEncoderGraphs(encode, rows=rows, token_widths=widths, lookahead=_LOOKAHEAD, upsample=_UPSAMPLE)
    keys = [(r, w, f) for r in rows for w in widths for f in frames]
    graphs.capture(
        keys, cnn_shape=_CNN, att_layout=(_DEPTH, _HEADS, _WIDTH), hidden_dim=1, dtype=torch.float32, device="cpu"
    )
    return graphs, encode


def _inputs(rows: int, width: int, frames: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, 50, (rows, width), generator=generator)
    cnn = [torch.randn(1, *_CNN, generator=generator) for _ in range(rows)]
    att = [torch.randn(_DEPTH, 1, _HEADS, frames, _WIDTH, generator=generator) for _ in range(rows)]
    return tokens, cnn, att


@pytest.mark.parametrize(("rows", "frames"), [(1, 8), (2, 18), (3, 28), (3, 8)])
def test_replay_matches_the_eager_encoder_exactly(rows, frames) -> None:
    graphs, encode = _graphs()
    tokens, cnn, att = _inputs(rows, 6, frames, seed=rows * 100 + frames)

    graphed = graphs.run(tokens, cnn, att)
    expected = encode(tokens, torch.cat(cnn, dim=0), torch.cat(att, dim=1))

    assert graphed is not None
    for actual, reference in zip(graphed, expected, strict=True):
        assert actual.shape == reference.shape
        assert torch.equal(actual, reference)
    assert graphs.stats["replays"] == 1


@pytest.mark.parametrize(
    ("rows", "width", "frames"),
    [
        (4, 6, 8),  # more rows than captured: never padded down or split
        (1, 5, 8),  # another token width
        (1, 6, 10),  # a cache length no prompt stream reaches
    ],
)
def test_shapes_without_their_own_graph_run_eager(rows, width, frames) -> None:
    graphs, _ = _graphs()
    tokens, cnn, att = _inputs(rows, width, frames, seed=1)

    assert graphs.run(tokens, cnn, att) is None
    assert graphs.stats["replays"] == 0


def test_mismatched_cache_layouts_run_eager() -> None:
    graphs, _ = _graphs()
    tokens, cnn, att = _inputs(2, 6, 8, seed=2)

    assert graphs.run(tokens, cnn, [att[0], att[1].double()]) is None
    assert graphs.run(tokens, [cnn[0], cnn[1][:, :2]], att) is None
    assert graphs.run(tokens.int(), cnn, att) is None
    assert graphs.run(tokens[:1], cnn, att) is None
    assert graphs.run(tokens, cnn, att) is not None


def test_graphs_share_one_storage_per_buffer_and_capture_the_largest_first(_fake_capture) -> None:
    graphs, _ = _graphs()

    assert _fake_capture[0] == (3, 6, 28)
    assert _fake_capture[-1] == (1, 6, 8)
    for name in ("tokens", "cnn", "att", "hidden", "new_cnn", "new_att"):
        pointers = {getattr(entry, name).data_ptr() for entry in graphs.graphs.values()}
        assert pointers == {graphs._storage[name].data_ptr()}
    largest = graphs.graphs[(3, 6, 28)]
    assert graphs._storage["att"].numel() == largest.att.numel()
    assert graphs._storage["new_att"].numel() == largest.new_att.numel()
    assert graphs.storage_bytes() == sum(t.numel() * t.element_size() for t in graphs._storage.values())


def test_results_are_shared_views_valid_until_the_next_replay() -> None:
    graphs, encode = _graphs()
    first_inputs = _inputs(1, 6, 8, seed=3)
    first = graphs.run(*first_inputs)
    kept = [value.clone() for value in first]

    graphs.run(*_inputs(1, 6, 8, seed=4))

    assert not torch.equal(first[0], kept[0])  # rewritten in place by the second replay
    expected = encode(first_inputs[0], torch.cat(first_inputs[1]), torch.cat(first_inputs[2], dim=1))
    for value, reference in zip(kept, expected, strict=True):
        assert torch.equal(value, reference)


def test_a_precision_change_after_capture_runs_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    graphs, _ = _graphs()
    inputs = _inputs(1, 6, 8, seed=5)
    monkeypatch.setattr(graph_module, "_precision_state", lambda: ("tf32",))

    assert graphs.run(*inputs) is None
    assert graphs.stats["eager_precision"] == 1
    assert graphs.enabled


def test_a_replaced_relpos_table_disables_the_graphs() -> None:
    table = {"pe": torch.zeros(3)}
    graphs = FlowEncoderGraphs(
        lambda tokens, cnn, att: (tokens.float()[:, :, None].repeat(1, 2, 1)[:, :-2], cnn, att),
        rows=[1],
        token_widths=[2],
        lookahead=1,
        upsample=2,
        held_tensors=lambda: (table["pe"],),
    )
    graphs.capture(
        [(1, 2, 4)],
        cnn_shape=_CNN,
        att_layout=(_DEPTH, _HEADS, _WIDTH),
        hidden_dim=1,
        dtype=torch.float32,
        device="cpu",
    )
    assert graphs._held == (table["pe"],)

    table["pe"] = torch.zeros(3)
    tokens, cnn, att = _inputs(1, 2, 4, seed=6)

    assert graphs.run(tokens, cnn, att) is None
    assert not graphs.enabled


def test_capture_is_all_or_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def record(self, entry):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("capture failed")
        return _FakeGraph(self, entry)

    monkeypatch.setattr(FlowEncoderGraphs, "_record", record)
    graphs = FlowEncoderGraphs(lambda *a: a, rows=[1, 2], token_widths=[6], lookahead=1, upsample=2)

    with pytest.raises(RuntimeError, match="capture failed"):
        graphs.capture(
            [(1, 6, 8), (2, 6, 8)],
            cnn_shape=_CNN,
            att_layout=(_DEPTH, _HEADS, _WIDTH),
            hidden_dim=1,
            dtype=torch.float32,
            device="cpu",
        )
    assert graphs.graphs == {}
    assert graphs._storage == {}
    assert not graphs.enabled


def test_an_output_layout_mismatch_fails_the_capture() -> None:
    graphs = FlowEncoderGraphs(
        lambda tokens, cnn, att: (torch.zeros(1, 3, 1), cnn, att),
        rows=[1],
        token_widths=[6],
        lookahead=1,
        upsample=2,
    )
    graphs._record = lambda entry: graphs._body(entry)  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="hidden output"):
        graphs.capture(
            [(1, 6, 8)],
            cnn_shape=_CNN,
            att_layout=(_DEPTH, _HEADS, _WIDTH),
            hidden_dim=1,
            dtype=torch.float32,
            device="cpu",
        )


# ---------------------------------------------------------------------------
# Backend wiring


def test_default_width_is_the_duplex_unit_chunk() -> None:
    widths = BatchedToken2Wav._default_encoder_token_widths
    assert widths({"codec_chunk_frames": 75, "initial_codec_chunk_frames": 25, "codec_left_context_frames": 3}) == [28]
    assert widths({"codec_chunk_frames": 25, "codec_left_context_frames": 3}) == [28]
    assert widths(None) == [28]


def test_encoder_graphs_are_off_by_default_and_off_cuda() -> None:
    assert BatchedToken2Wav(_Token2Wav())._encoder_graphs is None
    # The stand-in flow lives on the CPU: the switch alone does not build graphs there.
    assert BatchedToken2Wav(_Token2Wav(), encoder_graph_config={"enabled": True, "rows": [1]})._encoder_graphs is None


def test_build_requires_the_upsample_conformer_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(BatchedToken2Wav, "_flow_on_cuda", lambda self: True)
    adapter = BatchedToken2Wav(_Token2Wav())
    # No RelPos table on the stand-in encoder: stays eager.
    assert adapter._build_encoder_graphs({"enabled": True, "rows": [1]}, None) is None

    pe = torch.zeros(1, 9, 4)
    adapter.flow.encoder.embed = SimpleNamespace(pos_enc=SimpleNamespace(pe=pe))
    adapter.flow.encoder.up_embed = SimpleNamespace(pos_enc=SimpleNamespace(pe=pe + 1))
    graphs = adapter._build_encoder_graphs(
        {"enabled": True, "rows": [1, 2]}, {"initial_codec_chunk_frames": 25, "codec_left_context_frames": 3}
    )
    assert graphs is not None
    assert graphs.rows == (1, 2)
    assert graphs.token_widths == (28,)
    assert (graphs.lookahead, graphs.upsample) == (_LOOKAHEAD, _UPSAMPLE)
    held = graphs._held_tensors()
    assert len(held) == 2
    assert held[0] is pe and held[1] is adapter.flow.encoder.up_embed.pos_enc.pe

    adapter.flow.encoder.embed.extend_pe = lambda x: None
    assert adapter._build_encoder_graphs({"enabled": True, "rows": [1]}, None) is None


def test_precapture_covers_the_prompt_stream_cache_lengths() -> None:
    adapter = _backend(rows=(1, 2), widths=(6,))
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")

    assert adapter.precapture_flow_encoder(prompt) == 2 * 11
    # Prompt state: 4 tokens + 1 lookahead -> 8 frames; +10 per 6-token chunk; trimmed at 8 + 100.
    assert sorted({key[2] for key in adapter._encoder_graphs.graphs}) == list(range(8, 109, 10))
    assert adapter.precapture_flow_encoder(prompt) == 0  # once


def test_decode_batch_with_graphs_is_bitwise_the_eager_decode() -> None:
    eager, graphed = _backend(rows=()), _backend(rows=(1, 2))
    prompts = [backend.prepare_prompt("shared", "/fake/prompt.wav") for backend in (eager, graphed)]
    graphed.precapture_flow_encoder(prompts[1])
    states = [backend.setup_batch(prompt, 2) for backend, prompt in zip((eager, graphed), prompts, strict=True)]

    for chunk in range(14):  # through the cache trim and well into the steady length
        last = chunk == 13
        tokens = _tokens(chunk, 2, 6)
        outputs = [
            backend.decode_batch(tokens, prompt, state, last_chunk=last)
            for backend, prompt, state in zip((eager, graphed), prompts, states, strict=True)
        ]
        (eager_audio, eager_states), (graph_audio, graph_states) = outputs
        for got, want in zip(graph_audio, eager_audio, strict=True):
            assert torch.equal(got, want)
        _assert_states_equal(graph_states, eager_states)
        states = [eager_states, graph_states]

    # Every chunk but the last (lookahead padding) replayed a graph.
    assert graphed._encoder_graphs.stats["replays"] == 13


def test_one_row_streams_replay_the_one_row_graph() -> None:
    eager, graphed = _backend(rows=()), _backend(rows=(1, 2))
    prompts = [backend.prepare_prompt("shared", "/fake/prompt.wav") for backend in (eager, graphed)]
    graphed.precapture_flow_encoder(prompts[1])
    states = [backend.setup_batch(prompt, 1) for backend, prompt in zip((eager, graphed), prompts, strict=True)]

    for chunk in range(4):
        tokens = _tokens(chunk, 1, 6)
        (eager_audio, eager_states), (graph_audio, graph_states) = (
            backend.decode_batch(tokens, prompt, state, last_chunk=False)
            for backend, prompt, state in zip((eager, graphed), prompts, states, strict=True)
        )
        assert torch.equal(graph_audio[0], eager_audio[0])
        _assert_states_equal(graph_states, eager_states)
        states = [eager_states, graph_states]
    assert graphed._encoder_graphs.stats["replays"] == 4


def _ragged_step(backends, prompts, states, rows, last_chunks):
    (eager_audio, eager_states), (graph_audio, graph_states) = (
        backend.decode_ragged_batch(rows, prompt, state, last_chunks=last_chunks)
        for backend, prompt, state in zip(backends, prompts, states, strict=True)
    )
    for got, want in zip(graph_audio, eager_audio, strict=True):
        assert torch.equal(got, want)
    _assert_states_equal(graph_states, eager_states)


@pytest.mark.parametrize("warm_chunks", [0, 11])
def test_ragged_batch_keeps_each_encoder_group_result_past_the_next_replay(warm_chunks) -> None:
    """Two encoder groups (token widths 6 and 4) replay two graphs that share their result buffers."""
    backends = (_backend(rows=()), _backend(rows=(1, 2, 3), widths=(4, 6)))
    prompts = [backend.prepare_prompt("shared", "/fake/prompt.wav") for backend in backends]
    backends[1].precapture_flow_encoder(prompts[1])
    states = [backend.setup_batch(prompt, 3) for backend, prompt in zip(backends, prompts, strict=True)]
    for chunk in range(warm_chunks):  # 11 chunks: the steady, trimmed cache length
        tokens = _tokens(chunk, 3, 6)
        (eager_audio, eager_states), (graph_audio, graph_states) = (
            backend.decode_batch(tokens, prompt, state, last_chunk=False)
            for backend, prompt, state in zip(backends, prompts, states, strict=True)
        )
        _assert_states_equal(graph_states, eager_states)
        states = [eager_states, graph_states]
    stats = backends[1]._encoder_graphs.stats
    before = stats["replays"]

    six = _tokens(warm_chunks, 2, 6)
    _ragged_step(backends, prompts, states, [six[0], six[1], _tokens(warm_chunks, 1, 4)[0]], [False] * 3)

    assert stats["replays"] - before == 2


def test_ragged_final_rows_stay_eager_next_to_graphed_ones() -> None:
    backends = (_backend(rows=()), _backend(rows=(1, 2), widths=(4, 6)))
    prompts = [backend.prepare_prompt("shared", "/fake/prompt.wav") for backend in backends]
    backends[1].precapture_flow_encoder(prompts[1])
    states = [backend.setup_batch(prompt, 3) for backend, prompt in zip(backends, prompts, strict=True)]

    six = _tokens(0, 2, 6)
    _ragged_step(backends, prompts, states, [six[0], six[1], _tokens(0, 1, 4)[0]], [False, False, True])

    assert backends[1]._encoder_graphs.stats["replays"] == 1


# ---------------------------------------------------------------------------
# Code2Wav connector extras


def _code2wav(extra: dict, max_num_seqs: int | None = 20):
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import MiniCPMO45Code2Wav

    config = SimpleNamespace(
        model_config=SimpleNamespace(model="/fake/model", stage_connector_config={"extra": dict(extra)}),
    )
    if max_num_seqs is not None:
        config.scheduler_config = SimpleNamespace(max_num_seqs=max_num_seqs)
    return MiniCPMO45Code2Wav(vllm_config=config)


@pytest.mark.parametrize(
    ("extra", "max_num_seqs", "expected"),
    [
        ({}, 20, {"enabled": False, "rows": list(range(1, 9)), "token_widths": None}),
        ({"cfm_encoder_cuda_graph": True}, 20, {"enabled": True, "rows": list(range(1, 9)), "token_widths": None}),
        ({"cfm_encoder_cuda_graph": True, "cfm_encoder_graph_rows": 4}, 2, {"enabled": True, "rows": [1, 2]}),
        ({"cfm_encoder_graph_rows": [4, 1, 4]}, None, {"enabled": False, "rows": [1, 4]}),
        ({"cfm_encoder_cuda_graph": True, "cfm_encoder_graph_token_widths": [28]}, 20, {"token_widths": [28]}),
        ({"cfm_encoder_cuda_graph": "true"}, 20, {"enabled": False}),
    ],
)
def test_encoder_graph_extras_reach_the_backend_config(extra, max_num_seqs, expected) -> None:
    config = _code2wav(extra, max_num_seqs)._encoder_graph_config
    assert {key: config[key] for key in expected} == expected


def test_encoder_graph_rows_must_be_positive() -> None:
    with pytest.raises(ValueError, match="cfm_encoder_graph_rows"):
        _code2wav({"cfm_encoder_graph_rows": [0, 1]})


@pytest.mark.parametrize("fails", [False, True])
def test_precapture_captures_the_encoder_graphs_after_whole_euler(fails) -> None:
    model = _code2wav({"cfm_encoder_cuda_graph": True})
    calls: list[str] = []
    features = object()

    def encoder(got):
        assert got is features
        calls.append("encoder")
        if fails:
            raise RuntimeError("capture failed")
        return 3

    model.backend = SimpleNamespace(
        precapture_hift=lambda: calls.append("hift") or 0,
        prepare_prompt=lambda *args: features,
        precapture_whole_euler=lambda got: calls.append("whole_euler") or 0,
        precapture_flow_encoder=encoder,
        _encoder_graphs=object(),
    )
    model._default_prompt_normalized = ("/fake/prompt.wav", "shared")

    model._precapture_default_prompt()  # a failed encoder capture only logs: the encoder stays eager

    assert calls == ["hift", "whole_euler", "encoder"]


def test_precapture_skips_the_encoder_without_graphs() -> None:
    model = _code2wav({})
    calls: list[str] = []
    model.backend = SimpleNamespace(
        precapture_hift=lambda: 0,
        prepare_prompt=lambda *args: object(),
        precapture_whole_euler=lambda got: 0,
        precapture_flow_encoder=lambda got: calls.append("encoder"),
        _encoder_graphs=None,
    )
    model._default_prompt_normalized = ("/fake/prompt.wav", "shared")

    model._precapture_default_prompt()

    assert calls == []
