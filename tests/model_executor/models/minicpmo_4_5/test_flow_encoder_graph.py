# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact-shape flow-encoder graphs (``cfm_encoder_cuda_graph``): keys, eager fallbacks, bit-exact replays."""

import functools
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.model_executor.models.minicpmo_4_5.flow_encoder_graph as graph_module
from vllm_omni.model_executor.models.minicpmo_4_5.flow_encoder_graph import (
    FlowEncoderGraphs,
    reachable_conformer_cache_frames,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

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


@pytest.mark.parametrize(
    ("growths", "expected"),
    [([50], [300, 350, 400]), ([150], [300, 400]), ([30], [300, 330, 360, 390, 400]), ([], [300])],
)
def test_reachable_cache_frames_follow_the_streaming_trim(growths, expected) -> None:
    # 6 s default voice (300 frames); 28-token duplex units grow the cache by 50 frames.
    assert reachable_conformer_cache_frames(start=300, prompt_len=300, suffix=100, growths=growths) == expected


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


@pytest.mark.parametrize(("rows", "frames"), [(1, 8), (3, 18), (3, 8)])
def test_replay_matches_the_eager_encoder_exactly(fake_capture, rows, frames) -> None:
    graphs, encode = _graphs("cpu")
    _check_replay(graphs, encode, rows, frames, "cpu")
    assert graphs.replays == 1


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
