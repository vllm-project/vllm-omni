# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact-shape HiFT graphs: the duplex vocoder shapes outside the capture buckets (CPU, fake graphs)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

import vllm_omni.model_executor.models.minicpmo_4_5.cuda_graph_wrapper as wrapper_module
from vllm_omni.model_executor.models.minicpmo_4_5.cuda_graph_wrapper import HiFTGraphWrapper, codec_frame_range

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MEL_CACHE = 8
_SOURCE_CACHE = 3840


class _HiFT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_pre = nn.Conv1d(80, 4, 1)
        self.inference = Mock(return_value=(torch.tensor([[99.0]]), torch.tensor([[[98.0]]])))
        self.finalize_calls: list[tuple[int, ...]] = []

    def _inference_pre_istft(self, speech_feat, cache_source):  # pragma: no cover - fake graphs never run it
        raise AssertionError("the fake capture never runs the graph function")

    def _finalize_decode(self, magnitude, phase):
        self.finalize_calls.append(tuple(magnitude.shape))
        return magnitude + phase


def _token2wav() -> SimpleNamespace:
    return SimpleNamespace(
        hift=_HiFT(),
        flow=SimpleNamespace(
            encoder=SimpleNamespace(pre_lookahead_layer=SimpleNamespace(pre_lookahead_len=3)),
            token_mel_ratio=2,
        ),
        mel_cache_len=_MEL_CACHE,
        source_cache_len=_SOURCE_CACHE,
    )


def _connector(**extra) -> dict:
    return {
        "codec_chunk_frames": 75,
        "initial_codec_chunk_frames": 25,
        "codec_left_context_frames": 3,
        "hift_graph_codec_chunk_frames": [25, 75],
        **extra,
    }


class _FakeGraph:
    """Replays like the captured HiFT prefix: magnitude = 2 * mel[:, :1], phase = 0, source = mel[:, :1]."""

    def __init__(self, wrapper: HiFTGraphWrapper, key: tuple[int, int, int]) -> None:
        self.wrapper = wrapper
        self.key = key
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1
        mel = self.wrapper.static_speech_inputs[self.key]
        self.wrapper.static_magnitude_outputs[self.key].copy_(2 * mel[:, :1])
        self.wrapper.static_phase_outputs[self.key].zero_()
        self.wrapper.static_cache_source_outputs[self.key].copy_(mel[:, :1])


def _wrapper(monkeypatch: pytest.MonkeyPatch, **extra) -> HiFTGraphWrapper:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    wrapper = HiFTGraphWrapper(_token2wav(), _connector(**extra), capture_batch_sizes=[1, 2])

    def capture(batch_size: int, num_frames: int, cache_len: int) -> None:
        key = (batch_size, num_frames, cache_len)
        if key in wrapper.graph:
            return
        wrapper.graph[key] = _FakeGraph(wrapper, key)
        wrapper.static_speech_inputs[key] = torch.zeros(batch_size, 80, num_frames)
        wrapper.static_cache_source_inputs[key] = torch.zeros(batch_size, 1, cache_len)
        wrapper.static_magnitude_outputs[key] = torch.zeros(batch_size, 1, num_frames)
        wrapper.static_phase_outputs[key] = torch.zeros(batch_size, 1, num_frames)
        wrapper.static_cache_source_outputs[key] = torch.zeros(batch_size, 1, num_frames)

    wrapper._capture = Mock(side_effect=capture)
    return wrapper


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, range(0)), ([], range(0)), ((), range(0)), ([1, 30], range(1, 31)), ([26, 26], range(26, 27))],
)
def test_codec_frame_range_parses_inclusive_ranges(value, expected) -> None:
    assert codec_frame_range(value, name="x") == expected


@pytest.mark.parametrize("value", [[0, 3], [5, 2], [1], [1, 2, 3], 7, "1,30"])
def test_codec_frame_range_rejects_malformed_ranges(value) -> None:
    with pytest.raises(ValueError, match="hift_graph_first_chunk_frames"):
        codec_frame_range(value, name="hift_graph_first_chunk_frames")


def test_exact_shapes_are_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapper = _wrapper(monkeypatch)

    assert wrapper.exact_shapes == []
    assert wrapper._exact_keys == frozenset()
    wrapper.capture()
    # Only the buckets, at every capture batch size: the default is unchanged.
    assert sorted(wrapper.graph) == sorted(
        (b, frames, cache) for b in (1, 2) for frames, cache in ((150, 0), (158, 3840), (50, 0), (58, 3840))
    )
    assert wrapper.finalize_fn.__self__.finalize_calls == []


def test_exact_shapes_cover_first_chunks_and_continuations_outside_the_buckets(monkeypatch) -> None:
    wrapper = _wrapper(
        monkeypatch,
        hift_graph_first_chunk_frames=[1, 30],
        hift_graph_continuation_frames=[26, 74],
    )

    first = [(2 * f, 0) for f in range(1, 31) if f != 25]  # (50, 0) is the first-chunk bucket
    continuation = [(_MEL_CACHE + 2 * f, _SOURCE_CACHE) for f in range(26, 75)]
    assert wrapper.exact_shapes == first + continuation
    assert (14, 0) in wrapper.exact_shapes  # dx3's most frequent eager fallback (1, 14, 0)
    assert (62, _SOURCE_CACHE) in wrapper.exact_shapes  # and (1, 62, 3840)
    assert wrapper._exact_keys == frozenset((1, frames, cache) for frames, cache in wrapper.exact_shapes)


def test_continuation_range_skips_the_bucket_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapper = _wrapper(monkeypatch, hift_graph_continuation_frames=[24, 26])

    # f=25 is the duplex unit continuation bucket (58, 3840).
    assert wrapper.exact_shapes == [(56, _SOURCE_CACHE), (60, _SOURCE_CACHE)]


def test_capture_takes_exact_shapes_at_the_exact_batch_sizes_only(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapper = _wrapper(monkeypatch, hift_graph_first_chunk_frames=[6, 8])
    hift = wrapper.finalize_fn.__self__

    wrapper.capture()

    exact = {key for key in wrapper.graph if (key[1], key[2]) in {(12, 0), (14, 0), (16, 0)}}
    assert exact == {(1, 12, 0), (1, 14, 0), (1, 16, 0)}
    # Each exact shape builds its ISTFT envelope at capture time, not on its first live replay.
    assert hift.finalize_calls == [(1, 1, 12), (1, 1, 14), (1, 1, 16)]


def test_exact_batch_sizes_are_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapper = _wrapper(monkeypatch, hift_graph_first_chunk_frames=[7, 7], hift_graph_exact_batch_sizes=[1, 2])

    wrapper.capture()

    assert {(1, 14, 0), (2, 14, 0)} <= set(wrapper.graph)


def test_exact_shape_replays_its_own_graph_unpadded(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapper = _wrapper(monkeypatch, hift_graph_first_chunk_frames=[7, 7])
    wrapper.capture()
    wrapper._capture.reset_mock()
    speech_feat = torch.randn(1, 80, 14)

    speech, source = wrapper.replay(speech_feat, torch.zeros(1, 1, 0))

    graph = wrapper.graph[(1, 14, 0)]
    assert graph.replays == 1
    assert wrapper.static_speech_inputs[(1, 14, 0)].shape == (1, 80, 14)
    torch.testing.assert_close(speech, 2 * speech_feat[:, :1], rtol=0, atol=0)
    torch.testing.assert_close(source, speech_feat[:, :1], rtol=0, atol=0)
    wrapper._capture.assert_not_called()
    wrapper.decode_fn.assert_not_called()


def test_exact_shape_at_other_batch_sizes_stays_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the captured batch sizes replay: no padding to a larger graph, no lazy capture."""
    wrapper = _wrapper(monkeypatch, hift_graph_first_chunk_frames=[7, 7])
    wrapper.capture()
    wrapper._capture.reset_mock()
    speech_feat = torch.randn(2, 80, 14)
    cache_source = torch.zeros(2, 1, 0)

    result = wrapper.replay(speech_feat, cache_source)

    wrapper._capture.assert_not_called()
    wrapper.decode_fn.assert_called_once_with(speech_feat, cache_source)
    assert result is wrapper.decode_fn.return_value


def test_shapes_outside_the_ranges_still_run_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapper = _wrapper(monkeypatch, hift_graph_first_chunk_frames=[7, 7])
    wrapper.capture()
    speech_feat = torch.randn(1, 80, 16)
    cache_source = torch.zeros(1, 1, 0)

    wrapper.replay(speech_feat, cache_source)

    wrapper.decode_fn.assert_called_once_with(speech_feat, cache_source)


def test_bucket_shapes_keep_their_padded_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapper = _wrapper(monkeypatch, hift_graph_first_chunk_frames=[1, 30])
    wrapper.capture()
    speech_feat = torch.randn(1, 80, 50)

    wrapper.replay(speech_feat, torch.zeros(1, 1, 0))

    assert wrapper.graph[(1, 50, 0)].replays == 1
    wrapper.decode_fn.assert_not_called()


def test_capture_exact_reports_memory_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapper = _wrapper(monkeypatch, hift_graph_first_chunk_frames=[2, 3])
    assert wrapper_module._device_used_bytes(torch.device("cpu")) is None

    assert wrapper.capture_exact() == 2
    assert wrapper.capture_exact() == 0


def test_exact_shape_extras_reach_the_connector_config() -> None:
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import MiniCPMO45Code2Wav

    def connector(extra: dict) -> dict:
        config = SimpleNamespace(
            model_config=SimpleNamespace(model="/fake/model", stage_connector_config={"extra": extra})
        )
        return MiniCPMO45Code2Wav(vllm_config=config)._connector_config

    default = connector({})
    assert default["hift_graph_first_chunk_frames"] == []
    assert default["hift_graph_continuation_frames"] == []
    assert default["hift_graph_exact_batch_sizes"] == [1]
    configured = connector(
        {
            "hift_graph_first_chunk_frames": [1, 30],
            "hift_graph_continuation_frames": [26, 74],
            "hift_graph_exact_batch_sizes": [1, 2],
        }
    )
    assert configured["hift_graph_first_chunk_frames"] == [1, 30]
    assert configured["hift_graph_continuation_frames"] == [26, 74]
    assert configured["hift_graph_exact_batch_sizes"] == [1, 2]
