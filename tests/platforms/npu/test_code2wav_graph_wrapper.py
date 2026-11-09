# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""NPU codec graph routing preserves exact shapes and short-output chunks."""

import pytest
import torch

from vllm_omni.platforms.npu.models.code2wav_graph import NPUGraphDecoderWrapper

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _ShortDecoder(torch.nn.Module):
    total_upsample = 4

    def forward(self, codes, *, graph_safe_mask=False):
        return codes[:, :1].float().repeat_interleave(4, dim=-1)[..., :-1]


def test_exact_shapes_and_outputs_are_preserved(monkeypatch):
    decoder = _ShortDecoder()
    wrapper = NPUGraphDecoderWrapper(decoder, 1)
    calls = []

    def run(operation, inputs, constants, compute):
        calls.append(tuple(inputs[0].shape))
        return compute(*inputs)

    monkeypatch.setattr(wrapper.runner, "run", run)
    for length in (3, 5, 3):
        codes = torch.arange(length).reshape(1, 1, length)
        torch.testing.assert_close(wrapper.decode(codes), decoder(codes), rtol=0, atol=0)
    assert calls == [(1, 1, 3), (1, 1, 5), (1, 1, 3)]


def test_chunked_decode_preserves_short_output_trim():
    decoder = _ShortDecoder()
    wrapper = NPUGraphDecoderWrapper(decoder, 1)
    codes = torch.arange(13).reshape(1, 1, 13)
    expected = torch.cat(
        [decoder(codes[..., :5]), decoder(codes[..., 3:10])[..., 8:], decoder(codes[..., 8:13])[..., 8:]], dim=-1
    )
    actual = wrapper.chunked_decode_with_cudagraph(codes, chunk_size=5, left_context_size=2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class _FunctionalGraph:
    def __init__(self, compute):
        self.compute = compute

    def replay(self, inputs):
        return tuple(output.detach().clone() for output in self.compute(*inputs))


def _capturing_wrapper(monkeypatch, decoder):
    wrapper = NPUGraphDecoderWrapper(decoder, 1)
    captured = []

    def capture(inputs, compute):
        captured.append(tuple(inputs[0].shape))
        return _FunctionalGraph(compute)

    monkeypatch.setattr(wrapper.runner, "_eligible", lambda inputs: True)
    monkeypatch.setattr(wrapper.runner, "capture", capture)
    return wrapper, captured


def test_warmup_captures_planned_shapes_then_serves_others_eagerly(monkeypatch):
    decoder = _ShortDecoder()
    wrapper, captured = _capturing_wrapper(monkeypatch, decoder)
    wrapper.warmup(torch.device("cpu"), [(1, 4), (2, 6)])
    assert captured == [(1, 1, 4), (2, 1, 6)]

    for shape in [(2, 1, 6), (3, 1, 6), (1, 1, 5)]:
        codes = torch.arange(shape[0] * shape[2]).reshape(*shape)
        torch.testing.assert_close(wrapper.decode(codes), decoder(codes), rtol=0, atol=0)
    assert captured == [(1, 1, 4), (2, 1, 6)]
    assert wrapper.runner.stats == {"captures": 2, "failed": 0, "hits": 1}


def test_default_streaming_plan_matches_talker_chunk_windows():
    from vllm_omni.platforms.npu.models.code2wav_graph import plan_code2wav_graph_shapes

    # initial_codec_chunk_frames=4, codec_chunk_frames=25, left context 25:
    # chunk 0 decodes 4 frames, chunk 1 decodes 4 + 25, then 25 + 25.
    shapes = plan_code2wav_graph_shapes(steady_window=50, start_windows=[4, 29], max_batch_size=64)
    assert shapes == [(1, 4), (1, 29), *((rows, 50) for rows in range(1, 31))]
    assert plan_code2wav_graph_shapes(steady_window=50, start_windows=[50], max_batch_size=8) == [
        (rows, 50) for rows in range(1, 9)
    ]
    assert plan_code2wav_graph_shapes(steady_window=50, batch_sizes=[4, 16, 2], max_batch_size=8) == [
        (1, 50),
        (2, 50),
        (4, 50),
    ]


@pytest.mark.parametrize(
    ("extra", "streaming", "expected_start"),
    [
        ({"initial_codec_chunk_frames": 4, "codec_chunk_frames": 25, "codec_left_context_frames": 25}, True, [4, 29]),
        (
            {"codec_chunk_ramp": [2, 8, 25], "codec_chunk_frames": 25, "codec_left_context_frames": 25},
            True,
            [2, 10, 35],
        ),
        ({"codec_chunk_frames": 25, "codec_left_context_frames": 25}, True, [25]),
        ({"codec_chunk_frames": 25, "codec_left_context_frames": 25}, False, [300]),
    ],
)
def test_qwen3_omni_plans_npu_shapes_from_connector_config(extra, streaming, expected_start):
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import Qwen3OmniMoeForConditionalGeneration

    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    model.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(stage_connector_config={"extra": extra}),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    steady = 50 if streaming else 325
    shapes = model._plan_npu_code2wav_graph_shapes(streaming)
    assert shapes == [*((1, window) for window in expected_start), *((rows, steady) for rows in range(1, 5))]


def test_codec_captures_npu_graphs_before_enabling(monkeypatch):
    from unittest.mock import Mock

    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeCode2WavConfig

    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_code2wav import Qwen3OmniMoeCode2Wav
    from vllm_omni.platforms.npu.models import code2wav_graph

    decoder = object.__new__(Qwen3OmniMoeCode2Wav)
    torch.nn.Module.__init__(decoder)
    decoder.config = Qwen3OmniMoeCode2WavConfig()
    decoder._cudagraph_wrapper = None
    decoder._cudagraph_enabled = False
    wrapper = Mock()
    wrapper.return_value.warmup.side_effect = RuntimeError("capture failed")
    monkeypatch.setattr(code2wav_graph, "prepare_code2wav_graph_runtime", Mock())
    monkeypatch.setattr(code2wav_graph, "NPUGraphDecoderWrapper", wrapper)
    device = Mock(type="npu")

    with pytest.raises(RuntimeError, match="capture failed"):
        decoder.enable_cudagraph(device=device, npu_graph_shapes=[(1, 4)])
    assert not decoder._cudagraph_enabled
    assert decoder._cudagraph_wrapper is None

    wrapper.return_value.warmup.side_effect = None
    decoder.enable_cudagraph(device=device, npu_graph_shapes=[(1, 4), (2, 50)])
    wrapper.return_value.warmup.assert_called_with(device, [(1, 4), (2, 50)])
    assert decoder._cudagraph_enabled
    assert decoder._cudagraph_wrapper is wrapper.return_value


@pytest.mark.parametrize("length", [1, 7, 9, 19])
def test_graph_masks_match_transformers(length):
    from unittest.mock import Mock

    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeCode2WavConfig

    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_code2wav import Qwen3OmniMoeCode2Wav

    decoder = object.__new__(Qwen3OmniMoeCode2Wav)
    torch.nn.Module.__init__(decoder)
    decoder.config = Qwen3OmniMoeCode2WavConfig(sliding_window=8)
    decoder.config._attn_implementation = "eager"
    decoder.pre_transformer = Mock(has_sliding_layers=True)
    hidden = torch.empty(2, length, 16)
    actual = decoder._graph_attention_mask(hidden)
    args = dict(config=decoder.config, inputs_embeds=hidden, attention_mask=None, past_key_values=None)
    for name, create in [
        ("full_attention", create_causal_mask),
        ("sliding_attention", create_sliding_window_causal_mask),
    ]:
        expected = create(**args)
        torch.testing.assert_close(actual[name].expand_as(expected), expected, rtol=0, atol=0)
