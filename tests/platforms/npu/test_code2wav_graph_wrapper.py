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


def test_codec_enables_npu_wrapper_at_graph_selection(monkeypatch):
    from unittest.mock import Mock

    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeCode2WavConfig

    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_code2wav import Qwen3OmniMoeCode2Wav
    from vllm_omni.platforms.npu.models import code2wav_graph

    decoder = object.__new__(Qwen3OmniMoeCode2Wav)
    torch.nn.Module.__init__(decoder)
    decoder.config = Qwen3OmniMoeCode2WavConfig()
    wrapper = Mock()
    prepare = Mock()
    monkeypatch.setattr(code2wav_graph, "prepare_code2wav_graph_runtime", prepare)
    monkeypatch.setattr(code2wav_graph, "NPUGraphDecoderWrapper", wrapper)
    decoder.enable_cudagraph(device=Mock(type="npu"))
    prepare.assert_called_once_with()
    wrapper.assert_called_once_with(decoder, decoder.config.num_quantizers)
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


def test_graph_runtime_disables_internal_format_and_jit(monkeypatch):
    from unittest.mock import Mock

    from vllm_omni.platforms.npu.models.code2wav_graph import prepare_code2wav_graph_runtime

    npu = Mock()
    npu.config.allow_internal_format = True
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    prepare_code2wav_graph_runtime()
    assert npu.config.allow_internal_format is False
    npu.set_compile_mode.assert_called_once_with(jit_compile=False)
