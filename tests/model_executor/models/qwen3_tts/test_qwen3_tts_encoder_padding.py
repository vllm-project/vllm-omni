# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import copy

import pytest
import torch
from torch import nn
from transformers import MimiConfig
from transformers.models.mimi.modeling_mimi import MimiConv1d, MimiConv1dPaddingCache

from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.mimi_conv import (
    _integer_padding_forward,
    use_integer_padding,
)
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Encoder,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.usefixtures("single_threaded_cpu")]


def _conv(*, causal=True, mode="constant", kernel=7, stride=3, dilation=1):
    config = MimiConfig(use_causal_conv=causal, pad_mode=mode)
    return MimiConv1d(config, 4, 4, kernel, stride=stride, dilation=dilation, layer_idx=0).eval()


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("mode", ["constant", "reflect", "replicate"])
@pytest.mark.parametrize("kernel,stride,dilation", [(1, 1, 1), (3, 1, 1), (7, 3, 1), (3, 2, 2)])
@torch.inference_mode()
def test_integer_padding_matches_mimi(causal, mode, kernel, stride, dilation):
    original = _conv(causal=causal, mode=mode, kernel=kernel, stride=stride, dilation=dilation)
    candidate = copy.deepcopy(original)
    use_integer_padding(candidate)
    # Loop over boundary lengths without multiplying the collected test matrix.
    for length in sorted({1, stride, stride + 1, 17}):
        x = torch.randn(2, 4, length)
        saved = x.clone()
        expected, actual = original(x), candidate(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(x, saved, rtol=0, atol=0)
        assert actual.stride() == expected.stride()
        assert actual.dtype == expected.dtype


@pytest.mark.parametrize("mode", ["constant", "replicate"])
@pytest.mark.parametrize("kernel,stride", [(3, 1), (7, 3)])
@torch.inference_mode()
def test_integer_padding_preserves_streaming_cache(mode, kernel, stride):
    original = _conv(mode=mode, kernel=kernel, stride=stride)
    candidate = copy.deepcopy(original)
    use_integer_padding(candidate)
    caches = [MimiConv1dPaddingCache(1, [kernel - stride], [mode], [4]) for _ in range(2)]
    for length in (stride, stride + 1, 17):
        x = torch.randn(2, 4, length)
        torch.testing.assert_close(candidate(x, caches[1]), original(x, caches[0]), rtol=0, atol=0)
        torch.testing.assert_close(caches[1].padding_cache[0], caches[0].padding_cache[0], rtol=0, atol=0)


@torch.inference_mode()
def test_padding_override_is_local_and_preserves_state(mocker):
    conv = _conv()
    unrelated = _conv()
    state = {name: value.clone() for name, value in conv.state_dict().items()}
    params = dict(conv.named_parameters())
    buffers = dict(conv.named_buffers())
    use_integer_padding(nn.Sequential(conv))
    use_integer_padding(conv)  # Reapplying must not wrap or replace any state.
    assert conv.forward.__func__ is _integer_padding_forward
    assert unrelated.forward.__func__ is MimiConv1d.forward
    assert dict(conv.named_parameters()).keys() == params.keys()
    assert all(dict(conv.named_parameters())[name] is value for name, value in params.items())
    assert all(dict(conv.named_buffers())[name] is value for name, value in buffers.items())
    assert conv.state_dict().keys() == state.keys()
    for name, value in state.items():
        torch.testing.assert_close(conv.state_dict()[name], value, rtol=0, atol=0)
    # Avoiding upstream device-scalar padding arithmetic is the optimization.
    mocker.patch.object(conv, "_get_extra_padding_for_conv1d", side_effect=AssertionError("tensor padding used"))
    conv(torch.randn(1, 4, 17))


def test_encoder_constructor_installs_integer_padding(mocker):
    config = MimiConfig(
        hidden_size=16,
        num_filters=4,
        num_residual_layers=1,
        upsampling_ratios=[2, 2],
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=32,
        codebook_size=16,
        codebook_dim=16,
        vector_quantization_hidden_dimension=16,
        num_quantizers=4,
        num_semantic_quantizers=1,
        upsample_groups=1,
        use_streaming=False,
    )
    encoder = Qwen3TTSTokenizerV2Encoder(config).eval()
    convs = [module for module in encoder.modules() if isinstance(module, MimiConv1d)]
    assert convs
    assert encoder.decoder is None
    assert encoder.upsample is None
    assert encoder.decoder_transformer is None
    assert all(module.forward.__func__ is _integer_padding_forward for module in convs)
    # The local implementation must keep the strict checkpoint-loading schema.
    mocker.patch(
        "vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2.use_integer_padding"
    )
    original = Qwen3TTSTokenizerV2Encoder(config).eval()
    original.load_state_dict(encoder.state_dict(), strict=True)
    with torch.inference_mode():
        x = torch.randn(1, 1, 128)
        torch.testing.assert_close(encoder.encoder(x), original.encoder(x), rtol=0, atol=0)
