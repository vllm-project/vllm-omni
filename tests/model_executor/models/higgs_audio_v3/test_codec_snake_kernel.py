# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The fused DAC Snake / decoder forward is bit-identical to transformers' DacDecoder."""

import copy

import pytest
import torch
import torch.nn as nn

from tests.helpers.mark import hardware_test
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model]

requires_cuda = pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="requires CUDA")

_DAC_CONFIG = {
    "codebook_dim": 8,
    "codebook_size": 1024,
    "decoder_hidden_size": 256,
    "downsampling_ratios": [8, 5, 4, 2, 3],
    "encoder_hidden_size": 16,
    "hidden_size": 64,
    "hop_length": 960,
    "n_codebooks": 9,
    "sampling_rate": 16000,
    "upsampling_ratios": [8, 5, 4, 2, 3],
}


def _dac_decoder_pair(device, dtype):
    from transformers import DacConfig, DacModel

    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_decoder import (
        _fuse_dac_decoder,
        adjust_conv_transpose_output_padding,
    )

    torch.manual_seed(0)
    reference = DacModel(DacConfig(**_DAC_CONFIG)).decoder
    adjust_conv_transpose_output_padding(reference)
    with torch.no_grad():
        for name, param in reference.named_parameters():
            if name.endswith("alpha"):
                param.copy_(torch.rand_like(param) * 2 + 0.05)
            elif name.endswith("bias"):
                param.normal_(0.0, 0.1)
    reference = reference.to(device, dtype).eval()
    fused = copy.deepcopy(reference)
    assert _fuse_dac_decoder(fused) == 1 + 7 * len(reference.block)
    fused.load_state_dict(reference.state_dict())
    return reference, fused


@requires_cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 64, 33), (3, 1024, 8), (2, 96, 5000)])
@torch.inference_mode()
def test_fused_snake_matches_transformers_snake(dtype, shape):
    from transformers.models.dac.modeling_dac import Snake1d

    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_decoder import _FusedDacSnake1d

    torch.manual_seed(0)
    reference = Snake1d(shape[1]).to("cuda", dtype)
    reference.alpha.copy_(torch.rand_like(reference.alpha) * 4 + 0.05)
    fused = _FusedDacSnake1d(shape[1]).to("cuda", dtype)
    fused.load_state_dict(reference.state_dict())
    x = (torch.randn(shape, device="cuda") * 3).to(dtype)
    residual = (torch.randn(shape, device="cuda") * 3).to(dtype)
    bias = (torch.randn(shape[1], device="cuda") * 3).to(dtype)
    assert torch.equal(fused(x), reference(x))

    total, out = fused.fused(x, bias, residual, write_sum=True)
    expected_total = residual + (x + bias.reshape(1, -1, 1))
    assert torch.equal(total, expected_total)
    assert torch.equal(out, reference(expected_total))
    assert fused.fused(x, bias)[0] is None
    assert torch.equal(fused.fused(x, bias)[1], reference(x + bias.reshape(1, -1, 1)))


@requires_cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batch,frames", [(1, 33), (4, 8)])
def test_fused_dac_decoder_matches_transformers_decoder(dtype, batch, frames):
    reference, fused = _dac_decoder_pair("cuda", dtype)
    x = torch.randn(batch, _DAC_CONFIG["hidden_size"], frames, device="cuda").to(dtype)
    with torch.inference_mode():
        assert torch.equal(fused(x), reference(x))

    # Reloaded weights must not reuse the cached Snake reciprocals.
    with torch.no_grad():
        for module in (reference, fused):
            module.snake1.alpha.mul_(0.5)
    with torch.inference_mode():
        assert torch.equal(fused(x), reference(x))


@pytest.mark.cpu
def test_fuse_dac_decoder_keeps_state_dict_and_falls_back_on_cpu():
    from transformers.models.dac.modeling_dac import Snake1d

    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_decoder import _FusedDacSnake1d

    reference, fused = _dac_decoder_pair("cpu", torch.float32)
    assert not any(isinstance(module, Snake1d) for module in fused.modules())
    assert set(fused.state_dict()) == set(reference.state_dict())
    x = torch.randn(2, _DAC_CONFIG["hidden_size"], 5)
    with torch.no_grad():
        assert torch.equal(fused(x), reference(x))

    decoder = nn.Sequential(Snake1d(8), nn.Conv1d(8, 8, 3))
    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_decoder import _fuse_dac_decoder

    assert _fuse_dac_decoder(decoder) == 1
    assert isinstance(decoder[0], _FusedDacSnake1d)
    assert "forward" not in vars(decoder)


@requires_cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_fused_decoder_graph_replays_changed_inputs(dtype):
    reference, fused = _dac_decoder_pair("cuda", dtype)
    x = torch.randn(2, _DAC_CONFIG["hidden_size"], 8, device="cuda").to(dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fused(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = fused(x)
    for _ in range(2):
        x.normal_()
        graph.replay()
        assert torch.equal(output, reference(x))


@requires_cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.inference_mode()
def test_fused_snake_uses_eager_on_non_nvidia_platform(monkeypatch):
    from transformers.models.dac.modeling_dac import Snake1d

    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_decoder import _FusedDacSnake1d

    reference = Snake1d(8).cuda()
    fused = _FusedDacSnake1d(8).cuda()
    fused.load_state_dict(reference.state_dict())
    x = torch.randn(2, 8, 33, device="cuda")
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: False)
    assert not fused.can_fuse(x)
    assert torch.equal(fused(x), reference(x))


@requires_cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.inference_mode()
def test_fused_snake_preserves_float64_precision():
    from transformers.models.dac.modeling_dac import Snake1d

    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_decoder import _FusedDacSnake1d

    reference = Snake1d(8).to("cuda", torch.float64)
    fused = _FusedDacSnake1d(8).to("cuda", torch.float64)
    fused.load_state_dict(reference.state_dict())
    x = torch.randn(2, 8, 33, device="cuda", dtype=torch.float64)
    assert not fused.can_fuse(x)
    assert torch.equal(fused(x), reference(x))
