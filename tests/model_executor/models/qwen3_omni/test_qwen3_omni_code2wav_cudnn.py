# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeCode2WavConfig

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_code2wav import Qwen3OmniMoeCode2Wav

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="requires CUDA"),
]


def _codec(enabled):
    config = Qwen3OmniMoeCode2WavConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        decoder_dim=32,
        upsample_rates=[2, 2],
        upsampling_ratios=[2],
        codebook_size=16,
        num_quantizers=2,
    )
    config._attn_implementation = "sdpa"
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=config, stage_connector_config={"extra": {"codec_cudnn_benchmark": enabled}}
        )
    )
    model = Qwen3OmniMoeCode2Wav(vllm_config=vllm_config).cuda().eval()
    model.precompute_snake_caches()
    return model


def _flags():
    c = torch.backends.cudnn
    return c.enabled, c.benchmark, c.benchmark_limit, c.deterministic, c.allow_tf32


@torch.inference_mode()
def test_autotuned_codec_graph_replay_restores_flags():
    torch.manual_seed(21)
    model = _codec(True)
    codes = torch.randint(0, 16, (2, 2, 8), device="cuda")
    original = _flags()
    for _ in range(3):
        model(codes)
    assert _flags() == original
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = model(codes)
    assert _flags() == original
    codes.random_(0, 16)
    graph.replay()
    expected = model(codes)
    torch.testing.assert_close(captured, expected)
    assert _flags() == original


@pytest.mark.parametrize("enabled", [False, True])
@torch.inference_mode()
def test_codec_failure_restores_cudnn_settings(mocker, enabled):
    model = _codec(enabled)
    codes = torch.randint(0, 16, (1, 2, 8), device="cuda")
    original = _flags()

    def fail(hidden):
        if enabled:
            assert torch.backends.cudnn.benchmark
            assert torch.backends.cudnn.benchmark_limit == 10
        else:
            assert _flags() == original
        raise RuntimeError("decode failed")

    mocker.patch.object(model, "_decode_waveform", side_effect=fail)
    with pytest.raises(RuntimeError, match="decode failed"):
        model(codes)
    assert _flags() == original
