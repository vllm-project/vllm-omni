# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.moss_tts import codec_gemm

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("split", [1, 4])
def test_gemm_rounding_and_capture(mode, split):
    torch.manual_seed(42)
    x = torch.randn(15, 768, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(3072, 768, device="cuda", dtype=x.dtype) / 28
    scale = torch.randn(3072, device="cuda", dtype=x.dtype) * 0.01
    residual = torch.randn(15, 3072, device="cuda", dtype=x.dtype)
    ref = F.linear(x, w)
    ref = F.gelu(ref) if mode == 1 else (residual + ref * scale if mode == 2 else ref)

    def fn():
        return codec_gemm.ffn_gemm(x, w, scale, residual, mode, 16, 64, 64, split)

    value = fn()
    error = (value.float() - ref.float()).norm() / ref.float().norm()
    assert error < 0.004
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fn()
    graph.replay()
    torch.testing.assert_close(captured, value, rtol=0, atol=0)


def test_selected_compile(monkeypatch):
    monkeypatch.setattr(codec_gemm, "CONFIG", {"15,3072,768,1": [16, 64, 64, 1]})
    x = torch.randn(15, 768, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(3072, 768, device="cuda", dtype=x.dtype) / 28
    scale = torch.ones(3072, device="cuda", dtype=x.dtype)
    residual = torch.zeros(15, 3072, device="cuda", dtype=x.dtype)
    fn = torch.compile(codec_gemm.selected_linear, fullgraph=True, dynamic=True)
    for rows in [15, 30]:
        xx = x if rows == 15 else x.repeat(2, 1)
        rr = residual if rows == 15 else residual.repeat(2, 1)
        value = fn(xx, w, scale, rr, 1)
        torch.testing.assert_close(
            value,
            codec_gemm.selected_linear(xx, w, scale, rr, 1),
            atol=0 if rows == 15 else 2e-5,
            rtol=0 if rows == 15 else 0.008,
        )


@torch.inference_mode()
def test_ffn_strict_dynamic_dimensions(monkeypatch):
    from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import (
        MossAudioTokenizerTransformerLayer,
        StreamingExecutionContext,
    )

    monkeypatch.setattr(codec_gemm, "CONFIG", {"30,3072,768,1": [16, 64, 64, 1]})
    layer = MossAudioTokenizerTransformerLayer(
        d_model=768,
        num_heads=12,
        dim_feedforward=3072,
        layer_scale=0.01,
        device="cuda",
        dtype=torch.bfloat16,
    ).eval()
    compiled = torch.compile(layer._ff_block, fullgraph=True, dynamic=True)
    for b, t in [(2, 15), (4, 30)]:
        x = torch.randn(b, t, 768, device="cuda", dtype=torch.bfloat16)
        slots = torch.arange(b, device="cuda")
        valid = torch.ones(b, device="cuda", dtype=torch.bool)
        torch._dynamo.mark_dynamic(x, [0, 1])
        torch._dynamo.mark_dynamic(slots, 0)
        torch._dynamo.mark_dynamic(valid, 0)
        ctx = StreamingExecutionContext(slots, valid)
        value = compiled(x, ctx)
        torch.testing.assert_close(value, layer._ff_block(x, ctx), atol=0.008, rtol=0.008)
