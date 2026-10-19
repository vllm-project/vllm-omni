# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.common.snake_activation import SnakeBeta

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"npu": "A2"}, num_cards=1)]


@pytest.mark.parametrize("shape", [(1, 64, 1920), (4, 32, 480), (2, 8, 17)])
@torch.inference_mode()
def test_npu_fused_matches_eager(shape):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    torch.manual_seed(42)
    snake = SnakeBeta(shape[1]).to(device="npu", dtype=torch.bfloat16).eval()
    snake.alpha.uniform_(-0.5, 0.5)
    snake.beta.uniform_(-0.5, 0.5)
    snake.precompute_exp_cache()
    x = torch.randn(shape, device="npu", dtype=torch.bfloat16)
    expected = snake(x)
    snake.npu_fused = True
    actual = snake(x)
    torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.01)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_unsupported_layout_uses_eager(dtype):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    snake = SnakeBeta(8).to(device="npu", dtype=dtype).eval()
    snake.precompute_exp_cache()
    x = torch.randn(2, 8, 34, device="npu", dtype=dtype)[..., ::2]
    expected = snake(x)
    snake.npu_fused = True
    torch.testing.assert_close(snake(x), expected, atol=0, rtol=0)


@torch.inference_mode()
def test_small_decoder_combines_codebook_cache_and_snake():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    from tests.model_executor.models.qwen3_tts.test_time_major_decoder import _make_decoder
    from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        EuclideanCodebook,
    )

    decoder = _make_decoder().to(device="npu", dtype=torch.bfloat16).eval()
    decoder.precompute_snake_caches()
    for module in decoder.modules():
        if isinstance(module, EuclideanCodebook):
            module.embedding_sum.normal_()
            module.cluster_usage.uniform_(0.5, 2.0)
    codes = torch.randint(0, 32, (1, 2, 4), device="npu")
    expected = decoder(codes)
    for module in decoder.modules():
        if isinstance(module, EuclideanCodebook):
            module.precompute_embedding()
        elif isinstance(module, SnakeBeta):
            module.npu_fused = True
    actual = decoder(codes)
    torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.01)


@torch.inference_mode()
def test_fused_snake_graph_replay():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    snake = SnakeBeta(8).to(device="npu", dtype=torch.bfloat16).eval()
    snake.precompute_exp_cache()
    snake.npu_fused = True
    x = torch.randn(2, 8, 128, device="npu", dtype=torch.bfloat16)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            snake(x)
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph, stream=stream):
            actual = snake(x)
        for _ in range(2):
            x.normal_()
            graph.replay()
            torch.testing.assert_close(actual, snake._eager_forward(x), atol=0.03125, rtol=0.01)
    finally:
        torch.npu.synchronize()
        graph.reset()
