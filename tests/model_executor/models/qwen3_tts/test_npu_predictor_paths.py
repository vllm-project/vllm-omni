# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"npu": "A2"}, num_cards=1)]


def require_npu():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU required")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("top_k", [0, 1, 7, 64])
@torch.inference_mode()
def test_sampling_matches_reference(dtype, top_k):
    require_npu()
    from vllm_omni.platforms.npu.layers.code_predictor_sampling import sample_topk_gumbel

    torch.manual_seed(321)
    for _ in range(3):
        logits = torch.randn(4, 64, device="npu", dtype=dtype)
        uniforms = torch.rand(4, 64, device="npu", dtype=torch.float32).clamp_(1e-6, 1 - 1e-6)
        # Include ties and boundary random values, preserving argmax semantics.
        logits[0].zero_()
        uniforms[0].fill_(0.5)
        uniforms[1, 0] = 0
        uniforms[2, 0] = 1
        scaled = logits * (1 / 0.9)
        if top_k:
            threshold = scaled.topk(top_k, dim=-1).values[:, -1:]
            scaled = scaled.masked_fill(scaled < threshold, -float("inf"))
        expected = (scaled.float() - torch.log(-torch.log(uniforms))).argmax(-1, keepdim=True)
        actual = sample_topk_gumbel(logits, uniforms, top_k, 1 / 0.9)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.inference_mode()
@pytest.mark.parametrize("top_k", [0, 7, 64])
def test_sampling_nonfinite_and_strided_inputs(top_k):
    require_npu()
    from vllm_omni.platforms.npu.layers.code_predictor_sampling import sample_topk_gumbel

    logits = torch.randn(64, 4, device="npu").T
    uniforms = torch.rand(4, 3, 64, device="npu")[:, 1]
    logits[0, 2] = torch.nan
    logits[1].fill_(-torch.inf)
    logits[2, 4] = torch.inf
    scaled = logits * (1 / 0.9)
    if top_k:
        scaled = scaled.masked_fill(scaled < scaled.topk(top_k, dim=-1).values[:, -1:], -torch.inf)
    expected = (scaled.float() - torch.log(-torch.log(uniforms))).argmax(-1, keepdim=True)
    torch.testing.assert_close(sample_topk_gumbel(logits, uniforms, top_k, 1 / 0.9), expected, atol=0, rtol=0)


@torch.inference_mode()
def test_sampling_graph_replay_uses_new_uniforms():
    require_npu()
    from vllm_omni.platforms.npu.layers.code_predictor_sampling import sample_topk_gumbel

    logits = torch.randn(4, 64, device="npu", dtype=torch.bfloat16)
    uniforms = torch.full((4, 64), 0.5, device="npu")
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            sample_topk_gumbel(logits, uniforms, 7, 1 / 0.9)
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph, stream=stream):
            output = sample_topk_gumbel(logits, uniforms, 7, 1 / 0.9)
        for _ in range(3):
            logits.normal_()
            uniforms.uniform_(1e-6, 1 - 1e-6)
            graph.replay()
            scaled = logits * (1 / 0.9)
            scaled = scaled.masked_fill(scaled < scaled.topk(7, dim=-1).values[:, -1:], -float("inf"))
            expected = (scaled.float() - torch.log(-torch.log(uniforms))).argmax(-1, keepdim=True)
            torch.testing.assert_close(output, expected, atol=0, rtol=0)
    finally:
        torch.npu.synchronize()
        graph.reset()


@pytest.mark.parametrize("batch,seq_len", [(1, 1), (1, 17), (4, 17)])
@torch.inference_mode()
def test_native_gqa_matches_expanded_heads_and_replays(batch, seq_len):
    require_npu()
    from vllm_omni.model_executor.models.common.qwen3_code_predictor import CodePredictorAttention
    from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSTalkerCodePredictorConfig

    config = Qwen3TTSTalkerCodePredictorConfig(
        hidden_size=512, num_attention_heads=4, num_key_value_heads=2, head_dim=128, num_code_groups=16
    )
    attention = CodePredictorAttention(config).to(device="npu", dtype=torch.bfloat16).eval()
    q = torch.randn(batch, 4, seq_len, 128, device="npu", dtype=torch.bfloat16)
    k = torch.randn(batch, 2, seq_len, 128, device="npu", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    def forward(native):
        attention.use_npu_native_gqa = native
        return attention._forward_npu_attention(q, k, v, batch, seq_len)

    expected = forward(False)
    torch.testing.assert_close(forward(True), expected, atol=0.05, rtol=0.05)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            forward(True)
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph, stream=stream):
            output = forward(True)
        for _ in range(3):
            q.normal_()
            k.normal_()
            v.normal_()
            graph.replay()
            torch.testing.assert_close(output, forward(False), atol=0.05, rtol=0.05)
        if seq_len > 1:
            first = forward(True)[:, :, :1].clone()
            v[:, :, 1:].fill_(100)
            k[:, :, 1:].fill_(100)
            torch.testing.assert_close(forward(True)[:, :, :1], first, atol=0, rtol=0)
    finally:
        torch.npu.synchronize()
        graph.reset()


@torch.inference_mode()
def test_native_swiglu_matches_packed_mlp_and_replays():
    require_npu()
    from vllm_omni.model_executor.models.common.qwen3_code_predictor import CodePredictorMLP
    from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSTalkerCodePredictorConfig

    config = Qwen3TTSTalkerCodePredictorConfig(hidden_size=128, intermediate_size=256)
    mlp = CodePredictorMLP(config).to(device="npu", dtype=torch.bfloat16).eval()
    x = torch.randn(4, 17, 128, device="npu", dtype=torch.bfloat16)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    mlp.use_npu_fused_swiglu = True
    with torch.npu.stream(stream):
        for _ in range(3):
            mlp(x)
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph, stream=stream):
            output = mlp(x)
        mlp.use_npu_fused_swiglu = False
        for _ in range(3):
            x.normal_()
            graph.replay()
            torch.testing.assert_close(output, mlp(x), atol=0.05, rtol=0.05)
    finally:
        torch.npu.synchronize()
        graph.reset()
