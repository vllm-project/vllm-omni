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
def test_incremental_kv_matches_prefill_and_resets_frames():
    require_npu()
    from vllm_omni.model_executor.models.common.qwen3_code_predictor import CodePredictorBaseModel
    from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSTalkerCodePredictorConfig

    config = Qwen3TTSTalkerCodePredictorConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_code_groups=4,
    )
    torch.manual_seed(171)
    model = CodePredictorBaseModel(config).to(device="npu", dtype=torch.bfloat16).eval()
    for batch in (1, 3, 1):
        x = torch.randn(batch, 5, 64, device="npu", dtype=torch.bfloat16)
        positions = torch.arange(5, device="npu").expand(batch, -1)
        expected = model(x, positions)[:, :4]
        caches = [
            (
                torch.zeros(batch, 2, 5, 16, device="npu", dtype=torch.bfloat16),
                torch.zeros(batch, 2, 5, 16, device="npu", dtype=torch.bfloat16),
            )
            for _ in model.layers
        ]
        outputs = []
        for step in (1, 2, 3):
            lo = 0 if step == 1 else step
            pos = torch.arange(lo, step + 1, device="npu")
            mask = (torch.arange(5, device="npu")[None, :] <= pos[:, None])[None, None]
            outputs.append(model.forward_cached(x[:, lo : step + 1], pos.expand(batch, -1), caches, pos, mask))
        torch.testing.assert_close(torch.cat(outputs, dim=1), expected, atol=0.05, rtol=0.05)


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


@torch.inference_mode()
def test_kv_graph_replay_discards_previous_frame():
    require_npu()
    from vllm_omni.model_executor.models.common.qwen3_code_predictor import CodePredictorBaseModel
    from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSTalkerCodePredictorConfig

    config = Qwen3TTSTalkerCodePredictorConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_code_groups=4,
    )
    model = CodePredictorBaseModel(config).to(device="npu", dtype=torch.bfloat16).eval()
    x = torch.randn(3, 5, 64, device="npu", dtype=torch.bfloat16)
    buffer = torch.zeros(2, 2, 3, 2, 5, 16, device="npu", dtype=torch.bfloat16)
    caches = [(layer[0], layer[1]) for layer in buffer]
    steps = []
    for step in (1, 2, 3):
        lo = 0 if step == 1 else step
        pos = torch.arange(lo, step + 1, device="npu")
        mask = (torch.arange(5, device="npu")[None, :] <= pos[:, None])[None, None]
        steps.append((lo, step + 1, pos, mask))

    def frame():
        buffer.zero_()
        return torch.cat(
            [model.forward_cached(x[:, lo:hi], pos.expand(3, -1), caches, pos, mask) for lo, hi, pos, mask in steps],
            dim=1,
        )

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            frame()
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph, stream=stream):
            result = frame()
        for _ in range(3):
            x.normal_()
            buffer.fill_(torch.nan)
            graph.replay()
            expected = model(x, torch.arange(5, device="npu").expand(3, -1))[:, :4]
            torch.testing.assert_close(result, expected, atol=0.05, rtol=0.05)
    finally:
        torch.npu.synchronize()
        graph.reset()
