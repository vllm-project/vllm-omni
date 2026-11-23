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


@pytest.mark.parametrize("batch", [1, 4, 8])
@pytest.mark.parametrize("native_gqa", [False, True])
@torch.inference_mode()
def test_cached_predictor_matches_prefill_and_resets_each_frame(batch, native_gqa):
    require_npu()
    from vllm_omni.model_executor.models.common.qwen3_code_predictor import CodePredictorBaseModel
    from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSTalkerCodePredictorConfig

    torch.manual_seed(123)
    config = Qwen3TTSTalkerCodePredictorConfig(
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        num_code_groups=16,
        vocab_size=64,
    )
    model = CodePredictorBaseModel(config).to(device="npu", dtype=torch.bfloat16).eval()
    for layer in model.layers:
        layer.self_attn.use_npu_native_gqa = native_gqa
    inputs = torch.randn(batch, 16, 512, device="npu", dtype=torch.bfloat16)
    positions = torch.arange(16, device="npu").expand(batch, -1)
    caches = [
        (
            torch.zeros(batch, 2, 16, 128, device="npu", dtype=torch.bfloat16),
            torch.zeros(batch, 2, 16, 128, device="npu", dtype=torch.bfloat16),
        )
        for _ in model.layers
    ]
    steps = []
    for step in range(1, 16):
        begin = 0 if step == 1 else step
        writes = torch.arange(begin, step + 1, device="npu")
        mask = (torch.arange(16, device="npu")[None, :] <= writes[:, None])[None, None]
        steps.append((begin, step + 1, writes, mask))

    def compute():
        outputs = []
        for begin, end, writes, mask in steps:
            hidden = model.forward_cached(inputs[:, begin:end], positions[:, begin:end], caches, writes, mask)
            outputs.append(hidden[:, -1:])
        return torch.cat(outputs, dim=1)

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            compute()
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph, stream=stream):
            output = compute()
        for _ in range(2):
            inputs.normal_()
            for keys, values in caches:
                keys.fill_(100)
                values.fill_(100)
            graph.replay()
            expected = model(inputs, positions)[:, 1:]
            torch.testing.assert_close(output, expected, atol=0.06, rtol=0.06)
    finally:
        torch.npu.synchronize()
        graph.reset()


@pytest.mark.parametrize("batch", [1, 4, 8])
@torch.inference_mode()
def test_frame_graph_fused_sampling_matches_reference(batch):
    require_npu()
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.common.qwen3_code_predictor import CodePredictorBaseModel
    from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSTalkerCodePredictorConfig
    from vllm_omni.platforms.npu.layers.predictor_frame_graph import PredictorFrameGraph

    torch.manual_seed(719)
    config = Qwen3TTSTalkerCodePredictorConfig(
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        num_code_groups=16,
        vocab_size=64,
    )
    model = CodePredictorBaseModel(config).to(device="npu", dtype=torch.bfloat16).eval()
    for layer in model.layers:
        layer.self_attn.use_npu_native_gqa = True
    predictor = SimpleNamespace(
        _num_groups=16,
        model=model,
        config=config,
        _proj_buf=torch.zeros(batch, 17, 512, device="npu", dtype=torch.bfloat16),
        _bucket_pos_ids={batch: torch.arange(17, device="npu").expand(batch, -1)},
        _compiled_model_fwd=model,
        _lm_heads_list=[torch.nn.Linear(512, 64, bias=False, device="npu", dtype=torch.bfloat16) for _ in range(15)],
        _codec_embeds_list=[torch.nn.Embedding(64, 512, device="npu", dtype=torch.bfloat16) for _ in range(15)],
        small_to_mtp_projection=torch.nn.Identity(),
        _kv_requested=True,
        _fused_sampling=False,
    )
    graphs = []
    try:
        for fused in (False, True):
            predictor._fused_sampling = fused
            graph = PredictorFrameGraph(predictor, batch, 50, 1 / 0.9)
            graph.uniforms.fill_(0.5)
            graph.codes[:, 0] = 0
            graph.capture()
            graphs.append(graph)
        for _ in range(3):
            inputs = torch.randn(batch, 2, 512, device="npu", dtype=torch.bfloat16)
            draws = torch.rand(batch, 15, 64, device="npu").clamp_min_(1e-20)
            outputs = []
            for graph in graphs:
                predictor._proj_buf[:, :2].copy_(inputs)
                outputs.append(graph.run(draws).clone())
            torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    finally:
        torch.npu.synchronize()
        for graph in graphs:
            graph.graph.reset()


@pytest.mark.parametrize("batch", [1, 4, 8])
@pytest.mark.parametrize("steps", [1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_rope_cache_preserves_eager_rounding_and_unwritten_slots(batch, steps, dtype):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    from vllm_omni.model_executor.models.common.qwen3_code_predictor import _rotate_half
    from vllm_omni.platforms.npu.layers.predictor_rope_cache import rope_cache

    q = torch.randn(batch, steps, 16, 128, device="npu", dtype=dtype).transpose(1, 2)
    k = torch.randn(batch, steps, 8, 128, device="npu", dtype=dtype).transpose(1, 2)
    v = torch.randn_like(k)
    cos = torch.randn(batch, steps, 128, device="npu", dtype=dtype)
    sin = torch.randn_like(cos)
    positions = torch.arange(steps, device="npu") + 3
    kc = torch.randn(batch, 8, 16, 128, device="npu", dtype=dtype)
    vc = torch.randn_like(kc)
    # Warm before capture, then change inputs to verify that replay reads them.
    rope_cache(q, k, v, cos, sin, positions, kc, vc)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = rope_cache(q, k, v, cos, sin, positions, kc, vc)
    for _ in range(2):
        q.normal_()
        k.normal_()
        v.normal_()
        expected = q * cos[:, None] + _rotate_half(q) * sin[:, None]
        ek, ev = kc.clone(), vc.clone()
        ek.index_copy_(2, positions, k * cos[:, None] + _rotate_half(k) * sin[:, None])
        ev.index_copy_(2, positions, v)
        graph.replay()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(kc, ek, atol=0, rtol=0)
        torch.testing.assert_close(vc, ev, atol=0, rtol=0)
    graph.reset()
