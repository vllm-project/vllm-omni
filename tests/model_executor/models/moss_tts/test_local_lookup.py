# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Local lookup contracts: frame state, distribution, seed and graph replay."""

import pytest
import torch
from transformers import GPT2Config

from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_local_depth import MossTTSLocalDepthTransformer

pytestmark = [pytest.mark.core_model, pytest.mark.tts]


def _model(device, dtype=torch.float32, n_head=1, n_vq=3):
    torch.manual_seed(17)
    hidden = 80 * n_head
    cfg = GPT2Config(n_embd=hidden, n_head=n_head, n_inner=hidden * 2, layer_norm_epsilon=1e-6)
    model = MossTTSLocalDepthTransformer(cfg).to(device=device, dtype=dtype).eval()
    heads = torch.nn.ModuleList([torch.nn.Linear(hidden, 32) for _ in range(n_vq)]).to(device=device, dtype=dtype)
    embeddings = torch.nn.ModuleList([torch.nn.Embedding(32, hidden) for _ in range(n_vq)]).to(
        device=device, dtype=dtype
    )
    binary = torch.nn.Linear(hidden, 2).to(device=device, dtype=dtype)
    return model, heads, embeddings, binary


@pytest.mark.cpu
def test_lookup_unavailable_on_cpu_preserves_original_seeded_frame(monkeypatch):
    model, heads, embeddings, binary = _model("cpu")
    hidden = torch.randn(2, 80)
    generator = torch.Generator().manual_seed(31)
    expected = model.generate_frame(hidden, heads, embeddings, binary, n_vq=3, generator=generator)
    monkeypatch.setenv("VLLM_OMNI_MOSS_LOCAL_QKV_LOOKUP", "1")
    model.prepare_qkv_lookup(embeddings, 3)
    assert model._lookup_n_vq == 0
    generator.manual_seed(31)
    actual = model.generate_frame(hidden, heads, embeddings, binary, n_vq=3, generator=generator)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("batch,n_head,n_vq", [(4, 1, 3), (16, 4, 12)])
def test_lookup_prefix_matches_live_projection_with_fixed_codes_and_frame_reset(
    monkeypatch, dtype, batch, n_head, n_vq
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vllm_omni.model_executor.models.moss_tts.local_kernels import lookup_attention

    model, _, embeddings, _ = _model("cuda", dtype, n_head, n_vq)
    monkeypatch.setenv("VLLM_OMNI_MOSS_LOCAL_QKV_LOOKUP", "1")
    model.prepare_qkv_lookup(embeddings, n_vq)
    attn = model.h[0].attn
    key, value = (torch.empty(batch, n_head, n_vq, 80, device="cuda", dtype=dtype) for _ in range(2))
    codes = torch.randint(32, (batch, n_vq - 1), device="cuda")
    with torch.inference_mode():
        for _ in range(2):
            first = torch.randn(batch, n_head * 80, device="cuda", dtype=dtype)
            qkv = attn.c_attn(model.h[0].ln_1(first))
            actual = model._run_lookup_step(first, qkv, key, value, 0)
            prefix = first[:, None]
            expected = model._forward_prefix(prefix)[:, -1]
            torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
            for channel in range(n_vq - 1):
                embed = embeddings[channel](codes[:, channel])
                audio, residual = lookup_attention(
                    model._qkv_lookup[channel],
                    key,
                    value,
                    channel + 1,
                    tokens=codes[:, channel],
                    embedding=embeddings[channel].weight,
                )
                actual = model._forward_lookup(residual, audio)
                prefix = torch.cat([prefix, embed[:, None]], dim=1)
                expected = model._forward_prefix(prefix)[:, -1]
                torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype,width,k,p", [(torch.float32, 2, 2, 1.0), (torch.bfloat16, 1024, 25, 0.8), (torch.float16, 33, 7, 0.9)]
)
def test_fused_sampling_matches_independent_inverse_cdf_and_strided_output(dtype, width, k, p):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vllm_omni.model_executor.models.moss_tts.local_kernels import fused_sample

    # Stable descending sort specifies the documented lower-ID tie break.
    torch.manual_seed(53)
    logits = torch.randn(128, width, device="cuda").to(dtype)
    uniforms = torch.rand(128, 3, device="cuda")[:, 1]
    ordered, ids = torch.sort(logits.float(), descending=True, stable=True)
    ordered, ids = ordered[:, :k] / 1.7, ids[:, :k]
    probs = ordered.softmax(-1)
    keep = probs.cumsum(-1) - probs <= p
    keep[:, 0] = True
    probs = probs * keep
    cdf = probs.cumsum(-1)
    target = uniforms[:, None] * probs.sum(-1, keepdim=True)
    index = (cdf <= target).sum(-1).clamp_max(k - 1)
    expected = ids.gather(-1, index[:, None])[:, 0]
    buffer = torch.full((128, 3), -1, device="cuda", dtype=torch.long)
    actual = fused_sample(logits, uniforms, top_k=k, top_p=p, out=buffer[:, 1])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert (buffer[:, 0] == -1).all() and (buffer[:, 2] == -1).all()


@pytest.mark.cuda
def test_seeded_generators_and_unsupported_sampler_settings_bypass_fusion(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model, heads, embeddings, binary = _model("cuda", torch.bfloat16)
    monkeypatch.setenv("VLLM_OMNI_MOSS_LOCAL_QKV_LOOKUP", "1")
    monkeypatch.setenv("VLLM_OMNI_MOSS_LOCAL_FUSED_SAMPLING", "1")
    model.prepare_qkv_lookup(embeddings, 3)
    hidden = torch.randn(2, 80, device="cuda", dtype=torch.bfloat16)
    generators = [torch.Generator(device="cuda").manual_seed(seed) for seed in [11, 71]]
    expected = model.generate_frame(hidden, heads, embeddings, binary, n_vq=3, generators=generators)

    def forbidden(*args, **kwargs):
        raise AssertionError("fused sampler selected")

    monkeypatch.setattr("vllm_omni.model_executor.models.moss_tts.local_kernels.fused_sample", forbidden)
    for generator, seed in zip(generators, [11, 71]):
        generator.manual_seed(seed)
    actual = model.generate_frame(hidden, heads, embeddings, binary, n_vq=3, generators=generators)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    model.generate_frame(hidden, heads, embeddings, binary, n_vq=3, do_sample=False)
    model.generate_frame(hidden, heads, embeddings, binary, n_vq=3, top_k=33)
    with pytest.raises(AssertionError, match="fused sampler selected"):
        model.generate_frame(hidden, heads, embeddings, binary, n_vq=3, top_k=25)


@pytest.mark.cuda
def test_lookup_graph_replay_reads_changed_token_column_and_preserves_other_slots():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vllm_omni.model_executor.models.moss_tts.local_kernels import lookup_attention

    table = torch.randn(32, 240, device="cuda", dtype=torch.bfloat16)
    embedding = torch.randn(32, 80, device="cuda", dtype=torch.bfloat16)
    codes = torch.zeros(4, 3, device="cuda", dtype=torch.long)
    key = torch.zeros(4, 1, 3, 80, device="cuda", dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        lookup_attention(table, key, value, 0, tokens=codes[:, 1], embedding=embedding)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        audio, residual = lookup_attention(table, key, value, 0, tokens=codes[:, 1], embedding=embedding)
    for tokens in [[1, 2, 3, 4], [31, 7, 0, 15]]:
        codes[:, 1] = torch.tensor(tokens, device="cuda")
        graph.replay()
        torch.testing.assert_close(audio, table[codes[:, 1], 160:], rtol=0, atol=0)
        torch.testing.assert_close(residual, embedding[codes[:, 1]], rtol=0, atol=0)
        assert (key[:, :, 1:] == 0).all() and (value[:, :, 1:] == 0).all()


@pytest.mark.cuda
def test_fused_sampler_graph_replay_advances_pytorch_rng():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vllm_omni.model_executor.models.moss_tts.local_kernels import fused_sample

    # Equal logits expose stale random inputs: a frozen graph would produce
    # exactly the same 128 choices on every replay.
    logits = torch.zeros(128, 32, device="cuda", dtype=torch.bfloat16)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fused_sample(logits, torch.rand(128, device="cuda"), top_k=25, top_p=1.0)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tokens = fused_sample(logits, torch.rand(128, device="cuda"), top_k=25, top_p=1.0)
    graph.replay()
    first = tokens.clone()
    graph.replay()
    assert not torch.equal(tokens, first)
    assert ((tokens >= 0) & (tokens < 25)).all()
