# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Parity and graph-replay coverage for frame-local QKV lookup decoding."""

import copy

import pytest
import torch
from torch import nn
from transformers import GPT2Config

from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_local_depth import MossTTSLocalDepthTransformer

pytestmark = [pytest.mark.core_model, pytest.mark.tts]


def _models(n_vq, device="cpu", dtype=torch.float32):
    torch.manual_seed(7)
    config = GPT2Config(n_embd=32, n_head=4, n_inner=64, layer_norm_epsilon=1e-6)
    model = MossTTSLocalDepthTransformer(config).to(device=device, dtype=dtype).eval()
    embeddings = nn.ModuleList([nn.Embedding(19, 32) for _ in range(n_vq)]).to(device=device, dtype=dtype)
    heads = nn.ModuleList([nn.Linear(32, 19) for _ in range(n_vq)]).to(device=device, dtype=dtype)
    stop_head = nn.Linear(32, 2).to(device=device, dtype=dtype)
    return model, embeddings, heads, stop_head


@pytest.mark.cpu
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("n_vq", [1, 12])
@torch.inference_mode()
def test_lookup_matches_every_prefix(batch_size, n_vq):
    model, embeddings, heads, _ = _models(n_vq)
    original_keys = set(model.state_dict())
    model.prepare_qkv_lookup(embeddings, n_vq)
    assert set(model.state_dict()) == original_keys
    table_ptr = model._qkv_lookup.data_ptr()
    attn = model.h[0].attn
    key = torch.full((batch_size, attn.n_head, n_vq, attn.head_dim), float("nan"))
    value = torch.full_like(key, float("nan"))
    # Deliberately reuse dirty storage across frames. Future slots contain
    # old-frame values and must be excluded; position zero must be overwritten.
    for _ in range(3):
        prefix = torch.randn(batch_size, n_vq, model.hidden_size)
        qkv = attn.c_attn(model.h[0].ln_1(prefix[:, 0]))
        for position in range(n_vq):
            if position:
                token = torch.randint(19, (batch_size,))
                prefix[:, position] = embeddings[position - 1](token)
                qkv = model._qkv_lookup[position - 1][token]
            actual = model._run_lookup_step(prefix[:, position], qkv, key, value, position)
            expected = model._forward_prefix(prefix[:, : position + 1])[:, -1]
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(heads[position](actual), heads[position](expected), atol=2e-6, rtol=2e-5)
    assert model._qkv_lookup.data_ptr() == table_ptr


@pytest.mark.cpu
@torch.inference_mode()
def test_generate_lookup_matches_reference_and_only_projects_backbone():
    model, embeddings, heads, stop_head = _models(12)
    reference = copy.deepcopy(model)
    model.prepare_qkv_lookup(embeddings, 12)
    calls = []
    hook = model.h[0].attn.c_attn.register_forward_hook(lambda *args: calls.append(1))
    try:
        for batch_size in (4, 1, 3):
            hidden = torch.randn(batch_size, 32)
            kwargs = dict(n_vq=12, do_sample=False)
            actual = model.generate_frame(hidden, heads, embeddings, stop_head, **kwargs)
            expected = reference.generate_frame(hidden, heads, embeddings, stop_head, **kwargs)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b)
    finally:
        hook.remove()
    assert len(calls) == 3


@pytest.mark.cpu
def test_lookup_rejects_multiple_blocks():
    model, embeddings, _, _ = _models(12)
    model.h.append(copy.deepcopy(model.h[0]))
    with pytest.raises(ValueError, match="exactly one"):
        model.prepare_qkv_lookup(embeddings, 12)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_lookup_compiled_cuda_graph_replay():
    model, embeddings, heads, stop_head = _models(12, "cuda", torch.bfloat16)
    reference = copy.deepcopy(model)
    model.prepare_qkv_lookup(embeddings, 12)
    model.setup_compile()
    graphs = []
    # Capture both sizes, then replay the first after capturing the second.
    # Graphs must not share mutable frame-local KV storage.
    for batch_size in (1, 4):
        hidden = torch.randn(batch_size, 32, device="cuda", dtype=torch.bfloat16)

        def run():
            return model.generate_frame(hidden, heads, embeddings, stop_head, n_vq=12, do_sample=False)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run()
        graphs.append((graph, hidden, output))
    for graph, hidden, output in graphs * 2:
        hidden.normal_()
        expected = model.generate_frame(hidden, heads, embeddings, stop_head, n_vq=12, do_sample=False)
        graph.replay()
        for a, b in zip(output, expected):
            torch.testing.assert_close(a, b)
        # Fixed-input logits are the meaningful BF16 parity check; greedy
        # tokens can differ at ties and sampled tokens need not be bit-exact.
        first = reference._forward_prefix(hidden[:, None])[:, 0]
        attn = model.h[0].attn
        key = hidden.new_zeros((hidden.shape[0], attn.n_head, 12, attn.head_dim))
        qkv = attn.c_attn(model.h[0].ln_1(hidden))
        actual = model._run_lookup_step(hidden, qkv, key, torch.zeros_like(key), 0)
        torch.testing.assert_close(actual, first, atol=0.03, rtol=0.03)
