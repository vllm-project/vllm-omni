# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused MOSS Local depth preserves sampling, RNG and frame outputs."""

from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.tts]

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _equal(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@cuda_only
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("batch", [1, 37, 128])
@pytest.mark.parametrize("top_k,top_p", [(25, 0.8), (32, 1.0), (1, 0.8)])
@torch.inference_mode()
def test_sample_attend_matches_sample_then_attention(batch, top_k, top_p):
    """Real MOSS Local shapes: 32 heads x 80, 12 codebooks, 1024-entry vocab."""
    from vllm_omni.model_executor.models.moss_tts.local_kernels import fused_sample, lookup_attention, sample_attend

    torch.manual_seed(batch + top_k)
    heads, dim, depth, vocab = 32, 80, 12, 1024
    hidden = heads * dim
    dtype = torch.bfloat16
    tables = torch.randn(depth - 1, vocab, 3 * hidden, device="cuda", dtype=dtype)
    embeddings = torch.randn(depth - 1, vocab, hidden, device="cuda", dtype=dtype)
    # Dirty storage: stale values must never leak into results.
    key_ref = torch.full((batch, heads, depth, dim), float("nan"), device="cuda", dtype=dtype)
    value_ref = torch.full_like(key_ref, float("nan"))
    key, value = key_ref.clone(), value_ref.clone()
    for _ in range(2):
        qkv0 = torch.randn(batch, 3 * hidden, device="cuda", dtype=dtype)
        expected0, _ = lookup_attention(qkv0, key_ref, value_ref, 0)
        actual0, _ = lookup_attention(qkv0, key, value, 0, late_store=True)
        _equal(actual0, expected0)
        uniforms = torch.rand(batch, depth, device="cuda")
        codes = torch.full((batch, depth), -1, device="cuda", dtype=torch.long)
        for channel in range(depth):
            logits = torch.randn(batch, vocab, device="cuda", dtype=dtype) * 3
            last = channel + 1 == depth
            token = fused_sample(logits, uniforms[:, channel], top_k=top_k, temperature=1.7, top_p=top_p)
            fused = sample_attend(
                logits,
                uniforms[:, channel],
                codes[:, channel],
                top_k=top_k,
                temperature=1.7,
                top_p=top_p,
                qkv=None if last else tables[channel],
                embedding=None if last else embeddings[channel],
                key=key,
                value=value,
                position=channel + 1,
            )
            _equal(codes[:, channel], token.long())
            if last:
                assert fused is None
                continue
            expected = lookup_attention(
                tables[channel], key_ref, value_ref, channel + 1, tokens=token, embedding=embeddings[channel]
            )
            for actual_tensor, expected_tensor in zip(fused, expected, strict=True):
                _equal(actual_tensor, expected_tensor)
        # The final position's K/V are never read, so the fused path skips them.
        _equal(key[:, :, : depth - 1], key_ref[:, :, : depth - 1])
        _equal(value[:, :, : depth - 1], value_ref[:, :, : depth - 1])


@cuda_only
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("temperature", [1.0, 1.7, 0.6])
@pytest.mark.parametrize("top_k,top_p", [(50, 1.0), (0, 1.0), (2, 1.5)])
@torch.inference_mode()
def test_binary_continue_matches_reference_sampler(temperature, top_k, top_p):
    from vllm_omni.model_executor.models.moss_tts.local_kernels import binary_continue
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_local import _sample_token

    torch.manual_seed(3)
    batch = 8192
    logits = (torch.randn(batch, 2, device="cuda") * torch.logspace(-3, 1.5, batch, device="cuda")[:, None]).to(
        torch.bfloat16
    )
    logits[:64, 1] = logits[:64, 0]  # exact ties
    reference_gen = torch.Generator(device="cuda").manual_seed(11)
    fused_gen = torch.Generator(device="cuda").manual_seed(11)
    for _ in range(3):
        expected = _sample_token(logits.float(), temperature, top_k, top_p, True, generator=reference_gen).eq(0)
        exponential = torch.empty((batch, 2), device="cuda").exponential_(generator=fused_gen)
        actual = binary_continue(logits, exponential, temperature)
        _equal(actual, expected)
        # Same Philox consumption as torch.multinomial.
        _equal(fused_gen.get_state(), reference_gen.get_state())
    assert 0.05 < actual.float().mean().item() < 0.95


def _talker_stub(fused_depth, n_vq, vocab, hidden, codes, should_continue, dtype):
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import MossTTSLocalTalkerForGeneration

    embeddings = torch.randn(n_vq, vocab, hidden, device="cuda", dtype=dtype)
    stub = SimpleNamespace(
        n_vq=n_vq,
        audio_vocab_size=vocab,
        audio_pad_token_id=vocab,
        audio_lm_heads=None,
        audio_embeddings=None,
        local_text_lm_head=None,
        _stacked_audio_emb_w=embeddings,
        local_transformer=SimpleNamespace(
            fused_depth=fused_depth,
            generate_frame=lambda *args, **kwargs: (should_continue.clone(), codes.clone()),
        ),
    )
    stub._audio_embed = lambda c: MossTTSLocalTalkerForGeneration._audio_embed(stub, c)
    return MossTTSLocalTalkerForGeneration.talker_mtp, stub


@cuda_only
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("batch", [1, 128])
@torch.inference_mode()
def test_mtp_epilogue_matches_talker_reference(batch):
    n_vq, vocab, hidden, dtype = 12, 1024, 2560, torch.bfloat16
    torch.manual_seed(batch)
    codes = torch.randint(vocab, (batch, n_vq), device="cuda")
    codes[torch.rand(batch, n_vq, device="cuda") < 0.2] = vocab  # pad = no contribution
    should_continue = torch.rand(batch, device="cuda") < 0.8
    text_step = (torch.rand(batch, 1, device="cuda") < 0.9).long()
    input_embeds = torch.randn(batch, 2 * hidden, device="cuda", dtype=dtype)[:, :hidden]  # strided rows
    outputs = []
    for fused in (False, True):
        torch.manual_seed(0)  # identical embedding tables
        talker_mtp, stub = _talker_stub(fused, n_vq, vocab, hidden, codes, should_continue, dtype)
        hidden_in = torch.randn(batch, hidden, device="cuda", dtype=dtype)
        outputs.append(talker_mtp(stub, None, input_embeds, hidden_in, text_step))
    for actual, expected in zip(outputs[1], outputs[0], strict=True):
        _equal(actual, expected)


def _cuda_models():
    from tests.model_executor.models.moss_tts.test_local_depth_qkv_lookup import _models

    model, embeddings, heads, stop_head = _models(12, "cuda", torch.bfloat16)
    model.prepare_qkv_lookup(embeddings, 12)
    model.setup_compile()
    return model, embeddings, heads, stop_head


@cuda_only
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("per_row", [False, True])
@torch.inference_mode()
def test_generate_frame_fused_depth_is_bit_identical(per_row):
    model, embeddings, heads, stop_head = _cuda_models()
    assert model._fused_attention and model._fused_sampling
    hidden = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16) * 4
    results = []
    for fused in (False, True):
        model.fused_depth = fused
        generator = torch.Generator(device="cuda").manual_seed(5)
        generators = [torch.Generator(device="cuda").manual_seed(row) for row in range(16)] if per_row else None
        results.append(
            [
                model.generate_frame(
                    hidden,
                    heads,
                    embeddings,
                    stop_head,
                    n_vq=12,
                    top_k=16,
                    top_p=0.8,
                    temperature=1.7,
                    generator=generator,
                    generators=generators,
                )
                for _ in range(4)
            ]
        )
    for actual_frame, expected_frame in zip(results[1], results[0], strict=True):
        for actual, expected in zip(actual_frame, expected_frame, strict=True):
            _equal(actual, expected)


@cuda_only
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.inference_mode()
def test_generate_frame_fused_depth_graph_replay_matches_eager():
    model, embeddings, heads, stop_head = _cuda_models()
    model.fused_depth = True
    hidden = torch.randn(8, 32, device="cuda", dtype=torch.bfloat16) * 4

    def run():
        return model.generate_frame(hidden, heads, embeddings, stop_head, n_vq=12, top_k=16, top_p=0.8, temperature=1.7)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for _ in range(3):
        hidden.normal_()
        state = torch.cuda.get_rng_state()
        graph.replay()
        replayed = [tensor.clone() for tensor in captured]
        torch.cuda.set_rng_state(state)
        model.fused_depth = False
        eager = run()
        model.fused_depth = True
        for actual, expected in zip(replayed, eager, strict=True):
            _equal(actual, expected)
