# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.helpers.mark import hardware_test
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="requires CUDA")]


class MelOutput(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.m_source = SimpleNamespace(l_linear=SimpleNamespace(weight=self.anchor))

    def inference(self, speech_feat, finalize):
        assert finalize
        return speech_feat, None, None


def _items(shapes, **extra):
    return [
        dict(
            token=torch.randint(0, 64, (1, length), device="cuda"),
            prompt_token=torch.randint(0, 64, (1, prompt)),
            prompt_feat=torch.randn(1, prompt * 2, 80, device="cuda"),
            embedding=torch.randn(1, 192, device="cuda"),
            **extra,
        )
        for prompt, length in shapes
    ]


def tiny_flow(monkeypatch):
    import vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav as module
    from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config

    monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")
    monkeypatch.setattr(module, "CausalHiFTGenerator", MelOutput)
    config = CosyVoice3Config()
    config.flow["pre_lookahead_layer"]["channels"] = 32
    config.flow["decoder"]["estimator"].update(dim=32, depth=2, heads=4, dim_head=8)
    return module.CosyVoice3Code2Wav(config).cuda().eval()


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("offset", [0, 3])
@pytest.mark.parametrize("count", [3, 11])
@torch.inference_mode()
def test_full_batch_ragged_prompt_matches_independent_flow(monkeypatch, offset, count):
    model = tiny_flow(monkeypatch)
    items = _items(([(7, 17), (13, 9), (3, 31)] * 4)[:count], token_offset_tokens=offset)
    # Fix Flow noise so scheduling changes do not change the comparison input.
    monkeypatch.setattr(torch, "randn", lambda shape, **kw: torch.zeros(shape, **kw))
    expected = [model.forward(**item, n_timesteps=3) for item in items]
    actual = model.forward_batch(items, n_timesteps=3)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=3e-4, atol=3e-5)
    # Reordering unrelated requests must not move their conditioning or crop points.
    reverse = model.forward_batch(items[::-1], n_timesteps=3)
    for result, reference in zip(reverse, expected[::-1]):
        torch.testing.assert_close(result, reference, rtol=3e-4, atol=3e-5)
    assert model.forward_batch([]) == []


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@torch.inference_mode()
def test_packed_full_response_preserves_request_isolation(monkeypatch):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("the opt-in packed backend requires Hopper FA3")
    monkeypatch.setenv("COSYVOICE3_FULL_RESPONSE_OPTIMIZATIONS", "1")
    model = tiny_flow(monkeypatch).bfloat16()
    model.hift.upsample_rates = [1]
    model.hift.istft_params = {"hop_len": 1}
    items = _items([(7, 17), (13, 9), (3, 31)])
    monkeypatch.setattr(torch, "randn", lambda shape, **kw: torch.zeros(shape, **kw))
    expected = [model.forward(**item, n_timesteps=3) for item in items]
    actual = model.forward_batch(items, n_timesteps=3)
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference, rtol=0.04, atol=0.04)
    # A different batch order must not leak the other requests' conditioning.
    actual = model.forward_batch(items[::-1], n_timesteps=3)
    for output, reference in zip(actual, expected[::-1]):
        torch.testing.assert_close(output, reference, rtol=0.04, atol=0.04)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_packed_stream_noise_growth_preserves_prefix_and_global_rng():
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import CosyVoice3Code2Wav

    model = CosyVoice3Code2Wav.__new__(CosyVoice3Code2Wav)
    nn.Module.__init__(model)
    like = torch.empty(1, 80, 1, dtype=torch.bfloat16)
    rng = torch.random.get_rng_state().clone()
    first = model._stream_position_noise(97, like).clone()
    grown = model._stream_position_noise(30001, like).clone()
    torch.testing.assert_close(grown[..., :97], first, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    direct = CosyVoice3Code2Wav.__new__(CosyVoice3Code2Wav)
    nn.Module.__init__(direct)
    torch.testing.assert_close(direct._stream_position_noise(30001, like), grown, rtol=0, atol=0)
    assert model._stream_position_noise(0, like).shape == (1, 80, 0)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@torch.inference_mode()
def test_packed_stream_attention_matches_chunk_causal_reference():
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("the opt-in packed backend requires Hopper FA3")
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.packed_dit import (
        RaggedRowAttention,
        pack_rows,
        packed_fa3,
    )

    lengths = [67, 103]
    rows = pack_rows(lengths, torch.device("cuda"))
    attention = RaggedRowAttention(rows, heads=2, head_dim=64, chunk_size=50)
    generator = torch.Generator(device="cuda").manual_seed(7)
    q, k, v = [
        torch.randn(sum(lengths), 2, 64, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(3)
    ]
    actual = packed_fa3(
        q,
        k[:, None],
        v[:, None],
        attention.cache_seqlens,
        attention.page_table,
        attention.cu_seqlens_q,
        attention.max_seqlen_q,
    )
    start = 0
    for length in lengths:
        position = torch.arange(length, device="cuda")
        mask = position[None, :] < ((position[:, None] // 50 + 1) * 50)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q[start : start + length].transpose(0, 1).float(),
            k[start : start + length].transpose(0, 1).float(),
            v[start : start + length].transpose(0, 1).float(),
            attn_mask=mask,
        ).transpose(0, 1)
        torch.testing.assert_close(actual[start : start + length].float(), expected, rtol=0.03, atol=0.01)
        start += length


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@torch.inference_mode()
def test_packed_stream_mixed_finalization_and_ragged_requests_stay_aligned(monkeypatch):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("the opt-in packed backend requires Hopper FA3")
    monkeypatch.setenv("COSYVOICE3_FULL_RESPONSE_OPTIMIZATIONS", "0")
    monkeypatch.setenv("COSYVOICE3_PACKED_STREAMING", "1")
    model = tiny_flow(monkeypatch).bfloat16()
    monkeypatch.setattr(
        model, "_stream_hift_from_feat", lambda mel, cache_state, finalize: (mel, None if finalize else cache_state)
    )
    items = []
    for index, (prompt, generated) in enumerate(([(7, 17), (13, 9), (3, 31)] * 6)[:17]):
        items.append(
            dict(
                token=torch.randint(0, 64, (1, generated), device="cuda"),
                prompt_token=torch.randint(0, 64, (1, prompt)),
                prompt_feat=torch.randn(1, prompt * 2, 80, device="cuda"),
                embedding=torch.randn(1, 192, device="cuda"),
                token_offset_tokens=3,
                finalize=index % 3 == 0,
                cache_state={"request_marker": torch.tensor(index)},
            )
        )
    expected = [model.forward_streaming_batch([item], n_timesteps=3)[0] for item in items]
    for ordered, reference in [(items, expected), (items[::-1], expected[::-1])]:
        actual = model.forward_streaming_batch(ordered, n_timesteps=3)
        for (audio, state), (ref_audio, ref_state), item in zip(actual, reference, ordered):
            torch.testing.assert_close(audio, ref_audio, rtol=0.04, atol=0.04)
            assert state is ref_state
            assert state is (None if item["finalize"] else item["cache_state"])
    assert model.forward_streaming_batch([]) == []
    with pytest.raises(ValueError, match="mixed finalization"):
        model.forward_batch(items, n_timesteps=3, stream_items=True)
    monkeypatch.setenv("COSYVOICE3_PACKED_STREAMING", "0")
    with pytest.raises(ValueError, match="COSYVOICE3_PACKED_STREAMING"):
        model.forward_batch(items[:1], n_timesteps=3, stream_items=True)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@torch.inference_mode()
def test_incremental_packed_streaming_matches_full_prefix_and_keeps_tail_full(monkeypatch):
    import vllm_omni.model_executor.models.cosyvoice3.code2wav_core.incremental as incremental
    from vllm_omni.model_executor.models.cosyvoice3.runtime import cosyvoice3_packed_streaming_enabled

    monkeypatch.setenv("COSYVOICE3_PACKED_STREAMING", "1")
    monkeypatch.setenv("COSYVOICE3_FLOW_GRAPH", "0")
    monkeypatch.setenv("COSYVOICE3_FLOW_CUDAGRAPH", "0")
    if not cosyvoice3_packed_streaming_enabled():
        pytest.skip("packed FlashAttention is unavailable on this GPU/build")
    torch.manual_seed(42)
    model = tiny_flow(monkeypatch).bfloat16()
    # Isolate Flow parity; HiFT consumes the resulting new-mel slice unchanged.
    monkeypatch.setattr(
        model,
        "_stream_hift_from_feat",
        lambda feat, cache_state=None, finalize=False: (feat, None if finalize else {}),
    )
    base = _items([(7, 80)])[0]
    tokens = base.pop("token")
    cache = None
    offset = 0
    # 7 prompt tokens: valid combined lengths are 25, 50 and 65 tokens.
    # The last call leaves a partial attention block, which must not be cached.
    for length, complete in ((21, 50), (46, 100), (61, 100)):
        item = dict(base, token=tokens[:, :length], token_offset_tokens=offset, finalize=False)
        monkeypatch.setenv("COSYVOICE3_INCREMENTAL_FLOW", "0")
        expected = model.forward_batch([item], n_timesteps=3, stream_items=True)[0][0]
        monkeypatch.setenv("COSYVOICE3_INCREMENTAL_FLOW", "1")
        actual, cache = model.forward_batch([dict(item, cache_state=cache)], n_timesteps=3, stream_items=True)[0]
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
        assert cache["flow_cache"].frames == complete
        assert cache["lookahead_cache"].tokens == 7 + length - 3
        offset = length - 3

    # Final changes to full-context attention. It must ignore all cached KV.
    final = dict(base, token=tokens[:, :65], token_offset_tokens=offset, finalize=True)
    monkeypatch.setenv("COSYVOICE3_INCREMENTAL_FLOW", "0")
    expected = model.forward_batch([final], n_timesteps=3, stream_items=True)[0][0]

    def unexpected_incremental(*args, **kwargs):
        pytest.fail("final tail must use the existing full-context solve")

    monkeypatch.setattr(incremental, "solve_flow_euler_incremental", unexpected_incremental)
    monkeypatch.setattr(model.flow_model.pre_lookahead_layer, "forward_incremental", unexpected_incremental)
    monkeypatch.setattr(model.flow_model.pre_lookahead_layer, "forward_incremental_packed", unexpected_incremental)
    monkeypatch.setenv("COSYVOICE3_INCREMENTAL_FLOW", "1")
    actual, final_cache = model.forward_batch([dict(final, cache_state=cache)], n_timesteps=3, stream_items=True)[0]
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    assert final_cache is None


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@torch.inference_mode()
def test_incremental_packs_ragged_requests_into_one_attention_call_per_layer_step(monkeypatch):
    import vllm_omni.model_executor.models.cosyvoice3.code2wav_core.incremental as incremental
    from vllm_omni.model_executor.models.cosyvoice3.runtime import cosyvoice3_packed_streaming_enabled

    monkeypatch.setenv("COSYVOICE3_PACKED_STREAMING", "1")
    monkeypatch.setenv("COSYVOICE3_FLOW_GRAPH", "0")
    monkeypatch.setenv("COSYVOICE3_FLOW_CUDAGRAPH", "0")
    if not cosyvoice3_packed_streaming_enabled():
        pytest.skip("packed FlashAttention is unavailable on this GPU/build")
    torch.manual_seed(19)
    model = tiny_flow(monkeypatch).bfloat16()
    monkeypatch.setattr(model, "_stream_hift_from_feat", lambda feat, cache_state=None, finalize=False: (feat, {}))
    bases = _items([(7, 80), (3, 80)])
    tokens = [item.pop("token") for item in bases]
    caches = [None, None]
    offsets = [0, 0]
    original_attention = incremental.ragged_attention_tensor_geometry
    query_rows = []

    def attend(query, *args):
        query_rows.append(query.shape[1])
        return original_attention(query, *args)

    monkeypatch.setattr(incremental, "ragged_attention_tensor_geometry", attend)
    for lengths in ((21, 35), (46, 60)):
        items = [
            dict(base, token=token[:, :length], token_offset_tokens=offset, finalize=False)
            for base, token, length, offset in zip(bases, tokens, lengths, offsets)
        ]
        monkeypatch.setenv("COSYVOICE3_INCREMENTAL_FLOW", "0")
        expected = model.forward_batch(items, n_timesteps=3, stream_items=True)
        old_frames = [0 if cache is None else cache["flow_cache"].frames for cache in caches]
        query_rows.clear()
        monkeypatch.setenv("COSYVOICE3_INCREMENTAL_FLOW", "1")
        actual = model.forward_batch(
            [dict(item, cache_state=cache) for item, cache in zip(items, caches)], n_timesteps=3, stream_items=True
        )
        for (mel, _), (reference, _) in zip(actual, expected):
            torch.testing.assert_close(mel, reference, rtol=3e-2, atol=3e-2)
        # Tiny DiT has 2 layers and 3 Euler steps, independent of request count.
        assert len(query_rows) == 6
        suffix_lengths = [(prompt + length - 3) * 2 - old for prompt, length, old in zip((7, 3), lengths, old_frames)]
        assert query_rows == [2 * sum(suffix_lengths)] * 6
        caches = [cache for _, cache in actual]
        offsets = [length - 3 for length in lengths]
