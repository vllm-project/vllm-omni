# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from transformers.cache_utils import DynamicCache
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.utils.network_utils import get_file_store_init_method

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.sensenova_u1.paged_decode import PagedDecodeCache
from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import (
    FlashKVCache,
    SenseNovaU1ForCausalLM,
    clear_flash_kv_cache,
    prepare_flash_kv_cache,
)
from vllm_omni.transformers_utils.configs.sensenova_u1 import SenseNovaU1Config

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.cpu
@pytest.mark.parametrize("batch_size", [1, 2])
def test_flash_cache_update_and_reserve(batch_size):
    actual, expected = FlashKVCache(), DynamicCache()
    for seq_len in (3, 1, 5):
        k, v = torch.randn(2, 1, 2, seq_len, 8)
        for cache in (actual, expected):
            cache.update(k, v, 0)
        assert actual.get_seq_length() == expected.get_seq_length()
        for name in ("keys", "values"):
            value = getattr(actual.layers[0], name)
            torch.testing.assert_close(value, getattr(expected.layers[0], name), rtol=0, atol=0)
            assert value.transpose(1, 2).is_contiguous()
    prepare_flash_kv_cache(actual, 4, batch_size)
    layer = actual.layers[0]
    for name, buffer in (("keys", layer.flash_k_cache), ("values", layer.flash_v_cache)):
        value = getattr(layer, name)
        assert value.untyped_storage().data_ptr() == buffer.untyped_storage().data_ptr()
        torch.testing.assert_close(value, getattr(expected.layers[0], name).expand(batch_size, -1, -1, -1))
    clear_flash_kv_cache(actual)
    assert not hasattr(layer, "flash_k_cache")
    assert actual.get_seq_length() == 9


@pytest.mark.cpu
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("suffix_len", [4, 1024])
def test_clear_flash_cache_retires_suffix_storage(batch_size, suffix_len):
    cache = FlashKVCache()
    expected = []
    for layer_idx in range(2):
        k, v = torch.randn(2, 1, 2, 9, 8)
        cache.update(k, v, layer_idx)
        expected.append((k.expand(batch_size, -1, -1, -1), v.expand(batch_size, -1, -1, -1)))

    for _ in range(2):
        prepare_flash_kv_cache(cache, suffix_len, batch_size)
        clear_flash_kv_cache(cache)
        # Callers keep the prefix cache alive during output conversion. Its
        # logical shape alone does not show whether it still owns the suffix.
        for layer, prefix in zip(cache.layers, expected):
            for name, wanted in zip(("keys", "values"), prefix):
                value = getattr(layer, name)
                torch.testing.assert_close(value, wanted, rtol=0, atol=0)
                assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
                assert value.transpose(1, 2).is_contiguous()
            for attr in ("flash_prefix_len", "flash_total_len", "flash_k_cache", "flash_v_cache"):
                assert not hasattr(layer, attr)

    pointers = [(layer.keys.data_ptr(), layer.values.data_ptr()) for layer in cache.layers]
    clear_flash_kv_cache(cache)
    assert pointers == [(layer.keys.data_ptr(), layer.values.data_ptr()) for layer in cache.layers]
    for layer_idx, (k, v) in enumerate(expected):
        next_k, next_v = torch.randn(2, batch_size, 2, 1, 8)
        cache.update(next_k, next_v, layer_idx)
        torch.testing.assert_close(cache.layers[layer_idx].keys, torch.cat((k, next_k), dim=2), rtol=0, atol=0)
        torch.testing.assert_close(cache.layers[layer_idx].values, torch.cat((v, next_v), dim=2), rtol=0, atol=0)


@pytest.mark.cpu
def test_paged_handoff_keeps_flash_layout_and_request_ownership():
    cache = FlashKVCache()
    k, v = torch.randn(2, 1, 2, 7, 8)
    cache.update(k, v, 0)
    paged = PagedDecodeCache.from_dynamic_cache(cache, 1, torch.device("cpu"), torch.float32)
    paged.to_dynamic_cache(cache)
    for name in ("keys", "values"):
        assert getattr(cache.layers[0], name).transpose(1, 2).is_contiguous()
    paged.k[0].zero_()
    paged.v[0].zero_()
    torch.testing.assert_close(cache.layers[0].keys, k)
    torch.testing.assert_close(cache.layers[0].values, v)
    next_k, next_v = torch.randn(2, 1, 2, 2, 8)
    cache.update(next_k, next_v, 0)
    torch.testing.assert_close(cache.layers[0].keys, torch.cat((k, next_k), dim=2))
    prepare_flash_kv_cache(cache, 4, 1)
    assert cache.layers[0].keys.transpose(1, 2).stride()[-2:] == (8, 1)


@pytest.fixture
def tiny_model():
    init_distributed_environment(
        world_size=1, rank=0, local_rank=0, distributed_init_method=get_file_store_init_method()
    )
    initialize_model_parallel()
    try:
        config = SenseNovaU1Config(
            llm_config={
                "hidden_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "vocab_size": 32,
                "max_position_embeddings": 128,
                "max_position_embeddings_hw": 128,
            }
        )
        with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cuda"))):
            model = SenseNovaU1ForCausalLM(config.llm_config).to(device="cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                torch.manual_seed(42)
                for parameter in model.parameters():
                    parameter.normal_(0, 0.05)
            yield model
    finally:
        cleanup_dist_env_and_memory()


@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("fused", [False, True])
@torch.inference_mode()
def test_resident_cache_matches_dynamic_prefill_and_denoise(tiny_model, masked, fused, monkeypatch):
    if not fused:
        from vllm_omni.diffusion.models.sensenova_u1 import fused_rmsnorm_rope

        monkeypatch.setattr(fused_rmsnorm_rope, "triton_qk_norm_rope", None)
    inputs = torch.randn(1, 3, 128, device="cuda", dtype=torch.bfloat16)
    indexes = torch.zeros(3, 3, device="cuda", dtype=torch.long)
    indexes[0] = torch.arange(3, device="cuda")
    ordinary = tiny_model(inputs_embeds=inputs, indexes=indexes, past_key_values=DynamicCache(), use_cache=True)
    resident = tiny_model(inputs_embeds=inputs, indexes=indexes, use_cache=True)
    assert isinstance(resident.past_key_values, FlashKVCache)
    torch.testing.assert_close(resident.logits, ordinary.logits, rtol=0, atol=0)
    prefix = [(layer.keys.clone(), layer.values.clone()) for layer in resident.past_key_values.layers]
    prepare_flash_kv_cache(resident.past_key_values, 4, 1)
    ptrs = [layer.flash_k_cache.data_ptr() for layer in resident.past_key_values.layers]
    mask = None
    if masked:
        mask = torch.zeros(1, 1, 4, 7, device="cuda", dtype=torch.bfloat16)
        mask[..., 0, -1] = -torch.inf
    for _ in range(3):
        kwargs = dict(
            inputs_embeds=torch.randn(1, 4, 128, device="cuda", dtype=torch.bfloat16),
            indexes=torch.tensor([[3, 3, 3, 3], [0, 0, 1, 1], [0, 1, 0, 1]], device="cuda"),
            image_gen_indicators=torch.ones(1, 4, device="cuda", dtype=torch.bool),
            attention_mask={"full_attention": mask},
            update_cache=False,
            use_cache=True,
            compute_logits=False,
        )
        expected = tiny_model(past_key_values=ordinary.past_key_values, **kwargs)
        actual = tiny_model(past_key_values=resident.past_key_values, **kwargs)
        torch.testing.assert_close(actual.hidden_states, expected.hidden_states, rtol=0, atol=0)
        for layer, (k, v), ptr in zip(resident.past_key_values.layers, prefix, ptrs):
            torch.testing.assert_close(layer.keys, k, rtol=0, atol=0)
            torch.testing.assert_close(layer.values, v, rtol=0, atol=0)
            assert layer.flash_k_cache.data_ptr() == ptr

    clear_flash_kv_cache(resident.past_key_values)
    for layer, (k, v) in zip(resident.past_key_values.layers, prefix):
        torch.testing.assert_close(layer.keys, k, rtol=0, atol=0)
        torch.testing.assert_close(layer.values, v, rtol=0, atol=0)
        for value in (layer.keys, layer.values):
            assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
            assert value.transpose(1, 2).is_contiguous()
    # A caller may still read the prefix after retiring the denoise buffers.
    actual = tiny_model(past_key_values=resident.past_key_values, **kwargs)
    torch.testing.assert_close(actual.hidden_states, expected.hidden_states, rtol=0, atol=0)


@hardware_test(res={"cuda": "L4"})
@torch.inference_mode()
def test_resident_suffix_replays_with_new_inputs(tiny_model):
    attn = tiny_model.model.layers[0].self_attn
    cache = FlashKVCache()
    k, v = torch.randn(2, 1, attn.num_kv_heads, 3, attn.head_dim, device="cuda", dtype=torch.bfloat16)
    cache.update(k, v, 0)
    prepare_flash_kv_cache(cache, 4, 1)
    x = torch.randn(1, 4, 128, device="cuda", dtype=torch.bfloat16)
    indexes = torch.tensor([[3, 3, 3, 3], [0, 0, 1, 1], [0, 1, 0, 1]], device="cuda")
    rope = (
        tiny_model.model.rotary_emb(x, indexes[0].unsqueeze(0)),
        tiny_model.model.rotary_emb_hw(x, indexes[1].unsqueeze(0)),
        tiny_model.model.rotary_emb_hw(x, indexes[2].unsqueeze(0)),
    )

    def run():
        return attn.forward_gen(x, indexes, None, cache, position_embeddings=rope, update_cache=False)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    for _ in range(3):
        x.normal_()
        expected = run().clone()
        graph.replay()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(cache.layers[0].keys, k, rtol=0, atol=0)
        torch.testing.assert_close(cache.layers[0].values, v, rtol=0, atol=0)
