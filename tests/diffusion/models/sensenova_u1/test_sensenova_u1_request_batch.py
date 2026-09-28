# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request batching: independent prefixes/RNG, fused denoise and ordered outputs."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
from vllm_omni.diffusion.models.sensenova_u1.batching import image_count, merge_cfg_branches, merge_conditioning
from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import (
    SenseNovaU1Pipeline,
    get_sensenova_u1_pre_process_func,
)
from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import (
    SenseNovaU1Attention,
    prepare_flash_kv_cache,
    write_packed_image_kv,
)
from vllm_omni.diffusion.output_formatter import format_diffusion_outputs, normalize_diffusion_postprocess_output
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def single_gpu_model_parallel(monkeypatch):
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "29544")
    init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


def _request(name="first", *, count=1, seed=42, extra=None, mode="t2i"):
    prompt = {"prompt": name}
    if mode == "text":
        prompt["modalities"] = ["text"]
    elif mode == "it2i":
        prompt["multi_modal_data"] = {"image": [object()]}
    return OmniDiffusionRequest(
        request_id=name,
        prompt=prompt,
        sampling_params=OmniDiffusionSamplingParams(
            width=32, height=32, num_inference_steps=2, num_outputs_per_prompt=count, seed=seed, extra_args=extra or {}
        ),
    )


def _prefix(length, count=1, offset=0, branches=("cond", "uncond")):
    result = {}
    for branch in branches:
        cache = DynamicCache()
        for layer in range(2):
            keys = torch.randn(1, 2, length, 4).expand(count, -1, -1, -1)
            values = torch.randn(1, 2, length, 4).expand(count, -1, -1, -1)
            cache.update(keys, values, layer)
        result[branch] = cache
        result[f"idx_{branch}"] = torch.arange(3).repeat(3, 1) + offset
    return result


@pytest.mark.parametrize("lengths", [(2, 5), (4, 4)])
@pytest.mark.parametrize("branches", [("cond", "uncond"), ("cond", "img_cond", "uncond")])
def test_padded_attention_matches_independent_requests(lengths, branches):
    torch.manual_seed(10)
    prefixes = [_prefix(length, 2, offset=length, branches=branches) for length in lengths]
    merged = merge_conditioning(prefixes, [2, 2], image_tokens=3)
    query, image_k, image_v = [torch.randn(4, 2, 3, 4) for _ in range(3)]
    for branch in branches:
        mask = merged[f"mask_{branch}"]["full_attention"]
        assert (mask is None) == (lengths[0] == lengths[1])
        if mask is not None:
            assert mask.dtype == torch.bool
            assert mask.shape == (4, max(lengths) + 3)
        for layer_idx in range(2):
            layer = merged[branch].layers[layer_idx]
            actual = F.scaled_dot_product_attention(
                query,
                torch.cat([layer.keys, image_k], dim=2),
                torch.cat([layer.values, image_v], dim=2),
                attn_mask=None if mask is None else mask[:, None, None, :],
            )
            expected = []
            for i, prefix in enumerate(prefixes):
                rows = slice(i * 2, (i + 1) * 2)
                source = prefix[branch].layers[layer_idx]
                expected.append(
                    F.scaled_dot_product_attention(
                        query[rows],
                        torch.cat([source.keys, image_k[rows]], dim=2),
                        torch.cat([source.values, image_v[rows]], dim=2),
                    )
                )
                torch.testing.assert_close(
                    merged[f"idx_{branch}"][:, rows], prefix[f"idx_{branch}"].unsqueeze(1).expand(-1, 2, -1)
                )
            torch.testing.assert_close(actual, torch.cat(expected), atol=1e-6, rtol=1e-6)


def test_packed_conditioning_places_prefixes_and_reserves_image_tokens():
    prefixes = [_prefix(2), _prefix(5)]
    merged = merge_conditioning(prefixes, [1, 1], image_tokens=3, packed_varlen=True)
    assert merged["mask_cond"]["full_attention"] is None
    layer = merged["cond"].layers[0]
    assert layer.keys.shape == (1, 2, 13, 4)
    assert layer.sensenova_cu_seqlens_q.tolist() == [0, 3, 6]
    assert layer.sensenova_cu_seqlens_k.tolist() == [0, 5, 13]
    assert layer.sensenova_image_positions.tolist() == [2, 3, 4, 10, 11, 12]
    torch.testing.assert_close(layer.keys[0, :, :2], prefixes[0]["cond"].layers[0].keys[0])
    torch.testing.assert_close(layer.keys[0, :, 5:10], prefixes[1]["cond"].layers[0].keys[0])
    assert not layer.keys[0, :, layer.sensenova_image_positions].count_nonzero()
    image_keys = torch.randn(2, 3, 2, 4)
    image_values = torch.randn_like(image_keys)
    keys, values = write_packed_image_kv(layer, image_keys, image_values)
    torch.testing.assert_close(keys[0, layer.sensenova_image_positions], image_keys.flatten(0, 1))
    torch.testing.assert_close(values[0, layer.sensenova_image_positions], image_values.flatten(0, 1))
    torch.testing.assert_close(keys[0, :2], prefixes[0]["cond"].layers[0].keys[0].transpose(0, 1))
    torch.testing.assert_close(keys[0, 5:10], prefixes[1]["cond"].layers[0].keys[0].transpose(0, 1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FlashAttention")
@pytest.mark.cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("merge_kind", ["request", "cfg"])
def test_packed_flash_attention_matches_independent_ragged_requests(merge_kind):
    torch.manual_seed(17)
    image_tokens, query_heads, kv_heads, head_dim = 12, 32, 8, 128
    prefixes = []
    for length in (7, 19):
        cache = DynamicCache()
        key = torch.randn(1, kv_heads, length, head_dim, device="cuda", dtype=torch.bfloat16)
        value = torch.randn_like(key)
        cache.update(key, value, 0)
        prefixes.append({"cond": cache, "idx_cond": torch.zeros(3, image_tokens, device="cuda")})

    if merge_kind == "request":
        merged = merge_conditioning(prefixes, [1, 1], image_tokens, packed_varlen=True)["cond"]
    else:
        merged, _, _ = merge_cfg_branches(
            [prefix["cond"] for prefix in prefixes],
            [prefix["idx_cond"] for prefix in prefixes],
            image_tokens,
            packed_varlen=True,
        )
    layer = merged.layers[0]
    query = torch.randn(2, image_tokens, query_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    image_key = torch.randn(2, image_tokens, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    image_value = torch.randn_like(image_key)
    packed_key, packed_value = write_packed_image_kv(layer, image_key, image_value)

    impl = FlashAttentionImpl(
        num_heads=query_heads,
        head_size=head_dim,
        softmax_scale=head_dim**-0.5,
        num_kv_heads=kv_heads,
    )
    actual = impl._forward_varlen_packed(
        query.reshape(1, 2 * image_tokens, query_heads, head_dim),
        packed_key,
        packed_value,
        cu_seqlens_q=layer.sensenova_cu_seqlens_q,
        cu_seqlens_k=layer.sensenova_cu_seqlens_k,
        max_seqlen_q=image_tokens,
        max_seqlen_k=layer.sensenova_max_seqlen_k,
    ).reshape_as(query)
    expected = []
    for row, prefix in enumerate(prefixes):
        source = prefix["cond"].layers[0]
        key = torch.cat([source.keys, image_key[row].transpose(0, 1).unsqueeze(0)], dim=2)
        value = torch.cat([source.values, image_value[row].transpose(0, 1).unsqueeze(0)], dim=2)
        out = F.scaled_dot_product_attention(query[row].transpose(0, 1).unsqueeze(0), key, value, enable_gqa=True)
        expected.append(out.squeeze(0).transpose(0, 1))
    torch.testing.assert_close(actual, torch.stack(expected), atol=0.03, rtol=0.03)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FlashAttention")
@pytest.mark.cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_packed_sensenova_attention_dispatch_matches_independent_requests(single_gpu_model_parallel):
    """Exercise the real Attention dispatcher, not FlashAttentionImpl directly."""
    torch.manual_seed(23)
    device, dtype = "cuda", torch.bfloat16
    image_tokens, heads, kv_heads, head_dim = 4, 4, 2, 64
    config = SimpleNamespace(
        hidden_size=heads * head_dim,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        attention_dropout=0.0,
        attention_bias=False,
        rms_norm_eps=1e-6,
    )
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device=device))):
        attention = SenseNovaU1Attention(config, layer_idx=0, prefix="test.attn").to(device=device, dtype=dtype)
    assert attention.attn.attn_backend.supports_multi_doc_packed_varlen()
    with torch.no_grad():
        for param in attention.parameters():
            if param.ndim == 1:
                param.fill_(1)
            else:
                param.normal_(std=0.1)

    prefixes = []
    for length, sign in ((5, 1), (8, -1)):
        cache = DynamicCache()
        keys = torch.zeros(1, kv_heads, length, head_dim, device=device, dtype=dtype)
        values = torch.full_like(keys, 4 * sign)
        cache.update(keys, values, 0)
        prefixes.append({"cond": cache, "idx_cond": torch.zeros(3, image_tokens, device=device)})
    merged = merge_conditioning(prefixes, [1, 1], image_tokens, packed_varlen=True)["cond"]
    assert merged.layers[0].sensenova_packed_varlen

    hidden = torch.randn(2, image_tokens, heads * head_dim, device=device, dtype=dtype)
    cos_t = torch.ones(2, image_tokens, head_dim // 2, device=device, dtype=dtype)
    sin_t = torch.zeros_like(cos_t)
    cos_hw = torch.ones(2, image_tokens, head_dim // 4, device=device, dtype=dtype)
    sin_hw = torch.zeros_like(cos_hw)
    position_embeddings = ((cos_t, sin_t), (cos_hw, sin_hw), (cos_hw, sin_hw))

    def run(states, cache, rope):
        return attention.forward(
            states,
            image_gen_indicators=torch.ones(states.shape[:2], device=device, dtype=torch.bool),
            exist_und=False,
            exist_gen=True,
            indexes=torch.zeros(3, states.shape[0], image_tokens, device=device),
            attention_mask=None,
            past_key_values=cache,
            update_cache=False,
            position_embeddings=rope,
        )

    with torch.inference_mode():
        actual = run(hidden, merged, position_embeddings)
        expected = torch.cat(
            [
                run(
                    hidden[row : row + 1],
                    prefix["cond"],
                    tuple((cos[row : row + 1], sin[row : row + 1]) for cos, sin in position_embeddings),
                )
                for row, prefix in enumerate(prefixes)
            ]
        )
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)
    assert not hasattr(merged.layers[0], "flash_k_cache")
    assert all(prefix["cond"].layers[0].flash_k_cache is not None for prefix in prefixes)

    # A stale SDPA pin leaves attn_backend advertising Flash support while
    # ignoring the packed sequence boundaries. This control must be distinct.
    native_impl = attention.attn.attention
    try:
        attention.attn.attention = attention.attn.sdpa_fallback
        with torch.inference_mode():
            wrong = run(hidden, merged, position_embeddings)
    finally:
        attention.attn.attention = native_impl
    assert (wrong - expected).abs().max().item() > 0.1


def test_flash_kv_preparation_can_allocate_only_the_used_layer():
    cache = DynamicCache()
    for layer_idx in range(2):
        cache.update(torch.ones(1, 2, 3, 4), torch.ones(1, 2, 3, 4), layer_idx)
    _pipeline()._expand_and_prepare_kv(cache, token_hw=2, batch_size=1)
    assert all(not hasattr(layer, "flash_k_cache") for layer in cache.layers)
    prepare_flash_kv_cache(cache, current_len=2, batch_size=1, layer_idx=1)
    assert not hasattr(cache.layers[0], "flash_k_cache")
    assert cache.layers[1].flash_k_cache.shape == (1, 5, 2, 4)


def test_merge_cfg_branches_preserves_branch_rows_and_indexes():
    branches = [_prefix(2, count=2), _prefix(5, count=2, offset=10)]
    merged, indexes, mask = merge_cfg_branches(
        [branch["cond"] for branch in branches],
        [branch["idx_cond"] for branch in branches],
        image_tokens=3,
    )
    layer = merged.layers[0]
    assert layer.keys.shape == (4, 2, 5, 4)
    torch.testing.assert_close(layer.keys[:2, :, :2], branches[0]["cond"].layers[0].keys)
    torch.testing.assert_close(layer.keys[2:, :, :5], branches[1]["cond"].layers[0].keys)
    assert indexes.shape == (3, 4, 3)
    assert mask["full_attention"].tolist() == [
        [True, True, False, False, False, True, True, True],
        [True, True, False, False, False, True, True, True],
        [True] * 8,
        [True] * 8,
    ]

    packed, packed_indexes, packed_mask = merge_cfg_branches(
        [branch["cond"] for branch in branches],
        [branch["idx_cond"] for branch in branches],
        image_tokens=3,
        packed_varlen=True,
    )
    packed_layer = packed.layers[0]
    assert packed_layer.keys.shape == (1, 2, 26, 4)
    assert packed_layer.sensenova_cu_seqlens_q.tolist() == [0, 3, 6, 9, 12]
    assert packed_layer.sensenova_cu_seqlens_k.tolist() == [0, 5, 10, 18, 26]
    assert packed_indexes.shape == indexes.shape
    assert packed_mask["full_attention"] is None


@pytest.mark.parametrize("uncond_lengths", [(3, 3), (4, 6)])
@pytest.mark.parametrize("branches", [("cond", "uncond"), ("cond", "img_cond", "uncond")])
def test_merge_cfg_branches_accepts_request_packed_and_dense_sources(uncond_lengths, branches):
    counts = [1, 2]
    prefixes = [_prefix(length, count, offset=length, branches=branches) for length, count in zip((2, 5), counts)]
    for request, count, length in zip(prefixes, counts, uncond_lengths, strict=True):
        request["uncond"] = _prefix(length, count, branches=("uncond",))["uncond"]
        if "img_cond" in branches:
            request["img_cond"] = _prefix(4, count, branches=("img_cond",))["img_cond"]
    request_cache = merge_conditioning(prefixes, counts, image_tokens=3, packed_varlen=True)
    assert request_cache["cond"].layers[0].sensenova_packed_varlen
    if "img_cond" in branches or uncond_lengths[0] == uncond_lengths[1]:
        dense_branch = "img_cond" if "img_cond" in branches else "uncond"
        assert not getattr(request_cache[dense_branch].layers[0], "sensenova_packed_varlen", False)

    # Packed sources contain image slots; their previous contents must not be
    # copied into the new CFG cache's freshly reserved slots.
    for branch in branches:
        source = request_cache[branch]
        if getattr(source.layers[0], "sensenova_packed_varlen", False):
            for layer in source.layers:
                layer.keys[0, :, layer.sensenova_image_positions] = 7
                layer.values[0, :, layer.sensenova_image_positions] = 7

    merged, indexes, mask = merge_cfg_branches(
        [request_cache[branch] for branch in branches],
        [request_cache[f"idx_{branch}"] for branch in branches],
        image_tokens=3,
        packed_varlen=True,
    )
    assert mask["full_attention"] is None
    expected_indexes = [request_cache[f"idx_{branch}"] for branch in branches]
    lengths = [
        request[branch].get_seq_length()
        for branch in branches
        for request, count in zip(prefixes, counts, strict=True)
        for _ in range(count)
    ]
    expected_offsets = [0]
    for length in lengths:
        expected_offsets.append(expected_offsets[-1] + length + 3)
    torch.testing.assert_close(indexes, torch.cat(expected_indexes, dim=1))
    for layer_idx in range(2):
        layer = merged.layers[layer_idx]
        assert layer.keys.shape == (1, 2, expected_offsets[-1], 4)
        assert layer.sensenova_cu_seqlens_q.tolist() == list(range(0, 3 * (len(lengths) + 1), 3))
        assert layer.sensenova_cu_seqlens_k.tolist() == expected_offsets
        for row, (branch, request_idx, source_row) in enumerate(
            (branch, request_idx, source_row)
            for branch in branches
            for request_idx, count in enumerate(counts)
            for source_row in range(count)
        ):
            length = lengths[row]
            source = prefixes[request_idx][branch].layers[layer_idx]
            start = expected_offsets[row]
            torch.testing.assert_close(layer.keys[0, :, start : start + length], source.keys[source_row])
            torch.testing.assert_close(layer.values[0, :, start : start + length], source.values[source_row])
            assert not layer.keys[0, :, start + length : expected_offsets[row + 1]].count_nonzero()
            assert not layer.values[0, :, start + length : expected_offsets[row + 1]].count_nonzero()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FlashAttention")
@pytest.mark.cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("uncond_lengths", [(9, 9), (5, 17)])
@pytest.mark.parametrize("branches", [("cond", "uncond"), ("cond", "img_cond", "uncond")])
def test_mixed_source_cfg_flash_matches_independent_branches(uncond_lengths, branches):
    torch.manual_seed(23)
    image_tokens, query_heads, kv_heads, head_dim = 12, 32, 8, 128
    requests = []
    for cond_length, uncond_length in zip((7, 19), uncond_lengths, strict=True):
        request = {}
        branch_lengths = {"cond": cond_length, "img_cond": 8, "uncond": uncond_length}
        for branch in branches:
            length = branch_lengths[branch]
            cache = DynamicCache()
            key = torch.randn(1, kv_heads, length, head_dim, device="cuda", dtype=torch.bfloat16)
            cache.update(key, torch.randn_like(key), 0)
            request[branch] = cache
            request[f"idx_{branch}"] = torch.zeros(3, image_tokens, device="cuda")
        requests.append(request)
    request_cache = merge_conditioning(requests, [1, 1], image_tokens, packed_varlen=True)
    merged, _, mask = merge_cfg_branches(
        [request_cache[branch] for branch in branches],
        [request_cache[f"idx_{branch}"] for branch in branches],
        image_tokens,
        packed_varlen=True,
    )
    assert mask["full_attention"] is None
    layer = merged.layers[0]
    total = 2 * len(branches)
    query = torch.randn(total, image_tokens, query_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    image_key = torch.randn(total, image_tokens, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    image_value = torch.randn_like(image_key)
    packed_key, packed_value = write_packed_image_kv(layer, image_key, image_value)
    impl = FlashAttentionImpl(
        num_heads=query_heads,
        head_size=head_dim,
        softmax_scale=head_dim**-0.5,
        num_kv_heads=kv_heads,
    )
    actual = impl._forward_varlen_packed(
        query.reshape(1, total * image_tokens, query_heads, head_dim),
        packed_key,
        packed_value,
        cu_seqlens_q=layer.sensenova_cu_seqlens_q,
        cu_seqlens_k=layer.sensenova_cu_seqlens_k,
        max_seqlen_q=image_tokens,
        max_seqlen_k=layer.sensenova_max_seqlen_k,
    ).reshape_as(query)
    expected = []
    for row, (branch, request) in enumerate((branch, request) for branch in branches for request in requests):
        source = request[branch].layers[0]
        key = torch.cat([source.keys, image_key[row].transpose(0, 1).unsqueeze(0)], dim=2)
        value = torch.cat([source.values, image_value[row].transpose(0, 1).unsqueeze(0)], dim=2)
        out = F.scaled_dot_product_attention(query[row].transpose(0, 1).unsqueeze(0), key, value, enable_gqa=True)
        expected.append(out.squeeze(0).transpose(0, 1))
    torch.testing.assert_close(actual, torch.stack(expected), atol=0.03, rtol=0.03)


def test_pipeline_fuses_packed_request_cfg_and_reuses_prefix_cache(monkeypatch):
    pipe = _pipeline()
    backend = SimpleNamespace(supports_multi_doc_packed_varlen=lambda: True)
    attention = SimpleNamespace(attn_backend=backend)
    pipe.language_model = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(attn=attention))])
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1.is_cfg_group_initialized", lambda: False
    )
    prefixes = [_prefix(2), _prefix(5)]
    for request in prefixes:
        request["uncond"] = _prefix(3, branches=("uncond",))["uncond"]
    request_cache = merge_conditioning(prefixes, [1, 1], image_tokens=3, packed_varlen=True)
    assert request_cache["cond"].layers[0].sensenova_packed_varlen

    calls = []

    def predict_noise(**kwargs):
        calls.append(kwargs)
        return kwargs["input_embeds"][..., :1]

    pipe.predict_noise = predict_noise
    branches_kwargs = [
        {
            "input_embeds": torch.full((2, 3, 4), fill),
            "z": torch.zeros(2, 3, 4),
            "past_key_values": request_cache[branch],
            "indexes_image": request_cache[f"idx_{branch}"],
            "attn_mask": request_cache[f"mask_{branch}"],
            "image_token_num": 3,
        }
        for branch, fill in (("cond", 1.0), ("uncond", 2.0))
    ]
    fused = pipe._predict_fused_cfg_branches(branches_kwargs, request_cache)
    assert fused is not None and len(fused) == 2
    torch.testing.assert_close(fused[0], torch.ones(2, 3, 1))
    torch.testing.assert_close(fused[1], torch.full((2, 3, 1), 2.0))
    assert calls[0]["past_key_values"].layers[0].sensenova_packed_varlen
    assert calls[0]["indexes_image"].shape == (3, 4, 3)
    merged_cache = request_cache["_fused_cfg"][1]
    pipe._predict_fused_cfg_branches(branches_kwargs, request_cache)
    assert request_cache["_fused_cfg"][1] is merged_cache
    assert len(calls) == 2

    branches_kwargs[0]["attn_mask"] = {"full_attention": torch.ones(2, 8, dtype=torch.bool)}
    assert pipe._predict_fused_cfg_branches(branches_kwargs, request_cache) is None
    assert len(calls) == 2

    branches_kwargs[0]["attn_mask"] = {"full_attention": None}
    pipe.od_config.step_execution = True
    assert pipe._predict_fused_cfg_branches(branches_kwargs, request_cache) is None
    assert len(calls) == 2


@pytest.mark.parametrize("count,legacy,expected", [(1, None, 1), (2, None, 2), (1, 3, 3), (3, 3, 3)])
def test_image_count_alias(count, legacy, expected):
    req = _request(count=count, extra={"batch_size": legacy} if legacy is not None else {})
    assert image_count(req.sampling_params) == expected
    pre = get_sensenova_u1_pre_process_func(SimpleNamespace(step_execution=False))
    assert pre(req).sampling_params.num_outputs_per_prompt == expected


def test_step_mode_legacy_image_count_matches_output_metric(monkeypatch):
    req = _request(extra={"batch_size": 3})
    pre = get_sensenova_u1_pre_process_func(SimpleNamespace(step_execution=True))
    assert pre(req).sampling_params.num_outputs_per_prompt == 3
    assert req.batch_compatibility_key is None

    pipe = _pipeline()
    p = pipe._parse_request(DiffusionRequestBatch([req]))
    assert p.batch_size == 3
    output = pipe._denoising_output({}, pipe._init_noise_and_schedule(p).image_prediction)
    monkeypatch.setattr("vllm_omni.diffusion.output_formatter.supports_audio_output", lambda _: False)
    [result] = format_diffusion_outputs(
        request=req,
        od_config=SimpleNamespace(model_class_name="SenseNovaU1Pipeline"),
        diffusion_output=output,
        output_data=output.output,
        postprocess_output=normalize_diffusion_postprocess_output(output.output),
    )
    assert len(result.images) == result.metrics["image_num"] == 3


def test_preprocess_resolves_through_real_registry():
    from vllm_omni.diffusion import registry

    config = SimpleNamespace(model_class_name="SenseNovaU1Pipeline", step_execution=False)
    assert registry._DIFFUSION_PRE_PROCESS_FUNCS[config.model_class_name] == "get_sensenova_u1_pre_process_func"
    pre = registry.get_diffusion_pre_process_func(config)
    assert callable(pre)
    req = _request(extra={"batch_size": 2})
    assert pre(req) is req
    assert req.sampling_params.num_outputs_per_prompt == 2
    assert req.batch_compatibility_key is not None


@pytest.mark.parametrize("count,legacy", [(2, 3), (0, None), (1, -1), (1, True), (1, 1.5)])
def test_invalid_image_counts(count, legacy):
    with pytest.raises(ValueError):
        image_count(SimpleNamespace(num_outputs_per_prompt=count, extra_args={"batch_size": legacy}))


def test_admission_separates_cfg_and_modes_but_not_seed_or_think():
    pre = get_sensenova_u1_pre_process_func(SimpleNamespace(step_execution=False))
    first = pre(_request()).batch_compatibility_key
    assert pre(_request("other", seed=3, extra={"think": True})).batch_compatibility_key == first
    for req in [_request(extra={"cfg_scale": 5}), _request(mode="it2i"), _request(mode="text")]:
        assert pre(req).batch_compatibility_key != first
    assert (
        pre(_request("a", mode="text")).batch_compatibility_key
        != pre(_request("b", mode="text")).batch_compatibility_key
    )
    step_request = _request()
    get_sensenova_u1_pre_process_func(SimpleNamespace(step_execution=True))(step_request)
    assert step_request.batch_compatibility_key is None


def _pipeline():
    pipe = object.__new__(SenseNovaU1Pipeline)
    pipe.device = torch.device("cpu")
    pipe.patch_size = 8
    pipe.merge_size = 2
    pipe.od_config = SimpleNamespace(dtype=torch.float32, cache_backend="none")
    pipe.model_cfg = SimpleNamespace(noise_scale=1.0, noise_scale_mode="constant", noise_scale_max_value=1.0)
    pipe._apply_time_schedule = lambda timesteps, *args: timesteps
    return pipe


def test_noise_is_seeded_per_request_and_supports_generator_lists():
    pipe = _pipeline()
    for seed in (42, 73):
        p = pipe._parse_request(DiffusionRequestBatch([_request(count=2, seed=seed)]))
        actual = pipe._init_noise_and_schedule(p).image_prediction
        expected = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(seed))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    p.image_generator = [torch.Generator().manual_seed(seed) for seed in (42, 73)]
    actual = pipe._init_noise_and_schedule(p).image_prediction
    expected = torch.cat(
        [torch.randn(1, 3, 32, 32, generator=torch.Generator().manual_seed(seed)) for seed in (42, 73)]
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    p.image_generator.pop()
    with pytest.raises(ValueError, match="generator lists"):
        pipe._init_noise_and_schedule(p)


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("mode", ["t2i", "it2i"])
def test_batch_finishes_each_ar_prefix_then_fuses_denoise_and_splits_outputs(count, mode):
    pipe = _pipeline()
    events = []
    requests = [_request("first", count=count, mode=mode), _request("second", count=count, seed=73, mode=mode)]

    def prefix(p, ns, *args):
        events.append(("prefix", p.prompt))
        return SimpleNamespace(cursor=SimpleNamespace(finished=False, name=p.prompt))

    def think(cursor):
        events.append(("think", cursor.name))
        cursor.finished = True

    def caches(p, ns, ctx, *, prepare_flash):
        assert ctx.cursor.finished and not prepare_flash
        events.append(("save", p.prompt))
        cache = _prefix(2 if p.prompt == "first" else 5, count)
        for branch in ("cond", "uncond"):
            cache[f"idx_{branch}"] = torch.zeros(3, ns.token_h * ns.token_w, dtype=torch.long)
        return cache, p.prompt

    def denoise(images, ns, caches, p, step, is_it2i):
        assert is_it2i == (mode == "it2i")
        assert images.shape[0] == p.batch_size == 2 * count
        assert caches["cond"].layers[0].keys.shape[0] == 2 * count
        events.append(("denoise", step))
        return images, torch.zeros_like(images)

    pipe._t2i_prefix = pipe._it2i_prefix = prefix
    pipe._t2i_caches = pipe._it2i_caches = caches
    pipe._think_step = think
    pipe._denoise_one = denoise
    pipe._advance_latents = lambda z, *args: z
    # Input encoding isn't part of this orchestration test.
    pipe._extract_input_images = lambda prompt: [object()] if mode == "it2i" else None
    outputs = pipe.forward(DiffusionRequestBatch(requests))
    assert events == [(stage, name) for name in ("first", "second") for stage in ("prefix", "think", "save")] + [
        ("denoise", 0),
        ("denoise", 1),
    ]
    assert len(outputs) == 2
    for req, output in zip(requests, outputs, strict=True):
        assert output.output["metadata"]["text"]["think_text"] == req.request_id
        ns = pipe._init_noise_and_schedule(pipe._parse_request(DiffusionRequestBatch([req])))
        expected = pipe._denoising_output({}, ns.image_prediction).output["payload"]["image"]
        actual = output.output["payload"]["image"]
        if count == 1:
            assert actual.tobytes() == expected.tobytes()
        else:
            assert len(actual) == count
            assert [im.tobytes() for im in actual] == [im.tobytes() for im in expected]


def test_incompatible_batch_rejected_before_preparation():
    pipe = _pipeline()
    with pytest.raises(ValueError, match="compatible"):
        pipe.forward(DiffusionRequestBatch([_request(), _request("other", extra={"cfg_scale": 9})]))
    with pytest.raises(ValueError, match="text output"):
        pipe.forward(DiffusionRequestBatch([_request(mode="text"), _request("other", mode="text")]))
    pipe.od_config.cache_backend = "cache_dit"
    with pytest.raises(ValueError, match="cache_backend"):
        pipe.forward(DiffusionRequestBatch([_request(), _request("other")]))


@pytest.mark.parametrize("count", [1, 2])
def test_single_request_uses_list_contract_and_formatter_preserves_all_images(count, monkeypatch):
    pipe = _pipeline()
    req = _request(count=count)
    pipe._forward_t2i = lambda p: pipe._denoising_output({}, pipe._init_noise_and_schedule(p).image_prediction)
    outputs = pipe.forward(DiffusionRequestBatch([req]))
    assert isinstance(outputs, list) and len(outputs) == 1
    monkeypatch.setattr("vllm_omni.diffusion.output_formatter.supports_audio_output", lambda _: False)
    [result] = format_diffusion_outputs(
        request=req,
        od_config=SimpleNamespace(model_class_name="SenseNovaU1Pipeline"),
        diffusion_output=outputs[0],
        output_data=outputs[0].output,
        postprocess_output=normalize_diffusion_postprocess_output(outputs[0].output),
    )
    assert result.request_id == req.request_id
    assert len(result.images) == count
    assert result.metrics["image_num"] == count
