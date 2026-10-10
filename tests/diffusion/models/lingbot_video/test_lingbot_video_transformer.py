# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _tiny_transformer(**overrides):
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.lingbot_video import LingBotVideoTransformer3DModel

    config = {
        "patch_size": (1, 1, 1),
        "in_channels": 2,
        "out_channels": 2,
        "hidden_size": 16,
        "num_attention_heads": 1,
        "depth": 0,
        "intermediate_size": 32,
        "text_dim": 8,
        "freq_dim": 8,
        "axes_dims": (4, 4, 8),
        "axes_lens": (32, 32, 32),
    }
    config.update(overrides)
    # Keep real CPU attention when these tests run on a CUDA CI worker.
    od_config = OmniDiffusionConfig(diffusion_attention_config={"default": {"backend": "TORCH_SDPA"}})
    with set_current_diffusion_config(od_config):
        return LingBotVideoTransformer3DModel(**config)


def test_joint_position_ids_video_then_text_order():
    from vllm_omni.diffusion.models.lingbot_video.lingbot_video_transformer import make_joint_position_ids

    positions = make_joint_position_ids(text_len=3, grid_t=1, grid_h=2, grid_w=2, device=torch.device("cpu"))

    assert positions.shape == (7, 3)
    assert positions[:4, 0].tolist() == [4, 4, 4, 4]
    assert positions[:4, 1:].tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert positions[4:].tolist() == [[1, 0, 0], [2, 0, 0], [3, 0, 0]]


def test_tiny_transformer_depth_zero_forward_shape():
    model = _tiny_transformer()
    hidden_states = torch.randn(1, 2, 1, 2, 2)
    timestep = torch.tensor([300.0])
    encoder_hidden_states = torch.randn(1, 3, 8)
    encoder_attention_mask = torch.ones(1, 3, dtype=torch.long)

    with torch.no_grad():
        out = model(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=False,
        )[0]

    assert out.shape == hidden_states.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("num_experts", [0, 2], ids=["dense", "moe"])
@pytest.mark.parametrize("batch_size", [1, 2], ids=["single", "packed"])
def test_prepared_metadata_matches_fallback_across_timesteps(mocker, num_experts, batch_size):
    from vllm_omni.diffusion.models.lingbot_video.lingbot_video_transformer import LingBotVideoMetadataCache

    generator = torch.Generator().manual_seed(17)
    model = _tiny_transformer(
        depth=1,
        num_experts=num_experts,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
    ).eval()
    # MoE expert weights are allocated with empty(), then normally loaded from a checkpoint.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.1, generator=generator)
    hidden_states = torch.randn(batch_size, 2, 1, 2, 2, generator=generator)
    encoder_hidden_states = torch.randn(batch_size, 3, 8, generator=generator)
    encoder_attention_mask = torch.tensor([[1, 1, 1], [1, 0, 0]])[:batch_size]
    metadata = model.prepare_metadata(hidden_states, encoder_hidden_states, encoder_attention_mask)
    cache = LingBotVideoMetadataCache(metadata)
    original_rotary = metadata.rotary.clone()
    prepare = mocker.spy(model, "prepare_metadata")
    block_inputs = []
    hook = model.blocks[0].register_forward_pre_hook(
        lambda module, args, kwargs: block_inputs.append((args, kwargs)), with_kwargs=True
    )

    try:
        with torch.no_grad():
            for step, time in enumerate([900.0, 500.0, 100.0]):
                # Both timestep and latents change while request geometry stays fixed.
                latents = hidden_states + step * 0.25
                timestep = torch.full((batch_size,), time)
                expected = model(latents, timestep, encoder_hidden_states, encoder_attention_mask, False)[0]
                prepare.reset_mock()
                actual = model(
                    latents,
                    timestep,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    False,
                    metadata_cache=cache,
                )[0]

                prepare.assert_not_called()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert torch.isfinite(actual).all()
                reference_args, reference_kwargs = block_inputs[-2]
                cached_args, cached_kwargs = block_inputs[-1]
                torch.testing.assert_close(cached_args[0], reference_args[0], rtol=0, atol=0)
                torch.testing.assert_close(cached_args[2], reference_args[2], rtol=0, atol=0)
                assert cached_args[2] is metadata.rotary
                assert cached_args[3] is metadata.attention_mask
                assert cached_args[4] is metadata.moe_padding_mask
                assert cached_kwargs["packed_indices"] is metadata.packed_indices
                if batch_size == 2:
                    torch.testing.assert_close(
                        cached_kwargs["packed_indices"]["cu_seqlens_kv"],
                        reference_kwargs["packed_indices"]["cu_seqlens_kv"],
                        rtol=0,
                        atol=0,
                    )
    finally:
        hook.remove()

    assert metadata.text_lens_list == ([3] if batch_size == 1 else [3, 1])
    torch.testing.assert_close(metadata.rotary, original_rotary, rtol=0, atol=0)
    if batch_size == 2:
        assert metadata.packed_indices["cu_seqlens_kv"].tolist() == [0, 7, 12]
        mask = metadata.packed_indices["attention_mask"][0, 0]
        assert mask[:7, :7].all() and mask[7:, 7:].all()
        assert not mask[:7, 7:].any() and not mask[7:, :7].any()


@pytest.mark.parametrize(
    ("batch_size", "degree", "expected_cu_seqlens", "padding_size"),
    [(1, 4, [0, 7, 8], 1), (2, 8, [0, 7, 12, 16], 4)],
)
def test_prepared_metadata_preserves_context_parallel_padding(batch_size, degree, expected_cu_seqlens, padding_size):
    from diffusers.models._modeling_parallel import ContextParallelConfig, ParallelConfig

    from vllm_omni.diffusion.models.lingbot_video.lingbot_video_transformer import LingBotVideoMetadataCache

    model = _tiny_transformer()
    # Exercise packing without replacing metadata construction or requiring collectives.
    model._parallel_config = ParallelConfig(context_parallel_config=ContextParallelConfig(ulysses_degree=degree))
    hidden_states = torch.randn(batch_size, 2, 1, 2, 2)
    encoder_hidden_states = torch.randn(batch_size, 3, 8)
    encoder_attention_mask = torch.tensor([[1, 1, 1], [1, 0, 0]])[:batch_size]
    metadata = model.prepare_metadata(hidden_states, encoder_hidden_states, encoder_attention_mask)
    cache = LingBotVideoMetadataCache(metadata)

    assert metadata.padding_size == padding_size
    assert metadata.packed_indices["cu_seqlens_kv"].tolist() == expected_cu_seqlens
    assert metadata.packed_indices["max_seqlen_in_batch_kv"] == 7
    assert "attention_mask" not in metadata.packed_indices
    assert metadata.rotary.shape == (1, expected_cu_seqlens[-1], 8)
    assert not metadata.rotary[:, -padding_size:].any()
    with torch.no_grad():
        for time in [900.0, 100.0]:
            timestep = torch.full((batch_size,), time)
            expected = model(hidden_states, timestep, encoder_hidden_states, encoder_attention_mask).sample
            actual = model(
                hidden_states,
                timestep,
                encoder_hidden_states,
                encoder_attention_mask,
                metadata_cache=cache,
            ).sample
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Reuse must not append CP padding repeatedly.
    assert metadata.packed_indices["cu_seqlens_kv"].tolist() == expected_cu_seqlens


def test_prepared_metadata_preserves_single_sample_padding_masks():
    model = _tiny_transformer()
    hidden_states = torch.randn(1, 2, 1, 2, 2)
    encoder_hidden_states = torch.randn(1, 4, 8)
    encoder_attention_mask = torch.tensor([[1, 1, 0, 0]])
    metadata = model.prepare_metadata(hidden_states, encoder_hidden_states, encoder_attention_mask)

    assert metadata.text_lens_list == [2]
    assert metadata.attention_mask.shape == (1, 1, 1, 8)
    assert metadata.attention_mask.flatten().tolist() == [True] * 6 + [False] * 2
    assert metadata.moe_padding_mask.tolist() == [1.0] * 6 + [0.0] * 2
    assert metadata.packed_indices is None
    assert metadata.padding_size == 0


@pytest.mark.parametrize("batch_size", [1, 2])
def test_prepared_metadata_supports_maskless_inputs(batch_size):
    from vllm_omni.diffusion.models.lingbot_video.lingbot_video_transformer import LingBotVideoMetadataCache

    model = _tiny_transformer(depth=1).eval()
    hidden_states = torch.randn(batch_size, 2, 1, 2, 2)
    encoder_hidden_states = torch.randn(batch_size, 3, 8)
    metadata = model.prepare_metadata(hidden_states, encoder_hidden_states)

    assert metadata.text_lens_list == [3] * batch_size
    assert metadata.attention_mask is None
    assert metadata.moe_padding_mask is None
    with torch.no_grad():
        timestep = torch.full((batch_size,), 300.0)
        expected = model(hidden_states, timestep, encoder_hidden_states).sample
        actual = model(
            hidden_states, timestep, encoder_hidden_states, metadata_cache=LingBotVideoMetadataCache(metadata)
        ).sample
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_prepared_metadata_is_fresh_for_new_geometry_and_text_lengths():
    from vllm_omni.diffusion.models.lingbot_video.lingbot_video_transformer import LingBotVideoMetadataCache

    model = _tiny_transformer(depth=1).eval()
    first_latents = torch.randn(2, 2, 1, 2, 2)
    first_text = torch.randn(2, 3, 8)
    first_mask = torch.tensor([[1, 1, 1], [1, 0, 0]])
    first = model.prepare_metadata(first_latents, first_text, first_mask)
    first_rotary = first.rotary.clone()

    second_latents = torch.randn(2, 2, 1, 2, 3)
    second_text = torch.randn(2, 4, 8)
    second_mask = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 1]])
    second = model.prepare_metadata(second_latents, second_text, second_mask)

    assert second is not first
    assert second.text_lens_list == [1, 4]
    assert second.packed_indices["cu_seqlens_kv"].tolist() == [0, 7, 17]
    assert second.rotary.shape == (1, 17, 8)
    assert first.text_lens_list == [3, 1]
    assert first.packed_indices["cu_seqlens_kv"].tolist() == [0, 7, 12]
    torch.testing.assert_close(first.rotary, first_rotary, rtol=0, atol=0)
    with torch.no_grad():
        timestep = torch.full((2,), 300.0)
        expected = model(second_latents, timestep, second_text, second_mask).sample
        actual = model(
            second_latents, timestep, second_text, second_mask, metadata_cache=LingBotVideoMetadataCache(second)
        ).sample
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_metadata_cache_is_initialized_after_pre_forward_hooks(mocker):
    from vllm_omni.diffusion.models.lingbot_video.lingbot_video_transformer import LingBotVideoMetadataCache

    model = _tiny_transformer()
    hidden_states = torch.randn(1, 2, 1, 2, 2)
    encoder_hidden_states = torch.randn(1, 3, 8)
    cache = LingBotVideoMetadataCache()
    prepare = mocker.spy(model, "prepare_metadata")
    metadata_at_hook = []
    hook = model.register_forward_pre_hook(lambda module, args: metadata_at_hook.append(cache.metadata))

    try:
        with torch.no_grad():
            for time in [900.0, 100.0]:
                model(hidden_states, torch.tensor([time]), encoder_hidden_states, metadata_cache=cache)
    finally:
        hook.remove()

    prepare.assert_called_once()
    assert metadata_at_hook[0] is None
    assert metadata_at_hook[1] is cache.metadata
    assert cache.metadata.text_lens_list == [3]


def test_packed_attention_uses_sdpa_fallback_without_flash_varlen(monkeypatch):
    from vllm_omni.diffusion.models.lingbot_video import lingbot_video_transformer as module

    monkeypatch.setattr(module, "flash_attn_varlen_func_v3", None)
    attn = module.LingBotVideoAttention(
        hidden_size=8,
        num_heads=2,
        norm_eps=1e-6,
        qkv_bias=False,
        out_bias=False,
    )
    captured = {}

    def fake_sdpa_forward(query, key, value, attn_metadata):
        captured["mask"] = attn_metadata.attn_mask
        return torch.zeros_like(query)

    monkeypatch.setattr(attn.attn.sdpa_fallback, "forward", fake_sdpa_forward)
    x = torch.randn(1, 5, 8)
    rotary = torch.ones(1, 5, 2, dtype=torch.complex64)
    packed_indices = {
        "cu_seqlens_kv": torch.tensor([0, 2, 5], dtype=torch.int32),
        "max_seqlen_in_batch_kv": 3,
        "attention_mask": module._packed_block_attention_mask([2, 3], x.device),
    }

    out = attn(x, rotary, packed_indices=packed_indices)

    assert out.shape == x.shape
    mask = captured["mask"]
    assert mask.shape == (1, 1, 5, 5)
    assert mask[0, 0, :2, :2].all()
    assert mask[0, 0, 2:, 2:].all()
    assert not mask[0, 0, :2, 2:].any()
    assert not mask[0, 0, 2:, :2].any()


def test_tiny_transformer_rejects_invalid_rope_dims():
    from vllm_omni.diffusion.models.lingbot_video import LingBotVideoTransformer3DModel

    with pytest.raises(AssertionError, match="head_dim"):
        LingBotVideoTransformer3DModel(
            hidden_size=16,
            num_attention_heads=1,
            axes_dims=(4, 4, 4),
            depth=0,
        )


def test_transformer_to_keeps_sensitive_modules_in_fp32():
    model = _tiny_transformer()

    model.to(dtype=torch.bfloat16)

    assert model.patch_embedder.weight.dtype == torch.bfloat16
    assert model.time_embedder.linear_1.weight.dtype == torch.float32
    assert model.norm_out_modulation[1].weight.dtype == torch.float32


def test_router_group_limited_topk_uses_bias_corrected_choice():
    from vllm_omni.diffusion.models.lingbot_video import lingbot_video_transformer as module

    router = module.LingBotVideoRouter(
        hidden_size=2,
        num_experts=4,
        top_k=2,
        score_func="sigmoid",
        norm_topk_prob=True,
        n_group=2,
        topk_group=1,
        route_scale=2.5,
    )
    with torch.no_grad():
        router.weight.copy_(
            torch.tensor(
                [
                    [8.0, 0.0],
                    [7.0, 0.0],
                    [-8.0, 0.0],
                    [-7.0, 0.0],
                ]
            )
        )
        router.e_score_correction_bias.copy_(torch.tensor([0.0, 0.0, 2.0, 2.0]))

    top_indices, top_scores = router(torch.tensor([[1.0, 0.0]]))

    assert set(top_indices[0].tolist()) == {2, 3}
    assert torch.allclose(top_scores.sum(dim=-1), torch.tensor([2.5]))
    low_score, high_score = sorted(top_scores[0].tolist())
    assert low_score < 0.8
    assert high_score > 1.7


def test_sparse_moe_block_masks_padding_tokens():
    from vllm_omni.diffusion.models.lingbot_video import lingbot_video_transformer as module

    block = module.LingBotVideoSparseMoeBlock(
        hidden_size=4,
        num_experts=2,
        top_k=1,
        moe_intermediate_size=3,
        score_func="sigmoid",
        norm_topk_prob=True,
        n_group=None,
        topk_group=None,
        routed_scaling_factor=1.0,
        n_shared_experts=None,
    )
    with torch.no_grad():
        block.router.weight.copy_(
            torch.tensor(
                [
                    [8.0, 0.0, 0.0, 0.0],
                    [-8.0, 0.0, 0.0, 0.0],
                ]
            )
        )
        block.router.e_score_correction_bias.zero_()
        block.experts.w1.fill_(0.5)
        block.experts.w2.fill_(0.5)
        block.experts.w3.fill_(0.5)

    hidden_states = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]])
    padding_mask = torch.tensor([1.0, 0.0])

    out = block(hidden_states, padding_mask=padding_mask)

    assert out.shape == hidden_states.shape
    assert torch.isfinite(out).all()
    assert not torch.allclose(out[0, 0], torch.zeros_like(out[0, 0]))
    assert torch.allclose(out[0, 1], torch.zeros_like(out[0, 1]))


@pytest.mark.parametrize(
    "counts_list",
    [
        [0, 0, 0, 0],
        [1, 0, 7, 8, 9],
        [16, 0, 1, 0],
        [0, 0, 33, 0],
    ],
)
def test_grouped_padding_matches_reference(counts_list):
    from vllm_omni.diffusion.models.lingbot_video import lingbot_video_transformer as module

    align = 8
    counts = torch.tensor(counts_list, dtype=torch.int64)
    num_tokens = int(counts.sum())
    tokens = torch.arange(num_tokens * 4, dtype=torch.float32).reshape(num_tokens, 4)

    actual = module.LingBotVideoSparseMoeBlock._pad_grouped_tokens(
        tokens,
        counts,
        align,
    )

    num_experts = len(counts_list)
    max_len = ((num_tokens + num_experts * align + align - 1) // align) * align
    aligned_counts = [((max(count, align) + align - 1) // align) * align for count in counts_list]
    expected_indices = [num_tokens] * max_len
    source_start = 0
    write_start = 0
    for count, aligned_count in zip(counts_list, aligned_counts):
        expected_indices[write_start : write_start + count] = range(
            source_start,
            source_start + count,
        )
        source_start += count
        write_start += aligned_count

    expected_indices_tensor = torch.tensor(expected_indices, dtype=torch.int64)
    tokens_with_pad = torch.vstack((tokens, tokens.new_zeros((tokens.shape[-1],))))
    expected_aligned_counts = torch.tensor(aligned_counts, dtype=torch.int32)

    assert actual[0] == tokens_with_pad.shape
    torch.testing.assert_close(actual[1], tokens_with_pad[expected_indices_tensor])
    torch.testing.assert_close(actual[2], expected_indices_tensor)
    torch.testing.assert_close(actual[3], expected_aligned_counts)


def test_tiny_transformer_constructs_moe_and_dense_layers():
    from vllm_omni.diffusion.models.lingbot_video import lingbot_video_transformer as module

    model = _tiny_transformer(
        depth=2,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        decoder_sparse_step=1,
        mlp_only_layers=(1,),
        n_shared_experts=1,
        n_group=2,
        topk_group=1,
        routed_scaling_factor=2.5,
    )

    assert isinstance(model.blocks[0].ffn, module.LingBotVideoSparseMoeBlock)
    assert isinstance(model.blocks[1].ffn, module.LingBotVideoMLP)
    assert "blocks.0.ffn.experts.w1" in model.state_dict()
    assert "blocks.0.ffn.shared_experts.gate_proj.weight" in model.state_dict()

    model.to(dtype=torch.bfloat16)

    assert model.blocks[0].ffn.router.weight.dtype == torch.float32
    assert model.blocks[0].ffn.experts.w1.dtype == torch.bfloat16
