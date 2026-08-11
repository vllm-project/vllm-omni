# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from itertools import count

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_MODEL_PREFIXES = count()


@pytest.fixture(autouse=True)
def _single_rank_tp_group(init_fake_tp_group):
    """Common FusedMoE's gate is created inside the vLLM TP lifecycle."""
    yield


def _tiny_transformer(**overrides):
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
        "prefix": f"test_lingbot_{next(_MODEL_PREFIXES)}",
    }
    config.update(overrides)
    return LingBotVideoTransformer3DModel(**config)


def test_transformer_declares_pattern_3_cache_dit_contract():
    from cache_dit import ForwardPattern

    from vllm_omni.diffusion.cache.cachedit import CacheDiTAdapterConfig
    from vllm_omni.diffusion.models.lingbot_video.lingbot_video_transformer import (
        LingBotVideoBlock,
        LingBotVideoTransformer3DModel,
    )

    adapter_config = LingBotVideoTransformer3DModel._cache_dit_adapter_config
    assert isinstance(adapter_config, CacheDiTAdapterConfig)
    assert adapter_config.block_forward_patterns == {"blocks": ForwardPattern.Pattern_3}
    assert adapter_config.has_separate_cfg is True
    assert "hidden_states" in inspect.signature(LingBotVideoBlock.forward).parameters


def test_tiny_transformer_installs_and_refreshes_cache_dit():
    import cache_dit

    from vllm_omni.diffusion.cache.cachedit import CacheDiTBackend
    from vllm_omni.diffusion.data import DiffusionCacheConfig

    model = _tiny_transformer(depth=2)
    refiner = _tiny_transformer(depth=2)
    pipeline_type = type("LingBotVideoPipeline", (), {})
    pipeline = pipeline_type()
    pipeline.transformer = model
    pipeline.refiner_transformer = refiner
    pipeline._cache_dit_stage_refreshers = {}
    backend = CacheDiTBackend(DiffusionCacheConfig())

    try:
        backend.enable(pipeline)
        refresh = pipeline._cache_dit_stage_refreshers["transformer"]
        refresh(pipeline, 6, False)
        assert backend.is_enabled()
        assert cache_dit.BlockAdapter.is_cached(model)
        assert cache_dit.BlockAdapter.is_cached(refiner)
        assert set(pipeline._cache_dit_stage_refreshers) == {
            "transformer",
            "refiner_transformer",
        }

        hidden_states = torch.randn(1, 2, 1, 2, 2)
        timestep = torch.tensor([300.0])
        encoder_hidden_states = torch.randn(1, 3, 8)
        encoder_attention_mask = torch.ones(1, 3, dtype=torch.long)

        with torch.no_grad():
            outputs = [
                model(
                    hidden_states,
                    timestep,
                    encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                for _ in range(2)
            ]
        assert all(output.shape == hidden_states.shape for output in outputs)
        assert all(torch.isfinite(output).all() for output in outputs)
    finally:
        backend.disable(pipeline)
        assert not cache_dit.BlockAdapter.is_cached(model)
        assert not cache_dit.BlockAdapter.is_cached(refiner)
        assert pipeline._cache_dit_stage_refreshers == {}


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


def test_packed_attention_mask_excludes_sp_padding_tail():
    from vllm_omni.diffusion.models.lingbot_video import lingbot_video_transformer as module

    mask = module._packed_block_attention_mask(torch.tensor([0, 0, 1, 1, 1]), total_seq_len=8)

    assert mask.shape == (1, 1, 8, 8)
    assert not mask[..., 5:].any()
    assert not mask[..., 5:, :].any()


def test_tiny_transformer_packs_variable_text_lengths():
    model = _tiny_transformer()
    hidden_states = torch.randn(2, 2, 1, 2, 2)
    timestep = torch.tensor([300.0, 300.0])
    encoder_hidden_states = torch.randn(2, 3, 8)
    encoder_attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

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


def test_packed_batch_matches_independent_sample_forwards():
    torch.manual_seed(7)
    model = _tiny_transformer(depth=1)
    hidden_states = torch.randn(2, 2, 1, 2, 2)
    timestep = torch.tensor([300.0, 700.0])
    encoder_hidden_states = torch.randn(2, 3, 8)
    encoder_attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

    with torch.no_grad():
        packed = model(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=False,
        )[0]
        independent = torch.cat(
            [
                model(
                    hidden_states[sample_idx : sample_idx + 1],
                    timestep[sample_idx : sample_idx + 1],
                    encoder_hidden_states[sample_idx : sample_idx + 1, :text_len],
                    return_dict=False,
                )[0]
                for sample_idx, text_len in enumerate((3, 2))
            ],
            dim=0,
        )

    torch.testing.assert_close(packed, independent, rtol=1e-5, atol=1e-6)


def test_transformer_exposes_standard_sp_plan():
    from vllm_omni.diffusion.distributed.sp_plan import validate_sp_plan
    from vllm_omni.diffusion.models.lingbot_video import LingBotVideoTransformer3DModel

    model = _tiny_transformer()
    plan = LingBotVideoTransformer3DModel._sp_plan
    validate_sp_plan(plan)

    assert set(plan) == {"sp_input_boundary", "sp_output_boundary"}
    input_specs = plan["sp_input_boundary"]
    assert all(spec.auto_pad for spec in input_specs.values())
    assert model.sp_input_boundary is not None
    assert model.sp_output_boundary is not None


def test_packed_attention_uses_sdpa_after_standard_attention_resharding(monkeypatch):
    from vllm_omni.diffusion.models.lingbot_video import lingbot_video_transformer as module

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
    out = attn(x, rotary, attention_mask=module._packed_block_attention_mask(torch.tensor([0, 0, 1, 1, 1])))

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


def test_transformer_exposes_block_level_hsdp_contract_for_dense_and_moe():
    dense_model = _tiny_transformer(depth=2)
    moe_model = _tiny_transformer(
        depth=2,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        n_shared_experts=1,
    )

    for model in (dense_model, moe_model):
        model.to(dtype=torch.bfloat16)
        matched = [
            name
            for name, module in model.named_modules()
            if any(condition(name, module) for condition in model._hsdp_shard_conditions)
        ]

        assert matched == ["blocks.0", "blocks.1"]
        assert model._hsdp_preserve_param_dtype is True
        assert model.blocks[0].attn.to_q.weight.dtype == torch.bfloat16
        assert model.blocks[0].norm1.weight.dtype == torch.float32

    assert moe_model.blocks[0].ffn.experts.gate.weight.dtype == torch.float32
    assert moe_model.blocks[0].ffn.experts.routed_experts.e_score_correction_bias.dtype == torch.float32

    assert "blocks.0.ffn.experts" in dict(moe_model.named_modules())
    assert not any(
        condition("blocks.0.ffn.experts", moe_model.blocks[0].ffn.experts)
        for condition in moe_model._hsdp_shard_conditions
    )


def test_transformer_is_a_native_torch_module():
    from diffusers.configuration_utils import ConfigMixin
    from diffusers.models.modeling_utils import ModelMixin

    from vllm_omni.diffusion.models.lingbot_video import LingBotVideoTransformer3DModel

    assert issubclass(LingBotVideoTransformer3DModel, torch.nn.Module)
    assert not issubclass(LingBotVideoTransformer3DModel, (ModelMixin, ConfigMixin))


def test_batched_joint_positions_match_single_sample_reference():
    from vllm_omni.diffusion.models.lingbot_video import lingbot_video_transformer as module

    text_lens = torch.tensor([3, 2])
    batched = module.make_batched_joint_position_ids(
        text_lens,
        max_text_len=3,
        grid_t=1,
        grid_h=2,
        grid_w=2,
    )
    num_video_tokens = 4

    for sample_idx, text_len in enumerate(text_lens):
        reference = module.make_joint_position_ids(
            int(text_len),
            grid_t=1,
            grid_h=2,
            grid_w=2,
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(
            batched[sample_idx, :num_video_tokens],
            reference[:num_video_tokens],
        )
        torch.testing.assert_close(
            batched[sample_idx, num_video_tokens : num_video_tokens + text_len],
            reference[num_video_tokens:],
        )


def test_transformer_hot_path_avoids_host_token_list_conversion():
    from vllm_omni.diffusion.models.lingbot_video import LingBotVideoTransformer3DModel

    source = inspect.getsource(LingBotVideoTransformer3DModel.forward)

    assert ".cpu().tolist()" not in source
    assert "text_lens_list" not in source


def test_sparse_moe_uses_common_runner_and_packs_checkpoint_weights():
    from vllm.model_executor.layers.fused_moe.layer import MoERunner

    from vllm_omni.diffusion.models.lingbot_video import lingbot_video_transformer as module

    model = _tiny_transformer(
        depth=1,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        n_group=2,
        topk_group=1,
    )
    sparse_moe = model.blocks[0].ffn
    assert isinstance(sparse_moe, module.LingBotVideoSparseMoeBlock)
    assert isinstance(sparse_moe.experts, MoERunner)

    w1 = torch.randn(4, 8, 16)
    w2 = torch.randn(4, 16, 8)
    w3 = torch.randn(4, 8, 16)
    gate = torch.randn(4, 16)
    correction_bias = torch.randn(4)
    loaded = model.load_weights(
        [
            ("blocks.0.ffn.experts.w1", w1),
            ("blocks.0.ffn.experts.w2", w2),
            ("blocks.0.ffn.experts.w3", w3),
            ("blocks.0.ffn.router.weight", gate),
            (
                "blocks.0.ffn.router.e_score_correction_bias",
                correction_bias,
            ),
        ]
    )
    params = dict(model.named_parameters())
    w13 = params["blocks.0.ffn.experts.routed_experts.w13_weight"]

    assert loaded == {
        "blocks.0.ffn.experts.gate.weight",
        "blocks.0.ffn.experts.routed_experts.e_score_correction_bias",
        "blocks.0.ffn.experts.routed_experts.w13_weight",
        "blocks.0.ffn.experts.routed_experts.w2_weight",
    }
    torch.testing.assert_close(w13[:, :8], w1)
    torch.testing.assert_close(w13[:, 8:], w3)
    torch.testing.assert_close(
        params["blocks.0.ffn.experts.routed_experts.w2_weight"],
        w2,
    )
    torch.testing.assert_close(params["blocks.0.ffn.experts.gate.weight"], gate)
    torch.testing.assert_close(
        params["blocks.0.ffn.experts.routed_experts.e_score_correction_bias"],
        correction_bias,
    )


def test_moe_block_api_has_no_global_token_shape_workaround():
    from vllm_omni.diffusion.models.lingbot_video import lingbot_video_transformer as module

    block_parameters = inspect.signature(module.LingBotVideoBlock.forward).parameters
    sparse_moe_parameters = inspect.signature(module.LingBotVideoSparseMoeBlock.forward).parameters

    assert "moe_padding_mask" not in block_parameters
    assert "moe_router_target_m" not in block_parameters
    assert "padding_mask" not in sparse_moe_parameters
    assert "router_target_m" not in sparse_moe_parameters


def test_sparse_moe_runner_is_a_narrow_compile_boundary():
    from vllm_omni.diffusion.models.lingbot_video import lingbot_video_transformer as module

    assert getattr(
        module.LingBotVideoSparseMoeBlock._run_routed_experts,
        "_torchdynamo_disable",
        False,
    )


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
    assert "blocks.0.ffn.experts.routed_experts.w13_weight" in model.state_dict()
    assert "blocks.0.ffn.experts.routed_experts.w2_weight" in model.state_dict()
    assert "blocks.0.ffn.shared_experts.gate_proj.weight" in model.state_dict()

    model.to(dtype=torch.bfloat16)

    assert model.blocks[0].ffn.experts.gate.weight.dtype == torch.float32
    assert model.blocks[0].ffn.experts.routed_experts.e_score_correction_bias.dtype == torch.float32
    assert model.blocks[0].ffn.experts.routed_experts.w13_weight.dtype == torch.bfloat16
