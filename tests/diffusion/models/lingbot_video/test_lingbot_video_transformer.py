# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


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
    }
    config.update(overrides)
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


def test_sparse_moe_configures_common_runner_with_lingbot_router_semantics(mocker):
    from vllm_omni.diffusion.models.lingbot_video import lingbot_video_transformer as module

    gate = torch.nn.Linear(16, 8, bias=False)
    runner = torch.nn.Identity()
    gate_factory = mocker.patch.object(module, "GateLinear", return_value=gate)
    fused_moe_factory = mocker.patch.object(module, "FusedMoE", return_value=runner)

    block = module.LingBotVideoSparseMoeBlock(
        hidden_size=16,
        num_experts=8,
        top_k=2,
        moe_intermediate_size=8,
        score_func="sigmoid",
        norm_topk_prob=True,
        n_group=4,
        topk_group=1,
        routed_scaling_factor=2.5,
        n_shared_experts=None,
        prefix="transformer.blocks.0.ffn.experts",
    )

    assert block.experts is runner
    gate_factory.assert_called_once_with(
        16,
        8,
        bias=False,
        out_dtype=torch.float32,
        params_dtype=torch.float32,
        force_fp32_compute=True,
        prefix="transformer.blocks.0.ffn.experts.gate",
    )
    kwargs = fused_moe_factory.call_args.kwargs
    assert kwargs["renormalize"] is True
    assert kwargs["use_grouped_topk"] is True
    assert kwargs["num_expert_group"] == 4
    assert kwargs["topk_group"] == 1
    assert kwargs["scoring_func"] == "sigmoid"
    assert kwargs["routed_scaling_factor"] == 2.5
    assert kwargs["gate"] is gate
    assert kwargs["ckpt_names"] == ("w1", "w2", "w3")
    assert kwargs["e_score_correction_bias"].dtype == torch.float32
    assert not kwargs["e_score_correction_bias"].requires_grad


def test_sparse_moe_block_compacts_and_restores_padding_tokens(mocker):
    from vllm_omni.diffusion.models.lingbot_video import lingbot_video_transformer as module

    mocker.patch.object(
        module,
        "GateLinear",
        return_value=torch.nn.Linear(4, 2, bias=False),
    )
    mocker.patch.object(module, "FusedMoE", return_value=torch.nn.Identity())
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
    routed = mocker.patch.object(
        block,
        "_run_routed_experts",
        side_effect=lambda tokens: tokens * 2,
    )

    hidden_states = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]])
    padding_mask = torch.tensor([1.0, 0.0])

    out = block(hidden_states, padding_mask=padding_mask)

    assert out.shape == hidden_states.shape
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[0, 0], hidden_states[0, 0] * 2)
    assert torch.allclose(out[0, 1], torch.zeros_like(out[0, 1]))
    routed.assert_called_once()
    torch.testing.assert_close(routed.call_args.args[0], hidden_states[:, :1].reshape(1, 4))


def test_sparse_moe_runner_is_a_narrow_compile_boundary():
    from vllm_omni.diffusion.models.lingbot_video import lingbot_video_transformer as module

    assert getattr(
        module.LingBotVideoSparseMoeBlock._run_routed_experts,
        "_torchdynamo_disable",
        False,
    )


def test_sparse_moe_packs_checkpoint_weights_for_common_runner(mocker):
    from vllm_omni.diffusion.models.lingbot_video import lingbot_video_transformer as module

    class FakeRoutedExperts(torch.nn.Module):
        def __init__(self, num_experts, hidden_size, intermediate_size, correction_bias):
            super().__init__()
            self.w13_weight = torch.nn.Parameter(torch.empty(num_experts, 2 * intermediate_size, hidden_size))
            self.w2_weight = torch.nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
            self.e_score_correction_bias = correction_bias

            def load_w13(param, loaded_weight, name, shard_id, expert_id):
                del name
                offset = 0 if shard_id == "w1" else intermediate_size
                param.data[expert_id, offset : offset + intermediate_size].copy_(loaded_weight)

            def load_w2(param, loaded_weight, name, shard_id, expert_id):
                del name, shard_id
                param.data[expert_id].copy_(loaded_weight)

            self.w13_weight.weight_loader = load_w13
            self.w2_weight.weight_loader = load_w2

    class FakeRunner(torch.nn.Module):
        def __init__(self, kwargs):
            super().__init__()
            self.gate = kwargs["gate"]
            self.routed_experts = FakeRoutedExperts(
                kwargs["num_experts"],
                kwargs["hidden_size"],
                kwargs["intermediate_size"],
                kwargs["e_score_correction_bias"],
            )

    mocker.patch.object(
        module,
        "GateLinear",
        side_effect=lambda input_size, output_size, **kwargs: torch.nn.Linear(
            input_size,
            output_size,
            bias=kwargs["bias"],
        ),
    )
    mocker.patch.object(
        module,
        "FusedMoE",
        side_effect=lambda **kwargs: FakeRunner(kwargs),
    )
    mocker.patch.object(
        module,
        "LingBotVideoAttention",
        side_effect=lambda *args, **kwargs: torch.nn.Identity(),
    )
    model = _tiny_transformer(
        depth=1,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        n_shared_experts=1,
        n_group=2,
        topk_group=1,
        routed_scaling_factor=2.5,
    )
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

    model.to(dtype=torch.bfloat16)

    assert model.blocks[0].ffn.experts.gate.weight.dtype == torch.float32
    assert model.blocks[0].ffn.experts.routed_experts.e_score_correction_bias.dtype == torch.float32
    assert model.blocks[0].ffn.experts.routed_experts.w13_weight.dtype == torch.bfloat16
