# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def _init_distributed(monkeypatch):
    """The native transformer uses vLLM parallel linear layers, which require a
    tensor-parallel group; initialize a single-process group for CPU tests."""
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda: default_unquantized_gemm,
    )
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")
    init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


_TINY_CONFIG = {
    "in_channels": 4,
    "out_channels": 4,
    "num_attention_heads": 2,
    "attention_head_dim": 12,
    "num_layers": 1,
    "num_cross_attention_heads": 2,
    "cross_attention_head_dim": 12,
    "cross_attention_dim": 24,
    "caption_channels": 8,
    "mlp_ratio": 2.0,
    "patch_size": (1, 2, 2),
    "sample_size": 4,
    "rope_max_seq_len": 32,
}

_GOLDEN_PREFIX = torch.tensor(
    [
        -0.1159307063,
        0.2003862262,
        -0.3754127920,
        -0.2403523028,
        -1.2376221418,
        -0.6513092518,
        -0.3358457983,
        -0.9894289970,
        0.7858567238,
        0.5089095831,
        -0.5362522602,
        -0.3732060790,
    ]
)


@pytest.mark.parametrize(
    ("elementwise_affine", "bias", "expected_state_keys"),
    [
        (False, False, set()),
        (False, True, set()),
        (True, False, {"weight"}),
        (True, True, {"weight", "bias"}),
    ],
)
@pytest.mark.parametrize(
    ("input_dtype", "parameter_dtype"),
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float16),
        (torch.float32, torch.bfloat16),
    ],
)
def test_sana_rms_norm_matches_diffusers(
    elementwise_affine,
    bias,
    expected_state_keys,
    input_dtype,
    parameter_dtype,
):
    from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm

    from vllm_omni.diffusion.models.sana_video.transformer_sana_video import SanaRMSNorm

    reference = DiffusersRMSNorm(8, eps=1e-5, elementwise_affine=elementwise_affine, bias=bias).to(
        dtype=parameter_dtype
    )
    actual = SanaRMSNorm(8, eps=1e-5, elementwise_affine=elementwise_affine, bias=bias).to(dtype=parameter_dtype)

    assert actual.eps == reference.eps
    assert actual.elementwise_affine == reference.elementwise_affine
    assert actual.dim == reference.dim
    assert set(actual.state_dict()) == expected_state_keys
    assert set(actual.state_dict()) == set(reference.state_dict())

    if elementwise_affine:
        weight = torch.linspace(0.5, 1.5, 8, dtype=parameter_dtype)
        actual.weight.data.copy_(weight)
        reference.weight.data.copy_(weight)
        if bias:
            norm_bias = torch.linspace(-0.25, 0.25, 8, dtype=parameter_dtype)
            actual.bias.data.copy_(norm_bias)
            reference.bias.data.copy_(norm_bias)
    else:
        assert actual.weight is None
        assert actual.bias is None

    hidden_states = torch.linspace(-2.0, 2.0, 48, dtype=input_dtype).reshape(2, 3, 8)
    expected = reference(hidden_states)
    result = actual(hidden_states)

    assert result.dtype == expected.dtype
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("num_channels", "flip_sin_to_cos", "downscale_freq_shift", "scale"),
    [
        (256, True, 0, 1),
        (7, False, 1, 0.5),
        (12, True, 0.5, 2),
    ],
)
@pytest.mark.parametrize("timestep_dtype", [torch.int64, torch.float32, torch.float64])
def test_sana_timesteps_matches_diffusers(
    num_channels,
    flip_sin_to_cos,
    downscale_freq_shift,
    scale,
    timestep_dtype,
):
    from diffusers.models.embeddings import Timesteps

    from vllm_omni.diffusion.models.sana_video.transformer_sana_video import SanaTimesteps

    reference = Timesteps(
        num_channels=num_channels,
        flip_sin_to_cos=flip_sin_to_cos,
        downscale_freq_shift=downscale_freq_shift,
        scale=scale,
    )
    actual = SanaTimesteps(
        num_channels=num_channels,
        flip_sin_to_cos=flip_sin_to_cos,
        downscale_freq_shift=downscale_freq_shift,
        scale=scale,
    )
    timesteps = torch.tensor([0, 1, 500, 999], dtype=timestep_dtype)

    assert actual.state_dict() == reference.state_dict() == {}
    assert actual.num_channels == reference.num_channels
    assert actual.flip_sin_to_cos == reference.flip_sin_to_cos
    assert actual.downscale_freq_shift == reference.downscale_freq_shift
    assert actual.scale == reference.scale
    torch.testing.assert_close(actual(timesteps), reference(timesteps), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("kwargs", "sample_shape", "condition_shape"),
    [
        ({}, (2, 256), None),
        (
            {
                "out_dim": 12,
                "post_act_fn": "silu",
                "cond_proj_dim": 6,
                "sample_proj_bias": False,
            },
            (2, 256),
            (2, 6),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sana_timestep_embedding_matches_diffusers(
    kwargs,
    sample_shape,
    condition_shape,
    dtype,
):
    from diffusers.models.embeddings import TimestepEmbedding

    from vllm_omni.diffusion.models.sana_video.transformer_sana_video import (
        SanaTimestepEmbedding,
    )

    torch.manual_seed(17)
    reference = TimestepEmbedding(256, 24, **kwargs).to(dtype=dtype)
    actual = SanaTimestepEmbedding(256, 24, **kwargs).to(dtype=dtype)
    assert set(actual.state_dict()) == set(reference.state_dict())
    actual.load_state_dict(reference.state_dict())

    sample = torch.randn(sample_shape, dtype=dtype)
    condition = torch.randn(condition_shape, dtype=dtype) if condition_shape else None
    expected = reference(sample, condition)
    result = actual(sample, condition)

    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_sana_timestep_embedding_rejects_unsupported_activation():
    from vllm_omni.diffusion.models.sana_video.transformer_sana_video import (
        SanaTimestepEmbedding,
    )

    with pytest.raises(ValueError, match="act_fn='silu'"):
        SanaTimestepEmbedding(8, 16, act_fn="gelu")
    with pytest.raises(ValueError, match="post_act_fn"):
        SanaTimestepEmbedding(8, 16, post_act_fn="gelu")

    module = SanaTimestepEmbedding(8, 16)
    with pytest.raises(ValueError, match="cond_proj_dim"):
        module(torch.randn(1, 8), torch.randn(1, 4))


@pytest.mark.parametrize("use_additional_conditions", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sana_ada_layer_norm_single_matches_diffusers(
    use_additional_conditions,
    dtype,
):
    from diffusers.models.normalization import AdaLayerNormSingle

    from vllm_omni.diffusion.models.sana_video.transformer_sana_video import (
        SanaAdaLayerNormSingle,
    )

    torch.manual_seed(19)
    reference = AdaLayerNormSingle(
        24,
        use_additional_conditions=use_additional_conditions,
    ).to(dtype=dtype)
    actual = SanaAdaLayerNormSingle(
        24,
        use_additional_conditions=use_additional_conditions,
    ).to(dtype=dtype)
    assert set(actual.state_dict()) == set(reference.state_dict())
    actual.load_state_dict(reference.state_dict())

    timestep = torch.tensor([10, 500])
    added_cond_kwargs = None
    if use_additional_conditions:
        added_cond_kwargs = {
            "resolution": torch.tensor([[480, 832], [704, 1280]]),
            "aspect_ratio": torch.tensor([[832 / 480], [1280 / 704]]),
        }

    expected = reference(
        timestep,
        added_cond_kwargs=added_cond_kwargs,
        batch_size=2,
        hidden_dtype=dtype,
    )
    result = actual(
        timestep,
        added_cond_kwargs=added_cond_kwargs,
        batch_size=2,
        hidden_dtype=dtype,
    )

    assert len(result) == len(expected) == 2
    for actual_tensor, expected_tensor in zip(result, expected):
        assert actual_tensor.shape == expected_tensor.shape
        assert actual_tensor.dtype == expected_tensor.dtype
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


@pytest.mark.parametrize("act_fn", ["gelu_tanh", "silu", "silu_fp32"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sana_pixart_text_projection_matches_diffusers(act_fn, dtype):
    from diffusers.models.embeddings import PixArtAlphaTextProjection

    from vllm_omni.diffusion.models.sana_video.transformer_sana_video import (
        SanaPixArtAlphaTextProjection,
    )

    torch.manual_seed(23)
    reference = PixArtAlphaTextProjection(
        in_features=8,
        hidden_size=24,
        out_features=12,
        act_fn=act_fn,
    ).to(dtype=dtype)
    actual = SanaPixArtAlphaTextProjection(
        in_features=8,
        hidden_size=24,
        out_features=12,
        act_fn=act_fn,
    ).to(dtype=dtype)
    assert set(actual.state_dict()) == set(reference.state_dict())
    actual.load_state_dict(reference.state_dict())

    caption = torch.randn(2, 5, 8, dtype=dtype)
    expected = reference(caption)
    result = actual(caption)

    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("freqs_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(("dim", "max_seq_len", "theta"), [(4, 8, 10000.0), (12, 32, 256.0)])
def test_native_rope_matches_diffusers(dim, max_seq_len, theta, freqs_dtype):
    from diffusers.models.embeddings import get_1d_rotary_pos_embed

    from vllm_omni.diffusion.models.sana_video.transformer_sana_video import (
        _get_1d_rotary_pos_embed,
    )

    expected_cos, expected_sin = get_1d_rotary_pos_embed(
        dim,
        max_seq_len,
        theta,
        use_real=True,
        repeat_interleave_real=True,
        freqs_dtype=freqs_dtype,
    )
    actual_cos, actual_sin = _get_1d_rotary_pos_embed(dim, max_seq_len, theta, freqs_dtype)

    torch.testing.assert_close(actual_cos, expected_cos, rtol=0, atol=0)
    torch.testing.assert_close(actual_sin, expected_sin, rtol=0, atol=0)


def test_tiny_transformer_matches_diffusers_and_frozen_output():
    from diffusers import SanaVideoTransformer3DModel as DiffusersTransformer
    from diffusers.configuration_utils import ConfigMixin
    from diffusers.models.modeling_utils import ModelMixin

    from vllm_omni.diffusion.attention.layer import Attention
    from vllm_omni.diffusion.models.sana_video import SanaVideoTransformer3DModel
    from vllm_omni.diffusion.models.sana_video.transformer_sana_video import (
        SanaDistributedRMSNorm,
        SanaLinearAttention,
        SanaRMSNorm,
        SanaVideoTransformerConfig,
        SanaVideoTransformerOutput,
    )

    torch.manual_seed(7)
    reference = DiffusersTransformer(**_TINY_CONFIG).eval()
    model = SanaVideoTransformer3DModel(**_TINY_CONFIG).eval()
    assert not isinstance(model, ModelMixin)
    assert not isinstance(model, ConfigMixin)
    assert isinstance(model.config, SanaVideoTransformerConfig)
    assert set(model.state_dict()) == set(reference.state_dict())
    model.load_state_dict(reference.state_dict())

    block = model.transformer_blocks[0]
    assert isinstance(block.attn1, SanaLinearAttention)
    assert isinstance(block.attn2.attn, Attention)
    assert block.attn2.attn.role == "cross"
    assert isinstance(block.attn1.norm_q, SanaDistributedRMSNorm)
    assert isinstance(block.attn1.norm_k, SanaDistributedRMSNorm)
    assert isinstance(block.attn2.norm_q, SanaDistributedRMSNorm)
    assert isinstance(block.attn2.norm_k, SanaDistributedRMSNorm)
    assert isinstance(model.caption_norm, SanaRMSNorm)

    torch.manual_seed(11)
    hidden_states = torch.randn(1, 4, 3, 4, 4)
    encoder_hidden_states = torch.randn(1, 5, 8)
    encoder_attention_mask = torch.tensor([[1, 1, 1, 1, 0]])
    timestep = torch.tensor([500.0])

    with torch.no_grad():
        expected = reference(
            hidden_states,
            encoder_hidden_states,
            timestep,
            encoder_attention_mask=encoder_attention_mask,
        ).sample
        actual_output = model(
            hidden_states,
            encoder_hidden_states,
            timestep,
            encoder_attention_mask=encoder_attention_mask,
        )
        actual = actual_output.sample

    assert isinstance(actual_output, SanaVideoTransformerOutput)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual.flatten()[: len(_GOLDEN_PREFIX)], _GOLDEN_PREFIX, rtol=1e-5, atol=1e-5)


def test_encoder_attention_mask_uses_bool_sdpa_contract(monkeypatch):
    from vllm_omni.diffusion.models.sana_video import SanaVideoTransformer3DModel

    torch.manual_seed(37)
    model = SanaVideoTransformer3DModel(**_TINY_CONFIG).eval()
    captured = {}
    omni_attention = model.transformer_blocks[0].attn2.attn

    def fake_sdpa_forward(query, key, value, attn_metadata):
        captured["query_shape"] = query.shape
        captured["key_shape"] = key.shape
        captured["mask"] = attn_metadata.attn_mask.detach().clone()
        return torch.zeros_like(query)

    def unexpected_backend_forward(*args, **kwargs):
        raise AssertionError("masked SANA cross-attention must use SDPA")

    monkeypatch.setattr(omni_attention.sdpa_fallback, "forward", fake_sdpa_forward)
    monkeypatch.setattr(omni_attention, "forward", unexpected_backend_forward)

    hidden_states = torch.randn(2, 4, 3, 4, 4)
    encoder_hidden_states = torch.randn(2, 5, 8)
    input_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
        ],
        dtype=torch.int64,
    )

    with torch.no_grad():
        output = model(
            hidden_states,
            encoder_hidden_states,
            torch.tensor([100.0, 500.0]),
            encoder_attention_mask=input_mask,
        ).sample

    assert output.shape == hidden_states.shape
    assert captured["query_shape"][:2] == (2, 12)
    assert captured["key_shape"][:2] == (2, 5)
    assert captured["query_shape"][1] != captured["key_shape"][1]
    assert captured["mask"].shape == (2, 5)
    assert captured["mask"].dtype == torch.bool
    assert torch.equal(captured["mask"], input_mask.bool())


def test_encoder_attention_mask_rejects_additive_bias():
    from vllm_omni.diffusion.models.sana_video import SanaVideoTransformer3DModel

    model = SanaVideoTransformer3DModel(**_TINY_CONFIG).eval()
    additive_mask = torch.tensor([[0.0, 0.0, 0.0, -10000.0, -10000.0]])

    with pytest.raises(TypeError, match="not a floating-point additive attention bias"):
        model(
            torch.randn(1, 4, 3, 4, 4),
            torch.randn(1, 5, 8),
            torch.tensor([500.0]),
            encoder_attention_mask=additive_mask,
        )


def test_linear_attention_requires_rotary_embeddings():
    from vllm_omni.diffusion.models.sana_video.transformer_sana_video import SanaLinearAttention

    attention = SanaLinearAttention(
        dim=24,
        num_heads=2,
        head_dim=12,
        dropout=0.0,
        bias=True,
        qk_norm="rms_norm_across_heads",
    )

    with pytest.raises(ValueError, match="requires rotary_emb"):
        attention(torch.randn(1, 4, 24))


def test_linear_attention_bfloat16_matches_diffusers_mixed_precision():
    from diffusers.models.attention_processor import Attention as DiffusersAttention
    from diffusers.models.transformers.transformer_sana_video import SanaLinearAttnProcessor3_0

    from vllm_omni.diffusion.models.sana_video.transformer_sana_video import SanaLinearAttention

    torch.manual_seed(17)
    reference = DiffusersAttention(
        query_dim=24,
        heads=2,
        dim_head=12,
        kv_heads=2,
        qk_norm="rms_norm_across_heads",
        dropout=0.0,
        bias=True,
        cross_attention_dim=None,
        processor=SanaLinearAttnProcessor3_0(),
    ).to(dtype=torch.bfloat16)
    actual = SanaLinearAttention(
        dim=24,
        num_heads=2,
        head_dim=12,
        dropout=0.0,
        bias=True,
        qk_norm="rms_norm_across_heads",
    ).to(dtype=torch.bfloat16)
    actual.load_state_dict(reference.state_dict())

    hidden_states = torch.randn(1, 5, 24, dtype=torch.bfloat16)
    rotary_emb = (
        torch.randn(1, 5, 1, 12, dtype=torch.bfloat16),
        torch.randn(1, 5, 1, 12, dtype=torch.bfloat16),
    )

    with torch.no_grad():
        expected = reference(hidden_states, rotary_emb=rotary_emb)
        result = actual(hidden_states, rotary_emb=rotary_emb)

    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_transformer_block_without_cross_attention_keeps_feed_forward_norm():
    from vllm_omni.diffusion.models.sana_video.transformer_sana_video import SanaVideoTransformerBlock

    block = SanaVideoTransformerBlock(
        dim=24,
        num_attention_heads=2,
        attention_head_dim=12,
        num_cross_attention_heads=2,
        cross_attention_head_dim=12,
        cross_attention_dim=None,
        mlp_ratio=2.0,
    )

    assert block.attn2 is None
    assert isinstance(block.norm2, torch.nn.LayerNorm)


def test_guidance_transformer_matches_diffusers_and_tuple_output():
    from diffusers import SanaVideoTransformer3DModel as DiffusersTransformer

    from vllm_omni.diffusion.models.sana_video import SanaVideoTransformer3DModel
    from vllm_omni.diffusion.models.sana_video.transformer_sana_video import (
        SanaTimestepEmbedding,
        SanaTimesteps,
    )

    config = _TINY_CONFIG | {"guidance_embeds": True}
    torch.manual_seed(29)
    reference = DiffusersTransformer(**config).eval()
    model = SanaVideoTransformer3DModel(**config).eval()
    assert set(model.state_dict()) == set(reference.state_dict())
    model.load_state_dict(reference.state_dict())
    assert isinstance(model.time_embed.time_proj, SanaTimesteps)
    assert isinstance(model.time_embed.timestep_embedder, SanaTimestepEmbedding)
    assert isinstance(model.time_embed.guidance_condition_proj, SanaTimesteps)
    assert isinstance(model.time_embed.guidance_embedder, SanaTimestepEmbedding)

    torch.manual_seed(31)
    hidden_states = torch.randn(1, 4, 3, 4, 4)
    encoder_hidden_states = torch.randn(1, 5, 8)
    encoder_attention_mask = torch.tensor([[1, 1, 1, 1, 0]])
    timestep = torch.tensor([500.0])
    guidance = torch.tensor([6.0])

    with torch.no_grad():
        expected = reference(
            hidden_states,
            encoder_hidden_states,
            timestep,
            guidance=guidance,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=False,
        )
        actual = model(
            hidden_states,
            encoder_hidden_states,
            timestep,
            guidance=guidance,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=False,
        )

    assert isinstance(actual, tuple)
    assert len(actual) == len(expected) == 1
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)


def test_native_transformer_config_filters_diffusers_metadata():
    from vllm_omni.diffusion.models.sana_video.transformer_sana_video import SanaVideoTransformerConfig

    config = SanaVideoTransformerConfig.from_dict(
        _TINY_CONFIG | {"_class_name": "SanaVideoTransformer3DModel", "_diffusers_version": "0.38.0"}
    )
    assert config.patch_size == (1, 2, 2)

    with pytest.raises(ValueError, match="unsupported_field"):
        SanaVideoTransformerConfig.from_dict(_TINY_CONFIG | {"unsupported_field": True})


def test_dual_vae_variant_resolution():
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline

    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import (
        DistributedAutoencoderKLLTX2Video,
    )
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        DistributedAutoencoderKLWan,
    )
    from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
    from vllm_omni.diffusion.models.sana_video.pipeline_sana_video import (
        SanaVideoPipeline,
        _resolve_vae_class_and_dtype,
    )

    assert not issubclass(SanaVideoPipeline, DiffusionPipeline)
    assert issubclass(SanaVideoPipeline, ProgressBarMixin)

    vae_class, dtype = _resolve_vae_class_and_dtype("AutoencoderKLWan", torch.bfloat16)
    assert vae_class is DistributedAutoencoderKLWan
    assert dtype is torch.float32

    vae_class, dtype = _resolve_vae_class_and_dtype("AutoencoderKLLTX2Video", torch.bfloat16)
    assert vae_class is DistributedAutoencoderKLLTX2Video
    assert dtype is torch.bfloat16

    with pytest.raises(ValueError, match="Unsupported SANA-Video VAE"):
        _resolve_vae_class_and_dtype("UnknownVAE", torch.bfloat16)
