# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Layer-level contracts for BAGEL online FP8 quantization."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

from tests.helpers.runtime import get_open_port
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from vllm_omni.diffusion.models.bagel.bagel_transformer import (
    Bagel,
    Qwen2MoTConfig,
    Qwen2MoTForCausalLM,
)
from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization import build_quant_config

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    pytest.mark.gpu,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="BAGEL online FP8 requires CUDA"),
]

_NUM_LAYERS = 2


@pytest.fixture(scope="module", autouse=True)
def _single_rank_tp_env():
    """Initialize the single-rank TP environment required by vLLM linears."""
    os.environ.update(
        {
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(get_open_port()),
        }
    )
    current_omni_platform.set_device(0)
    if not torch.distributed.is_initialized():
        init_distributed_environment(world_size=1, rank=0, local_rank=0)
    if not model_parallel_is_initialized():
        initialize_model_parallel(
            data_parallel_size=1,
            cfg_parallel_size=1,
            sequence_parallel_size=1,
            ulysses_degree=1,
            ring_degree=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
    yield
    destroy_distributed_env()


def _tiny_bagel_config() -> tuple[Qwen2MoTConfig, SimpleNamespace]:
    llm_config = Qwen2MoTConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=_NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
    )
    bagel_config = SimpleNamespace(
        llm_config=llm_config,
        visual_gen=True,
        visual_und=True,
        vae_config=SimpleNamespace(z_channels=4, downsample=8),
        vit_config=SimpleNamespace(hidden_size=32, patch_size=2),
        latent_patch_size=2,
        max_latent_size=4,
        timestep_shift=1.0,
        vit_max_num_patch_per_side=4,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
    )
    return llm_config, bagel_config


def _build_tiny_bagel(quant_config) -> Bagel:
    llm_config, bagel_config = _tiny_bagel_config()
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        language_model = Qwen2MoTForCausalLM(
            llm_config,
            quant_config=quant_config,
            prefix="bagel.language_model",
        )
        model = Bagel(
            language_model=language_model,
            vit_model=torch.nn.Identity(),
            config=bagel_config,
            quant_config=quant_config,
            prefix="bagel",
        )
    finally:
        torch.set_default_dtype(previous_dtype)
    # vLLM creates online-quantized weights on the meta device so the loader
    # can materialize them without an intermediate allocation.
    return model.to_empty(device="cuda")


def _test_vllm_config() -> VllmConfig:
    config = VllmConfig()
    config.model_config = SimpleNamespace(dtype=torch.bfloat16)
    return config


def _is_quantized_linear(module: torch.nn.Module) -> bool:
    quant_method = getattr(module, "quant_method", None)
    return (
        not isinstance(module, VocabParallelEmbedding)
        and quant_method is not None
        and not isinstance(quant_method, UnquantizedLinearMethod)
    )


def _expected_quantized_modules() -> set[str]:
    expected: set[str] = set()
    for layer_idx in range(_NUM_LAYERS):
        layer = f"language_model.model.layers.{layer_idx}"
        expected.update(
            {
                f"{layer}.self_attn.qkv_proj",
                f"{layer}.self_attn.qkv_proj.gen_exp",
                f"{layer}.self_attn.o_proj",
                f"{layer}.self_attn.o_proj.gen_exp",
                f"{layer}.mlp.gate_up_proj",
                f"{layer}.mlp.down_proj",
                f"{layer}.mlp_moe_gen.gate_up_proj",
                f"{layer}.mlp_moe_gen.down_proj",
            }
        )
    expected.update({"connector.fc1", "connector.fc2"})
    return expected


def _initialize_float_parameters(model: torch.nn.Module) -> None:
    generator = torch.Generator(device="cuda").manual_seed(17)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.is_floating_point():
                parameter.normal_(mean=0.0, std=0.02, generator=generator)


def _process_quantized_weights(model: torch.nn.Module) -> None:
    for module in model.modules():
        if _is_quantized_linear(module):
            module.quant_method.process_weights_after_loading(module)


def _mixed_route_reference(
    x: torch.Tensor,
    text_indices: torch.Tensor,
    vae_indices: torch.Tensor,
    text_weight: torch.Tensor,
    vae_weight: torch.Tensor,
    text_bias: torch.Tensor | None,
    vae_bias: torch.Tensor | None,
) -> torch.Tensor:
    output = torch.empty(
        x.shape[0],
        text_weight.shape[0],
        device=x.device,
        dtype=x.dtype,
    )
    output[text_indices] = torch.nn.functional.linear(x[text_indices], text_weight, text_bias)
    output[vae_indices] = torch.nn.functional.linear(x[vae_indices], vae_weight, vae_bias)
    return output


def _marked_weight(shape: torch.Size | tuple[int, ...], value: float) -> torch.Tensor:
    return torch.full(shape, value, device="cuda", dtype=torch.bfloat16)


def test_bagel_fp8_config_covers_all_quantization_aware_linears():
    """The public flat FP8 config reaches both MoT paths and the connector."""
    quant_config = build_quant_config("fp8")

    with set_current_vllm_config(_test_vllm_config()):
        model = _build_tiny_bagel(quant_config)

    actual = {name for name, module in model.named_modules() if _is_quantized_linear(module)}
    assert actual == _expected_quantized_modules()

    # These precision-sensitive or non-vLLM bridge/output layers intentionally
    # stay in BF16 because they do not use vLLM quantization-aware linears.
    bf16_linears = (
        model.vae2llm,
        model.llm2vae,
        model.time_embedder.mlp[0],
        model.time_embedder.mlp[2],
        model.language_model.lm_head,
    )
    for module in bf16_linears:
        assert not _is_quantized_linear(module)
        assert module.weight.dtype == torch.bfloat16


def test_bagel_online_fp8_materializes_text_and_generation_weights():
    """Post-load conversion creates FP8 weights/scales for text and gen_exp."""
    quant_config = build_quant_config("fp8")

    with set_current_vllm_config(_test_vllm_config()):
        model = _build_tiny_bagel(quant_config)
        _initialize_float_parameters(model)
        _process_quantized_weights(model)

    modules = dict(model.named_modules())
    for name in _expected_quantized_modules():
        module = modules[name]
        # Native FP8-capable GPUs retain E4M3 weights. Pre-Ada GPUs such as
        # A800 use vLLM's Marlin fallback, which packs FP8 bytes into int32.
        assert module.weight.dtype in (torch.float8_e4m3fn, torch.int32), name
        assert hasattr(module, "weight_scale"), name
        assert module.weight_scale.numel() > 0, name
        assert torch.isfinite(module.weight_scale).all(), name

    # Make the dual-weight MoT contract explicit: a successful text-weight
    # conversion is insufficient unless the generation expert was converted too.
    for layer in model.language_model.model.layers:
        for projection in (layer.self_attn.qkv_proj, layer.self_attn.o_proj):
            assert projection.weight.dtype in (torch.float8_e4m3fn, torch.int32)
            assert projection.gen_exp.weight.dtype == projection.weight.dtype
            assert hasattr(projection, "weight_scale")
            assert hasattr(projection.gen_exp, "weight_scale")
            assert projection.weight_scale.shape == projection.gen_exp.weight_scale.shape


def test_bagel_online_fp8_mot_mixed_route_matches_bf16_reference():
    """FP8 QKV/O projections compute both text and generation token routes."""
    quant_config = build_quant_config("fp8")

    with set_current_vllm_config(_test_vllm_config()):
        model = _build_tiny_bagel(quant_config)
        _initialize_float_parameters(model)

        layer = model.language_model.model.layers[0]
        projections = (layer.self_attn.qkv_proj, layer.self_attn.o_proj)
        references = [
            (
                projection.weight.detach().clone(),
                projection.gen_exp.weight.detach().clone(),
                None if projection.bias is None else projection.bias.detach().clone(),
                None if projection.gen_exp.bias is None else projection.gen_exp.bias.detach().clone(),
            )
            for projection in projections
        ]
        _process_quantized_weights(model)

        text_indices = torch.tensor([0, 3, 5, 7], device="cuda")
        vae_indices = torch.tensor([1, 2, 4, 6], device="cuda")
        generator = torch.Generator(device="cuda").manual_seed(29)

        for projection, (text_weight, vae_weight, text_bias, vae_bias) in zip(
            projections,
            references,
            strict=True,
        ):
            x = torch.randn(
                8,
                projection.input_size_per_partition,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            expected = _mixed_route_reference(
                x,
                text_indices,
                vae_indices,
                text_weight,
                vae_weight,
                text_bias,
                vae_bias,
            )
            actual, _ = projection(x, text_indices, vae_indices)

            assert torch.isfinite(actual).all()
            route_similarity = torch.nn.functional.cosine_similarity(
                actual.float(),
                expected.float(),
                dim=-1,
            )
            assert route_similarity.min().item() > 0.98


def test_bagel_checkpoint_remaps_text_and_generation_experts():
    """Separated checkpoint projections load into the correct fused MoT slices."""
    with set_current_vllm_config(_test_vllm_config()):
        model = _build_tiny_bagel(quant_config=None)

    language_model = model.language_model
    layer = language_model.model.layers[0]
    attention = layer.self_attn
    hidden_size = language_model.config.hidden_size
    intermediate_size = language_model.config.intermediate_size

    checkpoint_weights = [
        ("model.layers.0.self_attn.q_proj.weight", _marked_weight((attention.q_size, hidden_size), 1)),
        ("model.layers.0.self_attn.k_proj.weight", _marked_weight((attention.kv_size, hidden_size), 2)),
        ("model.layers.0.self_attn.v_proj.weight", _marked_weight((attention.kv_size, hidden_size), 3)),
        ("model.layers.0.self_attn.q_proj_moe_gen.weight", _marked_weight((attention.q_size, hidden_size), 11)),
        ("model.layers.0.self_attn.k_proj_moe_gen.weight", _marked_weight((attention.kv_size, hidden_size), 12)),
        ("model.layers.0.self_attn.v_proj_moe_gen.weight", _marked_weight((attention.kv_size, hidden_size), 13)),
        ("model.layers.0.mlp.gate_proj.weight", _marked_weight((intermediate_size, hidden_size), 21)),
        ("model.layers.0.mlp.up_proj.weight", _marked_weight((intermediate_size, hidden_size), 22)),
        ("model.layers.0.mlp_moe_gen.gate_proj.weight", _marked_weight((intermediate_size, hidden_size), 31)),
        ("model.layers.0.mlp_moe_gen.up_proj.weight", _marked_weight((intermediate_size, hidden_size), 32)),
        ("model.layers.0.self_attn.o_proj_moe_gen.weight", _marked_weight((hidden_size, hidden_size), 41)),
        ("model.layers.0.input_layernorm_moe_gen.weight", _marked_weight((hidden_size,), 51)),
        ("model.layers.0.post_attention_layernorm_moe_gen.weight", _marked_weight((hidden_size,), 52)),
        ("model.layers.0.self_attn.q_norm_moe_gen.weight", _marked_weight((attention.head_dim,), 53)),
        ("model.layers.0.self_attn.k_norm_moe_gen.weight", _marked_weight((attention.head_dim,), 54)),
        ("model.norm_moe_gen.weight", _marked_weight((hidden_size,), 55)),
    ]

    loaded = language_model.load_weights(checkpoint_weights)

    expected_loaded = {
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.self_attn.qkv_proj.gen_exp.weight",
        "model.layers.0.mlp.gate_up_proj.weight",
        "model.layers.0.mlp_moe_gen.gate_up_proj.weight",
        "model.layers.0.self_attn.o_proj.gen_exp.weight",
        "model.layers.0.input_layernorm.gen_weight",
        "model.layers.0.post_attention_layernorm.gen_weight",
        "model.layers.0.self_attn.q_norm.gen_weight",
        "model.layers.0.self_attn.k_norm.gen_weight",
        "model.norm.gen_weight",
    }
    assert loaded == expected_loaded

    torch.testing.assert_close(
        attention.qkv_proj.weight,
        torch.cat(
            [
                _marked_weight((attention.q_size, hidden_size), 1),
                _marked_weight((attention.kv_size, hidden_size), 2),
                _marked_weight((attention.kv_size, hidden_size), 3),
            ]
        ),
    )
    torch.testing.assert_close(
        attention.qkv_proj.gen_exp.weight,
        torch.cat(
            [
                _marked_weight((attention.q_size, hidden_size), 11),
                _marked_weight((attention.kv_size, hidden_size), 12),
                _marked_weight((attention.kv_size, hidden_size), 13),
            ]
        ),
    )
    torch.testing.assert_close(
        layer.mlp.gate_up_proj.weight,
        torch.cat(
            [
                _marked_weight((intermediate_size, hidden_size), 21),
                _marked_weight((intermediate_size, hidden_size), 22),
            ]
        ),
    )
    torch.testing.assert_close(
        layer.mlp_moe_gen.gate_up_proj.weight,
        torch.cat(
            [
                _marked_weight((intermediate_size, hidden_size), 31),
                _marked_weight((intermediate_size, hidden_size), 32),
            ]
        ),
    )
    torch.testing.assert_close(
        attention.o_proj.gen_exp.weight,
        _marked_weight(attention.o_proj.gen_exp.weight.shape, 41),
    )
    torch.testing.assert_close(
        layer.input_layernorm.gen_weight,
        _marked_weight(layer.input_layernorm.gen_weight.shape, 51),
    )
    torch.testing.assert_close(
        layer.post_attention_layernorm.gen_weight,
        _marked_weight(layer.post_attention_layernorm.gen_weight.shape, 52),
    )
    torch.testing.assert_close(
        attention.q_norm.gen_weight,
        _marked_weight(attention.q_norm.gen_weight.shape, 53),
    )
    torch.testing.assert_close(
        attention.k_norm.gen_weight,
        _marked_weight(attention.k_norm.gen_weight.shape, 54),
    )
    torch.testing.assert_close(
        language_model.model.norm.gen_weight,
        _marked_weight(language_model.model.norm.gen_weight.shape, 55),
    )
