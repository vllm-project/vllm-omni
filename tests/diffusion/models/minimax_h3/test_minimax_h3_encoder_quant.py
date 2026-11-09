# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Serialized AutoRound MXFP4 loading for the MiniMax-H3 text encoder."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.utils.network_utils import get_open_port

from vllm_omni.diffusion.models.minimax_h3 import encoder as encoder_module
from vllm_omni.diffusion.models.minimax_h3.encoder import MiniMaxH3Qwen3VLEncoder

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _init_distributed(monkeypatch: pytest.MonkeyPatch):
    """Minimal world_size=1 TP group required by vLLM weight parameters."""
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )

    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(get_open_port()))
    try:
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="env://",
            backend="gloo",
        )
        initialize_model_parallel()
        yield
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.fixture(autouse=True)
def _stub_mxfp4_kernel(monkeypatch: pytest.MonkeyPatch):
    """The loader tests do not dispatch MXFP4 kernels on the CPU lane."""
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.inc.schemes.inc_mxfp4_linear.init_mxfp4_linear_kernel",
        lambda **kwargs: None,
    )


_HIDDEN = 64
_INTERMEDIATE = 128
_NUM_HEADS = 4
_NUM_KV_HEADS = 2
_HEAD_DIM = 16
_VOCAB = 128
_GROUP_SIZE = 32


class _SingleRankGroup:
    rank_in_group = 0
    world_size = 1


def _mx_quant_config() -> dict[str, object]:
    return {
        "quant_method": "auto-round",
        "bits": 4,
        "group_size": _GROUP_SIZE,
        "sym": True,
        "data_type": "mx_fp",
        "packing_format": "auto_round:llm_compressor",
        "block_name_to_quantize": "model.language_model.layers",
        "extra_config": {
            "model.language_model.layers.0.mlp.down_proj": {
                "bits": 16,
                "data_type": "float",
            }
        },
    }


def _tiny_qwen3_vl_config() -> Qwen3VLConfig:
    text_config = Qwen3VLTextConfig(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        num_hidden_layers=2,
        num_attention_heads=_NUM_HEADS,
        num_key_value_heads=_NUM_KV_HEADS,
        head_dim=_HEAD_DIM,
        max_position_embeddings=128,
    )
    vision_config = Qwen3VLVisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        num_position_embeddings=16,
        out_hidden_size=_HIDDEN,
    )
    config = Qwen3VLConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
    )
    setattr(config, "quantization_config", _mx_quant_config())
    return config


def _build_encoder(monkeypatch: pytest.MonkeyPatch) -> MiniMaxH3Qwen3VLEncoder:
    config = _tiny_qwen3_vl_config()
    monkeypatch.setattr(
        Qwen3VLConfig,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: config),
    )
    monkeypatch.setattr(encoder_module, "MiniMaxH3Qwen3VLVisionModel", lambda cfg: nn.Identity())
    return MiniMaxH3Qwen3VLEncoder(
        "unused",
        device=torch.device("cpu"),
        load_model=True,
        encoder_group=_SingleRankGroup(),
    )


def _mxfp4_shard(
    prefix: str,
    rows: int,
    columns: int,
    *,
    packed_value: int,
    scale_value: int,
) -> list[tuple[str, torch.Tensor]]:
    return [
        (f"{prefix}.weight_packed", torch.full((rows, columns // 2), packed_value, dtype=torch.uint8)),
        (f"{prefix}.weight_scale", torch.full((rows, columns // _GROUP_SIZE), scale_value, dtype=torch.uint8)),
    ]


def _checkpoint_weights(text_model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    weights: list[tuple[str, torch.Tensor]] = [
        ("model.language_model.embed_tokens.weight", torch.zeros(_VOCAB, _HIDDEN, dtype=torch.bfloat16)),
    ]
    q_width = _NUM_HEADS * _HEAD_DIM
    kv_width = _NUM_KV_HEADS * _HEAD_DIM
    for index in range(len(text_model.layers)):
        prefix = f"model.language_model.layers.{index}"
        for part, width in (("q", q_width), ("k", kv_width), ("v", kv_width)):
            weights += _mxfp4_shard(
                f"{prefix}.self_attn.{part}_proj",
                width,
                _HIDDEN,
                packed_value=index + 1,
                scale_value=index + 2,
            )
        weights += _mxfp4_shard(
            f"{prefix}.self_attn.o_proj",
            _HIDDEN,
            q_width,
            packed_value=index + 3,
            scale_value=index + 4,
        )
        for part in ("gate", "up"):
            weights += _mxfp4_shard(
                f"{prefix}.mlp.{part}_proj",
                _INTERMEDIATE,
                _HIDDEN,
                packed_value=index + 5,
                scale_value=index + 6,
            )
        # The checkpoint stores layer 0's exempted projection in BF16 and every
        # other decoder projection as serialized MXFP4, independent of how the
        # runtime config happens to classify the layer.
        if index == 0:
            weights.append(
                (
                    f"{prefix}.mlp.down_proj.weight",
                    torch.full((_HIDDEN, _INTERMEDIATE), 7.0, dtype=torch.bfloat16),
                )
            )
        else:
            weights += _mxfp4_shard(
                f"{prefix}.mlp.down_proj",
                _HIDDEN,
                _INTERMEDIATE,
                packed_value=index + 8,
                scale_value=index + 9,
            )
        weights += [
            (f"{prefix}.self_attn.q_norm.weight", torch.zeros(_HEAD_DIM, dtype=torch.bfloat16)),
            (f"{prefix}.self_attn.k_norm.weight", torch.zeros(_HEAD_DIM, dtype=torch.bfloat16)),
            (f"{prefix}.input_layernorm.weight", torch.zeros(_HIDDEN, dtype=torch.bfloat16)),
            (f"{prefix}.post_attention_layernorm.weight", torch.zeros(_HIDDEN, dtype=torch.bfloat16)),
        ]
    return weights


def test_mixed_precision_encoder_loads_bf16_exception_and_mxfp4_layers(monkeypatch):
    encoder = _build_encoder(monkeypatch)
    loaded = encoder.load_weights(_checkpoint_weights(encoder.text_model))

    quant_config = encoder.quant_config
    assert quant_config.block_name_to_quantize == ["text_encoder.text_model.layers"]
    assert "text_encoder.text_model.layers.0.mlp.down_proj" in quant_config.extra_config

    bf16_down = encoder.text_model.layers[0].mlp.down_proj
    quantized_down = encoder.text_model.layers[1].mlp.down_proj
    assert isinstance(bf16_down.quant_method, UnquantizedLinearMethod)
    assert hasattr(bf16_down, "weight") and not hasattr(bf16_down, "weight_packed")
    assert not isinstance(quantized_down.quant_method, UnquantizedLinearMethod)
    assert hasattr(quantized_down, "weight_packed") and hasattr(quantized_down, "weight_scale")

    params = dict(encoder.named_parameters())
    assert loaded == set(params)
    torch.testing.assert_close(
        bf16_down.weight,
        torch.full((_HIDDEN, _INTERMEDIATE), 7.0, dtype=torch.bfloat16),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        quantized_down.weight_packed,
        torch.full((_HIDDEN, _INTERMEDIATE // 2), 9, dtype=torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        quantized_down.weight_scale,
        torch.full((_HIDDEN, _INTERMEDIATE // _GROUP_SIZE), 10, dtype=torch.uint8),
        rtol=0,
        atol=0,
    )


def test_row_parallel_slices_packed_and_scale_against_stored_width():
    from vllm_omni.quantization.inc_config import OmniINCConfig

    quant_config = OmniINCConfig.from_config(_mx_quant_config())
    quant_config.apply_prefix_map({"model.language_model.": "text_encoder.text_model."})
    packed = (
        torch.arange(_HIDDEN * (_INTERMEDIATE // 2), dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(_HIDDEN, _INTERMEDIATE // 2)
    )
    scale = (
        torch.arange(_HIDDEN * (_INTERMEDIATE // _GROUP_SIZE), dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(_HIDDEN, _INTERMEDIATE // _GROUP_SIZE)
    )

    for rank in range(2):
        group = SimpleNamespace(rank_in_group=rank, world_size=2)
        linear = encoder_module.MiniMaxH3Qwen3VLRowParallelLinear(
            group,
            input_size=_INTERMEDIATE,
            output_size=_HIDDEN,
            dtype=torch.bfloat16,
            quant_config=quant_config,
            prefix="text_encoder.text_model.layers.1.mlp.down_proj",
        )
        linear.weight_loader(linear.weight_packed, packed)
        linear.weight_loader(linear.weight_scale, scale)
        packed_width = linear.weight_packed.shape[1]
        scale_width = linear.weight_scale.shape[1]
        torch.testing.assert_close(
            linear.weight_packed,
            packed[:, rank * packed_width : (rank + 1) * packed_width],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            linear.weight_scale,
            scale[:, rank * scale_width : (rank + 1) * scale_width],
            rtol=0,
            atol=0,
        )


def test_encoder_rejects_missing_mxfp4_fused_scale_shard(monkeypatch):
    encoder = _build_encoder(monkeypatch)
    omitted = "model.language_model.layers.1.self_attn.v_proj.weight_scale"
    weights = [item for item in _checkpoint_weights(encoder.text_model) if item[0] != omitted]

    with pytest.raises(RuntimeError, match=r"qkv_proj\.weight_scale: \['v'\]"):
        encoder.load_weights(weights)
