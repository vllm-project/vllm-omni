# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Shared DAC codec construction for Fish Speech S2 Pro.

Used by both the encoder (voice cloning, CPU) and the decoder (Stage 1, GPU).
"""

from __future__ import annotations

import torch.nn as nn
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from torch.nn.utils.weight_norm import WeightNorm

# Default DAC codec config matching s2-pro's modded_dac_vq.yaml.
DAC_SAMPLE_RATE = 44100
DAC_HOP_LENGTH = 2048  # 512 (decoder upsample) * 4 (quantizer upsample)
DAC_NUM_CODEBOOKS = 10  # 1 semantic + 9 residual
DAC_SEMANTIC_CODEBOOK_SIZE = 4096
DAC_RESIDUAL_CODEBOOK_SIZE = 1024


def dac_codebook_sizes() -> tuple[int, ...]:
    """Entries of each codebook, in code order: the semantic one, then the residual ones."""
    return (DAC_SEMANTIC_CODEBOOK_SIZE,) + (DAC_RESIDUAL_CODEBOOK_SIZE,) * (DAC_NUM_CODEBOOKS - 1)


def build_dac_codec() -> nn.Module:
    """Construct a DAC codec model (uninitialized weights).

    Returns the model on CPU in eval mode -- caller is responsible for
    loading weights and moving to the target device.
    """
    from vllm_omni.model_executor.models.fish_speech.dac_modules import (
        DAC,
        DownsampleResidualVectorQuantize,
        ModelArgs,
        WindowLimitedTransformer,
    )

    base_transformer_kwargs = dict(
        block_size=16384,
        n_local_heads=-1,
        head_dim=64,
        rope_base=10000,
        norm_eps=1e-5,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        channels_first=True,
    )

    def _make_transformer_config(**kw):
        return ModelArgs(**{**base_transformer_kwargs, **kw})

    quantizer_transformer_config = ModelArgs(
        block_size=4096,
        n_layer=8,
        n_head=16,
        dim=1024,
        intermediate_size=3072,
        n_local_heads=-1,
        head_dim=64,
        rope_base=10000,
        norm_eps=1e-5,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        channels_first=True,
    )
    post_module = WindowLimitedTransformer(
        causal=True,
        window_size=128,
        input_dim=1024,
        config=quantizer_transformer_config,
    )
    pre_module = WindowLimitedTransformer(
        causal=True,
        window_size=128,
        input_dim=1024,
        config=quantizer_transformer_config,
    )
    quantizer = DownsampleResidualVectorQuantize(
        input_dim=1024,
        n_codebooks=DAC_NUM_CODEBOOKS - 1,
        codebook_size=DAC_RESIDUAL_CODEBOOK_SIZE,
        codebook_dim=8,
        quantizer_dropout=0.0,
        downsample_factor=[2, 2],
        post_module=post_module,
        pre_module=pre_module,
        semantic_codebook_size=DAC_SEMANTIC_CODEBOOK_SIZE,
    )
    codec = DAC(
        encoder_dim=64,
        encoder_rates=[2, 4, 8, 8],
        decoder_dim=1536,
        decoder_rates=[8, 8, 4, 2],
        quantizer=quantizer,
        sample_rate=DAC_SAMPLE_RATE,
        causal=True,
        encoder_transformer_layers=[0, 0, 0, 4],
        decoder_transformer_layers=[4, 0, 0, 0],
        transformer_general_config=_make_transformer_config,
    )
    return codec


def bake_weight_norm(codec: nn.Module) -> int:
    """Fold every weight norm (parametrized and legacy hook) into plain weights; returns the count.

    The quantizer's legacy ``weight_norm`` recomputes ``weight`` from
    ``weight_g`` / ``weight_v`` in a forward pre-hook, so until a forward runs
    the ``weight`` attribute is stale; baking computes it from the loaded
    parameters once.
    """
    baked = 0
    for module in codec.modules():
        parametrizations = getattr(module, "parametrizations", None)
        for name in list(parametrizations.keys()) if parametrizations else []:
            remove_parametrizations(module, name, leave_parametrized=True)
            baked += 1
        for hook in [h for h in module._forward_pre_hooks.values() if isinstance(h, WeightNorm)]:
            remove_weight_norm(module, hook.name)
            baked += 1
    return baked
