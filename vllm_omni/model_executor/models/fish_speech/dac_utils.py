"""Shared DAC codec construction for Fish Speech S2 Pro.

Used by both the encoder (voice cloning, CPU) and the decoder (Stage 1, GPU).
"""

from __future__ import annotations

import torch.nn as nn

# Default DAC codec config matching s2-pro's modded_dac_vq.yaml.
DAC_SAMPLE_RATE = 44100
DAC_HOP_LENGTH = 2048  # 512 (decoder upsample) * 4 (quantizer upsample)
DAC_NUM_CODEBOOKS = 10  # 1 semantic + 9 residual


def build_dac_codec() -> nn.Module:
    """Construct a DAC codec model (uninitialized weights).

    Returns the model on CPU in eval mode -- caller is responsible for
    loading weights and moving to the target device.
    """
    from fish_speech.models.dac.modded_dac import (
        DAC,
        ModelArgs,
        WindowLimitedTransformer,
    )
    from fish_speech.models.dac.rvq import DownsampleResidualVectorQuantize

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
        n_codebooks=9,
        codebook_size=1024,
        codebook_dim=8,
        quantizer_dropout=0.0,
        downsample_factor=[2, 2],
        post_module=post_module,
        pre_module=pre_module,
        semantic_codebook_size=4096,
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
