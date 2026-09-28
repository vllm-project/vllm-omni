# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Minimal HF metadata for Spark's non-HF BiCodec checkpoint."""

from transformers import PretrainedConfig


class SparkBiCodecConfig(PretrainedConfig):
    model_type = "spark_bicodec"

    def __init__(self, **kwargs):
        kwargs.setdefault("architectures", ["SparkBiCodecDecoder"])
        kwargs.setdefault("hidden_size", 1)
        kwargs.setdefault("num_attention_heads", 1)
        kwargs.setdefault("num_hidden_layers", 1)
        kwargs.setdefault("vocab_size", 8192)
        kwargs.setdefault("max_position_embeddings", 8192)
        super().__init__(**kwargs)
        self.sample_rate = 16000
