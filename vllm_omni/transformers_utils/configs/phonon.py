# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from transformers import AutoConfig, PretrainedConfig


class PhononConfig(PretrainedConfig):
    model_type = "gradium_phonon_tts"

    def __init__(self, flow_lm=None, mimi=None, **kwargs):
        super().__init__(**kwargs)
        self.flow_lm = flow_lm or {}
        self.mimi = mimi or {}
        self.hidden_size = self.flow_lm.get("d_model", 512)
        self.num_attention_heads = self.flow_lm.get("num_heads", 8)
        self.num_key_value_heads = self.num_attention_heads
        self.num_hidden_layers = self.flow_lm.get("num_layers", 6)
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.vocab_size = self.flow_lm.get("n_bins", 4000) + 1
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 8192)
        self.architectures = ["PhononTalkerForConditionalGeneration"]


AutoConfig.register(PhononConfig.model_type, PhononConfig)
