# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from transformers import AutoConfig, PretrainedConfig


class PhononConfig(PretrainedConfig):
    model_type = "gradium_phonon_tts"

    def __init__(self, flow_lm=None, mimi=None, **kwargs):
        eos = kwargs.get("eos_token_id")
        if isinstance(eos, list):
            if len(eos) != 1:
                raise ValueError("Phonon requires one EOS protocol token")
            eos = eos[0]
        kwargs["eos_token_id"] = 1 if eos is None else eos
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
        self.audio_continue_token_id = kwargs.get("audio_continue_token_id", 0)
        if not (0 <= self.eos_token_id < self.vocab_size):
            raise ValueError("Phonon EOS token must be inside the vocabulary")
        if (
            not (0 <= self.audio_continue_token_id < self.vocab_size)
            or self.audio_continue_token_id == self.eos_token_id
        ):
            raise ValueError("Phonon continue token must be valid and distinct from EOS")
        self.architectures = ["PhononTalkerForConditionalGeneration"]


AutoConfig.register(PhononConfig.model_type, PhononConfig)
