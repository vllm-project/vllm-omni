# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native control/configuration facade for the non-Transformers Zonos3 release."""

from typing import Any

from transformers import AutoConfig, PretrainedConfig

from vllm_omni.model_executor.models.zonos3.ar_config import Zonos3Architecture


class Zonos3Config(PretrainedConfig):
    model_type = "zonos3"
    has_no_defaults_at_init = True

    def __init__(
        self,
        zonos3_params: dict[str, Any],
        zonos3_lm_subdir: str = "lm/zonos3_mini",
        zonos3_decoder: str = "keep2_plain",
        **kwargs: Any,
    ) -> None:
        architecture = Zonos3Architecture.from_params(zonos3_params)
        if zonos3_decoder not in ("keep2_plain", "full_mr"):
            raise ValueError("Unknown Zonos3 decoder variant")
        self.zonos3_params = zonos3_params
        self.zonos3_lm_subdir = zonos3_lm_subdir
        self.zonos3_decoder = zonos3_decoder
        derived = {
            "hidden_size": architecture.dim,
            "intermediate_size": architecture.ffn_hidden,
            "num_hidden_layers": architecture.n_layers,
            "num_attention_heads": architecture.n_heads,
            "num_key_value_heads": architecture.n_kv_heads,
            "head_dim": architecture.head_dim,
            "max_position_embeddings": architecture.max_seqlen,
            "rms_norm_eps": architecture.norm_eps,
            "rope_theta": architecture.rope_theta,
            "vocab_size": 3,
        }
        for name, value in derived.items():
            if kwargs.pop(name, value) != value:
                raise ValueError(f"Zonos3 {name} must be derived from the release params.json")
            setattr(self, name, value)
        kwargs.pop("codec_config", None)
        kwargs.pop("model_type", None)
        architectures = kwargs.pop("architectures", ["Zonos3ForCausalLM"])
        dtype = kwargs.pop("dtype", kwargs.pop("torch_dtype", "bfloat16"))
        for name, value in {
            "bos_token_id": 0,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "tie_word_embeddings": False,
            "is_encoder_decoder": False,
        }.items():
            if kwargs.pop(name, value) != value:
                raise ValueError(f"Zonos3 requires logical control setting {name}={value!r}")
        super().__init__(
            architectures=architectures,
            dtype=dtype,
            bos_token_id=0,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=False,
            is_encoder_decoder=False,
            **kwargs,
        )
        # This stage has no language-model KV. Benign standard dimensions let
        # native runner/task inspection handle its one placeholder per chunk.
        self.codec_config = PretrainedConfig(
            architectures=["Zonos3CodecModel"],
            hidden_size=64,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=64,
            vocab_size=3,
            max_position_embeddings=64,
            bos_token_id=0,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=False,
            dtype="float32",
        )


AutoConfig.register("zonos3", Zonos3Config)
