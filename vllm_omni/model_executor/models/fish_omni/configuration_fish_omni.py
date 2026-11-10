# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""HF config for Fish Audio Drama 3 (``model_type = "fish_omni"``).

The text backbone is a Qwen3.5 MoE (hybrid gated-delta-net + full attention),
so ``text_config`` converts to transformers' ``Qwen3_5MoeTextConfig``. The
codebook decoder keeps the checkpoint's own field names.
"""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig, Qwen3_5MoeTextConfig

# Checkpoint text_config keys that Qwen3_5MoeTextConfig has no field for. The
# backbone hard-codes what they describe (NeoX rope, q/k norm, zero-centered
# norms, gated shared expert, MoE in every layer), so they are checked, then dropped.
_FISH_ONLY_TEXT_KEYS = {
    "model_type": None,
    "mlp_layer_types": None,
    "rotary_interleaved": False,
    "use_qk_norm": True,
    "layernorm_zero_centered_gamma": True,
    "shared_expert_gate": True,
}


def to_qwen3_5_moe_text_config(text_config: dict[str, Any]) -> Qwen3_5MoeTextConfig:
    kwargs = dict(text_config)
    for key, expected in _FISH_ONLY_TEXT_KEYS.items():
        value = kwargs.pop(key, expected)
        if expected is not None and value != expected:
            raise ValueError(f"fish_omni text_config.{key}={value!r} is not supported (expected {expected!r})")
    mlp_types = text_config.get("mlp_layer_types")
    if mlp_types is not None and any(kind != "moe" for kind in mlp_types):
        raise ValueError("fish_omni text_config: every backbone layer must be MoE")
    for key in ("id2label", "label2id", "architectures", "_name_or_path", "dtype"):
        kwargs.pop(key, None)
    return Qwen3_5MoeTextConfig(**kwargs)


class FishOmniAudioDecoderConfig(PretrainedConfig):
    """Codebook tower config under the checkpoint's field names.

    Not registered with AutoConfig: its model_type ``fish_qwen3_audio_decoder``
    already names the Fish S2-Pro fast-AR config.
    """

    model_type = "fish_omni_audio_decoder"

    def __init__(
        self,
        dim: int = 2048,
        n_head: int = 16,
        n_local_heads: int = 2,
        head_dim: int = 256,
        intermediate_size: int = 6144,
        norm_eps: float = 1e-6,
        num_codebooks: int = 10,
        vocab_size: int = 4096,
        shared_enrichment_layers: int = 4,
        rope_base: float = 1e7,
        rotary_percent: float = 0.25,
        vq_backbone_codebooks: int = 1,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("model_type", None)
        self.dim = dim
        self.n_head = n_head
        self.n_local_heads = n_local_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.norm_eps = norm_eps
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.shared_enrichment_layers = shared_enrichment_layers
        self.rope_base = rope_base
        self.rotary_percent = rotary_percent
        self.rotary_dim = int(head_dim * rotary_percent)
        self.vq_backbone_codebooks = vq_backbone_codebooks
        super().__init__(**kwargs)


class FishOmniConfig(PretrainedConfig):
    """Top-level Drama 3 config: Qwen3.5 MoE backbone plus codebook tower."""

    model_type = "fish_omni"

    def __init__(
        self,
        text_config: dict | Qwen3_5MoeTextConfig | None = None,
        audio_decoder_config: dict | FishOmniAudioDecoderConfig | None = None,
        semantic_start_token_id: int = 248099,
        semantic_end_token_id: int = 252194,
        audio_pad_token_id: int = 248076,
        eos_token_id: int = 248046,
        speech_template_version: str = "reference-bracket-v1",
        **kwargs: Any,
    ) -> None:
        if isinstance(text_config, dict):
            text_config = to_qwen3_5_moe_text_config(text_config)
        if isinstance(audio_decoder_config, dict):
            audio_decoder_config = FishOmniAudioDecoderConfig(**audio_decoder_config)
        self.text_config = text_config or Qwen3_5MoeTextConfig()
        self.audio_decoder_config = audio_decoder_config or FishOmniAudioDecoderConfig()
        self.semantic_start_token_id = semantic_start_token_id
        self.semantic_end_token_id = semantic_end_token_id
        self.audio_pad_token_id = audio_pad_token_id
        self.speech_template_version = speech_template_version
        kwargs.pop("tie_word_embeddings", None)
        super().__init__(eos_token_id=eos_token_id, tie_word_embeddings=False, **kwargs)

    def get_text_config(self, **kwargs: Any) -> Qwen3_5MoeTextConfig:
        return self.text_config
