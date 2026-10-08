# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Configuration for Zyphra ZONOS2 (model_type="zonos2").

The official checkpoint ships a flat ``params.json`` (no ``config.json`` and no
``architectures`` field). This config keeps the official field names verbatim
(``n_layers``, ``dim``, ``moe_*`` ...) so model code maps 1:1 onto the upstream
implementation, and additionally exposes standard Transformers attribute names
(``hidden_size``, ``num_hidden_layers`` ...) for vLLM internals.

``tools/convert_zonos2_to_safetensors.py`` wraps ``params.json`` into a
``config.json`` carrying ``architectures=["Zonos2ForConditionalGeneration"]``,
which is what this class deserializes from.
"""

from __future__ import annotations

from transformers import PretrainedConfig


class Zonos2Config(PretrainedConfig):
    """ZONOS2 backbone + codec + speaker configuration (flat params.json layout)."""

    model_type = "zonos2"

    def __init__(
        self,
        # backbone
        n_layers: int = 28,
        dim: int = 2048,
        head_dim: int = 128,
        n_heads: int | None = None,
        n_kv_heads: int = 4,
        ffn_dim_multiplier: float = 1.5,
        multiple_of: int = 256,
        norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        max_seqlen: int = 6144,
        # codec token layout
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        eoa_id: int = 1024,
        audio_pad_id: int = 1025,
        text_vocab: int = 519,
        loss_softcap: float = 15.0,
        # speaker conditioning
        speaker_enabled: bool = True,
        speaker_embedding_dim: int = 2048,
        speaker_lda_dim: int = 1024,
        speaker_background_token_enabled: bool = True,
        accurate_mode_token_enabled: bool = True,
        # conditioning buckets
        speaking_rate_num_buckets: int = 8,
        speaking_rate_buckets: list[str] | None = None,
        quality_num_buckets: int = 60,
        quality_features: list[str] | None = None,
        quality_buckets: dict[str, list[str]] | None = None,
        quality_dropout: dict[str, float] | None = None,
        # sonic EDA MoE
        moe_impl: str = "sonic",
        moe_n_experts: int = 16,
        moe_router_topk: int = 1,
        special_topk_layers: dict[str | int, int] | None = None,
        moe_router_dim: int = 128,
        moe_start_from_layer: int = 3,
        moe_end_from_layer: int = 1,
        **kwargs,
    ):
        # ---- official field names, stored verbatim ----
        self.n_layers = n_layers
        self.dim = dim
        self.head_dim = head_dim
        self.n_heads = n_heads if n_heads is not None else dim // head_dim
        self.n_kv_heads = n_kv_heads
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.max_seqlen = max_seqlen

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.eoa_id = eoa_id
        self.audio_pad_id = audio_pad_id
        self.text_vocab = text_vocab
        self.loss_softcap = loss_softcap

        self.speaker_enabled = speaker_enabled
        self.speaker_embedding_dim = speaker_embedding_dim
        self.speaker_lda_dim = speaker_lda_dim
        self.speaker_background_token_enabled = speaker_background_token_enabled
        self.accurate_mode_token_enabled = accurate_mode_token_enabled

        self.speaking_rate_num_buckets = speaking_rate_num_buckets
        self.speaking_rate_buckets = speaking_rate_buckets
        self.quality_num_buckets = quality_num_buckets
        self.quality_features = quality_features
        self.quality_buckets = quality_buckets
        self.quality_dropout = quality_dropout

        self.moe_impl = moe_impl
        self.moe_n_experts = moe_n_experts
        self.moe_router_topk = moe_router_topk
        # JSON object keys are strings; normalize to int keys for runtime lookups.
        self.special_topk_layers = {int(k): int(v) for k, v in (special_topk_layers or {"26": 2}).items()}
        self.moe_router_dim = moe_router_dim
        self.moe_start_from_layer = moe_start_from_layer
        self.moe_end_from_layer = moe_end_from_layer

        # ---- derived MoE layer partition ----
        moe = set(range(moe_start_from_layer, n_layers - moe_end_from_layer))
        self.moe_layer_ids = sorted(moe)
        self.dense_layer_ids = [i for i in range(n_layers) if i not in moe]

        # ---- standard Transformers/vLLM attribute aliases ----
        self.hidden_size = dim
        self.num_hidden_layers = n_layers
        self.num_attention_heads = self.n_heads
        self.num_key_value_heads = n_kv_heads
        self.max_position_embeddings = max_seqlen
        self.rms_norm_eps = norm_eps

        # per-codebook audio vocabulary including eoa/pad
        self.codebook_vocab_size = codebook_size + 2
        # 9 audio columns + 1 text column per frame
        self.frame_width = n_codebooks + 1

        super().__init__(**kwargs)
        # One scheduler token per frame. The continue/stop sentinels and all
        # prompt text-column IDs must be within the declared lifecycle vocab.
        self.vocab_size = self.codebook_vocab_size

    def router_topk(self, layer_id: int) -> int:
        """Top-k for the MoE router at ``layer_id`` (layer 26 uses top-2)."""
        return self.special_topk_layers.get(layer_id, self.moe_router_topk)

    def is_moe_layer(self, layer_id: int) -> bool:
        return layer_id in set(self.moe_layer_ids)
