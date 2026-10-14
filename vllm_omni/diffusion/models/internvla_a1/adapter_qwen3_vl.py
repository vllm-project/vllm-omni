from __future__ import annotations

import torch
import torch.nn as nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    ALL_ATTENTION_FUNCTIONS,
    BaseModelOutputWithPast,
    Cache,
    DynamicCache,
    FlashAttentionKwargs,
    Qwen3VLConfig,
    Qwen3VLPreTrainedModel,
    Qwen3VLTextConfig,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionModel,
    Unpack,
    apply_rotary_pos_emb,
    check_model_inputs,
    create_causal_mask,
    deprecate_kwarg,
    eager_attention_forward,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration as HFQwen3VLForConditionalGeneration,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel as HFQwen3VLModel,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextAttention as HFQwen3VLTextAttention,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextDecoderLayer as HFQwen3VLTextDecoderLayer,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextModel as HFQwen3VLTextModel,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextRMSNorm as HFQwen3VLTextRMSNorm,
)


class Qwen3VLTextRMSNorm(HFQwen3VLTextRMSNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class Qwen3VLTextAttention(HFQwen3VLTextAttention):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            if use_cache:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            else:
                key_states = torch.cat([past_key_values[self.layer_idx][0], key_states], dim=2)
                value_states = torch.cat([past_key_values[self.layer_idx][1], value_states], dim=2)

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3VLTextDecoderLayer(HFQwen3VLTextDecoderLayer):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen3VLTextAttention(config=config, layer_idx=layer_idx)
        self.input_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3VLTextModel(HFQwen3VLTextModel):
    config: Qwen3VLTextConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer"]
    _repeated_blocks = ["Qwen3VLTextDecoderLayer"]

    def __init__(self, config: Qwen3VLTextConfig):
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3VLTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Qwen3VLModel(HFQwen3VLModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    config: Qwen3VLConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]

    def __init__(self, config):
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3VLTextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()


class Qwen3VLForConditionalGeneration(HFQwen3VLForConditionalGeneration):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = ["lm_head.weight"]
    accepts_loss_kwargs = False
    config: Qwen3VLConfig

    def __init__(self, config):
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.model = Qwen3VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()


__all__ = [
    "Qwen3VLVisionModel",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLModel",
    "Qwen3VLPreTrainedModel",
    "Qwen3VLTextModel",
    "apply_rotary_pos_emb",
    "eager_attention_forward",
]
