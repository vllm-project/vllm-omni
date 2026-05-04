# coding=utf-8
"""
Minimal, optimized T5Gemma Encoder implementation.

This file subclasses the original Hugging Face T5GemmaEncoder and
T5GemmaEncoderModel to override the slow, general-purpose mask creation
with a direct, fast 2D -> 4D mask conversion for encoder-only inference.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers.utils.generic import check_model_inputs

# --- Key Imports from Transformers ---
# We import the original classes to subclass them
from transformers.models.t5gemma.modeling_t5gemma import (
    T5GemmaConfig,
    T5GemmaEncoder,
    T5GemmaEncoderModel,
    T5GemmaPreTrainedModel,
)

logger = logging.get_logger(__name__)

# --- Helper Functions for Fast Masking ---
# (These are the only small helpers we need)


def _make_default_2d_attention_mask(
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Construct the default 2D attention mask (all ones)."""
    return torch.ones(
        (hidden_states.shape[0], hidden_states.shape[1]),
        device=hidden_states.device,
        dtype=torch.long,
    )


def _prepare_encoder_attention_mask(
    attention_mask: torch.Tensor, hidden_states: torch.Tensor
) -> torch.Tensor:
    """
    Creates a 4D additive attention mask for an encoder from a 2D mask.
    Input: [batch_size, seq_len]
    Output: [batch_size, 1, 1, seq_len] (with -inf for padding)
    """
    extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = extended_attention_mask.to(dtype=hidden_states.dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
        hidden_states.dtype
    ).min
    return extended_attention_mask


# --- The Overridden Encoder Class ---


class OptimizedT5GemmaEncoder(T5GemmaEncoder):
    """
    Overrides the T5GemmaEncoder to replace the slow, general-purpose
    mask creation with a direct and efficient 2D -> 4D mask conversion.

    This class inherits its __init__ method from T5GemmaEncoder,
    so all layer names (embed_tokens, layers, norm, etc.) are
    guaranteed to match for weight loading.
    """

    # We inherit the __init__ method from T5GemmaEncoder.
    # No need to redefine it! It will build all the sub-modules
    # (T5GemmaSelfAttention, T5GemmaMLP, etc.) for us.

    @check_model_inputs()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs: Unpack[torch.nn.Module.forward],  # Use base class Unpack
    ) -> BaseModelOutput:

        # --- 1. Input Validation and Embedding ---
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        kwargs.pop("past_key_values", None)  # No KV cache in encoder

        if inputs_embeds is None:
            # self.embed_tokens is inherited from the parent T5GemmaEncoder
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        seq_length = hidden_states.shape[1]

        # --- 2. Position IDs and RoPE ---
        cache_position = torch.arange(seq_length, device=hidden_states.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # self.rotary_emb is inherited
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # --- 3. Optimized Mask Creation (The Core Change) ---
        if attention_mask is None:
            attention_mask = _make_default_2d_attention_mask(hidden_states)

        # This is the fast 2D -> 4D mask conversion
        # It replaces the slow `create_causal_mask` calls.
        additive_attention_mask = _prepare_encoder_attention_mask(
            attention_mask, hidden_states
        )
        layer_attention_mask = additive_attention_mask

        # --- 4. Main Encoder Layers ---
        # self.config, self.dropout, self.layers, self.norm are all inherited
        normalizer = torch.tensor(
            self.config.hidden_size**0.5, dtype=hidden_states.dtype
        )
        hidden_states = hidden_states * normalizer
        hidden_states = self.dropout(hidden_states)

        for layer_module in self.layers:
            hidden_states = layer_module(
                hidden_states,
                position_embeddings,
                layer_attention_mask,  # Pass the optimized 4D mask
                position_ids,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )


# --- The Overridden Wrapper Model Class ---


class OptimizedT5GemmaEncoderModel(T5GemmaEncoderModel):
    """
    Wrapper model that is identical to T5GemmaEncoderModel, but
    instantiates our OptimizedT5GemmaEncoder instead of the original
    T5GemmaEncoder.

    We must override __init__ to swap the encoder class.
    The `forward`, `get_input_embeddings`, etc., are inherited
    and will work correctly as they just call `self.encoder.*`.
    """

    def __init__(self, config: T5GemmaConfig):
        # We must call the __init__ of the *grandparent* class
        # (T5GemmaPreTrainedModel) to skip the original
        # T5GemmaEncoderModel.__init__ which instantiates the wrong encoder.
        super(T5GemmaEncoderModel, self).__init__(config)

        if config.is_encoder_decoder:
            raise ValueError(
                "OptimizedT5GemmaEncoderModel only supports encoder-only model."
            )

        # --- KEY CHANGE ---
        # Instantiate our optimized encoder instead of the original
        self.encoder = OptimizedT5GemmaEncoder(config.encoder)
        # ------------------

        self.post_init()

    # All other methods (forward, get_input_embeddings, set_input_embeddings)
    # are inherited from T5GemmaEncoderModel and will work perfectly.
