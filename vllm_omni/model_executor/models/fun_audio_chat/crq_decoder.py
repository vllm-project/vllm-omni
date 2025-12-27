# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CRQ Transformer Decoder for Fun-Audio-Chat speech synthesis.

The CRQ (Codec Residual Quantization) Decoder is the second stage in the
Fun-Audio-Chat S2S pipeline. It takes LLM hidden states and generates
discrete speech tokens at 25Hz that can be converted to audio by CosyVoice.

Architecture:
- Pre-matching: Linear projection to expand hidden states by group_size (5x)
- CRQ Transformer: Qwen3-based transformer for autoregressive token generation
- LM Head: Projects to codebook vocabulary for token prediction

Reference: https://huggingface.co/FunAudioLLM/Fun-Audio-Chat-8B
"""

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


class FunAudioChatCRQDecoder(nn.Module, SupportsPP):
    """
    CRQ Transformer Decoder for speech token generation.

    This implements the autoregressive decoder that converts LLM hidden states
    into discrete speech tokens. The decoder uses a Qwen3-based transformer
    with interleaved token generation across group_size (5) positions.

    Input: LLM hidden states [batch, seq_len, hidden_size]
    Output: Speech tokens [batch, seq_len * group_size] with values in [0, codebook_size)
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.audio_config = self.config.audio_config

        # Get decoder config from audio_config
        self.group_size = getattr(self.audio_config, "group_size", 5)
        self.hidden_size = getattr(self.audio_config, "output_dim", 4096)
        self.codebook_size = getattr(self.audio_config, "codebook_size", 6565)
        self.bos_token_id = getattr(self.audio_config, "bos_token_id", 6561)
        self.eos_token_id = getattr(self.audio_config, "eos_token_id", 6562)
        self.pad_token_id = getattr(self.audio_config, "pad_token_id", 6564)

        logger.info(
            f"Initializing CRQ Decoder: group_size={self.group_size}, "
            f"hidden_size={self.hidden_size}, codebook_size={self.codebook_size}"
        )

        # Pre-matching: Expand hidden states by group_size
        self.pre_matching = nn.Linear(self.hidden_size, self.hidden_size * self.group_size, bias=True)

        # CRQ Transformer: Qwen3-based decoder
        crq_config_dict = getattr(self.audio_config, "crq_transformer_config", None)
        if crq_config_dict is None:
            # Default CRQ config based on Fun-Audio-Chat
            crq_config_dict = {
                "model_type": "qwen3",
                "hidden_size": 1024,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "intermediate_size": 3072,
                "vocab_size": self.codebook_size,
                "max_position_embeddings": 32768,
                "head_dim": 64,
                "rope_theta": 1000000.0,
            }

        self.crq_config = AutoConfig.for_model(**crq_config_dict)
        self.crq_transformer = AutoModel.from_config(self.crq_config)

        # Remove embedding layer (we use our own embeddings)
        if hasattr(self.crq_transformer, "embed_tokens"):
            del self.crq_transformer.embed_tokens

        crq_hidden_size = self.crq_config.hidden_size

        # Input/Output matching layers
        self.input_matching = nn.Linear(self.hidden_size, crq_hidden_size, bias=False)
        self.output_matching = nn.Linear(crq_hidden_size, self.hidden_size, bias=False)

        # LM head for codebook prediction
        self.lm_head = nn.Linear(self.hidden_size, self.codebook_size, bias=False)

        # For empty intermediate tensors
        self.make_empty_intermediate_tensors = lambda: None

    def get_embeddings(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        """Get embeddings for audio tokens from LM head weights."""
        return self.lm_head.weight.data[audio_tokens]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        additional_information: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """
        Forward pass for CRQ decoder.

        Args:
            input_ids: Not used directly (placeholder for vLLM interface)
            positions: Position IDs
            additional_information: Contains:
                - thinker_hidden_states: [batch, seq_len, hidden_size] from Stage 0
                - text_embeds: [batch, seq_len, hidden_size] from Stage 0

        Returns:
            OmniOutput with speech_tokens in multimodal_outputs
        """
        if additional_information is None:
            logger.warning("CRQ Decoder: No additional_information provided")
            return OmniOutput(
                text_hidden_states=torch.zeros(1, self.hidden_size),
                multimodal_outputs={"speech_tokens": None},
            )

        # Extract hidden states from Stage 0
        thinker_hidden_states = additional_information.get("thinker_hidden_states")
        text_embeds = additional_information.get("text_embeds")

        if thinker_hidden_states is None:
            logger.warning("CRQ Decoder: No thinker_hidden_states provided")
            return OmniOutput(
                text_hidden_states=torch.zeros(1, self.hidden_size),
                multimodal_outputs={"speech_tokens": None},
            )

        device = thinker_hidden_states.device
        dtype = thinker_hidden_states.dtype

        # Combine hidden states with text embeddings (following official impl)
        if text_embeds is not None:
            speech_inputs_embeds = thinker_hidden_states + text_embeds.detach().to(device, dtype)
        else:
            speech_inputs_embeds = thinker_hidden_states

        # Generate speech tokens autoregressively
        speech_tokens = self._generate_speech_tokens(speech_inputs_embeds)

        return OmniOutput(
            text_hidden_states=thinker_hidden_states.reshape(-1, thinker_hidden_states.shape[-1]),
            multimodal_outputs={"speech_tokens": speech_tokens},
        )

    def _generate_speech_tokens(
        self,
        inputs_embeds: torch.Tensor,
        max_steps: int | None = None,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Generate speech tokens autoregressively using CRQ decoder.

        Args:
            inputs_embeds: LLM hidden states [batch, seq_len, hidden_size]
            max_steps: Maximum generation steps (defaults to seq_len)
            temperature: Sampling temperature
            do_sample: Whether to sample or use greedy decoding

        Returns:
            Speech tokens [batch, seq_len * group_size]
        """
        batch_size, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        if max_steps is None:
            max_steps = seq_len

        # Pre-matching: expand to group_size
        expanded = self.pre_matching(inputs_embeds)  # [batch, seq_len, hidden_size * group_size]
        hidden_states = expanded.reshape(batch_size, seq_len * self.group_size, -1)

        all_tokens = []
        past_key_values = None

        # Initialize with BOS embedding
        audio_embeds = (
            self.get_embeddings(torch.tensor([self.bos_token_id], device=device))
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
            .to(dtype=dtype)
        )

        for step in range(max_steps):
            step_tokens = []

            for i in range(self.group_size):
                # Compute input for this position
                if i == 0:
                    # First position: use full context up to this step
                    input_embeds = (
                        hidden_states[:, : (step + 1) * self.group_size - (self.group_size - 1)] + audio_embeds
                    )
                else:
                    # Subsequent positions: just the current position
                    pos_idx = step * self.group_size + i
                    input_embeds = hidden_states[:, pos_idx : pos_idx + 1] + audio_embeds.unsqueeze(1)

                # Project to CRQ transformer dimension
                input_embeds = self.input_matching(input_embeds)

                # Forward through CRQ transformer
                outputs = self.crq_transformer(
                    inputs_embeds=input_embeds,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values

                # Project back and compute logits
                lhidden_states = self.output_matching(outputs.last_hidden_state)
                logits = self.lm_head(lhidden_states)

                # Sample next token
                next_token_logits = logits[:, -1, :].float()

                if do_sample and temperature > 0:
                    next_token_logits = next_token_logits / temperature
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)

                step_tokens.append(next_tokens)

                # Update audio embeddings for next position
                audio_embeds = self.get_embeddings(next_tokens).to(dtype=dtype)

            # Concatenate tokens for this step
            all_tokens.append(torch.stack(step_tokens, dim=1))  # [batch, group_size]

            # Check for EOS
            step_tokens_tensor = torch.stack(step_tokens, dim=1)
            if (step_tokens_tensor == self.eos_token_id).any():
                break

        # Concatenate all tokens
        if all_tokens:
            speech_tokens = torch.cat(all_tokens, dim=1)  # [batch, num_steps * group_size]
        else:
            speech_tokens = torch.empty(batch_size, 0, dtype=torch.long, device=device)

        return speech_tokens

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load CRQ decoder weights."""
        loaded_weights: set[str] = set()

        # Convert to state_dict format
        weights_dict = {}
        for name, weight in weights:
            # Handle prefix from full model
            if name.startswith("audio_invert_tower."):
                param_name = name.replace("audio_invert_tower.", "")
                weights_dict[param_name] = weight

        # Load into current model
        state_dict = self.state_dict()
        for name, weight in weights_dict.items():
            if name in state_dict:
                if state_dict[name].shape == weight.shape:
                    state_dict[name].copy_(weight)
                    loaded_weights.add(f"audio_invert_tower.{name}")
                else:
                    logger.warning(f"Shape mismatch for {name}: expected {state_dict[name].shape}, got {weight.shape}")
            else:
                logger.debug(f"Skipping weight: {name}")

        logger.info(f"Loaded {len(loaded_weights)} CRQ decoder weights")
        return loaded_weights


__all__ = ["FunAudioChatCRQDecoder"]
