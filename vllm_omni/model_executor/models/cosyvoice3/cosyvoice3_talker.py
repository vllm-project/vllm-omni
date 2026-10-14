# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adopted from https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/llm/llm.py
from collections.abc import Callable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen2 import Qwen2Model


class VLLMQwen2Encoder(torch.nn.Module):
    """Qwen2 encoder using vLLM's Qwen2Model with external KV cache management.

    This replaces the HuggingFace Qwen2ForCausalLM with vLLM's optimized implementation
    that uses PagedAttention and external KV cache via ForwardContext.
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.model = Qwen2Model(vllm_config=vllm_config, prefix=prefix)
        self.hidden_size = vllm_config.model_config.hf_config.hidden_size

    def forward(self, inputs_embeds: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Forward pass using vLLM's attention with external KV cache.

        Args:
            inputs_embeds: Input embeddings [total_tokens, hidden_size] or [batch, seq, hidden]
            positions: Position tensor for RoPE [total_tokens]

        Returns:
            hidden_states: Output hidden states [total_tokens, hidden_size]
        """
        # vLLM model expects flattened tensors [total_tokens, hidden_size]
        if inputs_embeds.dim() == 3:
            inputs_flat = inputs_embeds.view(-1, self.hidden_size)
        else:
            inputs_flat = inputs_embeds
        positions_flat = positions.view(-1)

        # KV cache managed externally via ForwardContext (set by GPUARModelRunner)
        # input_ids is required but ignored when inputs_embeds is provided
        hidden_states = self.model(
            input_ids=torch.zeros(inputs_flat.size(0), dtype=torch.long, device=inputs_flat.device),
            positions=positions_flat,
            intermediate_tensors=None,
            inputs_embeds=inputs_flat,
        )
        return hidden_states


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        text_encoder_input_size: int,
        llm_input_size: int,
        llm_output_size: int,
        text_token_size: int,
        speech_token_size: int,
        text_encoder: torch.nn.Module,
        llm: torch.nn.Module,
        length_normalized_loss: bool = True,
        lsm_weight: float = 0.0,
        spk_embed_dim: int = 192,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.speech_token_size = speech_token_size
        # 1. build text token inputs related modules
        self.text_embedding = torch.nn.Embedding(text_token_size, text_encoder_input_size)
        self.text_encoder = text_encoder
        self.text_encoder_affine_layer = nn.Linear(self.text_encoder.output_size(), llm_input_size)

        # 2. build speech token language model related modules
        self.sos = 0
        self.task_id = 1
        self.eos_token = self.speech_token_size
        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 1)

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size, llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, llm_input_size)


class Qwen2LM(TransformerLM):
    def __init__(
        self,
        llm_input_size: int,
        llm_output_size: int,
        speech_token_size: int,
        llm: torch.nn.Module,
        sampling: Callable,
        length_normalized_loss: bool = True,
        lsm_weight: float = 0.0,
        mix_ratio: list[int] = [5, 15],
    ):
        torch.nn.Module.__init__(self)
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size
        # 2. build speech token language model related modules
        self.sos = 0
        self.task_id = 1
        self.eos_token = speech_token_size
        self.fill_token = speech_token_size + 2

        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 3)

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size + 3, llm_input_size)

        # 4. sampling method
        self.sampling = sampling
        self.mix_ratio = mix_ratio

        # 5. vllm related
        self.stop_token_ids = [speech_token_size + i for i in range(3)]
        self.vllm_output_queue = {}


class CosyVoice3LM(Qwen2LM):
    def __init__(
        self,
        llm_input_size: int,
        llm_output_size: int,
        speech_token_size: int,
        llm: torch.nn.Module,
        length_normalized_loss: bool = True,
        lsm_weight: float = 0.0,
        mix_ratio: list[int] = [5, 15],
    ):
        torch.nn.Module.__init__(self)
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size
        # 2. build speech token language model related modules
        self.sos = speech_token_size + 0
        self.eos_token = speech_token_size + 1
        self.task_id = speech_token_size + 2
        self.fill_token = speech_token_size + 3

        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 200, bias=False)

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size + 200, llm_input_size)

        # 4. sampling method
        # self.sampling = sampling
        self.mix_ratio = mix_ratio

        # 5. vllm related
        self.stop_token_ids = [speech_token_size + i for i in range(200)]
        self.vllm_output_queue = {}
