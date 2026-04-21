# SPDX-License-Identifier: Apache-2.0
"""Fun-Audio-Chat CRQ decoder (`audio_invert_tower`).

Line-by-line port of the reference
    github.com/FunAudioLLM/Fun-Audio-Chat
    funaudiochat/modeling_funaudiochat.py L563-768
(FunAudioChatDecoder: pre_matching, input_matching, output_matching,
lm_head, crq_transformer, get_embeddings, sampling_step,
crq_generate_forward, forward).

Deviation from reference: generation state (past_kv, audio_embeds,
speech_ids, logits_processor, do_sample) is threaded through a
`CRQState` dataclass rather than stashed on `self`. This lets
vllm-omni's `model_intermediate_buffer[req_id]` own the state per
request and makes BS>=1 concurrency safe.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import CausalLMOutput

from vllm_omni.transformers_utils.configs.fun_audio_chat import (
    FunAudioChatAudioEncoderConfig,
)

__all__ = ["CRQState", "FunAudioChatDecoder"]


@dataclass
class CRQState:
    """Per-request CRQ generation state.

    Everything the reference stashes on `audio_invert_tower.*` lives here.
    Mutable: pass in, return updated copy (or mutate if owned by the caller).
    """

    # past_key_values for the CRQ transformer's KV cache; `None` at first step.
    past_key_values: Any = None
    # Last sampled audio-token embedding(s); `None` at first step (ref seeds
    # with embed(bos_token_id)).
    audio_embeds: torch.Tensor | None = None
    # All speech tokens accumulated so far for this request (1D long tensor).
    # Used by the logits processor (repetition penalty, etc).
    speech_ids: torch.Tensor = field(
        default_factory=lambda: torch.empty(1, 0, dtype=torch.long)
    )
    # LogitsProcessorList-compatible callable; may be None for pure greedy.
    logits_processor: Any = None
    # True => torch.multinomial sampling; False => argmax.
    do_sample: bool = False
    # Tokens produced on the most recent forward pass: LongTensor [1, group_size].
    generate_tokens: torch.Tensor = field(
        default_factory=lambda: torch.empty(1, 0, dtype=torch.long)
    )


class FunAudioChatDecoder(nn.Module):
    """CRQ decoder. Ref L563-768.

    Weight layout (checkpoint):
        audio_invert_tower.pre_matching
        audio_invert_tower.input_matching
        audio_invert_tower.output_matching
        audio_invert_tower.lm_head
        audio_invert_tower.crq_transformer.{layers, norm, ...}  # no embed_tokens
    """

    main_input_name = "audio_ids"

    def __init__(self, config: FunAudioChatAudioEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.group_size = config.group_size
        self.hidden_size = config.output_dim  # outer dim; input to pre_matching
        self.pre_matching = nn.Linear(
            self.hidden_size, self.hidden_size * self.group_size, bias=True
        )
        # Build crq_transformer (Qwen3-like) from config.crq_transformer_config.
        crq_cfg_input = config.crq_transformer_config
        if isinstance(crq_cfg_input, dict):
            crq_transformer_config = AutoConfig.for_model(**crq_cfg_input)
        else:
            crq_transformer_config = crq_cfg_input
        self.crq_transformer = AutoModel.from_config(crq_transformer_config)
        # Ref L584: delete the CRQ transformer's own input embedding; we feed
        # inputs_embeds directly.
        del self.crq_transformer.embed_tokens

        self.input_matching = nn.Linear(
            self.hidden_size, crq_transformer_config.hidden_size, bias=False
        )
        self.output_matching = nn.Linear(
            crq_transformer_config.hidden_size, self.hidden_size, bias=False
        )
        self.lm_head = nn.Linear(config.output_dim, config.codebook_size, bias=False)

    # Ref L594-595.
    def get_embeddings(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        return self.lm_head.weight.data[audio_tokens]

    # Ref L597-613.
    def _sampling_step(
        self,
        logits: torch.Tensor,
        state: CRQState,
        accumulated_step_tokens: list[torch.Tensor],
    ) -> torch.Tensor:
        next_token_logits = logits[:, -1, :].to(
            copy=True, dtype=torch.float32, device=logits.device
        )
        if state.logits_processor is not None:
            # Ref builds input to the processor from crq_speech_ids + already-
            # emitted step tokens concatenated, matching the training-time
            # convention so repetition penalties cover the whole stream.
            history = torch.cat([state.speech_ids.to(next_token_logits.device),
                                 *accumulated_step_tokens], dim=-1)
            next_token_scores = state.logits_processor(history, next_token_logits)
        else:
            next_token_scores = next_token_logits
        if state.do_sample:
            probs = torch.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)
        return next_tokens

    # Ref L615-683.
    def crq_generate_forward(
        self,
        inputs_embeds: torch.Tensor,   # [B, slen, H]
        state: CRQState,
    ) -> tuple[torch.Tensor, CRQState]:
        """Generate `group_size` CRQ tokens conditioned on `inputs_embeds`.

        Returns
        -------
        generate_tokens : LongTensor [B, group_size]
        new_state       : CRQState with updated past_key_values, audio_embeds,
                          and `generate_tokens` populated for convenience.
        """
        x = self.pre_matching(inputs_embeds)
        bs, slen, _ = x.shape
        hidden_states = x.reshape(bs, slen * self.group_size, -1)

        # Seed audio_embeds with BOS-token embedding at i==0; otherwise reuse
        # the last-step audio embed (unsqueezed to be broadcastable).
        if state.audio_embeds is None:
            bos_embed = (
                self.get_embeddings(torch.tensor(self.config.bos_token_id,
                                                 device=hidden_states.device))
                .to(dtype=hidden_states.dtype, device=hidden_states.device)
                .view(1, 1, -1)
                .repeat(bs, 1, 1)
            )
            audio_embeds = bos_embed
        else:
            audio_embeds = state.audio_embeds.unsqueeze(1)

        past_key_values = state.past_key_values
        step_tokens: list[torch.Tensor] = []
        all_logits: list[torch.Tensor] = []

        for i in range(self.group_size):
            if i == 0:
                # Ref L642-644: first sub-step sees the whole group_size-1
                # prefix of hidden_states plus the audio_embed seed.
                input_embeds = (
                    hidden_states[:, : slen * self.group_size - (self.group_size - i - 1)]
                    + audio_embeds
                )
            else:
                input_embeds = (
                    hidden_states[:, slen * self.group_size - (self.group_size - i)]
                    + audio_embeds
                ).unsqueeze(1)
            input_embeds = self.input_matching(input_embeds)
            outputs = self.crq_transformer(
                inputs_embeds=input_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            lh = self.output_matching(outputs.last_hidden_state)
            logits = self.lm_head(lh)
            next_tokens = self._sampling_step(logits, state, step_tokens)
            step_tokens.append(next_tokens.unsqueeze(1))
            if i == 0:
                all_logits.append(logits)
            else:
                all_logits.append(logits[:, -1, :].unsqueeze(1))
            audio_embeds = self.get_embeddings(next_tokens)  # ref stashes as 2D

        generate_tokens = torch.cat(step_tokens, dim=1)  # [B, group_size]
        new_state = replace(
            state,
            past_key_values=past_key_values,
            audio_embeds=audio_embeds,  # [B, H] (2D per ref)
            generate_tokens=generate_tokens,
        )
        return generate_tokens, new_state

    # Ref L685-768. Teacher-forcing forward used during training; kept for weight
    # loading and potential future fine-tune paths.
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        audio_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> CausalLMOutput | tuple:
        x = self.pre_matching(inputs_embeds)
        bs, slen, _ = x.shape
        hidden_states = x.reshape(bs, slen * self.group_size, -1)
        if audio_embeds is not None:
            audio_embeds = audio_embeds.view(bs, slen * self.group_size, -1)
            audio_embeds = torch.roll(audio_embeds, shifts=-(self.group_size - 1), dims=1)
            my_inputs_embeds = hidden_states + audio_embeds
        else:
            my_inputs_embeds = hidden_states
        my_inputs_embeds = self.input_matching(my_inputs_embeds)

        crq_attention_mask = None
        crq_position_ids = None
        if attention_mask is not None:
            assert attention_mask.dim() == 2
            crq_attention_mask = attention_mask.repeat_interleave(self.group_size, dim=1)
        if position_ids is not None:
            base = position_ids * self.group_size
            crq_position_ids = base.unsqueeze(-1).repeat(1, 1, self.group_size)
            offsets = torch.arange(self.group_size, device=position_ids.device,
                                   dtype=position_ids.dtype)
            crq_position_ids = crq_position_ids + offsets.view(1, 1, -1)
            crq_position_ids = crq_position_ids.view(bs, -1)
            # Ref L729-740: derive padding mask from position_ids (rightmost
            # non-zero position) rather than attention_mask.
            nonzero_mask = position_ids != 0
            reversed_nonzero = torch.flip(nonzero_mask, dims=[-1])
            first_nonzero_from_end = reversed_nonzero.to(torch.long).argmax(dim=-1)
            has_nonzero = nonzero_mask.any(dim=-1)
            orig_slen = position_ids.shape[1]
            ending_pos = torch.where(
                has_nonzero,
                orig_slen - first_nonzero_from_end,
                torch.zeros_like(first_nonzero_from_end),
            )
            seq_indices = torch.arange(orig_slen, device=position_ids.device).unsqueeze(0)
            padding_mask = seq_indices >= ending_pos.unsqueeze(-1)
            crq_padding_mask = padding_mask.repeat_interleave(self.group_size, dim=1)
            crq_position_ids = crq_position_ids.masked_fill(crq_padding_mask, 0)

        outputs = self.crq_transformer(
            inputs_embeds=my_inputs_embeds,
            attention_mask=crq_attention_mask,
            position_ids=crq_position_ids,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = self.output_matching(outputs.last_hidden_state)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            labels = nn.functional.pad(labels, (0, self.group_size), value=-100)
            shift_labels = labels[..., self.group_size:].contiguous()
            # Use a simple CE loss; matches reference behavior for shifted labels.
            from torch.nn.functional import cross_entropy
            loss = cross_entropy(
                logits.reshape(-1, self.config.codebook_size),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
        if not return_dict:
            out = (logits,)
            return (loss,) + out if loss is not None else out
        return CausalLMOutput(loss=loss, logits=logits)
