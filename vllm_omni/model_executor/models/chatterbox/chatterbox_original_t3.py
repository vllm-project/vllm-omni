# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""English Original T3 on vLLM's Llama backbone with paired CFG sampling."""

from collections.abc import Iterable

import torch
from torch import nn
from transformers import LlamaConfig
from vllm.config import VllmConfig
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.utils import maybe_prefix
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.chatterbox.cfg_sampler import ChatterboxCFGSampler, ChatterboxMRv2CFGSampler
from vllm_omni.model_executor.models.chatterbox.chatterbox_t3 import ChatterboxT3ForConditionalGeneration
from vllm_omni.model_executor.models.chatterbox.original_heads import (
    OriginalHeads,
    original_prefill_embeds,
    original_speech_embeds,
    split_original_weights,
)

CFG_SUFFIX = "__cfg_uncond"


class ChatterboxOriginalT3(ChatterboxT3ForConditionalGeneration):
    """Keep both guidance branches in step through the atomic CFG scheduler."""

    prefer_model_sampler = True
    model_sampler_wants_input_batch = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("Chatterbox requires enable_prefix_caching=False for conditioned placeholder prompts")
        self.config = vllm_config.model_config.hf_config
        backbone = LlamaConfig(
            vocab_size=8,
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=30,
            num_attention_heads=16,
            num_key_value_heads=16,
            head_dim=64,
            max_position_embeddings=131072,
            hidden_act="silu",
            attention_bias=False,
            mlp_bias=False,
            rms_norm_eps=1e-5,
            rope_theta=500000.0,
            rope_scaling={
                "factor": 8.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
            tie_word_embeddings=False,
        )
        self.tfmr = LlamaModel(
            vllm_config=vllm_config.with_hf_config(backbone, architectures=["ChatterboxForConditionalGeneration"]),
            prefix=maybe_prefix(prefix, "tfmr"),
        )
        self.heads = OriginalHeads(self.config)
        self.make_empty_intermediate_tensors = self.tfmr.make_empty_intermediate_tensors
        self.allow_patterns_overrides = [self.config.t3_weights]
        self.cfg_pairs: dict[str, tuple[str, str, float]] = {}
        self.cfg_sampler = ChatterboxCFGSampler(Sampler(), self.cfg_pairs)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        *,
        _omni_is_prefill: bool,
        _omni_prompt_len: int,
        _omni_num_computed_tokens: int,
        ids: dict,
        embed: dict,
        req_id: str,
        global_request_id: list[str],
        chatterbox: dict,
        cfg_group: dict | None = None,
        **runner_metadata: object,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Reconstruct learned positions from scheduler progress, including recomputation."""
        role = "cond" if cfg_group is None else cfg_group["role"]
        external_id = global_request_id[0]
        pair_id = external_id.removesuffix(CFG_SUFFIX) if role == "uncond" else external_id
        self.cfg_pairs[req_id] = (pair_id, role, chatterbox["cfg_weight"])
        device, dtype = input_ids.device, self.heads.text_emb.weight.dtype
        prompt_rows = input_ids.new_empty((0, self.config.hidden_size), dtype=dtype)
        in_prompt = 0
        if _omni_is_prefill:
            prompt = original_prefill_embeds(
                self.heads,
                torch.tensor(ids["prompt"], dtype=torch.long, device=device),
                torch.tensor(ids["speech_token"], dtype=torch.long, device=device),
                embed["voice"].to(device=device, dtype=dtype),
                exaggeration=chatterbox["exaggeration"],
                unconditional=role == "uncond",
            )
            if prompt.shape[0] != _omni_prompt_len:
                raise ValueError("Original Chatterbox placeholder length does not match its conditioned prompt")
            in_prompt = min(input_ids.shape[0], _omni_prompt_len - _omni_num_computed_tokens)
            prompt_rows = prompt[_omni_num_computed_tokens : _omni_num_computed_tokens + in_prompt]
        speech_ids = input_ids[in_prompt:]
        first_position = _omni_num_computed_tokens + in_prompt - _omni_prompt_len + 1
        positions = torch.arange(first_position, first_position + speech_ids.numel(), device=device)
        speech_rows = original_speech_embeds(self.heads, speech_ids, positions)
        return input_ids, torch.cat((prompt_rows, speech_rows)), {}

    def preprocess_decode_batch(self, *, input_ids: torch.Tensor, req_infos: list[dict]) -> tuple:
        """Batch speech embeddings while retaining each request's own learned position."""
        positions = input_ids.new_tensor(
            [info["_omni_num_computed_tokens"] - info["_omni_prompt_len"] + 1 for info in req_infos]
        )
        return input_ids, original_speech_embeds(self.heads, input_ids, positions), [{} for _ in req_infos]

    def preprocess_decode_batch_mrv2(
        self, *, input_ids: torch.Tensor, input_embeds: torch.Tensor, req_infos: list[dict]
    ) -> tuple:
        ids, embeds, updates = self.preprocess_decode_batch(input_ids=input_ids, req_infos=req_infos)
        empty = embeds.new_empty((len(req_infos), 0))
        return ids, embeds, empty, empty, updates

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return raw logits; reserved-token masking follows the CFG subtraction."""
        return self.heads.speech_head(hidden_states)

    def sample(self, logits, sampling_metadata, *, input_batch):
        return self.cfg_sampler.sample(logits, sampling_metadata, input_batch)

    def mrv2_custom_sampler(self, sampler):
        return ChatterboxMRv2CFGSampler(sampler, self.cfg_pairs), None

    def on_requests_finished(self, req_ids: set[str]) -> None:
        for req_id in req_ids:
            self.cfg_pairs.pop(req_id, None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        backbone, heads = split_original_weights(weights)
        loaded = self.tfmr.load_weights(backbone)
        self.heads.load_state_dict(heads, strict=True)
        return {f"tfmr.{name}" for name in loaded} | {f"heads.{name}" for name in heads}
