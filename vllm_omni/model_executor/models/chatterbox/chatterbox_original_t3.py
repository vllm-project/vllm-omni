# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""English Original T3 on vLLM's Llama backbone with paired CFG sampling."""

from collections.abc import Iterable

import torch
from transformers import LlamaConfig, PretrainedConfig
from vllm.config import VllmConfig
from vllm.model_executor.models.llama import LlamaModel
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.chatterbox.cfg_sampler import ChatterboxCFGSampler, ChatterboxMRv2CFGSampler
from vllm_omni.model_executor.models.chatterbox.chatterbox_t3 import ChatterboxT3ForConditionalGeneration
from vllm_omni.model_executor.models.chatterbox.original_heads import (
    ORIGINAL_SPEECH_POSITIONS,
    OriginalHeads,
    original_prefill_embeds,
    original_speech_embeds,
    split_original_weights,
)
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig


class ChatterboxOriginalT3(ChatterboxT3ForConditionalGeneration):
    """Keep both guidance branches in step through the atomic CFG scheduler."""

    prefer_model_sampler = True
    model_sampler_wants_extra_args = True

    backbone_class = LlamaModel
    heads_class = OriginalHeads

    @staticmethod
    def backbone_config(config: ChatterboxConfig) -> PretrainedConfig:
        """Upstream's ``LLAMA_520M_CONFIG_DICT``: the checkpoint's own eight-row embedding table
        (the heads embed every token) and llama3 RoPE."""
        return LlamaConfig(
            vocab_size=8,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            hidden_act=config.activation_function,
            attention_bias=False,
            mlp_bias=False,
            rms_norm_eps=config.layer_norm_epsilon,
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

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # A request's speech positions stay below its context, so a context
        # within the checkpoint's speech position table bounds every request.
        if vllm_config.model_config.max_model_len > ORIGINAL_SPEECH_POSITIONS:
            raise ValueError(f"Chatterbox Original supports max_model_len up to {ORIGINAL_SPEECH_POSITIONS}")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
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
        chatterbox: dict,
        cfg_group: dict | None = None,
        sampling_params: SamplingParams | None = None,
        req_id: str | None = None,
        **runner_metadata: object,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Reconstruct learned positions from scheduler progress, including recomputation.

        Model Runner V2 samples by row identity, so it registers here the
        guidance pair that ``prepare_original_prompt`` and
        ``expand_original_cfg_prompts`` put in the request's
        ``extra_args``; the default runner hands the sampler that same tuple
        itself and passes no sampling parameters.
        """
        if sampling_params is not None:
            self.cfg_pairs[req_id] = sampling_params.extra_args["chatterbox_cfg"]
        role = "cond" if cfg_group is None else cfg_group["role"]
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

    def sample(self, logits, sampling_metadata, *, per_req_extra_args):
        return self.cfg_sampler.sample(logits, sampling_metadata, per_req_extra_args)

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
