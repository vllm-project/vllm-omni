from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from transformers import LlamaConfig
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.minicpmo import MultiModalProjector
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
)
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


class MiniCPMO4_5TalkerForConditionalGeneration(nn.Module, SupportsPP):
    def _build_llama_backbone_config(self) -> LlamaConfig:
        if hasattr(self.config, "to_dict"):
            llama_config = LlamaConfig.from_dict(dict(self.config.to_dict()))
        else:
            llama_config = LlamaConfig()

        llama_config.hidden_size = int(self.config.hidden_size)
        llama_config.intermediate_size = int(self.config.intermediate_size)
        llama_config.num_attention_heads = int(self.config.num_attention_heads)
        llama_config.num_hidden_layers = int(self.config.num_hidden_layers)
        llama_config.num_key_value_heads = int(self.config.num_key_value_heads)
        llama_config.max_position_embeddings = int(self.config.max_position_embeddings)
        return llama_config

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.root_config = vllm_config.model_config.hf_config
        stage_config_name = getattr(vllm_config.model_config, "hf_config_name", None)
        stage_config = getattr(self.root_config, stage_config_name, None) if stage_config_name else None
        if stage_config_name and stage_config is None:
            logger.warning(
                "MiniCPMO4_5 talker could not find hf_config.%s; falling back to root hf_config.",
                stage_config_name,
            )
        self.config = stage_config if stage_config is not None else self.root_config

        self.num_vq = int(getattr(self.config, "num_vq", 1))
        if self.num_vq != 1:
            raise NotImplementedError(
                f"MiniCPMO4_5 native talker currently supports num_vq == 1 only, got {self.num_vq}."
            )

        self.have_multimodal_outputs = False
        self.has_preprocess = True
        self.has_postprocess = False
        self.requires_raw_input_tokens = True

        self.audio_bos_token_id = int(self.config.audio_bos_token_id)
        self.text_eos_token_id = int(self.config.text_eos_token_id)

        self.emb_text = nn.Embedding(int(self.config.num_text_tokens), int(self.config.hidden_size))
        self.projector_semantic = MultiModalProjector(int(self.config.llm_dim), int(self.config.hidden_size))
        self.projector_spk = MultiModalProjector(int(self.config.llm_dim), int(self.config.hidden_size))

        self.emb_code = nn.ModuleList(
            [nn.Embedding(int(self.config.num_audio_tokens), int(self.config.hidden_size)) for _ in range(self.num_vq)]
        )
        self.head_code = nn.ModuleList(
            [
                weight_norm(
                    nn.Linear(
                        int(self.config.hidden_size),
                        int(self.config.num_audio_tokens),
                        bias=False,
                    )
                )
                for _ in range(self.num_vq)
            ]
        )

        llama_config = self._build_llama_backbone_config()
        self.model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix="model",
            hf_config=llama_config,
            architectures=["LlamaModel"],
        )
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        self.hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_prefix={
                "tts.model.": "model.model.",
                "tts.emb_text.": "emb_text.",
                "tts.projector_semantic.": "projector_semantic.",
                "tts.projector_spk.": "projector_spk.",
                "tts.emb_code.": "emb_code.",
                "tts.head_code.": "head_code.",
            }
        )

    def embed_multimodal(self, **kwargs: Any) -> list[torch.Tensor]:
        if not kwargs:
            return []

        logger.warning(
            "MiniCPM talker received multimodal encoder inputs during profile run; "
            "returning dummy embeddings because Stage 1 does not consume them."
        )

        hidden_size = int(self.config.hidden_size)
        device = self.emb_text.weight.device
        dtype = self.emb_text.weight.dtype
        num_items = 0

        for value in kwargs.values():
            if isinstance(value, torch.Tensor):
                device = value.device
                num_items = int(value.shape[0]) if value.ndim > 0 else 1
                break
            if isinstance(value, list):
                num_items = len(value)
                if value and isinstance(value[0], torch.Tensor):
                    device = value[0].device
                break

        return [torch.zeros((1, hidden_size), device=device, dtype=dtype) for _ in range(num_items)]

    @staticmethod
    def estimate_prompt_len_from_additional_information(additional_information: dict[str, Any] | None) -> int:
        info = additional_information or {}
        llm_tokens = info.get("llm_tokens")
        if isinstance(llm_tokens, torch.Tensor):
            text_len = int(llm_tokens.numel())
        elif isinstance(llm_tokens, list):
            text_len = len(llm_tokens)
        else:
            text_len = 0

        spk_embeds = info.get("spk_embeds")
        if isinstance(spk_embeds, torch.Tensor) and spk_embeds.ndim >= 2:
            spk_len = int(spk_embeds.shape[0])
        else:
            spk_len = 0

        return spk_len + text_len + 2

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=[],
        )
        loaded = loader.load_weights(
            ((name, tensor) for name, tensor in weights if name.startswith("tts.")),
            mapper=self.hf_to_vllm_mapper,
        )
        loaded.add("model.lm_head.weight")
        logger.info("Loaded %d weights for MiniCPMO4_5TalkerForConditionalGeneration", len(loaded))
        return loaded

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        audio_ids = input_ids.to(device=self.emb_code[0].weight.device, dtype=torch.long)
        return self.emb_code[0](audio_ids)

    def _get_text_eos_embed(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.emb_text(torch.tensor([self.text_eos_token_id], device=device, dtype=torch.long)).to(dtype=dtype)

    def _get_audio_bos_embed(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.emb_text(torch.tensor([self.audio_bos_token_id], device=device, dtype=torch.long)).to(dtype=dtype)

    def prepare_condition_inputs(
        self,
        additional_information: dict[str, Any],
    ) -> torch.Tensor:
        info = additional_information or {}
        llm_tokens = info.get("llm_tokens")
        hidden_states = info.get("tts_hidden_states")

        device = self.emb_text.weight.device
        dtype = self.emb_text.weight.dtype
        hidden_size = int(self.config.hidden_size)
        llm_dim = int(self.config.llm_dim)

        if llm_tokens is None:
            tts_embeds = torch.empty((0, hidden_size), device=device, dtype=dtype)
        elif isinstance(llm_tokens, list) and len(llm_tokens) == 0:
            tts_embeds = torch.empty((0, hidden_size), device=device, dtype=dtype)
        elif hasattr(llm_tokens, "numel") and llm_tokens.numel() == 0:
            tts_embeds = torch.empty((0, hidden_size), device=device, dtype=dtype)
        else:
            llm_tokens_tensor = torch.as_tensor(llm_tokens, dtype=torch.long, device=device).reshape(-1)
            hidden_states_tensor = torch.as_tensor(hidden_states, device=device, dtype=dtype).reshape(
                llm_tokens_tensor.shape[0], llm_dim
            )
            hidden_embeds = self.projector_semantic(hidden_states_tensor)
            if bool(getattr(self.config, "normalize_projected_hidden", False)):
                hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)
            llm_embeds = self.emb_text(llm_tokens_tensor)
            tts_embeds = llm_embeds + hidden_embeds

        spk_embeds = torch.empty((0, hidden_size), device=device, dtype=dtype)
        text_eos_embed = self._get_text_eos_embed(device=device, dtype=dtype)
        audio_bos_embed = self._get_audio_bos_embed(device=device, dtype=dtype)

        pieces: list[torch.Tensor] = []
        for tensor in (spk_embeds, tts_embeds):
            if tensor.numel() > 0:
                pieces.append(tensor)
        pieces.append(text_eos_embed)
        pieces.append(audio_bos_embed)
        if not pieces:
            return torch.empty((0, hidden_size), device=device, dtype=dtype)
        return torch.cat(pieces, dim=0)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        additional_information = info_dict.get("additional_information")
        if isinstance(additional_information, dict):
            merged: dict[str, Any] = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in additional_information.items():
                merged.setdefault(k, v)
            info_dict = merged

        span_len = int(input_ids.shape[0])
        if span_len <= 0:
            if input_embeds is None:
                input_embeds = self.embed_input_ids(input_ids)
            return input_ids, input_embeds, {}

        # Prefill: build condition embeddings from additional_information.
        if span_len > 1:
            prompt_embeds = self.prepare_condition_inputs(info_dict).detach().to("cpu").contiguous()
            if not prompt_embeds.is_pinned():
                prompt_embeds = prompt_embeds.pin_memory()

            total_prompt_len = int(prompt_embeds.shape[0])
            s = max(0, min(0, total_prompt_len))
            e = max(0, min(span_len, total_prompt_len))
            take = prompt_embeds[s:e]
            if int(take.shape[0]) < span_len:
                pad_n = int(span_len - int(take.shape[0]))
                pad = (
                    take[-1:].expand(pad_n, -1)
                    if take.numel() > 0
                    else torch.zeros(
                        (pad_n, int(self.config.hidden_size)),
                        dtype=prompt_embeds.dtype,
                    )
                )
                take = torch.cat([take, pad], dim=0)

            update_dict: dict[str, Any] = {"talker_prefill_offset": int(span_len)}
            next_offset = span_len
            if next_offset < total_prompt_len:
                update_dict["talker_prompt_embeds"] = prompt_embeds

            prompt_slice = take.to(device=input_ids.device, dtype=self.emb_text.weight.dtype, non_blocking=True)
            return input_ids.clone(), prompt_slice, update_dict

        # Decode: vLLM supplies the previously sampled audio token id as input_id.
        audio_embeds = self.embed_input_ids(input_ids.reshape(-1)).reshape(span_len, -1)
        return input_ids, audio_embeds.to(device=input_ids.device, dtype=self.emb_text.weight.dtype), {}

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.head_code[0](hidden_states)
