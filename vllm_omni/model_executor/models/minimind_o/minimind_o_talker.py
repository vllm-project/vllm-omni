# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# MiniMind-O Talker stage with MTP head for 8-layer Mimi codes.

from collections.abc import Iterable
from functools import cached_property
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.minimind_o.config import MiniMindOTalkerConfig

logger = init_logger(__name__)


class MiniMindOMTPHead(nn.Module):
    """MTP (Multi-Token Prediction) head for generating 8-layer Mimi codes.

    Pattern from MiniMind-O: base linear + 8 adapter layers.
    Each adapter is a low-rank bottleneck (Linear → GELU → Linear).
    """

    def __init__(self, in_features: int, out_features: int, num_layers: int = 8, rank: int = 256):
        super().__init__()
        self.num_layers = num_layers
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_features, rank, bias=False), nn.GELU(), nn.Linear(rank, out_features, bias=False)
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        """Generate logits for all 8 layers.

        Args:
            x: [batch, seq_len, in_features] hidden states

        Returns:
            List of [batch, seq_len, out_features] logits for each layer
        """
        base_out = self.base(x)
        return [base_out + adapter(x) for adapter in self.adapters]


class MiniMindOTalkerEmbedding(nn.Module):
    """Embedding with adapters for 8-layer Mimi codes.

    Pattern from MiniMind-O: base embedding + 8 adapter layers.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, num_layers: int = 8, rank: int = 256):
        super().__init__()
        self.num_layers = num_layers
        self.base = nn.Embedding(num_embeddings, embedding_dim)
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(nn.Embedding(num_embeddings, rank), nn.GELU(), nn.Linear(rank, embedding_dim, bias=False))
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        """Embed 8-layer codes and average across layers.

        Args:
            x: [batch, 8, seq_len] 8-layer code IDs

        Returns:
            [batch, seq_len, embedding_dim] averaged embeddings
        """
        base_out = self.base(x)
        # Sum base + adapter outputs for each layer, then average
        layer_outputs = []
        for i in range(len(self.adapters)):
            layer_outputs.append(base_out[:, i, :] + self.adapters[i](x[:, i, :]))
        return sum(layer_outputs) / self.num_layers


class MiniMindOTalkerForConditionalGeneration(
    nn.Module,
    SupportsPP,
):
    """
    MiniMind-O Talker stage with MTP head.

    Components:
    - thinker_to_talker_proj: Projection from thinker to talker dimension
    - language_model: LLM backbone (generates layer 0)
    - mtp_head: MTP head (generates layers 1-7)
    - codec_head: Projects to Mimi vocabulary
    - embed_tokens: Embedding for 8-layer codes
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "talker.layers.": "language_model.model.layers.",
            "talker.lm_head.": "mtp_head.",
            "talker.embed_tokens.": "embed_tokens.",
            "talker.codec_proj.": "codec_proj.",
            "talker.embed_proj.": "embed_proj.",
            "talker.spk_proj.": "spk_proj.",
            "talker.text_scale": "text_scale",
            "talker.audio_scale": "audio_scale",
            "talker.": "",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: MiniMindOTalkerConfig = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.prefix = prefix
        self.config = config

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            hf_config=config.text_config,
            architectures=["MiniMindForCausalLM"],
        )

        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

        # MTP head for 8-layer code generation
        self.mtp_head = MiniMindOMTPHead(
            config.talker_hidden_size,
            config.audio_vocab_size,
            num_layers=config.mtp_num_layers,
            rank=config.mtp_rank,
        )

        # Embedding for 8-layer codes
        self.embed_tokens = MiniMindOTalkerEmbedding(
            config.audio_vocab_size,
            config.talker_hidden_size,
            num_layers=config.mtp_num_layers,
            rank=config.mtp_rank,
        )

        # Codec projection (from MiniMind-O)
        self.codec_proj = nn.Sequential(
            nn.Linear(config.talker_hidden_size, config.talker_hidden_size),
            nn.GELU(),
            nn.Linear(config.talker_hidden_size, config.talker_hidden_size),
            RMSNorm(config.talker_hidden_size, eps=config.rms_norm_eps),
        )

        self.embed_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.talker_hidden_size),
            RMSNorm(config.talker_hidden_size, eps=config.rms_norm_eps),
        )

        # Speaker projection (from MiniMind-O)
        self.spk_proj = nn.Linear(config.spk_emb_size, config.talker_hidden_size, bias=False)

        # Learnable scales (from MiniMind-O)
        self.text_scale = nn.Parameter(torch.tensor(3.0))
        self.audio_scale = nn.Parameter(torch.tensor(1.0))

        # Special tokens
        self.audio_pad_token = config.audio_pad_token
        self.audio_stop_token = config.audio_stop_token
        self.audio_spk_token = config.audio_spk_token

        # suppress start id
        self.suppress_start_id = None

        self.talker_mtp = self
        self.talker_mtp_output_key = ("codes", "audio")
        self.mtp_temperature = 0.2
        self.mtp_top_k = 50

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler
        return Sampler()

    @staticmethod
    def _parse_stacked_input_ids(
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if input_ids is None:
            return None, None
        if input_ids.dim() == 3:
            if input_ids.shape[1] == 9:
                return input_ids[:, :8, :], input_ids[:, 8, :]
            if input_ids.shape[1] == 8:
                return input_ids, None
        return None, input_ids

    def build_fused_embeds(
        self,
        bridge_states: torch.Tensor,
        audio_ids: torch.Tensor | None = None,
        *,
        spk_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """HF: embed_proj(bridge)*text_scale + codec_proj(embed_tokens(audio))*audio_scale."""
        if bridge_states.dim() == 2:
            bridge_states = bridge_states.unsqueeze(0)
        batch_size, seq_len, _ = bridge_states.shape
        device = bridge_states.device
        dtype = bridge_states.dtype

        if audio_ids is None:
            audio_ids = torch.full(
                (batch_size, self.config.mtp_num_layers, seq_len),
                self.audio_pad_token,
                dtype=torch.long,
                device=device,
            )
        elif audio_ids.dim() == 2:
            audio_ids = audio_ids.unsqueeze(0)
        if audio_ids.size(0) == 1 and batch_size > 1:
            audio_ids = audio_ids.expand(batch_size, -1, -1)
        if audio_ids.size(2) != seq_len:
            if audio_ids.size(2) > seq_len:
                audio_ids = audio_ids[..., :seq_len]
            else:
                pad = audio_ids.new_full(
                    (audio_ids.size(0), audio_ids.size(1), seq_len - audio_ids.size(2)),
                    self.audio_pad_token,
                )
                audio_ids = torch.cat([audio_ids, pad], dim=-1)

        talker_emb = self.embed_tokens(audio_ids)
        if spk_emb is not None:
            spk_mask = (audio_ids[:, 0, :] == self.audio_spk_token).unsqueeze(-1)
            spk = self.spk_proj(spk_emb.to(device=device, dtype=dtype))
            if spk.dim() == 1:
                spk = spk.unsqueeze(0)
            talker_emb = torch.where(spk_mask, spk.unsqueeze(1), talker_emb)

        text_part = self.embed_proj(bridge_states) * self.text_scale
        audio_part = self.codec_proj(talker_emb) * self.audio_scale
        return text_part + audio_part

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Flat codec ids → base talker embedding (prefill without bridge states)."""
        if input_ids.dim() == 3:
            audio_ids, _ = self._parse_stacked_input_ids(input_ids)
            return self.embed_tokens(audio_ids)
        return self.language_model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        positions: torch.Tensor = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        assert input_ids is not None or inputs_embeds is not None, "input_ids or inputs_embeds must be provided"

        if intermediate_tensors is not None:
            inputs_embeds = None
        else:
            bridge_states = kwargs.pop("bridge_states", None)
            audio_ids = kwargs.pop("audio_ids", None)
            spk_emb = kwargs.pop("spk_emb", None)
            if audio_ids is None:
                parsed_audio, _ = self._parse_stacked_input_ids(input_ids)
                audio_ids = parsed_audio

            if bridge_states is not None or audio_ids is not None:
                if bridge_states is None:
                    bridge_states = inputs_embeds
                inputs_embeds = self.build_fused_embeds(
                    bridge_states,
                    audio_ids=audio_ids,
                    spk_emb=spk_emb if isinstance(spk_emb, torch.Tensor) else None,
                )
            elif inputs_embeds is None and input_ids is not None:
                inputs_embeds = self.embed_input_ids(input_ids)

        input_ids = None

        hidden_states = self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )
        return hidden_states

    def compute_mtp_logits(self, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        """Compute logits for all 8 layers using MTP head."""
        return self.mtp_head(hidden_states)

    def bad_word_processor(self, logits: torch.Tensor) -> torch.Tensor:
        """Suppress unsupported token IDs."""
        if self.suppress_start_id and self.suppress_start_id < logits.size(-1):
            end_id = int(getattr(self.config, "tts_codec_end_token_id", self.audio_stop_token))
            if self.suppress_start_id == end_id:
                logits[..., end_id + 1 : logits.size(-1)] = -1e9
            elif self.suppress_start_id < end_id:
                logits[..., self.suppress_start_id : end_id] = -1e9
                logits[..., end_id + 1 : logits.size(-1)] = -1e9
            else:
                logits[..., self.suppress_start_id : logits.size(-1)] = -1e9

        if hasattr(self.config, "tts_codec_start_token_id"):
            bos_id = int(getattr(self.config, "tts_codec_start_token_id", self.audio_pad_token))
            logits[..., bos_id] = -1e9
        return logits

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        # Layer-0 logits from MTP head (matches talker.lm_head.base + adapters[0])
        logits = self.mtp_head(hidden_states)[0]
        logits = self.bad_word_processor(logits)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        return self.language_model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["thinker.", "code2wav."],
        )
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

        # Log load summary
        try:
            total_bytes = 0
            for name, param in self.named_parameters():
                if param is not None and param.data is not None:
                    total_bytes += param.data.numel() * param.data.element_size()
            device = next(self.parameters()).device
            logger.info(
                "[Model Loaded] name=%s, success=%s, size=%.2f MB, device=%s",
                self.__class__.__name__,
                True,
                total_bytes / (1024**2),
                str(device),
            )
        except Exception:
            pass
        return loaded

    def set_suppress_start_id(self, start_id: int):
        self.suppress_start_id = start_id
        logger.debug(f"Set suppress start id to {self.suppress_start_id}")

    def talker_postprocess(self, hidden_states: torch.Tensor, **info_dict: object) -> dict:
        return {"hidden_states": {"last": hidden_states[-1].detach()}}

    @staticmethod
    def _sample_mimi_layer_logits(
        logits: torch.Tensor,
        prev_codes: list[int],
        *,
        generator: torch.Generator | None = None,
        temperature: float = 0.2,
        top_k: int = 50,
    ) -> int:
        logits = logits.clone() / max(temperature, 1e-9)
        for code in prev_codes[-3:]:
            if 0 <= code < logits.numel():
                logits[code] /= 1.05
        top_val, top_idx = torch.topk(logits, top_k)
        pick = torch.multinomial(F.softmax(top_val, dim=-1), 1, generator=generator).item()
        return int(top_idx[pick].item())

    @torch.inference_mode()
    def talker_mtp(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        text_step: torch.Tensor,
        *,
        audio_step: int | None = None,
        audio_ids: torch.Tensor | None = None,
        audio_code_history: torch.Tensor | None = None,
        audio_steps: torch.Tensor | None = None,
        audio_ids_list: list | None = None,
        generator: torch.Generator | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GPU fast-path: sample 8-layer Mimi codes (HF stream_generate delay pattern)."""
        del kwargs
        bsz = int(input_ids.reshape(-1).shape[0])
        device = input_embeds.device
        dtype = input_embeds.dtype
        hidden = last_talker_hidden.reshape(bsz, 1, -1)
        text_step = text_step.reshape(bsz, 1, -1).to(device=device, dtype=dtype)
        logits_layers = self.mtp_head(hidden)

        temp = self.mtp_temperature if temperature is None else float(temperature)
        k = self.mtp_top_k if top_k is None else int(top_k)

        codes = torch.full((bsz, self.config.mtp_num_layers), self.audio_pad_token, dtype=torch.long, device=device)
        for b in range(bsz):
            step_i = int(audio_steps[b].item()) if audio_steps is not None and b < audio_steps.numel() else (
                int(audio_step) if audio_step is not None else -1
            )
            row_audio = audio_ids
            if audio_ids_list is not None and b < len(audio_ids_list):
                row_audio = audio_ids_list[b]
            if row_audio is not None and row_audio.dim() == 2:
                audio_code_history = row_audio

            for layer_idx, layer_logits in enumerate(logits_layers):
                if step_i < layer_idx:
                    continue
                prev: list[int] = []
                if audio_code_history is not None and audio_code_history.ndim == 2:
                    hist = audio_code_history[layer_idx]
                    prev = [int(x) for x in hist.tolist() if int(x) != self.audio_pad_token][-3:]
                codes[b, layer_idx] = self._sample_mimi_layer_logits(
                    layer_logits[b, 0, :],
                    prev,
                    generator=generator,
                    temperature=temp,
                    top_k=k,
                )

            if row_audio is None:
                row_audio = codes[b : b + 1].unsqueeze(-1)
            else:
                if row_audio.dim() == 2:
                    row_audio = row_audio.unsqueeze(0)
                pad_col = row_audio.new_full((row_audio.size(0), row_audio.size(1), 1), self.audio_pad_token)
                row_audio = torch.cat([row_audio, pad_col], dim=-1)
                row_audio[0, :, -1] = codes[b]

            if b == 0:
                audio_ids = row_audio
            else:
                audio_ids = torch.cat([audio_ids, row_audio], dim=0)

        bridge = text_step
        fused = self.build_fused_embeds(bridge, audio_ids=audio_ids)
        return fused.reshape(bsz, -1), codes

