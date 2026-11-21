# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright 2026 NetEase Youdao
"""Confucius4-TTS Stage 0: vLLM-native text-to-semantic talker."""

from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import GPT2Config
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.gpt2 import GPT2Block
from vllm.model_executor.models.utils import (
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from vllm_omni.data_entry_keys import OmniPayload
from vllm_omni.model_executor.models.indextts2.preprocess_utils import (
    resolve_model_file,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.utils.speaker_cache import get_speaker_cache

from .configuration_confucius4_tts import Confucius4TTSConfig
from .position_embeddings import LearnedPositionalEmbedding
from .preprocessing import extract_reference_features, load_wav2vec_stats
from .prompt_utils import tokenize_confucius4_prompt
from .speaker_encoder import build_speaker_encoder
from .text_encoder import TextEmbeddingProjector

logger = init_logger(__name__)

_STAGE1_CONDITIONING_KEYS = ("text_token_ids", "condition_vector", "ref_mel", "style")


def _gpt2_block_factory(
    config: GPT2Config,
    cache_config: Any,
    quant_config: Any,
):
    return partial(GPT2Block, config, cache_config, quant_config)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Confucius4TTSTalkerForConditionalGeneration(nn.Module):
    # forward() returns the hidden-state tensor; the multimodal payload is built
    # afterwards in make_omni_output. Declaring this keeps FULL CUDA graphs
    # available to MRv2 instead of dropping to PIECEWISE.
    mrv2_forward_returns_tensor = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        self.config: Confucius4TTSConfig = vllm_config.model_config.hf_config  # type: ignore[assignment]

        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = False
        self.requires_raw_input_tokens = True
        self.enable_update_additional_information = True
        self.omni_pooler_payload_include_hidden = False
        self.omni_payload_at_request_end = True
        self.omni_request_end_token_ids = ()
        # Stage 1 rebuilds the latent from the finished token sequence, so
        # Stage 0 only tracks decode counters on the GPU.
        self.gpu_resident_buffer_keys = {
            ("meta", "semantic_start_offset"),
            ("meta", "semantic_code_count"),
        }

        cfg = self.config
        self.text_projector = TextEmbeddingProjector(
            cfg.text_vocab_size,
            cfg.text_embedding_dim,
            cfg.hidden_size,
        )
        self.semantic_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.text_position_embedding = LearnedPositionalEmbedding(
            cfg.max_text_tokens,
            cfg.hidden_size,
        )
        self.semantic_position_embedding = LearnedPositionalEmbedding(
            cfg.max_semantic_tokens,
            cfg.hidden_size,
        )
        self.speaker_encoder = build_speaker_encoder(cfg)

        gpt_config = GPT2Config(
            vocab_size=cfg.vocab_size,
            n_positions=cfg.max_text_tokens + cfg.max_semantic_tokens + 1,
            n_ctx=cfg.max_text_tokens + cfg.max_semantic_tokens + 1,
            n_embd=cfg.hidden_size,
            n_layer=cfg.num_layers,
            n_head=cfg.num_attention_heads,
            use_cache=True,
        )
        self.transformer = nn.Module()
        self.start_layer, self.end_layer, self.transformer.h = make_layers(
            cfg.num_layers,
            _gpt2_block_factory(
                gpt_config,
                vllm_config.cache_config,
                vllm_config.quant_config,
            ),
            prefix=f"{prefix}.transformer.h",
        )
        self.transformer.ln_f = nn.LayerNorm(
            cfg.hidden_size,
            eps=gpt_config.layer_norm_epsilon,
        )
        self.final_norm = nn.LayerNorm(cfg.hidden_size)
        if get_pp_group().is_last_rank:
            self.semantic_head = ParallelLMHead(
                cfg.vocab_size,
                cfg.hidden_size,
                bias=True,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "semantic_head"),
            )
        else:
            self.semantic_head = None  # type: ignore[assignment]
        self.logits_processor = LogitsProcessor(cfg.vocab_size)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"],
            cfg.hidden_size,
        )
        self._w2v_stats: torch.Tensor | None = None
        self._speaker_cache = get_speaker_cache()

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.semantic_embedding(input_ids)

    def _semantic_offsets(
        self,
        positions: torch.Tensor,
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        infos = kwargs.get("model_intermediate_buffer") or []
        if not infos:
            return torch.zeros_like(positions)
        offsets = []
        for info in infos:
            meta = info.get("meta", {}) if isinstance(info, dict) else {}
            offsets.append(int(meta.get("semantic_start_offset", 0)))
        return torch.tensor(offsets, dtype=positions.dtype, device=positions.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                semantic_pos = torch.clamp(
                    positions - self._semantic_offsets(positions, kwargs),
                    min=0,
                )
                inputs_embeds = self.semantic_embedding(input_ids)
                inputs_embeds = inputs_embeds + self.semantic_position_embedding.embedding(semantic_pos)
            hidden_states = inputs_embeds
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for layer in self.transformer.h[self.start_layer : self.end_layer]:
            hidden_states = layer(hidden_states)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        return self.transformer.ln_f(hidden_states)

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None or self.semantic_head is None:
            return None
        return self.logits_processor(
            self.semantic_head,
            self.final_norm(hidden_states),
            embedding_bias=getattr(self.semantic_head, "bias", None),
        )

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        del input_embeds
        meta = info_dict.get("meta", {})
        if input_ids.shape[0] == 1 and meta.get("semantic_start_offset") is not None:
            count = int(meta.get("semantic_code_count", 0)) + 1
            position = input_ids.new_full((1,), count)
            embeds = self.semantic_embedding(input_ids)
            embeds = embeds + self.semantic_position_embedding.embedding(position)
            return input_ids, embeds, {"meta": {"semantic_code_count": count}}

        additional = info_dict.get("additional_information")
        if isinstance(additional, dict):
            info_dict = {**info_dict, **additional}

        def first(key: str, default: Any = None) -> Any:
            value = info_dict.get(key, default)
            return value[0] if isinstance(value, list) and value else value

        text = str(first("text", "")).strip()
        lang = str(first("lang", "zh"))
        voice = first("voice")
        if not text:
            raise ValueError("Confucius4-TTS requires non-empty text")
        if voice is None:
            raise ValueError("Confucius4-TTS requires reference audio")

        token_ids = first("text_token_ids")
        if token_ids is None:
            token_ids = tokenize_confucius4_prompt(self.model_path, text, lang)
        token_ids = [int(token) for token in token_ids]
        if len(token_ids) > self.config.max_text_tokens:
            raise ValueError(f"Confucius4-TTS formatted prompt exceeds {self.config.max_text_tokens} tokens")

        device = input_ids.device
        voice_name = first("voice_name")
        voice_created_at = int(first("voice_created_at", 0))
        ref_audio_cache_key = first("ref_audio_cache_key")
        cache_key = None
        cached = None
        if voice_name:
            cache_key = self._speaker_cache.make_cache_key(
                str(voice_name),
                "confucius4_tts",
                voice_created_at,
            )
        elif ref_audio_cache_key:
            cache_key = self._speaker_cache.make_cache_key(
                str(ref_audio_cache_key),
                "confucius4_tts_ref",
            )
        if cache_key is not None:
            cached = self._speaker_cache.get(cache_key)
        if cached is None:
            if self._w2v_stats is None:
                self._w2v_stats = load_wav2vec_stats(
                    self.model_path,
                    self.config.w2v_stat,
                )
            features = extract_reference_features(
                voice,
                model_path=self.model_path,
                device=device,
                w2v_stats=self._w2v_stats,
                max_audio_length_seconds=self.config.max_reference_audio_seconds,
            )
            if cache_key is not None:
                self._speaker_cache.put(
                    cache_key,
                    {k: v.detach().cpu().contiguous() for k, v in features.items()},
                )
        else:
            features = {k: v.to(device) for k, v in cached.items()}

        condition_vector = features["condition_vector"].to(
            device=device,
            dtype=next(self.speaker_encoder.parameters()).dtype,
        )
        speaker_emb = self.speaker_encoder(condition_vector).unsqueeze(1)
        text_tensor = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=device,
        )
        text_emb = self.text_position_embedding(self.text_projector(text_tensor))
        bos = torch.tensor(
            [[self.config.bos_token_id]],
            dtype=torch.long,
            device=device,
        )
        bos_emb = self.semantic_position_embedding(self.semantic_embedding(bos))
        inputs_embeds = torch.cat([speaker_emb, text_emb, bos_emb], dim=1).squeeze(0)
        if inputs_embeds.shape[0] != input_ids.shape[0]:
            raise ValueError(
                "Confucius4-TTS prefill placeholder mismatch: "
                f"scheduled={input_ids.shape[0]} actual={inputs_embeds.shape[0]}"
            )
        semantic_start_offset = 1 + len(token_ids)
        input_ids_out = torch.full_like(input_ids, self.config.bos_token_id)
        return (
            input_ids_out,
            inputs_embeds,
            {
                "meta": {
                    "semantic_start_offset": semantic_start_offset,
                    "semantic_code_count": 0,
                    # Conditioning Stage 1 needs to recompute the latent and
                    # to condition the S2A model. Kept on CPU in float32.
                    "text_token_ids": torch.tensor(token_ids, dtype=torch.long),
                    "condition_vector": features["condition_vector"].detach().float().cpu().contiguous(),
                    "ref_mel": features["ref_mel"].cpu().contiguous(),
                    "style": features["style"].cpu().contiguous(),
                },
            },
        )

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput:
        """Publish conditioning once; token ids come from the finished request.

        The conditioning tensors are emitted on the prefill step, which also
        samples the first semantic token. A request can finish on that step
        (``max_tokens=1`` or an immediate EOS), so waiting for a decode step
        would leave Stage 1 without input until its wait deadline expires.
        Decode steps only refresh the counter.
        """
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        hidden = model_outputs
        infos = kwargs.get("model_intermediate_buffer")
        if infos is None:
            infos = kwargs.get("runtime_additional_information") or []
        if "runtime_additional_information" in kwargs and "model_intermediate_buffer" not in kwargs:
            logger.warning_once("runtime_additional_information is deprecated, use model_intermediate_buffer")
        meta_rows: list[dict[str, Any]] = []
        for info in infos:
            meta = info.get("meta", {}) if isinstance(info, dict) else {}
            if meta.get("semantic_start_offset") is None:
                # Profiling and dummy rows carry no request state.
                meta_rows.append({})
                continue
            count = int(meta.get("semantic_code_count", 0))
            row_meta: dict[str, Any] = {"semantic_code_count": count}
            if count == 0:
                for key in _STAGE1_CONDITIONING_KEYS:
                    value = meta.get(key)
                    if isinstance(value, torch.Tensor):
                        row_meta[key] = value
            meta_rows.append(row_meta)
        if not any(meta_rows):
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})
        payload: OmniPayload = {"meta": meta_rows}
        return OmniOutput(text_hidden_states=hidden, multimodal_outputs=payload)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        del weights
        checkpoint = resolve_model_file(
            self.model_path,
            self.config.t2s_checkpoint,
        )
        if checkpoint is None:
            raise FileNotFoundError(f"Confucius4-TTS checkpoint {self.config.t2s_checkpoint!r} not found")
        state = load_file(checkpoint, device="cpu")
        params = dict(self.named_parameters(remove_duplicate=False))
        loaded: set[str] = set()
        unexpected: set[str] = set()
        for name, weight in state.items():
            if ".attn.bias" in name or ".attn.masked_bias" in name:
                continue
            if is_pp_missing_parameter(name, self):
                continue
            if name not in params:
                unexpected.add(name)
                continue
            if name.startswith("transformer.h.") and name.endswith(
                (
                    ".c_attn.weight",
                    ".c_proj.weight",
                    ".c_fc.weight",
                )
            ):
                weight = weight.t()
            param = params[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, weight)
            loaded.add(name)

        pp_group = get_pp_group()
        required_prefixes: list[str] = []
        if pp_group.is_first_rank:
            required_prefixes.extend(
                [
                    "text_projector.",
                    "speaker_encoder.",
                    "text_position_embedding.",
                    "semantic_position_embedding.",
                    "semantic_embedding.",
                ]
            )
        required_prefixes.extend(
            f"transformer.h.{layer_index}." for layer_index in range(self.start_layer, self.end_layer)
        )
        if pp_group.is_last_rank:
            required_prefixes.extend(
                [
                    "transformer.ln_f.",
                    "final_norm.",
                    "semantic_head.",
                ]
            )
        required: dict[int, str] = {}
        for name, param in params.items():
            if any(name.startswith(prefix) for prefix in required_prefixes):
                required.setdefault(id(param), name)
        loaded_param_ids = {id(params[name]) for name in loaded}
        missing = sorted(name for param_id, name in required.items() if param_id not in loaded_param_ids)
        if missing:
            raise RuntimeError("Confucius4-TTS checkpoint is missing required parameters: " + ", ".join(missing))
        if unexpected:
            logger.warning(
                "Ignored %d unexpected Confucius4-TTS checkpoint keys: %s",
                len(unexpected),
                ", ".join(sorted(unexpected)),
            )
        logger.info("Loaded %d Confucius4-TTS talker weights", len(loaded))
        return loaded
