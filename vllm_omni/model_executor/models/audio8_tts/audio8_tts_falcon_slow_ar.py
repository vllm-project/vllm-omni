# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio8 TTS Preview 0.1b -- Slow AR model (Stage 0), Falcon-H1 backbone.

Same DualAR design as the 0.6b Slow AR, but the slow backbone is Falcon-H1
(Mamba2 + attention parallel hybrid, muP multipliers) instead of Qwen2, and the
semantic head is a dedicated ``semantic_output`` linear (4096 codes + EOS)
rather than a sliced tied embedding. Everything else -- multi-codebook input
embedding, Repetition-Aware sampling, voice cloning, the nested Fast AR, the
streaming delta contract -- is inherited unchanged from
``Audio8TTSSlowARForConditionalGeneration``.

``slow_backbone: "falcon_h1"`` in the arktts config routes here (see
``vllm_omni/config/pipeline_registry`` / the pipeline ``hf_config_predicate``).
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.config.vllm import set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.falcon_h1 import FalconH1ForCausalLM, FalconH1Model
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.utils.speaker_cache import get_speaker_cache

from .audio8_tts_fast_ar import Audio8TTSFastAR
from .audio8_tts_slow_ar import (
    Audio8TTSSlowARForConditionalGeneration,
    _remap_audio8_tts_weights,
)

logger = init_logger(__name__)


class Audio8TTS01BSlowARForConditionalGeneration(Audio8TTSSlowARForConditionalGeneration):
    """Stage 0 for Audio8 TTS Preview 0.1b (Falcon-H1 hybrid Slow AR)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Deliberately does NOT call super().__init__: the parent builds a
        # Qwen2 backbone + tied lm_head. We build a Falcon-H1 backbone + a
        # dedicated semantic head, then reuse every parent method.
        nn.Module.__init__(self)
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        config = vllm_config.model_config.hf_config
        self.config = config
        # For 0.1b, get_text_config() returns a FalconH1Config annotated with the
        # DualAR fields (semantic_begin_id, codebook_size, ...) the parent reads.
        self.text_config = config.get_text_config()
        self.fast_ar_config = config.fast_ar_config

        self._semantic_begin_id = int(self.text_config.semantic_begin_id)
        self._semantic_end_id = int(self.text_config.semantic_end_id)
        self._num_semantic_ids = self._semantic_end_id - self._semantic_begin_id + 1
        self._eos_token_id = int(self.text_config.eos_token_id)
        self._pad_token_id = int(self.text_config.pad_token_id)
        self._codebook_size = int(self.text_config.codebook_size)
        self._num_codebooks = int(self.text_config.num_codebooks)
        self._ras_window_size = int(config.ras_window_size)
        self._ras_temperature = float(config.ras_temperature)
        self._ras_top_p = float(config.ras_top_p)

        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = True
        self.mtp_hidden_size = int(self.text_config.hidden_size)
        self.talker_mtp_output_key = ("codes", "audio")
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {("hidden_states", "last")}
        self.talker_mtp_graph_safe = True

        # Falcon-H1 backbone. embedding_multiplier is applied inside its forward
        # to inputs_embeds, so the wrapper must pass un-multiplied embeddings.
        self.model = FalconH1Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

        # Dedicated compact semantic head: 4096 semantic codes + 1 EOS. The
        # reference does not scale it by lm_head_multiplier, so neither do we.
        self.semantic_output = nn.Linear(
            self.text_config.hidden_size,
            self._num_semantic_ids + 1,
            bias=False,
        )
        # Kept for framework compatibility; compute_logits below uses
        # semantic_output directly rather than a full-vocabulary lm_head.
        self.logits_processor = LogitsProcessor(int(self.text_config.vocab_size))
        self.make_empty_intermediate_tensors = getattr(self.model, "make_empty_intermediate_tensors", None)

        # Summed multi-codebook input embedding table (identical to 0.6b).
        self.codebook_embeddings = nn.Embedding(
            self._codebook_size * self._num_codebooks,
            self.text_config.hidden_size,
        )

        # Fast AR: separate compilation context, same as the 0.6b Slow AR.
        fast_ar_compilation = copy.copy(vllm_config.compilation_config)
        fast_ar_compilation.static_forward_context = {}
        self._fast_ar_vllm_config = copy.copy(vllm_config)
        self._fast_ar_vllm_config.compilation_config = fast_ar_compilation
        with set_current_vllm_config(self._fast_ar_vllm_config):
            self.fast_ar = Audio8TTSFastAR(
                vllm_config=self._fast_ar_vllm_config,
                config=self.fast_ar_config,
                slow_ar_config=self.text_config,
                prefix="fast_ar",
            )

        self._speaker_cache = get_speaker_cache()
        self._tokenizer = None

    def _fix_rope_style(self) -> None:
        """No-op: Falcon-H1 builds RoPE from ``rope_parameters`` (NeoX style,
        rope_theta 1e11), which already matches the reference. Unlike the 0.6b
        Qwen2 backbone, no interleaved-RoPE switch is needed."""

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None

        # semantic_output emits [B, num_semantic_ids + 1] (codes then EOS).
        compact = self.semantic_output(hidden_states)
        # Scatter into full-vocabulary width so the inherited sample() /
        # _compact_semantic_logits (which slice the semantic range + EOS out of
        # a vocab-wide tensor) work unchanged. Everything else stays -inf.
        vocab = int(self.text_config.vocab_size)
        full = compact.new_full((compact.shape[0], vocab), float("-inf"))
        end = min(self._semantic_end_id + 1, vocab)
        full[:, self._semantic_begin_id : end] = compact[:, : end - self._semantic_begin_id]
        if self._eos_token_id < vocab:
            full[:, self._eos_token_id] = compact[:, self._num_semantic_ids]
        return full

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        fast_config = self.fast_ar_config
        fast_q_size = fast_config.num_attention_heads * fast_config.head_dim
        fast_kv_size = fast_config.num_key_value_heads * fast_config.head_dim

        slow_weights: list[tuple[str, torch.Tensor]] = []
        fast_src: list[tuple[str, torch.Tensor]] = []
        other: list[tuple[str, torch.Tensor]] = []
        for name, tensor in weights:
            if name.endswith(("freqs_cis", "fast_freqs_cis")) or "rotary_emb.inv_freq" in name:
                continue
            if name.startswith("slow."):
                slow_weights.append((name[len("slow.") :], tensor))
            elif name.startswith(
                ("fast_layers.", "fast_embeddings.", "fast_output.", "fast_norm.", "fast_project_in.")
            ):
                fast_src.append((name, tensor))
            else:
                other.append((name, tensor))

        loaded_params: set[str] = set()

        # Slow backbone: reuse vLLM's Falcon-H1 loader (handles q/k/v -> qkv,
        # gate/up -> gate_up, A_log -> A, mamba -> mamba.mamba).
        slow_loaded = AutoWeightsLoader(self.model).load_weights(
            slow_weights, mapper=FalconH1ForCausalLM.hf_to_vllm_mapper
        )
        loaded_params |= {f"model.{name}" for name in slow_loaded}

        # Fast AR + heads: load through this module's params with the same
        # stacked mapping the 0.6b Slow AR uses. The fast remapper renames
        # fast_layers.* -> fast_ar.* and splits wqkv; heads load by name.
        remapped_fast = _remap_audio8_tts_weights(
            fast_src,
            q_size=0,
            kv_size=0,
            fast_q_size=fast_q_size,
            fast_kv_size=fast_kv_size,
        )
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        unexpected: list[str] = []
        for name, loaded_weight in list(remapped_fast) + other:
            handled = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                loader = getattr(param, "weight_loader", default_weight_loader)
                if loader == default_weight_loader:
                    loader(param, loaded_weight)
                else:
                    loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped)
                handled = True
                break
            if handled:
                continue
            if name in params_dict:
                param = params_dict[name]
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight)
                loaded_params.add(name)
            else:
                unexpected.append(name)

        if unexpected:
            raise ValueError(
                f"Audio8 TTS 0.1b Slow AR received {len(unexpected)} unmapped checkpoint tensors, "
                f"e.g. {sorted(unexpected)[:5]}. The weight remapper is out of sync with the checkpoint."
            )
        missing = sorted(set(params_dict) - loaded_params)
        if missing:
            raise ValueError(f"Audio8 TTS 0.1b Slow AR is missing weights for {missing[:5]} ({len(missing)} total)")
        logger.info("Loaded %d weights for Audio8TTS01BSlowARForConditionalGeneration", len(loaded_params))
        return loaded_params


__all__ = ["Audio8TTS01BSlowARForConditionalGeneration"]
