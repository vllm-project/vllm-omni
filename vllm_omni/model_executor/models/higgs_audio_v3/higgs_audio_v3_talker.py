# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-0 talker for higgs-audio v3 (Qwen3 backbone, fused multi-codebook).

Architecture:
- Backbone: Qwen3 (~4B, 36 layers, 2560 hidden, GQA 32/8). No DualFFN.
- Fused multi-codebook embedding: [N*V, D] weight, offset lookup, sum across N
- Fused multi-codebook head: same weight (tied), reshape to [L, N, V]
- MusicGen-style delay pattern [0,1,...,7] with BOC/EOC
- Audio feedback: replace audio_token_id embedding with fused codebook embed

Weight loading maps from the HF checkpoint's prefixes:
  tied.embedding.text_embedding. -> model.embed_tokens.
  body.layers.                   -> model.layers.
  body.norm.                     -> model.norm.
  tied.head.text_head.           -> lm_head.
  tied.embedding.modality_embeddings.0.embedding. -> multimodal_embedding.
  tied.embedding.modality_embeddings.0.model.*    -> skipped (codec for code2wav)
  tied.head.modality_heads.0.*                    -> skipped when tied
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor

from vllm_omni.model_executor.models.higgs_audio_v3.configuration_higgs_audio_v3 import (
    HiggsAudioV3Config,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

__all__ = ["HiggsAudioV3TalkerForConditionalGeneration"]

logger = init_logger(__name__)

# Delay pattern constants
BOC_ID = 1024  # beginning of codebook
EOC_ID = 1025  # end of codebook
NUM_CODEBOOKS = 8

# Checkpoint prefix mapping: HF checkpoint -> vLLM parameter names
_BACKBONE_PREFIX_MAP = {
    "tied.embedding.text_embedding.": "model.embed_tokens.",
    "body.layers.": "model.layers.",
    "body.norm.": "model.norm.",
    "tied.head.text_head.": "lm_head.",
}

_MODALITY_EMBEDDING_PREFIX = "tied.embedding.modality_embeddings.0.embedding."
_MODALITY_HEAD_PREFIX = "tied.head.modality_heads.0."
_CODEC_PREFIX = "tied.embedding.modality_embeddings.0.model."


class HiggsFusedMultiTextEmbedding(nn.Module):
    """Fused multi-codebook embedding: [N*V, D] weight + offset lookup."""

    def __init__(self, num_codebooks: int, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_codebooks * vocab_size, hidden_size))
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: [..., N] -> [..., D] summed across codebook axis."""
        N = self.num_codebooks
        V = self.vocab_size
        offsets = torch.arange(N, device=codes.device, dtype=codes.dtype) * V
        fused_ids = codes + offsets
        return F.embedding(fused_ids, self.weight).sum(dim=-2)


class HiggsFusedMultiTextHead(nn.Module):
    """Fused multi-codebook head: [L, D] -> [L, N, V] via one linear."""

    def __init__(self, num_codebooks: int, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_codebooks * vocab_size, hidden_size))
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size

    def generate(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden, self.weight)
        return logits.reshape(hidden.shape[0], self.num_codebooks, self.vocab_size)


class HiggsAudioV3TalkerForConditionalGeneration(nn.Module):
    """Stage-0 talker for higgs-audio v3.

    Wraps vLLM's Qwen3ForCausalLM backbone and adds fused multi-codebook
    embedding/head for multi-codebook audio generation with MusicGen-style
    delay pattern.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        hf_config = vllm_config.model_config.hf_config
        if isinstance(hf_config, HiggsAudioV3Config):
            self.config = hf_config
        else:
            self.config = HiggsAudioV3Config(**hf_config.to_dict())

        self.vllm_config = vllm_config

        # Audio constants
        self.num_codebooks = int(self.config.num_codebooks)
        self.codebook_size = int(self.config.codebook_size)
        hidden_size = int(self.config.audio_hidden_size)
        self.tie_modality = self.config.tie_modality_embeddings

        # Fused multi-codebook modules
        self.multimodal_embedding = HiggsFusedMultiTextEmbedding(self.num_codebooks, self.codebook_size, hidden_size)
        self.modality_head = HiggsFusedMultiTextHead(self.num_codebooks, self.codebook_size, hidden_size)
        if self.tie_modality:
            self.modality_head.weight = self.multimodal_embedding.weight

        # Qwen3 backbone - we'll build it using the text_config
        # We need to patch vllm_config to use text_config for the backbone
        self._backbone_config = self.config.text_config
        self._build_backbone(vllm_config, prefix)

        # LM logits processor (for text-level token sampling)
        self.logits_processor = LogitsProcessor(self._backbone_config.vocab_size)

        # Engine hooks
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False

        # Per-request audio state (populated during sampling)
        self._audio_state: dict[int, dict[str, Any]] = {}
        self._last_logits_hidden: torch.Tensor | None = None
        self._last_step_input_ids: torch.Tensor | None = None

    def _build_backbone(self, vllm_config: VllmConfig, prefix: str) -> None:
        """Build the Qwen3 backbone model using the text_config."""
        import copy

        from vllm.model_executor.models.qwen3 import Qwen3Model

        # Create a modified vllm_config that uses text_config as hf_config
        backbone_vllm_config = copy.copy(vllm_config)
        backbone_model_config = copy.copy(vllm_config.model_config)
        backbone_model_config.hf_config = self._backbone_config
        backbone_vllm_config.model_config = backbone_model_config

        self.model = Qwen3Model(
            vllm_config=backbone_vllm_config,
            prefix=f"{prefix}.model" if prefix else "model",
        )

        # LM head from the backbone
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

        if self._backbone_config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                self._backbone_config.vocab_size,
                self._backbone_config.hidden_size,
                prefix=f"{prefix}.lm_head" if prefix else "lm_head",
            )

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward through Qwen3 backbone with audio feedback."""
        if inputs_embeds is None:
            hidden_states = self.model.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        # Stash input_ids for audio mode bias in sample()
        if input_ids is not None:
            self._last_step_input_ids = input_ids

        # Audio feedback: at decode time, replace audio_token_id positions
        # with fused codebook embedding of last generated frame
        if input_ids is not None and inputs_embeds is None:
            hidden_states = self._apply_audio_feedback(hidden_states, input_ids)

        # Run through Qwen3 transformer layers
        residual: torch.Tensor | None = None
        for layer in self.model.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

        # Final norm
        norm_out = self.model.norm(hidden_states, residual)
        if isinstance(norm_out, tuple):
            norm_out = norm_out[0]
        return norm_out

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Any = None,
    ) -> torch.Tensor:
        """Text-vocab logits for the LM-level sampler."""
        self._last_logits_hidden = hidden_states
        return self.logits_processor(self.lm_head, hidden_states, sampling_metadata)

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        """Embed with audio feedback substitution."""
        text_embed = self.model.embed_tokens(input_ids)
        return self._apply_audio_feedback(text_embed, input_ids)

    # ------------------------------------------------------------------ audio feedback
    def _apply_audio_feedback(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Replace audio_token_id positions with fused codebook embedding of last codes."""
        # V3 uses audio_token_id from the config (typically from tokenizer)
        # For now, detect audio positions by checking audio_state
        if not self._audio_state:
            return hidden_states

        # Find audio_token_id - in v3, the model uses <|audio|> as the
        # continuation token during audio generation. We need to detect
        # decode-time audio positions from the audio state.
        flat_ids = input_ids.reshape(-1)
        rep_positions: list[int] = []
        rep_embeds: list[torch.Tensor] = []

        for pos in range(flat_ids.numel()):
            req_state = self._audio_state.get(pos)
            if req_state is None:
                continue
            last_codes = req_state.get("last_codes")
            if not isinstance(last_codes, torch.Tensor) or last_codes.numel() == 0:
                continue
            # Fused codebook embedding: [1, N] -> [1, D] summed across codebooks
            codes_1n = last_codes.unsqueeze(0).to(hidden_states.device)
            audio_emb = self.multimodal_embedding(codes_1n)  # [1, D]
            rep_positions.append(pos)
            rep_embeds.append(audio_emb[0].to(dtype=hidden_states.dtype))

        if not rep_positions:
            return hidden_states

        new_hidden = hidden_states.clone()
        flat_hidden = new_hidden.reshape(-1, new_hidden.shape[-1])
        idx = torch.tensor(rep_positions, dtype=torch.long, device=new_hidden.device)
        rep = torch.stack(rep_embeds, dim=0)
        flat_hidden.index_copy_(0, idx, rep)
        return new_hidden

    # ------------------------------------------------------------------ sampling
    def sample(self, logits: torch.Tensor, sampling_metadata: Any) -> Any:
        """Model-owned sampler with delay-pattern audio dispatch."""
        sampler = getattr(self, "_stock_sampler", None)
        if sampler is None:
            from vllm.v1.sample.sampler import Sampler

            sampler = Sampler()
            self._stock_sampler = sampler

        # Apply audio mode bias: force audio continuation token during generation
        self._apply_audio_mode_bias(logits, sampling_metadata)
        sampler_output = sampler(logits=logits, sampling_metadata=sampling_metadata)

        hidden = self._last_logits_hidden
        self._last_logits_hidden = None
        if hidden is None:
            return sampler_output

        sampled = getattr(sampler_output, "sampled_token_ids", None)
        if sampled is None:
            return sampler_output
        sampled_flat = sampled.reshape(-1)
        if int(sampled_flat.numel()) != int(hidden.shape[0]):
            return sampler_output

        # Detect audio positions from output_token_ids
        # In v3, after <|audio|>, the model should continue generating audio
        audio_token_id = self._get_audio_continuation_token_id()
        if audio_token_id is None:
            return sampler_output

        is_audio = sampled_flat == audio_token_id

        # Check for first-after-audio-start transitions
        first_after_audio = getattr(self, "_last_first_audio_step", None)
        self._last_first_audio_step = None

        seeded_rows: list[int] = []
        if isinstance(first_after_audio, torch.Tensor) and first_after_audio.numel() == is_audio.shape[0]:
            first_after_audio = first_after_audio.to(is_audio.device)
            skip_rows = first_after_audio & is_audio
            if bool(skip_rows.any()):
                seeded_frame = torch.full((self.num_codebooks,), BOC_ID, dtype=torch.long, device=hidden.device)
                for bi in torch.nonzero(skip_rows, as_tuple=False).reshape(-1).tolist():
                    bi = int(bi)
                    self._audio_state[bi] = {
                        "delay_count": 0,
                        "eoc_countdown": -1,
                        "generation_done": False,
                        "last_codes": seeded_frame.clone(),
                        "audio_out_ids": seeded_frame.unsqueeze(-1).clone(),
                    }
                    seeded_rows.append(bi)
                is_audio = is_audio & ~first_after_audio

        if not bool(is_audio.any()) and not seeded_rows:
            return sampler_output

        audio_row_indices = torch.nonzero(is_audio, as_tuple=False).reshape(-1).tolist()

        if not audio_row_indices:
            return sampler_output

        # Per-codebook logits at audio positions
        cb_logits = self._audio_codebook_logits(hidden, is_audio)  # [N_audio, Q, V]

        # Apply delay pattern masking
        self._apply_delay_pattern_masking(cb_logits, audio_row_indices)

        # Sample per-codebook
        cb_logits_2d = cb_logits.reshape(-1, cb_logits.shape[-1])
        codes_2d = self._sample_audio_codes(cb_logits_2d)
        codes_flat = codes_2d.view(cb_logits.shape[0], cb_logits.shape[1]).to(torch.long)

        # Update delay pattern state and audio_out_ids
        for local_i, batch_i in enumerate(audio_row_indices):
            batch_i = int(batch_i)
            state = self._audio_state.get(batch_i)
            if state is None:
                state = {
                    "delay_count": 0,
                    "eoc_countdown": -1,
                    "generation_done": False,
                    "last_codes": torch.full((self.num_codebooks,), BOC_ID, dtype=torch.long, device=hidden.device),
                    "audio_out_ids": torch.empty((self.num_codebooks, 0), dtype=torch.long, device=hidden.device),
                }
                self._audio_state[batch_i] = state

            this_codes = codes_flat[local_i]  # [Q]
            delay_count = state["delay_count"]
            eoc_countdown = state["eoc_countdown"]

            # Delay phase: increment and mask
            if delay_count < self.num_codebooks:
                next_cb = delay_count + 1
                if next_cb < self.num_codebooks:
                    this_codes[next_cb:] = BOC_ID
                state["delay_count"] = delay_count + 1
            # Wind-down phase
            elif eoc_countdown >= 0:
                state["eoc_countdown"] = eoc_countdown - 1
                if state["eoc_countdown"] <= 0:
                    state["generation_done"] = True
            # EOC detection on codebook 0
            elif int(this_codes[0].item()) == EOC_ID:
                if self.num_codebooks <= 2:
                    state["generation_done"] = True
                else:
                    state["eoc_countdown"] = self.num_codebooks - 2

            # Update last_codes and audio_out_ids
            if not state["generation_done"]:
                state["last_codes"] = this_codes.clone()
            state["audio_out_ids"] = torch.cat([state["audio_out_ids"], this_codes.unsqueeze(-1)], dim=-1)

            # Force audio_eos at ramp-down completion
            if state["generation_done"]:
                eos_token_id = self._get_audio_eos_token_id()
                if eos_token_id is not None:
                    sampler_output.sampled_token_ids[batch_i] = eos_token_id

        return sampler_output

    def _audio_codebook_logits(self, hidden_states: torch.Tensor, audio_mask: torch.Tensor) -> torch.Tensor:
        """Per-codebook logits at audio positions: [N_audio, N, V]."""
        mask = audio_mask.reshape(-1).to(hidden_states.device)
        hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        if not mask.any():
            return torch.empty(
                (0, self.num_codebooks, self.codebook_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        return self.modality_head.generate(hidden_flat[mask])

    def _apply_delay_pattern_masking(self, cb_logits: torch.Tensor, audio_row_indices: list[int]) -> None:
        """Mask per-codebook logits according to delay pattern state."""
        for local_i, batch_i in enumerate(audio_row_indices):
            state = self._audio_state.get(int(batch_i))
            delay_count = int(state["delay_count"]) if state else 0
            eoc_countdown = state.get("eoc_countdown", -1) if state else -1

            if eoc_countdown >= 0:
                # Ramp-down: lock codebooks [0:lock_until] to EOC only
                lock_until = self.num_codebooks - int(eoc_countdown)
                for q in range(self.num_codebooks):
                    row = cb_logits[local_i, q]
                    if q < lock_until:
                        mask = torch.full_like(row, float("-inf"))
                        mask[EOC_ID] = row[EOC_ID]
                        cb_logits[local_i, q] = mask
                    else:
                        cb_logits[local_i, q, BOC_ID] = float("-inf")
                        cb_logits[local_i, q, EOC_ID] = float("-inf")
            else:
                # Delay phase or normal generation
                for q in range(self.num_codebooks):
                    row = cb_logits[local_i, q]
                    if q > delay_count:
                        # Force BOC for codebooks not yet active
                        mask = torch.full_like(row, float("-inf"))
                        mask[BOC_ID] = row[BOC_ID]
                        cb_logits[local_i, q] = mask
                    else:
                        # Disallow BOC for active codebooks
                        cb_logits[local_i, q, BOC_ID] = float("-inf")
                        # Only codebook 0 can trigger ramp-down
                        if q != 0:
                            cb_logits[local_i, q, EOC_ID] = float("-inf")

    def _sample_audio_codes(self, logits_2d: torch.Tensor) -> torch.Tensor:
        """Sample from per-codebook logits with temperature/top-k/top-p."""
        temperature = 1.0
        top_k = 50
        top_p = 0.95

        logits = logits_2d / temperature
        # Top-k filtering
        if top_k > 0:
            topk_vals, _ = logits.topk(top_k, dim=-1)
            threshold = topk_vals[:, -1].unsqueeze(-1)
            logits = logits.where(logits >= threshold, torch.full_like(logits, float("-inf")))
        # Top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _apply_audio_mode_bias(self, logits: torch.Tensor, sampling_metadata: Any) -> None:
        """Force audio continuation token during audio generation."""
        # During audio generation, the LM should keep emitting the audio
        # continuation token so the model stays in audio mode.
        # The exact mechanism depends on how the prompt is structured.
        # For v3: after <|audio|>, the model should continue with audio_token_id
        prev_ids = self._last_step_input_ids
        if prev_ids is None:
            return

        audio_token_id = self._get_audio_continuation_token_id()
        if audio_token_id is None:
            return

        flat_prev = prev_ids.reshape(-1)
        for i in range(flat_prev.numel()):
            state = self._audio_state.get(i)
            if state is not None and not state.get("generation_done", False):
                # In audio generation mode: force audio continuation token
                mask = torch.full_like(logits[i], float("-inf"))
                mask[audio_token_id] = logits[i, audio_token_id]
                logits[i] = mask
            elif state is not None and state.get("generation_done", False):
                # Generation done: force eos
                eos_id = self._get_audio_eos_token_id()
                if eos_id is not None:
                    mask = torch.full_like(logits[i], float("-inf"))
                    mask[eos_id] = logits[i, eos_id]
                    logits[i] = mask

    def _get_audio_continuation_token_id(self) -> int | None:
        """Get the token ID used to continue audio generation in the LM stream."""
        # In v3, this is the <|audio|> token or a similar audio continuation token
        # We need to resolve this from the config/tokenizer
        # For now, use audio_token_id from config (-100 is placeholder, actual ID
        # will be resolved at runtime from the tokenizer)
        audio_id = getattr(self.config, "audio_token_id", None)
        if audio_id is not None and audio_id != -100:
            return int(audio_id)
        return None

    def _get_audio_eos_token_id(self) -> int | None:
        """Get the token ID that signals end of audio generation."""
        eos_id = getattr(self.config, "eos_token_id", None)
        if eos_id is not None:
            return int(eos_id)
        return None

    # ------------------------------------------------------------------ omni output
    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput:
        """Wrap decoder outputs into OmniOutput with audio codes."""
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        hidden = model_outputs

        info_dicts = kwargs.get("model_intermediate_buffer")
        if info_dicts is None:
            info_dicts = kwargs.get("runtime_additional_information")
        if info_dicts is None:
            info_dicts = []

        audio_codes_list: list[torch.Tensor] = []
        any_nonempty = False
        for info in info_dicts:
            ac: torch.Tensor | None = None
            if isinstance(info, dict):
                codes_field = info.get("codes")
                if isinstance(codes_field, dict):
                    ac = codes_field.get("audio")
                else:
                    ac = info.get("audio_codes")
            if isinstance(ac, torch.Tensor) and ac.numel() > 0:
                audio_codes_list.append(ac)
                any_nonempty = True
            else:
                audio_codes_list.append(torch.empty(0, dtype=torch.long))

        if any_nonempty:
            return OmniOutput(
                text_hidden_states=hidden,
                multimodal_outputs={
                    "codes": {"audio": audio_codes_list},
                },
            )
        return OmniOutput(text_hidden_states=hidden, multimodal_outputs=None)

    # ------------------------------------------------------------------ weight loading
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load from v3 checkpoint with prefix remapping."""
        backbone_weights: list[tuple[str, torch.Tensor]] = []
        loaded_params: set[str] = set()

        own_params = dict(self.named_parameters())

        for name, tensor in weights:
            mapped = self._map_weight_name(name)
            if mapped is None:
                continue  # Skip codec weights, etc.

            if mapped.startswith("model.") or mapped.startswith("lm_head."):
                backbone_weights.append((mapped, tensor))
            elif mapped in own_params:
                param = own_params[mapped]
                if param.shape == tensor.shape:
                    param.data.copy_(tensor.to(param.dtype))
                    loaded_params.add(mapped)
                else:
                    logger.warning(
                        "Shape mismatch for %s: expected %s, got %s",
                        mapped,
                        param.shape,
                        tensor.shape,
                    )

        # Load backbone weights via Qwen3's standard loader
        if backbone_weights:
            # Build a temporary wrapper to use AutoWeightsLoader
            backbone_module = _BackboneWrapper(self.model, self.lm_head, self._backbone_config)
            try:
                loaded = backbone_module.load_weights(iter(backbone_weights))
                loaded_params.update(f"model.{k}" if not k.startswith("lm_head") else k for k in loaded)
            except Exception as exc:
                logger.warning("Backbone weight loading via AutoWeightsLoader failed: %s", exc)
                # Fallback: manual assignment
                self._manual_load_backbone(backbone_weights, loaded_params)

        logger.info(
            "HiggsAudioV3Talker: loaded %d parameters, modality_embedding shape=%s, tied=%s",
            len(loaded_params),
            tuple(self.multimodal_embedding.weight.shape),
            self.tie_modality,
        )
        return loaded_params

    def _map_weight_name(self, name: str) -> str | None:
        """Map a checkpoint weight name to our parameter name."""
        # Skip codec weights
        if name.startswith(_CODEC_PREFIX):
            return None

        # Skip modality head when tied
        if name.startswith(_MODALITY_HEAD_PREFIX):
            if self.tie_modality:
                return None
            return name.replace(_MODALITY_HEAD_PREFIX, "modality_head.")

        # Map modality embedding
        if name.startswith(_MODALITY_EMBEDDING_PREFIX):
            return name.replace(_MODALITY_EMBEDDING_PREFIX, "multimodal_embedding.")

        # Map backbone prefixes
        for ckpt_prefix, model_prefix in _BACKBONE_PREFIX_MAP.items():
            if name.startswith(ckpt_prefix):
                return name.replace(ckpt_prefix, model_prefix, 1)

        # Unknown prefix - skip with warning
        logger.debug("Skipping unknown weight: %s", name)
        return None

    def _manual_load_backbone(
        self,
        backbone_weights: list[tuple[str, torch.Tensor]],
        loaded_params: set[str],
    ) -> None:
        """Manual fallback for backbone weight loading."""
        own_params = dict(self.named_parameters())
        for name, tensor in backbone_weights:
            if name in own_params:
                param = own_params[name]
                if param.shape == tensor.shape:
                    param.data.copy_(tensor.to(param.dtype))
                    loaded_params.add(name)


class _BackboneWrapper(nn.Module):
    """Temporary wrapper to use AutoWeightsLoader for Qwen3 backbone."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, model, lm_head, config):
        super().__init__()
        self.model = model
        self.lm_head = lm_head
        self.config = config

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm.model_executor.model_loader.weight_utils import AutoWeightsLoader

        skip = ["lm_head."] if getattr(self.config, "tie_word_embeddings", False) else None
        loader = AutoWeightsLoader(self, skip_prefixes=skip)
        return loader.load_weights(weights)
