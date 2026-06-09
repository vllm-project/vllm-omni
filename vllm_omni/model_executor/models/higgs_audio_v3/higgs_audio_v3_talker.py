# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-0 talker for higgs-audio v3 (Qwen3 backbone, fused multi-codebook).

Architecture:
- Backbone: Qwen3 (~4B, 36 layers, 2560 hidden, GQA 32/8). No DualFFN.
- Fused multi-codebook embedding: [N*V, D] weight, offset lookup, sum across N
- Fused multi-codebook head: same weight (tied), reshape to [L, N, V]
- MusicGen-style delay pattern [0,1,...,7] with BOC/EOC
- Audio feedback: replace continuation-token embedding with fused codebook embed

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

import copy
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3 import Qwen3Model

from vllm_omni.model_executor.models.higgs_audio_v3.configuration_higgs_audio_v3 import (
    HiggsAudioV3Config,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

__all__ = ["HiggsAudioV3TalkerForConditionalGeneration"]

logger = init_logger(__name__)

# Delay pattern constants
BOC_ID = 1024  # beginning of codebook
EOC_ID = 1025  # end of codebook

# Checkpoint prefix mapping
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

    Wraps Qwen3Model backbone + fused multi-codebook modules for TTS generation
    with MusicGen-style delay pattern sampling and audio feedback embedding.
    """

    # Tell the AR runner to call model.sample() instead of the stock sampler.
    prefer_model_sampler: bool = True
    # Tell the runner to call postprocess() to emit per-step audio codes.
    have_multimodal_outputs: bool = True
    has_postprocess: bool = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        hf_config = vllm_config.model_config.hf_config
        if isinstance(hf_config, HiggsAudioV3Config):
            self.config = hf_config
        else:
            self.config = HiggsAudioV3Config(**hf_config.to_dict())

        self.vllm_config = vllm_config
        self.num_codebooks = int(self.config.num_codebooks)
        self.codebook_size = int(self.config.codebook_size)
        hidden_size = int(self.config.audio_hidden_size)
        self.tie_modality = self.config.tie_modality_embeddings

        # Fused multi-codebook modules
        self.multimodal_embedding = HiggsFusedMultiTextEmbedding(self.num_codebooks, self.codebook_size, hidden_size)
        self.modality_head = HiggsFusedMultiTextHead(self.num_codebooks, self.codebook_size, hidden_size)
        if self.tie_modality:
            self.modality_head.weight = self.multimodal_embedding.weight

        # Qwen3 backbone
        self._backbone_config = self.config.text_config
        backbone_vllm_config = copy.copy(vllm_config)
        backbone_model_config = copy.copy(vllm_config.model_config)
        backbone_model_config.hf_config = self._backbone_config
        backbone_vllm_config.model_config = backbone_model_config

        self.model = Qwen3Model(
            vllm_config=backbone_vllm_config,
            prefix=f"{prefix}.model" if prefix else "model",
        )

        if self._backbone_config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                self._backbone_config.vocab_size,
                self._backbone_config.hidden_size,
                prefix=f"{prefix}.lm_head" if prefix else "lm_head",
            )

        self.logits_processor = LogitsProcessor(self._backbone_config.vocab_size)

        # Audio continuation token ID — resolved lazily from tokenizer.
        # This is the <|audio|> token that serves as the LM-level continuation
        # marker during audio generation (equivalent to v2's audio_token_id).
        self._audio_continuation_id: int | None = None
        self._eos_token_id: int | None = None
        self._resolved_tokens = False

        # Per-request audio state keyed by batch row index.
        # Reset per slot via _slot_output_len tracking (same pattern as v2).
        self._audio_state: dict[int, dict[str, Any]] = {}
        self._slot_output_len: dict[int, int] = {}
        self._last_logits_hidden: torch.Tensor | None = None
        self._last_step_input_ids: torch.Tensor | None = None
        self._last_step_query_start_loc: torch.Tensor | None = None
        self._last_first_audio_after_start: torch.Tensor | None = None
        self._last_audio_codes: torch.Tensor | None = None
        self._postprocess_cursor: int = 0

        # Pre-allocated decode-step audio feedback buffers (CUDA-graph safe).
        # Populated by sample(), read by forward() via torch.where (no dict).
        max_bs = 64  # safe upper bound; will grow if needed
        self._decode_last_codes = torch.zeros(max_bs, self.num_codebooks, dtype=torch.long)
        self._decode_has_codes = torch.zeros(max_bs, dtype=torch.bool)

        # PrefixCache opt-outs (mirror qwen3_tts pattern):
        # 1. The talker only consumes the last token's hidden state, so the
        #    runner can skip the per-step full hidden-state GPU->CPU merge
        #    that PrefixCache otherwise does.
        # 2. Per-step ``codes.audio`` rows stay GPU-resident; defer the CPU
        #    write of the prefix-cache mm-output copy to request finish so
        #    the per-step bookkeeping does not block batching. Stage 0 can
        #    then set ``enable_prefix_caching: true`` without the regression
        #    observed in qwen3_tts (#3665).
        self.requires_full_prefix_cached_hidden_states = False
        self.deferred_prefix_cache_mm_keys = {"codes.audio"}

    def _resolve_token_ids(self) -> None:
        """Resolve <|audio|> and eos token IDs.

        Prefers config's pre-resolved IDs (from ``resolve_special_tokens()``),
        falls back to loading the HF tokenizer directly.
        """
        if self._resolved_tokens:
            return
        self._resolved_tokens = True

        # Try config first (populated by resolve_special_tokens or from_pretrained)
        cfg_audio = getattr(self.config, "audio_continuation_id", None)
        cfg_eos = getattr(self.config, "eos_token_id", None)
        if cfg_audio is not None:
            self._audio_continuation_id = int(cfg_audio)
        if cfg_eos is not None:
            self._eos_token_id = int(cfg_eos)

        if self._audio_continuation_id is not None:
            logger.info(
                "Resolved v3 token IDs from config: audio_continuation=%s, eos=%s",
                self._audio_continuation_id,
                self._eos_token_id,
            )
            return

        # Fallback: load tokenizer directly
        model_path = getattr(self.vllm_config.model_config, "model", None)
        if model_path is None:
            return
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            vocab = dict(tokenizer.get_added_vocab())
            if "<|audio|>" in vocab:
                self._audio_continuation_id = vocab["<|audio|>"]
            if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
                self._eos_token_id = int(tokenizer.eos_token_id)
            logger.info(
                "Resolved v3 token IDs from tokenizer: audio_continuation=%s, eos=%s",
                self._audio_continuation_id,
                self._eos_token_id,
            )
        except Exception as exc:
            logger.warning("Failed to resolve token IDs from tokenizer: %s", exc)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            # Mask -100 placeholders to 0 before embedding. Use torch.where
            # (no Python data-dependent branch) so this is CUDA-graph safe.
            safe_ids = torch.where(input_ids < 0, torch.zeros_like(input_ids), input_ids)
            hidden_states = self.model.embed_tokens(safe_ids)
        else:
            hidden_states = inputs_embeds

        if input_ids is not None:
            self._last_step_input_ids = input_ids

        # Stash query_start_loc and max_query_len for prefill detection
        _max_query_len = None
        try:
            from vllm.forward_context import get_forward_context

            attn_metadata = get_forward_context().attn_metadata
            if isinstance(attn_metadata, dict) and attn_metadata:
                attn = next(iter(attn_metadata.values()))
            else:
                attn = attn_metadata
            qsl = getattr(attn, "query_start_loc", None)
            if isinstance(qsl, torch.Tensor):
                self._last_step_query_start_loc = qsl.detach().clone()
            else:
                self._last_step_query_start_loc = None
            _max_query_len = getattr(attn, "max_query_len", None)
        except Exception:
            self._last_step_query_start_loc = None

        # Detect prefill vs decode using attn_metadata.max_query_len.
        # max_query_len == 1 means pure decode (even with N concurrent
        # requests, each contributes exactly 1 token). numel > 1 is NOT
        # reliable because decode with N>1 concurrent requests gives
        # input_ids.numel() == N.
        if _max_query_len is not None:
            is_prefill = int(_max_query_len) > 1
        else:
            is_prefill = input_ids is not None and inputs_embeds is None and int(input_ids.numel()) > 1
        if is_prefill:
            # Voice clone: replace -100 placeholder positions with ref audio embeddings
            info_dicts = kwargs.get("model_intermediate_buffer")
            if info_dicts is None:
                info_dicts = kwargs.get("runtime_additional_information")
            hidden_states = self._apply_ref_audio_substitution(hidden_states, input_ids, info_dicts)

        # Audio feedback: replace continuation token embeddings with audio
        # embeddings from the last decoded frame (CUDA-graph safe).
        if input_ids is not None and inputs_embeds is None:
            hidden_states = self._apply_audio_feedback(hidden_states, input_ids)

        residual: torch.Tensor | None = None
        for layer in self.model.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        norm_out = self.model.norm(hidden_states, residual)
        if isinstance(norm_out, tuple):
            norm_out = norm_out[0]
        return norm_out

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: Any = None) -> torch.Tensor:
        self._last_logits_hidden = hidden_states
        return self.logits_processor(self.lm_head, hidden_states, sampling_metadata)

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        safe_ids = input_ids
        if input_ids is not None and (input_ids < 0).any():
            safe_ids = torch.where(input_ids < 0, torch.zeros_like(input_ids), input_ids)
        text_embed = self.model.embed_tokens(safe_ids)
        return self._apply_audio_feedback(text_embed, input_ids)

    # ------------------------------------------------------------------ ref audio substitution
    def _apply_ref_audio_substitution(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        info_dicts: list[dict[str, Any]] | None,
    ) -> torch.Tensor:
        """Replace -100 placeholder positions with fused multi-codebook embeddings
        of the delay-pattern-encoded reference audio codes.

        Called at prefill to inject voice clone reference. ``info_dicts`` is a
        list of per-request dicts from ``model_intermediate_buffer``, each
        containing ``audio_input_ids`` ([T, N] delayed codes) and
        ``audio_input_ids_mask`` ([T] bool mask).
        """
        if not info_dicts:
            return hidden_states

        PLACEHOLDER = -100
        flat_ids = input_ids.reshape(-1)
        placeholder_mask = flat_ids == PLACEHOLDER
        if not placeholder_mask.any():
            return hidden_states

        # Use query_start_loc to map placeholders to per-request spans
        q_start = self._last_step_query_start_loc
        if not isinstance(q_start, torch.Tensor) or q_start.numel() < 2:
            # Fallback: single-request batch
            q_start_list = [0, int(flat_ids.numel())]
        else:
            q_start_list = q_start.detach().to("cpu").tolist()

        new_hidden: torch.Tensor | None = None
        num_requests = min(len(info_dicts), len(q_start_list) - 1)

        for i in range(num_requests):
            info = info_dicts[i]
            if not isinstance(info, dict):
                continue

            codes = info.get("audio_input_ids")
            mask = info.get("audio_input_ids_mask")

            # Handle msgspec serialization (may be list-wrapped)
            if isinstance(codes, list):
                codes = codes[0] if codes else None
            if isinstance(mask, list):
                mask = mask[0] if mask else None
            if not isinstance(codes, torch.Tensor):
                continue

            # codes shape: [T, num_codebooks] delayed reference codes
            if codes.ndim == 3:
                codes = codes[0]
            if codes.ndim != 2:
                continue

            if isinstance(mask, torch.Tensor):
                if mask.ndim == 2:
                    mask = mask[0]
                codes = codes[mask.to(dtype=torch.bool)]

            if codes.numel() == 0:
                continue

            # Find placeholder positions in this request's span
            s = int(q_start_list[i])
            e = int(q_start_list[i + 1])
            if e - s <= 1:
                continue  # Decode step, skip

            span_mask = placeholder_mask[s:e]
            placeholders = span_mask.nonzero(as_tuple=True)[0]
            n_codes = int(codes.shape[0])

            if int(placeholders.numel()) < n_codes:
                continue  # Mismatch

            # Embed delayed codes via fused multi-codebook embedding
            target = placeholders[:n_codes] + s
            codes_device = codes.to(device=hidden_states.device, dtype=torch.long)
            embeds = self.multimodal_embedding(codes_device)  # [n_codes, hidden]

            if new_hidden is None:
                new_hidden = hidden_states.clone()
            flat_hidden = new_hidden.reshape(-1, new_hidden.shape[-1])
            flat_hidden[target] = embeds.to(new_hidden.dtype)

        return new_hidden if new_hidden is not None else hidden_states

    # ------------------------------------------------------------------ audio feedback
    def _apply_audio_feedback(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Replace decode-step embeddings with audio feedback from pre-allocated buffers.

        CUDA-graph safe: reads from pre-allocated _decode_last_codes and
        _decode_has_codes tensors using torch.where (no Python dict lookup).
        The buffers are populated by sample() after each step.

        For decode steps (1 token per request): position i maps to row i.
        For prefill: audio feedback is not needed (ref audio substitution
        handles the prefill path separately).
        """
        bs = hidden_states.shape[0]

        # Ensure buffers are on the right device and large enough
        if self._decode_last_codes.device != hidden_states.device:
            self._decode_last_codes = self._decode_last_codes.to(hidden_states.device)
            self._decode_has_codes = self._decode_has_codes.to(hidden_states.device)
        if bs > self._decode_last_codes.shape[0]:
            old_size = self._decode_last_codes.shape[0]
            new_size = max(bs, old_size * 2)
            new_codes = torch.zeros(new_size, self.num_codebooks, dtype=torch.long, device=hidden_states.device)
            new_has = torch.zeros(new_size, dtype=torch.bool, device=hidden_states.device)
            new_codes[:old_size] = self._decode_last_codes
            new_has[:old_size] = self._decode_has_codes
            self._decode_last_codes = new_codes
            self._decode_has_codes = new_has

        # Compute audio embeddings from last_codes for ALL rows (graph-safe)
        codes_slice = self._decode_last_codes[:bs]  # [bs, N]
        has_codes = self._decode_has_codes[:bs].unsqueeze(-1)  # [bs, 1]
        audio_embeds = self.multimodal_embedding(codes_slice)  # [bs, D]
        audio_embeds = audio_embeds.to(dtype=hidden_states.dtype)

        # Select: where has_codes, use audio embed; else keep text embed
        return torch.where(has_codes, audio_embeds, hidden_states)

    # ------------------------------------------------------------------ sampling
    def sample(self, logits: torch.Tensor, sampling_metadata: Any) -> Any:
        """Model-owned sampler with delay-pattern audio dispatch.

        Mirrors v2's pattern: bias LM logits to force audio continuation,
        sample multi-codebook codes via the fused head, apply delay pattern,
        and accumulate per-request state.
        """
        self._resolve_token_ids()

        sampler = getattr(self, "_stock_sampler", None)
        if sampler is None:
            from vllm.v1.sample.sampler import Sampler

            sampler = Sampler()
            self._stock_sampler = sampler

        audio_id = self._audio_continuation_id
        num_codebooks = self.num_codebooks

        # Slot reuse detection: check output_token_ids length for each row.
        # If a slot's output length decreased (fresh request reusing a finished
        # slot), drop stale _audio_state for that row.
        output_ids = getattr(sampling_metadata, "output_token_ids", None)
        if output_ids is not None:
            for i in range(len(output_ids)):
                cur_len = len(output_ids[i]) if output_ids[i] else 0
                prev_len = self._slot_output_len.get(i, 0)
                if cur_len < prev_len:
                    # Fresh request reusing this slot — clear stale state
                    self._audio_state.pop(i, None)
                    self._slot_output_len[i] = 0
                self._slot_output_len[i] = cur_len

        # Bias LM logits for audio continuation
        self._apply_audio_mode_bias(logits, sampling_metadata)
        sampler_output = sampler(logits=logits, sampling_metadata=sampling_metadata)

        hidden = self._last_logits_hidden
        self._last_logits_hidden = None
        if hidden is None or audio_id is None:
            self._last_audio_codes = None
            return sampler_output

        sampled = getattr(sampler_output, "sampled_token_ids", None)
        if sampled is None:
            self._last_audio_codes = None
            return sampler_output
        sampled_flat = sampled.reshape(-1)
        if int(sampled_flat.numel()) != int(hidden.shape[0]):
            self._last_audio_codes = None
            return sampler_output

        is_audio = sampled_flat == audio_id

        # Handle first-after-<|audio|> transition: initialize state but do NOT
        # skip sampling. Unlike v2 (which seeds an all-BOC frame and skips the
        # first audio step), v3/sglang samples codebook 0 on the very first
        # step with delay_count=0 (codebooks 1-7 masked to BOC).
        first_after_start = self._last_first_audio_after_start
        self._last_first_audio_after_start = None

        if isinstance(first_after_start, torch.Tensor) and first_after_start.numel() == is_audio.shape[0]:
            first_after_start = first_after_start.to(is_audio.device)
            init_rows = first_after_start & is_audio
            if bool(init_rows.any()):
                boc_frame = torch.full((num_codebooks,), BOC_ID, dtype=torch.long, device=hidden.device)
                for bi in torch.nonzero(init_rows, as_tuple=False).reshape(-1).tolist():
                    bi = int(bi)
                    self._slot_output_len[bi] = 0
                    self._audio_state[bi] = {
                        "num_delay": 0,
                        "num_remaining_delays": None,
                        "audio_out_ids": None,
                        "last_codes": boc_frame.clone(),
                        "should_terminate": False,
                    }
                    # Update CUDA-graph-safe feedback buffers
                    self._decode_last_codes[bi] = boc_frame.to(self._decode_last_codes.device)
                    self._decode_has_codes[bi] = True
                # Do NOT remove init_rows from is_audio — let them be sampled

        if not bool(is_audio.any()):
            self._last_audio_codes = None
            return sampler_output

        audio_row_indices = torch.nonzero(is_audio, as_tuple=False).reshape(-1).tolist()

        # Per-codebook logits at audio positions
        cb_logits = self._audio_codebook_logits(hidden, is_audio)

        # Apply delay pattern masking BEFORE sampling
        self._apply_delay_pattern_masking(cb_logits, audio_row_indices)

        # Sample per-codebook
        cb_logits_2d = cb_logits.reshape(-1, cb_logits.shape[-1])
        codes_2d = self._sample_audio_codes(cb_logits_2d)
        codes_flat = codes_2d.view(cb_logits.shape[0], cb_logits.shape[1]).to(torch.long)

        # Update delay pattern state
        eos_stream = EOC_ID
        bos = BOC_ID
        new_codes_flat: list[torch.Tensor] = []
        for local_i, batch_i in enumerate(audio_row_indices):
            state = self._audio_state.setdefault(
                int(batch_i),
                {
                    "num_delay": 0,
                    "num_remaining_delays": None,
                    "audio_out_ids": None,
                    "last_codes": torch.full((num_codebooks,), bos, dtype=torch.long, device=hidden.device),
                    "should_terminate": False,
                },
            )
            num_delay: int = state["num_delay"]
            num_remaining_delays: int | None = state["num_remaining_delays"]
            this_codes = codes_flat[local_i].clone()

            # Leading delay-pattern BOS pad
            if num_delay + 1 < num_codebooks:
                this_codes[num_delay + 1 :] = bos
                num_delay += 1

            # Trailing eos ramp-down
            if num_remaining_delays is not None:
                this_codes[: num_codebooks - num_remaining_delays] = eos_stream
                num_remaining_delays -= 1
            else:
                eos_positions = (this_codes == eos_stream).nonzero(as_tuple=False).reshape(-1)
                if eos_positions.numel() > 0:
                    last_eos_idx = int(eos_positions[-1].item())
                    this_codes[: last_eos_idx + 1] = eos_stream
                    num_remaining_delays = num_codebooks - last_eos_idx - 1

            if num_remaining_delays is not None and num_remaining_delays <= 0:
                # Ramp-down complete — terminate without emitting this frame.
                state["num_delay"] = 0
                state["num_remaining_delays"] = None
                state["should_terminate"] = True
                new_codes_flat.append(torch.full_like(this_codes, -1))
                continue

            state["num_delay"] = num_delay
            state["num_remaining_delays"] = num_remaining_delays
            state["last_codes"] = this_codes.clone()
            # Update CUDA-graph-safe feedback buffers
            self._decode_last_codes[int(batch_i)] = this_codes.to(self._decode_last_codes.device)
            self._decode_has_codes[int(batch_i)] = True
            if state["audio_out_ids"] is None:
                state["audio_out_ids"] = this_codes.unsqueeze(-1).clone()
            else:
                state["audio_out_ids"] = torch.cat([state["audio_out_ids"], this_codes.unsqueeze(-1)], dim=-1)

            new_codes_flat.append(this_codes)

        # Build full codes tensor [batch_size, num_codebooks]
        # -1 marks "no audio code at this position"
        codes_full = torch.full(
            (int(sampled_flat.numel()), num_codebooks),
            -1,
            dtype=torch.long,
            device=hidden.device,
        )
        if new_codes_flat:
            stacked = torch.stack(new_codes_flat, dim=0).to(hidden.device)
            codes_full[is_audio] = stacked

        self._last_audio_codes = codes_full
        self._postprocess_cursor = 0
        return sampler_output

    # ------------------------------------------------------------------ postprocess
    def postprocess(
        self,
        hidden_states_slice: torch.Tensor,
        multimodal_outputs: Any = None,
        **req_infos: Any,
    ) -> dict[str, Any]:
        """Publish per-request audio codes into model_intermediate_buffer.

        Called once per request in batch order. Indexes _last_audio_codes
        by a running cursor (one row per request per step).
        """
        _ = multimodal_outputs
        codes_full = self._last_audio_codes
        if codes_full is None:
            return {}

        cursor = int(self._postprocess_cursor)
        if cursor >= int(codes_full.shape[0]):
            self._postprocess_cursor = 0
            return {}
        slice_codes = codes_full[cursor : cursor + 1]
        self._postprocess_cursor = cursor + 1

        # Drop placeholder rows (-1)
        audio_rows = slice_codes[:, 0] >= 0
        if not bool(audio_rows.any()):
            return {}
        new_codes = slice_codes[audio_rows].to(torch.int32)
        return {"codes": {"audio": new_codes}}

    # ------------------------------------------------------------------ helpers
    def _audio_codebook_logits(self, hidden_states: torch.Tensor, audio_mask: torch.Tensor) -> torch.Tensor:
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
        """Mask per-codebook logits according to delay pattern state, in-place.

        During delay phase: codebooks beyond delay_count only allow BOC.
        During ramp-down: locked codebooks only allow EOC.
        Normal generation: BOC disallowed; only cb0 allows EOC.
        """
        bos_pre = BOC_ID
        eos_pre = EOC_ID
        num_codebooks = self.num_codebooks
        for local_i, batch_i in enumerate(audio_row_indices):
            state = self._audio_state.get(int(batch_i))
            num_delay = int(state["num_delay"]) if state else 0
            num_rem = state.get("num_remaining_delays") if state else None

            if num_rem is not None:
                lock_until = num_codebooks - int(num_rem)
                for q in range(num_codebooks):
                    row = cb_logits[local_i, q]
                    if q < lock_until:
                        mask = torch.full_like(row, float("-inf"))
                        mask[eos_pre] = row[eos_pre]
                        cb_logits[local_i, q] = mask
                    else:
                        cb_logits[local_i, q, bos_pre] = float("-inf")
                        cb_logits[local_i, q, eos_pre] = float("-inf")
            else:
                for q in range(num_codebooks):
                    row = cb_logits[local_i, q]
                    if q > num_delay:
                        mask = torch.full_like(row, float("-inf"))
                        mask[bos_pre] = row[bos_pre]
                        cb_logits[local_i, q] = mask
                    else:
                        cb_logits[local_i, q, bos_pre] = float("-inf")
                        if q != 0:
                            cb_logits[local_i, q, eos_pre] = float("-inf")

    def _sample_audio_codes(self, logits_2d: torch.Tensor) -> torch.Tensor:
        """Replicate upstream sampling: temperature → top-k → top-p → multinomial."""
        x = logits_2d.float()
        top_k = 50
        top_p = 0.95
        if 0 < top_k < x.shape[-1]:
            kth = x.topk(top_k, dim=-1).values[..., -1:]
            x = x.masked_fill(x < kth, float("-inf"))
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = x.sort(dim=-1, descending=True)
            cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            sorted_mask = cumprobs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            mask = torch.zeros_like(x, dtype=torch.bool)
            mask.scatter_(-1, sorted_idx, sorted_mask)
            x = x.masked_fill(mask, float("-inf"))
        # Detect all-masked rows BEFORE softmax (softmax of all-inf yields NaN).
        has_finite = torch.isfinite(x).any(dim=-1)
        all_masked = ~has_finite
        if all_masked.any():
            # For all-masked rows, fall back to argmax (picks least-negative).
            fallback = x.argmax(dim=-1)
            if has_finite.any():
                probs = x[has_finite].softmax(dim=-1)
                sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
                result = fallback.clone()
                result[has_finite] = sampled
                return result
            return fallback
        probs = x.softmax(dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _apply_audio_mode_bias(self, logits: torch.Tensor, sampling_metadata: Any) -> None:
        """Detect <|audio|> transition, force continuation, force eos at ramp-down.

        Mirrors v2's _apply_audio_mode_bias: walks per-request to find the
        previous token, and if it was <|audio|> or the continuation token,
        forces the next emit to <|audio|>. On ramp-down completion, forces eos.
        """
        if logits is None or logits.ndim != 2:
            return

        audio_id = self._audio_continuation_id
        eos_id = self._eos_token_id
        if audio_id is None:
            return

        num_rows = int(logits.shape[0])
        prompt_ids = getattr(sampling_metadata, "prompt_token_ids", None)
        output_ids = getattr(sampling_metadata, "output_token_ids", None)

        # Fallback prev-token source from stashed input_ids
        stash_ids = self._last_step_input_ids
        stash_tail: list[int] | None = None
        if isinstance(stash_ids, torch.Tensor) and stash_ids.numel() > 0:
            q_start = self._last_step_query_start_loc
            if isinstance(q_start, torch.Tensor) and int(q_start.numel()) == num_rows + 1:
                q_start_cpu = q_start.detach().to("cpu").tolist()
                tail_idx = [max(0, int(q_start_cpu[i + 1]) - 1) for i in range(num_rows)]
                flat_ids = stash_ids.detach().to("cpu").tolist()
                stash_tail = [int(flat_ids[idx]) if idx < len(flat_ids) else -1 for idx in tail_idx]
            elif int(stash_ids.numel()) >= num_rows:
                stash_tail = stash_ids[-num_rows:].detach().to("cpu").tolist()

        first_after_start = torch.zeros(num_rows, dtype=torch.bool, device=logits.device)

        for i in range(num_rows):
            prev: int | None = None
            if output_ids is not None and i < len(output_ids):
                hist = output_ids[i]
                if hist:
                    prev = int(hist[-1])
            if prev is None and prompt_ids is not None:
                try:
                    p_i = prompt_ids[i]
                    if hasattr(p_i, "tolist"):
                        p_i = p_i.tolist()
                    if p_i:
                        prev = int(p_i[-1])
                except (IndexError, TypeError):
                    prev = None
            if prev is None and stash_tail is not None and i < len(stash_tail):
                prev = int(stash_tail[i])
            if prev is None:
                continue

            # Only bias if previous token was <|audio|> (the continuation token)
            if prev != audio_id:
                continue

            # Check if this is the FIRST step after <|audio|> appears
            # (i.e., transitioning from prompt to audio generation)
            audio_state = self._audio_state.get(i)
            if audio_state is None or audio_state.get("should_terminate"):
                # No state yet, or stale state from a finished prior request
                # reusing this slot — treat as first audio step
                self._audio_state.pop(i, None)
                # Clear CUDA-graph feedback buffers for this slot
                self._decode_has_codes[i] = False
                first_after_start[i] = True

            # Check for ramp-down termination
            should_terminate = bool(isinstance(audio_state, dict) and audio_state.get("should_terminate"))
            if should_terminate and eos_id is not None and 0 <= eos_id < int(logits.shape[-1]):
                row = logits[i]
                mask = torch.full_like(row, float("-inf"))
                mask[eos_id] = row[eos_id]
                logits[i].copy_(mask)
                audio_state["should_terminate"] = False
                continue

            # Force audio continuation token
            row = logits[i]
            mask = torch.full_like(row, float("-inf"))
            if 0 <= audio_id < row.shape[-1]:
                mask[audio_id] = row[audio_id]
            logits[i].copy_(mask)

        self._last_first_audio_after_start = first_after_start

    # ------------------------------------------------------------------ omni output
    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
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
                multimodal_outputs={"codes": {"audio": audio_codes_list}},
            )
        return OmniOutput(text_hidden_states=hidden, multimodal_outputs=None)

    # ------------------------------------------------------------------ weight loading

    # Per-layer suffixes from the actual V3 checkpoint (results/higgs_v3_checkpoint_analysis.txt)
    _V3_LAYER_SUFFIXES = (
        "input_layernorm.weight",
        "mlp.down_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "post_attention_layernorm.weight",
        "self_attn.k_norm.weight",
        "self_attn.k_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.q_proj.weight",
        "self_attn.v_proj.weight",
    )

    @classmethod
    def _build_required_keys(cls, num_layers: int) -> set[str]:
        """Build the exact set of required V3 checkpoint keys."""
        keys = {
            "tied.embedding.text_embedding.weight",
            "body.norm.weight",
            f"{_MODALITY_EMBEDDING_PREFIX}weight",
        }
        for i in range(num_layers):
            for suffix in cls._V3_LAYER_SUFFIXES:
                keys.add(f"body.layers.{i}.{suffix}")
        return keys

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        backbone_weights: list[tuple[str, torch.Tensor]] = []
        loaded_params: set[str] = set()
        own_params = dict(self.named_parameters())
        seen_checkpoint_keys: set[str] = set()

        for name, tensor in weights:
            seen_checkpoint_keys.add(name)

            mapped = self._map_weight_name(name)
            if mapped is None:
                continue

            if mapped.startswith("model.") or mapped.startswith("lm_head."):
                backbone_weights.append((mapped, tensor))
            elif mapped in own_params:
                param = own_params[mapped]
                if param.shape != tensor.shape:
                    raise ValueError(
                        f"Shape mismatch for {mapped}: expected {param.shape}, "
                        f"got {tensor.shape} (checkpoint key: {name})"
                    )
                param.data.copy_(tensor.to(param.dtype))
                loaded_params.add(mapped)

        if backbone_weights:
            backbone_module = _BackboneWrapper(self.model, self.lm_head, self._backbone_config)
            loaded = backbone_module.load_weights(iter(backbone_weights))
            loaded_params.update(loaded)

        # Resolve special token IDs from the tokenizer
        model_path = getattr(self.vllm_config.model_config, "model", None)
        if model_path:
            self.config.resolve_special_tokens(model_path)
        self._resolve_token_ids()

        # Verify every required checkpoint key was seen.
        num_layers = int(self._backbone_config.num_hidden_layers)
        required = self._build_required_keys(num_layers)
        missing = required - seen_checkpoint_keys
        if missing:
            raise RuntimeError(
                f"HiggsAudioV3Talker: {len(missing)} required checkpoint keys missing: {sorted(missing)[:5]}..."
            )

        logger.info(
            "HiggsAudioV3Talker: loaded %d params, modality_embedding=%s, tied=%s",
            len(loaded_params),
            tuple(self.multimodal_embedding.weight.shape),
            self.tie_modality,
        )
        return loaded_params

    def _map_weight_name(self, name: str) -> str | None:
        if name.startswith(_CODEC_PREFIX):
            return None
        if name.startswith(_MODALITY_HEAD_PREFIX):
            if self.tie_modality:
                return None
            return name.replace(_MODALITY_HEAD_PREFIX, "modality_head.")
        if name.startswith(_MODALITY_EMBEDDING_PREFIX):
            return name.replace(_MODALITY_EMBEDDING_PREFIX, "multimodal_embedding.")
        for ckpt_prefix, model_prefix in _BACKBONE_PREFIX_MAP.items():
            if name.startswith(ckpt_prefix):
                return name.replace(ckpt_prefix, model_prefix, 1)
        # Reject unexpected non-codec Higgs checkpoint keys
        raise ValueError(
            f"Unexpected checkpoint key with no known mapping: {name!r}. "
            f"Known prefixes: {list(_BACKBONE_PREFIX_MAP.keys())}, "
            f"{_MODALITY_EMBEDDING_PREFIX!r}, {_MODALITY_HEAD_PREFIX!r}, {_CODEC_PREFIX!r}"
        )


class _BackboneWrapper(nn.Module):
    """Wrapper to use AutoWeightsLoader for Qwen3 backbone."""

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
        from vllm.model_executor.models.utils import AutoWeightsLoader

        skip = ["lm_head."] if getattr(self.config, "tie_word_embeddings", False) else None
        loader = AutoWeightsLoader(self, skip_prefixes=skip)
        return loader.load_weights(weights)
