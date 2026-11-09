# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage-0 talker for higgs-audio v2 (vLLM-native, DualFFN-aware).

Architecture summary (verified against `transformers.models.higgs_audio_v2`
source and the boson-ai checkpoint config.json):

- Backbone: Llama-3.2-3B (hidden=3072, 28 layers, GQA 24Q/8KV, head_dim=128,
  vocab=128256, max_position_embeddings=2048, RoPE llama3 scaling factor=32).
  We reuse vLLM's compiled :class:`vllm.model_executor.models.llama.LlamaModel`
  for the attention path so PagedAttention scheduling stays intact.

- DualFFN: every transformer block carries a parallel audio expert
  (``audio_input_layernorm`` + ``audio_post_attention_layernorm`` + ``audio_mlp``)
  next to the standard text path (``input_layernorm`` + ``post_attention_layernorm``
  + ``mlp``). A per-position ``audio_token_mask`` of shape ``[B, S]`` selects
  between the two. The mask is the union of positions where the input token id
  equals ``audio_token_id=128016`` or ``audio_delay_token_id=128014``.

- Multi-codebook output head: at audio positions, Stage 0 emits one ID per
  codebook (8 codebooks of vocabulary 1026 each) in parallel via a single
  fused ``audio_lm_head: Linear(hidden, num_codebooks * codebook_size)``
  whose output is reshaped to ``[N_audio, num_codebooks, codebook_size]``.

- Delay pattern: codebook k is staggered by k frames using
  ``audio_delay_token_id=128014`` as a filler at the LM-token positions where
  codebook k has not yet started emitting real codes. This matches
  ``HiggsAudioV2DelayPatternLogitsProcessor`` upstream and is the canonical
  MusicGen pattern ``[0, 1, 2, 3, 4, 5, 6, 7]``.

This file delivers the structural pieces (module classes, weight mapping,
forward pass, AR sampler dispatch, and the upstream delay-pattern logits
processor) to reproduce the upstream behavior under vLLM's compiled forward
path.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaMLP

from vllm_omni.model_executor.models.higgs_audio_v2.configuration_higgs_audio_v2 import (
    HiggsAudioV2Config,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

__all__ = [
    "HiggsAudioV2DecoderLayer",
    "HiggsAudioV2TalkerForConditionalGeneration",
]

logger = init_logger(__name__)


class HiggsAudioRMSNorm(nn.Module):
    """Minimal RMSNorm matching the upstream HiggsAudioV2RMSNorm semantics.

    Kept self-contained so the parameter names (``weight``) line up with the
    upstream state dict for direct load.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class HiggsAudioV2DecoderLayer(LlamaDecoderLayer):
    """One transformer block: vLLM-native attention + DualFFN-routed MLPs.

    Wraps a stock :class:`vllm.model_executor.models.llama.LlamaDecoderLayer`
    for the self-attention path (so PagedAttention KV caches keep working) and
    overlays the upstream HiggsAudioV2 DualFFN routing for the layernorm + MLP
    pairs. The implementation follows the upstream rule from
    ``transformers.models.higgs_audio_v2.HiggsAudioV2DecoderLayer.forward``:

    1. Pre-attention norm: per-position split into ``audio_input_layernorm``
       (audio mask True) vs ``input_layernorm`` (audio mask False). The mixed
       output is fed to a single shared self-attention.
    2. Post-attention residual + dual MLP: text positions go through
       ``mlp(post_attention_layernorm(.))`` and audio positions go through
       ``audio_mlp(audio_post_attention_layernorm(.))``. Both deltas are added
       to the residual.

    The classical (non-fused-residual) transformer pattern is used so the per-
    position split is straightforward. vLLM's compiled forward path passes
    ``audio_token_mask`` through as an extra layer kwarg via
    :class:`HiggsAudioV2TalkerForConditionalGeneration.forward`.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Subclass LlamaDecoderLayer so self_attn registers at the
        # CANONICAL ``layers.{i}.self_attn.attn`` prefix (not nested under
        # ``.base``). This is the architecturally-correct registration path
        # for vLLM's compilation_config.static_forward_context.
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        hidden_size = int(config.hidden_size)
        intermediate_size = int(config.intermediate_size)
        hidden_act = str(config.hidden_act)
        mlp_bias = bool(getattr(config, "mlp_bias", False))
        rms_norm_eps = float(config.rms_norm_eps)
        # Parallel audio-expert norms + MLP (mirrors upstream parameter names).
        self.audio_input_layernorm = HiggsAudioRMSNorm(hidden_size, eps=rms_norm_eps)
        self.audio_post_attention_layernorm = HiggsAudioRMSNorm(hidden_size, eps=rms_norm_eps)
        self.audio_mlp = LlamaMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            bias=mlp_bias,
            prefix=f"{prefix}.audio_mlp",
        )

    # ------------------------------------------------------------------ helpers
    def _routed_norm(
        self,
        hidden: torch.Tensor,
        text_norm: nn.Module,
        audio_norm: HiggsAudioRMSNorm,
        audio_mask: torch.Tensor | None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-position routed RMSNorm with residual handling.

        Matches the vLLM fused-residual pattern: takes (hidden, residual) and
        returns (normed_hidden, post_add_residual). Routes per position:
        audio positions through ``audio_norm`` (HiggsAudioRMSNorm, plain RMS),
        text positions through ``text_norm`` (vLLM compiled RMSNorm).

        For consistency we explicitly compute the residual sum once, then call
        the vLLM text_norm on text positions WITHOUT its residual argument
        (else it'd add the residual again).
        """
        # Fold residual into hidden once.
        if residual is not None:
            hidden = hidden + residual
        new_residual = hidden
        if audio_mask is None:
            return audio_norm(hidden), new_residual
        mask_flat = audio_mask.reshape(-1)
        if hidden.ndim == 3:
            hidden_flat = hidden.reshape(-1, hidden.shape[-1])
        else:
            hidden_flat = hidden
        out = hidden_flat.clone()
        if (~mask_flat).any():
            text_out = text_norm(hidden_flat[~mask_flat])
            # vLLM compiled RMSNorm.forward(x) returns Tensor; some variants
            # return (Tensor, residual). Normalize to Tensor.
            if isinstance(text_out, tuple):
                text_out = text_out[0]
            out[~mask_flat] = text_out.to(out.dtype)
        if mask_flat.any():
            audio_out = audio_norm(hidden_flat[mask_flat])
            out[mask_flat] = audio_out.to(out.dtype)
        return out.reshape_as(hidden), new_residual

    def _routed_mlp(
        self,
        hidden: torch.Tensor,
        audio_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply ``audio_mlp`` to mask positions and ``mlp`` to non-mask positions."""
        if audio_mask is None:
            return self.audio_mlp(hidden)
        mask_flat = audio_mask.reshape(-1)
        if hidden.ndim == 3:
            hidden_flat = hidden.reshape(-1, hidden.shape[-1])
        else:
            hidden_flat = hidden
        out = torch.zeros_like(hidden_flat)
        if (~mask_flat).any():
            out[~mask_flat] = self.mlp(hidden_flat[~mask_flat]).to(out.dtype)
        if mask_flat.any():
            out[mask_flat] = self.audio_mlp(hidden_flat[mask_flat]).to(out.dtype)
        return out.reshape_as(hidden)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        *,
        audio_token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Fused-residual pattern (compatible with vLLM's LlamaModel forward
        # loop which calls ``hidden, residual = self.norm(hidden, residual)``
        # at the end).
        attn_input, residual = self._routed_norm(
            hidden_states,
            self.input_layernorm,
            self.audio_input_layernorm,
            audio_token_mask,
            residual,
        )
        attn_out = self.self_attn(positions=positions, hidden_states=attn_input)

        mlp_input, residual = self._routed_norm(
            attn_out,
            self.post_attention_layernorm,
            self.audio_post_attention_layernorm,
            audio_token_mask,
            residual,
        )
        mlp_out = self._routed_mlp(mlp_input, audio_token_mask)
        return mlp_out, residual


class HiggsAudioV2TalkerForConditionalGeneration(nn.Module):
    """Stage-0 talker class registered under
    ``HiggsAudioV2ForConditionalGeneration`` (canonical HF arch identifier)
    AND ``HiggsAudioV2TalkerForConditionalGeneration`` (explicit alias).

    The class wires together:
      1. vLLM-native ``LlamaModel`` backbone (PagedAttention scheduling).
      2. A parallel set of DualFFN modules (one per decoder layer) for the
         audio-expert path.
      3. Multi-codebook output via a single fused ``audio_lm_head`` of shape
         ``[num_codebooks * codebook_size, hidden]`` (all codebooks emitted
         in parallel per audio position).
      4. ``load_weights`` HF -> vLLM mapping (GQA reshape, llama3 RoPE,
         text/audio MLP split, audio code heads, audio token embeddings).

    The forward path interleaves DualFFN routing with vLLM's compiled
    attention path so the talker integrates with the rest of the wiring
    (pipeline_registry, serving_speech, deploy yaml, stage_input_processor).
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
        if isinstance(hf_config, HiggsAudioV2Config):
            self.config: HiggsAudioV2Config = hf_config
        else:
            # When loaded via AutoConfig the class is the upstream
            # transformers config; wrap it in our typed shell preserving
            # every attribute via PretrainedConfig.
            self.config = HiggsAudioV2Config(**hf_config.to_dict())

        # ------------------------------------------------------------------ embedding + final norm
        # directly instantiate the text-side embedding and final norm
        # so we don't pull in a full LlamaModel.layers list that the engine's
        # weight loader would require initialized. Each HiggsAudioV2DecoderLayer
        # below already owns its own LlamaDecoderLayer for the attention path;
        # duplicating those via a parallel LlamaModel made the loader fail.
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.hidden_size,
            org_num_embeddings=self.config.vocab_size,
            quant_config=None,
            prefix=f"{prefix}.embed_tokens",
        )
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

        # ------------------------------------------------------------------ DualFFN-aware layer stack
        # use vLLM's ``make_layers`` helper to register attention layers
        # at canonical ``layers.{i}.self_attn.attn`` paths in
        # ``compilation_config.static_forward_context``. This is what gets
        # picked up by ``bind_kv_cache`` so prefill writes K/V to the cache
        # (root cause for empty cache + zero attention output).
        from vllm.model_executor.models.utils import make_layers

        _, _, self.layers = make_layers(
            self.config.num_hidden_layers,
            lambda prefix: HiggsAudioV2DecoderLayer(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        # ------------------------------------------------------------------ audio embedding (shared)
        # Audio frames are embedded by summing per-codebook lookups; the LM
        # input_ids stream uses ``audio_token_id`` / ``audio_delay_token_id``
        # placeholders at the corresponding positions, and the talker
        # substitutes the audio embedding via the audio_codebook_embeddings
        # table declared further below.
        self.register_buffer(
            "audio_tokens_offsets",
            torch.arange(self.config.num_codebooks) * self.config.codebook_size,
            persistent=False,
        )

        # ------------------------------------------------------------------ heads
        # Standard text LM head (Llama 128256-wide vocab).
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            org_num_embeddings=self.config.vocab_size,
            quant_config=None,
            prefix=f"{prefix}.lm_head",
        )

        # Audio side: the upstream ``audio_decoder_proj.audio_lm_head`` is a
        # single fused ``Linear(hidden, num_codebooks * (codebook_size + 2))``.
        # In our v2 config ``codebook_size = 1026`` ALREADY includes the two
        # stream specials, so the fused head is shape ``(8 * 1026, 3072) =
        # (8208, 3072)``. At sample time we run it once and reshape
        # ``[N, 8208] -> [N, 8, 1026]`` to read each codebook's logits.
        self.audio_lm_head = nn.Linear(
            self.config.hidden_size,
            self.config.num_codebooks * self.config.codebook_size,
            bias=False,
        )
        # Multi-codebook input embedding used both for prompt-side audio frame
        # substitution and for next-step audio-input-embedding feedback during
        # decode. Mirrors upstream ``HiggsAudioModel.audio_codebook_embeddings``.
        self.audio_codebook_embeddings = nn.Embedding(
            self.config.num_codebooks * self.config.codebook_size,
            self.config.hidden_size,
        )

        # vLLM's logits processor wires sampling metadata into the lm_head.
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    # ----------------------------------------------------------------- masks
    @torch.inference_mode()
    def audio_token_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute the per-position audio_token_mask used by DualFFN routing.

        Matches ``HiggsAudioV2Model.get_placeholder_mask``:

            mask = (input_ids == audio_token_id) | (input_ids == audio_delay_token_id)
        """
        return (input_ids == self.config.audio_token_id) | (input_ids == self.config.audio_delay_token_id)

    # ------------------------------------------------------------ ref-audio
    @torch.inference_mode()
    def _maybe_apply_ref_audio_substitution(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        info_dicts: Any,
    ) -> torch.Tensor:
        """Substitute reference-audio embeddings at prompt-side audio placeholders.

        Voice-clone path: the serving layer threads per-request
        ``audio_input_ids`` (shape ``[T_frames, num_codebooks]``) +
        ``audio_input_ids_mask`` (``[T_frames]``) into
        ``model_intermediate_buffer`` (one dict per request in batch order).
        For each request that has those tensors, we look up its span in the
        flat ``input_ids`` via ``query_start_loc`` from the forward context,
        embed the codes through ``audio_codebook_embeddings`` (with the
        per-codebook ``audio_tokens_offsets`` shift, mirroring
        ``HiggsAudioV2Embeddings.forward`` upstream), and overwrite the
        text-side embedding at every audio_token / audio_delay placeholder
        position in that span.

        Decode steps (span length 1) are skipped here — the audio-feedback
        path in ``_maybe_apply_audio_feedback`` takes over from the moment
        the talker emits its first codebook frame via ``sample()``.
        """
        if not info_dicts:
            return hidden_states

        # Pull the per-request (codes, mask) tensors in batch order. Values
        # may be on CPU after IPC serialization (msgspec ships tensors as raw
        # bytes); the loop below moves them to the input_ids device on use.
        per_req_codes: list[torch.Tensor | None] = []
        any_codes = False
        for info in info_dicts:
            if not isinstance(info, dict):
                per_req_codes.append(None)
                continue
            ids = info.get("audio_input_ids")
            mask = info.get("audio_input_ids_mask")
            if isinstance(ids, list):
                ids = ids[0] if ids else None
            if isinstance(mask, list):
                mask = mask[0] if mask else None
            if not isinstance(ids, torch.Tensor) or not isinstance(mask, torch.Tensor):
                per_req_codes.append(None)
                continue
            if ids.ndim == 3:
                ids = ids[0]
            if mask.ndim == 2:
                mask = mask[0]
            valid = ids[mask.to(dtype=torch.bool)]  # [T_valid, num_codebooks]
            if valid.numel() == 0:
                per_req_codes.append(None)
                continue
            per_req_codes.append(valid)
            any_codes = True

        if not any_codes:
            return hidden_states

        # Per-request slices in the flat input_ids are recovered from the
        # forward context's attention metadata (mimo_audio_llm.py:1022-1025
        # uses the same pattern). Without spans we cannot route codes to the
        # right placeholder positions, so we no-op instead of risking a
        # cross-request mix-up.
        from vllm.forward_context import get_forward_context

        try:
            attn_metadata = get_forward_context().attn_metadata
        except Exception:
            return hidden_states
        if isinstance(attn_metadata, dict):
            if not attn_metadata:
                return hidden_states
            attn = next(iter(attn_metadata.values()))
        else:
            attn = attn_metadata
        q_start = getattr(attn, "query_start_loc", None)
        if not isinstance(q_start, torch.Tensor):
            return hidden_states
        q_start_cpu = q_start.detach().to("cpu").tolist()
        if len(q_start_cpu) != len(info_dicts) + 1:
            return hidden_states

        audio_token_id = int(self.config.audio_token_id)
        audio_delay_id = int(self.config.audio_delay_token_id)
        offsets = self.audio_tokens_offsets.to(input_ids.device)

        new_hidden: torch.Tensor | None = None
        for i, codes in enumerate(per_req_codes):
            if codes is None:
                continue
            s = int(q_start_cpu[i])
            e = int(q_start_cpu[i + 1])
            if e - s <= 1:
                # Decode step — let _maybe_apply_audio_feedback handle this row.
                continue
            slice_ids = input_ids[s:e]
            mask = (slice_ids == audio_token_id) | (slice_ids == audio_delay_id)
            placeholders = mask.nonzero(as_tuple=True)[0]
            n_real = int(codes.shape[0])
            if int(placeholders.numel()) < n_real:
                # Mismatch (likely a truncated prompt): skip this request rather
                # than corrupt the batch. The user-visible failure surfaces as
                # the model not honoring the ref-audio rather than a crash.
                continue
            target = placeholders[:n_real] + s
            embeds = self.audio_codebook_embeddings(codes.to(device=input_ids.device, dtype=torch.long) + offsets).sum(
                dim=-2
            )
            if new_hidden is None:
                new_hidden = hidden_states.clone()
            new_hidden[target] = embeds.to(new_hidden.dtype)

        return new_hidden if new_hidden is not None else hidden_states

    # ----------------------------------------------------------------- weights
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Map HF state dict names to the vLLM-native layout.

        The upstream tensor names live under ``model.layers.<L>...`` plus a few
        top-level entries. The vLLM-native layout uses fused QKV/MLP tensors,
        so we transform the relevant pairs at load time:

          * Attention: HF ``q_proj/k_proj/v_proj`` -> vLLM ``qkv_proj`` (packed
            ``[hidden + 2 * kv_dim, hidden]``; for GQA with 24 Q heads, 8 KV
            heads, head_dim=128 the slabs are
            ``q_proj[3072, 3072]``, ``k_proj[1024, 3072]``, ``v_proj[1024, 3072]``
            -> ``qkv_proj[5120, 3072]``).
          * RoPE parameters are read from ``self.config.rope_parameters`` and
            consumed by vLLM's LlamaAttention without any weight transform
            (``rope_type="llama3"``, factor=32, low_freq_factor=0.125,
            high_freq_factor=0.5, original_max_position_embeddings=1024).
          * Text MLP: HF ``mlp.gate_proj`` + ``mlp.up_proj`` ->
            vLLM ``mlp.gate_up_proj`` (concatenated on dim 0).
          * Audio MLP: HF ``audio_mlp.gate_proj`` + ``audio_mlp.up_proj`` ->
            vLLM ``dual_ffns.<L>.audio_mlp.gate_up_proj`` (same shape transform).
          * Layer norms: HF ``input_layernorm`` / ``post_attention_layernorm``
            -> ``dual_ffns.<L>.{input,post_attention}_layernorm``. Audio norms
            -> ``dual_ffns.<L>.{audio_input,audio_post_attention}_layernorm``.
          * Audio token embedding: HF ``model.embed_audio_tokens.weight`` (with
            optional outer ``embed_audio_tokens.`` prefix from upstream's
            HiggsAudioV2Embeddings) -> ``self.audio_codebook_embeddings.weight``
            (see ``_map_simple_name``).
          * Audio codebook heads: HF ``audio_lm_head.weight`` -> the fused 8x1026
            head from which Stage 0 reads codebook 0 directly and codebook 1..7
            via the fast-AR code predictor's residual heads. (Boson-ai's actual
            checkpoint exposes a single fused ``audio_lm_head`` of shape
            ``[num_codebooks * codebook_size, hidden]``, not one head per
            codebook; we split it into the per-codebook heads at load time.)

        this routine covers the simple, fused-QKV, and fused-MLP
        transcriptions. The remaining gap (full Stage-0 forward integration with
        vLLM's PagedAttention scheduler so the loaded weights drive a working
        greedy decode) is tracked as an Open Issue in the goal tracker.
        """
        loaded: set[str] = set()
        own_params = dict(self.named_parameters())
        # Debug: log the first few HF keys we receive so we can verify the
        # mapper hits them. Also log our own param-name shape so missing
        # mappings are obvious.
        logger.info(
            "higgs_audio_v2 load_weights: %d own params (sample: %s)",
            len(own_params),
            sorted(list(own_params.keys()))[:5],
        )
        debug_seen: list[str] = []

        # First pass: collect tensors that need fusing (q/k/v, gate/up for each
        # MLP slot). We accumulate into per-(layer, slot, kind) buckets and
        # commit the fused tensor once all parts have arrived.
        fuse_buckets: dict[tuple[int, str, str], dict[str, torch.Tensor]] = {}

        def _strip_model_prefix(name: str) -> str:
            """vLLM's DefaultModelLoader strips the leading ``model.`` from HF
            state-dict keys before invoking ``load_weights``, but we also want
            to accept raw HF names (for unit tests that pass the state dict
            directly). Normalize by stripping at most one ``model.`` prefix.
            """
            if name.startswith("model."):
                return name[len("model.") :]
            return name

        def _try_simple(name: str, tensor: torch.Tensor) -> bool:
            normalized = _strip_model_prefix(name)
            # Direct match: many keys (embed_tokens.weight, norm.weight,
            # layers.<L>.{audio_,}input_layernorm.weight, layers.<L>.{audio_,}post_attention_layernorm.weight,
            # layers.<L>.self_attn.o_proj.weight, layers.<L>.{,audio_}mlp.down_proj.weight) already match
            # our named_parameters() keys after the model-prefix strip.
            if normalized in own_params:
                target = own_params[normalized]
                if tuple(target.shape) == tuple(tensor.shape):
                    with torch.no_grad():
                        target.copy_(tensor)
                    loaded.add(normalized)
                    return True
                logger.warning(
                    "higgs_audio_v2 load_weights: shape mismatch %s -> %s: %s vs %s",
                    name,
                    normalized,
                    tuple(tensor.shape),
                    tuple(target.shape),
                )
                return False

            # After making HiggsAudioV2DecoderLayer inherit from
            # LlamaDecoderLayer, text-side params live directly at
            # ``layers.<L>.<tail>`` (no ``.base`` indirection). audio side
            # remains at ``layers.<L>.audio_<...>``.
            parts = normalized.split(".")
            if len(parts) >= 3 and parts[0] == "layers":
                tail = ".".join(parts[2:])
                # Audio_mlp.down_proj lives at our top-level audio_mlp slot.
                if tail == "audio_mlp.down_proj.weight":
                    audio_name = f"layers.{parts[1]}.audio_mlp.down_proj.weight"
                    if audio_name in own_params:
                        target = own_params[audio_name]
                        if tuple(target.shape) == tuple(tensor.shape):
                            with torch.no_grad():
                                target.copy_(tensor)
                            loaded.add(audio_name)
                            return True

            # Special simple cases (nested audio embedding, text_lm_head alias, etc.)
            mapped = self._map_simple_name(name)
            if mapped is None:
                # Also try after stripping the model. prefix so the _map_simple_name
                # branches written for un-stripped names continue to work.
                mapped = self._map_simple_name("model." + normalized)
            if mapped is None or mapped not in own_params:
                return False
            target = own_params[mapped]
            if tuple(target.shape) != tuple(tensor.shape):
                logger.warning(
                    "higgs_audio_v2 load_weights: shape mismatch %s -> %s: %s vs %s",
                    name,
                    mapped,
                    tuple(tensor.shape),
                    tuple(target.shape),
                )
                return False
            with torch.no_grad():
                target.copy_(tensor)
            loaded.add(mapped)
            return True

        def _stash_fusion(name: str, tensor: torch.Tensor) -> bool:
            normalized = _strip_model_prefix(name)
            parts = normalized.split(".")
            if len(parts) < 4 or parts[0] != "layers":
                return False
            try:
                layer_idx = int(parts[1])
            except ValueError:
                return False
            tail = ".".join(parts[2:])
            for slot in ("mlp", "audio_mlp"):
                for kind in ("gate_proj", "up_proj"):
                    if tail == f"{slot}.{kind}.weight":
                        fuse_buckets.setdefault((layer_idx, slot, "gate_up_proj"), {})[kind] = tensor
                        return True
            for kind in ("q_proj", "k_proj", "v_proj"):
                if tail == f"self_attn.{kind}.weight":
                    fuse_buckets.setdefault((layer_idx, "self_attn", "qkv_proj"), {})[kind] = tensor
                    return True
            return False

        unhandled: list[str] = []
        for name, tensor in weights:
            if len(debug_seen) < 30:
                debug_seen.append(name)
            if _try_simple(name, tensor):
                continue
            if _stash_fusion(name, tensor):
                continue
            # Audio code head: upstream has a fused [num_codebooks*codebook_size, hidden]
            # head; split into codebook-0 head + residual heads. Accept both
            # ``audio_lm_head.weight`` and ``model.audio_lm_head.weight``.
            if name in (
                "audio_lm_head.weight",
                "model.audio_lm_head.weight",
                "audio_decoder_proj.audio_lm_head.weight",
                "model.audio_decoder_proj.audio_lm_head.weight",
            ):
                self._consume_fused_audio_head(tensor, loaded)
                continue
            unhandled.append(name)

        # Second pass: emit fused tensors. Targets are the per-layer slots on
        # the HiggsAudioV2DecoderLayer stack: text-side params at
        # ``layers.<L>.<tail>`` directly (HiggsAudioV2DecoderLayer inherits
        # from LlamaDecoderLayer); ``layers.<L>.audio_mlp...`` for the audio
        # expert.
        for (layer_idx, slot, fused_kind), parts in fuse_buckets.items():
            if fused_kind == "gate_up_proj":
                gate = parts.get("gate_proj")
                up = parts.get("up_proj")
                if gate is None or up is None:
                    continue
                fused = torch.cat([gate, up], dim=0)
                if slot == "mlp":
                    target_name = f"layers.{layer_idx}.mlp.gate_up_proj.weight"
                else:  # audio_mlp
                    target_name = f"layers.{layer_idx}.audio_mlp.gate_up_proj.weight"
            elif fused_kind == "qkv_proj":
                q = parts.get("q_proj")
                k = parts.get("k_proj")
                v = parts.get("v_proj")
                if q is None or k is None or v is None:
                    continue
                fused = torch.cat([q, k, v], dim=0)
                target_name = f"layers.{layer_idx}.self_attn.qkv_proj.weight"
            else:
                continue
            if target_name in own_params and tuple(own_params[target_name].shape) == tuple(fused.shape):
                with torch.no_grad():
                    own_params[target_name].copy_(fused)
                loaded.add(target_name)
            else:
                logger.warning(
                    "higgs_audio_v2 load_weights: fused %s not found or shape mismatch (%s vs target %s)",
                    target_name,
                    tuple(fused.shape),
                    tuple(own_params[target_name].shape) if target_name in own_params else "<missing>",
                )
        logger.info(
            "higgs_audio_v2 load_weights: %d/%d params initialized "
            "(sample seen: %s; unhandled count=%d, sample unhandled=%s)",
            len(loaded),
            len(own_params),
            debug_seen[:10],
            len(unhandled),
            unhandled[:20],
        )
        return loaded

    def _consume_fused_audio_head(self, fused: torch.Tensor, loaded: set[str]) -> None:
        """Load the fused ``audio_lm_head[num_codebooks*codebook_size, hidden]``
        directly into ``self.audio_lm_head.weight``.

        Upstream uses a single big head; there is no separate codebook-0
        head or residual predictor in the actual checkpoint.
        """
        own_params = dict(self.named_parameters())
        num_codebooks = int(self.config.num_codebooks)
        codebook_size = int(self.config.codebook_size)
        hidden = int(self.config.hidden_size)
        target = own_params.get("audio_lm_head.weight")
        if target is None:
            logger.warning("higgs_audio_v2: no audio_lm_head.weight slot to load into")
            return
        expected = (num_codebooks * codebook_size, hidden)
        if tuple(fused.shape) != expected:
            logger.warning(
                "higgs_audio_v2: unexpected audio_lm_head shape %s; expected %s",
                tuple(fused.shape),
                expected,
            )
            return
        with torch.no_grad():
            target.copy_(fused)
        loaded.add("audio_lm_head.weight")

    # --------------------------------------------------------------- helpers
    def _map_simple_name(self, hf_name: str) -> str | None:
        """Translate the simple (non-fused) HF parameter names to vLLM names."""
        # Upstream nests the audio embedding as
        # ``model.embed_audio_tokens.embed_audio_tokens.weight``; the boson-ai
        # checkpoint also uses this exact key. Accept both nested and flat.
        # the upstream multi-codebook embedding is ``audio_codebook_embeddings``
        # (a flat ``Embedding(8 * codebook_size, hidden)``). also had a separate
        # ``embed_audio_tokens`` (used in prefill-time DualFFN routing helpers);
        # we keep that slot so the existing prompt-side embedding logic stays
        # functional. Both slots are loaded from the same upstream tensor.
        if hf_name in (
            "embed_audio_tokens.weight",
            "embed_audio_tokens.embed_audio_tokens.weight",  # stripped form
            "model.embed_audio_tokens.embed_audio_tokens.weight",  # raw HF form
            "model.embed_audio_tokens.weight",
            # Actual key used in the boson-ai checkpoint:
            "audio_codebook_embeddings.weight",
            "model.audio_codebook_embeddings.weight",
        ):
            return "audio_codebook_embeddings.weight"
        if hf_name in (
            "lm_head.weight",
            "text_lm_head.weight",
            "model.text_lm_head.weight",
            # Actual key used in the boson-ai checkpoint:
            "audio_decoder_proj.text_lm_head.weight",
            "model.audio_decoder_proj.text_lm_head.weight",
        ):
            return "lm_head.weight"
        # Dropped LlamaModel wrapper; text embedding + final norm now
        # live directly on the talker (self.embed_tokens / self.norm).
        if hf_name == "model.embed_tokens.weight":
            return "embed_tokens.weight"
        if hf_name == "model.norm.weight":
            return "norm.weight"
        # Per-layer audio norms -> our HiggsAudioV2DecoderLayer side.
        if hf_name.startswith("model.layers.") and hf_name.endswith(".audio_input_layernorm.weight"):
            layer_idx = hf_name.split(".")[2]
            return f"layers.{layer_idx}.audio_input_layernorm.weight"
        if hf_name.startswith("model.layers.") and hf_name.endswith(".audio_post_attention_layernorm.weight"):
            layer_idx = hf_name.split(".")[2]
            return f"layers.{layer_idx}.audio_post_attention_layernorm.weight"
        # Per-layer text-side params now live directly at
        # ``layers.{i}.<tail>`` (no ``.base`` indirection) since
        # HiggsAudioV2DecoderLayer inherits from LlamaDecoderLayer.
        if hf_name.startswith("model.layers.") and hf_name.endswith(".input_layernorm.weight"):
            layer_idx = hf_name.split(".")[2]
            return f"layers.{layer_idx}.input_layernorm.weight"
        if hf_name.startswith("model.layers.") and hf_name.endswith(".post_attention_layernorm.weight"):
            layer_idx = hf_name.split(".")[2]
            return f"layers.{layer_idx}.post_attention_layernorm.weight"
        if hf_name.startswith("model.layers.") and hf_name.endswith(".self_attn.o_proj.weight"):
            layer_idx = hf_name.split(".")[2]
            return f"layers.{layer_idx}.self_attn.o_proj.weight"
        if hf_name.startswith("model.layers.") and hf_name.endswith(".mlp.down_proj.weight"):
            layer_idx = hf_name.split(".")[2]
            return f"layers.{layer_idx}.mlp.down_proj.weight"
        if hf_name.startswith("model.layers.") and hf_name.endswith(".audio_mlp.down_proj.weight"):
            layer_idx = hf_name.split(".")[2]
            return f"layers.{layer_idx}.audio_mlp.down_proj.weight"
        return None

    # ----------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """DualFFN-routed Stage-0 forward.

        Drives the custom :class:`HiggsAudioV2DecoderLayer` stack with a
        per-position ``audio_token_mask`` derived from the input ``input_ids``
        (matching :func:`HiggsAudioV2Model.get_placeholder_mask`). Each layer
        applies the upstream DualFFN routing internally; this method only
        composes the embedding and final-norm steps around the layer loop.
        """
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        # Stash the step's input_ids for :meth:`_apply_audio_mode_bias` to read
        # in :meth:`sample`. ``sampling_metadata.prompt_token_ids`` is only
        # populated when penalties or token-id-aware logits processors are
        # active, so for our use case we have to keep our own copy.
        if input_ids is not None:
            self._last_step_input_ids = input_ids

        # Stash query_start_loc here (inside the forward context) so
        # _apply_audio_mode_bias (which runs LATER in sample(), after the
        # forward context exits) can recover each request's last token
        # position from the flat input_ids — see the per-row prev-token
        # lookup below. Without this, mixed prefill+decode batches would
        # fall back to ``input_ids[-num_rows:]``, which silently misroutes
        # the audio_out_bos detection in batch>1 voice clone (was the root
        # cause of the slot-shifted gibberish at batch=4).
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
        except Exception:
            self._last_step_query_start_loc = None

        if input_ids is not None and inputs_embeds is None:
            # Voice-clone ref-audio substitution: at the prompt-side audio
            # placeholder span we overwrite the text-embed lookups with the
            # multi-codebook embedding of the encoded reference clip carried in
            # ``model_intermediate_buffer``. No-op for plain-text requests and
            # for decode steps (which fall through to the audio-feedback path).
            info_dicts = kwargs.get("model_intermediate_buffer")
            if info_dicts is None:
                info_dicts = kwargs.get("runtime_additional_information")
            hidden_states = self._maybe_apply_ref_audio_substitution(
                hidden_states,
                input_ids,
                info_dicts,
            )

            # Audio-feedback substitution: vLLM's AR runner calls forward()
            # with ``inputs_embeds=None`` and lets the model embed its own
            # input_ids inline. ``embed_input_ids`` is only invoked via the
            # supports_mm_inputs/prompt_embeds branches, neither of which fires
            # for the talker. So we apply the substitution directly here: at
            # decode positions where input_ids == audio_token_id, swap the text
            # embedding with embed_audio_codes(last audio frame stashed in
            # ``self._audio_state[req_id]`` by :meth:`sample`).
            hidden_states = self._maybe_apply_audio_feedback(hidden_states, input_ids)

        audio_mask = self.audio_token_mask(input_ids) if input_ids is not None else None

        # Fused-residual loop: layers return (hidden, residual)
        # where ``residual`` carries everything up to the post-MLP add.
        residual: torch.Tensor | None = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                audio_token_mask=audio_mask,
            )

        # Final fused norm: combine final mlp_out with carried residual.
        norm_out = self.norm(hidden_states, residual)
        if isinstance(norm_out, tuple):
            norm_out = norm_out[0]
        return norm_out

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Any = None,
    ) -> torch.Tensor:
        """Sampler-compatible logits tensor.

        Returns a single ``[N, vocab_size]`` tensor that the stock
        vLLM AR runner pipes through ``.contiguous()`` and a generic
        sampler. The contract is intentionally tensor-only:

        - Text-position logits come from ``lm_head`` (128256-wide Llama vocab).
        - Audio-position logits at audio_token_id placeholders are also
          read from ``lm_head``; the per-position codebook routing path
          lives in :meth:`audio_codebook_logits` and is consumed by a
          separate codebook sampling adapter.
        """
        # Stash for the audio-sampler dispatch in :meth:`sample`. Limited to a
        # single step at a time; cleared in :meth:`sample` after consumption
        # so a missed call doesn't leak stale tensors.
        self._last_logits_hidden = hidden_states
        return self.logits_processor(self.lm_head, hidden_states)

    def audio_codebook_logits(
        self,
        hidden_states: torch.Tensor,
        audio_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-codebook logits at audio positions, ``[N_audio, num_codebooks, codebook_size]``.

        Runs the single fused ``audio_lm_head: Linear(hidden, num_codebooks *
        codebook_size)`` and reshapes the output. This mirrors upstream's
        ``HiggsAudioDecoderProjector.audio_lm_head`` + the
        ``view(-1, num_codebooks, codebook_size)`` reshape in
        ``HiggsAudioModel.forward``.

        Returns an empty tensor with the correct trailing shape when the mask
        selects no positions.
        """
        num_codebooks = int(self.config.num_codebooks)
        codebook_size = int(self.config.codebook_size)
        mask = audio_token_mask.reshape(-1).to(hidden_states.device)
        hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        if not mask.any():
            return torch.empty(
                (0, num_codebooks, codebook_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        flat = self.audio_lm_head(hidden_flat[mask])
        return flat.view(-1, num_codebooks, codebook_size)

    def embed_audio_codes(self, audio_ids: torch.Tensor) -> torch.Tensor:
        """Embed a ``[num_codebooks, T]`` audio-ids slice into a ``[T, hidden]`` tensor.

        Mirrors upstream ``HiggsAudioModel._embed_audio_ids``:

            shift = arange(num_codebooks) * codebook_size  # 1026 per slot
            audio_embed = sum_over_codebooks(audio_codebook_embeddings(audio_ids + shift))

        The 8 codebooks share a single ``Embedding(8 * codebook_size, hidden)``
        table; each codebook indexes into its own per-codebook slot via the
        shift. The output is the per-frame mixture of 8 codebook embeddings
        and is what the model re-feeds at the next AR step's
        ``<|AUDIO_OUT|>`` placeholder position.
        """
        if audio_ids.ndim != 2:
            raise ValueError(f"audio_ids must be [num_codebooks, T]; got shape {tuple(audio_ids.shape)}")
        codebook_size = int(self.config.codebook_size)
        num_codebooks = int(self.config.num_codebooks)
        shift = torch.arange(num_codebooks, device=audio_ids.device) * codebook_size
        shifted = audio_ids + shift.unsqueeze(-1)  # [num_codebooks, T]
        embed = self.audio_codebook_embeddings(shifted)  # [num_codebooks, T, hidden]
        return embed.sum(dim=0)  # [T, hidden]

    def _maybe_apply_audio_feedback(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Return ``hidden_states`` with audio-token positions replaced by the
        multi-codebook embedding of the LAST audio frame stashed by
        :meth:`sample`.

        Called from :meth:`forward` when ``inputs_embeds`` is ``None`` (vLLM's
        AR runner path) — the equivalent of upstream's
        ``_embed_audio_ids(audio_out_ids)`` injection at audio positions.
        We use ``index_copy`` (out-of-place) instead of mutating the original
        ``hidden_states`` because the embed-tokens output may be a view of a
        compiled buffer that doesn't tolerate in-place writes.
        """
        audio_token_id = int(self.config.audio_token_id)
        flat_ids = input_ids.reshape(-1)
        audio_positions = (flat_ids == audio_token_id).nonzero(as_tuple=False).reshape(-1)
        if audio_positions.numel() == 0:
            return hidden_states
        # Under MRV2 the hook publishes a batch-row-keyed view of the
        # req_id-keyed state; under V1 _audio_state is already batch-row-keyed.
        state = getattr(self, "_audio_state_view", None) or getattr(self, "_audio_state", None)
        if not state:
            return hidden_states
        # Resolve each flat position to its owning row via this step's
        # query_start_loc, stashed earlier in the same forward. A flat
        # position is not a row index in mixed prefill+decode steps, and a
        # wrong guess would inject another request's frame. If the layout is
        # unavailable, skip feedback rather than guess.
        q_start = getattr(self, "_last_step_query_start_loc", None)
        if not isinstance(q_start, torch.Tensor) or int(q_start.numel()) < 2:
            logger.warning_once(
                "Higgs Audio v2 talker: audio feedback skipped — no usable "
                "query_start_loc for this forward's batch layout"
            )
            return hidden_states
        q_start_cpu = q_start.detach().to("cpu").tolist()
        # Build a list of (pos, audio_emb) overrides and apply via index_copy
        # on a cloned hidden_states.
        rep_positions: list[int] = []
        rep_embeds: list[torch.Tensor] = []
        for pos in audio_positions.tolist():
            row = bisect_right(q_start_cpu, int(pos)) - 1
            if row < 0 or row + 1 >= len(q_start_cpu):
                # Past the padded tail or an empty span: no owning request.
                continue
            req_state = state.get(int(row))
            if req_state is None:
                continue
            audio_out_ids = req_state.get("audio_out_ids")
            if not isinstance(audio_out_ids, torch.Tensor) or audio_out_ids.numel() == 0:
                continue
            last_frame = audio_out_ids[:, -1:].to(hidden_states.device)
            audio_emb = self.embed_audio_codes(last_frame)  # [1, hidden]
            rep_positions.append(int(pos))
            rep_embeds.append(audio_emb[0].to(dtype=hidden_states.dtype))
        if not rep_positions:
            return hidden_states
        new_hidden = hidden_states.clone()
        flat_hidden = new_hidden.reshape(-1, new_hidden.shape[-1])
        idx = torch.tensor(rep_positions, dtype=torch.long, device=new_hidden.device)
        rep = torch.stack(rep_embeds, dim=0)
        flat_hidden.index_copy_(0, idx, rep)
        return new_hidden

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        """Embed the LM token stream, swapping in audio embeddings at placeholders.

        At audio positions the text embedding is replaced with the
        multi-codebook embedding of the last stashed audio frame, mirroring
        upstream's ``_embed_audio_ids`` injection at ``<|AUDIO_OUT|>``.

        Dead path for this talker under both runners: the V1 AR runner embeds
        inline via :meth:`_maybe_apply_audio_feedback`, and MRV2 disables
        ``supports_mm_inputs``. If ever wired live, the caller must supply the
        current step's ``query_start_loc``: this hook runs pre-forward, the
        stash still holds the previous step's layout, and the positional
        lookup is only valid in pure-decode steps.
        """
        text_embed = self.embed_tokens(input_ids)
        audio_token_id = int(self.config.audio_token_id)
        flat_ids = input_ids.reshape(-1)
        audio_positions = (flat_ids == audio_token_id).nonzero(as_tuple=False).reshape(-1)
        if audio_positions.numel() == 0:
            return text_embed
        # Under MRV2 the hook publishes a batch-row-keyed view of the
        # req_id-keyed state; under V1 _audio_state is already batch-row-keyed.
        state = getattr(self, "_audio_state_view", None) or getattr(self, "_audio_state", None)
        if not state:
            return text_embed
        # One audio position per active request per decode step;
        # ``audio_positions`` indexes into the flat stream in batch order.
        flat_embed = text_embed.reshape(-1, text_embed.shape[-1])
        for pos in audio_positions.tolist():
            # Row keys match flat positions in pure-decode steps. Prefill
            # prompt positions have no state yet; keep the text embed.
            req_state = state.get(int(pos))
            if req_state is None:
                continue
            audio_out_ids = req_state.get("audio_out_ids")
            if not isinstance(audio_out_ids, torch.Tensor) or audio_out_ids.numel() == 0:
                continue
            last_frame = audio_out_ids[:, -1:].to(text_embed.device)  # [Q, 1]
            audio_emb = self.embed_audio_codes(last_frame)  # [1, hidden]
            flat_embed[pos] = audio_emb[0].to(dtype=text_embed.dtype)
        return text_embed

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput:
        """Wrap raw decoder outputs into the :class:`OmniOutput` contract.

        Mirrors the canonical Qwen3-TTS / Fish-Speech recovery pattern:
        the runner threads per-request ``codes.audio`` into
        ``model_intermediate_buffer`` (a list of dicts in batch order); we
        concatenate those and trim ``text_hidden_states`` to the emitted
        audio span. Falls back to the deprecated ``runtime_additional_information``
        kwarg for older runners, and finally to ``audio_codes`` /
        ``model_kwargs[audio_codes]`` for direct callers.
        """
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        hidden = model_outputs

        # Primary contract: model_intermediate_buffer (Qwen3-TTS / Fish-Speech).
        info_dicts = kwargs.get("model_intermediate_buffer")
        if info_dicts is None:
            info_dicts = kwargs.get("runtime_additional_information")
        if info_dicts is None:
            info_dicts = []

        # Build a per-request audio-codes list whose length exactly equals
        # ``len(info_dicts)`` (== batch size). Requests with no emitted codes
        # this step get an empty tensor placeholder — that way the runner's
        # ``to_payload_element`` (which indexes ``element[idx]`` per request)
        # can never silently fall back to ``element[0]`` for higher slots,
        # which was the root of the batch>1 voice-clone slot-shift bug
        # (slot k ended up speaking slot k-1's text).
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

        # Fallbacks: explicit kwarg or wrapped model_kwargs dicts (preserved for
        # direct-API callers that don't go through the runner buffer).
        audio_codes = kwargs.get("audio_codes")
        if audio_codes is None:
            for source_name in ("model_kwargs", "model_kwargs_extra"):
                source = kwargs.get(source_name)
                if isinstance(source, dict) and "audio_codes" in source:
                    audio_codes = source["audio_codes"]
                    break
        if audio_codes is None:
            audio_codes = torch.empty(0, dtype=torch.long)
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={"codes": {"audio": audio_codes}},
        )

    # ----------------------------------------------------- model-owned sampler
    # Opt into the AR runner's model-sampler hook (see
    # vllm_omni/worker/gpu_ar_model_runner.py:_sample). The runner calls
    # ``self.model.sample(logits, sampling_metadata)`` when this flag is True;
    # we delegate to the stock vLLM sampler for now and document the
    # per-position audio-codebook-0 dispatch as the immediate follow-up. The
    # hook itself is what unblocks an external integrator from plugging in the
    # final audio-side sampler without touching the runner.
    prefer_model_sampler: bool = True

    # Tell the runner's :func:`extract_multimodal_outputs` to forward the OmniOutput
    # produced by :meth:`make_omni_output`. Without this flag the multimodal
    # payload (audio codes) is dropped and the downstream stage receives only
    # text hidden states.
    have_multimodal_outputs: bool = True

    def sample(self, logits: torch.Tensor, sampling_metadata: Any) -> Any:
        """Model-owned sampler with delay-pattern audio dispatch.

        Each generation step the model emits one LM token. At audio positions
        (where the previous LM token was ``audio_bos_token_id`` or
        ``audio_token_id``), the same hidden state also drives the fused
        ``audio_lm_head`` to produce ``[num_codebooks, codebook_size]``
        per-codebook logits. We argmax those per-codebook, then APPLY THE
        UPSTREAM DELAY PATTERN:

          - For the first ``num_codebooks - 1`` audio steps, force codebooks
            ``[num_delay + 1 :]`` to ``audio_stream_bos_id``; only codebooks
            up to ``num_delay`` are real.
          - When ``audio_stream_eos_id`` first appears in a codebook k, force
            all earlier codebooks to ``eos`` too and start the trailing
            ``num_codebooks - k - 1`` "ramp-down" steps where successive
            codebooks also get ``eos``.
          - When all 8 codebooks have ramped down, switch the LM next-token
            from ``audio_token_id`` to ``audio_eos_token_id`` so the engine
            sees a stop signal at the right place.

        Per-request state (``num_delay``, ``num_remaining_delays``,
        cumulative ``audio_out_ids: [num_codebooks, T_so_far]``) lives in
        ``self._audio_state[req_id]``; ``embed_input_ids`` reads it on the
        next step to substitute the audio-token embedding via
        :meth:`embed_audio_codes`.
        """
        sampler = getattr(self, "_stock_sampler", None)
        if sampler is None:
            from vllm.v1.sample.sampler import Sampler

            sampler = Sampler()
            self._stock_sampler = sampler

        # Bias the text-vocab logits so the LM stays on the audio path. This
        # forces ``audio_token_id`` at audio positions; the delay-pattern
        # logic below later overrides the LM token to ``audio_eos_token_id``
        # when the ramp-down completes.
        self._apply_audio_mode_bias(logits, sampling_metadata)
        sampler_output = sampler(logits=logits, sampling_metadata=sampling_metadata)

        hidden = getattr(self, "_last_logits_hidden", None)
        return self._dispatch_audio_codes(sampler_output, hidden, sampling_metadata)

    def _dispatch_audio_codes(self, sampler_output, hidden, sampling_metadata, row_req_ids=None):
        """Turn this step's LM samples into per-request audio codes.

        Shared by the V1 model sampler and the MRV2 sampler hook; both callers
        sample LM tokens themselves, so audio-code sampling and per-request
        state progression stay identical. ``hidden`` holds this step's logits
        rows; the ``_last_logits_hidden`` stash is cleared on entry.

        ``row_req_ids`` maps each sampled row to its owning request id, as
        built by :meth:`_row_req_ids` from ``cu_num_logits``. Passing it keys
        ``_audio_state`` and ``_slot_output_len`` by req_id, which survives
        MRV2's batch compaction between steps. Under V1 (``None``) batch-row
        keys apply, matching the V1 runner's stable row layout.
        """
        self._last_logits_hidden = None
        if hidden is None:
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

        audio_token_id = int(self.config.audio_token_id)
        audio_delay_id = int(self.config.audio_delay_token_id)
        is_audio = (sampled_flat == audio_token_id) | (sampled_flat == audio_delay_id)

        if not hasattr(self, "_audio_state"):
            self._audio_state: dict[Any, dict[str, Any]] = {}
        if not hasattr(self, "_slot_output_len"):
            self._slot_output_len: dict[Any, int] = {}

        # Under MRV2, key per-request state by req_id so it survives batch
        # compaction; under V1, batch-row keys match the runner's stable layout.
        def _key(bi: int) -> Any:
            if row_req_ids is not None and bi < len(row_req_ids):
                return row_req_ids[bi]
            return int(bi)

        # Skip the audio-code sampling for rows that just transitioned
        # OUT of audio_out_bos. The hidden state at this prefill-end step is
        # at the audio_out_bos position, where upstream's audio_out_mask is
        # False and no audio_logits are computed. The LM bias above already
        # forced ``audio_token_id`` at this step, and the next forward will
        # produce a hidden at the new audio_token_id position — that is what
        # upstream samples its first audio frame from. Also reset any stale
        # ``_audio_state`` and ``_slot_output_len`` for the slot, so a fresh
        # request reusing a finished slot starts clean.
        first_after_bos = getattr(self, "_last_first_audio_after_bos", None)
        self._last_first_audio_after_bos = None
        num_codebooks_local = int(self.config.num_codebooks)
        bos_local = int(self.config.audio_stream_bos_id)
        seeded_rows: list[int] = []
        seeded_frame: torch.Tensor | None = None
        # HF seeds frame 0 with all-BOS at the audio_bos position and ignores
        # the model's prediction there; the first real audio frame is sampled
        # one step later at the audio_token_id position, so we skip the
        # prediction at the bos row and emit the all-BOS frame instead.
        if isinstance(first_after_bos, torch.Tensor) and first_after_bos.numel() == is_audio.shape[0]:
            first_after_bos = first_after_bos.to(is_audio.device)
            skip_rows = first_after_bos & is_audio
            if bool(skip_rows.any()):
                # Upstream's AUDIO_INIT step seeds ``audio_out_ids`` with
                # a single all-``audio_stream_bos_id`` frame and the NEXT
                # forward feeds ``embed_audio_codes(all_bos)`` at the new
                # audio_out_token_idx position. Without this, our audio MLP
                # receives a text embedding at the audio_token_id position
                # and produces NaN logits. Seed per-slot state with the
                # all-BOS frame here, AND emit it as the first output frame
                # so the downstream delay-pattern revert sees the leading
                # BOS-only frame upstream emits.
                seeded_frame = torch.full((num_codebooks_local,), bos_local, dtype=torch.long, device=hidden.device)
                for bi in torch.nonzero(skip_rows, as_tuple=False).reshape(-1).tolist():
                    bi = int(bi)
                    # Reset stale state from a finished prior request.
                    k = _key(bi)
                    self._slot_output_len[k] = 0
                    self._audio_state[k] = {
                        "num_delay": 0,
                        "num_remaining_delays": None,
                        "audio_out_ids": seeded_frame.unsqueeze(-1).clone(),  # [Q, 1]
                    }
                    seeded_rows.append(bi)
                is_audio = is_audio & ~first_after_bos

        if not bool(is_audio.any()) and not seeded_rows:
            self._last_audio_codes = None
            return sampler_output

        # Map audio-position rows to their absolute batch indices. The
        # delay-pattern + state-update loop below uses this for per-request
        # state lookup; the RAS check between the head sample and the
        # delay-pattern apply also reads it.
        audio_row_indices = torch.nonzero(is_audio, as_tuple=False).reshape(-1).tolist()

        # Per-codebook logits at audio positions only.
        cb_logits = self.audio_codebook_logits(hidden, is_audio)  # [N_audio, Q, V]

        # Emulate upstream's ``HiggsAudioV2DelayPatternLogitsProcessor``.
        # The raw audio_lm_head has very large weights at index 1025
        # (audio_stream_eos): cb1 |w[1025]|=9.25 vs mean content 1.53 (6x).
        # Without masking, the model naturally fires 1025 at codebook 1..6
        # with huge confidence (logit 32 vs runner-up 6) and triggers
        # premature ramp-down at frame 2. Upstream's processor MASKS
        # forbidden vocab BEFORE sampling so each codebook k is bound to:
        #   - emit ``audio_stream_bos_id`` (1024) for k leading frames,
        #   - emit content tokens [0..1023] thereafter,
        #   - emit ``audio_stream_eos_id`` (1025) only during ramp-down.
        bos_pre = int(self.config.audio_stream_bos_id)
        eos_pre = int(self.config.audio_stream_eos_id)
        for local_i, batch_i in enumerate(audio_row_indices):
            state_pre = self._audio_state.get(_key(int(batch_i)))
            num_delay_pre = int(state_pre.get("num_delay", 0)) if state_pre else 0
            num_rem_pre = state_pre.get("num_remaining_delays") if state_pre else None
            if num_rem_pre is not None:
                # Ramp-down: mask all but EOS for the EOS-locked codebooks; the
                # remaining ones still emit content. Locked range: codebooks
                # [0 : num_codebooks - num_rem_pre].
                lock_until = int(self.config.num_codebooks) - int(num_rem_pre)
                for q in range(int(self.config.num_codebooks)):
                    row = cb_logits[local_i, q]
                    if q < lock_until:
                        mask = torch.full_like(row, float("-inf"))
                        mask[eos_pre] = row[eos_pre]
                        cb_logits[local_i, q] = mask
                    else:
                        # Disallow stream specials during ramp-down content frames
                        cb_logits[local_i, q, bos_pre] = float("-inf")
                        cb_logits[local_i, q, eos_pre] = float("-inf")
            else:
                # Leading delay phase. Codebook k starts emitting real content
                # at step k (delay = k). Steps with k > num_delay_pre are still
                # in BOS phase — force 1024. Steps with k <= num_delay_pre are
                # content — disallow 1024 (BOS) and 1025 (EOS).
                for q in range(int(self.config.num_codebooks)):
                    row = cb_logits[local_i, q]
                    if q > num_delay_pre:
                        mask = torch.full_like(row, float("-inf"))
                        mask[bos_pre] = row[bos_pre]
                        cb_logits[local_i, q] = mask
                    else:
                        cb_logits[local_i, q, bos_pre] = float("-inf")
                        # Allow EOS at codebook 0 only (it gates ramp-down).
                        # Disallow at all other content codebooks.
                        if q != 0:
                            cb_logits[local_i, q, eos_pre] = float("-inf")

        # Sample per-codebook, following the upstream
        # ``HiggsAudioModel._sample_audio_tokens`` pipeline byte-for-byte.
        cb_logits_2d = cb_logits.reshape(-1, cb_logits.shape[-1])
        codes_2d = self._sample_audio_codes_upstream(
            cb_logits_2d,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
        )
        codes_flat = codes_2d.view(cb_logits.shape[0], cb_logits.shape[1]).to(torch.long)

        # (PyTorch align): Repetition-Aware Sampling (RAS). Upstream
        # ``HiggsAudioModel._sample_audio_tokens`` re-samples any codebook
        # that just emitted the same token >= ras_win_max_num_repeat times
        # in the last ras_win_len steps. This breaks codebook collapse loops
        # — a 3B model with greedy/low-temp sampling often locks into one
        # value otherwise.
        ras_win_len = 7
        ras_win_max_num_repeat = 2
        # Codes_flat is [N_audio, Q]; we re-sample PER (audio_position, codebook)
        # row that hits the repeat threshold over the last ras_win_len audio
        # frames of that codebook's history (from state.audio_out_ids).
        if codes_flat.shape[0] > 0:
            audio_idx_to_batch = audio_row_indices  # already a list[int]
            for local_i in range(codes_flat.shape[0]):
                bi = int(audio_idx_to_batch[local_i])
                req_state = self._audio_state.get(_key(bi))
                if req_state is None:
                    continue
                aoi = req_state.get("audio_out_ids")
                if not isinstance(aoi, torch.Tensor) or aoi.shape[-1] < ras_win_len:
                    continue
                window = aoi[:, -ras_win_len:].to(codes_flat.device)  # [Q, W]
                this_codes = codes_flat[local_i]  # [Q]
                rep_count = (window == this_codes.unsqueeze(-1)).sum(dim=-1)  # [Q]
                row_indices = torch.nonzero(rep_count >= ras_win_max_num_repeat, as_tuple=False).reshape(-1)
                if row_indices.numel() == 0:
                    continue
                # Re-sample those codebooks from cb_logits directly (no top-k/top-p
                # in RAS resample per upstream — pure softmax + multinomial).
                resample_logits = cb_logits[local_i, row_indices].float()
                resample_probs = resample_logits.softmax(dim=-1)
                resampled = torch.multinomial(resample_probs, num_samples=1).squeeze(-1)
                codes_flat[local_i, row_indices] = resampled.to(codes_flat.dtype)

        # Apply the upstream delay pattern + EOS ramp-down per request.
        num_codebooks = int(self.config.num_codebooks)
        bos = int(self.config.audio_stream_bos_id)
        eos_stream = int(self.config.audio_stream_eos_id)
        audio_eos_vocab = int(getattr(self.config, "audio_eos_token_id", -1))

        # ``audio_row_indices`` and ``self._audio_state`` were initialised
        # earlier (right after ``is_audio`` was computed) so the RAS check
        # can reference them.

        # Per-request reset: under MRV2 a req_id not yet tracked is a new
        # request; under V1 the output_token_ids length heuristic detects
        # slot reuse. ``_slot_output_len`` is initialised above next to
        # ``_audio_state``; this reset block runs after every early return.
        output_ids = getattr(sampling_metadata, "output_token_ids", None)
        for batch_i in audio_row_indices:
            bi = int(batch_i)
            k = _key(bi)
            if row_req_ids is not None:
                # MRV2: a req_id not yet tracked is a new request.
                if k not in self._slot_output_len and k in self._audio_state:
                    self._audio_state.pop(k, None)
                self._slot_output_len[k] = 0
            else:
                current_len = len(output_ids[bi]) if output_ids is not None and bi < len(output_ids) else 0
                prior_len = self._slot_output_len.get(k, -1)
                is_new_request = (prior_len > current_len) or (prior_len == -1 and current_len == 0)
                if is_new_request and k in self._audio_state:
                    self._audio_state.pop(k, None)
                self._slot_output_len[k] = current_len

        # Simplification: we DON'T override ``sampler_output.sampled_token_ids``
        # back to ``audio_eos_token_id`` at ramp-down completion. The vLLM
        # ``SamplerOutput.sampled_token_ids`` is plumbed through several
        # downstream structures that may have already cached the GPU pointer;
        # an in-place mutation here was crashing the worker with exit code
        # None (silent CUDA error). Instead, ``max_tokens`` (set in the deploy
        # yaml) is the stop bound while the audio span ramps down naturally.
        new_codes_flat: list[torch.Tensor] = []
        for local_i, batch_i in enumerate(audio_row_indices):
            state = self._audio_state.setdefault(
                _key(int(batch_i)),
                {"num_delay": 0, "num_remaining_delays": None, "audio_out_ids": None},
            )
            num_delay: int = state["num_delay"]
            num_remaining_delays: int | None = state["num_remaining_delays"]
            this_codes = codes_flat[local_i].clone()  # [Q]

            # Leading delay-pattern BOS pad.
            if num_delay + 1 < num_codebooks:
                this_codes[num_delay + 1 :] = bos
                num_delay += 1

            # Trailing eos ramp-down.
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
                # Audio ramp-down finished. Set a per-slot termination flag
                # so the NEXT step's bias forces audio_eos_token_id (upstream's
                # ``next_tokens[...] = audio_eos_token_id`` override at the
                # ramp-completion point — see modeling_higgs_audio.py:1564).
                _ = audio_eos_vocab
                num_delay = 0
                num_remaining_delays = None
                state["num_delay"] = num_delay
                state["num_remaining_delays"] = num_remaining_delays
                state["should_terminate"] = True
                new_codes_flat.append(torch.full_like(this_codes, -1))
                continue

            state["num_delay"] = num_delay
            state["num_remaining_delays"] = num_remaining_delays
            # Append to the per-request audio_out_ids buffer.
            if state["audio_out_ids"] is None:
                state["audio_out_ids"] = this_codes.unsqueeze(-1).clone()
            else:
                state["audio_out_ids"] = torch.cat([state["audio_out_ids"], this_codes.unsqueeze(-1)], dim=-1)
            new_codes_flat.append(this_codes)

        # Reassemble the per-position codes tensor with the post-delay-pattern
        # values; `-1` rows mark "no audio code at this position".
        codes_full = torch.full(
            (int(sampled_flat.numel()), num_codebooks),
            -1,
            dtype=torch.long,
            device=hidden.device,
        )
        if new_codes_flat:
            stacked = torch.stack(new_codes_flat, dim=0).to(hidden.device)
            codes_full[is_audio] = stacked
        # Emit the seeded all-BOS frame at the prefill-end (AUDIO_INIT)
        # rows so the talker output has the leading delay-pattern frame the
        # downstream revert helper expects.
        if seeded_rows and seeded_frame is not None:
            seeded_rows_idx = torch.tensor(seeded_rows, dtype=torch.long, device=hidden.device)
            codes_full.index_copy_(
                0,
                seeded_rows_idx,
                seeded_frame.unsqueeze(0).expand(len(seeded_rows), -1).to(hidden.device),
            )

        self._last_audio_codes = codes_full
        self._postprocess_cursor = 0
        return sampler_output

    def mrv2_custom_sampler(self, sampler: Any) -> tuple[Any, Any] | None:
        """MRV2 sampling hook: delegate LM token sampling to the native staged sampler.

        The V1 AR runner calls :meth:`sample` directly; under MRV2 the stock
        OmniModelState dispatches here instead. The native gpu sampler samples
        LM tokens, so its staged per-request state such as temperature, top-k,
        top-p, min-p, penalties and logprobs applies unchanged. Audio codes
        flow through :meth:`_dispatch_audio_codes` and ``postprocess`` as on
        V1.
        """
        from vllm_omni.worker_v2.omni_sampler import OmniSampler, OmniSamplingOutput
        from vllm_omni.worker_v2.warmup import is_warmup_request

        talker = self

        class _StepSampler(OmniSampler):
            def sample_step(self, hidden_states, input_batch, req_states, grammar_output, standard_sample):
                if all(is_warmup_request(r) for r in input_batch.req_ids):
                    talker._last_audio_codes = None
                    talker._pending_codes_by_req = {}
                    return super().sample_step(hidden_states, input_batch, req_states, grammar_output, standard_sample)
                if grammar_output is not None or input_batch.num_draft_tokens:
                    raise ValueError("Higgs Audio v2 MRV2 does not support grammar or speculative sampling")
                hidden = hidden_states[input_batch.logits_indices]
                logits = talker.compute_logits(hidden, None)
                # Each row's owning request id, from cu_num_logits.
                row_req_ids = talker._row_req_ids(input_batch)
                # The context token at each row is input_ids[logits_indices].
                row_prev_tokens = input_batch.input_ids[input_batch.logits_indices].tolist()
                talker._apply_audio_mode_bias(logits, None, row_req_ids=row_req_ids, row_prev_tokens=row_prev_tokens)
                out = self.base_sampler(logits, input_batch)
                out = talker._dispatch_audio_codes(out, hidden, None, row_req_ids=row_req_ids)
                talker._pending_codes_by_req = talker._split_codes_by_request(talker._last_audio_codes, input_batch)
                # Do not rebuild the batch-row view here: the next forward's
                # batch layout differs, so HiggsAudioV2ModelState.run_preprocess
                # rebuilds it from the batch that forward consumes.
                return OmniSamplingOutput(out, out.num_sampled, out.num_rejected)

        return _StepSampler(sampler), None

    @staticmethod
    def _row_req_ids(input_batch: Any) -> list[Any]:
        """Map each logits row of this step to its owning request id.

        ``cu_num_logits`` is an inclusive prefix of length ``num_reqs + 1``:
        request ``j`` owns rows ``[cu[j], cu[j+1])``. Derive the mapping from
        the prefix rather than assuming row == request: a mid-prefill request
        owns no row, and the upstream construction may change.
        """
        n = input_batch.num_reqs
        cu_np = getattr(input_batch, "cu_num_logits_np", None)
        cu = cu_np.tolist() if cu_np is not None else input_batch.cu_num_logits.tolist()
        req_ids = input_batch.req_ids
        return [req_ids[j] for j in range(n) for _ in range(int(cu[j + 1]) - int(cu[j]))]

    def _split_codes_by_request(self, codes: torch.Tensor | None, input_batch: Any) -> dict[Any, torch.Tensor]:
        """Split the sampled code rows into a req_id-keyed map for postprocess.

        The MRV2 runner's postprocess runs before the step's sampling, so it
        serves each request the previous step's entry. Row ownership follows
        the ``cu_num_logits`` prefix, as in :meth:`_row_req_ids`.
        """
        from vllm_omni.worker_v2.warmup import is_warmup_request

        if codes is None:
            return {}
        cu = input_batch.cu_num_logits
        ends = cu[1 : input_batch.num_reqs + 1].tolist()
        starts = cu[: input_batch.num_reqs].tolist()
        pending: dict[Any, torch.Tensor] = {}
        for i, req_id in enumerate(input_batch.req_ids):
            lo, hi = int(starts[i]), int(ends[i])
            if hi > lo and not is_warmup_request(req_id):
                pending[req_id] = codes[lo:hi]
        return pending

    def _rebuild_audio_state_view(self, input_batch: Any) -> None:
        """Publish a batch-row-keyed view of the req_id-keyed audio state.

        ``_maybe_apply_audio_feedback`` looks state up by batch row, while
        the authoritative ``_audio_state`` is keyed by req_id under MRV2;
        this view bridges the two key spaces. Called by
        :class:`HiggsAudioV2ModelState.run_preprocess` before every forward:
        the batch layout can change between steps, so a view materialized at
        sampling time would be stale by the next lookup.
        """
        state = getattr(self, "_audio_state", None)
        if not isinstance(state, dict):
            return
        view: dict[int, dict[str, Any]] = {}
        for i in range(input_batch.num_reqs):
            req_id = input_batch.req_ids[i]
            if req_id in state:
                view[i] = state[req_id]
        self._audio_state_view = view

    def on_requests_finished(self, finished_req_ids: set[str]) -> None:
        """Drop finished requests' per-request audio state.

        Under MRV2 the per-request dicts are keyed by req_id and a key never
        repeats, so finished entries would otherwise retain their
        ``audio_out_ids`` GPU tensors for the process lifetime. Under V1 the
        batch-row keys are bounded by slot reuse, so the hook is a no-op
        there. Aborted requests may also leave an unconsumed entry in
        ``_pending_codes_by_req``, since postprocess only walks the current
        batch. Preempted requests keep their state: a recomputed request
        must resume its delay-pattern history.
        """
        state = getattr(self, "_audio_state", None)
        lens = getattr(self, "_slot_output_len", None)
        pending = getattr(self, "_pending_codes_by_req", None)
        for req_id in finished_req_ids:
            if isinstance(state, dict):
                state.pop(req_id, None)
            if isinstance(lens, dict):
                lens.pop(req_id, None)
            if isinstance(pending, dict):
                pending.pop(req_id, None)

    def create_omni_model_state(self, vllm_config, encoder_cache, device):
        """MRV2 model state with per-forward audio-state view realignment."""
        from .model_state import HiggsAudioV2ModelState

        return HiggsAudioV2ModelState(vllm_config, self, encoder_cache, device)

    @staticmethod
    def _sample_audio_codes_upstream(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """Replicate the upstream HF ``LogitsProcessorList`` pipeline that
        ``HiggsAudioModel._sample_audio_tokens`` calls via
        ``logits_processor(None, next_audio_token_logits)``.

        Order (matching transformers' standard ``TemperatureLogitsWarper``,
        ``TopKLogitsWarper``, ``TopPLogitsWarper``):

        1. Divide by temperature.
        2. Top-k: keep the K largest, set others to -inf.
        3. Top-p (nucleus): keep the smallest prefix whose cumulative
           softmax probability is <= top_p; mask the rest. Always keep at
           least the top-1.
        4. Softmax → multinomial sample.

        ``logits`` shape: ``[N, V]``. Returns ``[N]`` of int sampled indices.
        """
        x = logits.float()
        if temperature is not None and temperature > 0.0 and temperature != 1.0:
            x = x / temperature
        if top_k is not None and 0 < top_k < x.shape[-1]:
            kth = x.topk(top_k, dim=-1).values[..., -1:]
            x = x.masked_fill(x < kth, float("-inf"))
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = x.sort(dim=-1, descending=True)
            cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            sorted_mask = cumprobs > top_p
            # Shift right by one so we always keep at least the top-1 token.
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            # Scatter back to original index space and apply.
            mask = torch.zeros_like(x, dtype=torch.bool)
            mask.scatter_(-1, sorted_idx, sorted_mask)
            x = x.masked_fill(mask, float("-inf"))
        if temperature is None or temperature <= 0.0:
            return torch.argmax(x, dim=-1)
        probs = x.softmax(dim=-1)
        # Guard against any all-masked rows (shouldn't happen with the shift
        # above) — fall back to argmax of original logits.
        valid = probs.sum(dim=-1) > 0
        sampled = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
        if valid.any():
            sampled[valid] = torch.multinomial(probs[valid], num_samples=1).squeeze(-1)
        if (~valid).any():
            sampled[~valid] = torch.argmax(logits[~valid].float(), dim=-1)
        return sampled

    def _apply_audio_mode_bias(
        self,
        logits: torch.Tensor,
        sampling_metadata: Any,
        row_req_ids: list[Any] | None = None,
        row_prev_tokens: list[int] | None = None,
    ) -> None:
        """Mask non-audio tokens at audio-mode positions, in-place on ``logits``.

        Per request, find the last token seen so far: the last of
        ``output_token_ids[i]`` if non-empty, else the last of
        ``prompt_token_ids[i]``. If it is ``audio_bos_token_id`` or
        ``audio_token_id``, force the next sample to one of
        ``{audio_token_id, audio_eos_token_id}``. The un-biased argmax over
        the 128k vocab favours text-token ranges, in particular the global
        ``eos_token_id``.

        The MRV2 hook passes ``row_req_ids``, each logits row's owning
        request id, and ``row_prev_tokens``, the context token at each row
        (``input_ids[logits_indices]``). Rows exclude mid-prefill requests,
        so row ``i`` is not request ``i`` in mixed steps. Both default to
        ``None`` on V1, which keeps the metadata/stash walk below; the
        ``should_terminate`` lookup needs ``row_req_ids`` under MRV2 because
        ``_audio_state`` is keyed by request id there.

        The standard ``eos_token_id`` is deliberately not in the allowed
        set: letting it through would end the whole sequence at the first
        step. Stopping inside the audio span is the ``audio_eos`` token's
        job; once it fires the bias drops and the next call falls back to
        the stock sampler.
        """
        if logits is None or logits.ndim != 2:
            return
        audio_bos = int(self.config.audio_bos_token_id)
        audio_id = int(self.config.audio_token_id)
        # (PyTorch align, corrected): upstream NEVER lets the model
        # naturally emit ``audio_eos_token_id`` in the LM stream. The LM
        # next-token is force-set to ``audio_out_token_idx`` every step;
        # the audio span only ends when the delay-pattern ramp-down completes
        # (driven by ``audio_stream_eos_id`` appearing in a codebook), at
        # which point upstream OVERRIDES the LM token to ``audio_eos``.
        # Our diag showed the model's ``audio_eos`` logit is very high right
        # after the first audio step; allowing it via sampling stops the
        # audio span with 1 frame and the rest of the LM stream goes off-rails.
        # Force ONLY ``audio_token_id``.
        allowed_extra: list[int] = []
        # Walk per-request to decide which rows to mask.
        prompt_ids = getattr(sampling_metadata, "prompt_token_ids", None)
        output_ids = getattr(sampling_metadata, "output_token_ids", None)
        num_rows = int(logits.shape[0])
        # Fallback "previous token" source: read each request's LAST input
        # position from ``self._last_step_input_ids`` using ``query_start_loc``
        # from the forward context. The previous version of this code took
        # ``stash_ids[-num_rows:]`` which only works for pure-decode batches
        # (where each row contributes exactly one token to the flat input_ids).
        # In MIXED prefill+decode batches the last N tokens of flat input_ids
        # all belong to the LAST prefilling request, so the "prev token" for
        # other rows came back as random tokens from inside that request's
        # prompt — which caused the audio_out_bos detection to fail for some
        # rows, the audio_token_id bias to skip them, and the per-row codes to
        # collapse to -1 silently (see the batch>1 voice-clone gibberish bug).
        stash_ids = None if row_prev_tokens is not None else getattr(self, "_last_step_input_ids", None)
        stash_tail: list[int] | None = None
        if isinstance(stash_ids, torch.Tensor) and stash_ids.numel() > 0:
            # forward() stashed query_start_loc for us inside the forward
            # context — get_forward_context() would fail here because we're
            # past the with-block by the time sample() runs.
            q_start = getattr(self, "_last_step_query_start_loc", None)
            if isinstance(q_start, torch.Tensor) and int(q_start.numel()) == num_rows + 1:
                # Per-request last position: q_start_loc[i+1] - 1.
                q_start_cpu = q_start.detach().to("cpu").tolist()
                tail_idx = [max(0, int(q_start_cpu[i + 1]) - 1) for i in range(num_rows)]
                flat_ids = stash_ids.detach().to("cpu").tolist()
                stash_tail = [int(flat_ids[idx]) if idx < len(flat_ids) else -1 for idx in tail_idx]
            elif int(stash_ids.numel()) >= num_rows:
                # Pure-decode fallback when query_start_loc is unavailable:
                # input_ids has exactly num_rows tokens (one per request) and
                # the slice is order-preserving.
                stash_tail = stash_ids[-num_rows:].detach().to("cpu").tolist()
        # Also stash a per-row "is this the first audio-mode step right
        # after audio_out_bos" flag. Upstream NEVER samples audio codes at
        # the audio_out_bos position (its audio_out_mask is False there);
        # the first audio frame is sampled one step later, at the new
        # audio_out_token_idx position. ``sample()`` reads this flag to
        # suppress the audio-code sampling at the prefill-end step so the
        # first audio frame is sampled from the correct hidden state.
        first_after_bos = torch.zeros(num_rows, dtype=torch.bool, device=logits.device)
        for i in range(num_rows):
            prev: int | None = None
            if row_prev_tokens is not None and i < len(row_prev_tokens):
                prev = int(row_prev_tokens[i])
            elif output_ids is not None and i < len(output_ids):
                hist = output_ids[i]
                if hist:
                    prev = int(hist[-1])
            if prev is None and prompt_ids is not None:
                # Prompt_token_ids may be a tensor or a list-of-lists
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
            if prev not in (audio_bos, audio_id):
                continue
            if prev == audio_bos:
                first_after_bos[i] = True
            # If this slot's audio ramp finished last step, force LM token
            # to audio_eos so the engine stops naturally. Matches upstream's
            # ``next_tokens[...] = audio_eos_token_id`` override.
            audio_state = getattr(self, "_audio_state", {}) or {}
            # Under MRV2 the state dict is keyed by req_id; row i's owner is
            # row_req_ids[i] (rows and requests desync in mixed prefill steps).
            state_key = row_req_ids[i] if row_req_ids is not None and i < len(row_req_ids) else i
            slot_state = audio_state.get(state_key)
            should_terminate = bool(isinstance(slot_state, dict) and slot_state.get("should_terminate"))
            audio_eos_vocab = int(getattr(self.config, "audio_eos_token_id", -1))
            if should_terminate and 0 <= audio_eos_vocab < int(logits.shape[-1]):
                # Force audio_eos at this position; upstream override at
                # ramp-down completion (modeling_higgs_audio.py:1564).
                row = logits[i]
                mask = torch.full_like(row, float("-inf"))
                mask[audio_eos_vocab] = row[audio_eos_vocab]
                logits[i].copy_(mask)
                slot_state["should_terminate"] = False
                continue
            # Always force ``audio_token_id`` — upstream mirror.
            allowed: set[int] = {audio_id, *allowed_extra}
            row = logits[i]
            mask = torch.full_like(row, float("-inf"))
            for tok in allowed:
                if 0 <= tok < row.shape[-1]:
                    mask[tok] = row[tok]
            logits[i].copy_(mask)
        self._last_first_audio_after_bos = first_after_bos

    has_postprocess: bool = True

    def postprocess(
        self,
        hidden_states_slice: torch.Tensor,
        multimodal_outputs: Any = None,
        **req_infos: Any,
    ) -> dict[str, Any]:
        """Publish per-request audio codes into model_intermediate_buffer.

        Under MRV2 the runner calls this once per request after every
        forward, BEFORE the step's sampling, so codes produced by a sampling
        step are consumed one step later. A positional cursor desyncs in
        mixed prefill+decode steps, so the MRV2 sampling hook fills
        ``_pending_codes_by_req`` keyed by request id and this method serves
        entries from that map. The V1 runner calls postprocess in batch
        order right after sampling, so the positional cursor over
        ``_last_audio_codes`` remains the fallback there.
        """
        _ = multimodal_outputs  # consumed by the runner directly
        pending = getattr(self, "_pending_codes_by_req", None)
        if pending is not None:
            slice_codes = pending.pop(req_infos.get("req_id"), None)
            if slice_codes is None:
                return {}
            return self._emit_code_rows(slice_codes)
        codes_full = getattr(self, "_last_audio_codes", None)
        if codes_full is None:
            return {}

        # The runner walks requests in batch order; pop from the cursor as we go.
        # ``_last_audio_codes`` is one row per request slot (== one logits_indices
        # row), not one row per prompt token, so advance by 1 per call.
        cursor = int(getattr(self, "_postprocess_cursor", 0))
        if cursor >= int(codes_full.shape[0]):
            # Defensive: shape drift between forward and postprocess.
            self._postprocess_cursor = 0
            return {}
        slice_codes = codes_full[cursor : cursor + 1]
        self._postprocess_cursor = cursor + 1
        return self._emit_code_rows(slice_codes)

    @staticmethod
    def _emit_code_rows(slice_codes: torch.Tensor) -> dict[str, Any]:
        # Drop placeholder rows (-1): only emit codes for actual audio positions.
        audio_rows = slice_codes[:, 0] >= 0
        if not bool(audio_rows.any()):
            return {}
        new_codes = slice_codes[audio_rows].to(torch.int32)

        # Return only this step's frames: the engine accumulates per-step
        # multimodal_outputs across steps and concatenates at request finish.
        return {"codes": {"audio": new_codes}}
