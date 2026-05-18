# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
  equals ``audio_token_id=128016`` or ``audio_delay_token_id=128014``; see
  ``UPSTREAM_TRACE.md`` for the full derivation.

- Multi-codebook output head: at audio positions, Stage 0 emits one ID per
  codebook (8 codebooks of vocabulary 1026 each). Codebook 0 comes from the
  audio-side LM head; codebooks 1..7 come from a nested fast-AR
  :class:`HiggsAudioCodePredictor` that runs once per AR step using the last
  hidden state plus the previously emitted codebook embedding.

- Delay pattern: codebook k is staggered by k frames using
  ``audio_delay_token_id=128014`` as a filler at the LM-token positions where
  codebook k has not yet started emitting real codes. This matches
  ``HiggsAudioV2DelayPatternLogitsProcessor`` upstream and is the canonical
  MusicGen pattern ``[0, 1, 2, 3, 4, 5, 6, 7]``.

This file delivers the structural pieces (module classes, weight mapping,
forward-pass skeleton). The integration-test polish on the AR hot loop is
gated on the reference fixtures produced by
``examples/offline_inference/text_to_speech/higgs_audio_v2/reference_hf.py``
(AC-1, AC-2, AC-3). Until that lands the talker is registered in the model
registry but should be considered "structural scaffold" rather than
production-ready inference; this is documented in the round summary so the
RLCR loop tracks it as a known gap.
"""

from __future__ import annotations

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

from vllm_omni.model_executor.models.output_templates import OmniOutput

from vllm_omni.model_executor.models.higgs_audio_v2.configuration_higgs_audio_v2 import (
    HiggsAudioV2Config,
)

__all__ = [
    "DualFFNLayer",
    "HiggsAudioV2DecoderLayer",
    "HiggsAudioCodePredictor",
    "HiggsAudioV2TalkerForConditionalGeneration",
]

logger = init_logger(__name__)


class DualFFNLayer(nn.Module):
    """Routed FFN that runs the text or audio expert per token position.

    The routing follows the exact rule from
    ``transformers.models.higgs_audio_v2.HiggsAudioV2DecoderLayer.forward``:

      * Pre-attention norm: positions with ``audio_token_mask`` True use
        ``audio_input_layernorm``; the rest use ``input_layernorm``. The two
        outputs are stitched via ``masked_scatter`` and fed to a single
        self-attention.
      * Post-attention norm + FFN: text positions take the text path, audio
        positions take the audio path. Both deltas are ADDED to the residual
        (not replacing) so the residual stream remains shared.

    This module owns the four parallel sub-layers and a forward helper that
    consumes a precomputed routing mask.
    """

    def __init__(self, config: HiggsAudioV2Config, prefix: str = ""):
        super().__init__()
        rms_norm_eps = float(config.rms_norm_eps)
        hidden_size = int(config.hidden_size)
        intermediate_size = int(config.intermediate_size)
        hidden_act = str(config.hidden_act)
        mlp_bias = bool(getattr(config, "mlp_bias", False))

        # Two normalization pairs.  We use stock nn.RMSNorm-equivalents to keep
        # the parameter names aligned with the upstream state dict.
        self.input_layernorm = HiggsAudioRMSNorm(hidden_size, eps=rms_norm_eps)
        self.audio_input_layernorm = HiggsAudioRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = HiggsAudioRMSNorm(hidden_size, eps=rms_norm_eps)
        self.audio_post_attention_layernorm = HiggsAudioRMSNorm(hidden_size, eps=rms_norm_eps)

        # Two parallel MLPs.  vLLM's LlamaMLP fuses gate_proj+up_proj for TP;
        # we use it for both expert paths and rely on load_weights to repack
        # HF's separate gate/up tensors into the fused gate_up_proj.
        self.mlp = LlamaMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            bias=mlp_bias,
            prefix=f"{prefix}.mlp",
        )
        self.audio_mlp = LlamaMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            bias=mlp_bias,
            prefix=f"{prefix}.audio_mlp",
        )

    def pre_attention_norm(
        self,
        hidden_states: torch.Tensor,
        audio_token_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply input_layernorm / audio_input_layernorm based on the mask."""
        if audio_token_mask is None:
            return self.audio_input_layernorm(hidden_states)
        mask = audio_token_mask.to(hidden_states.device)
        out = hidden_states.clone()
        if mask.any():
            out = out.masked_scatter(
                mask.unsqueeze(-1),
                self.audio_input_layernorm(hidden_states[mask]).to(hidden_states.device),
            )
        if (~mask).any():
            out = out.masked_scatter(
                (~mask).unsqueeze(-1),
                self.input_layernorm(hidden_states[~mask]).to(hidden_states.device),
            )
        return out

    def post_attention_ffn(
        self,
        hidden_states: torch.Tensor,
        audio_token_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run text MLP on text positions and audio MLP on audio positions.

        Both expert outputs are ADDED to ``hidden_states`` (mirroring upstream).
        When the mask is None, audio expert is applied to all positions.
        """
        if audio_token_mask is None:
            audio_h = self.audio_post_attention_layernorm(hidden_states)
            audio_h = self.audio_mlp(audio_h)
            return hidden_states + audio_h
        mask = audio_token_mask.to(hidden_states.device)
        out = hidden_states.clone()
        if (~mask).any():
            text_h = self.post_attention_layernorm(hidden_states[~mask])
            text_h = self.mlp(text_h)
            out[~mask] = out[~mask] + text_h.to(out.dtype).to(out.device)
        if mask.any():
            audio_h = self.audio_post_attention_layernorm(hidden_states[mask])
            audio_h = self.audio_mlp(audio_h)
            out[mask] = out[mask] + audio_h.to(out.dtype).to(out.device)
        return out


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
        return (self.weight * hidden_states.to(input_dtype))


class HiggsAudioV2DecoderLayer(nn.Module):
    """One transformer block: vLLM-native attention + DualFFN-routed MLPs.

    Wraps a stock :class:`vllm.model_executor.models.llama.LlamaDecoderLayer`
    for the self-attention path (so PagedAttention KV caches keep working) and
    overlays the upstream HiggsAudioV2 DualFFN routing for the layernorm + MLP
    pairs. The implementation follows the upstream rule from
    ``transformers.models.higgs_audio_v2.HiggsAudioV2DecoderLayer.forward``
    described in ``UPSTREAM_TRACE.md``:

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

    def __init__(self, vllm_config: VllmConfig, prefix: str, config: HiggsAudioV2Config):
        super().__init__()
        self.config = config

        # vLLM-native attention + reference text-side norms / MLP. We reuse the
        # whole LlamaDecoderLayer scaffold so the attention sub-module is
        # wired into the engine's KV cache + RoPE machinery; only the forward
        # path below diverges from the canonical Llama one.
        self.base = LlamaDecoderLayer(vllm_config=vllm_config, prefix=f"{prefix}.base")

        # Parallel audio-expert norms + MLP (mirrors upstream parameter names).
        hidden_size = int(config.hidden_size)
        intermediate_size = int(config.intermediate_size)
        hidden_act = str(config.hidden_act)
        mlp_bias = bool(getattr(config, "mlp_bias", False))
        rms_norm_eps = float(config.rms_norm_eps)
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
    ) -> torch.Tensor:
        """Apply ``text_norm`` to non-mask positions and ``audio_norm`` to mask positions."""
        if audio_mask is None:
            # Bare audio context (no LM placeholder): all positions go through audio path.
            return audio_norm(hidden)
        mask_flat = audio_mask.reshape(-1)
        if hidden.ndim == 3:
            hidden_flat = hidden.reshape(-1, hidden.shape[-1])
        else:
            hidden_flat = hidden
        out = hidden_flat.clone()
        if (~mask_flat).any():
            text_out = text_norm(hidden_flat[~mask_flat])
            # vLLM RMSNorm forward(x) returns a single Tensor; some variants
            # return (Tensor, residual). Normalize to Tensor.
            if isinstance(text_out, tuple):
                text_out = text_out[0]
            out[~mask_flat] = text_out.to(out.dtype)
        if mask_flat.any():
            audio_out = audio_norm(hidden_flat[mask_flat])
            out[mask_flat] = audio_out.to(out.dtype)
        return out.reshape_as(hidden)

    def _routed_mlp(
        self,
        hidden: torch.Tensor,
        audio_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply ``audio_mlp`` to mask positions and ``base.mlp`` to non-mask positions."""
        if audio_mask is None:
            return self.audio_mlp(hidden)
        mask_flat = audio_mask.reshape(-1)
        if hidden.ndim == 3:
            hidden_flat = hidden.reshape(-1, hidden.shape[-1])
        else:
            hidden_flat = hidden
        out = torch.zeros_like(hidden_flat)
        if (~mask_flat).any():
            out[~mask_flat] = self.base.mlp(hidden_flat[~mask_flat]).to(out.dtype)
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
        # Stage 1: pre-attention norm, classical residual pattern.
        # (We deliberately skip the fused-residual optimization the stock
        # LlamaDecoderLayer uses; the DualFFN per-position split is incompatible
        # with the in-place residual mutation.)
        if residual is not None:
            hidden_states = hidden_states + residual
        residual = hidden_states
        attn_input = self._routed_norm(
            hidden_states,
            self.base.input_layernorm,
            self.audio_input_layernorm,
            audio_token_mask,
        )

        # Stage 2: shared self-attention. The vLLM LlamaAttention takes the
        # standard (positions, hidden_states) signature.
        attn_out = self.base.self_attn(positions=positions, hidden_states=attn_input)
        hidden_states = residual + attn_out

        # Stage 3: post-attention norm + dual MLP, classical residual pattern.
        residual = hidden_states
        mlp_input = self._routed_norm(
            hidden_states,
            self.base.post_attention_layernorm,
            self.audio_post_attention_layernorm,
            audio_token_mask,
        )
        mlp_out = self._routed_mlp(mlp_input, audio_token_mask)
        hidden_states = residual + mlp_out
        # We return ``None`` for the next-step residual so the next layer also
        # uses the classical pattern (residual is fully baked into hidden_states
        # at this point).
        return hidden_states, None


class HiggsAudioCodePredictor(nn.Module):
    """Fast-AR head for residual codebooks 1..N-1.

    At each AR step the talker produces a hidden state at the current audio
    position; this module emits codebooks 1..N-1 sequentially, each consuming
    the previously emitted codebook's embedding plus the running hidden state.

    The structure mirrors ``qwen3_tts.CodePredictorWrapper`` so existing
    talker plumbing (``embed_input_ids``, ``make_omni_output``, ``postprocess``)
    can be lifted with minimal changes during the AR-loop integration round.
    """

    def __init__(self, config: HiggsAudioV2Config, prefix: str = ""):
        super().__init__()
        self.config = config
        self.num_codebooks = int(config.num_codebooks)
        self.codebook_size = int(config.codebook_size)
        self.hidden_size = int(config.hidden_size)

        # Embedding for codebook k looks up among `codebook_size` entries.
        # We mirror upstream's HiggsAudioV2Embeddings layout: a single fused
        # embedding sized `num_codebooks * codebook_size`, addressed via a
        # per-codebook offset of ``k * codebook_size``.
        self.embed_codebooks = nn.Embedding(
            self.num_codebooks * self.codebook_size, self.hidden_size
        )
        self.register_buffer(
            "codebook_offsets",
            torch.arange(self.num_codebooks) * self.codebook_size,
            persistent=False,
        )

        # One output head per residual codebook (1..N-1). Codebook 0 is
        # emitted by the talker's audio LM head.
        self.residual_heads = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.codebook_size, bias=False)
                for _ in range(self.num_codebooks - 1)
            ]
        )

    def embed_codebook(self, codebook_idx: int, code_ids: torch.Tensor) -> torch.Tensor:
        """Embedding lookup for a single codebook with the offset applied."""
        offset = int(self.codebook_offsets[codebook_idx].item())
        return self.embed_codebooks(code_ids + offset)

    def predict_residual(self, hidden_state: torch.Tensor, codebook_idx: int) -> torch.Tensor:
        """Logits for residual codebook ``codebook_idx`` (1 <= idx < N)."""
        if codebook_idx < 1 or codebook_idx >= self.num_codebooks:
            raise IndexError(
                f"codebook_idx must be in [1, {self.num_codebooks - 1}]; got {codebook_idx}"
            )
        head = self.residual_heads[codebook_idx - 1]
        return head(hidden_state)


class HiggsAudioV2TalkerForConditionalGeneration(nn.Module):
    """Stage-0 talker class registered under
    ``HiggsAudioV2ForConditionalGeneration`` (canonical HF arch identifier)
    AND ``HiggsAudioV2TalkerForConditionalGeneration`` (explicit alias).

    The class wires together:
      1. vLLM-native ``LlamaModel`` backbone (PagedAttention scheduling).
      2. A parallel set of DualFFN modules (one per decoder layer) for the
         audio-expert path.
      3. Multi-codebook output: audio LM head (codebook 0) +
         :class:`HiggsAudioCodePredictor` (codebooks 1..N-1).
      4. ``load_weights`` HF -> vLLM mapping (GQA reshape, llama3 RoPE,
         text/audio MLP split, audio code heads, audio token embeddings).

    NOTE: The forward path that interleaves DualFFN routing with vLLM's
    compiled attention path is the largest single integration concern in this
    package and is gated on the reference fixtures + the upstream-trace
    runtime instrumentation (see ``UPSTREAM_TRACE.md``). This file lays down
    the structural scaffold; subsequent rounds will harden the forward path
    and the AR hot loop. The class is registered so the rest of the wiring
    (pipeline_registry, serving_speech, deploy yaml, stage_input_processor)
    can be smoke-tested end-to-end as soon as the runtime is exercised.
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
        # Round 7: directly instantiate the text-side embedding and final norm
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
        self.layers = nn.ModuleList(
            [
                HiggsAudioV2DecoderLayer(
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.layers.{i}",
                    config=self.config,
                )
                for i in range(self.config.num_hidden_layers)
            ]
        )

        # ------------------------------------------------------------------ audio embedding (shared)
        # Audio frames are embedded by summing per-codebook lookups; the LM
        # input_ids stream uses ``audio_token_id`` / ``audio_delay_token_id``
        # placeholders at the corresponding positions, and the talker
        # substitutes the audio embedding via ``masked_scatter`` before
        # entering the decoder stack.
        self.embed_audio_tokens = nn.Embedding(
            self.config.num_codebooks * self.config.codebook_size,
            self.config.hidden_size,
        )
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

        # Audio side: codebook-0 head emits 1026 logits; codebooks 1..7 come
        # from the fast-AR ``HiggsAudioCodePredictor`` below.
        self.audio_codebook0_head = nn.Linear(
            self.config.hidden_size, self.config.codebook_size, bias=False
        )
        self.code_predictor = HiggsAudioCodePredictor(
            self.config, prefix=f"{prefix}.code_predictor"
        )

        # vLLM's logits processor wires sampling metadata into the lm_head.
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

        # Round 7: share the audio embedding table with the code predictor's
        # per-codebook lookup. Upstream ships a single
        # ``model.embed_audio_tokens.embed_audio_tokens.weight`` tensor that
        # services both the prompt-side audio-frame substitution and the
        # residual-codebook-embedding path in the fast-AR predictor. Tying the
        # weights here means load_weights only needs to write the single
        # ``embed_audio_tokens.weight`` slot and the code predictor
        # automatically picks up the same tensor.
        self.code_predictor.embed_codebooks.weight = self.embed_audio_tokens.weight

    # ----------------------------------------------------------------- masks
    @torch.inference_mode()
    def audio_token_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute the per-position audio_token_mask used by DualFFN routing.

        Matches ``HiggsAudioV2Model.get_placeholder_mask``:

            mask = (input_ids == audio_token_id) | (input_ids == audio_delay_token_id)
        """
        return (input_ids == self.config.audio_token_id) | (
            input_ids == self.config.audio_delay_token_id
        )

    # ----------------------------------------------------------------- weights
    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
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
            HiggsAudioV2Embeddings) -> ``self.embed_audio_tokens.weight``.
          * Audio codebook heads: HF ``audio_lm_head.weight`` -> the fused 8x1026
            head from which Stage 0 reads codebook 0 directly and codebook 1..7
            via the fast-AR code predictor's residual heads. (Boson-ai's actual
            checkpoint exposes a single fused ``audio_lm_head`` of shape
            ``[num_codebooks * codebook_size, hidden]``, not one head per
            codebook; we split it into the per-codebook heads at load time.)

        Round-2 status: this routine covers the simple, fused-QKV, and fused-MLP
        transcriptions. The remaining gap (full Stage-0 forward integration with
        vLLM's PagedAttention scheduler so the loaded weights drive a working
        greedy decode) is tracked as an Open Issue in the goal tracker.
        """
        loaded: set[str] = set()
        own_params = dict(self.named_parameters())
        # R7 debug: log the first few HF keys we receive so we can verify the
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

            # Per-layer text-side params currently live one level deeper in our
            # HiggsAudioV2DecoderLayer (under ``.base``). Map ``layers.<L>.<tail>``
            # -> ``layers.<L>.base.<tail>`` when the tail belongs to the wrapped
            # LlamaDecoderLayer (input_layernorm / post_attention_layernorm /
            # self_attn.o_proj / mlp.down_proj).
            parts = normalized.split(".")
            if len(parts) >= 3 and parts[0] == "layers":
                tail = ".".join(parts[2:])
                if tail in (
                    "input_layernorm.weight",
                    "post_attention_layernorm.weight",
                    "self_attn.o_proj.weight",
                    "mlp.down_proj.weight",
                ):
                    base_name = f"layers.{parts[1]}.base.{tail}"
                    if base_name in own_params:
                        target = own_params[base_name]
                        if tuple(target.shape) == tuple(tensor.shape):
                            with torch.no_grad():
                                target.copy_(tensor)
                            loaded.add(base_name)
                            return True
                # audio_mlp.down_proj lives at our top-level audio_mlp slot.
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
            # audio code head: upstream has a fused [num_codebooks*codebook_size, hidden]
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
        # our HiggsAudioV2DecoderLayer stack (``layers.<L>.base...`` for text-
        # side; ``layers.<L>.audio_mlp...`` for the audio expert).
        for (layer_idx, slot, fused_kind), parts in fuse_buckets.items():
            if fused_kind == "gate_up_proj":
                gate = parts.get("gate_proj")
                up = parts.get("up_proj")
                if gate is None or up is None:
                    continue
                fused = torch.cat([gate, up], dim=0)
                if slot == "mlp":
                    target_name = f"layers.{layer_idx}.base.mlp.gate_up_proj.weight"
                else:  # audio_mlp
                    target_name = f"layers.{layer_idx}.audio_mlp.gate_up_proj.weight"
            elif fused_kind == "qkv_proj":
                q = parts.get("q_proj")
                k = parts.get("k_proj")
                v = parts.get("v_proj")
                if q is None or k is None or v is None:
                    continue
                fused = torch.cat([q, k, v], dim=0)
                target_name = f"layers.{layer_idx}.base.self_attn.qkv_proj.weight"
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
            "higgs_audio_v2 load_weights: %d/%d params initialized (sample seen: %s; unhandled count=%d, sample unhandled=%s)",
            len(loaded),
            len(own_params),
            debug_seen[:10],
            len(unhandled),
            unhandled[:20],
        )
        return loaded

    def _consume_fused_audio_head(
        self, fused: torch.Tensor, loaded: set[str]
    ) -> None:
        """Split the fused ``audio_lm_head[num_codebooks*codebook_size, hidden]``
        into ``audio_codebook0_head`` + the code predictor's residual heads.
        """
        own_params = dict(self.named_parameters())
        num_codebooks = int(self.config.num_codebooks)
        codebook_size = int(self.config.codebook_size)
        hidden = int(self.config.hidden_size)
        if tuple(fused.shape) != (num_codebooks * codebook_size, hidden):
            logger.warning(
                "higgs_audio_v2: unexpected audio_lm_head shape %s; expected %s",
                tuple(fused.shape),
                (num_codebooks * codebook_size, hidden),
            )
            return
        chunks = fused.split(codebook_size, dim=0)
        if "audio_codebook0_head.weight" in own_params:
            with torch.no_grad():
                own_params["audio_codebook0_head.weight"].copy_(chunks[0])
            loaded.add("audio_codebook0_head.weight")
        for k, chunk in enumerate(chunks[1:], start=1):
            name = f"code_predictor.residual_heads.{k - 1}.weight"
            if name in own_params:
                with torch.no_grad():
                    own_params[name].copy_(chunk)
                loaded.add(name)

    # --------------------------------------------------------------- helpers
    def _map_simple_name(self, hf_name: str) -> str | None:
        """Translate the simple (non-fused) HF parameter names to vLLM names."""
        # Upstream nests the audio embedding as
        # ``model.embed_audio_tokens.embed_audio_tokens.weight``; the boson-ai
        # checkpoint also uses this exact key. Accept both nested and flat.
        if hf_name in (
            "embed_audio_tokens.weight",
            "embed_audio_tokens.embed_audio_tokens.weight",  # stripped form
            "model.embed_audio_tokens.embed_audio_tokens.weight",  # raw HF form
            "model.embed_audio_tokens.weight",
            # Actual key used in the boson-ai checkpoint:
            "audio_codebook_embeddings.weight",
            "model.audio_codebook_embeddings.weight",
        ):
            return "embed_audio_tokens.weight"
        if hf_name in (
            "lm_head.weight",
            "text_lm_head.weight",
            "model.text_lm_head.weight",
            # Actual key used in the boson-ai checkpoint:
            "audio_decoder_proj.text_lm_head.weight",
            "model.audio_decoder_proj.text_lm_head.weight",
        ):
            return "lm_head.weight"
        # Round 7: dropped LlamaModel wrapper; text embedding + final norm now
        # live directly on the talker (self.embed_tokens / self.norm).
        if hf_name == "model.embed_tokens.weight":
            return "embed_tokens.weight"
        if hf_name == "model.norm.weight":
            return "norm.weight"
        # Per-codebook head, when the upstream exports it un-fused.
        if hf_name == "codebook_head_0.weight":
            return "audio_codebook0_head.weight"
        if hf_name.startswith("codebook_head_") and hf_name.endswith(".weight"):
            try:
                idx = int(hf_name[len("codebook_head_") : -len(".weight")])
            except ValueError:
                return None
            if idx >= 1:
                return f"code_predictor.residual_heads.{idx - 1}.weight"
        # Per-layer audio norms -> our HiggsAudioV2DecoderLayer side.
        if hf_name.startswith("model.layers.") and hf_name.endswith(
            ".audio_input_layernorm.weight"
        ):
            layer_idx = hf_name.split(".")[2]
            return f"layers.{layer_idx}.audio_input_layernorm.weight"
        if hf_name.startswith("model.layers.") and hf_name.endswith(
            ".audio_post_attention_layernorm.weight"
        ):
            layer_idx = hf_name.split(".")[2]
            return f"layers.{layer_idx}.audio_post_attention_layernorm.weight"
        # Per-layer text norms -> the wrapped LlamaDecoderLayer's slots.
        if hf_name.startswith("model.layers.") and hf_name.endswith(
            ".input_layernorm.weight"
        ):
            layer_idx = hf_name.split(".")[2]
            return f"layers.{layer_idx}.base.input_layernorm.weight"
        if hf_name.startswith("model.layers.") and hf_name.endswith(
            ".post_attention_layernorm.weight"
        ):
            layer_idx = hf_name.split(".")[2]
            return f"layers.{layer_idx}.base.post_attention_layernorm.weight"
        # Per-layer non-fused projections (Round 7 fix): o_proj for attention
        # and the down_proj of both MLP experts pass through unchanged in
        # shape, so they map directly to the corresponding slot on our
        # HiggsAudioV2DecoderLayer.
        if hf_name.startswith("model.layers.") and hf_name.endswith(
            ".self_attn.o_proj.weight"
        ):
            layer_idx = hf_name.split(".")[2]
            return f"layers.{layer_idx}.base.self_attn.o_proj.weight"
        if hf_name.startswith("model.layers.") and hf_name.endswith(
            ".mlp.down_proj.weight"
        ):
            layer_idx = hf_name.split(".")[2]
            return f"layers.{layer_idx}.base.mlp.down_proj.weight"
        if hf_name.startswith("model.layers.") and hf_name.endswith(
            ".audio_mlp.down_proj.weight"
        ):
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
        **_: Any,
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

        audio_mask = self.audio_token_mask(input_ids) if input_ids is not None else None

        residual: torch.Tensor | None = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                audio_token_mask=audio_mask,
            )

        # Final norm. vLLM's compiled RMSNorm returns ``(hidden, residual)`` when
        # called with the fused signature; we call it with the single-tensor
        # signature since our layers already baked residual into hidden_states.
        norm_out = self.norm(hidden_states)
        if isinstance(norm_out, tuple):
            norm_out = norm_out[0]
        return norm_out

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Any = None,
        *,
        audio_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sampler-compatible logits tensor.

        Returns a single ``[N, vocab_size]`` tensor that the stock
        vLLM AR runner pipes through ``.contiguous()`` and a generic
        sampler. For Round-5 the contract is intentionally tensor-only:

        - Text-position logits come from ``lm_head`` (128256-wide Llama vocab).
        - Audio-position logits at audio_token_id placeholders are also
          read from ``lm_head`` for now; the per-position codebook-0
          routing path described in ``UPSTREAM_TRACE.md`` lives in
          :meth:`audio_codebook0_logits` and is consumed by a separate
          codebook-0 sampling adapter (round-6 follow-up). The previous
          dict-return form broke the AR runner's
          ``compute_logits(...).contiguous()`` contract; this restores
          tensor-only output so the engine can boot the talker even before
          the dedicated audio-sampler dispatch lands.

        Call :meth:`audio_codebook0_logits` for the audio-side 1026-wide
        head; that helper is exercised by unit tests + the upcoming
        sampler dispatch.
        """
        _ = audio_token_mask  # The mask is consumed by audio_codebook0_logits, not here.
        # Stash for the audio-sampler dispatch in :meth:`sample` (round-7+).
        # Limited to a single step at a time; cleared in :meth:`sample` after
        # consumption so a missed call doesn't leak stale tensors.
        self._last_logits_hidden = hidden_states
        return self.logits_processor(self.lm_head, hidden_states, sampling_metadata)

    def audio_codebook0_logits(
        self,
        hidden_states: torch.Tensor,
        audio_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Codebook-0 logits at audio positions, ``[N_audio, codebook_size]``.

        Separated from :meth:`compute_logits` so the AR runner can pipe a
        tensor through the stock sampler while a dedicated codebook-0
        sampling adapter consumes this tensor in parallel. Returns an
        empty tensor when ``audio_token_mask`` selects no positions.
        """
        mask = audio_token_mask.reshape(-1).to(hidden_states.device)
        hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        if not mask.any():
            return torch.empty(
                (0, int(self.config.codebook_size)),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        return self.audio_codebook0_head(hidden_flat[mask])

    def predict_audio_residual_codebooks(
        self,
        hidden_states: torch.Tensor,
        codebook0_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Emit codebooks 1..N-1 for audio positions via the fast-AR predictor.

        Inputs:
            hidden_states: ``[N_audio, hidden]`` -- last hidden state at each audio position.
            codebook0_ids: ``[N_audio]`` -- the sampled codebook-0 token id for each position.

        Returns:
            ``[N_audio, num_codebooks - 1]`` int64 tensor of predicted codes.
        """
        if hidden_states.ndim != 2:
            raise ValueError(f"hidden_states must be [N, hidden]; got shape {tuple(hidden_states.shape)}")
        n_audio = int(hidden_states.shape[0])
        out = torch.empty(
            (n_audio, self.config.num_codebooks - 1), dtype=torch.long, device=hidden_states.device
        )
        num_real = int(getattr(self.config, "num_real_codes", self.config.codebook_size))
        running_hidden = hidden_states
        prev_codes = codebook0_ids
        for k in range(1, self.config.num_codebooks):
            # Inject the previous codebook embedding as a residual into the
            # running hidden state, then predict the next codebook's logits.
            cb_embed = self.code_predictor.embed_codebook(k - 1, prev_codes)  # [N, hidden]
            running_hidden = running_hidden + cb_embed
            logits = self.code_predictor.predict_residual(running_hidden, k)  # [N, codebook_size]
            # Mask stream specials so argmax stays in the real-code range.
            if logits.shape[-1] > num_real:
                logits = logits.clone()
                logits[:, num_real:] = float("-inf")
            next_codes = torch.argmax(logits, dim=-1)
            out[:, k - 1] = next_codes
            prev_codes = next_codes
        return out

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        """Embed the LM token stream, swapping in audio embeddings at placeholders.

        The vLLM runner calls this before forward(); the embedding produced
        here is what the decoder stack consumes. For text positions we use
        the standard ``embed_tokens`` lookup; for audio positions (input id
        equals ``audio_token_id`` or ``audio_delay_token_id``) we leave the
        text embedding in place -- the actual audio embedding lookup is
        deferred to the engine path that has ``audio_input_ids`` in scope.
        Until the live engine wires audio_input_ids through, the text
        embedding at placeholder positions acts as a deterministic dummy
        that the DualFFN audio expert still routes correctly.
        """
        return self.embed_tokens(input_ids)

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

        audio_codes_list: list[torch.Tensor] = []
        for info in info_dicts:
            if not isinstance(info, dict):
                continue
            codes_field = info.get("codes")
            if isinstance(codes_field, dict):
                ac = codes_field.get("audio")
            else:
                ac = info.get("audio_codes")
            if isinstance(ac, torch.Tensor) and ac.numel() > 0:
                audio_codes_list.append(ac)

        if audio_codes_list:
            audio_codes = torch.cat(audio_codes_list, dim=0)
            span_len = int(audio_codes.shape[0])
            if hidden is not None and span_len <= int(hidden.shape[0]):
                hidden = hidden[:span_len]
            return OmniOutput(
                text_hidden_states=hidden,
                multimodal_outputs={"codes": {"audio": audio_codes}},
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
        """Model-owned sampler with audio-codebook-0 dispatch.

        - Run the stock vLLM sampler against the text-vocab logits to produce
          per-position token ids. These already cover the standard text path.
        - Detect positions where the *sampled* token is an audio placeholder
          (``audio_token_id`` or ``audio_delay_token_id``). At those positions
          we sample codebook 0 from :meth:`audio_codebook0_logits` and run
          :meth:`predict_audio_residual_codebooks` for codebooks 1..7. The
          resulting ``[N_audio, 8]`` tensor is stashed on the talker so the
          per-request ``postprocess`` hook can publish it under
          ``model_intermediate_buffer[req_id]["codes"]["audio"]``.

        Hidden states are the ones cached by :meth:`compute_logits`.
        """
        sampler = getattr(self, "_stock_sampler", None)
        if sampler is None:
            from vllm.v1.sample.sampler import Sampler

            sampler = Sampler()
            self._stock_sampler = sampler

        # Audio-mode logits bias: when the previous token (last in
        # output_token_ids[i] if non-empty, else last in prompt_token_ids[i])
        # is ``audio_out_bos`` (=audio_bos_token_id) or ``audio_token_id``, the
        # next emitted token must be either another ``audio_token_id`` (audio
        # frame placeholder) or ``audio_eos`` (stop). Mask everything else.
        self._apply_audio_mode_bias(logits, sampling_metadata)
        sampler_output = sampler(logits=logits, sampling_metadata=sampling_metadata)

        hidden = getattr(self, "_last_logits_hidden", None)
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
            # Shape mismatch (e.g. spec-decode draft path); skip audio dispatch.
            self._last_audio_codes = None
            return sampler_output

        audio_token_id = int(self.config.audio_token_id)
        audio_delay_id = int(self.config.audio_delay_token_id)
        is_audio = (sampled_flat == audio_token_id) | (sampled_flat == audio_delay_id)
        if not bool(is_audio.any()):
            self._last_audio_codes = None
            return sampler_output

        audio_hidden = hidden[is_audio]
        cb0_logits = self.audio_codebook0_head(audio_hidden)
        # Mask stream-special positions so argmax stays in the real-code range
        # ``[0, num_real_codes)``. Without this the per-codebook argmax can
        # land on ``audio_stream_bos_id`` (1024) or ``audio_stream_eos_id``
        # (1025), and the stage-input adapter's :func:`_extract_last_frame`
        # filter drops those frames entirely.
        num_real = int(getattr(self.config, "num_real_codes", cb0_logits.shape[-1]))
        if cb0_logits.shape[-1] > num_real:
            cb0_logits = cb0_logits.clone()
            cb0_logits[:, num_real:] = float("-inf")
        cb0 = torch.argmax(cb0_logits, dim=-1)
        residual = self.predict_audio_residual_codebooks(audio_hidden, cb0)
        # Defensive clamp on residual codebooks too.
        residual = residual.clamp_(max=num_real - 1).clamp_(min=0)
        codes_flat = torch.cat([cb0.unsqueeze(1), residual], dim=1).to(torch.long)

        # Scatter [N_audio, 8] back to a [N, 8] tensor (-1 at non-audio rows)
        # so postprocess can slice per-request.
        num_codebooks = int(self.config.num_codebooks)
        codes_full = torch.full(
            (int(sampled_flat.numel()), num_codebooks),
            -1,
            dtype=torch.long,
            device=hidden.device,
        )
        codes_full[is_audio] = codes_flat
        self._last_audio_codes = codes_full
        self._postprocess_cursor = 0
        return sampler_output

    def _apply_audio_mode_bias(self, logits: torch.Tensor, sampling_metadata: Any) -> int:
        """Mask non-audio tokens at audio-mode positions, in-place on ``logits``.

        Heuristic: per-request, find the last token seen so far (last of
        ``output_token_ids[i]`` if non-empty, else last of
        ``prompt_token_ids[i]``). If that token is ``audio_bos_token_id`` or
        ``audio_token_id``, force the next emit to be one of
        ``{audio_token_id, audio_eos_token_id}``. This unblocks live audio
        generation since the un-biased argmax over the 128k vocab favours
        text-token-id ranges (in particular the global ``eos_token_id``).

        We deliberately do NOT include the standard ``eos_token_id`` in the
        allowed set — letting it through would end the whole sequence at the
        first step. Stopping inside the audio span is the ``audio_eos`` token's
        job; once it fires we drop the bias and the next call falls back to
        the stock sampler.
        """
        if logits is None or logits.ndim != 2:
            return 0
        audio_bos = int(self.config.audio_bos_token_id)
        audio_id = int(self.config.audio_token_id)
        # Force ``audio_token_id`` only — including ``audio_eos`` in the allow
        # set lets greedy decode pick the EOS at step 1 (the model's unbiased
        # logit for the audio placeholder is weak relative to neighbouring
        # special tokens). The framework's ``max_tokens`` is the stop bound
        # while the codebook-emission contract stabilises (R8). When the
        # upstream sampler is faithful we'll add ``audio_eos`` back here.
        allowed_extra: list[int] = []
        # Walk per-request to decide which rows to mask.
        prompt_ids = getattr(sampling_metadata, "prompt_token_ids", None)
        output_ids = getattr(sampling_metadata, "output_token_ids", None)
        num_rows = int(logits.shape[0])
        biased = 0
        # Fallback "previous token" source: input_ids stashed by :meth:`forward`.
        # The runner concatenates per-request input_ids; ``logits`` rows are
        # produced from logits_indices which (for the common AR case) are the
        # last position of each request slice. So input_ids' tail matches the
        # logits' rows in order.
        stash_ids = getattr(self, "_last_step_input_ids", None)
        stash_tail: list[int] | None = None
        if isinstance(stash_ids, torch.Tensor) and stash_ids.numel() >= num_rows:
            stash_tail = stash_ids[-num_rows:].detach().to("cpu").tolist()
        for i in range(num_rows):
            prev: int | None = None
            if output_ids is not None and i < len(output_ids):
                hist = output_ids[i]
                if hist:
                    prev = int(hist[-1])
            if prev is None and prompt_ids is not None:
                # prompt_token_ids may be a tensor or a list-of-lists
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
            # Mask all logits at this row except the audio + EOS tokens.
            allowed = {audio_id, *allowed_extra}
            row = logits[i]
            mask = torch.full_like(row, float("-inf"))
            for tok in allowed:
                if 0 <= tok < row.shape[-1]:
                    mask[tok] = row[tok]
            logits[i].copy_(mask)
            biased += 1
        return biased

    has_postprocess: bool = True

    def postprocess(
        self,
        hidden_states_slice: torch.Tensor,
        multimodal_outputs: Any = None,
        **req_infos: Any,
    ) -> dict[str, Any]:
        """Publish per-request audio codes into model_intermediate_buffer.

        The runner calls this once per request, passing the request's slice of
        hidden states. We use the slice's length to pick out the matching rows
        from ``self._last_audio_codes`` (a per-batch [N_total, num_codebooks]
        tensor scattered by :meth:`sample`). The returned dict is merged into
        ``model_intermediate_buffer[req_id]`` for downstream Stage-1 use.
        """
        _ = multimodal_outputs  # consumed by the runner directly
        codes_full = getattr(self, "_last_audio_codes", None)
        if codes_full is None:
            return {}

        # The runner walks requests in batch order; pop from the cursor as we go.
        cursor = int(getattr(self, "_postprocess_cursor", 0))
        n = int(hidden_states_slice.shape[0])
        end = cursor + n
        if end > int(codes_full.shape[0]):
            # Defensive: shape drift between forward and postprocess.
            self._postprocess_cursor = 0
            return {}
        slice_codes = codes_full[cursor:end]
        self._postprocess_cursor = end
        # Drop placeholder rows (-1) — only emit codes for actual audio positions.
        audio_rows = slice_codes[:, 0] >= 0
        if not bool(audio_rows.any()):
            return {}
        new_codes = slice_codes[audio_rows].to(torch.int32)

        # Append to any existing codes.audio in the runner buffer (req_infos)
        # so that codes accumulate across decode steps; without this, each
        # postprocess overwrite drops earlier frames.
        existing = req_infos.get("codes")
        prior = None
        if isinstance(existing, dict):
            cand = existing.get("audio")
            if isinstance(cand, torch.Tensor) and cand.numel() > 0:
                prior = cand.to(device=new_codes.device, dtype=new_codes.dtype)
        codes_out = (
            torch.cat([prior, new_codes], dim=0) if prior is not None else new_codes
        )
        return {"codes": {"audio": codes_out}}
