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
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.llama import LlamaMLP, LlamaModel

from vllm_omni.model_executor.models.higgs_audio_v2.configuration_higgs_audio_v2 import (
    HiggsAudioV2Config,
)

__all__ = [
    "DualFFNLayer",
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

        # ------------------------------------------------------------------ backbone
        # We reuse vLLM's compiled LlamaModel for the attention path. The
        # MLP/LayerNorm slots inside each decoder layer are owned by
        # ``LlamaDecoderLayer``; the DualFFN audio expert is held alongside
        # in ``self.dual_ffns`` (one module per layer) so weight loading and
        # the bypass-the-MLP path remain explicit.
        self.model = LlamaModel(vllm_config=vllm_config, prefix=f"{prefix}.model")

        # ------------------------------------------------------------------ DualFFN tier
        # One DualFFN module per transformer layer. The text-side parameters
        # are also held by ``self.model.layers[i].mlp`` / ``input_layernorm``
        # / ``post_attention_layernorm``; we mirror them in DualFFN purely for
        # the audio expert side and for the upstream-aligned weight names.
        # During forward, the audio expert is applied as a delta on top of
        # the LlamaModel output for positions where audio_token_mask is True.
        self.dual_ffns = nn.ModuleList(
            [
                DualFFNLayer(self.config, prefix=f"{prefix}.dual_ffns.{i}")
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

        # First pass: collect tensors that need fusing (q/k/v, gate/up for each
        # MLP slot). We accumulate into per-(layer, slot, kind) buckets and
        # commit the fused tensor once all parts have arrived.
        fuse_buckets: dict[tuple[int, str, str], dict[str, torch.Tensor]] = {}

        def _try_simple(name: str, tensor: torch.Tensor) -> bool:
            mapped = self._map_simple_name(name)
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
            parts = name.split(".")
            if len(parts) < 5 or parts[0] != "model" or parts[1] != "layers":
                return False
            try:
                layer_idx = int(parts[2])
            except ValueError:
                return False
            tail = ".".join(parts[3:])
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

        for name, tensor in weights:
            if _try_simple(name, tensor):
                continue
            if _stash_fusion(name, tensor):
                continue
            # audio code head: upstream has a fused [num_codebooks*codebook_size, hidden]
            # head; split into codebook-0 head + residual heads.
            if name in ("audio_lm_head.weight",):
                self._consume_fused_audio_head(tensor, loaded)
                continue

        # Second pass: emit fused tensors.
        for (layer_idx, slot, fused_kind), parts in fuse_buckets.items():
            if fused_kind == "gate_up_proj":
                gate = parts.get("gate_proj")
                up = parts.get("up_proj")
                if gate is None or up is None:
                    continue
                fused = torch.cat([gate, up], dim=0)
                if slot == "mlp":
                    target_name = f"model.layers.{layer_idx}.mlp.gate_up_proj.weight"
                else:
                    target_name = f"dual_ffns.{layer_idx}.audio_mlp.gate_up_proj.weight"
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
            elif fused_kind == "qkv_proj":
                q = parts.get("q_proj")
                k = parts.get("k_proj")
                v = parts.get("v_proj")
                if q is None or k is None or v is None:
                    continue
                fused = torch.cat([q, k, v], dim=0)
                target_name = f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"
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
            "model.embed_audio_tokens.embed_audio_tokens.weight",
            "model.embed_audio_tokens.weight",
        ):
            return "embed_audio_tokens.weight"
        if hf_name in ("lm_head.weight", "text_lm_head.weight"):
            return "lm_head.weight"
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
        # Per-layer audio norms.
        if hf_name.startswith("model.layers.") and hf_name.endswith(
            ".audio_input_layernorm.weight"
        ):
            layer_idx = hf_name.split(".")[2]
            return f"dual_ffns.{layer_idx}.audio_input_layernorm.weight"
        if hf_name.startswith("model.layers.") and hf_name.endswith(
            ".audio_post_attention_layernorm.weight"
        ):
            layer_idx = hf_name.split(".")[2]
            return f"dual_ffns.{layer_idx}.audio_post_attention_layernorm.weight"
        if hf_name.startswith("model.layers.") and hf_name.endswith(
            ".input_layernorm.weight"
        ):
            layer_idx = hf_name.split(".")[2]
            return f"dual_ffns.{layer_idx}.input_layernorm.weight"
        if hf_name.startswith("model.layers.") and hf_name.endswith(
            ".post_attention_layernorm.weight"
        ):
            layer_idx = hf_name.split(".")[2]
            return f"dual_ffns.{layer_idx}.post_attention_layernorm.weight"
        return None

    # ----------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass.

        Status (Round 2): the DualFFN audio expert is NOT yet wired into the
        compiled LlamaModel forward path. Returning the bare LlamaModel output
        without DualFFN routing would silently produce wrong codebook
        predictions (the audio_mlp / audio_input_layernorm / audio_post_attention_layernorm
        weights would be loaded but unused), so we raise here instead of
        returning a misleading "structural" hidden state.

        Round 3 will replace this raise with the routed path described in
        UPSTREAM_TRACE.md (DualFFN audio expert applied per layer at positions
        where audio_token_mask is True; multi-codebook output via
        ``audio_codebook0_head`` + ``code_predictor``). The weight loader
        already places the audio-side parameters at the correct sites
        (``dual_ffns.<L>.audio_mlp.gate_up_proj``, ``audio_input_layernorm``,
        ``audio_post_attention_layernorm``, ``audio_codebook0_head``,
        ``code_predictor.residual_heads.<k>``), so this is a pure forward-path
        integration task with no remaining state-dict mapping work.
        """
        raise NotImplementedError(
            "higgs_audio_v2 Stage-0 forward path is gated on the DualFFN "
            "per-layer routing integration with vLLM's compiled LlamaModel. "
            "Round 2 ships the weight mapping (fused QKV, fused gate_up_proj "
            "for both MLP experts, audio_lm_head split into codebook heads) "
            "and reference fixtures. Round 3 will wire the routed forward. "
            "See vllm_omni/model_executor/models/higgs_audio_v2/UPSTREAM_TRACE.md "
            "for the exact routing rule the integration must reproduce."
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Any,
    ) -> torch.Tensor:
        """Standard vLLM-style logits computation against the text LM head.

        At audio positions, the talker emits codebook-0 via
        :attr:`audio_codebook0_head` (and codebooks 1..7 via the code
        predictor) instead; see ``forward`` for that branch.
        """
        return self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
