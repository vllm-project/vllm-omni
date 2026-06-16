# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CSM-1B Stage-0 backbone AR model (with the 31-step depth decoder INSIDE).

This is Stage 0 of the 2-stage CSM pipeline (A3 design §2). It runs the CSM
backbone as a vLLM-native ``LlamaModel`` under PagedAttention, samples codebook-0
(cb0) per 80 ms frame, runs the 31-step depth decoder inline to produce cb1..cb31,
composes the next-frame Sigma-embedding, caches it per request, and surfaces the
finished 32-code frame forward to Stage 1 (Mimi vocoder) as a latent payload.

WHY 2-stage / why this shape is illegal-address-proof (A3 design §1, §3)
------------------------------------------------------------------------
The single-stage model (csm.py ``inference_stream``, d92e35f7) drove the backbone
with a PRIVATE python loop that advanced its own ``positions`` thousands of times
per request, OUTSIDE the vLLM scheduler. vLLM derives every KV slot purely from
``positions`` + the per-request ``block_table``; positions the scheduler never
allocated scatter ``reshape_and_cache`` writes into other requests' KV blocks ->
silent heap corruption -> the deferred ``cudaErrorIllegalAddress`` on the next
request's first kernel. That is NOT one clamp away from working.

The fix (this file) is structural and mirrors MiMo-Audio (``mimo_audio_llm.py``,
the feedback twin) + Qwen3-TTS (``qwen3_tts_talker.py``, the engine-contract
twin):

  1. Exactly ONE real, in-vocab token per frame stays in the sequence (cb0). vLLM
     schedules it as a normal 1-token decode; the scheduler owns ``positions`` /
     ``block_table`` / ``slot_mapping``. The model only overwrites the token's
     EMBEDDING value -- never a position. (Qwen3 invariant verbatim:
     ``qwen3_tts_talker.py:687-689``.)
  2. The composed Sigma-embedding is delivered through the supported decode-time
     ``preprocess`` hook (``has_preprocess=True``); the runner writes the returned
     embedding back at the request's exact span
     (``gpu_model_runner.py:1536-1563``). Positions stay runner-owned.
  3. ``forward()`` runs the backbone EXACTLY once per step -> h_t; cb0 via the
     ``LogitsProcessor`` (NOT a direct ``ParallelLMHead.forward`` -- that was the
     b3c bug); then the 31-step depth loop INLINE (de-risked eager option (b),
     A3 §6). The scheduler advances one frame per step.
  4. Feedback via per-request cache + re-inject: ``forward()`` computes the
     Sigma-embed and stashes it in ``self._cached_sigma_by_req[req_id]``; the next
     ``preprocess`` call adds it onto the in-vocab token's base embedding. Because
     the next step is also a real scheduled token, its KV slot is always
     allocated -> no IMA. (MiMo cache+reinject:
     ``mimo_audio_llm.py:939-969`` / ``:1112-1131``.)

Every KV write therefore lands at a scheduler-allocated slot. The model's only
GPU influence is the VALUE of ``inputs_embeds`` and of the sampled next token --
never the position arithmetic. This is the mechanism MiMo-Audio ships today.

Weight loading (A3 §5, B2 fix): the ``sesame/csm-1b`` checkpoint is split by
prefix. ``backbone_model.* / lm_head / *embed*`` and ``depth_decoder.*`` are
loaded HERE (Stage 0). ``codec_model.*`` is drained here and (re)loaded by Stage 1
(``csm_mimi``) from the same checkpoint. Every returned name is a REAL registered
parameter name so vLLM's load-completeness validator passes.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.utils import maybe_prefix

from vllm_omni.model_executor.models.csm.configuration_csm import (
    CsmConfig,
    build_backbone_llama_config,
)
from vllm_omni.model_executor.models.csm.csm_depth import (
    CsmDepthDecoder,
    sample_logits,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

__all__ = ["CsmBackboneForConditionalGeneration"]

# Default sampling parameters (public CSM-1B / Mimi facts).
_DEFAULT_TEMPERATURE = 0.9
_DEFAULT_TOP_K = 50

# Public CSM/Mimi frame constants.
_NUM_CODEBOOKS = 32
_CODEBOOK_EOS_ID = 0  # codebook_eos_token_id; a frame with cb0..cb30==0 is EOS.

# Hard frame-cap fallback (GATE-B follow-up). One scheduler decode step == one
# 80 ms frame, so the AR loop length is bounded by SamplingParams.max_tokens.
# But CSM-1B at do_sample has a non-terminating greedy/repetition attractor that
# can keep emitting non-EOS frames indefinitely; if the scheduler stop on
# max_tokens ever races the in-flight async streaming step, a request can run
# PAST its cap (observed: 113 frames at cap 64). This model-owned counter is the
# fail-closed guard: forward() counts emitted frames per request and
# compute_logits forces the frame EOS (token id 0) the step the count reaches the
# cap, so the request ALWAYS stops at the cap even with no natural EOS frame.
# 2048 mirrors the deploy default_sampling_params max_tokens for Stage 0.
_DEFAULT_MAX_FRAMES = 2048


def _pick(info: dict, key: str, default):
    """Extract scalar from additional_information dict (list or plain value)."""
    val = info.get(key, default)
    if isinstance(val, (list, tuple)) and len(val) > 0:
        return val[0]
    return val if val is not None else default


def _req_key(info: dict[str, Any]) -> str:
    """Resolve the per-request key (I5).

    The runner sets ``request_id`` for the generic preprocess path
    (``gpu_model_runner.py:1521``) and passes the same scheduler ids to
    ``on_requests_finished`` (``gpu_ar_model_runner.py:379-380``). ``_omni_req_id``
    is a legacy fallback that is never set in this runner; we still honor it (I5
    canonical name) plus ``global_request_id`` so the key used at re-inject time
    matches the key freed at finish time.
    """
    return str(
        info.get("request_id") or info.get("global_request_id") or info.get("_omni_req_id") or info.get("req_id") or "0"
    )


class CsmBackbone(nn.Module):
    """The CSM-1B backbone, rebuilt on vLLM-native Llama layers (REUSE: csm.py:116).

    Wraps :class:`vllm.model_executor.models.llama.LlamaModel` (PagedAttention,
    fused QKV, RoPE) driven by a synthetic ``LlamaConfig`` from ``CsmConfig``. The
    cb0 logits head is ``cb0_head`` (the untied ``lm_head`` in the checkpoint).
    """

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        csm_config: CsmConfig = vllm_config.model_config.hf_config
        self.csm_config = csm_config

        backbone_llama_config = build_backbone_llama_config(csm_config)
        # Swap the synthetic LlamaConfig in so vLLM's native LlamaModel sees a
        # standard Llama config (parent caches the CsmConfig first).
        vllm_config.model_config.hf_config = backbone_llama_config
        vllm_config.model_config.hf_text_config = backbone_llama_config

        self.model = LlamaModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.cb0_head = ParallelLMHead(
            csm_config.vocab_size,
            csm_config.hidden_size,
            prefix=maybe_prefix(prefix, "cb0_head"),
        )
        self.logits_processor = LogitsProcessor(csm_config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One backbone forward (the Sigma-embed is composed by the caller)."""
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Delegate the backbone decoder layers + norm to vLLM's LlamaModel."""
        return self.model.load_weights(weights)


class CsmBackboneForConditionalGeneration(nn.Module):
    """CSM-1B Stage-0 backbone AR model + inline 31-step depth decoder.

    Contract surface (A3 §4.3):
      * ``has_preprocess`` / ``has_postprocess`` -> the runner delivers the
        composed inputs_embeds via ``preprocess`` and captures h_t via
        ``postprocess``.
      * ``have_multimodal_outputs`` -> ``forward`` returns an ``OmniOutput`` whose
        ``multimodal_outputs`` carries the per-step 32-code frame as a latent
        payload (``codes.audio``), surfaced forward to Stage 1.
      * cb0 via ``logits_processor`` (NOT direct LMHead -- the b3c bug).
      * Real frame EOS: cb0..cb30 all-zero (A3 §5; the *real* stop authority,
        replacing the single-stage fake ``compute_logits`` EOS grid).

    NOTE: this model deliberately does NOT declare ``talker_mtp_output_key`` /
    ``mtp_hidden_size``. That keeps ``has_talker_mtp`` False on the runner so the
    runner takes the plain ``preprocess`` path and the depth loop runs INSIDE this
    model's ``forward()`` (the de-risked eager option (b), A3 §6) rather than via
    the runner's single-shot MTP residual fast-path (which assumes a fixed-length
    graph-safe predictor -- CSM's depth is a genuine 31-step sequential AR).
    """

    requires_raw_input_tokens = True
    have_multimodal_outputs = True
    has_preprocess = True
    has_postprocess = True
    enable_update_additional_information = True
    inject_omni_request_id_into_runtime_info = True

    packed_modules_mapping = CsmBackbone.packed_modules_mapping

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config: CsmConfig = vllm_config.model_config.hf_config
        self.model_path: str = vllm_config.model_config.model

        self.num_codebooks = int(getattr(self.config, "num_codebooks", _NUM_CODEBOOKS))
        self.hidden_size = int(getattr(self.config, "hidden_size", 2048))

        # Backbone (vLLM-native layers). Constructing it mutates
        # vllm_config.model_config.hf_config into the synthetic LlamaConfig, so we
        # cache the CsmConfig above first.
        self.backbone = CsmBackbone(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "backbone"),
        )

        # Aux modules (built in load_weights, after distributed init).
        self._depth_decoder: nn.Module | None = None  # CsmDepthDecoderForCausalLM
        self._frame_embed: nn.Module | None = None  # CsmBackboneModelEmbeddings
        self._text_embed: nn.Module | None = None  # nn.Embedding (text prompt)
        self._hf_config: Any = None
        self._device: torch.device | None = None
        self._dtype: torch.dtype = torch.float32  # aux (depth/embeds) compute dtype
        self._backbone_dtype: torch.dtype = torch.bfloat16  # vLLM Llama dtype
        self._lock = threading.Lock()

        # Single-call depth wrapper (N2); its HF module is wired in load_weights.
        self.depth = CsmDepthDecoder(
            num_codebooks=self.num_codebooks,
            hidden_size=self.hidden_size,
            aux_dtype=self._dtype,
        )

        # I5: per-request feedback cache. Keyed by _req_key(info); holds the
        # previous frame's Sigma-embed (1, hidden) to add onto the next step's
        # in-vocab token embedding inside preprocess(). Freed in
        # on_requests_finished.
        self._cached_sigma_by_req: dict[str, torch.Tensor] = {}
        # Per-request EOS latch so compute_logits can stop the scheduler on the
        # real frame-level EOS (cb0..cb30 all-zero) computed in forward().
        # Kept for debugging only; NOT used for row mapping (dict order is
        # first-seen insertion order, which is NOT the per-step sampler row
        # order once a request finishes or requests arrive out of order).
        self._eos_by_req: dict[str, bool] = {}
        # Authoritative EOS->logits-row mapping (unknown #2 fix). forward() fills
        # this in EXACT batch-row order (the same order the runner gathers
        # runtime_additional_information and the same order hidden_states are
        # gathered into compute_logits' sample rows -- input_batch.req_ids).
        # compute_logits consumes it positionally so EOS lands on the correct
        # request's row even under concurrency / staggered finishes.
        self._eos_flags_by_row: list[bool] = []
        # Per-request sampling params resolved at first preprocess.
        self._sampling_by_req: dict[str, tuple[float, int]] = {}
        # GATE-B hard frame cap. _max_frames_by_req[req] is the resolved cap
        # (from the request's SamplingParams.max_tokens, falling back to
        # _DEFAULT_MAX_FRAMES); _frames_emitted_by_req[req] counts non-prefill
        # frames surfaced from forward(). When the count reaches the cap,
        # compute_logits forces the frame EOS for that request's row so the AR
        # loop stops at the cap regardless of the natural all-zero EOS firing.
        self._max_frames_by_req: dict[str, int] = {}
        self._frames_emitted_by_req: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Weight loading (A3 §5; split: codec_model.* -> Stage 1)
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Split the CSM checkpoint by prefix; route Stage-0 parts, drain codec.

        REUSE: csm.py:278 prefix routing, minus the Mimi codec (Stage 1 loads it).
        Every returned name is a REAL registered parameter name (the
        ``backbone.`` / leading-underscore aux prefixes) to satisfy vLLM's
        load-completeness validator (B2).
        """
        with self._lock:
            if self._depth_decoder is not None:
                return set()
            try:
                self._device = next(self.parameters()).device
            except StopIteration:
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._dtype = self.config.torch_dtype or torch.float32
            try:
                self._backbone_dtype = next(self.backbone.parameters()).dtype
            except StopIteration:
                self._backbone_dtype = self.vllm_config.model_config.dtype or self._dtype
            self._build_aux_modules()

        backbone_layer_weights: list[tuple[str, torch.Tensor]] = []
        depth_weights: list[tuple[str, torch.Tensor]] = []
        loaded: set[str] = set()

        frame_embed_state: dict[str, torch.Tensor] = {}
        text_embed_state: dict[str, torch.Tensor] = {}
        cb0_head_state: dict[str, torch.Tensor] = {}

        for name, w in weights:
            if name.startswith("backbone_model.embed_tokens."):
                sub = name[len("backbone_model.embed_tokens.") :]
                frame_embed_state[sub] = w
                loaded.add(name)
            elif name.startswith("backbone_model."):
                backbone_layer_weights.append((name[len("backbone_model.") :], w))
            elif name == "lm_head.weight":
                cb0_head_state["weight"] = w
                loaded.add(name)
            elif name == "embed_text_tokens.weight":
                text_embed_state["weight"] = w
                loaded.add(name)
            elif name.startswith("depth_decoder."):
                depth_weights.append((name[len("depth_decoder.") :], w))
            elif name.startswith("codec_model."):
                # Stage 1 (csm_mimi) loads the Mimi codec from the same
                # checkpoint; drain it here so this stage's loader does not
                # report it as unrouted, and credit it as "loaded" for this
                # iterator (Stage 0 owns no codec parameters).
                loaded.add(name)
            else:
                logger.warning("CSM Stage-0 load_weights: unrouted weight %s", name)

        self.backbone.load_weights(iter(backbone_layer_weights))
        loaded |= {n for n, _ in self.backbone.named_parameters(prefix="backbone")}

        if "weight" in cb0_head_state:
            self._load_into(self.backbone.cb0_head, {"weight": cb0_head_state["weight"]})

        if self._frame_embed is not None and frame_embed_state:
            _, fe_unexpected = self._frame_embed.load_state_dict(frame_embed_state, strict=False)
            if fe_unexpected:
                logger.warning("CSM frame_embed unexpected keys: %s", fe_unexpected[:8])
            loaded |= {n for n, _ in self._frame_embed.named_parameters(prefix="_frame_embed")}
        if self._text_embed is not None and text_embed_state:
            self._text_embed.load_state_dict(text_embed_state, strict=False)
            loaded |= {n for n, _ in self._text_embed.named_parameters(prefix="_text_embed")}

        if self._depth_decoder is not None:
            _, dd_unexpected = self._depth_decoder.load_state_dict(dict(depth_weights), strict=False)
            if list(dd_unexpected):
                logger.warning("CSM depth_decoder unexpected keys: %s", list(dd_unexpected)[:8])
            loaded |= {n for n, _ in self._depth_decoder.named_parameters(prefix="_depth_decoder")}
            # The same HF depth module is also wired into the ``depth`` wrapper
            # (``self.depth.set_module(self._depth_decoder)``), which auto-
            # registers it under ``depth._module.*`` (nn.Module submodule
            # registration). Credit that aliased name span too, else vLLM's
            # load-completeness validator (default_loader.track_weights_loading)
            # flags ``depth._module.*`` as uninitialized at boot (B2).
            loaded |= {n for n, _ in self.depth.named_parameters(prefix="depth")}

        self._maybe_tie_depth_embed()
        return loaded

    def _build_aux_modules(self) -> None:
        """Construct the depth decoder + frame-embed + text embed (REUSE: csm.py:416).

        Mimi codec is NOT built here (Stage 1 owns it). Built from the
        authoritative HF ``CsmConfig`` loaded from the model path.
        """
        from transformers import AutoConfig
        from transformers.models.csm.modeling_csm import (
            CsmBackboneModelEmbeddings,
            CsmDepthDecoderForCausalLM,
        )

        hf_cfg = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self._hf_config = hf_cfg
        depth_cfg = hf_cfg.depth_decoder_config

        self._frame_embed = CsmBackboneModelEmbeddings(hf_cfg)
        self._text_embed = nn.Embedding(int(hf_cfg.text_vocab_size), int(hf_cfg.hidden_size))
        self._depth_decoder = CsmDepthDecoderForCausalLM(depth_cfg)

        for m in (self._frame_embed, self._text_embed, self._depth_decoder):
            m.to(device=self._device, dtype=self._dtype)
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

        # Wire the HF depth module into the single-call wrapper.
        self.depth._aux_dtype = self._dtype
        self.depth.set_module(self._depth_decoder)

    def _maybe_tie_depth_embed(self) -> None:
        """Tie depth audio embed to backbone audio embed (REUSE: csm.py:458)."""
        if self._frame_embed is None or self._depth_decoder is None:
            return
        if not getattr(self.config, "tie_codebooks_embeddings", True):
            return
        try:
            src = self._frame_embed.embed_audio_tokens.weight
            self._depth_decoder.model.embed_tokens.weight = src
        except AttributeError:
            logger.warning("CSM: could not tie depth embed to backbone audio embed")

    @staticmethod
    def _load_into(module: nn.Module, state: dict[str, torch.Tensor]) -> None:
        """REUSE: csm.py:476."""
        params = dict(module.named_parameters())
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        for name, w in state.items():
            if name not in params:
                logger.warning("CSM: %s not in %s", name, type(module).__name__)
                continue
            param = params[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, w)

    # ------------------------------------------------------------------
    # Dummy run support
    # ------------------------------------------------------------------

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict]:
        return [{"text": "hello", "_is_dummy": True} for _ in range(num_reqs)]

    def embed_input_ids(self, input_ids: torch.Tensor, *_: Any, **__: Any) -> torch.Tensor:
        """Stable dummy embedding for the runner's pre-embed path.

        The real per-step embedding is composed in ``preprocess`` (cb0 base embed
        + cached prev-frame Sigma). The runner pre-embeds the token here only to
        seed the CUDA-graph-stable inputs_embeds buffer; ``preprocess`` overwrites
        the request's span.
        """
        return torch.zeros(
            (input_ids.shape[0], self.hidden_size),
            device=input_ids.device,
            dtype=self._backbone_dtype,
        )

    # ------------------------------------------------------------------
    # Embedding primitives
    # ------------------------------------------------------------------

    def _embed_text_prompt(self, info: dict[str, Any], device: torch.device) -> torch.Tensor:
        """Embed the text prompt token ids -> (T, hidden). REUSE: csm.py:525."""
        token_ids = _pick(info, "prompt_token_ids", None)
        if token_ids is None:
            token_ids = info.get("prompt_token_ids")
        if token_ids is None:
            bos = int(getattr(self.config, "bos_token_id", 128000) or 128000)
            token_ids = [bos]
        ids = torch.as_tensor(token_ids, dtype=torch.long, device=device).view(-1)
        return self._text_embed(ids).to(self._backbone_dtype)

    def _compose_frame_embed(self, frame_codes: torch.Tensor) -> torch.Tensor:
        """Sigma over the 32 codebooks -> one backbone input embedding.

        REUSE: csm.py:544 ``_compose_frame_embed``. ``frame_codes`` is ``(B, 32)``
        Long; ``CsmBackboneModelEmbeddings`` expects ``(B, T, num_codebooks)`` and
        sums over the codebook dim. Returns ``(B, hidden)`` (we squeeze the
        length-1 frame axis) cast to the BACKBONE dtype (the b3b dtype fix: the
        fp32 frame-embed must match the bf16 QKV matmul or it raises
        "mat1 and mat2 must have the same dtype").
        """
        emb = self._frame_embed(frame_codes.unsqueeze(1))  # (B, 1, hidden)
        return emb.squeeze(1).to(self._backbone_dtype)  # (B, hidden)

    # ------------------------------------------------------------------
    # preprocess: re-inject the cached Sigma-embed (the feedback, A3 §3)
    # ------------------------------------------------------------------

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Deliver the composed inputs_embeds for this request's span.

        Mirrors Qwen3-TTS ``preprocess`` (``qwen3_tts_talker.py:571``) for the
        contract shape ``(input_ids_out, embeds_out, update_dict)``, and MiMo's
        cache-add semantics (``mimo_audio_llm.py:964`` ``+= prev_new_audio_emb``).

        PREFILL (span_len >= 1, first scheduled tokens): embed the text prompt;
        positions/slots are scheduler-owned. The in-vocab token ids are kept (they
        stay in vocab; the model reads only the embedding).

        DECODE (span_len == 1): the embedding is the base embedding of the in-vocab
        token PLUS the cached previous-frame Sigma-embed for this request. This is
        the ONLY place the feedback is re-injected; the next step is again a real
        scheduled token, so its KV slot is always allocated -> no IMA.
        """
        device = input_ids.device
        req_key = _req_key(info_dict)
        span_len = int(input_ids.shape[0])
        if span_len <= 0:
            base = input_embeds if input_embeds is not None else self.embed_input_ids(input_ids)
            return input_ids, base, {}

        # Resolve + cache per-request sampling once.
        if req_key not in self._sampling_by_req:
            temperature = float(_pick(info_dict, "temperature", _DEFAULT_TEMPERATURE))
            top_k = int(_pick(info_dict, "top_k", _DEFAULT_TOP_K))
            self._sampling_by_req[req_key] = (temperature, top_k)

        # Resolve + cache the hard frame cap once (GATE-B). Prefer an explicit
        # per-request cap forwarded via additional_information ("max_new_frames"
        # or its "max_tokens" alias, set by the serving param-builder), else the
        # deploy-level default. This is the fail-closed backstop to the
        # scheduler's own SamplingParams.max_tokens stop.
        if req_key not in self._max_frames_by_req:
            cap = _pick(info_dict, "max_new_frames", None)
            if cap is None:
                cap = _pick(info_dict, "max_tokens", None)
            try:
                cap_int = int(cap) if cap is not None else _DEFAULT_MAX_FRAMES
            except (TypeError, ValueError):
                cap_int = _DEFAULT_MAX_FRAMES
            if cap_int <= 0:
                cap_int = _DEFAULT_MAX_FRAMES
            self._max_frames_by_req[req_key] = cap_int
            self._frames_emitted_by_req[req_key] = 0

        is_prefill_raw = info_dict.get("_omni_is_prefill")
        if isinstance(is_prefill_raw, bool):
            is_prefill = is_prefill_raw
        else:
            try:
                is_prefill = int(info_dict["_omni_num_computed_tokens"]) < int(info_dict["_omni_prompt_len"])
            except Exception:
                is_prefill = span_len > 1

        # Keep token ids in-vocab for vLLM bookkeeping (Qwen3 invariant
        # qwen3_tts_talker.py:687-689). cb0 ids are already in [0, vocab); clamp
        # defensively so a stray id can never index out of the embedding table.
        input_ids_out = input_ids.clone()
        input_ids_out.clamp_(min=0, max=int(self.config.vocab_size) - 1)

        if is_prefill:
            prompt_embeds = self._embed_text_prompt(info_dict, device)  # (T, hidden)
            offset = int(info_dict.get("_omni_num_computed_tokens", 0) or 0)
            offset = max(0, offset)
            s = max(0, min(offset, int(prompt_embeds.shape[0])))
            e = max(0, min(offset + span_len, int(prompt_embeds.shape[0])))
            take = prompt_embeds[s:e]
            if int(take.shape[0]) < span_len:
                pad_n = span_len - int(take.shape[0])
                pad = torch.zeros((pad_n, self.hidden_size), device=device, dtype=self._backbone_dtype)
                take = torch.cat([take, pad], dim=0)
            return input_ids_out, take, {}

        # Decode: span_len == 1. base = embed(cb0_token); add cached Sigma.
        # The in-vocab token's base embedding is taken from the same per-codebook
        # cb0 row of the frame-embed table (codebook 0), so the composed embedding
        # equals embed_cb0(token) + Sigma(prev frame). When no cache yet (first
        # decode after prefill), fall back to the runner-provided base embed.
        cb0_token = input_ids_out.reshape(-1)[:1].to(torch.long)  # (1,)
        # Build a (1, 32) frame with only cb0 set so the Sigma table contributes
        # exactly the cb0 codebook-0 embedding for this token.
        cb0_frame = torch.zeros((1, self.num_codebooks), dtype=torch.long, device=device)
        cb0_frame[0, 0] = cb0_token[0]
        base = self._compose_frame_embed(cb0_frame)  # (1, hidden)

        cached = self._cached_sigma_by_req.get(req_key)
        if cached is not None:
            base = base + cached.to(device=device, dtype=self._backbone_dtype)
        return input_ids_out, base, {}

    def preprocess_decode_batch(
        self,
        *,
        input_ids: torch.Tensor,
        req_infos: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Batched decode-only preprocess (B>1 continuous batching, A3 N8).

        Loops ``preprocess`` over the batch and stacks. CSM does NOT use the
        runner's talker_mtp fast-path, so we return only ``(ids, embeds, [])``
        (no mtp_inputs); the runner's plain ``preprocess`` flush path consumes
        this. Kept here so ``has_talker_mtp`` stays False while still offering a
        batched entry point symmetric with ``preprocess``.
        """
        ids_flat = input_ids.reshape(-1)
        if int(ids_flat.numel()) != len(req_infos):
            raise ValueError(f"preprocess_decode_batch expected {len(req_infos)} ids, got {int(ids_flat.numel())}")
        out_ids: list[torch.Tensor] = []
        out_embeds: list[torch.Tensor] = []
        for i, info in enumerate(req_infos):
            rid, remb, _ = self.preprocess(input_ids=ids_flat[i : i + 1], input_embeds=None, **info)
            out_ids.append(rid.reshape(-1)[:1])
            out_embeds.append(remb.reshape(1, -1))
        return torch.cat(out_ids, dim=0), torch.cat(out_embeds, dim=0), {}

    def postprocess(self, hidden_states: torch.Tensor, **_: Any) -> dict[str, Any]:
        """No-op postprocess (h_t is consumed inline in forward).

        Declared (``has_postprocess=True``) only to keep the runner's
        per-request additional_information refresh path active
        (``gpu_model_runner.py:1208``). CSM forms its own next-step embedding
        inside ``forward()`` + ``preprocess``, so there is nothing to stash here.
        """
        return {}

    # ------------------------------------------------------------------
    # Core forward: one backbone step + inline depth + Sigma cache
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        model_intermediate_buffer: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """Run the backbone once, sample cb0, run inline depth, cache Sigma.

        Streaming contract (I1): DELTA -- ``codes.audio`` carries ONLY the new
        32-code frame(s) produced this step (forward-flowing latent to Stage 1);
        nothing is re-decoded. Hot-loop discipline (I3): no ``.item()`` / ``.cpu()``
        inside the depth loop (handled in ``CsmDepthDecoder.run``); the single EOS
        host-read is per-frame, outside the 31-step loop.
        """
        infos = runtime_additional_information or model_intermediate_buffer or [{}]

        # vLLM owns positions/slots; we only read h_t and overwrite embedding
        # VALUES. The runner already wrote the composed embedding (from
        # preprocess) into inputs_embeds[s:e].
        hidden = self.backbone.forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        # hidden: (sum_tokens, hidden). Resolve per-request last-row spans from
        # the forward context query_start_loc (mirrors mimo_audio_llm.py:1038).
        from vllm.forward_context import get_forward_context

        ctx = get_forward_context()
        if ctx is not None and ctx.attn_metadata is not None:
            qsl = next(iter(ctx.attn_metadata.values())).query_start_loc
        else:
            qsl = torch.tensor([0, hidden.shape[0]], device=hidden.device)

        num_reqs = max(0, int(qsl.numel()) - 1)
        device = hidden.device
        codes_out: list[torch.Tensor | None] = [None] * num_reqs
        # Per-row EOS flags for THIS step, in batch-row order (unknown #2). Index
        # i corresponds to sampler logits row i in compute_logits (both derive
        # from input_batch.req_ids order). Default False (incl. dummy rows).
        eos_flags_by_row: list[bool] = [False] * num_reqs

        for i in range(num_reqs):
            info = infos[i] if i < len(infos) else {}
            if info.get("_is_dummy"):
                continue
            req_key = _req_key(info)
            temperature, top_k = self._sampling_by_req.get(req_key, (_DEFAULT_TEMPERATURE, _DEFAULT_TOP_K))

            start = int(qsl[i].item())
            end = int(qsl[i + 1].item())
            if end <= start:
                continue
            last_hidden = hidden[end - 1 : end].to(self._backbone_dtype)  # (1, hidden)

            # cb0 via the LogitsProcessor (NOT a direct ParallelLMHead.forward --
            # the b3c bug). Applies the head matmul + TP gather + vocab trim.
            cb0_logits = self.backbone.logits_processor(self.backbone.cb0_head, last_hidden)
            cb0 = sample_logits(cb0_logits, temperature, top_k)  # (1,)

            # 31-step inline depth -> (1, 32) Long.
            frame_codes = self.depth.run(
                cb0=cb0,
                backbone_last_hidden_state=last_hidden,
                temperature=temperature,
                top_k=top_k,
            )

            # Real frame EOS (A3 §5): cb0..cb30 all-zero. One host-read per frame
            # (outside the depth loop, I3-compliant). Latch so compute_logits stops
            # the scheduler this step.
            natural_eos = bool((frame_codes[0, : self.num_codebooks - 1] == _CODEBOOK_EOS_ID).all().item())

            # GATE-B hard frame cap: count the frame we are about to surface and
            # force a stop once the per-request cap is reached, even if the
            # natural all-zero EOS never fires (the non-terminating greedy/
            # repetition attractor). This is the fail-closed backstop to the
            # scheduler's SamplingParams.max_tokens stop; it guarantees the AR
            # loop ends at the cap and protects the server from unbounded
            # requests. Distinct from natural_eos because a cap-forced frame is a
            # REAL audio frame (keep it), whereas a natural EOS frame is all-zero
            # (drop it).
            emitted = self._frames_emitted_by_req.get(req_key, 0) + 1
            self._frames_emitted_by_req[req_key] = emitted
            cap = self._max_frames_by_req.get(req_key, _DEFAULT_MAX_FRAMES)
            cap_forced = emitted >= cap and not natural_eos
            if cap_forced:
                logger.warning(
                    "CSM req %s hit frame cap %d without a natural EOS frame; "
                    "forcing stop (greedy non-terminating attractor guard).",
                    req_key,
                    cap,
                )

            # is_eos drives the scheduler stop for BOTH natural EOS and the
            # cap-forced stop; the emit branch below keeps the cap-forced frame's
            # audio but drops the all-zero natural-EOS frame.
            is_eos = natural_eos or cap_forced
            self._eos_by_req[req_key] = is_eos
            eos_flags_by_row[i] = is_eos  # authoritative row mapping (unknown #2)

            # Cache the next-frame Sigma-embed for this request (I5). On the
            # terminal EOS frame we still cache (harmless; freed at finish).
            sigma = self._compose_frame_embed(frame_codes)  # (1, hidden)
            self._cached_sigma_by_req[req_key] = sigma.detach()

            # Surface the finished 32-code frame forward to Stage 1 as latent.
            # Layout [F=1, 32] (frame-major; the stage processor flattens
            # codebook-major). On the NATURAL all-zero EOS emit an empty frame so
            # Stage 1 trims it; on a CAP-FORCED stop the frame is real audio --
            # keep it so the capped request still produces its final frame of
            # speech (the cap bounds length, it does not truncate the last frame).
            if natural_eos:
                codes_out[i] = torch.zeros((0, self.num_codebooks), dtype=torch.long, device=device)
            else:
                codes_out[i] = frame_codes.to(torch.long)  # (1, 32)

            # Default-off validation hook: dump per-request served frames to a
            # JSONL so an offline driver can assert frame-for-frame vs HF greedy
            # (GATE-B). Zero effect unless ``VLLM_CSM_DUMP_FRAMES`` is set.
            _dump = __import__("os").environ.get("VLLM_CSM_DUMP_FRAMES")
            if _dump:
                try:
                    import json as _json

                    _rec = {
                        "req": req_key,
                        "eos": bool(is_eos),
                        "frame": frame_codes[0].detach().cpu().tolist(),
                    }
                    # Drift probe (unknown #1): when VLLM_CSM_DUMP_HIDDEN is set,
                    # also dump the backbone last-hidden-state row used to drive
                    # cb0 + the inline depth loop, so an offline driver can diff
                    # it against an HF-eager run on the same prefix (kernel-drift
                    # signature: small ~1e-3 delta growing with t). Zero cost by
                    # default; fp32 list for exact compare.
                    if __import__("os").environ.get("VLLM_CSM_DUMP_HIDDEN"):
                        _rec["hidden"] = last_hidden[0].detach().float().cpu().tolist()
                    with open(_dump, "a") as _fh:
                        _fh.write(_json.dumps(_rec) + "\n")
                except Exception:
                    pass

        # Publish the per-row EOS mapping for compute_logits (called immediately
        # after forward in the same step, same batch-row order). Unknown #2 fix.
        self._eos_flags_by_row = eos_flags_by_row

        # text_hidden_states feeds compute_logits' sample-row gather; pass through.
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={"codes": {"audio": codes_out}},
        )

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        """Passthrough when forward already returned an OmniOutput.

        ``forward`` returns ``OmniOutput`` directly, so ``extract_multimodal_outputs``
        (``gpu_model_runner.py:686``) consumes it and the runner does NOT call this
        (``_model_forward`` only calls ``make_omni_output`` when the output is not an
        ``OmniOutput``). Provided for symmetry with qwen3_tts
        (``qwen3_tts_talker.py:505-507``) / robustness.
        """
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs={})

    # ------------------------------------------------------------------
    # AR scheduler interface: cb0 logits + real frame EOS stop
    # ------------------------------------------------------------------

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        """Emit cb0 logits so the external sampler advances one real in-vocab
        token per frame, with a real frame-EOS override.

        The cb0 token id the sampler picks becomes ``input_ids`` for the next
        step's ``preprocess`` (the scheduler-visible in-vocab token; A3 §3 inv.1).
        On the real frame EOS (cb0..cb30 all-zero, latched in ``forward``) we force
        the configured stop id so the scheduler finishes the request. NOT the fake
        ``(num_rows, 2051)`` EOS grid the single-stage model used (csm.py:847,
        discarded -- it was a second, conflicting stop authority, A3 §1).
        """
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        if hidden_states.ndim == 1:
            hidden_states = hidden_states.unsqueeze(0)
        logits = self.backbone.logits_processor(self.backbone.cb0_head, hidden_states.to(self._backbone_dtype))
        if logits is None:
            return None

        # Force EOS (token id 0) on rows whose frame was the real all-zero EOS.
        # The AR runner gathers one sample row per request in batch-row order
        # (hidden_states[logits_indices], same order as input_batch.req_ids);
        # forward() filled _eos_flags_by_row in that SAME order this step, so we
        # map positionally (unknown #2 fix). Using dict.values() here was wrong:
        # dict order is first-seen insertion order, which desyncs from the
        # current sampler-row order once any request finishes or requests arrive
        # out of order -> EOS could fire on the wrong request's row.
        eos_flags = self._eos_flags_by_row
        num_rows = int(logits.shape[0])
        for row in range(num_rows):
            if row < len(eos_flags) and eos_flags[row]:
                logits[row, :] = float("-inf")
                logits[row, _CODEBOOK_EOS_ID] = 1.0e6
        return logits

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        """Free per-request state on finish (I5). REUSE shape: csm.py:829."""
        for req_id in finished_req_ids:
            key = str(req_id)
            self._cached_sigma_by_req.pop(key, None)
            self._eos_by_req.pop(key, None)
            self._sampling_by_req.pop(key, None)
            self._max_frames_by_req.pop(key, None)
            self._frames_emitted_by_req.pop(key, None)
