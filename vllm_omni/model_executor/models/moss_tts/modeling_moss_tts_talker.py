# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""MOSS-TTS Stage-0 talker: Qwen3 backbone + (n_vq+1) parallel AR heads."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.sequence import IntermediateTensors
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.model_executor.models.moss_tts.configuration_moss_tts import (
    MossTTSDelayConfig,
    MossTTSLocalConfig,
    MossTTSRealtimeConfig,
)
from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_local import (
    MossTTSRealtimeLocalTransformer,
)
from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_local_depth import (
    MossTTSLocalDepthTransformer,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


def _maybe_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _iter_state_row_spans(
    states: Sequence[dict[str, Any]],
    spans: Sequence[tuple[int, int]] | None,
    num_rows: int,
) -> Iterable[tuple[dict[str, Any], int, int]]:
    """Yield state/logit-row ranges without assuming equal request row counts."""
    if num_rows <= 0 or not states:
        return

    if spans is not None:
        if len(spans) != len(states):
            raise RuntimeError(f"request_token_spans has {len(spans)} entries for {len(states)} request states")
        if all(row_end <= num_rows for _, row_end in spans):
            for state, (row_start, row_end) in zip(states, spans, strict=False):
                if row_start < row_end:
                    yield state, int(row_start), int(row_end)
            return
        if num_rows != len(states):
            raise RuntimeError(
                "request_token_spans describe full hidden rows, but current logits "
                f"have {num_rows} rows for {len(states)} states and are not one-row-per-request"
            )

    if num_rows == len(states):
        for i, state in enumerate(states):
            yield state, i, i + 1
        return

    raise RuntimeError(
        "MOSS-TTS compute_logits requires request_token_spans or one-row-per-request logits; "
        f"got {num_rows} rows for {len(states)} request states"
    )


# ---------------------------------------------------------------------------
# MossTTSDelayTalkerForGeneration
# ---------------------------------------------------------------------------


class MossTTSDelayTalkerForGeneration(nn.Module):
    """Stage-0 talker for MossTTSDelayModel variants.

    Covers all four repos that ship ``architectures: ["MossTTSDelayModel"]``:
      - MOSS-TTS (8B, n_vq=32)
      - MOSS-TTSD-v1.0 (8B, n_vq=16)
      - MOSS-SoundEffect (8B, n_vq=16)
      - MOSS-VoiceGenerator (1.7B, n_vq=16)

    Architecture
    ~~~~~~~~~~~~
    * Backbone: Qwen3 transformer (hidden_size, num_hidden_layers, etc. from
      ``config.language_config``).
    * Embedding: text_embed(t) + Σᵢ audio_embed_i(aᵢ)  — additive fusion,
      no cross-attention or concatenation.
    * Heads: (n_vq + 1) parallel linear heads over the final hidden state.
      – Head 0  → text logits  (drives AR scheduler)
      – Heads 1…n_vq → audio VQ logits  (one per RVQ codebook)

    Delay pattern
    ~~~~~~~~~~~~~
    Audio heads are only active after the model emits the delay-slot token
    (``audio_assistant_delay_slot_token_id``).  Before the slot fires all
    audio heads output a pad token (``audio_pad_code``).  After the slot:

        step  t:     collect audio_codebook_0  for frame t
        step t+1:    collect audio_codebook_0  for frame t+1
                     collect audio_codebook_1  for frame t     (1-step lag)
        …
        step t+k:    all k codebooks active; emit frame t

    The per-request ``delay_step`` counter (stored in the per-request info
    dict) tracks this. Stage-1 receives the codes in (T, NQ) shape and the
    codec's ``batch_decode`` handles the de-interleaving internally.
    """

    # vLLM-Omni integration flags
    have_multimodal_outputs: bool = True
    has_preprocess: bool = True
    has_postprocess: bool = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config: MossTTSDelayConfig = vllm_config.model_config.hf_config

        lm_cfg = self.config.language_config
        hidden_size: int = lm_cfg.hidden_size
        self.hidden_size = hidden_size
        self.n_vq: int = self.config.n_vq
        self.audio_vocab_size: int = self.config.audio_vocab_size
        self.audio_pad_code: int = self.config.audio_pad_code

        # Token IDs (mirrors upstream MossTTSDelayConfig defaults).
        self.audio_start_token_id: int = self.config.audio_start_token_id
        self.audio_end_token_id: int = self.config.audio_end_token_id
        self.audio_assistant_gen_slot_token_id: int = self.config.audio_assistant_gen_slot_token_id
        self.audio_assistant_delay_slot_token_id: int = self.config.audio_assistant_delay_slot_token_id
        self.pad_token_id: int = getattr(self.config, "pad_token_id", 151643)
        self.im_end_token_id: int = getattr(self.config, "im_end_token_id", 151645)

        # Qwen3 backbone — weights live under ``language_model.*``. vLLM's
        # Qwen3Model reads num_hidden_layers/hidden_size/etc. from
        # vllm_config.model_config.hf_config, but MOSS nests the LLM config under
        # ``language_config`` (a Qwen3Config). Rebind the backbone's hf_config to
        # it — same pattern as the Qwen3-Omni talker — so the inner model sees a
        # real Qwen3Config instead of MossTTSDelayConfig (which lacks
        # num_hidden_layers).
        lm_vllm_config = vllm_config.with_hf_config(lm_cfg, architectures=["Qwen3ForCausalLM"])
        lm_vllm_config.model_config.hf_text_config = lm_vllm_config.model_config.hf_config
        self.model = Qwen3Model(
            vllm_config=lm_vllm_config,
            prefix=_maybe_prefix(prefix, "model"),
        )

        # Text LM head (head 0)
        self.text_lm_head = ParallelLMHead(
            lm_cfg.vocab_size,
            hidden_size,
            bias=False,
            prefix=_maybe_prefix(prefix, "text_lm_head"),
        )
        self.logits_processor = LogitsProcessor(lm_cfg.vocab_size)

        # Audio VQ heads (heads 1…n_vq in upstream naming)
        # Each head predicts one RVQ codebook: vocab = audio_vocab_size + 1
        self.audio_heads = nn.ModuleList(
            [nn.Linear(hidden_size, self.audio_vocab_size + 1, bias=False) for _ in range(self.n_vq)]
        )

        # Per-codebook audio embeddings (emb_ext in upstream)
        self.audio_embeddings = nn.ModuleList(
            [nn.Embedding(self.audio_vocab_size + 1, hidden_size) for _ in range(self.n_vq)]
        )

        # GPU-resident per-request buffer keys (avoid CPU round-trips)
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {
            ("audio_codes", "current"),  # last step's audio codes
            ("audio_codes", "accumulated"),
            ("hidden_states", "last"),
        }

        # Static text-logit masks (lazy, built on first use to know device).
        self._pre_exclude_text_ids = (
            self.pad_token_id,
            self.audio_assistant_gen_slot_token_id,
            self.audio_assistant_delay_slot_token_id,
            self.audio_end_token_id,
        )
        self._audio_keep_text_ids = (
            self.audio_assistant_gen_slot_token_id,
            self.audio_assistant_delay_slot_token_id,
        )

        # Per-step state stash for `compute_logits` (populated by
        # `make_omni_output`, which runs immediately before the sampler).
        self._batch_state: list[dict[str, Any]] | None = None

        # Stacked weight caches built after load_weights(). Avoids a Python
        # loop over n_vq heads/embeddings at every decode step.
        # shapes: (n_vq, audio_vocab+1, hidden_size)
        self._stacked_audio_head_w: torch.Tensor | None = None
        self._stacked_audio_emb_w: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # vLLM-Omni hooks
    # ------------------------------------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
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

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor | None:
        """Return text-head logits with delay-pattern constraints applied.

        The mask follows upstream MOSS-TTS' generate loop:
          * Forced tokens override the sampler when delayed_lengths is in the
            audio-emit window (delay_slot for [0, n_vq), audio_end at n_vq).
          * Outside that window, mask audio control tokens unless the model
            is currently emitting audio (is_audio).
          * Mask delay_slot at step 0 and im_end during the first n_vq steps,
            matching upstream's anti-collapse heuristics.
        """
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        logits = self.logits_processor(self.text_lm_head, hidden_states)
        if logits is None or self._batch_state is None:
            return logits

        states = self._batch_state
        if not states:
            return logits

        n_vq = self.n_vq
        device = logits.device
        vocab_size = logits.shape[-1]

        for state, row_start, row_end in _iter_state_row_spans(
            states,
            getattr(self, "_batch_state_spans", None),
            logits.shape[0],
        ):
            if state is None:
                continue
            row = logits[row_start:row_end]

            delayed_lengths = int(state.get("delayed_lengths", -1))
            is_audio = bool(state.get("is_audio", False))
            step = int(state.get("step", 0))
            audio_lengths_cur = int(state.get("audio_lengths", 0))
            max_new_frames = int(state.get("max_new_frames", -1))

            # ---- max_new_frames cap (force im_end / EOS) ----
            # When the request specified ``max_new_frames`` and we've emitted
            # that many audio-bearing tokens, force ``im_end`` so the AR loop
            # stops. Without this, short prompts run to deploy-default
            # ``max_tokens`` and produce 100+ s of audio per ~10-word input.
            if (
                max_new_frames > 0
                and is_audio
                and audio_lengths_cur >= max_new_frames
                and 0 <= self.im_end_token_id < vocab_size
            ):
                neg_inf = torch.full_like(row, float("-inf"))
                neg_inf[..., self.im_end_token_id] = 0.0
                logits[row_start:row_end] = neg_inf
                continue

            # ---- Forced tokens (delay-slot run / audio-end) ----
            forced: int | None = None
            if 0 <= delayed_lengths < n_vq:
                forced = self.audio_assistant_delay_slot_token_id
            elif delayed_lengths == n_vq:
                forced = self.audio_end_token_id
            if forced is not None and 0 <= forced < vocab_size:
                neg_inf = torch.full_like(row, float("-inf"))
                neg_inf[..., forced] = 0.0
                row = neg_inf
                logits[row_start:row_end] = row
                continue

            # ---- Pre-exclusion masks (delayed_lengths == -1 sentinel) ----
            if is_audio:
                # Only delay_slot or gen_slot are valid in audio mode.
                mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
                for tok in self._audio_keep_text_ids:
                    if 0 <= tok < vocab_size:
                        mask[tok] = False
                row = row.masked_fill(mask, float("-inf"))
            else:
                # Mask audio control tokens during text generation.
                for tok in self._pre_exclude_text_ids:
                    if 0 <= tok < vocab_size:
                        row[..., tok] = float("-inf")

            # ---- Step-conditional masks ----
            if step == 0:
                tok = self.audio_assistant_delay_slot_token_id
                if 0 <= tok < vocab_size:
                    row[..., tok] = float("-inf")
            if step <= n_vq:
                tok = self.im_end_token_id
                if 0 <= tok < vocab_size:
                    row[..., tok] = float("-inf")

            logits[row_start:row_end] = row

        return logits

    def _initial_state(self, prompt_ids: torch.Tensor) -> dict[str, Any]:
        """Initialise the AR generation state machine from a prompt.

        Mirrors upstream's ``generate`` head:
          * is_audio = True iff the last prompt token is audio_start or gen_slot
            and the prompt contains a prior audio_start.
          * audio_lengths counts tokens since the last audio_start in that case.
          * delayed_lengths starts at the int64-max sentinel (-1 here).
        """
        if prompt_ids.numel() == 0:
            return {
                "audio_lengths": 0,
                "delayed_lengths": -1,
                "is_audio": False,
                "step": 0,
            }
        prompt_cpu = prompt_ids.detach().to("cpu", dtype=torch.long).reshape(-1)
        seq_len = int(prompt_cpu.shape[0])
        last_id = int(prompt_cpu[-1])
        is_continuation = last_id in (
            self.audio_start_token_id,
            self.audio_assistant_gen_slot_token_id,
        )
        last_audio_start = -1
        if is_continuation:
            matches = (prompt_cpu == self.audio_start_token_id).nonzero(as_tuple=True)[0]
            if matches.numel() > 0:
                last_audio_start = int(matches[-1])
        is_audio = is_continuation and last_audio_start != -1
        audio_lengths = (seq_len - last_audio_start) if is_audio else 0
        return {
            "audio_lengths": int(audio_lengths),
            "delayed_lengths": -1,
            "is_audio": bool(is_audio),
            "step": 0,
        }

    def _advance_state(self, state: dict[str, Any], sampled_id: int) -> dict[str, Any]:
        """Update state given the text token sampled at the previous step."""
        audio_lengths = int(state.get("audio_lengths", 0))
        delayed_lengths = int(state.get("delayed_lengths", -1))
        is_audio = bool(state.get("is_audio", False))

        if sampled_id in (
            self.audio_start_token_id,
            self.audio_assistant_gen_slot_token_id,
            self.audio_assistant_delay_slot_token_id,
        ):
            audio_lengths += 1
        if sampled_id == self.audio_end_token_id:
            audio_lengths = 0

        if delayed_lengths == -1:
            if sampled_id == self.audio_assistant_delay_slot_token_id:
                delayed_lengths = 0
        else:
            delayed_lengths += 1
            if delayed_lengths > self.n_vq:
                delayed_lengths = -1

        if sampled_id == self.audio_start_token_id:
            is_audio = True
        if sampled_id == self.audio_end_token_id:
            is_audio = False

        state = dict(state)
        state["audio_lengths"] = audio_lengths
        state["delayed_lengths"] = delayed_lengths
        state["is_audio"] = is_audio
        state["step"] = int(state.get("step", 0)) + 1
        return state

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build per-step input embeddings (text + audio additive fusion).

        Prefill: initialise the per-request state machine from the prompt.
        Decode: update the state with the just-sampled text token, then build
        the combined text+audio embedding using the previous step's codes.
        """
        device = input_ids.device
        span_len = int(input_ids.shape[0])
        audio_state = info_dict.get("audio_state")
        is_first_call = not isinstance(audio_state, dict)

        if span_len > 1 or is_first_call:
            # Prefill (or first call). Initialise the per-request state and
            # build text embeddings. If the request carries reference-audio
            # codes (``codes.ref`` from the upstream MossTTSDelayProcessor's
            # delay-pattern grid, shape ``(L_full, n_vq)``), additively embed
            # them at the matching prefill positions so the talker can attend
            # to the speaker's timbre when generating its response.
            audio_state = self._initial_state(input_ids)
            embeds = self.model.embed_tokens(input_ids)

            ref_codes = (info_dict.get("codes", {}) or {}).get("ref")
            last_ref_row = None
            ref_offset = int(info_dict.get("ref_offset", 0))
            if isinstance(ref_codes, torch.Tensor) and ref_codes.numel() > 0:
                if ref_codes.dim() == 1:
                    if ref_codes.numel() % self.n_vq == 0:
                        ref_codes = ref_codes.view(-1, self.n_vq)
                    else:
                        ref_codes = None  # malformed, skip silently
                if isinstance(ref_codes, torch.Tensor) and ref_codes.dim() == 2:
                    # Slice the window matching this prefill chunk (chunked
                    # prefill calls preprocess multiple times for one request).
                    end_off = ref_offset + span_len
                    chunk = ref_codes[ref_offset:end_off]
                    if chunk.numel() > 0 and chunk.shape[0] == span_len:
                        codes = chunk.to(device=device, dtype=torch.long).clamp_(0, self.audio_vocab_size)
                        if self._stacked_audio_emb_w is not None:
                            codes_nq = codes.t()  # (n_vq, L)
                            audio_embed = self._stacked_audio_emb_w[
                                torch.arange(self.n_vq, device=device)[:, None], codes_nq
                            ].sum(0)
                        else:
                            audio_embed = torch.zeros_like(embeds)
                            for i, emb_layer in enumerate(self.audio_embeddings):
                                audio_embed = audio_embed + emb_layer(codes[:, i])
                        embeds = embeds + audio_embed
                        last_ref_row = codes[-1].detach()

            current_codes = (
                last_ref_row
                if last_ref_row is not None
                else torch.full(
                    (self.n_vq,),
                    self.audio_pad_code,
                    dtype=torch.long,
                    device=device,
                )
            )
            # Lift max_new_frames from request additional_information into
            # audio_state so compute_logits can force im_end once we've
            # emitted that many audio frames (mirrors the moss_tts_nano cap).
            max_new_frames_req = info_dict.get("max_new_frames")
            if isinstance(max_new_frames_req, (list, tuple)) and max_new_frames_req:
                max_new_frames_req = max_new_frames_req[0]
            try:
                audio_state["max_new_frames"] = int(max_new_frames_req) if max_new_frames_req is not None else -1
            except (TypeError, ValueError):
                audio_state["max_new_frames"] = -1
            info_update: dict[str, Any] = {
                "audio_state": audio_state,
                "audio_codes": {"current": current_codes},
                "ref_offset": ref_offset + span_len,
            }
            return input_ids, embeds, info_update

        # Decode step: input_ids is the text token sampled at step n-1.
        sampled_id = int(input_ids.reshape(-1)[0].item())
        audio_state = self._advance_state(audio_state, sampled_id)

        # Combined embedding: text(t) + Σᵢ audio_emb_i(code_i_{t-1}).
        text_embed = self.model.embed_tokens(input_ids.reshape(1))  # (1, H)
        audio_codes_buf = (info_dict.get("audio_codes", {}) or {}).get("current")
        if isinstance(audio_codes_buf, torch.Tensor) and audio_codes_buf.numel() == self.n_vq:
            codes = audio_codes_buf.to(device=device, dtype=torch.long)
            if self._stacked_audio_emb_w is not None:
                # Single gather: stacked_emb[i, codes[i]] for each i → (n_vq, H) → sum → (1, H)
                n_vq_idx = torch.arange(self.n_vq, device=device)
                audio_embed = self._stacked_audio_emb_w[n_vq_idx, codes.clamp(0, self.audio_vocab_size)].sum(
                    0, keepdim=True
                )
            else:
                audio_embed = torch.zeros_like(text_embed)
                for i, emb_layer in enumerate(self.audio_embeddings):
                    code_i = codes[i].clamp(0, self.audio_vocab_size).reshape(1)
                    audio_embed = audio_embed + emb_layer(code_i)
            combined = text_embed + audio_embed
        else:
            combined = text_embed

        return input_ids, combined, {"audio_state": audio_state}

    def postprocess(self, hidden_states: torch.Tensor, **_: Any) -> dict[str, Any]:
        if hidden_states.numel() == 0:
            return {}
        return {"hidden_states": {"last": hidden_states[-1].detach()}}

    @staticmethod
    def _sample_with_top_k(
        logits: torch.Tensor,
        top_k: int,
        temperature: float,
    ) -> torch.Tensor:
        """Top-k sampling on a (..., V) logits tensor returning (...,) ids."""
        if temperature > 0:
            logits = logits / max(temperature, 1e-6)
        if top_k and top_k > 0 and top_k < logits.shape[-1]:
            top_vals, _ = torch.topk(logits, top_k, dim=-1)
            kth = top_vals[..., -1:].expand_as(logits)
            logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
        if temperature <= 0:
            return logits.argmax(dim=-1)
        probs = torch.softmax(logits, dim=-1)
        flat = probs.reshape(-1, probs.shape[-1])
        sampled = torch.multinomial(flat, num_samples=1).reshape(probs.shape[:-1])
        return sampled

    def _sample_audio_codes(
        self,
        last_h: torch.Tensor,  # (1, H)
        state: dict[str, Any],
    ) -> torch.Tensor:  # (n_vq,)
        """Sample one row of n_vq audio codes given current state.

        Mirrors upstream's pre/post audio masks:
          pre_audio_mask  = audio_lengths > arange(n_vq)
          post_audio_mask = arange(n_vq) > delayed_lengths - 1
                           (or all-True when delayed_lengths is sentinel)
          sampling_audio_mask = pre & post  → heads to sample; rest = pad_code
        """
        device = last_h.device
        n_vq = self.n_vq
        audio_lengths = int(state.get("audio_lengths", 0))
        delayed_lengths = int(state.get("delayed_lengths", -1))

        idx = torch.arange(n_vq, device=device)
        pre_mask = idx < audio_lengths
        if delayed_lengths == -1:
            post_mask = torch.ones(n_vq, dtype=torch.bool, device=device)
        else:
            post_mask = idx > (delayed_lengths - 1)
        sampling_mask = pre_mask & post_mask

        codes = torch.full((n_vq,), self.audio_pad_code, dtype=torch.long, device=device)
        if not bool(sampling_mask.any()):
            return codes

        audio_top_k = 25
        audio_temp = 1.7

        if self._stacked_audio_head_w is not None:
            # Single batched matmul: (n_vq, V+1, H) @ (H,) → (n_vq, V+1).
            # Replaces n_vq serial nn.Linear calls.
            all_logits = self._stacked_audio_head_w @ last_h.reshape(-1)  # (n_vq, V+1)
            all_logits[:, -1] = float("-inf")  # invalid sentinel
            all_logits[:, self.audio_pad_code] = float("-inf")

            active_idx = sampling_mask.nonzero(as_tuple=True)[0]  # (n_active,)
            active_logits = all_logits[active_idx]  # (n_active, V+1)
            active_logits = active_logits / max(audio_temp, 1e-6)
            if audio_top_k > 0 and audio_top_k < active_logits.shape[-1]:
                top_vals, _ = torch.topk(active_logits, audio_top_k, dim=-1)
                kth = top_vals[:, -1:].expand_as(active_logits)
                active_logits = torch.where(
                    active_logits < kth,
                    torch.full_like(active_logits, float("-inf")),
                    active_logits,
                )
            probs = torch.softmax(active_logits, dim=-1)
            codes[active_idx] = torch.multinomial(probs, num_samples=1).squeeze(-1).long()
            return codes

        # Fallback: per-head loop (used before load_weights() completes).
        for i in range(n_vq):
            if not bool(sampling_mask[i]):
                continue
            head = self.audio_heads[i]
            logits = head(last_h).reshape(-1)  # (V,)
            logits[..., -1] = float("-inf")
            logits[..., self.audio_pad_code] = float("-inf")
            sampled = self._sample_with_top_k(logits, audio_top_k, audio_temp)
            codes[i] = sampled.long()
        return codes

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput:
        """Sample audio codes per request and stash text-mask state.

        Per-request state lives in ``info["audio_state"]``.  Audio codes are
        accumulated in ``info["audio_codes"]["accumulated"]`` (T_acc, NQ) and
        the most recent row is stored in ``info["audio_codes"]["current"]``
        for the next preprocess step.
        """
        if isinstance(model_outputs, OmniOutput):
            self._batch_state = None
            self._batch_state_spans = None
            return model_outputs

        hidden = model_outputs  # (S, H)
        info_dicts: list[dict[str, Any]] = (
            kwargs.get("model_intermediate_buffer") or kwargs.get("runtime_additional_information") or []
        )

        # Ensure each request has an initialised state (defensive: typically
        # populated by preprocess on the first call).
        for info in info_dicts:
            if isinstance(info, dict) and not isinstance(info.get("audio_state"), dict):
                info["audio_state"] = {
                    "audio_lengths": 0,
                    "delayed_lengths": -1,
                    "is_audio": False,
                    "step": 0,
                }

        # Stash the per-row state for compute_logits to apply masks. logits
        # rows align with hidden rows, which align with info_dicts in order.
        self._batch_state = [(info["audio_state"] if isinstance(info, dict) else {}) for info in info_dicts]
        self._batch_state_spans = kwargs.get("request_token_spans")

        # Per-request (start, end) hidden-row spans from the runner. In mixed
        # prefill+decode steps the per-request token counts differ, so an equal
        # rows-per-request split would sample codes from the wrong rows.
        spans = kwargs.get("request_token_spans")

        # One accumulated-codes tensor per request, in batch order. The generic
        # splitter (``to_payload_element``) routes ``per_req_codes[idx]`` to
        # request ``idx``, so Stage 1 decodes each request's own codes instead
        # of request 0's. Skipped requests get an empty placeholder so indices
        # stay aligned (a shorter list would silently fall back to entry 0).
        per_req_codes: list[torch.Tensor] = []
        have_codes = False

        if hidden.numel() > 0:
            num_rows = hidden.shape[0]
            if spans is None or len(spans) != len(info_dicts):
                raise RuntimeError(
                    "MOSS-TTS make_omni_output requires request_token_spans "
                    f"for {len(info_dicts)} request infos, got {0 if spans is None else len(spans)}"
                )

            for i, info in enumerate(info_dicts):
                if not isinstance(info, dict):
                    per_req_codes.append(hidden.new_empty((0, self.n_vq), dtype=torch.long))
                    continue
                row_start, row_end = spans[i]
                row_end = min(int(row_end), num_rows)
                if row_start >= row_end:
                    per_req_codes.append(hidden.new_empty((0, self.n_vq), dtype=torch.long))
                    continue

                # Sample new audio codes from the last hidden state of this request.
                last_h = hidden[row_end - 1].unsqueeze(0)  # (1, H)
                state = info.get("audio_state", {}) or {}
                new_codes = self._sample_audio_codes(last_h, state)  # (n_vq,)

                acc = (info.get("audio_codes", {}) or {}).get("accumulated")
                if isinstance(acc, torch.Tensor) and acc.numel() > 0:
                    updated_acc = torch.cat([acc.to(new_codes.device), new_codes.unsqueeze(0)], dim=0)
                else:
                    updated_acc = new_codes.unsqueeze(0)

                info["audio_codes"] = {
                    "current": new_codes,
                    "accumulated": updated_acc,
                }
                per_req_codes.append(updated_acc)
                have_codes = True

        if not have_codes:
            return OmniOutput(
                text_hidden_states=hidden,
                multimodal_outputs={},
            )

        # Emit one accumulated-codes tensor per request as a list; the splitter
        # picks ``element[idx]`` per request on the way to Stage 1.
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={"codes": {"audio": per_req_codes}},
        )

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Map HF weight names to vLLM-Omni module names.

        HF layout (MossTTSDelayModel):
          language_model.model.*       → model.*
          language_model.lm_head.weight (if present)
          emb_ext.{i}.weight           → audio_embeddings.{i}.weight
          lm_heads.0.weight            → text_lm_head.weight
          lm_heads.{i+1}.weight        → audio_heads.{i}.weight  (i ≥ 0)
        """
        loaded: set[str] = set()
        params_dict = dict(self.named_parameters())

        # Buffer Qwen3 backbone weights (stripped of `language_model.` prefix)
        # and delegate to self.model.load_weights() so q/k/v and gate/up are
        # correctly fused into qkv_proj/gate_up_proj.
        backbone_weights: list[tuple[str, torch.Tensor]] = []

        for name, tensor in weights:
            # Qwen3 backbone — checkpoint stores keys as language_model.<layer>.
            if name.startswith("language_model.") and not name.startswith("language_model.lm_head"):
                backbone_weights.append((name[len("language_model.") :], tensor))
                continue

            # Text LM head (lm_heads.0 in upstream == language_model.lm_head)
            if name in ("lm_heads.0.weight", "language_model.lm_head.weight"):
                tgt = "text_lm_head.weight"
                if tgt in params_dict:
                    default_weight_loader(params_dict[tgt], tensor)
                    loaded.add(tgt)
                continue

            # Audio heads: lm_heads.{k}.weight for k >= 1 → audio_heads.{k-1}.weight
            if name.startswith("lm_heads."):
                parts = name.split(".")
                try:
                    k = int(parts[1])
                except (IndexError, ValueError):
                    continue
                if k >= 1:
                    tgt = f"audio_heads.{k - 1}.weight"
                    if tgt in params_dict:
                        default_weight_loader(params_dict[tgt], tensor)
                        loaded.add(tgt)
                continue

            # Audio embeddings: emb_ext.{i}.weight → audio_embeddings.{i}.weight
            if name.startswith("emb_ext."):
                tgt = name.replace("emb_ext.", "audio_embeddings.", 1)
                if tgt in params_dict:
                    default_weight_loader(params_dict[tgt], tensor)
                    loaded.add(tgt)
                continue

        backbone_loaded = self.model.load_weights(iter(backbone_weights))
        for n in backbone_loaded:
            loaded.add(f"model.{n}")

        # Build stacked weight caches for vectorised decode ops.
        self._stacked_audio_head_w = torch.stack(
            [h.weight.detach() for h in self.audio_heads], dim=0
        )  # (n_vq, audio_vocab+1, hidden_size)
        self._stacked_audio_emb_w = torch.stack(
            [e.weight.detach() for e in self.audio_embeddings], dim=0
        )  # (n_vq, audio_vocab+1, hidden_size)

        return loaded


# ---------------------------------------------------------------------------
# MossTTSRealtimeTalkerForGeneration
# ---------------------------------------------------------------------------


class MossTTSRealtimeTalkerForGeneration(nn.Module):
    """Stage-0 talker for MossTTSRealtime (1.7B, TTFB ~180 ms).

    Architecture differs from the delay model:
    * Backbone (Qwen3) consumes ``embed_tokens[0](text) + Σᵢ embed_tokens[i+1](audio_i)``.
    * The model does NOT have a text LM head — the text column at every
      decode step is forced to ``text_pad`` (or ``eos`` when the audio EOS
      token has been emitted), so we synthesise a deterministic logit row
      to feed the vLLM sampler.
    * Per-step audio generation runs the small ``local_transformer`` (4-layer
      Qwen3-style decoder, ``rvq=16`` codebooks) inside ``make_omni_output``.
      Stop condition: codebook-0 token equals ``audio_eos_token`` (1026).
    """

    have_multimodal_outputs: bool = True
    has_preprocess: bool = True
    has_postprocess: bool = True

    AUDIO_BOS = 1025
    AUDIO_EOS = 1026

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config: MossTTSRealtimeConfig = vllm_config.model_config.hf_config
        lang_cfg = self.config.language_config
        local_cfg = self.config.local_config

        self.hidden_size: int = int(lang_cfg.hidden_size)
        self.n_vq: int = int(self.config.rvq)
        self.audio_vocab_size: int = int(self.config.audio_vocab_size)
        self.audio_pad_token: int = int(self.config.audio_pad_token)
        self.text_pad_id: int = int(self.config.text_pad)
        self.audio_eos_id: int = int(getattr(self.config, "eos_token_id", 151645))
        self.text_vocab_size: int = int(lang_cfg.vocab_size)

        # Qwen3 backbone (uses the inner language_config). vLLM exposes its
        # internal weights under ``model.*`` so we keep that prefix and remap
        # the upstream ``language_model.*`` keys at load time.
        backbone_vllm_config = copy.copy(vllm_config)
        # Swap in the inner Qwen3 config so vLLM picks the right num_layers,
        # heads etc.; KV cache sizing already uses get_text_config().
        backbone_model_config = copy.copy(vllm_config.model_config)
        backbone_model_config.hf_config = lang_cfg
        backbone_model_config.hf_text_config = lang_cfg
        backbone_vllm_config.model_config = backbone_model_config
        self.model = Qwen3Model(
            vllm_config=backbone_vllm_config,
            prefix=_maybe_prefix(prefix, "model"),
        )

        # Outer per-channel embeddings: index 0 is text (vocab=text_vocab_size),
        # indices 1..rvq are audio codebooks (vocab=audio_vocab_size). Match
        # upstream's ``MossTTSRealtime.embed_tokens`` ModuleList exactly.
        self.embed_tokens = nn.ModuleList()
        self.embed_tokens.append(
            nn.Embedding(
                self.text_vocab_size,
                self.hidden_size,
                padding_idx=int(getattr(lang_cfg, "pad_token_id", 151643) or 151643),
            )
        )
        for _ in range(self.n_vq):
            self.embed_tokens.append(
                nn.Embedding(self.audio_vocab_size, self.hidden_size, padding_idx=self.audio_pad_token)
            )

        # Local depth transformer + per-codebook LM heads.
        self.local_transformer = MossTTSRealtimeLocalTransformer(local_cfg)
        self.local_lm_heads = nn.ModuleList(
            [nn.Linear(int(local_cfg.hidden_size), self.audio_vocab_size, bias=False) for _ in range(self.n_vq)]
        )

        # No real text LM head — compute_logits builds a one-hot row directly.
        # We still need a logits_processor so vLLM downstream stays happy with
        # something callable; a minimal pass-through suffices.
        self.logits_processor = LogitsProcessor(self.text_vocab_size)
        self._batch_state: list[dict[str, Any]] | None = None

        # Stacked weight cache built after load_weights().
        # shape: (n_vq, audio_vocab_size, hidden_size)
        self._stacked_audio_emb_w: torch.Tensor | None = None

        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {
            ("audio_codes", "current"),
            ("audio_codes", "accumulated"),
            ("hidden_states", "last"),
        }

    # ------------------------------------------------------------------
    # vLLM hooks
    # ------------------------------------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        # Used by vllm dummy profiling; real prefill goes through preprocess().
        return self.embed_tokens[0](input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
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

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor | None:
        """Synthesise a one-hot text logit row per request.

        The realtime model has no text LM head — text is always either
        ``text_pad`` (continue) or ``eos`` (stop because audio EOS just fired).
        """
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None or hidden_states.numel() == 0:
            return None

        B = hidden_states.shape[0]
        V = self.text_vocab_size
        logits = hidden_states.new_full((B, V), float("-inf"))

        states = self._batch_state or []
        for state, r0, r1 in _iter_state_row_spans(
            states,
            getattr(self, "_batch_state_spans", None),
            B,
        ):
            if state is None:
                logits[r0:r1, self.text_pad_id] = 0.0
                continue
            if state.get("is_stopping"):
                logits[r0:r1, self.audio_eos_id] = 0.0
                continue
            # Streaming text: emit the next remaining text token, then advance
            # the cursor. Once exhausted, fall back to ``text_pad``.
            cursor = int(state.get("text_cursor", 0))
            remaining = state.get("remaining_text") or []
            if 0 <= cursor < len(remaining):
                tok = int(remaining[cursor])
                state["text_cursor"] = cursor + 1
            else:
                tok = self.text_pad_id
            logits[r0:r1, tok] = 0.0
        # Defensive: if no states recorded yet (very first call before
        # make_omni_output ran), default to text_pad everywhere.
        if not states:
            logits[:, self.text_pad_id] = 0.0
        return logits

    # ------------------------------------------------------------------
    # Embedding (text + audio codebooks, additive)
    # ------------------------------------------------------------------

    def _build_input_embeds(
        self,
        text_ids: torch.Tensor,  # (T,)
        audio_codes: torch.Tensor | None,  # (T, n_vq) or None
    ) -> torch.Tensor:
        embeds = self.embed_tokens[0](text_ids)
        if audio_codes is None:
            return embeds
        if self._stacked_audio_emb_w is not None:
            # Single gather: (n_vq, vocab, H)[arange[:,None], codes.T] → (n_vq, T, H) → sum → (T, H)
            n_vq_idx = torch.arange(self.n_vq, device=audio_codes.device)
            codes_t = audio_codes.t().clamp(0, self.audio_vocab_size - 1)  # (n_vq, T)
            return embeds + self._stacked_audio_emb_w[n_vq_idx.unsqueeze(1), codes_t].sum(0)
        for i in range(self.n_vq):
            col = audio_codes[:, i].clamp_(0, self.audio_vocab_size - 1)
            embeds = embeds + self.embed_tokens[i + 1](col)
        return embeds

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        device = input_ids.device
        span_len = int(input_ids.shape[0])
        audio_state = info_dict.get("audio_state")
        is_first_call = not isinstance(audio_state, dict)

        if span_len > 1 or is_first_call:
            # Prefill: read the full reference-audio code grid (channels x T)
            # from ``info_dict["codes"]["ref"]``; slice the chunk that aligns
            # with this prefill window.
            ref_codes = (info_dict.get("codes", {}) or {}).get("ref")
            ref_offset = int(info_dict.get("ref_offset", 0))
            chunk_audio = None
            if isinstance(ref_codes, torch.Tensor) and ref_codes.numel() > 0:
                if ref_codes.dim() == 1 and ref_codes.numel() % self.n_vq == 0:
                    ref_codes = ref_codes.view(-1, self.n_vq)
                if isinstance(ref_codes, torch.Tensor) and ref_codes.dim() == 2:
                    end_off = ref_offset + span_len
                    sliced = ref_codes[ref_offset:end_off]
                    if sliced.shape[0] == span_len:
                        chunk_audio = sliced.to(device=device, dtype=torch.long)
            embeds = self._build_input_embeds(input_ids, chunk_audio)
            current_codes = (
                chunk_audio[-1].detach()
                if chunk_audio is not None
                else torch.full((self.n_vq,), self.audio_pad_token, dtype=torch.long, device=device)
            )
            # Capture the streaming-text list from the request (set by the
            # realtime end2end builder; empty for the delay variants which
            # don't take this code path anyway).
            remaining = (info_dict.get("ids", {}) or {}).get("all")
            if not isinstance(remaining, list):
                remaining = []
            info_update: dict[str, Any] = {
                "audio_state": {
                    "is_stopping": False,
                    "step": 0,
                    "text_cursor": 0,
                    "remaining_text": list(remaining),
                },
                "audio_codes": {"current": current_codes},
                "ref_offset": ref_offset + span_len,
            }
            return input_ids, embeds, info_update

        # Decode step: text_token from the just-sampled vLLM logit, plus the
        # audio codes the local transformer produced last step.
        prev_codes = (info_dict.get("audio_codes", {}) or {}).get("current")
        if prev_codes is None:
            prev_codes = torch.full((self.n_vq,), self.audio_pad_token, dtype=torch.long, device=device)
        embeds = self._build_input_embeds(
            input_ids.reshape(-1),
            prev_codes.to(device=device, dtype=torch.long).unsqueeze(0),
        )
        return input_ids, embeds, {}

    def postprocess(self, hidden_states: torch.Tensor, **_: Any) -> dict[str, Any]:
        if hidden_states.numel() == 0:
            return {}
        return {"hidden_states": {"last": hidden_states[-1].detach()}}

    # ------------------------------------------------------------------
    # Per-step audio generation via local transformer
    # ------------------------------------------------------------------

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            self._batch_state = None
            self._batch_state_spans = None
            return model_outputs

        hidden = model_outputs  # (S, H)
        info_dicts: list[dict[str, Any]] = (
            kwargs.get("model_intermediate_buffer") or kwargs.get("runtime_additional_information") or []
        )

        # Defensive state init.
        for info in info_dicts:
            if isinstance(info, dict) and not isinstance(info.get("audio_state"), dict):
                info["audio_state"] = {"is_stopping": False, "step": 0}

        self._batch_state = [(info["audio_state"] if isinstance(info, dict) else {}) for info in info_dicts]
        self._batch_state_spans = kwargs.get("request_token_spans")

        # See delay talker: real per-request row spans from the runner are
        # required because mixed prefill+decode steps have unequal row counts.
        spans = kwargs.get("request_token_spans")
        # Per-request accumulated codes in batch order, pre-filled with empty
        # placeholders so every skip path (stopped / eos / bos / pad) keeps
        # indices aligned for to_payload_element's per-request routing. A
        # shorter list would silently fall back to request 0's codes.
        per_req_codes: list[torch.Tensor] = [hidden.new_empty((0, self.n_vq), dtype=torch.long) for _ in info_dicts]
        have_codes = False
        if hidden.numel() > 0 and info_dicts:
            num_rows = hidden.shape[0]
            if spans is None or len(spans) != len(info_dicts):
                raise RuntimeError(
                    "MOSS-TTS make_omni_output requires request_token_spans "
                    f"for {len(info_dicts)} request infos, got {0 if spans is None else len(spans)}"
                )
            for i, info in enumerate(info_dicts):
                if not isinstance(info, dict):
                    continue
                row_start, row_end = spans[i]
                row_end = min(int(row_end), num_rows)
                if row_start >= row_end:
                    continue

                state = info.get("audio_state", {}) or {}
                if state.get("is_stopping"):
                    continue  # already stopped — no more audio frames

                last_h = hidden[row_end - 1].unsqueeze(0)  # (1, H)
                # Sampling parameters mirror upstream ``MossTTSRealtimeInference.generate``:
                # 0.8 / 0.6 / 30 + 1.1 repetition penalty over a 50-frame window.
                rep_window = 50
                hist_per_cb: list[list[int]] = []
                acc_for_hist = (info.get("audio_codes", {}) or {}).get("accumulated")
                if isinstance(acc_for_hist, torch.Tensor) and acc_for_hist.numel() > 0:
                    tail = acc_for_hist[-rep_window:].long().cpu().tolist()
                    for cb in range(self.n_vq):
                        hist_per_cb.append([row[cb] for row in tail])
                else:
                    hist_per_cb = [[] for _ in range(self.n_vq)]
                new_codes = self.local_transformer.generate_frame(
                    last_h,
                    self.local_lm_heads,
                    temperature=0.8,
                    top_p=0.6,
                    top_k=30,
                    do_sample=True,
                    repetition_penalty=1.1,
                    history_per_codebook=hist_per_cb,
                ).squeeze(0)  # (n_vq,)
                if int(state.get("step", 0)) < 5 or int(state.get("step", 0)) % 50 == 0:
                    logger.debug(
                        "[MossTTSRealtime make_omni] step=%d ch0=%d cursor=%d/%d",
                        int(state.get("step", 0)),
                        int(new_codes[0].item()),
                        int(state.get("text_cursor", 0)),
                        len(state.get("remaining_text") or []),
                    )

                ch0 = int(new_codes[0].item())
                # Stop condition mirrors upstream: codebook 0 == eos_audio_id.
                if ch0 == self.AUDIO_EOS:
                    state["is_stopping"] = True
                    state["step"] = int(state.get("step", 0)) + 1
                    info["audio_codes"] = {
                        "current": new_codes,
                        "accumulated": (info.get("audio_codes", {}) or {}).get("accumulated"),
                    }
                    continue  # don't append the eos frame to accumulated

                if ch0 in (self.AUDIO_BOS, self.audio_pad_token):
                    # Skip the bos / pad frames — they don't decode to real audio.
                    state["step"] = int(state.get("step", 0)) + 1
                    info["audio_codes"] = {
                        "current": new_codes,
                        "accumulated": (info.get("audio_codes", {}) or {}).get("accumulated"),
                    }
                    continue

                acc = (info.get("audio_codes", {}) or {}).get("accumulated")
                if isinstance(acc, torch.Tensor) and acc.numel() > 0:
                    updated_acc = torch.cat([acc.to(new_codes.device), new_codes.unsqueeze(0)], dim=0)
                else:
                    updated_acc = new_codes.unsqueeze(0)

                info["audio_codes"] = {"current": new_codes, "accumulated": updated_acc}
                state["step"] = int(state.get("step", 0)) + 1
                per_req_codes[i] = new_codes.unsqueeze(0)
                have_codes = True

        if not have_codes:
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

        # Emit one current-frame tensor per request as a list; the splitter
        # routes ``element[idx]`` to request idx, so each Stage-1 request receives
        # only its own newly generated raw codec row.
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={
                "codes": {"audio": per_req_codes},
            },
        )

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Remap upstream MossTTSRealtime checkpoint names → vendored layout.

        Mapping:
          embed_tokens.{i}.*               → embed_tokens.{i}.*               (kept)
          language_model.embed_tokens.*    → model.embed_tokens.*             (Qwen3 inner)
          language_model.layers.*          → model.layers.*
          language_model.norm.*            → model.norm.*
          local_transformer.model.embed_tokens.* → local_transformer.model.codec_embedding.*
          local_transformer.model.*        → local_transformer.model.*        (shared body, kept)
          local_transformer.local_lm_heads.* → local_lm_heads.*               (top-level)
        """
        loaded: set[str] = set()
        params_dict = dict(self.named_parameters())

        # Qwen3 backbone weights need vLLM's stacked-params loader to fuse
        # ``q_proj``/``k_proj``/``v_proj`` → ``qkv_proj`` and
        # ``gate_proj``/``up_proj`` → ``gate_up_proj``. ``default_weight_loader``
        # alone leaves those fused params un-initialised, which silently turns
        # the backbone into a slightly-corrupted model that never emits EOS.
        backbone_weights: list[tuple[str, torch.Tensor]] = []
        skipped: list[str] = []
        for name, tensor in weights:
            if name.startswith("language_model."):
                backbone_weights.append((name[len("language_model.") :], tensor))
                continue
            if name.startswith("local_transformer.model."):
                # Shared CodePredictorBaseModel keeps the ``.model.`` nesting and
                # the per-layer/norm names verbatim; only the codebook embedding
                # is renamed (upstream ``embed_tokens`` → ``codec_embedding``).
                sub = name[len("local_transformer.model.") :]
                if sub.startswith("rotary_emb.inv_freq"):
                    continue  # non-persistent buffer, recomputed at runtime
                if sub.startswith("embed_tokens."):
                    sub = "codec_embedding." + sub[len("embed_tokens.") :]
                tgt = "local_transformer.model." + sub
            elif name.startswith("local_transformer.local_lm_heads."):
                tgt = "local_lm_heads." + name[len("local_transformer.local_lm_heads.") :]
            else:
                tgt = name
            if tgt in params_dict:
                default_weight_loader(params_dict[tgt], tensor)
                loaded.add(tgt)
            else:
                skipped.append(f"{name}->{tgt}")

        # Delegate Qwen3 weights to its own loader (handles fused params).
        backbone_loaded = self.model.load_weights(iter(backbone_weights))
        for n in backbone_loaded:
            loaded.add(f"model.{n}")

        # Build stacked embedding cache for vectorised decode ops.
        # embed_tokens[0] is the text embedding; [1..n_vq] are audio codebooks.
        self._stacked_audio_emb_w = torch.stack(
            [self.embed_tokens[i + 1].weight.detach() for i in range(self.n_vq)], dim=0
        )  # (n_vq, audio_vocab_size, hidden_size)

        logger.info(
            "[MossTTSRealtime] loaded %d/%d params; skipped=%d (first 5: %s)",
            len(loaded),
            len(params_dict),
            len(skipped),
            skipped[:5],
        )
        not_loaded = [n for n in params_dict if n not in loaded]
        if not_loaded:
            logger.warning("[MossTTSRealtime] %d params NOT loaded (first 5: %s)", len(not_loaded), not_loaded[:5])
        return loaded


# ---------------------------------------------------------------------------
# MossTTSLocalTalkerForGeneration
# ---------------------------------------------------------------------------


class MossTTSLocalTalkerForGeneration(nn.Module):
    """Stage-0 talker for MossTTSLocalModel (MOSS-TTS-Local-Transformer-v1.5).

    Architecture differs from both existing variants:

    * Backbone (Qwen3; checkpoint prefix ``transformer.*``, unlike Delay/
      Realtime's ``language_model.*``) consumes additive text+audio fusion
      the same way Delay does (real text embeddings + masked per-codebook
      audio embeddings from ``info["codes"]["ref"]`` / ``info["audio_codes"]
      ["current"]``), but there is no delay-pattern countdown: the prompt
      already ends right at the audio_start boundary, and every step from
      the first decode onward drives exactly one audio frame.
    * The checkpoint's own ``text_lm_head`` is loaded (trained for a
      teacher-forced text loss) but never consulted at inference: the
      official reference's own ``generate()`` hardcodes the next
      text-channel input to ``audio_assistant_slot_token_id`` regardless of
      what ``text_lm_head`` would predict, because
      ``local_text_head_mode == "binary"`` for this checkpoint.
      ``compute_logits`` therefore synthesises a one-hot logit row
      (Realtime-style), forced to ``audio_assistant_slot_token_id`` while
      continuing or ``im_end_token_id`` once the local transformer's binary
      head decides to stop.
    * Per-step audio generation runs ``MossTTSLocalDepthTransformer``
      (1-layer GPT2-style local transformer, interleaved RoPE, ``n_vq=12``
      codebooks -- structurally different from Realtime's Qwen3-style
      ``CodePredictorBaseModel``) in the runner-owned ``talker_mtp`` path. Stop
      condition: the local transformer's binary head selects the
      "audio_end" candidate (logits index 1).
    """

    have_multimodal_outputs: bool = True
    has_preprocess: bool = True
    has_postprocess: bool = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config: MossTTSLocalConfig = vllm_config.model_config.hf_config

        qwen3_cfg = self.config.qwen3_config
        hidden_size = int(qwen3_cfg.hidden_size)
        self.hidden_size = hidden_size
        self.n_vq: int = int(self.config.n_vq)
        self.audio_vocab_size: int = int(self.config.audio_vocab_size)
        self.audio_pad_token_id: int = int(self.config.audio_pad_token_id)
        self.text_vocab_size: int = int(qwen3_cfg.vocab_size)

        self.audio_assistant_slot_token_id: int = int(self.config.audio_assistant_slot_token_id)
        self.im_end_token_id: int = int(self.config.im_end_token_id)

        # Qwen3 backbone. get_text_config() is sufficient for vLLM to size
        # KV cache / heads correctly, as long as AutoConfig resolves to the
        # local MossTTSLocalConfig rather than the HF remote-code class.
        self.model = Qwen3Model(
            vllm_config=vllm_config,
            prefix=_maybe_prefix(prefix, "model"),
        )

        # Real text LM head -- loaded from the checkpoint but never consulted
        # at inference (see class docstring); kept purely as a destination
        # for ``text_lm_head.weight`` so load_weights() loads every tensor.
        self.text_lm_head = ParallelLMHead(
            self.text_vocab_size,
            hidden_size,
            bias=False,
            prefix=_maybe_prefix(prefix, "text_lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.text_vocab_size)

        # Per-codebook audio embeddings/heads. audio_pad_token_id (==
        # audio_vocab_size) is outside the embedding table -- masked to zero
        # contribution rather than stored as a 1025th row (mirrors upstream
        # ``_build_inputs_embeds``'s ``valid_mask`` masking).
        self.audio_embeddings = nn.ModuleList(
            [nn.Embedding(self.audio_vocab_size, hidden_size) for _ in range(self.n_vq)]
        )
        self.audio_lm_heads = nn.ModuleList(
            [nn.Linear(hidden_size, self.audio_vocab_size, bias=False) for _ in range(self.n_vq)]
        )
        self.local_text_lm_head = nn.Linear(hidden_size, 2, bias=False)
        self.local_transformer = MossTTSLocalDepthTransformer(self.config.gpt2_config, hidden_size=hidden_size)

        self._batch_state: list[dict[str, Any]] | None = None
        # Stacked embedding cache built after load_weights() for vectorised
        # additive-fusion gathers (avoids an n_vq-iteration Python loop).
        self._stacked_audio_emb_w: torch.Tensor | None = None
        self.mtp_hidden_size = hidden_size
        self.talker_mtp_accepts_req_infos = True

        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {
            ("audio_codes", "current"),
            ("audio_codes", "accumulated"),
            ("hidden_states", "last"),
        }

    # ------------------------------------------------------------------
    # vLLM hooks
    # ------------------------------------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
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

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor | None:
        """Synthesise a one-hot text logit row per request.

        The text channel is never freely generated here: the upstream
        reference hardcodes it to ``audio_assistant_slot_token_id`` while
        audio frames keep coming, and only emits ``im_end_token_id`` once the
        local transformer's binary head decides to stop (see class
        docstring for why the real ``text_lm_head`` is bypassed).
        """
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None or hidden_states.numel() == 0:
            return None

        num_rows = hidden_states.shape[0]
        logits = hidden_states.new_full((num_rows, self.text_vocab_size), float("-inf"))

        states = self._batch_state or []
        for state, row_start, row_end in _iter_state_row_spans(
            states,
            getattr(self, "_batch_state_spans", None),
            num_rows,
        ):
            if state is not None and state.get("is_stopping"):
                forced_im_end_count = int(state.get("_forced_im_end_count", 0))
                if forced_im_end_count == 1:
                    logger.warning(
                        "MOSS-TTS Local request is still running after "
                        "im_end_token_id=%d was forced once; verify stage-0 "
                        "stop_token_ids/eos includes this id.",
                        self.im_end_token_id,
                    )
                state["_forced_im_end_count"] = forced_im_end_count + 1
                logits[row_start:row_end, self.im_end_token_id] = 0.0
            else:
                logits[row_start:row_end, self.audio_assistant_slot_token_id] = 0.0
        if not states:
            logits[:, self.audio_assistant_slot_token_id] = 0.0
        return logits

    # ------------------------------------------------------------------
    # Embedding (text + audio codebooks, additive)
    # ------------------------------------------------------------------

    def _audio_embed(self, codes: torch.Tensor) -> torch.Tensor:
        """``codes``: ``(T, n_vq)`` long, possibly containing ``audio_pad_token_id``
        (outside the valid embedding range). Returns the ``(T, H)`` additive
        audio embedding, masking out pad positions per codebook.
        """
        device = codes.device
        valid_mask = codes.ne(self.audio_pad_token_id)  # (T, n_vq)
        safe_codes = codes.masked_fill(~valid_mask, 0).clamp(0, self.audio_vocab_size - 1)
        if self._stacked_audio_emb_w is not None:
            n_vq_idx = torch.arange(self.n_vq, device=device)
            codes_t = safe_codes.t()  # (n_vq, T)
            gathered = self._stacked_audio_emb_w[n_vq_idx.unsqueeze(1), codes_t]  # (n_vq, T, H)
            gathered = gathered * valid_mask.t().unsqueeze(-1)
            return gathered.sum(0)
        embeds: torch.Tensor | None = None
        for i, emb_layer in enumerate(self.audio_embeddings):
            col = emb_layer(safe_codes[:, i]) * valid_mask[:, i].unsqueeze(-1)
            embeds = col if embeds is None else embeds + col
        return embeds

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build per-step input embeddings (text + audio additive fusion).

        Prefill: initialise the per-request ``{is_stopping, step}`` state.
        Decode: combine the forced text token (from ``compute_logits``) with
        the audio codes the local transformer produced last step.
        """
        device = input_ids.device
        span_len = int(input_ids.shape[0])
        audio_state = info_dict.get("audio_state")
        is_first_call = not isinstance(audio_state, dict)

        if span_len > 1 or is_first_call:
            embeds = self.model.embed_tokens(input_ids)

            ref_codes = (info_dict.get("codes", {}) or {}).get("ref")
            ref_offset = int(info_dict.get("ref_offset", 0))
            last_ref_row = None
            if isinstance(ref_codes, torch.Tensor) and ref_codes.numel() > 0:
                if ref_codes.dim() == 1 and ref_codes.numel() % self.n_vq == 0:
                    ref_codes = ref_codes.view(-1, self.n_vq)
                if isinstance(ref_codes, torch.Tensor) and ref_codes.dim() == 2:
                    end_off = ref_offset + span_len
                    chunk = ref_codes[ref_offset:end_off]
                    if chunk.numel() > 0 and chunk.shape[0] == span_len:
                        codes = chunk.to(device=device, dtype=torch.long)
                        embeds = embeds + self._audio_embed(codes)
                        last_ref_row = codes[-1].detach()

            current_codes = (
                last_ref_row
                if last_ref_row is not None
                else torch.full((self.n_vq,), self.audio_pad_token_id, dtype=torch.long, device=device)
            )

            max_new_frames = info_dict.get("max_new_frames", [-1])[0]

            info_update: dict[str, Any] = {
                "audio_state": {
                    "is_stopping": False,
                    "step": 0,
                    "max_new_frames": max_new_frames,
                },
                "audio_codes": {"current": current_codes, "emit": False},
                "ref_offset": ref_offset + span_len,
            }
            return input_ids, embeds, info_update

        # Decode step: return the plain forced text-token embedding. The local
        # depth transformer runs in OmniGPUModelRunner.talker_mtp before the
        # backbone forward; it adds the freshly sampled audio-code embedding
        # and stores the generated code frame in model_intermediate_buffer.
        text_embed = self.model.embed_tokens(input_ids.reshape(1))  # (1, H)
        hs = info_dict.get("hidden_states", {}) or {}
        last_hidden = hs.get("last")
        if isinstance(last_hidden, torch.Tensor) and last_hidden.numel() > 0:
            mtp_hidden = last_hidden.to(device=device, dtype=text_embed.dtype).reshape(1, -1)
        else:
            mtp_hidden = torch.zeros((1, self.hidden_size), device=device, dtype=text_embed.dtype)
        return input_ids, text_embed, {"mtp_inputs": (mtp_hidden, torch.zeros_like(mtp_hidden))}

    def postprocess(self, hidden_states: torch.Tensor, **_: Any) -> dict[str, Any]:
        if hidden_states.numel() == 0:
            return {}
        return {"hidden_states": {"last": hidden_states[-1].detach()}}

    @torch.inference_mode()
    def talker_mtp(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        text_step: torch.Tensor,
        do_sample: bool | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        generator: torch.Generator | None = None,
        req_infos: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, None]:
        bsz = int(input_embeds.shape[0])
        dev = input_embeds.device
        input_embeds_out = input_embeds.reshape(bsz, -1).clone()
        req_infos = req_infos or [{} for _ in range(bsz)]

        def write_update(info: dict[str, Any], update: dict[str, Any]) -> None:
            for key, value in update.items():
                if isinstance(value, dict):
                    target = info.setdefault(key, {})
                    target.update(value)
                else:
                    info[key] = value

        # Match MOSS-TTS Local v1.5's model-card/SGLang defaults. The generic
        # stage SamplingParams are for the backbone token loop and should not
        # silently retune the local depth transformer's audio-channel sampler.
        audio_temperature = 1.7
        audio_top_k = 25
        audio_top_p = 0.8
        do_sample_val = True if do_sample is None else bool(do_sample)

        for i in range(bsz):
            info = req_infos[i] if i < len(req_infos) and isinstance(req_infos[i], dict) else {}
            state = dict(info.get("audio_state", {}) or {})
            step = int(state.get("step", 0))
            max_new_frames = int(state.get("max_new_frames", -1))
            acc = (info.get("audio_codes", {}) or {}).get("accumulated")

            if state.get("is_stopping"):
                write_update(
                    info,
                    {
                        "audio_state": state,
                        "audio_codes": {
                            "current": input_embeds.new_empty((0, self.n_vq), dtype=torch.long),
                            "emit": False,
                        },
                    },
                )
                continue

            rep_window = 50
            if isinstance(acc, torch.Tensor) and acc.numel() > 0:
                tail = acc[-rep_window:].long().cpu().tolist()
                hist_per_cb = [[row[cb] for row in tail] for cb in range(self.n_vq)]
            else:
                hist_per_cb = [[] for _ in range(self.n_vq)]

            should_continue_t, new_codes_b = self.local_transformer.generate_frame(
                last_talker_hidden[i : i + 1],
                self.audio_lm_heads,
                self.audio_embeddings,
                self.local_text_lm_head,
                n_vq=self.n_vq,
                do_sample=do_sample_val,
                temperature=audio_temperature,
                top_k=audio_top_k,
                top_p=audio_top_p,
                repetition_penalty=1.0,
                history_per_codebook=hist_per_cb,
                generator=generator,
            )
            should_continue = bool(should_continue_t[0].item())
            new_codes = new_codes_b.squeeze(0).to(device=dev, dtype=torch.long)

            if not should_continue or (0 <= max_new_frames <= step):
                state["is_stopping"] = True
                state["step"] = step + 1
                write_update(
                    info,
                    {
                        "audio_state": state,
                        "audio_codes": {
                            "current": input_embeds.new_empty((0, self.n_vq), dtype=torch.long),
                            "accumulated": acc,
                            "emit": False,
                        },
                    },
                )
                continue

            if isinstance(acc, torch.Tensor) and acc.numel() > 0:
                updated_acc = torch.cat([acc.to(dev), new_codes.unsqueeze(0)], dim=0)
            else:
                updated_acc = new_codes.unsqueeze(0)

            state["step"] = step + 1
            current = new_codes.unsqueeze(0)
            input_embeds_out[i : i + 1] = input_embeds_out[i : i + 1] + self._audio_embed(current).to(
                dtype=input_embeds_out.dtype
            )
            write_update(
                info,
                {
                    "audio_state": state,
                    "audio_codes": {
                        "current": current,
                        "accumulated": updated_acc,
                        "emit": True,
                    },
                },
            )

        return input_embeds_out, None

    # ------------------------------------------------------------------
    # Package runner-generated audio frames
    # ------------------------------------------------------------------

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            self._batch_state = None
            self._batch_state_spans = None
            return model_outputs

        hidden = model_outputs  # (S, H)
        info_dicts: list[dict[str, Any]] = (
            kwargs.get("model_intermediate_buffer") or kwargs.get("runtime_additional_information") or []
        )

        for info in info_dicts:
            if isinstance(info, dict) and not isinstance(info.get("audio_state"), dict):
                info["audio_state"] = {"is_stopping": False, "step": 0, "max_new_frames": -1}

        self._batch_state = [(info["audio_state"] if isinstance(info, dict) else {}) for info in info_dicts]
        self._batch_state_spans = kwargs.get("request_token_spans")

        per_req_codes: list[torch.Tensor] = []
        have_codes = False
        for info in info_dicts:
            current = (info.get("audio_codes", {}) or {}).get("current") if isinstance(info, dict) else None
            emit = bool((info.get("audio_codes", {}) or {}).get("emit")) if isinstance(info, dict) else False
            if emit and isinstance(current, torch.Tensor) and current.numel() > 0:
                codes = current.to(device=hidden.device, dtype=torch.long)
                if codes.dim() == 1:
                    codes = codes.unsqueeze(0)
                per_req_codes.append(codes)
                have_codes = True
            else:
                per_req_codes.append(hidden.new_empty((0, self.n_vq), dtype=torch.long))

        if not have_codes:
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

        # Local-v1.5 emits the current raw code frame (no delay pattern); the
        # pipeline routes those rows through ``talker2codec_raw_async_chunk``.
        per_req_ref_codes: list[torch.Tensor] = []
        for info in info_dicts:
            ref_codes = (info.get("codes", {}) or {}).get("ref") if isinstance(info, dict) else None
            if isinstance(ref_codes, torch.Tensor) and ref_codes.numel() > 0:
                ref_codes = ref_codes.detach()
                if ref_codes.dim() == 1 and ref_codes.numel() % self.n_vq == 0:
                    ref_codes = ref_codes.view(-1, self.n_vq)
                if ref_codes.dim() == 2 and ref_codes.shape[1] == self.n_vq:
                    per_req_ref_codes.append(ref_codes.to(device=hidden.device, dtype=torch.long))
                    continue
            per_req_ref_codes.append(hidden.new_empty((0, self.n_vq), dtype=torch.long))

        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={
                "codes": {"audio": per_req_codes, "ref": per_req_ref_codes},
            },
        )

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Map checkpoint weight names to vLLM-Omni module names.

        Checkpoint layout (prefix ``transformer.*`` for the backbone --
        unlike Delay/Realtime's ``language_model.*``):
          transformer.*               → model.*            (Qwen3 backbone)
          text_lm_head.weight         → text_lm_head.weight
          audio_embeddings.{i}.weight → audio_embeddings.{i}.weight
          audio_lm_heads.{i}.weight   → audio_lm_heads.{i}.weight
          local_text_lm_head.weight   → local_text_lm_head.weight
          local_transformer.h.0.*     → local_transformer.h.0.*
          local_transformer.ln_f.*    → local_transformer.ln_f.*

        Every "tied" pair in the upstream config is stored as a full,
        separate tensor in this checkpoint (confirmed via the raw
        safetensors header: 438 keys, no dedup), so aside from the backbone
        prefix this is pure 1:1 name-based loading.
        """
        loaded: set[str] = set()
        params_dict = dict(self.named_parameters())

        backbone_weights: list[tuple[str, torch.Tensor]] = []
        skipped: list[str] = []

        for name, tensor in weights:
            if name.startswith("transformer."):
                backbone_weights.append((name[len("transformer.") :], tensor))
                continue
            if name in params_dict:
                default_weight_loader(params_dict[name], tensor)
                loaded.add(name)
            else:
                skipped.append(name)

        backbone_loaded = self.model.load_weights(iter(backbone_weights))
        for n in backbone_loaded:
            loaded.add(f"model.{n}")

        self._stacked_audio_emb_w = torch.stack(
            [e.weight.detach() for e in self.audio_embeddings], dim=0
        )  # (n_vq, audio_vocab_size, hidden_size)

        logger.info(
            "[MossTTSLocal] loaded %d/%d params; skipped=%d (first 5: %s)",
            len(loaded),
            len(params_dict),
            len(skipped),
            skipped[:5],
        )
        not_loaded = [n for n in params_dict if n not in loaded]
        if not_loaded:
            logger.warning("[MossTTSLocal] %d params NOT loaded (first 5: %s)", len(not_loaded), not_loaded[:5])
        return loaded


__all__ = [
    "MossTTSDelayTalkerForGeneration",
    "MossTTSRealtimeTalkerForGeneration",
    "MossTTSLocalTalkerForGeneration",
]
