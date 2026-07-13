# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IndexTTS2 Stage 0: GPT-2 AR Talker with vLLM-native PagedAttention.

Predicts mel codes autoregressively and collects hidden_states as latent
for Stage 1 (S2Mel + BigVGAN).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.utils.speaker_cache import get_speaker_cache

from .configuration_indextts2 import IndexTTS2Config
from .gpt.conformer_encoder import ConformerEncoder
from .gpt.embeddings import LearnedPositionEmbeddings
from .gpt.perceiver import PerceiverResampler
from .preprocess_utils import (
    compute_fbank,
    load_campplus,
    load_qwen_emotion,
    load_reference_audio,
    load_semantic_codec,
    load_wav2vec2,
    resolve_model_file,
    wav2vec_extract,
)

logger = init_logger(__name__)


def _find_most_similar_cosine(query: torch.Tensor, matrix: torch.Tensor) -> int:
    sims = F.cosine_similarity(query.float(), matrix.float(), dim=1)
    return int(torch.argmax(sims).item())


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class IndexTTS2TalkerForConditionalGeneration(nn.Module):
    """vLLM-native GPT-2 AR talker for IndexTTS2.

    Stage 0 of the two-stage pipeline. Predicts mel codes (8194 vocab)
    and accumulates hidden_states as latent for Stage 1 S2Mel decoder.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        self.config: IndexTTS2Config = vllm_config.model_config.hf_config  # type: ignore[assignment]
        gpt_cfg = self.config.gpt

        self.model_dim = gpt_cfg["model_dim"]
        self.num_layers = gpt_cfg["layers"]
        self.num_heads = gpt_cfg["heads"]
        self.max_mel_tokens = gpt_cfg["max_mel_tokens"]
        self.max_text_tokens = gpt_cfg["max_text_tokens"]
        self.number_mel_codes = gpt_cfg["number_mel_codes"]
        self.start_mel_token = gpt_cfg["start_mel_token"]
        self.stop_mel_token = gpt_cfg["stop_mel_token"]
        self.number_text_tokens = gpt_cfg.get("number_text_tokens", 12000)
        self.condition_num_latent = gpt_cfg.get("condition_num_latent", 32)

        # --- Flags for vLLM-Omni framework ---
        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = True
        self.requires_raw_input_tokens = True
        self.enable_update_additional_information = True
        # S2Mel consumes the complete code sequence only at request end, but
        # the connector accumulator needs to observe each appendable AR delta.
        # Keep runner request-end gating disabled and emit per-step rows from
        # make_omni_output; the full-payload builder reconstructs the sequence.
        self.omni_payload_at_request_end = False
        self.omni_request_end_token_ids = ()
        self._speaker_cache = get_speaker_cache()
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {
            ("codes", "mel"),
            ("hidden_states", "latent"),
            ("meta", "mel_start_offset"),
            ("meta", "latent_acc"),
            ("meta", "mel_code_count"),
        }

        # --- GPT-2 transformer (vLLM-native with PagedAttention) ---
        # Build a GPT2Config-like object for GPT2Block
        from transformers import GPT2Config as HFGpt2Config

        max_seq_len = self.max_mel_tokens + self.max_text_tokens + self.condition_num_latent + 4
        gpt2_config = HFGpt2Config(
            vocab_size=self.number_mel_codes,
            n_positions=max_seq_len,
            n_ctx=max_seq_len,
            n_embd=self.model_dim,
            n_layer=self.num_layers,
            n_head=self.num_heads,
            activation_function="gelu_new",
            layer_norm_epsilon=1e-5,
        )
        # Store for GPT2Block construction
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.start_layer, self.end_layer, self.h = make_layers(
            gpt2_config.n_layer,
            lambda prefix: GPT2Block(gpt2_config, cache_config, quant_config, prefix=prefix),
            prefix=f"{prefix}.h",
        )
        self.ln_f = nn.LayerNorm(self.model_dim, eps=gpt2_config.layer_norm_epsilon)
        self.final_norm = nn.LayerNorm(self.model_dim, eps=gpt2_config.layer_norm_epsilon)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.model_dim
        )

        # --- Embeddings ---
        self.text_embedding = nn.Embedding(self.number_text_tokens + 1, self.model_dim)
        self.mel_embedding = nn.Embedding(self.number_mel_codes, self.model_dim)
        self.text_pos_embedding = LearnedPositionEmbeddings(self.max_text_tokens + 2, self.model_dim)
        self.mel_pos_embedding = LearnedPositionEmbeddings(self.max_mel_tokens + 2 + 1, self.model_dim)
        self.speed_emb = nn.Embedding(2, self.model_dim)
        self.emo_layer = nn.Linear(self.model_dim, self.model_dim)
        self.emovec_layer = nn.Linear(1024, self.model_dim)

        # --- Mel head (logits) ---
        if get_pp_group().is_last_rank:
            self.mel_head = ParallelLMHead(
                self.number_mel_codes,
                self.model_dim,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "mel_head"),
            )
        else:
            self.mel_head = None  # type: ignore[assignment]
        self.logits_processor = LogitsProcessor(self.number_mel_codes)

        # --- Conditioning modules (speaker / emotion) ---
        cond_cfg = gpt_cfg.get("condition_module", {})
        emo_cond_cfg = gpt_cfg.get("emo_condition_module", {})

        self.conditioning_encoder = ConformerEncoder(
            input_size=1024,
            output_size=cond_cfg.get("output_size", 512),
            linear_units=cond_cfg.get("linear_units", 2048),
            attention_heads=cond_cfg.get("attention_heads", 8),
            num_blocks=cond_cfg.get("num_blocks", 6),
            input_layer=cond_cfg.get("input_layer", "conv2d2"),
        )
        self.perceiver_encoder = PerceiverResampler(
            self.model_dim,
            dim_context=cond_cfg.get("output_size", 512),
            ff_mult=cond_cfg.get("perceiver_mult", 2),
            heads=cond_cfg.get("attention_heads", 8),
            num_latents=self.condition_num_latent,
        )

        self.emo_conditioning_encoder = ConformerEncoder(
            input_size=1024,
            output_size=emo_cond_cfg.get("output_size", 512),
            linear_units=emo_cond_cfg.get("linear_units", 1024),
            attention_heads=emo_cond_cfg.get("attention_heads", 4),
            num_blocks=emo_cond_cfg.get("num_blocks", 4),
            input_layer=emo_cond_cfg.get("input_layer", "conv2d2"),
        )
        self.emo_perceiver_encoder = PerceiverResampler(
            1024,
            dim_context=emo_cond_cfg.get("output_size", 512),
            ff_mult=emo_cond_cfg.get("perceiver_mult", 2),
            heads=emo_cond_cfg.get("attention_heads", 4),
            num_latents=1,
        )

        # Padding masks for perceiver cross-attention
        self.cond_mask_pad = nn.ConstantPad1d((self.condition_num_latent, 0), True)
        self.emo_cond_mask_pad = nn.ConstantPad1d((1, 0), True)

        # --- Lazy-loaded external models ---
        self._w2v_stat: torch.Tensor | None = None
        self._emo_matrix: torch.Tensor | None = None
        self._spk_matrix: torch.Tensor | None = None
        self._text_tokenizer: Any = None

        # Initialize embeddings per GPT-2 convention
        for emb in [self.text_embedding, self.mel_embedding]:
            emb.weight.data.normal_(mean=0.0, std=0.02)
        self.speed_emb.weight.data.normal_(mean=0.0, std=0.0)

    # ------------------------------------------------------------------
    # vLLM required hooks
    # ------------------------------------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.mel_embedding(input_ids)

    def _extract_mel_offsets(self, positions: torch.Tensor, kwargs: dict[str, Any]) -> torch.Tensor:
        """Build per-token mel_start_offset from model_intermediate_buffer."""
        info_dicts = kwargs.get("model_intermediate_buffer") or kwargs.get("runtime_additional_information") or []
        if "runtime_additional_information" in kwargs and "model_intermediate_buffer" not in kwargs:
            logger.warning_once("runtime_additional_information is deprecated, use model_intermediate_buffer")
        if not info_dicts:
            logger.warning_once("[OFFSET] no model_intermediate_buffer available; defaulting mel offsets to 0")
            return torch.zeros_like(positions)
        offsets = []
        for info in info_dicts:
            if isinstance(info, dict):
                meta = info.get("meta", {})
                val = meta.get("mel_start_offset", 0)
                offsets.append(int(val) if not isinstance(val, int) else val)
            else:
                offsets.append(0)
        return torch.tensor(offsets, dtype=positions.dtype, device=positions.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        """AR transformer forward.

        During prefill: inputs_embeds is set by preprocess(), input_ids ignored.
        During decode: inputs_embeds is mel_embedding(input_ids) + mel_pos.
        """
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                logger.warning("[FORWARD] inputs_embeds=None — fallback decode path")
                mel_start_offset = self._extract_mel_offsets(positions, kwargs)
                mel_pos = positions - mel_start_offset
                mel_pos = torch.clamp(mel_pos, min=0)
                inputs_embeds = self.mel_embedding(input_ids) + self.mel_pos_embedding.emb(mel_pos)
            hidden_states = inputs_embeds
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for layer in self.h[self.start_layer : self.end_layer]:
            hidden_states = layer(hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        return self.final_norm(self.ln_f(hidden_states))

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None or self.mel_head is None:
            return None
        return self.logits_processor(self.mel_head, hidden_states)

    # ------------------------------------------------------------------
    # Omni multimodal output plumbing
    # ------------------------------------------------------------------

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        """Collect mel_codes and hidden_states from intermediate buffer."""
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        hidden = model_outputs
        info_dicts = kwargs.get("model_intermediate_buffer")
        if info_dicts is None:
            info_dicts = kwargs.get("runtime_additional_information") or []
        if "runtime_additional_information" in kwargs and "model_intermediate_buffer" not in kwargs:
            logger.warning_once("runtime_additional_information is deprecated, use model_intermediate_buffer")

        device = hidden.device
        latent_dim = hidden.shape[-1]
        _zero_codes = torch.zeros(0, 1, dtype=torch.long, device=device)
        _zero_latent = torch.zeros(0, latent_dim, dtype=hidden.dtype, device=device)

        # Per-request lists: to_payload_element dispatches lists via
        # element[idx], giving correct per-request mapping even during
        # mixed prefill+decode steps. Zero-length placeholders for prefill
        # requests are harmless — torch.cat ignores them in accumulation.
        #
        # Export semantics are appendable deltas for the existing full-payload
        # accumulator: codes.mel is a 2-D row [1, 1], hidden_states.latent is
        # the aligned one-row [1, D] slice from meta.latent_acc.
        mel_codes_list: list[torch.Tensor] = []
        latent_list: list[torch.Tensor] = []
        meta_list: list[dict[str, Any]] = []

        for info in info_dicts:
            if not isinstance(info, dict):
                mel_codes_list.append(_zero_codes)
                latent_list.append(_zero_latent)
                meta_list.append({})
                continue
            codes = info.get("codes", {})
            mc = codes.get("mel")
            info_meta = info.get("meta", {})
            # mel_len now comes from the int counter in meta, not from
            # code_seq.shape[0] (codes.mel is a single-token delta).
            mel_len = int(info_meta.get("mel_code_count", 0))
            if isinstance(mc, torch.Tensor) and mc.numel() > 0 and mel_len > 0:
                code_seq = mc.reshape(-1)
                mel_codes_list.append(code_seq[-1:].reshape(1, 1).contiguous())
            else:
                mel_codes_list.append(_zero_codes)

            lat_acc = info_meta.get("latent_acc")
            if isinstance(lat_acc, torch.Tensor) and lat_acc.numel() > 0 and mel_len > 0:
                lat_seq = lat_acc.reshape(-1, lat_acc.shape[-1]) if lat_acc.ndim != 2 else lat_acc
                idx = min(max(mel_len - 1, 0), int(lat_seq.shape[0]) - 1)
                latent_list.append(lat_seq[idx : idx + 1].contiguous())
            else:
                hs = info.get("hidden_states", {})
                lat = hs.get("latent")
                if isinstance(lat, torch.Tensor) and lat.numel() > 0:
                    lat_seq = lat.reshape(-1, lat.shape[-1]) if lat.ndim != 2 else lat
                    latent_list.append(lat_seq[-1:].contiguous())
                else:
                    latent_list.append(_zero_latent)

            req_meta: dict[str, Any] = {}
            if mel_len == 1:
                s_ref = info_meta.get("S_ref")
                ref_mel = info_meta.get("ref_mel")
                style = info_meta.get("style")
                if isinstance(s_ref, torch.Tensor):
                    req_meta["S_ref"] = s_ref
                if isinstance(ref_mel, torch.Tensor):
                    req_meta["ref_mel"] = ref_mel
                if isinstance(style, torch.Tensor):
                    req_meta["style"] = style
            meta_list.append(req_meta)

        if not any(t.numel() > 0 for t in mel_codes_list):
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

        mm: OmniPayload = {
            "codes": {"mel": mel_codes_list},
            "hidden_states": {"latent": latent_list},
        }
        if any(meta_list):
            mm["meta"] = meta_list

        return OmniOutput(text_hidden_states=hidden, multimodal_outputs=mm)

    # ------------------------------------------------------------------
    # preprocess / postprocess
    # ------------------------------------------------------------------

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build prompt embeddings for prefill; compute mel embeddings for decode.

        Prefill layout::

            [conds(32) + emo_vec(1) + duration(2)] [text_emb + text_pos] [start_mel + mel_pos(0)]

        Decode: ``mel_embedding(token) + mel_pos_embedding(step)``.
        """
        span_len = int(input_ids.shape[0])
        meta = info_dict.get("meta", {})

        # --- Decode path: span_len==1 and mel_start_offset already set ---
        if span_len == 1 and isinstance(meta, dict) and meta.get("mel_start_offset") is not None:
            return self._preprocess_decode(input_ids, info_dict, meta)

        # --- Prefill path ---
        return self._preprocess_prefill(input_ids, input_embeds, info_dict, span_len)

    def _preprocess_decode(
        self,
        input_ids: torch.Tensor,
        info_dict: dict[str, Any],
        meta: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Lightweight decode step: embed token + track code count."""
        # Track code count via int counter instead of accumulating a full tensor
        # (the connector reconstructs the full sequence from per-step deltas).
        prev_count = meta.get("mel_code_count", 0)
        if not isinstance(prev_count, int):
            prev_count = int(prev_count)
        decode_step = prev_count + 1

        # Only keep the current token; no O(T) history accumulation.
        new_codes = input_ids.detach()

        mel_pos = input_ids.new_full((1,), decode_step)
        embeds = self.mel_embedding(input_ids) + self.mel_pos_embedding.emb(mel_pos)

        return (
            input_ids,
            embeds,
            {
                "codes": {"mel": new_codes},
                "meta": {"mel_code_count": decode_step},
            },
        )

    def _preprocess_prefill(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        info_dict: dict[str, Any],
        span_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Full conditioning pipeline — runs once during prefill."""
        additional_information = info_dict.get("additional_information")
        if isinstance(additional_information, dict):
            merged: dict[str, Any] = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in additional_information.items():
                merged.setdefault(k, v)
            info_dict = merged

        device = input_ids.device

        def _first(key, default=None):
            v = info_dict.get(key)
            return v[0] if isinstance(v, list) and v else default

        text = _first("text")
        if not text:
            raise ValueError("Missing additional_information.text for IndexTTS2 talker.")

        voice_path = _first("voice")
        if voice_path is None:
            raise ValueError("IndexTTS2 requires voice (reference audio path) in additional_information.")

        emo_audio_path = _first("emo_audio")
        emo_text = _first("emo_text")
        use_emo_text = bool(_first("use_emo_text", False))
        emo_vector = _first("emo_vector")
        emo_alpha = float(_first("emo_alpha", 1.0))
        use_random = bool(_first("use_random", False))
        _raw_emo_voice = _first("emo_voice_name")
        emo_voice_name = str(_raw_emo_voice).strip().lower() if _raw_emo_voice else None

        # --- Speaker cache lookup ---
        _voice_name = _first("voice_name")
        _voice_created_at = int(_first("voice_created_at", 0))
        _speaker_cache_key = None
        _cached = None
        if _voice_name:
            _speaker_cache_key = self._speaker_cache.make_cache_key(_voice_name, "indextts2", _voice_created_at)
            _cached = self._speaker_cache.get(_speaker_cache_key)

        # --- Load audio and extract features ---
        # Keep the speaker-cache hit path truly lazy: cached speaker artifacts are
        # sufficient for conditioning and same-speaker emotion reuse, so avoid
        # loading/resampling reference audio and touching the Wav2Vec2 singleton
        # unless a miss (or an uncached separate emotion audio) actually needs it.
        wav_16k: torch.Tensor | None = None
        w2v_model = None
        w2v_proc = None

        if _cached is not None:
            s_ref = _cached["S_ref"].to(device)
            style = _cached["style"].to(device)
            ref_mel = _cached["ref_mel"].to(device)
            spk_cond_emb = _cached["spk_cond_emb"].to(device)
        else:
            wav_16k, wav_22k = load_reference_audio(voice_path, device, mode="speaker")
            w2v_model, w2v_proc = load_wav2vec2(self.model_path, device)
            self._ensure_w2v_stat_loaded(device)
            spk_cond_emb = wav2vec_extract(wav_16k, w2v_model, w2v_proc, device, self._w2v_stat)

            semantic_codec = load_semantic_codec(self.model_path, self.config.semantic_codec, device)
            with torch.no_grad():
                _, s_ref = semantic_codec.quantize(spk_cond_emb)  # [B, T, 1024] quantized embeddings

            campplus = load_campplus(self.model_path, device)
            fbank = compute_fbank(wav_16k, device)
            with torch.no_grad():
                style = campplus(fbank)  # [1, 192]

            ref_mel = self._compute_mel_22k(wav_22k, device)  # [1, 80, T_ref]

            if _speaker_cache_key is not None:
                self._speaker_cache.put(
                    _speaker_cache_key,
                    {
                        "S_ref": s_ref.detach().cpu().contiguous(),
                        "style": style.detach().cpu().contiguous(),
                        "ref_mel": ref_mel.detach().cpu().contiguous(),
                        "spk_cond_emb": spk_cond_emb.detach().cpu().contiguous(),
                    },
                )

        model_dtype = next(self.conditioning_encoder.parameters()).dtype
        spk_cond_emb = spk_cond_emb.to(device=device, dtype=model_dtype)
        spk_lens = torch.tensor([spk_cond_emb.shape[1]], device=device, dtype=torch.long)
        speech_cond, mask = self.conditioning_encoder(spk_cond_emb, spk_lens)
        conds_mask = self.cond_mask_pad(mask.squeeze(1))
        conds = self.perceiver_encoder(speech_cond, conds_mask)  # [1, 32, D]

        emo_vec = self._compute_emotion_vector(
            wav_16k=wav_16k,
            speaker_audio_path=voice_path,
            emo_audio_path=emo_audio_path,
            emo_voice_name=emo_voice_name,
            main_text=text,
            use_emo_text=use_emo_text,
            emo_text=emo_text,
            emo_vector=emo_vector,
            emo_alpha=emo_alpha,
            use_random=use_random,
            style=style,
            spk_cond_emb=spk_cond_emb,
            w2v_model=w2v_model,
            w2v_proc=w2v_proc,
            device=device,
        )

        speed_zero = torch.zeros(1, device=device, dtype=torch.long)
        speed_one = torch.ones(1, device=device, dtype=torch.long)
        duration_emb = self.speed_emb(speed_zero)
        duration_emb_half = self.speed_emb(speed_one)

        conds_with_emo = conds + emo_vec.unsqueeze(1)  # [1, 32, D]
        conds_prefix = torch.cat(
            [conds_with_emo, duration_emb_half.unsqueeze(1), duration_emb.unsqueeze(1)],
            dim=1,
        )  # [1, 34, D]

        text_tokens = self._tokenize_text(text, device)  # [1, L]
        text_emb = self.text_embedding(text_tokens) + self.text_pos_embedding(text_tokens)

        start_mel = self.mel_embedding(torch.tensor([[self.start_mel_token]], device=device, dtype=torch.long))
        start_mel_pos = self.mel_pos_embedding.get_fixed_embedding(0, device)
        start_mel = start_mel + start_mel_pos  # [1, 1, D]

        inputs_embeds = torch.cat([conds_prefix, text_emb, start_mel], dim=1).squeeze(0)
        mel_start_offset = int(conds_prefix.shape[1]) + int(text_emb.shape[1])
        prefill_mel_index = int(inputs_embeds.shape[0]) - 1

        if int(inputs_embeds.shape[0]) < span_len:
            pad_n = span_len - int(inputs_embeds.shape[0])
            pad_emb = torch.zeros(pad_n, self.model_dim, device=device, dtype=inputs_embeds.dtype)
            inputs_embeds = torch.cat([inputs_embeds, pad_emb], dim=0)
        elif int(inputs_embeds.shape[0]) > span_len:
            inputs_embeds = inputs_embeds[:span_len]

        input_ids_out = torch.full_like(input_ids, self.start_mel_token)

        info_update: dict[str, Any] = {
            "meta": {
                "mel_start_offset": mel_start_offset,
                "prefill_mel_index": prefill_mel_index,
                "mel_code_count": 0,
                "S_ref": s_ref.cpu().contiguous(),
                "ref_mel": ref_mel.cpu().contiguous(),
                "style": style.cpu().contiguous(),
                "latent_acc": torch.zeros(0, self.model_dim, dtype=inputs_embeds.dtype, device=device),
            },
            "codes": {"mel": torch.zeros(0, dtype=torch.long, device=device)},
            "hidden_states": {"latent": torch.zeros(0, self.model_dim, dtype=inputs_embeds.dtype, device=device)},
        }

        return input_ids_out, inputs_embeds, info_update

    def postprocess(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Store current decode-step hidden state for Stage 1.

        Only the current row is kept; the connector's full-payload accumulator
        reconstructs the complete latent sequence from per-step deltas emitted
        by make_omni_output.
        """
        if hidden_states.shape[0] != 1:
            meta = kwargs.get("meta", {})
            prefill_mel_index = int(meta.get("prefill_mel_index", hidden_states.shape[0] - 1))
            prefill_mel_index = min(max(prefill_mel_index, 0), hidden_states.shape[0] - 1)
            return {"meta": {"latent_acc": hidden_states[prefill_mel_index : prefill_mel_index + 1].detach()}}

        return {"meta": {"latent_acc": hidden_states.detach()}}

    def _compute_mel_22k(self, wav_22k: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Compute 80-band mel spectrogram at 22.05kHz for Stage 1."""
        s2mel_cfg = self.config.s2mel.get("preprocess_params", {})
        sr = s2mel_cfg.get("sr", 22050)
        spect = s2mel_cfg.get("spect_params", {})
        n_fft = spect.get("n_fft", 1024)
        hop_length = spect.get("hop_length", 256)
        win_length = spect.get("win_length", 1024)
        n_mels = spect.get("n_mels", 80)

        # mel_spectrogram is pure torch with per-device caches — run on the
        # wav's device (GPU after load_reference_audio) instead of CPU.
        wav = wav_22k.float()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        from .s2mel.modules.audio import mel_spectrogram as mel_fn

        mel = mel_fn(wav, n_fft, n_mels, sr, hop_length, win_length, 0, None)
        return mel.to(device=device)  # [1, 80, T]

    # ------------------------------------------------------------------
    # Emotion conditioning
    # ------------------------------------------------------------------

    def _compute_emotion_vector(
        self,
        *,
        wav_16k: torch.Tensor | None,
        speaker_audio_path: Any,
        emo_audio_path: str | None,
        emo_voice_name: str | None,
        main_text: str,
        use_emo_text: bool,
        emo_text: str | None,
        emo_vector: Any,
        emo_alpha: float,
        use_random: bool,
        style: torch.Tensor,
        spk_cond_emb: torch.Tensor,
        w2v_model: Any,
        w2v_proc: Any,
        device: torch.device,
    ) -> torch.Tensor:
        """Full emotion control logic aligned with official infer_v2.py. Returns [1, D]."""
        # --- Mutual exclusion: text/vector guidance clears emotion reference audio ---
        # Mirrors official infer_v2.py: use_emo_text is evaluated before explicit
        # vectors, so text guidance wins if both are supplied by an internal caller.
        if use_emo_text:
            if emo_text is None:
                emo_text = main_text
            emo_vector = self._predict_emotion_from_text(emo_text, device)
        if use_emo_text or emo_vector is not None:
            emo_audio_path = None

        # --- Validate emo_vector (user-provided or text-predicted) ---
        if emo_vector is not None:
            if isinstance(emo_vector, torch.Tensor):
                emo_vector = emo_vector.tolist()
            elif isinstance(emo_vector, np.ndarray):
                emo_vector = emo_vector.tolist()
            if not isinstance(emo_vector, list) or len(emo_vector) != len(self._DESIRED_ORDER):
                raise ValueError(
                    "IndexTTS2 emo_vector must contain 8 values in order: "
                    "happy, angry, sad, afraid, disgusted, melancholic, surprised, calm"
                )
            if any(isinstance(v, bool) for v in emo_vector):
                raise ValueError("IndexTTS2 emo_vector values must be numeric")
            try:
                emo_vector = [float(v) for v in emo_vector]
            except (TypeError, ValueError) as exc:
                raise ValueError("IndexTTS2 emo_vector values must be numeric") from exc
            if not all(math.isfinite(v) for v in emo_vector):
                raise ValueError("IndexTTS2 emo_vector values must be finite")
            if not all(0.0 <= v <= 1.2 for v in emo_vector):
                raise ValueError("IndexTTS2 emo_vector values must be between 0.0 and 1.2")

            # Alpha pre-scaling (official infer_v2.py:414-418):
            # when emo_alpha != 1.0, scale emo_vector down before overlay.
            emo_vector_scale = max(0.0, min(1.0, emo_alpha))
            if emo_vector_scale != 1.0:
                emo_vector = [int(x * emo_vector_scale * 10000) / 10000 for x in emo_vector]

        # --- Path B: emo_vector (8-dim) → spk_matrix cosine lookup ---
        emovec_mat = None
        weight_sum = 0.0
        if emo_vector is not None:
            weight_sum = sum(emo_vector)
            vec = torch.tensor(emo_vector, device=device, dtype=torch.float32)
            emovec_mat = self._compute_emo_from_distribution(vec, style, device, use_random=use_random)

        # --- Audio-based emotion with alpha blending ---
        if emo_audio_path is None:
            # No separate emotion audio: reuse speaker w2v-bert features instead of
            # reloading + re-extracting the same audio (saves ~0.5s per prefill).
            # Official code resamples orig→16k directly for emotion vs orig→22k→16k
            # for speaker; the resulting w2v features are near-identical.
            effective_alpha = 1.0
            emo_cond_emb = spk_cond_emb
        else:
            effective_alpha = emo_alpha
            # emotion audio cache (enabled when emo_voice_name is provided)
            _emo_cache_key = None
            _emo_cached = None
            if emo_voice_name:
                _emo_cache_key = self._speaker_cache.make_cache_key(emo_voice_name, "indextts2_emo", 0)
                _emo_cached = self._speaker_cache.get(_emo_cache_key)

            if _emo_cached is not None:
                emo_cond_emb = _emo_cached["spk_cond_emb"].to(device)
                logger.debug("[EMO] cache HIT for %s", emo_voice_name)
            else:
                emo_wav_16k, _ = load_reference_audio(emo_audio_path, device, mode="emotion")
                if w2v_model is None or w2v_proc is None:
                    w2v_model, w2v_proc = load_wav2vec2(self.model_path, device)
                self._ensure_w2v_stat_loaded(device)
                emo_cond_emb = wav2vec_extract(emo_wav_16k, w2v_model, w2v_proc, device, self._w2v_stat)
                if _emo_cache_key is not None:
                    self._speaker_cache.put(
                        _emo_cache_key,
                        {
                            "spk_cond_emb": emo_cond_emb.detach().cpu().contiguous(),
                        },
                    )
                    logger.debug("[EMO] cache MISS, cached for %s", emo_voice_name)

        emovec = self._merge_emovec(spk_cond_emb, emo_cond_emb, effective_alpha)

        # --- emo_vector overlay (official infer_v2.py:560-561) ---
        # emovec_mat is raw feat2.pt weighted sum [1, model_dim], NOT projected.
        # Mix: emovec_mat contributes the emotion prototype; (1-sum)*emovec
        # preserves the audio-based emotion as a "base color".
        if emovec_mat is not None:
            emovec = emovec_mat.to(dtype=emovec.dtype) + (1.0 - weight_sum) * emovec

        return emovec

    _MELANCHOLIC_WORDS = frozenset(
        {
            "低落",
            "melancholy",
            "melancholic",
            "depression",
            "depressed",
            "gloomy",
        }
    )

    _DESIRED_ORDER = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]

    def _predict_emotion_from_text(self, text: str, device: torch.device) -> list[float] | None:
        """Use QwenEmotion CausalLM to predict 8-dim emotion vector from text (aligned with official)."""
        model, tokenizer = load_qwen_emotion(
            self.model_path,
            device,
            trust_remote_code=self.vllm_config.model_config.trust_remote_code,
        )
        if model is None or tokenizer is None:
            return None

        messages = [
            {"role": "system", "content": "文本情感分类"},
            {"role": "user", "content": text},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
            )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        content_str = tokenizer.decode(output_ids[index:], skip_special_tokens=True)

        try:
            content = json.loads(content_str)
        except json.JSONDecodeError:
            content = {
                m.group(1): float(m.group(2)) for m in re.finditer(r'([^\s":.,]+?)"?\s*:\s*([\d.]+)', content_str)
            }

        text_lower = text.lower()
        if any(w in text_lower for w in self._MELANCHOLIC_WORDS):
            content["悲伤"], content["低落"] = content.get("低落", 0.0), content.get("悲伤", 0.0)

        max_score, min_score = 1.2, 0.0
        emo_vector = []
        for cn_key in self._DESIRED_ORDER:
            val = content.get(cn_key, 0.0)
            if isinstance(val, str):
                try:
                    val = float(val)
                except ValueError:
                    val = 0.0
            emo_vector.append(max(min_score, min(max_score, val)))

        if all(v <= 0.0 for v in emo_vector):
            emo_vector[7] = 1.0  # default calm

        return emo_vector

    def _compute_emo_from_distribution(
        self,
        emo_dist: torch.Tensor,
        style: torch.Tensor,
        device: torch.device,
        use_random: bool = False,
    ) -> torch.Tensor:
        """From 8-dim emotion distribution + CAMPPlus style, build emovec_mat via spk_matrix cosine lookup.

        Returns [1, model_dim] (feat2.pt rows are already model_dim).
        """
        self._ensure_matrix_loaded("_emo_matrix", "emo_matrix", "Emotion matrix")
        self._ensure_matrix_loaded("_spk_matrix", "spk_matrix", "Speaker matrix")
        if self._emo_matrix is None or self._spk_matrix is None:
            return torch.zeros(1, self.model_dim, device=device)

        weight_vector = emo_dist.to(device=device, dtype=torch.float32)
        if weight_vector.ndim == 2:
            weight_vector = weight_vector.squeeze(0)  # [8]

        if use_random:
            indices = [int(torch.randint(0, sub.shape[0], (1,), device=device).item()) for sub in self._spk_matrix]
        else:
            indices = [_find_most_similar_cosine(style, sub.to(device)) for sub in self._spk_matrix]
        emo_rows = torch.cat(
            [sub.to(device)[idx].unsqueeze(0) for idx, sub in zip(indices, self._emo_matrix)], dim=0
        )  # [8, model_dim]

        return (weight_vector.unsqueeze(1) * emo_rows).sum(0).unsqueeze(0)  # [1, model_dim]

    def _compute_audio_emo_vec(self, cond_emb: torch.Tensor) -> torch.Tensor:
        """Conformer → Perceiver → emovec_layer → emo_layer. Input: [1, T, 1024], output: [1, D]."""
        emo_dtype = next(self.emo_conditioning_encoder.parameters()).dtype
        cond_emb = cond_emb.to(device=cond_emb.device, dtype=emo_dtype)
        cond_lens = torch.tensor([cond_emb.shape[1]], device=cond_emb.device, dtype=torch.long)
        emo_cond, emo_mask = self.emo_conditioning_encoder(cond_emb, cond_lens)
        emo_conds_mask = self.emo_cond_mask_pad(emo_mask.squeeze(1))
        emo_percept = self.emo_perceiver_encoder(emo_cond, emo_conds_mask)  # [1, 1, 1024]
        emo_vec_syn = self.emovec_layer(emo_percept.squeeze(1))  # [1, D]
        return self.emo_layer(emo_vec_syn)  # [1, D]

    def _merge_emovec(
        self,
        spk_cond_emb: torch.Tensor,
        emo_cond_emb: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """Alpha-blend speaker and emotion audio vectors: base + alpha * (emo - base)."""
        if emo_cond_emb is spk_cond_emb or alpha == 1.0:
            return self._compute_audio_emo_vec(emo_cond_emb)
        if alpha == 0.0:
            return self._compute_audio_emo_vec(spk_cond_emb)
        base_vec = self._compute_audio_emo_vec(spk_cond_emb)
        emo_vec = self._compute_audio_emo_vec(emo_cond_emb)
        return base_vec + alpha * (emo_vec - base_vec)  # [1, D]

    # ------------------------------------------------------------------
    # Text tokenizer
    # ------------------------------------------------------------------

    def _tokenize_text(self, text: str, device: torch.device) -> torch.Tensor:
        """Tokenize text using the BPE tokenizer, add start/stop tokens."""
        if self._text_tokenizer is None:
            from .tokenizer import IndexTTS2Tokenizer

            bpe_path = resolve_model_file(self.model_path, "bpe.model")
            if bpe_path is None:
                raise FileNotFoundError(f"BPE model not found in {self.model_path}")
            self._text_tokenizer = IndexTTS2Tokenizer(bpe_path, model_dir=self.model_path)

        token_ids = self._text_tokenizer.encode(text, add_special_tokens=False)

        # Wrap with start/stop text tokens
        start_text = 0
        stop_text = 1
        token_ids = [start_text] + token_ids + [stop_text]
        return torch.tensor([token_ids], device=device, dtype=torch.long)

    # ------------------------------------------------------------------
    # Lazy loading helpers
    # ------------------------------------------------------------------

    def _ensure_w2v_stat_loaded(self, device: torch.device) -> None:
        if self._w2v_stat is not None:
            return
        stat_path = resolve_model_file(self.model_path, self.config.w2v_stat)
        if stat_path is not None:
            raw = torch.load(stat_path, map_location="cpu", weights_only=True)
            if isinstance(raw, dict):
                self._w2v_stat = torch.stack([raw["mean"], raw["var"]])  # [2, 1024]
            else:
                self._w2v_stat = raw
        else:
            logger.warning("Wav2Vec2-BERT stat file not found in %s", self.model_path)

    def _ensure_matrix_loaded(self, attr: str, cfg_key: str, label: str) -> None:
        if getattr(self, attr) is not None:
            return
        path = resolve_model_file(self.model_path, getattr(self.config, cfg_key))
        if path is not None:
            raw = torch.load(path, map_location="cpu", weights_only=True)
            setattr(self, attr, torch.split(raw, list(self.config.emo_num)))
        else:
            logger.warning("%s not found in %s", label, self.model_path)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from IndexTTS2 checkpoint (gpt.pth).

        Bypasses vllm's default .pt iterator because the model directory
        contains raw-tensor files (feat1.pt, feat2.pt) that are not state
        dicts and would crash ``pt_weights_iterator``.

        Weight mapping from gpt.pth checkpoint → vLLM model params:
          gpt.h.{i}.*                → h.{i}.*  (strip ``gpt.`` prefix)
          gpt.ln_f.*                 → ln_f.*
          final_norm.*               → final_norm.*
          text_head.*                → (skipped)
          <everything else>          → identity
        """
        _ = weights  # don't iterate — raw .pt files crash the default iterator

        ckpt_path = resolve_model_file(self.model_path, self.config.gpt_checkpoint)
        if ckpt_path is None:
            raise FileNotFoundError(f"IndexTTS2 GPT checkpoint {self.config.gpt_checkpoint!r} not found")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        prefix_map = {
            "gpt.h.": "h.",
            "gpt.ln_f.": "ln_f.",
            "final_norm.": "final_norm.",
            "text_head.": "_skip_text_head.",
        }

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in state.items():
            # Skip attention masks
            if ".attn.bias" in name or ".attn.masked_bias" in name:
                continue
            # Skip wpe (null position embeddings)
            if ".wpe." in name or ".wte." in name:
                continue

            # Remap checkpoint name to model name
            mapped_name = name
            for old_prefix, new_prefix in prefix_map.items():
                if name.startswith(old_prefix):
                    mapped_name = new_prefix + name[len(old_prefix) :]
                    break

            # Skip unmapped / intentionally skipped weights
            if mapped_name.startswith("_skip_"):
                continue

            if is_pp_missing_parameter(mapped_name, self):
                continue

            if mapped_name not in params_dict:
                logger.debug("Skipping unrecognized weight: %s → %s", name, mapped_name)
                continue

            param = params_dict[mapped_name]

            # GPT-2 Conv1D → Linear transpose
            for conv1d_name in ["c_attn", "c_proj", "c_fc"]:
                if conv1d_name not in mapped_name:
                    continue
                if not mapped_name.endswith(".weight"):
                    continue
                loaded_weight = loaded_weight.t()

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(mapped_name)

        logger.info(
            "Loaded %d weights for IndexTTS2TalkerForConditionalGeneration",
            len(loaded_params),
        )
        required_prefixes = [
            "text_embedding.",
            "mel_embedding.",
            "text_pos_embedding.",
            "mel_pos_embedding.",
            "speed_emb.",
            "emo_layer.",
            "emovec_layer.",
            "conditioning_encoder.",
            "perceiver_encoder.",
            "emo_conditioning_encoder.",
            "emo_perceiver_encoder.",
            "h.",
            "ln_f.",
            "final_norm.",
        ]
        if self.mel_head is not None:
            required_prefixes.append("mel_head.")
        missing_prefixes = [
            prefix for prefix in required_prefixes if not any(name.startswith(prefix) for name in loaded_params)
        ]
        if missing_prefixes:
            raise RuntimeError(
                "IndexTTS2 GPT checkpoint did not load required parameter groups: " + ", ".join(missing_prefixes)
            )

        # Ensure all sub-modules on correct device (some nn.Parameter created
        # with torch.Tensor() end up on CPU even when model is on CUDA).
        target_device = next((p.device for p in self.h.parameters()), torch.device("cpu"))
        for mod in [
            self.conditioning_encoder,
            self.emo_conditioning_encoder,
            self.perceiver_encoder,
            self.emo_perceiver_encoder,
        ]:
            mod.to(target_device)

        self._warmup_external_models(target_device)
        return loaded_params

    def _warmup_external_models(self, device: torch.device) -> None:
        """Preload external models to eliminate first-request latency."""
        try:
            load_wav2vec2(self.model_path, device)
            load_semantic_codec(self.model_path, self.config.semantic_codec, device)
            load_campplus(self.model_path, device)
            self._ensure_w2v_stat_loaded(device)
            logger.info("External models preloaded: Wav2Vec2, RepCodec, CAMPPlus")
        except Exception as e:
            logger.warning("Failed to preload external models: %s", e)
