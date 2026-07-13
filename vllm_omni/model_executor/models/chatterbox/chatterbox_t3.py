"""Chatterbox T3 AR models for vLLM-Omni.

T3 is the autoregressive text-to-speech-token stage of Chatterbox.

Two variants:
* ``ChatterboxTurboT3ForGeneration`` — GPT-2-medium backbone (350M)
* ``ChatterboxT3ForGeneration`` — LLaMA-520M backbone with perceiver
  (AR-stage CFG is a tracked follow-up; it is NOT implemented here)

The vLLM integration follows the same pattern as Qwen3-TTS Talker:
* ``preprocess`` builds full prompt embeddings from ``additional_information``
  (text, reference audio path, optional exaggeration).
* ``forward`` runs the backbone transformer.
* ``postprocess`` caches last hidden state for the next decode step.
* ``make_omni_output`` wraps generated speech tokens for the stage connector.

Weight loading maps from Chatterbox safetensors keys (``tfmr.*``,
``text_emb.*``, ``speech_emb.*``, ``speech_head.*``, ``cond_enc.*``).

Fixes over PR #1517:
- Prefill no longer emits fake zero speech_tokens (corrupted downstream)
- Reference audio conditions T3 locally (VoiceEncoder + cond prompt tokens);
  the stage input processor forwards the reference path to S3Gen separately
"""

from __future__ import annotations

import glob
import os
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPT2Config, LlamaConfig
from transformers.utils.hub import cached_file
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.gpt2 import GPT2Model
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.utils.audio import load_wav_mono

from .configuration_chatterbox import ChatterboxConfig, ChatterboxTurboConfig

logger = init_logger(__name__)

# Constants from Chatterbox
S3_SR = 16000  # S3Tokenizer input sample rate
ENC_COND_LEN = 15 * S3_SR  # 15 seconds conditioning for VoiceEncoder


def _build_speech_allowed_mask(config: Any) -> torch.Tensor:
    """Build the valid-token mask for the T3 speech head.

    Allows the real codec range ``[0, start_speech_token)`` plus
    ``stop_speech_token`` (so the model can terminate), and masks everything
    else — ``start_speech_token`` itself (re-emitting SOS mid-generation
    corrupts the decoder) and any padding slots above EOS.

    This matters for Original (``speech_vocab_size=8194``): the real S3
    codec only spans ``[0, 6561)`` plus SOS/EOS, so the padding slots
    ``[stop_speech_token + 1, speech_vocab_size)`` must be masked. Otherwise
    the sampler can emit ids that S3Gen silently drops (``>= 6561``),
    producing dropped frames / audio gaps. Turbo (vocab=6563) has no padding
    above EOS, so the result is unchanged for it.
    """
    mask = torch.zeros((config.speech_vocab_size,), dtype=torch.bool)
    mask[: config.start_speech_token] = True  # codec tokens [0, SOS)
    mask[config.stop_speech_token] = True  # EOS, so generation can terminate
    return mask


# ---------------------------------------------------------------------------
# Shared conditioning encoder
# ---------------------------------------------------------------------------


class ChatterboxT3CondEnc(nn.Module):
    """Conditioning encoder: project speaker embedding + cond prompt speech."""

    def __init__(self, speaker_embed_size: int, hidden_size: int):
        super().__init__()
        self.speaker_proj = nn.Linear(speaker_embed_size, hidden_size)

    def forward(
        self,
        speaker_emb: torch.Tensor,
        cond_prompt_speech_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return conditioning sequence: [speaker_proj || cond_speech_emb]."""
        spk = self.speaker_proj(speaker_emb)  # (1, hidden_size)
        if spk.ndim == 2:
            spk = spk.unsqueeze(1)  # (1, 1, hidden_size)
        if cond_prompt_speech_emb is not None:
            return torch.cat([spk, cond_prompt_speech_emb], dim=1)
        return spk


# ---------------------------------------------------------------------------
# Shared mixin for preprocess/postprocess/make_omni_output logic
# ---------------------------------------------------------------------------


class _ChatterboxT3Base(nn.Module):
    """Base class with shared preprocess/postprocess/omni logic for both variants."""

    have_multimodal_outputs = True
    has_preprocess = True
    has_postprocess = True

    # Subclasses must set these:
    config: ChatterboxTurboConfig | ChatterboxConfig
    model_path: str
    text_emb: nn.Embedding
    speech_emb: nn.Embedding
    cond_enc: ChatterboxT3CondEnc

    _tokenizer: Any
    _voice_encoder: Any
    _s3_tokenizer: Any
    _default_t3_cond: dict[str, Any] | None

    # -------------------- vLLM required hooks --------------------

    def compute_logits(
        self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        bias = getattr(self.speech_head, "bias", None)
        logits = self.logits_processor(self.speech_head, hidden_states, embedding_bias=bias)
        if logits is None:
            return None
        logits = logits.masked_fill(~self._speech_allowed_mask, float("-inf"))
        return logits

    # -------------------- Omni multimodal output plumbing --------------------

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        hidden = model_outputs
        info_dicts = kwargs.get("runtime_additional_information") or []

        speech_tokens_list: list[torch.Tensor] = []
        for info in info_dicts:
            if not isinstance(info, dict):
                continue
            st = info.get("speech_tokens")
            if isinstance(st, torch.Tensor):
                speech_tokens_list.append(st)

        if not speech_tokens_list:
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

        speech_tokens = torch.cat(speech_tokens_list, dim=0)
        span_len = int(speech_tokens.shape[0])
        hidden = hidden[:span_len]
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={"speech_tokens": speech_tokens},
        )

    # -------------------- preprocess / postprocess --------------------

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        additional_information = info_dict.get("additional_information")
        if isinstance(additional_information, dict):
            merged: dict[str, Any] = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in additional_information.items():
                merged.setdefault(k, v)
            info_dict = merged

        span_len = int(input_ids.shape[0])
        device = input_ids.device
        if span_len <= 0:
            return input_ids, input_embeds if input_embeds is not None else self.embed_input_ids(input_ids), {}

        text_list = info_dict.get("text")
        if not isinstance(text_list, list) or not text_list or not text_list[0]:
            raise ValueError("Missing additional_information.text for Chatterbox T3.")

        if span_len > 1:
            # Prefill
            prompt_embeds_cpu = info_dict.get("t3_prompt_embeds")
            is_first_prefill = not isinstance(prompt_embeds_cpu, torch.Tensor) or prompt_embeds_cpu.ndim != 2

            if is_first_prefill:
                prompt_embeds_full = self._build_prompt_embeds(info_dict, device)
                prompt_embeds_cpu = prompt_embeds_full.detach().to("cpu").contiguous()

                info_update: dict[str, Any] = {
                    "t3_prompt_embeds": prompt_embeds_cpu,
                    "t3_prefill_offset": 0,
                }

                # Propagate any keys _build_prompt_embeds seeded into info_dict
                # (e.g. ``t3_speech_pos`` for the Original variant's additive
                # speech position counter). Without forwarding them here, the
                # mutations on the local info_dict are lost and the decode hot
                # path starts at position 0 every step.
                for _seed_key in ("t3_speech_pos",):
                    if _seed_key in info_dict:
                        info_update[_seed_key] = info_dict[_seed_key]

                take = prompt_embeds_cpu[:span_len]
                if int(take.shape[0]) < span_len:
                    pad_n = span_len - int(take.shape[0])
                    pad_rows = torch.zeros(pad_n, take.shape[-1])
                    take = torch.cat([take, pad_rows], dim=0)
                prompt_embeds = take.to(device=device, dtype=torch.bfloat16)
                info_update["t3_prefill_offset"] = span_len
            else:
                offset = int(info_dict.get("t3_prefill_offset", 0) or 0)
                s = max(0, min(offset, int(prompt_embeds_cpu.shape[0])))
                e = max(0, min(offset + span_len, int(prompt_embeds_cpu.shape[0])))
                take = prompt_embeds_cpu[s:e]
                if int(take.shape[0]) < span_len:
                    pad_n = span_len - int(take.shape[0])
                    pad_rows = torch.zeros(pad_n, take.shape[-1])
                    take = torch.cat([take, pad_rows], dim=0)
                prompt_embeds = take.to(device=device, dtype=torch.bfloat16)
                info_update = {"t3_prefill_offset": offset + span_len}

            # Dummy input_ids (in-vocab for vLLM bookkeeping).
            input_ids_out = torch.zeros_like(input_ids)
            # Emit sentinel speech_tokens spanning the full prefill length so
            # make_omni_output keeps hidden_states sized to the full scheduled
            # batch.  Without this, a mixed prefill+decode batch truncates
            # hidden_states to only the decode rows, and sampling (which
            # gathers by logits_indices computed over the full batch) goes
            # out-of-bounds — CUDA device-side assert.
            #
            # We use the SOS token (start_speech_token) which is >=
            # SPEECH_VOCAB_SIZE (6561) and therefore filtered out by the
            # chatterbox stage input processor before reaching S3Gen.
            sentinel = int(self.config.start_speech_token)
            info_update["speech_tokens"] = torch.full((span_len,), sentinel, dtype=torch.long, device=device)
            return input_ids_out, prompt_embeds, info_update

        # Decode: span_len == 1
        last_hidden_cpu = info_dict.get("last_t3_hidden")
        if not isinstance(last_hidden_cpu, torch.Tensor):
            raise RuntimeError("Missing `last_t3_hidden` in additional_information; postprocess must run first.")

        last_token_embed = self.speech_emb(input_ids.clamp(0, self.config.speech_vocab_size - 1).long())
        # Original variant: the learned speech position embedding is additive
        # on top of the token embedding (native T3.prepare_input_embeds). The
        # position counter is seeded to 1 at prefill time (start token took 0).
        # Turbo does not declare a ``speech_pos_emb`` attribute and skips this.
        speech_pos = int(info_dict.get("t3_speech_pos", 0) or 0)
        speech_pos_emb = getattr(self, "speech_pos_emb", None)
        if speech_pos_emb is not None:
            last_token_embed = last_token_embed + speech_pos_emb.get_fixed_embedding(speech_pos).reshape(
                last_token_embed.shape
            )
        inputs_embeds_out = last_token_embed.reshape(1, -1)

        info_update = {
            "speech_tokens": input_ids.reshape(-1).to(torch.long),
            "t3_speech_pos": speech_pos + 1,
        }
        return input_ids, inputs_embeds_out, info_update

    def postprocess(self, hidden_states: torch.Tensor, **_: Any) -> dict[str, Any]:
        if hidden_states.numel() == 0:
            return {}
        last = hidden_states[-1, :].detach().to("cpu").contiguous()
        return {"last_t3_hidden": last}

    # -------------------- Prompt construction --------------------

    def _build_prompt_embeds(self, info_dict: dict[str, Any], device: torch.device) -> torch.Tensor:
        """Build the full prompt embedding sequence for T3 prefill.

        Sequence layout: [cond_enc_output || text_emb || start_speech_token_emb]

        Where cond_enc_output = [speaker_proj(speaker_emb) || speech_emb(cond_prompt)]
        """
        text = info_dict["text"][0]
        ref_audio_path = None
        ref_audio_list = info_dict.get("ref_audio")
        if isinstance(ref_audio_list, list) and ref_audio_list:
            ref_audio_path = ref_audio_list[0]

        text_token_ids = self._tokenize_text(text, device)
        text_embedded = self.text_emb(text_token_ids).unsqueeze(0)  # (1, T, H)

        speaker_emb = self._get_speaker_embedding(ref_audio_path, device)  # (1, 256)

        cond_speech_emb = self._get_cond_prompt_speech_emb(ref_audio_path, device)  # (1, plen, H) or None

        # Cast to model dtype — VoiceEncoder returns float32, model is bfloat16.
        model_dtype = self.cond_enc.speaker_proj.weight.dtype
        speaker_emb = speaker_emb.to(dtype=model_dtype)
        if cond_speech_emb is not None:
            cond_speech_emb = cond_speech_emb.to(dtype=model_dtype)

        cond_output = self.cond_enc(speaker_emb, cond_speech_emb)  # (1, 1+plen, H)

        start_token = torch.tensor([self.config.start_speech_token], dtype=torch.long, device=device)
        start_emb = self.speech_emb(start_token).unsqueeze(0)  # (1, 1, H)

        # Concatenate: [cond || text || start_speech]
        prompt_embeds = torch.cat([cond_output, text_embedded, start_emb], dim=1)  # (1, L, H)

        return prompt_embeds.squeeze(0)  # (L, H)

    def _tokenize_text(self, text: str, device: torch.device) -> torch.Tensor:
        """Tokenize text — subclasses may override for different tokenizers."""
        tokenizer = self._get_tokenizer()
        text_tokens = tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor(text_tokens, dtype=torch.long, device=device)

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
        return self._tokenizer

    def _load_default_t3_cond(self, device: torch.device) -> dict[str, Any]:
        """Load default T3 conditioning from conds.pt shipped with the model.

        conds.pt stores ``{t3: {speaker_emb, cond_prompt_speech_tokens, ...}, gen: {...}}``.
        T3 always requires conditioning — when no ref audio is provided, we use
        the builtin default voice stored in conds.pt.
        """
        if self._default_t3_cond is not None:
            return self._default_t3_cond

        conds_path = cached_file(self.model_path, "conds.pt")
        if conds_path is None:
            raise FileNotFoundError(
                "conds.pt not found in model checkpoint. Chatterbox T3 always requires "
                "speaker conditioning. Either supply --ref-audio or ensure conds.pt is "
                "present in the model directory."
            )
        conds = torch.load(conds_path, map_location=device, weights_only=True)
        t3_cond = conds.get("t3", {})
        # Move all tensors to the correct device.
        t3_cond = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t3_cond.items()}
        self._default_t3_cond = t3_cond
        logger.info("Loaded default T3 conditioning from conds.pt")
        return self._default_t3_cond

    def _get_speaker_embedding(self, ref_audio_path: str | None, device: torch.device) -> torch.Tensor:
        """Extract speaker embedding from reference audio using VoiceEncoder.

        Falls back to the default speaker embedding from conds.pt when no
        reference audio is provided.
        """
        if ref_audio_path is None:
            default_cond = self._load_default_t3_cond(device)
            spk = default_cond.get("speaker_emb")
            if spk is not None:
                return spk.to(device).float().view(1, -1)
            logger.warning("No speaker_emb in conds.pt; using zeros")
            return torch.zeros(1, self.config.speaker_embed_size, device=device)

        voice_encoder = self._ensure_voice_encoder(device)

        wav_np = load_wav_mono(ref_audio_path, target_sr=S3_SR)  # mono float32 at 16 kHz
        if wav_np.shape[-1] > ENC_COND_LEN:
            wav_np = wav_np[:ENC_COND_LEN]

        with torch.no_grad():
            emb_np = voice_encoder.embeds_from_wavs([wav_np], sample_rate=S3_SR)
        emb = torch.from_numpy(emb_np).float().mean(dim=0, keepdim=True).to(device)  # (1, 256)
        return emb

    def _get_cond_prompt_speech_emb(self, ref_audio_path: str | None, device: torch.device) -> torch.Tensor | None:
        """Tokenize reference audio and embed as conditioning prompt.

        Falls back to the default cond_prompt_speech_tokens from conds.pt
        when no reference audio is provided.
        """
        if ref_audio_path is None:
            default_cond = self._load_default_t3_cond(device)
            tokens = default_cond.get("cond_prompt_speech_tokens")
            if tokens is not None:
                tokens = torch.atleast_2d(tokens).to(device)
                plen = self.config.speech_cond_prompt_len
                if tokens.shape[-1] > plen:
                    tokens = tokens[:, :plen]
                cond_emb = self.speech_emb(tokens.clamp(0, self.config.speech_vocab_size - 1))
                logger.info("Using default speech conditioning: %d tokens", tokens.shape[-1])
                return cond_emb
            logger.warning("No cond_prompt_speech_tokens in conds.pt; no speech conditioning")
            return None

        s3_tokenizer = self._ensure_s3_tokenizer(device)

        wav_np = load_wav_mono(ref_audio_path, target_sr=S3_SR)  # mono float32 at 16 kHz
        if wav_np.shape[-1] > ENC_COND_LEN:
            wav_np = wav_np[:ENC_COND_LEN]

        plen = self.config.speech_cond_prompt_len
        with torch.no_grad():
            tokens, _ = s3_tokenizer([wav_np], max_len=plen)  # (1, plen)

        tokens = torch.atleast_2d(tokens).to(device)
        if tokens.shape[-1] > plen:
            tokens = tokens[:, :plen]

        cond_emb = self.speech_emb(tokens.clamp(0, self.config.speech_vocab_size - 1))  # (1, plen, H)
        return cond_emb

    def _ensure_voice_encoder(self, device: torch.device):
        """Lazy-load the VoiceEncoder (LSTM) for speaker embedding extraction.

        Upstream loads as: ``VoiceEncoder()`` + ``load_state_dict(load_file("ve.safetensors"))``.
        """
        if self._voice_encoder is not None:
            return self._voice_encoder

        try:
            from chatterbox.models.voice_encoder import VoiceEncoder
        except ImportError:
            raise ImportError(
                "chatterbox-tts package is required for Chatterbox TTS. Install it with: pip install chatterbox-tts"
            )

        from safetensors.torch import load_file

        ve_path = cached_file(self.model_path, "ve.safetensors")
        if ve_path is None:
            raise FileNotFoundError("ve.safetensors not found in model checkpoint")
        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ve_path))
        ve.to(device).eval()
        self._voice_encoder = ve
        return self._voice_encoder

    def _ensure_s3_tokenizer(self, device: torch.device):
        """Lazy-load S3Tokenizer for reference audio tokenization.

        S3Tokenizer downloads its own model via ``S3Tokenizer("speech_tokenizer_v2_25hz")``.
        No local checkpoint needed — it's built into the s3tokenizer package.
        """
        if self._s3_tokenizer is not None:
            return self._s3_tokenizer

        try:
            from chatterbox.models.s3tokenizer import S3Tokenizer
        except ImportError:
            raise ImportError(
                "chatterbox-tts package is required for Chatterbox TTS. Install it with: pip install chatterbox-tts"
            )

        self._s3_tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz")
        self._s3_tokenizer.to(device).eval()
        return self._s3_tokenizer


# ---------------------------------------------------------------------------
# Turbo variant (GPT-2 backbone)
# ---------------------------------------------------------------------------


class ChatterboxTurboT3ForGeneration(_ChatterboxT3Base):
    """Chatterbox Turbo T3 — GPT-2-medium AR speech token generator.

    Stage 0 of the Chatterbox Turbo TTS pipeline.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        config = vllm_config.model_config.hf_config
        if not isinstance(config, ChatterboxTurboConfig):
            config = ChatterboxTurboConfig()

        self.config = config
        hidden_size = config.hidden_size

        text_vocab = getattr(config, "text_vocab_size", 50276)
        gpt2_config = GPT2Config(
            vocab_size=text_vocab,
            n_embd=hidden_size,
            n_layer=config.num_hidden_layers,
            n_head=config.num_attention_heads,
            n_inner=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            add_cross_attention=False,
            scale_attn_by_inverse_layer_idx=False,
            reorder_and_upcast_attn=False,
        )
        # Patch hf_config so GPT2Model reads the right config.
        orig_hf_config = vllm_config.model_config.hf_config
        vllm_config.model_config.hf_config = gpt2_config
        self.model = GPT2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "tfmr"),
        )
        vllm_config.model_config.hf_config = orig_hf_config

        # Custom embeddings (separate from GPT-2 wte which we won't use).
        self.text_emb = nn.Embedding(text_vocab, hidden_size)
        self.speech_emb = nn.Embedding(config.speech_vocab_size, hidden_size)

        # GPT-2 variant has bias on speech_head (upstream: bias=self.is_gpt).
        self.speech_head = ParallelLMHead(
            config.speech_vocab_size,
            hidden_size,
            bias=True,
        )
        self.logits_processor = LogitsProcessor(config.speech_vocab_size)

        self.cond_enc = ChatterboxT3CondEnc(config.speaker_embed_size, hidden_size)

        # Valid token mask: allow every codec slot + EOS, mask out SOS
        # (re-emitting SOS mid-generation would break the decoder).  Derived
        # from config tokens so it generalises across Turbo and Original.
        self.register_buffer(
            "_speech_allowed_mask",
            _build_speech_allowed_mask(config),
            persistent=False,
        )

        # Lazy loaded.
        self._tokenizer = None
        self._voice_encoder = None
        self._s3_tokenizer = None
        self._default_t3_cond = None

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    # -------------------- Weight loading --------------------

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "tfmr.": "model.",
            "text_emb.": "text_emb.",
            "speech_emb.": "speech_emb.",
            "speech_head.": "speech_head.",
            "cond_enc.spkr_enc.": "cond_enc.speaker_proj.",
        }
    )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if ".attn.bias" in name or ".attn.masked_bias" in name:
                continue

            mapped_name = name
            for old_prefix, new_prefix in self.hf_to_vllm_mapper.orig_to_new_prefix.items():
                if name.startswith(old_prefix):
                    mapped_name = new_prefix + name[len(old_prefix) :]
                    break

            if mapped_name not in params_dict:
                continue

            param = params_dict[mapped_name]
            # GPT-2 Conv1D → Linear transpose.
            for conv1d_name in ["c_attn", "c_proj", "c_fc"]:
                if conv1d_name in mapped_name and mapped_name.endswith(".weight"):
                    loaded_weight = loaded_weight.t()
                    break

            weight_loader = getattr(param, "weight_loader", None)
            if weight_loader is not None:
                weight_loader(param, loaded_weight)
            else:
                param.data.copy_(loaded_weight)
            loaded_params.add(mapped_name)

        return loaded_params


# ---------------------------------------------------------------------------
# Original variant (LLaMA backbone)
# ---------------------------------------------------------------------------


class ChatterboxT3ForGeneration(_ChatterboxT3Base):
    """Chatterbox Original T3 — LLaMA-520M AR speech token generator.

    Stage 0 of the Chatterbox Original TTS pipeline.
    Supports exaggeration control. AR-stage classifier-free guidance (CFG),
    which native Chatterbox Original runs by default, is NOT yet implemented
    here and is tracked as a follow-up; without it Original output is muffled
    relative to native.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        config = vllm_config.model_config.hf_config
        if not isinstance(config, ChatterboxConfig):
            config = ChatterboxConfig()

        self.config = config
        hidden_size = config.hidden_size

        llama_config = LlamaConfig(
            vocab_size=8,  # Unused — we use custom embeddings
            hidden_size=hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            hidden_act="silu",
            attention_bias=False,
            mlp_bias=False,
            tie_word_embeddings=False,
        )
        # Patch hf_config so LlamaModel reads the right config.
        orig_hf_config = vllm_config.model_config.hf_config
        vllm_config.model_config.hf_config = llama_config
        self.model = LlamaModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "tfmr"),
        )
        vllm_config.model_config.hf_config = orig_hf_config

        # Custom embeddings.  ``text_vocab_size`` is the EnTokenizer vocab
        # (704), distinct from ``vocab_size`` which must match the speech head
        # for vLLM sampler compatibility.
        text_vocab = getattr(config, "text_vocab_size", 704)
        self.text_emb = nn.Embedding(text_vocab, hidden_size)
        self.speech_emb = nn.Embedding(config.speech_vocab_size, hidden_size)

        # Learned position embeddings (Original variant uses these)
        self.text_pos_emb = None
        self.speech_pos_emb = None
        if getattr(config, "input_pos_emb", None) == "learned":
            try:
                from chatterbox.models.t3.modules.learned_pos_emb import LearnedPositionEmbeddings
            except ImportError:
                raise ImportError(
                    "chatterbox-tts package is required for Chatterbox Original variant. "
                    "Install it with: pip install chatterbox-tts"
                )
            max_text_seq_len = config.max_text_tokens + 2
            self.text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, hidden_size)
            max_speech_seq_len = config.max_speech_tokens + 4
            self.speech_pos_emb = LearnedPositionEmbeddings(max_speech_seq_len, hidden_size)

        self.speech_head = ParallelLMHead(
            config.speech_vocab_size,
            hidden_size,
        )
        self.logits_processor = LogitsProcessor(config.speech_vocab_size)

        # Original variant uses the full native T3CondEnc (spkr_enc +
        # Perceiver resampler + emotion_adv_fc). The Turbo variant uses the
        # simpler stub (speaker_proj only). Delegating here avoids silently
        # dropping 12 conditioning tensors during weight load.
        try:
            from chatterbox.models.t3.modules.cond_enc import T3CondEnc
            from chatterbox.models.t3.modules.t3_config import T3Config
        except ImportError as e:
            raise ImportError(
                "chatterbox-tts package is required for Chatterbox Original variant. "
                "Install it with: pip install chatterbox-tts"
            ) from e
        _t3_hp = T3Config()
        self.cond_enc = T3CondEnc(_t3_hp)
        # SOT/EOT text-token ids are T3Config constants (255 / 0). Cache them
        # so _tokenize_text doesn't reconstruct T3Config on every prefill.
        self._sot = int(_t3_hp.start_text_token)
        self._eot = int(_t3_hp.stop_text_token)

        # Valid token mask: allow every codec slot + EOS, mask out SOS.
        self.register_buffer(
            "_speech_allowed_mask",
            _build_speech_allowed_mask(config),
            persistent=False,
        )

        # Lazy loaded.
        self._tokenizer = None
        self._voice_encoder = None
        self._s3_tokenizer = None
        self._default_t3_cond = None

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def _tokenize_text(self, text: str, device: torch.device) -> torch.Tensor:
        """Original variant uses custom EnTokenizer (704 vocab), not HuggingFace.

        Wraps the tokenized IDs with ``start_text_token`` (SOT) and
        ``stop_text_token`` (EOT) to match ``ChatterboxTTS.generate``.
        Native pads these in Python before the model sees them; our
        preprocess is the equivalent seam. Skipping them shifts every
        text position by 1 relative to what text_pos_emb expects and
        makes the backbone never emit EOS → infinite rambling until
        ``max_tokens``.
        """
        tokenizer = self._get_tokenizer()
        if hasattr(tokenizer, "encode"):
            text_tokens = list(tokenizer.encode(text))
        else:
            text_tokens = list(tokenizer(text))
        # Wrap with SOT/EOT (T3Config start_text_token=255, stop_text_token=0),
        # cached on self in __init__ to avoid rebuilding T3Config per prefill.
        text_tokens = [self._sot, *text_tokens, self._eot]
        return torch.tensor(text_tokens, dtype=torch.long, device=device)

    def _get_tokenizer(self):
        """Load the custom EnTokenizer for Original variant."""
        if self._tokenizer is None:
            try:
                from chatterbox.models.tokenizers import EnTokenizer
            except ImportError:
                raise ImportError(
                    "chatterbox-tts package is required for Chatterbox Original variant. "
                    "Install it with: pip install chatterbox-tts"
                )

            tok_path = cached_file(self.model_path, "tokenizer.json")
            if tok_path is None:
                raise FileNotFoundError("tokenizer.json not found in model checkpoint")
            self._tokenizer = EnTokenizer(tok_path)
        return self._tokenizer

    def _embed_cond_prompt_speech(self, clamped_tokens: torch.Tensor) -> torch.Tensor:
        """Mirror native ``T3.prepare_conditioning`` for the Original variant.

        Native adds ``speech_pos_emb`` on top of ``speech_emb`` for the non-GPT
        (LLaMA) backbone; without this the Perceiver resampler sees positionally
        ambiguous tokens and the downstream acoustic prior drifts (audible as a
        wrong speaker voice in generated audio).
        """
        emb = self.speech_emb(clamped_tokens)
        if self.speech_pos_emb is not None:
            emb = emb + self.speech_pos_emb(clamped_tokens)
        return emb

    def _get_cond_prompt_tokens_and_emb(
        self, ref_audio_path: str | None, device: torch.device
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return (tokens, embedding) for the speech conditioning prompt.

        Mirrors ``_get_cond_prompt_speech_emb`` but also returns the raw
        tokens so we can pass them to native ``T3Cond`` (which asserts that
        tokens/embedding null-status matches).
        """
        if ref_audio_path is None:
            default_cond = self._load_default_t3_cond(device)
            tokens = default_cond.get("cond_prompt_speech_tokens")
            if tokens is None:
                logger.warning("No cond_prompt_speech_tokens in conds.pt; no speech conditioning")
                return None, None
            tokens = torch.atleast_2d(tokens).to(device)
            plen = self.config.speech_cond_prompt_len
            if tokens.shape[-1] > plen:
                tokens = tokens[:, :plen]
            clamped = tokens.clamp(0, self.config.speech_vocab_size - 1)
            cond_emb = self._embed_cond_prompt_speech(clamped)
            logger.info("Using default speech conditioning: %d tokens", tokens.shape[-1])
            return tokens, cond_emb

        s3_tokenizer = self._ensure_s3_tokenizer(device)
        wav_np = load_wav_mono(ref_audio_path, target_sr=S3_SR)
        if wav_np.shape[-1] > ENC_COND_LEN:
            wav_np = wav_np[:ENC_COND_LEN]
        plen = self.config.speech_cond_prompt_len
        with torch.no_grad():
            tokens, _ = s3_tokenizer([wav_np], max_len=plen)
        tokens = torch.atleast_2d(tokens).to(device)
        if tokens.shape[-1] > plen:
            tokens = tokens[:, :plen]
        clamped = tokens.clamp(0, self.config.speech_vocab_size - 1)
        cond_emb = self._embed_cond_prompt_speech(clamped)
        return tokens, cond_emb

    def _build_prompt_embeds(self, info_dict: dict[str, Any], device: torch.device) -> torch.Tensor:
        """Original variant: route conditioning through the native ``T3CondEnc``.

        Sequence layout:
          [spkr_proj || perceiver(cond_prompt_speech_emb) || emotion_adv_fc(exag) ||
           text_emb || start_speech_token_emb]

        The native ``T3CondEnc.forward`` takes a ``T3Cond`` dataclass that we
        assemble here. We pre-embed ``cond_prompt_speech_tokens`` via our own
        ``speech_emb`` (weights-tied to the backbone) before handing it to the
        Perceiver resampler.
        """
        from chatterbox.models.t3.modules.cond_enc import T3Cond

        text = info_dict["text"][0]
        ref_audio_path = None
        ref_audio_list = info_dict.get("ref_audio")
        if isinstance(ref_audio_list, list) and ref_audio_list:
            ref_audio_path = ref_audio_list[0]

        text_token_ids = self._tokenize_text(text, device)
        text_embedded = self.text_emb(text_token_ids).unsqueeze(0)  # (1, T, H)
        # Native T3.prepare_input_embeds: for ``input_pos_emb == "learned"``
        # (Original variant) add learned position embeddings on top of the
        # token embedding. Without this the model hallucinates content and
        # wanders in speaker identity — the LLaMA backbone's RoPE alone is
        # NOT a substitute, because native stacks learned-pos + RoPE.
        if self.text_pos_emb is not None:
            text_embedded = text_embedded + self.text_pos_emb(text_embedded)

        speaker_emb = self._get_speaker_embedding(ref_audio_path, device)  # (1, 256)

        cond_tokens, cond_speech_emb = self._get_cond_prompt_tokens_and_emb(ref_audio_path, device)

        # Cast to model dtype (VoiceEncoder returns fp32, backbone is bf16).
        model_dtype = self.cond_enc.spkr_enc.weight.dtype
        speaker_emb = speaker_emb.to(dtype=model_dtype)
        if cond_speech_emb is not None:
            cond_speech_emb = cond_speech_emb.to(dtype=model_dtype)

        # Exaggeration / emotion_adv: T3Config.emotion_adv=True, so native
        # forward will assert that cond.emotion_adv is not None. Default to 0.5
        # (matches native library default) if CLI didn't supply one.
        exag_val = info_dict.get("exaggeration", 0.5)
        if isinstance(exag_val, list) and exag_val:
            exag_val = exag_val[0]
        try:
            exag_val = float(exag_val)
        except (TypeError, ValueError):
            logger.warning("Invalid exaggeration %r; falling back to 0.5", exag_val)
            exag_val = 0.5
        # Native Chatterbox trains exaggeration in roughly [0, 1]; clamp so an
        # out-of-range value can't push NaN/garbage into emotion_adv_fc.
        exag_val = min(max(exag_val, 0.0), 1.0)
        emotion_adv = torch.tensor([[exag_val]], dtype=model_dtype, device=device)  # (1, 1)

        # Native T3CondEnc asserts that cond_prompt_speech_tokens and
        # cond_prompt_speech_emb are both-None or both-not-None (tokens are
        # only used for the validation, the perceiver operates on the emb).
        t3_cond = T3Cond(
            speaker_emb=speaker_emb,
            cond_prompt_speech_tokens=cond_tokens,
            cond_prompt_speech_emb=cond_speech_emb,
            emotion_adv=emotion_adv,
        )

        cond_output = self.cond_enc(t3_cond)  # (1, 1 + plen_after_perceiver + 1, H)

        start_token = torch.tensor([self.config.start_speech_token], dtype=torch.long, device=device)
        start_emb = self.speech_emb(start_token).unsqueeze(0)  # (1, 1, H)
        # Start token sits at speech position 0; stack learned pos emb as
        # native does for the Original variant.
        if self.speech_pos_emb is not None:
            start_emb = start_emb + self.speech_pos_emb.get_fixed_embedding(0)

        prompt_embeds = torch.cat([cond_output, text_embedded, start_emb], dim=1)  # (1, L, H)

        # Seed the speech-position counter for the decode hot path. After the
        # prefill consumes position 0 (start token), the first generated speech
        # token is at position 1. preprocess() increments on each decode step.
        info_dict["t3_speech_pos"] = 1
        return prompt_embeds.squeeze(0)  # (L, H)

    # -------------------- Weight loading --------------------

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "tfmr.": "model.",
            "text_emb.": "text_emb.",
            "speech_emb.": "speech_emb.",
            "speech_head.": "speech_head.",
            # cond_enc.* flows through unchanged: we delegate to the native
            # T3CondEnc so spkr_enc / perceiver / emotion_adv_fc match.
            "cond_enc.": "cond_enc.",
            "text_pos_emb.": "text_pos_emb.",
            "speech_pos_emb.": "speech_pos_emb.",
        }
    )

    # vLLM's LLaMA backbone fuses q/k/v into qkv_proj and gate/up into
    # gate_up_proj. The Chatterbox Original checkpoint stores them split, so
    # we map each shard to its fused param via the shard_id kwarg that
    # MergedColumnParallelLinear.weight_loader understands.
    _stacked_params_mapping = [
        # (fused_param_suffix, shard_name_in_ckpt, shard_id)
        (".qkv_proj.", ".q_proj.", "q"),
        (".qkv_proj.", ".k_proj.", "k"),
        (".qkv_proj.", ".v_proj.", "v"),
        (".gate_up_proj.", ".gate_proj.", 0),
        (".gate_up_proj.", ".up_proj.", 1),
    ]

    # vLLM's weight iterator concatenates every safetensors file in the model
    # directory. For Chatterbox Original that means s3gen.safetensors (flow.*),
    # ve.safetensors (lstm.*, proj.*, similarity_*), and t3_cfg.safetensors
    # all flow through this loader. We only care about T3 prefixes here;
    # everything else belongs to other stages and is ignored silently.
    _t3_checkpoint_prefixes = (
        "tfmr.",
        "text_emb.",
        "speech_emb.",
        "speech_head.",
        "cond_enc.",
        "text_pos_emb.",
        "speech_pos_emb.",
        "text_head.",
    )

    # ResembleAI/chatterbox ships English weights (t3_cfg.safetensors) along
    # with multilingual variants (t3_23lang.safetensors, t3_mtl23ls_v2.safetensors)
    # that have a larger text vocab. The loader filters those out via the
    # ``ignore_patterns`` field in the stage config; this set is a belt-and-
    # suspenders filter in case a user-supplied config drops the ignore list.
    _shape_sensitive_keys = frozenset({"text_emb.weight", "text_head.weight"})

    # Checkpoint keys within the T3 namespace whose weights are intentionally
    # unused: ``text_head`` is an unused LM head over the text vocab, and the
    # backbone's ``embed_tokens`` is replaced by our ``text_emb``/``speech_emb``.
    _ignored_checkpoint_prefixes = (
        "text_head.",
        "tfmr.embed_tokens.",
    )

    # Multilingual T3 variants that share key names with the English weights
    # but represent a different model. vLLM's default loader globs every
    # *.safetensors in the cache folder (ignore_patterns only blocks downloads,
    # not local globs), so when these files are already cached they silently
    # overwrite the English weights and break generation (Original produced
    # 70+ seconds of runaway audio because tfmr.* was the 23-language variant
    # and never emitted the English EOS at 6562). Skip any weight yielded
    # from one of these files by inspecting the iterator source.
    _excluded_t3_shard_basenames = frozenset(
        {
            "t3_23lang.safetensors",
            "t3_mtl23ls_v2.safetensors",
            # Legacy .pt siblings, just in case load_format falls back to pt.
            "t3_23lang.pt",
            "t3_mtl23ls_v2.pt",
        }
    )

    def _t3_shard_path(self) -> str | None:
        """Return the absolute path of the English T3 shard (t3_cfg.safetensors).

        Returns None if we can't locate it (e.g. model dir is a custom local
        path missing the expected filename); callers fall back to the vLLM
        weight iterator in that case.
        """
        model_path = getattr(self.vllm_config.model_config, "model", None)
        if not model_path:
            return None
        candidates: list[str] = []
        if os.path.isdir(model_path):
            candidates.append(os.path.join(model_path, "t3_cfg.safetensors"))
        # HF cache case: look up the snapshot directory.
        try:
            from huggingface_hub import snapshot_download

            snap = snapshot_download(
                repo_id=model_path,
                allow_patterns=["t3_cfg.safetensors"],
            )
            candidates.append(os.path.join(snap, "t3_cfg.safetensors"))
        except Exception as exc:
            # Probe only — offline mode or a local-path model make
            # snapshot_download fail legitimately; the glob fallback below
            # still runs and a missing file fails loudly at load time.
            logger.debug("t3_cfg.safetensors snapshot probe failed: %s", exc)
        for c in candidates:
            if os.path.isfile(c):
                return c
        # Last resort: glob the HF cache, but scope to THIS repo's snapshot
        # directory (``models--<org>--<name>``) so we never silently load
        # ``t3_cfg.safetensors`` from an unrelated model. A local model_path
        # (not a repo id) simply won't match and we return None.
        if "/" not in model_path:
            return None
        repo_dir = "models--" + model_path.replace("/", "--")
        cache_root = os.environ.get("HF_HUB_CACHE") or os.path.expanduser("~/.cache/huggingface/hub")
        snap_pattern = os.path.join(cache_root, repo_dir, "snapshots", "*", "t3_cfg.safetensors")
        hits = glob.glob(snap_pattern)
        return hits[0] if hits else None

    def _english_only_weight_iterator(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Replace vLLM's glob-everything iterator with a t3_cfg-only one.

        This guarantees the 23-language and MTL variants sitting in the same
        HF cache snapshot never leak into the English T3's params. If we
        can't locate t3_cfg.safetensors we fall back to the original iterator
        (degraded but not worse than pre-fix behavior).
        """
        shard = self._t3_shard_path()
        if shard is None:
            logger.warning(
                "ChatterboxT3ForGeneration: could not locate t3_cfg.safetensors; "
                "falling back to vLLM's default weight iterator (may include "
                "multilingual shards if they are in the same cache folder)."
            )
            return weights

        from safetensors import safe_open

        logger.info(
            "ChatterboxT3ForGeneration: loading T3 weights exclusively from %s",
            shard,
        )

        def _iter():
            with safe_open(shard, framework="pt") as f:
                for key in f.keys():
                    yield key, f.get_tensor(key)

        return _iter()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        unconsumed: list[str] = []

        # Swap in an iterator that only reads t3_cfg.safetensors so we are
        # immune to stale multilingual shards sitting in the HF cache.
        weights = self._english_only_weight_iterator(weights)

        for name, loaded_weight in weights:
            # Filter to the T3 namespace — ignore weights from co-located
            # safetensors files that belong to other stages (S3Gen, VoiceEncoder).
            if not any(name.startswith(p) for p in self._t3_checkpoint_prefixes):
                continue
            if any(name.startswith(p) for p in self._ignored_checkpoint_prefixes):
                continue

            # Belt-and-suspenders: if somehow we are on the fallback path and
            # a multilingual variant slipped through, its text_emb / text_head
            # will have the wrong shape — skip rather than error out.
            if name in self._shape_sensitive_keys:
                target = params_dict.get(name)
                if target is not None and target.data.shape != loaded_weight.shape:
                    continue

            mapped_name = name
            for old_prefix, new_prefix in self.hf_to_vllm_mapper.orig_to_new_prefix.items():
                if name.startswith(old_prefix):
                    mapped_name = new_prefix + name[len(old_prefix) :]
                    break

            # Fused QKV / gate_up path: checkpoint stores them split; redirect
            # each shard to the fused param with a shard_id so the weight
            # loader places it in the right column slice.
            matched_stacked = False
            for fused_suffix, shard_suffix, shard_id in self._stacked_params_mapping:
                if shard_suffix not in mapped_name:
                    continue
                fused_name = mapped_name.replace(shard_suffix, fused_suffix)
                if fused_name not in params_dict:
                    continue
                param = params_dict[fused_name]
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader is None:
                    break
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(fused_name)
                matched_stacked = True
                break
            if matched_stacked:
                continue

            if mapped_name not in params_dict:
                unconsumed.append(f"{name} -> {mapped_name}")
                continue

            param = params_dict[mapped_name]
            weight_loader = getattr(param, "weight_loader", None)
            if weight_loader is not None:
                weight_loader(param, loaded_weight)
            else:
                if param.data.shape != loaded_weight.shape:
                    raise RuntimeError(
                        f"Shape mismatch loading '{name}' -> '{mapped_name}': "
                        f"param {tuple(param.data.shape)} vs loaded {tuple(loaded_weight.shape)}. "
                        "If this is a multilingual variant of the checkpoint, "
                        "add its key to _shape_sensitive_keys."
                    )
                param.data.copy_(loaded_weight)
            loaded_params.add(mapped_name)

        if unconsumed:
            # Fail loud: silently dropping checkpoint weights is how we ended
            # up shipping a T3 without its Perceiver resampler (12 tensors
            # silently discarded). Never again.
            raise RuntimeError(
                "ChatterboxT3ForGeneration.load_weights: "
                f"{len(unconsumed)} checkpoint key(s) could not be placed "
                "into the model. Unconsumed:\n  " + "\n  ".join(unconsumed)
            )

        # The LlamaModel backbone ships a vestigial ``model.embed_tokens`` layer
        # (we pass ``vocab_size=8`` to skip it and use our own ``text_emb`` /
        # ``speech_emb`` pair). The forward path never invokes it, but vLLM's
        # default loader runs a strict ``weights_to_load - loaded_weights``
        # check — so report it as loaded to satisfy the check.
        embed_name = "model.embed_tokens.weight"
        if embed_name in params_dict and embed_name not in loaded_params:
            with torch.no_grad():
                params_dict[embed_name].data.zero_()
            loaded_params.add(embed_name)

        return loaded_params
