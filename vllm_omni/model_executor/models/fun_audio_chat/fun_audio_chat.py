# SPDX-License-Identifier: Apache-2.0
"""Fun-Audio-Chat-8B: Whisper encoder + Qwen3 LM for S2S turn-based chat.

Weight layout in checkpoint:
  continuous_audio_tower.*  → FunAudioEncoder (self.continuous_audio_tower)
  audio_tower.output_matching.*  → self.audio_tower_output_matching
  language_model.*  → Qwen3ForCausalLM (self.language_model)
  audio_invert_tower.*  → FunAudioChatDecoder (self.audio_invert_tower)
  audio_tower.embed_tokens.*  → audio_invert_tower.lm_head (tied weights)
  audio_tower.continual_output_matching.*  → skipped (higher-res path, not wired)

S2S path uses the native vllm-omni two-stage pipeline:
  Stage 0 (this file, LLM_AR): text decode + per-step CRQ audio tokens
    - audio_invert_tower runs at each decode step in compute_logits()
    - Per-step CRQ tokens are accumulated via postprocess() into
      model_intermediate_buffer[req_id]["crq_tokens"] (single-request BS=1)
    - make_omni_output() emits {"crq_tokens": [per_req_tensor, ...]} for
      per-request distribution to Stage 1
  Stage 1 (token2wav.py, LLM_GENERATION): CRQ tokens → WAV via CosyVoice3

No ASGI middleware, no IPC file, no module-level state — all state flows
through the standard vllm-omni OmniOutput mechanism.

Concurrency note: BS=1 only. CRQ decoder state (crq_past_key_values,
crq_audio_embeds, crq_speech_ids) is on the module instance and not
separable per-request in a batched decode. Serve with --max-num-seqs 1.
"""

import logging
import math
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─── S2S: lazy import of FunAudioChatDecoder from src/funaudiochat ─────────────
_FUN_REF = os.environ.get(
    "FUN_AUDIO_REF_PATH",
    str(Path(__file__).parents[5] / "src" / "funaudiochat"),
)
FunAudioChatDecoder = None  # type: ignore[assignment]

def _try_import_decoder() -> None:
    """Import FunAudioChatDecoder; raise if unavailable (S2S is mandatory)."""
    global FunAudioChatDecoder
    if FunAudioChatDecoder is not None:
        return
    for p in [_FUN_REF]:
        if p not in sys.path:
            sys.path.insert(0, p)
    # Import FunAudioChatDecoder directly without register_funaudiochat() —
    # vllm already registered "funaudiochat" with AutoConfig; calling
    # register_funaudiochat() again would raise a duplicate-registration error.
    from funaudiochat.modeling_funaudiochat import FunAudioChatDecoder as _Dec  # type: ignore[import]
    FunAudioChatDecoder = _Dec

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature
from transformers import WhisperConfig as _WhisperConfig
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    AudioItem,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
try:
    from vllm.multimodal.processing import BaseDummyInputsBuilder
except ImportError:
    from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.transformers_utils.configs.fun_audio_chat import (
    FunAudioChatAudioConfig,
    FunAudioChatConfig,
)

# ─── Special token IDs (from tokenizer_config.json added_tokens_decoder) ─────
AUDIO_TOKEN_ID = 151669      # <|AUDIO|> — placeholder in prompt
AUDIO_BOS_TOKEN_ID = 151670  # <|audio_bos|>
AUDIO_EOS_TOKEN_ID = 151671  # <|audio_eos|>
AUDIO_PAD_TOKEN_ID = 151672  # <|audio_pad|>


# ─── Audio encoder ────────────────────────────────────────────────────────────

class FunAudioEncoder(nn.Module):
    """Whisper-Large-V3-compatible encoder for Fun-Audio-Chat-8B.

    Weight prefix in checkpoint: continuous_audio_tower.*

    Differences from stock transformers WhisperEncoder:
      - Final norm attribute is named `ln_post` (not `layer_norm`)
      - Additional `proj` Linear(d_model → output_dim) applied after ln_post
      - Additional `audio_bos_eos_token` Embedding(2, output_dim) (unused in S2T path)
      - No stored positional embeddings; sinusoidal table regenerated at init
    """

    def __init__(self, config: FunAudioChatAudioConfig) -> None:
        super().__init__()
        from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer

        # Build a WhisperConfig-compatible object for WhisperEncoderLayer
        whisper_cfg = _WhisperConfig(
            d_model=config.d_model,
            encoder_ffn_dim=config.encoder_ffn_dim,
            encoder_attention_heads=config.encoder_attention_heads,
            encoder_layers=config.encoder_layers,
            num_mel_bins=config.num_mel_bins,
            max_source_positions=config.max_source_positions,
            activation_function=config.activation_function,
            attention_dropout=config.attention_dropout,
            dropout=config.dropout,
            scale_embedding=config.scale_embedding,
            encoder_layerdrop=0.0,
            attn_implementation="eager",
        )
        self.config = config
        self.conv1 = nn.Conv1d(config.num_mel_bins, config.d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(config.d_model, config.d_model, kernel_size=3, stride=2, padding=1)
        self.layers = nn.ModuleList([
            WhisperEncoderLayer(whisper_cfg) for _ in range(config.encoder_layers)
        ])
        self.ln_post = nn.LayerNorm(config.d_model)
        self.proj = nn.Linear(config.d_model, config.output_dim, bias=True)
        self.audio_bos_eos_token = nn.Embedding(2, config.output_dim)

        # Fixed sinusoidal positional embeddings (not stored in checkpoint)
        self.register_buffer(
            "positional_embedding",
            self._sinusoidal_embedding(config.max_source_positions, config.d_model),
            persistent=False,
        )

    @staticmethod
    def _sinusoidal_embedding(max_positions: int, d_model: int) -> torch.Tensor:
        """Standard Whisper sinusoidal positional embeddings [1, max_positions, d_model]."""
        half = d_model // 2
        dims = torch.arange(half, dtype=torch.float32)
        inv_timescales = torch.exp(-math.log(10000.0) / (half - 1) * dims)
        positions = torch.arange(max_positions, dtype=torch.float32)
        scaled = positions.unsqueeze(1) * inv_timescales.unsqueeze(0)  # [T, half]
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1).unsqueeze(0)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Encode audio mel features to projected, grouped embeddings.

        Args:
            input_features: [batch, num_mel_bins, T] — from WhisperFeatureExtractor

        Returns:
            [batch, ceil(T/2/group_size), output_dim]

        Raises:
            ValueError: input_features.shape[1] != num_mel_bins
        """
        if input_features.shape[1] != self.config.num_mel_bins:
            raise ValueError(
                f"Expected {self.config.num_mel_bins} mel bins, "
                f"got {input_features.shape[1]}"
            )
        # Cast to model dtype (HF feature extractor outputs float32)
        input_features = input_features.to(self.conv1.weight.dtype)
        # CNN downsampling (Whisper pattern)
        x = F.gelu(self.conv1(input_features))
        x = F.gelu(self.conv2(x))       # [B, d_model, T/2]
        x = x.permute(0, 2, 1)          # [B, T/2, d_model]

        # Sinusoidal positional embeddings
        seq_len = x.size(1)
        x = x + self.positional_embedding[:, :seq_len, :].to(x.dtype)

        # Whisper encoder layers
        for layer in self.layers:
            x = layer(x, attention_mask=None, layer_head_mask=None)[0]

        # Final norm + projection
        x = self.ln_post(x)            # [B, T/2, d_model]
        x = self.proj(x)               # [B, T/2, output_dim]

        # Avg-pool into groups of audio_group_size (= 5)
        group_size = self.config.group_size
        B, T, D = x.shape
        pad_len = (group_size - T % group_size) % group_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        x = x.view(B, -1, group_size, D).mean(dim=2)  # [B, T/group_size, D]

        return x


# ─── Multi-modal processing ──────────────────────────────────────────────────

class FunAudioChatProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> FunAudioChatConfig:
        return self.ctx.get_hf_config(FunAudioChatConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_hf_processor(self, **kwargs: object):
        # Use WhisperFeatureExtractor directly (no FunAudioChatProcessor in transformers)
        from transformers import WhisperFeatureExtractor
        return WhisperFeatureExtractor(
            feature_size=128,
            sampling_rate=16000,
            hop_length=160,
            chunk_length=30,
            n_fft=400,
            padding_value=0.0,
            return_attention_mask=True,
        )

    def get_num_audio_tokens(self, audio_seconds: float) -> int:
        """Return number of <|AUDIO|> tokens for a given audio duration."""
        mel_frames = math.ceil(audio_seconds * 16000 / 160)
        encoder_frames = (mel_frames - 1) // 2 + 1
        group_size = self.get_hf_config().audio_config.group_size
        return math.ceil(encoder_frames / group_size)


class FunAudioChatDummyInputsBuilder(
    BaseDummyInputsBuilder[FunAudioChatProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        return "<|AUDIO|>" * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> dict[str, Any]:
        num_audios = mm_counts.get("audio", 0)
        # 1 second of silence per dummy audio
        audio_len = 16000
        return {
            "audio": self._get_dummy_audios(
                length=audio_len, num_audios=num_audios,
                overrides=mm_options.get("audio") if mm_options else None,
            )
        }


def _fun_audio_chat_field_config(
    hf_inputs: Mapping[str, torch.Tensor],
) -> dict[str, MultiModalFieldConfig]:
    # WhisperFeatureExtractor returns 'input_features' and 'attention_mask' (raw
    # waveform level). We only need input_features for the encoder; the raw-sample
    # attention_mask is not used.
    return {
        "input_features": MultiModalFieldConfig.batched("audio"),
    }


class FunAudioChatMultiModalProcessor(
    BaseMultiModalProcessor[FunAudioChatProcessingInfo]
):
    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=16000)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        processor = self.info.get_hf_processor(**mm_kwargs)
        audios = mm_data.get("audios") or mm_data.get("audio", [])
        if not audios:
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature({"input_ids": [prompt_ids]}, tensor_type="pt")

        if not isinstance(audios, list):
            audios = [audios]

        audio_inputs = processor(
            audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        tokenizer = self.info.get_tokenizer()
        text_inputs = tokenizer(prompt, return_tensors="pt")
        return BatchFeature({
            "input_ids": text_inputs["input_ids"],
            **audio_inputs,
        })

    def _hf_processor_applies_updates(self, prompt_text, mm_items, hf_processor_mm_kwargs, tokenization_kwargs) -> bool:
        # WhisperFeatureExtractor processes audio only — it does NOT modify prompt text.
        # Return False so vllm applies the PromptReplacement token substitution itself.
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _fun_audio_chat_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        # When all mm items are already in cache, mm_items is empty — no updates needed.
        if "audio" not in mm_items:
            return []

        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        audio_token_id = vocab.get("<|AUDIO|>", AUDIO_TOKEN_ID)
        audio_bos_id = vocab.get("<|audio_bos|>", AUDIO_BOS_TOKEN_ID)
        audio_eos_id = vocab.get("<|audio_eos|>", AUDIO_EOS_TOKEN_ID)

        out_data = out_mm_kwargs.get_data()
        feature_attention_mask = out_data.get("feature_attention_mask")

        audio_lengths: list[int] = []
        if feature_attention_mask is not None:
            assert isinstance(feature_attention_mask, torch.Tensor)
            for i in range(feature_attention_mask.shape[0]):
                active = int(feature_attention_mask[i].sum().item())
                encoder_frames = (active - 1) // 2 + 1
                group_size = self.info.get_hf_config().audio_config.group_size
                audio_lengths.append(math.ceil(encoder_frames / group_size))

        if not audio_lengths:
            audios = mm_items.get_items("audio", AudioProcessorItems)
            for i in range(len(audios)):
                audio_array = audios.get(i)
                duration = len(audio_array) / 16000.0
                audio_lengths.append(self.info.get_num_audio_tokens(duration))

        def make_replacement(item_idx: int):
            n = audio_lengths[item_idx] if item_idx < len(audio_lengths) else 10
            # Placeholder string already has <|audio_bos|>...<|audio_eos|> wrapping;
            # only replace the single <|AUDIO|> token with n audio feature tokens.
            tokens = [audio_token_id] * n
            return PromptUpdateDetails.select_token_id(tokens, embed_token_id=audio_token_id)

        return [
            PromptReplacement(
                modality="audio",
                target=[audio_token_id],
                replacement=make_replacement,
            )
        ]


# ─── Main model ──────────────────────────────────────────────────────────────

@MULTIMODAL_REGISTRY.register_processor(
    FunAudioChatMultiModalProcessor,
    info=FunAudioChatProcessingInfo,
    dummy_inputs=FunAudioChatDummyInputsBuilder,
)
class FunAudioChatForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    """Fun-Audio-Chat-8B Stage 0: Whisper encoder + Qwen3 LM + CRQ audio token decoder.

    Emits both text tokens (via standard LM head) and per-step CRQ audio token IDs
    (via audio_invert_tower) for Stage 1 (token2wav.py) consumption through the
    native vllm-omni OmniOutput pipeline.

    BS=1 only (CRQ decoder state is on the module, not per-request).
    """

    # Native vllm-omni pipeline hooks.
    have_multimodal_outputs: bool = True
    has_preprocess: bool = False
    has_postprocess: bool = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            # Reference uses exactly '<|audio_bos|><|AUDIO|><|audio_eos|>' as user content.
            # No "Audio N:" prefix — model not trained with it.
            return "<|audio_bos|><|AUDIO|><|audio_eos|>"
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: FunAudioChatConfig = vllm_config.model_config.hf_config
        self.config = config
        self.audio_token_index = config.audio_token_index

        self.continuous_audio_tower = FunAudioEncoder(config.audio_config)
        self.audio_tower_output_matching = nn.Linear(
            config.audio_config.output_dim,
            config.audio_config.output_dim,
            bias=False,
        )
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen3ForCausalLM"],
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # S2S: audio_invert_tower (FunAudioChatDecoder) — raises if unavailable
        _try_import_decoder()
        self.audio_invert_tower = FunAudioChatDecoder(config.audio_config)
        logger.info("fun_audio_chat: S2S path enabled (audio_invert_tower loaded)")

        # Per-request state for CRQ generation (reset at prefill)
        self._last_input_ids: torch.Tensor | None = None
        self._crq_initialized: bool = False
        self._speech_started: bool = False  # True after AUDIO_BOS_TOKEN_ID appears in decode input
        # Newly generated CRQ tokens from this decode step; consumed by postprocess()
        self._last_crq_tokens: torch.Tensor | None = None

    def _encode_audio(
        self,
        input_features: torch.Tensor | list[torch.Tensor],
        feature_attention_mask: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        """Encode audio features, return list of per-audio embedding tensors."""
        if isinstance(input_features, (list, tuple)):
            # Batch is a list of individual feature tensors
            results = []
            for feat in input_features:
                if feat.dim() == 2:
                    feat = feat.unsqueeze(0)
                enc = self.continuous_audio_tower(feat)   # [1, N, D]
                enc = self.audio_tower_output_matching(enc)
                results.append(enc.squeeze(0))            # [N, D]
            return results
        else:
            # Batch tensor [B, 128, T]
            enc = self.continuous_audio_tower(input_features)  # [B, N, D]
            enc = self.audio_tower_output_matching(enc)         # [B, N, D]
            return [enc[i] for i in range(enc.size(0))]

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        input_features = kwargs.get("input_features")
        if input_features is None:
            return []
        return self._encode_audio(input_features, feature_attention_mask=None)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # Save input_ids so compute_logits can build text_embeds for audio_invert_tower
        self._last_input_ids = input_ids

        hidden_states = self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        text_logits = self.language_model.compute_logits(hidden_states)

        # vllm passes hidden_states as 2D [num_tokens, hidden_size] (not [batch, seq, hidden]).
        # num_tokens > 1 at prefill (full prompt); num_tokens == 1 at decode (one token/sequence).
        ntokens = hidden_states.shape[0]
        dev = hidden_states.device

        # Reset per-step CRQ token cache; postprocess() consumes it below.
        self._last_crq_tokens = None

        if ntokens > 1:
            # Prefill: reset CRQ state for this request.
            self._speech_started = False
            self.audio_invert_tower.forward = self.audio_invert_tower.crq_generate_forward
            self.audio_invert_tower.crq_past_key_values = None
            self.audio_invert_tower.crq_audio_embeds = None
            self.audio_invert_tower.crq_do_sample = False  # greedy audio decoding
            self.audio_invert_tower.crq_speech_ids = torch.empty(
                1, 0, dtype=torch.long, device=dev
            )
            from transformers import LogitsProcessorList  # type: ignore[import]
            self.audio_invert_tower.crq_logits_processor = LogitsProcessorList()
            self.audio_invert_tower.crq_grobal_step = 0
            self._crq_initialized = True

        elif ntokens == 1 and self._crq_initialized and self._last_input_ids is not None:
            # Decode step: gate CRQ on audio_bos_index appearing in input_ids.
            # _last_input_ids is 1D [num_tokens] in vllm; at decode, num_tokens==1.
            if not self._speech_started:
                if AUDIO_BOS_TOKEN_ID in self._last_input_ids:
                    self._speech_started = True

            if self._speech_started:
                try:
                    # Reshape vllm's 2D [1, hidden_size] → 3D [1, 1, hidden_size] for crq_generate_forward
                    crq_hs = hidden_states.unsqueeze(0)  # [1, 1, hidden_size]

                    # Per reference: speech_inputs_embeds = hidden + text_embeds.
                    # embed_tokens accepts 1D input in vllm; returns [1, hidden_size].
                    last_tok = self._last_input_ids[-1:]  # [1] (last/only token)
                    text_embeds = self.language_model.model.embed_tokens(last_tok)  # [1, hidden_size]
                    text_embeds = text_embeds.unsqueeze(0)  # [1, 1, hidden_size]

                    crq_input = crq_hs + text_embeds.detach()

                    with torch.no_grad():
                        self.audio_invert_tower(inputs_embeds=crq_input)

                    # crq_generate_tokens: [batch, group_size] = [1, 5] after crq_generate_forward
                    # Save as 1D CPU tensor for postprocess() to consume and accumulate.
                    self._last_crq_tokens = (
                        self.audio_invert_tower.crq_generate_tokens[0]
                        .flatten()
                        .detach()
                        .to(dtype=torch.long, device="cpu")
                    )

                    # Append group_size tokens to speech history (single-request only)
                    self.audio_invert_tower.crq_speech_ids = torch.cat([
                        self.audio_invert_tower.crq_speech_ids,
                        self.audio_invert_tower.crq_generate_tokens[:1],  # [1, group_size]
                    ], dim=-1)
                except Exception as exc:
                    logger.warning("fun_audio_chat: audio_invert_tower decode error: %s", exc)

        return text_logits

    # ── Native vllm-omni pipeline hooks: postprocess + make_omni_output ────────

    def postprocess(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: Any = None,
        **req_infos: Any,
    ) -> dict[str, Any]:
        """Accumulate per-step CRQ tokens into this request's intermediate buffer.

        Called by the vllm-omni gpu_model_runner after each decode step, once
        per request. Reads the existing accumulated ``crq_tokens`` from
        ``req_infos`` (which is the current model_intermediate_buffer[req_id]),
        appends the tokens produced this step (stored in ``self._last_crq_tokens``
        by compute_logits), and returns the updated tensor. The framework
        overwrites the buffer with the returned dict, so accumulation must be
        done explicitly here.

        BS=1 only: CRQ decoder state is module-level, so only one request's
        worth of tokens can be produced per compute_logits call.
        """
        new_tokens = self._last_crq_tokens
        if new_tokens is None or new_tokens.numel() == 0:
            return {}
        existing = req_infos.get("crq_tokens")
        if isinstance(existing, torch.Tensor) and existing.numel() > 0:
            accumulated = torch.cat([existing.flatten().cpu(), new_tokens.flatten()])
        else:
            accumulated = new_tokens.flatten()
        # Consumed: guard against re-applying to later postprocess calls in the
        # unlikely event of a multi-request batch.
        self._last_crq_tokens = None
        return {"crq_tokens": accumulated}

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput:
        """Emit OmniOutput carrying per-request accumulated CRQ tokens.

        Returns multimodal_outputs={"crq_tokens": [req0_tokens, req1_tokens, ...]}
        indexed by request order. The downstream distribution logic (see
        vllm_omni.utils.mm_outputs.to_payload_element) picks the per-request
        entry by index so each CompletionOutput's multimodal_output gets its
        own tensor.
        """
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        hidden = model_outputs
        info_dicts = kwargs.get("model_intermediate_buffer")
        if info_dicts is None:
            info_dicts = kwargs.get("runtime_additional_information") or []

        per_req_tokens: list[torch.Tensor] = []
        for info in info_dicts:
            if isinstance(info, dict):
                tokens = info.get("crq_tokens")
                if isinstance(tokens, torch.Tensor):
                    per_req_tokens.append(tokens.flatten().to(torch.long).cpu())
                    continue
            per_req_tokens.append(torch.empty(0, dtype=torch.long))

        if not per_req_tokens:
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={"crq_tokens": per_req_tokens},
        )


    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Keys to skip entirely
        _SKIP_PREFIXES = (
            "audio_tower.continual_output_matching.",  # higher-res path, not wired
            # FunAudioChatDecoder deletes crq_transformer.embed_tokens in __init__;
            # skip these checkpoint weights to avoid AutoWeightsLoader key errors.
            "audio_invert_tower.crq_transformer.embed_tokens.",
        )

        # Name remapping: checkpoint key → module attribute path.
        # audio_tower.embed_tokens.* → audio_invert_tower.lm_head.* because
        # these weights are tied in the transformers model.
        _REMAP = {
            "audio_tower.output_matching.": "audio_tower_output_matching.",
            "audio_tower.embed_tokens.": "audio_invert_tower.lm_head.",
            "audio_invert_tower.": "audio_invert_tower.",
            "continuous_audio_tower.": "continuous_audio_tower.",
            "language_model.": "language_model.",
        }

        filtered: list[tuple[str, torch.Tensor]] = []
        for name, param in weights:
            if any(name.startswith(p) for p in _SKIP_PREFIXES):
                continue
            for old, new in _REMAP.items():
                if name.startswith(old):
                    name = new + name[len(old):]
                    break
            filtered.append((name, param))

        loader = AutoWeightsLoader(self)
        return loader.load_weights(iter(filtered))
