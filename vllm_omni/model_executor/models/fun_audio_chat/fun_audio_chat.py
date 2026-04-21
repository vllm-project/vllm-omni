# SPDX-License-Identifier: Apache-2.0
"""Fun-Audio-Chat-8B: native S2S model for vllm-omni.

Reference: github.com/FunAudioLLM/Fun-Audio-Chat
  funaudiochat/modeling_funaudiochat.py (FunAudioChatForConditionalGeneration)
  funaudiochat/processing_funaudiochat.py (FunAudioChatProcessor)

Differences from the reference (documented in design/plan-fun-audio-chat-s2s.md):
  * We use vllm-omni's two-stage pipeline. Stage 0 (this file) emits text tokens
    + per-step CRQ audio tokens via the audio_invert_tower. Stage 1
    (token2wav.py) converts CRQ tokens -> WAV via native CosyVoice3.
  * CRQ state (past_kv, audio_embeds) currently lives on the module attribute
    `audio_invert_tower.crq_state`; BS>=1 upgrade moves it into
    model_intermediate_buffer[req_id]. Documented; single-request deployment
    works today.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from transformers import BatchFeature, LogitsProcessorList
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    AudioItem,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
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

from vllm_omni.model_executor.models.fun_audio_chat.crq_decoder import (
    CRQState,
    FunAudioChatDecoder,
)
from vllm_omni.model_executor.models.fun_audio_chat.encoder import (
    FunAudioChatAudioEncoder,
    FunAudioChatDiscreteEncoder,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.transformers_utils.configs.fun_audio_chat import (
    FunAudioChatAudioEncoderConfig,
    FunAudioChatConfig,
)

logger = logging.getLogger(__name__)


# ─── Prompt / placeholder token ids ───────────────────────────────────────────
# Fallback defaults; actual ids come from config.
_AUDIO_TOKEN_ID_DEFAULT = 151669   # <|AUDIO|>
_AUDIO_BOS_INDEX_DEFAULT = 151670  # <|audio_bos|>
_AUDIO_EOS_INDEX_DEFAULT = 151671  # <|audio_eos|>
_AUDIO_PAD_TOKEN_ID = 151672       # <|audio_pad|>

# Sampling defaults for the CRQ side (ref: utils/constant.py DEFAULT_SP_GEN_KWARGS).
_DEFAULT_CRQ_DO_SAMPLE = False
_DEFAULT_TEXT_GREEDY = True
_DEFAULT_FORCE_TEXT_ABOS = True  # S2S turns always need to start in speech mode


# ─── Multi-modal processing ──────────────────────────────────────────────────

class FunAudioChatProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> FunAudioChatConfig:
        return self.ctx.get_hf_config(FunAudioChatConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}  # allow multi-turn

    def get_hf_processor(self, **kwargs: object):
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
        """Placeholder count per <|AUDIO|> audio. Matches reference processor.

        Reference (processing_funaudiochat.py L196-223, 25 Hz speech-token rate):
            num_frames = int(duration * 25)
            num_audio_tokens = ceil(num_frames / group_size)
        So roughly 5 placeholders per second (group_size=5).
        """
        group_size = self.get_hf_config().audio_config.group_size
        num_frames_25hz = int(audio_seconds * 25)
        return math.ceil(num_frames_25hz / group_size)


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
        audio_len = 16000  # 1 s
        return {
            "audio": self._get_dummy_audios(
                length=audio_len, num_audios=num_audios,
                overrides=mm_options.get("audio") if mm_options else None,
            )
        }


def _fun_audio_chat_field_config(
    hf_inputs: Mapping[str, torch.Tensor],
) -> dict[str, MultiModalFieldConfig]:
    return {
        "input_features": MultiModalFieldConfig.batched("audio"),
        "feature_attention_mask": MultiModalFieldConfig.batched("audio"),
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
            audios, sampling_rate=16000, return_tensors="pt", padding="max_length",
            return_attention_mask=True,
        )
        # Whisper returns attention_mask for the raw waveform; rename so it
        # doesn't conflict with the text attention_mask later.
        if "attention_mask" in audio_inputs:
            audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask")

        tokenizer = self.info.get_tokenizer()
        text_inputs = tokenizer(prompt, return_tensors="pt")
        return BatchFeature({
            "input_ids": text_inputs["input_ids"],
            **audio_inputs,
        })

    def _hf_processor_applies_updates(
        self, prompt_text, mm_items, hf_processor_mm_kwargs, tokenization_kwargs
    ) -> bool:
        # WhisperFeatureExtractor processes audio only; text placeholder
        # substitution is left to vllm's PromptReplacement.
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
        if "audio" not in mm_items:
            return []

        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        audio_token_id = vocab.get("<|AUDIO|>", _AUDIO_TOKEN_ID_DEFAULT)

        # Reference processing_funaudiochat.py L196-223: placeholder count comes
        # from 25 Hz speech-token rate, grouped by group_size. Derive duration
        # from the waveform; WhisperFeatureExtractor does NOT give us per-audio
        # lengths at this stage, so we use the parsed audio items directly.
        group_size = self.info.get_hf_config().audio_config.group_size
        audios = mm_items.get_items("audio", AudioProcessorItems)
        audio_lengths: list[int] = []
        for i in range(len(audios)):
            audio_array = audios.get(i)
            duration = len(audio_array) / 16000.0
            num_frames_25hz = int(duration * 25)
            audio_lengths.append(math.ceil(num_frames_25hz / group_size))

        def make_replacement(item_idx: int):
            n = audio_lengths[item_idx] if item_idx < len(audio_lengths) else 10
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
    """Stage 0: Whisper-style continuous encoder + discrete encoder + Qwen3 LM
    + CRQ audio-token decoder (audio_invert_tower).

    Emits:
      - text_logits for LM token sampling (normal vllm path)
      - multimodal_output["crq_tokens"]: accumulated 1D LongTensor of CRQ
        speech tokens; consumed by Stage 1 (token2wav.py).

    BS=1 today: CRQ state lives on audio_invert_tower as a CRQState attribute.
    BS>=1 upgrade path: move CRQState into model_intermediate_buffer[req_id]
    (design/plan-fun-audio-chat-s2s.md O6).
    """

    have_multimodal_outputs: bool = True
    has_preprocess: bool = False
    has_postprocess: bool = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<|audio_bos|><|AUDIO|><|audio_eos|>"
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: FunAudioChatConfig = vllm_config.model_config.hf_config
        self.config = config
        self.audio_token_index = config.audio_token_index

        audio_cfg: FunAudioChatAudioEncoderConfig = config.audio_config
        self.continuous_audio_tower = FunAudioChatAudioEncoder(audio_cfg)
        self.audio_tower = FunAudioChatDiscreteEncoder(audio_cfg)
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen3ForCausalLM"],
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        self.audio_invert_tower = FunAudioChatDecoder(audio_cfg)

        # Tied weight: reference stores the audio-token embedding only once,
        # under `audio_tower.embed_tokens.weight`, and the CRQ LM head uses the
        # same tensor. Enforce the tie so loading either one covers both.
        self.audio_invert_tower.lm_head.weight = self.audio_tower.embed_tokens.weight

        # Resolved special-token ids from config.
        self._audio_bos_index = getattr(
            config.text_config, "audio_bos_index", _AUDIO_BOS_INDEX_DEFAULT
        )
        self._audio_eos_index = getattr(
            config.text_config, "audio_eos_index", _AUDIO_EOS_INDEX_DEFAULT
        )
        self._crq_eos_token_id = audio_cfg.eos_token_id  # e.g. 6562
        self._crq_bos_token_id = audio_cfg.bos_token_id  # e.g. 6561
        self._crq_codebook_range = audio_cfg.bos_token_id  # valid tokens are 0..bos-1

        # Per-request CRQ state — BS=1 only in this revision.
        self._crq_state: CRQState | None = None
        self._generate_speech: bool = False
        self._speech_finished: bool = False
        self._last_input_ids: torch.Tensor | None = None
        self._pending_force_text_abos: bool = _DEFAULT_FORCE_TEXT_ABOS
        self._last_crq_tokens: torch.Tensor | None = None

    # ── Audio encoding (prefill path) ─────────────────────────────────────────

    def _encode_audio_prefill(
        self,
        input_features: torch.Tensor | list[torch.Tensor],
        feature_attention_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Run continuous encoder + discrete encoder fusion at prefill.

        Reference: modeling_funaudiochat.py L978-1064. For inference, the
        processor supplies speech_ids filled with `audio_pad_token_id`; the
        discrete encoder's continual_output_matching branch (controlled by
        `continuous_features_mode`, "replace" in the checkpoint) overwrites
        the pad embedding with the continuous features' group-pooled output.

        Returns a list of [T, output_dim] tensors (one per audio), ready for
        masked_scatter into `inputs_embeds` at <|AUDIO|> positions.
        """
        audio_cfg: FunAudioChatAudioEncoderConfig = self.config.audio_config
        group_size = audio_cfg.group_size
        pad_id = audio_cfg.pad_token_id

        # Normalize input: we want a list of per-audio [num_mel_bins, T] tensors.
        if isinstance(input_features, (list, tuple)):
            feats = [
                f.squeeze(0) if f.dim() == 3 else f for f in input_features
            ]
        else:
            feats = [input_features[i] for i in range(input_features.size(0))]

        results: list[torch.Tensor] = []
        for feat in feats:
            # feat: [num_mel_bins, T_mel]
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            feature_lens = torch.tensor([feat.shape[-1]], device=feat.device)
            aftercnn_lens, output_lens = self.continuous_audio_tower._get_feat_extract_output_lengths(
                feature_lens
            )
            # speech_maxlen is the flat speech_ids length (before group pooling).
            # Pad to multiple of group_size per ref.
            speech_maxlen = int(output_lens.item())
            speech_maxlen = ((speech_maxlen + group_size - 1) // group_size) * group_size
            cont = self.continuous_audio_tower(
                feat, feature_lens=feature_lens, aftercnn_lens=aftercnn_lens,
                speech_maxlen=speech_maxlen,
            ).last_hidden_state  # [1, speech_maxlen, output_dim]

            # Discrete side: all-pad speech_ids with shape [1, speech_maxlen].
            speech_ids = torch.full(
                (1, speech_maxlen), pad_id, dtype=torch.long, device=feat.device
            )
            feature_exist_mask = torch.ones(1, dtype=torch.bool, device=feat.device)
            fused = self.audio_tower(
                speech_ids,
                continuous_audio_features=cont,
                feature_exist_mask=feature_exist_mask,
                return_dict=True,
            ).last_hidden_state  # [1, speech_maxlen/group_size, output_dim]

            # Clip to the actual output length (unpadded) per ref L1047-1050.
            results.append(fused[0, : int(output_lens.item()) // 1])

        return results

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        input_features = kwargs.get("input_features")
        if input_features is None:
            return []
        feature_attention_mask = kwargs.get("feature_attention_mask")
        return self._encode_audio_prefill(
            input_features, feature_attention_mask=feature_attention_mask
        )

    # ── Core forward / logits / post-sample flow ──────────────────────────────

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

        # Save raw input_ids for compute_logits (needed for text_embed = E[last_tok])
        self._last_input_ids = input_ids

        # Decode-time feedback: once speech is active and we have >=group_size
        # accumulated speech tokens, replace the text embedding with
        # (text_emb + audio_tower(last group_size speech_ids)) / 2 for this
        # request (ref L1196-1208, "double generate"). BS=1 only here.
        if (
            self._generate_speech
            and self._crq_state is not None
            and self._crq_state.speech_ids.numel() >= self.config.audio_config.group_size
            and input_ids.numel() == 1  # decode step (one token)
        ):
            inputs_embeds = self._build_speech_mode_inputs_embeds(input_ids)

        hidden_states = self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )
        return hidden_states

    def _build_speech_mode_inputs_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Build LM input for decode steps during active speech generation.

        Reference: modeling_funaudiochat.py L1201-1207.
        inputs_embeds = (embed(last_text_token) + audio_tower(speech_ids[-group_size:])) / 2
        """
        group_size = self.config.audio_config.group_size
        # text side: vllm's embed
        text_emb = self.language_model.model.embed_tokens(input_ids)
        if text_emb.dim() == 2:
            text_emb = text_emb.unsqueeze(0)  # [1, 1, H]
        # audio side: encode the last group_size speech ids
        recent = self._crq_state.speech_ids[:, -group_size:].to(text_emb.device)
        audio_embed = self.audio_tower(recent, return_dict=True).last_hidden_state
        # blend per ref (single_modal=False path)
        fused = (text_emb + audio_embed) / 2
        return fused.reshape(-1, fused.size(-1))

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        text_logits = self.language_model.compute_logits(hidden_states)

        ntokens = hidden_states.shape[0]
        device = hidden_states.device
        self._last_crq_tokens = None

        if ntokens > 1:
            # Prefill: (re)initialize CRQ state for this request.
            self._crq_state = CRQState(
                logits_processor=LogitsProcessorList(),
                do_sample=_DEFAULT_CRQ_DO_SAMPLE,
                speech_ids=torch.empty(1, 0, dtype=torch.long, device=device),
            )
            self._generate_speech = False
            self._speech_finished = False
            self._pending_force_text_abos = _DEFAULT_FORCE_TEXT_ABOS
            return text_logits

        if (
            ntokens == 1
            and self._crq_state is not None
            and self._generate_speech
            and not self._speech_finished
            and self._last_input_ids is not None
        ):
            # CRQ runs only when speech is active (strict gating — see plan v2).
            try:
                crq_hs = hidden_states.unsqueeze(0)  # [1, 1, H]
                last_tok = self._last_input_ids[-1:]
                text_embeds = self.language_model.model.embed_tokens(last_tok).unsqueeze(0)
                crq_input = crq_hs + text_embeds.detach()
                with torch.no_grad():
                    new_tokens, new_state = self.audio_invert_tower.crq_generate_forward(
                        crq_input, self._crq_state
                    )
                self._crq_state = new_state
                # Detect audio EOS: any sub-token equals crq_eos_token_id.
                finish = (new_tokens == self._crq_eos_token_id).any()
                if finish:
                    self._speech_finished = True
                else:
                    # Append only the new group of 5 to speech_ids history.
                    self._crq_state.speech_ids = torch.cat(
                        [self._crq_state.speech_ids.to(device), new_tokens.long()],
                        dim=-1,
                    )
                self._last_crq_tokens = new_tokens.flatten().detach().to(
                    dtype=torch.long, device="cpu"
                )
            except Exception as exc:
                logger.warning("fun_audio_chat: CRQ step failed: %s", exc)

        return text_logits

    # ── Optional: force_text_abos hook ────────────────────────────────────────
    # vllm-omni does not currently expose a per-step sampling override hook
    # in this file, so we emulate force_text_abos by seeding speech mode the
    # first time compute_logits sees a post-prefill step if pending is set.
    # NOTE: this is approximate; proper implementation belongs in the sampling
    # hook and is the primary O6 work.

    def _maybe_force_text_abos(self) -> None:
        if self._pending_force_text_abos and not self._generate_speech:
            self._generate_speech = True
            self._pending_force_text_abos = False

    # ── vllm-omni pipeline hooks: postprocess + make_omni_output ──────────────

    def postprocess(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: Any = None,
        **req_infos: Any,
    ) -> dict[str, Any]:
        """Accumulate CRQ tokens into `model_intermediate_buffer[req_id]["crq_tokens"]`.

        Also flip `generate_speech` on if the most recent text token (stored
        before sampling in the vllm runner's loop) equals audio_bos_index.
        """
        # Determine if speech should start. The most recently *sampled* text
        # token is the second-to-last element of the runner's output_ids.
        # Without direct access here we fall back to inspecting the next
        # step's input_ids via _last_input_ids; this is what the reference
        # does in prepare_inputs_for_generation.
        if (
            not self._generate_speech
            and not self._speech_finished
            and self._last_input_ids is not None
            and self._audio_bos_index in self._last_input_ids
        ):
            self._generate_speech = True

        # force_text_abos: if enabled and speech hasn't started yet, flip now.
        if self._pending_force_text_abos and not self._generate_speech:
            self._generate_speech = True
            self._pending_force_text_abos = False

        new_tokens = self._last_crq_tokens
        if new_tokens is None or new_tokens.numel() == 0:
            return {}
        existing = req_infos.get("crq_tokens")
        if isinstance(existing, torch.Tensor) and existing.numel() > 0:
            accumulated = torch.cat([existing.flatten().cpu(), new_tokens.flatten()])
        else:
            accumulated = new_tokens.flatten()
        self._last_crq_tokens = None
        return {"crq_tokens": accumulated}

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput:
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

    # ── Weight loading ────────────────────────────────────────────────────────

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load checkpoint weights. Names match the reference layout 1:1.

        Skipped:
          - `audio_invert_tower.crq_transformer.embed_tokens.*` (ref deletes it)
          - `audio_tower.embed_tokens.*` is remapped through the tied weight
            established in __init__, so we load it into
            `audio_tower.embed_tokens.weight` directly.
        """
        _SKIP_PREFIXES = (
            "audio_invert_tower.crq_transformer.embed_tokens.",
        )
        filtered: list[tuple[str, torch.Tensor]] = []
        for name, param in weights:
            if any(name.startswith(p) for p in _SKIP_PREFIXES):
                continue
            filtered.append((name, param))

        loader = AutoWeightsLoader(self)
        return loader.load_weights(iter(filtered))
