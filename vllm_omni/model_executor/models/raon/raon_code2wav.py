# SPDX-License-Identifier: Apache-2.0

"""Raon stage-1: Mimi codec → waveform (one-shot and streaming)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from transformers import MimiConfig
from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.inputs import PromptType, TokensPrompt
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsPP,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.model_executor.models.whisper import ISO639_1_SUPPORTED_LANGS
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.raon.raon_audio_encoder import (
    compute_num_audio_input_tokens,
    compute_samples_per_frame,
)
from vllm_omni.model_executor.models.raon.raon_audio_tokenizer import (
    StreamingMimiDecoderOutput,
    StreamingMimiModel,
)
from vllm_omni.model_executor.models.raon.raon_utils import (
    IGNORE_WEIGHT_SUFFIXES,
    cfg_get,
    module_device,
    normalize_runtime_request_id,
    strip_raon_audio_markers,
    unwrap_singleton_list,
)
from vllm_omni.model_executor.models.raon.raon_utils import (
    get_placeholder_str as _get_placeholder_str,
)
from vllm_omni.tokenizers.raon_tokenizer import (
    align_tokenizer,
)
from vllm_omni.transformers_utils.configs.raon import (
    REQUEST_STATE_CLEANUP_KEYS,
    REQUEST_STATE_CLEANUP_PREFIXES,
)

logger = init_logger(__name__)


class RaonCode2WavModel(
    nn.Module,
    SupportsPP,
    CustomProcessMixin,
):
    """Codec codes to waveform via Mimi."""

    _STAGE1_ALIASES = {"stage1", "decode", "code2wav", "codec2wav"}
    _STAGE1_WEIGHT_PREFIX = "audio_tokenizer."
    _STAGE1_SKIP_PREFIXES = (
        "text_model.",
        "talker.",
        "lm_head.",
        "audio_lm_head.",
        "text_output_norm.",
        "thinker_to_talker_proj.",
        "logits_processor.",
        "proj_code.",
        "proj_speaker_code.",
        "output_adaptor.",
        "code_predictor.",
        "audio_encoder.",
        "input_adaptor.",
        "speaker_encoder.",
    )
    _TRACKED_WEIGHT_PREFIXES = (
        "text_model.",
        "talker.",
        "lm_head.",
        "audio_lm_head.",
        "text_output_norm.",
        "thinker_to_talker_proj.",
        "audio_encoder.",
        "input_adaptor.",
        "output_adaptor.",
        "proj_code.",
        "proj_speaker_code.",
        "code_predictor.",
        "speaker_encoder.",
        "audio_tokenizer.",
    )
    _IGNORE_WEIGHT_SUFFIXES = IGNORE_WEIGHT_SUFFIXES
    supported_languages = ISO639_1_SUPPORTED_LANGS

    # Worker cleanup: strip these keys from per-request additional_information on finish.
    request_state_cleanup_keys: tuple[str, ...] = REQUEST_STATE_CLEANUP_KEYS
    request_state_cleanup_prefixes: tuple[str, ...] = REQUEST_STATE_CLEANUP_PREFIXES

    @classmethod
    def _resolve_audio_tokenizer_config(cls, config: object) -> object:
        audio_cfg = cfg_get(config, "audio_tokenizer_config")
        if audio_cfg is not None:
            return audio_cfg

        logger.warning("hf_config.audio_tokenizer_config missing; treating stage hf_config as Mimi config.")
        return config

    @classmethod
    def _resolve_num_code_groups(cls, config: object, audio_tokenizer_config: object) -> int:
        code_predictor_cfg = cfg_get(config, "code_predictor_config")
        if code_predictor_cfg is not None:
            num_code_groups = cfg_get(code_predictor_cfg, "num_code_groups")
            if num_code_groups is not None:
                return int(num_code_groups)

        for fallback_key in ("num_codebooks", "num_quantizers"):
            fallback_value = cfg_get(audio_tokenizer_config, fallback_key)
            if fallback_value is not None:
                logger.warning(
                    "code_predictor_config missing; using %s=%s as num_code_groups.",
                    fallback_key,
                    fallback_value,
                )
                return int(fallback_value)

        logger.warning("code_predictor_config missing and no Mimi group metadata found; defaulting num_code_groups=32.")
        return 32

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix

        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config

        model_stage = str(getattr(vllm_config.model_config, "model_stage", "stage1")).lower()
        if model_stage not in self._STAGE1_ALIASES:
            logger.warning(
                "Expected stage1/code2wav model_stage, got %s; continuing with stage1 decode.",
                model_stage,
            )
        self.model_stage = "stage1"

        # Omni runner flags for this stage.
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.requires_raw_input_tokens = True

        # Sample rate, frame rate, and RVQ layout from Mimi config.
        audio_tokenizer_config = self._resolve_audio_tokenizer_config(self.config)
        self.sampling_rate = int(cfg_get(audio_tokenizer_config, "sampling_rate"))
        frame_rate_value = cfg_get(audio_tokenizer_config, "_frame_rate")
        if frame_rate_value is None:
            frame_rate_value = cfg_get(audio_tokenizer_config, "frame_rate")
        if frame_rate_value is None:
            raise ValueError("RaonCode2Wav requires audio tokenizer frame rate metadata ('_frame_rate').")
        self.frame_rate = float(frame_rate_value)
        self.samples_per_frame = compute_samples_per_frame(
            sampling_rate=self.sampling_rate,
            frame_rate=self.frame_rate,
        )
        self.codebook_size = int(cfg_get(audio_tokenizer_config, "codebook_size"))
        self.num_code_groups = self._resolve_num_code_groups(self.config, audio_tokenizer_config)

        # Placeholder embed dim and audio pad token ids (thinker vocab alignment).
        text_model_config = cfg_get(self.config, "text_model_config")
        hidden_size = cfg_get(text_model_config, "hidden_size")
        self.hidden_size = int(hidden_size) if hidden_size is not None else 1
        self.vocab_size = 0
        self.text_vocab_size = 0
        self.audio_output_token_id = int(cfg_get(self.config, "audio_output_token_id", self.codebook_size))
        audio_input_token_id = cfg_get(self.config, "audio_input_token_id")
        if audio_input_token_id is None:
            audio_input_token_id = self.audio_output_token_id + 1
        self.audio_input_token_id = int(audio_input_token_id)

        # Mimi decoder only (stage-1 weights under audio_tokenizer.*).
        self.dtype = (
            vllm_config.model_config.dtype
            if isinstance(vllm_config.model_config.dtype, torch.dtype)
            else torch.bfloat16
        )
        mimi_cfg = audio_tokenizer_config
        if isinstance(mimi_cfg, dict):
            mimi_cfg = MimiConfig(**mimi_cfg)
        elif not isinstance(mimi_cfg, MimiConfig) and hasattr(mimi_cfg, "to_dict"):
            mimi_cfg = MimiConfig(**mimi_cfg.to_dict())
        self.audio_tokenizer = StreamingMimiModel._from_config(mimi_cfg, dtype=self.dtype)
        self.make_empty_intermediate_tensors = lambda *args, **kwargs: IntermediateTensors({})

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        return _get_placeholder_str(modality, i)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        del kwargs
        return []

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del multimodal_embeddings, is_multimodal
        return torch.zeros((*input_ids.shape, self.hidden_size), dtype=self.dtype, device=input_ids.device)

    def _normalize_stage1_codec_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Normalize codec codes to [B, T, G] for Mimi.decode."""
        G = self.num_code_groups

        if codes.ndim == 3:
            if codes.shape[-1] == G:
                return codes
            if codes.shape[1] == G:
                return codes.transpose(1, 2)
            codes = codes.reshape(-1)

        if codes.ndim == 2:
            if codes.shape[-1] == G:
                return codes.unsqueeze(0)
            if codes.shape[0] == G:
                return codes.transpose(0, 1).unsqueeze(0)
            codes = codes.reshape(-1)

        if codes.ndim == 1:
            flat = codes.reshape(-1)
            if flat.numel() % G != 0:
                pad = G - (flat.numel() % G)
                logger.warning(
                    "Codec length %d is not divisible by num_code_groups=%d; padding %d zeros.",
                    flat.numel(),
                    G,
                    pad,
                )
                flat = torch.cat([flat, torch.zeros(pad, dtype=torch.long, device=flat.device)], dim=0)
            return flat.reshape(-1, G).unsqueeze(0)

        raise ValueError(f"Unsupported codec code shape: {tuple(codes.shape)}")

    def _resolve_stage1_codec_codes(
        self,
        input_ids: torch.Tensor | None,
        req_info: dict[str, Any] | None,
    ) -> torch.Tensor:
        candidate: Any = None
        if isinstance(req_info, dict):
            candidate = req_info.get("codec_codes")

        candidate = unwrap_singleton_list(candidate)
        if candidate is None:
            candidate = input_ids
        candidate = unwrap_singleton_list(candidate)
        if candidate is None:
            raise ValueError("Stage-1 Raon requires codec codes in input_ids or additional info.")

        device = module_device(self.audio_tokenizer)
        if isinstance(candidate, torch.Tensor):
            codes = candidate.to(device=device, dtype=torch.long)
        else:
            codes = torch.as_tensor(candidate, dtype=torch.long, device=device)

        return self._normalize_stage1_codec_codes(codes)

    def _build_padding_mask(self, codec_codes: torch.Tensor) -> torch.Tensor:
        """Valid-sample mask for Mimi decode (ConvTranspose tail trim)."""
        num_frames = codec_codes.shape[1] if codec_codes.ndim >= 2 else codec_codes.shape[0]
        audio_len = int(num_frames * self.samples_per_frame)
        return torch.arange(audio_len, device=codec_codes.device).unsqueeze(0) < audio_len

    def _decode_stage1_audio(self, codec_codes: torch.Tensor) -> torch.Tensor:
        """Mimi decode without streaming KV/conv caches."""
        padding_mask = self._build_padding_mask(codec_codes)
        outputs = self.audio_tokenizer.decode(
            codec_codes.transpose(1, 2),
            padding_mask=padding_mask.long(),
            return_dict=True,
        )
        if isinstance(outputs, StreamingMimiDecoderOutput):
            audio_values = outputs.audio_values
        elif isinstance(outputs, tuple):
            audio_values = outputs[0]
        elif isinstance(outputs, Mapping):
            audio_values = outputs.get("audio_values")
        else:
            audio_values = getattr(outputs, "audio_values", None)
        if audio_values is None:
            raise RuntimeError("Mimi decode returned no audio_values.")
        audio_values = audio_values.clamp(-1.0, 1.0)
        return audio_values.reshape(audio_values.shape[0], -1)

    def _resolve_stage1_streaming_info(
        self,
        req_info: dict[str, Any] | None,
    ) -> tuple[str | None, bool, bool]:
        """Parse (req_id, is_finished, flush_only) from runtime info."""
        if not isinstance(req_info, dict):
            return None, False, False

        finished_val = req_info.get("finished")
        is_finished = False
        if finished_val is not None:
            is_finished = bool(finished_val.item() if isinstance(finished_val, torch.Tensor) else finished_val)
        flush_only = bool(req_info.get("flush_only", False))

        req_id = normalize_runtime_request_id(req_info.get("global_request_id"))
        if req_id is None:
            req_id = normalize_runtime_request_id(req_info.get("_omni_req_id"))

        if req_id is not None:
            return req_id, is_finished, flush_only
        return None, False, False

    def _clear_streaming_state(self, req_id: str) -> None:
        """No-op hook: async path uses stateless Mimi decode per window."""
        del req_id

    @staticmethod
    def _empty_stage1_output() -> OmniOutput:
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs=None,
        )

    @staticmethod
    def _split_input_ids_by_request(
        input_ids: torch.Tensor | None,
        seq_token_counts: list[int] | None,
        num_reqs: int,
    ) -> list[torch.Tensor | None]:
        if input_ids is None or not seq_token_counts or len(seq_token_counts) != num_reqs:
            return [None] * num_reqs

        per_req_input_ids: list[torch.Tensor | None] = []
        offset = 0
        for count in seq_token_counts:
            if count > 0:
                per_req_input_ids.append(input_ids[offset : offset + count])
            else:
                per_req_input_ids.append(None)
            offset += count
        return per_req_input_ids

    def _forward_stage1(
        self,
        input_ids: torch.Tensor | None,
        runtime_additional_information: list[dict[str, Any]] | None,
        seq_token_counts: list[int] | None = None,
    ) -> OmniOutput:
        """Decode codec payload(s) to audio; supports batched runtime_info."""
        request_infos = runtime_additional_information or [None]
        num_reqs = len(request_infos)

        if num_reqs <= 1:
            req_info = request_infos[0]
            if isinstance(req_info, dict):
                return self._forward_stage1_single(input_ids, req_info)
            return self._forward_stage1_single(input_ids, None)

        per_req_input_ids = self._split_input_ids_by_request(input_ids, seq_token_counts, num_reqs)

        audio_list: list[torch.Tensor | None] = []
        sr_list: list[torch.Tensor | None] = []
        default_sr = torch.tensor(self.sampling_rate)
        empty_audio = torch.zeros(0)

        for i in range(num_reqs):
            req_info = request_infos[i]
            if not isinstance(req_info, dict):
                req_info = None

            result = self._forward_stage1_single(per_req_input_ids[i], req_info)
            mm = result.multimodal_outputs
            if mm is None:
                audio_list.append(empty_audio)
                sr_list.append(default_sr)
                continue

            if not isinstance(mm, dict):
                audio_list.append(empty_audio)
                sr_list.append(default_sr)
                continue

            audio_tensor = mm.get("model_outputs")
            sample_rate = mm.get("sr")
            audio_list.append(audio_tensor if audio_tensor is not None else empty_audio)
            sr_list.append(sample_rate if sample_rate is not None else default_sr)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": audio_list,
                "sr": sr_list,
            },
        )

    @staticmethod
    def _trim_left_context_audio(
        audio: torch.Tensor,
        left_context_frames: int,
        total_frames: int,
    ) -> torch.Tensor:
        """Trim the left-context prefix (proportional to frame counts; Mimi length mismatch)."""
        if left_context_frames <= 0 or total_frames <= 0:
            return audio
        cut = int(left_context_frames / total_frames * audio.shape[-1])
        if cut >= audio.shape[-1]:
            logger.warning(
                "Left-context trim %d >= decoded audio length %d; returning empty.",
                cut,
                audio.shape[-1],
            )
            return audio[..., :0]
        return audio[..., cut:]

    def _trim_right_padding_audio(
        self,
        audio: torch.Tensor,
        total_frames: int,
    ) -> torch.Tensor:
        """Clip trailing audio past ``total_frames * samples_per_frame`` (decode tail noise)."""
        if total_frames <= 0 or audio.numel() == 0:
            return audio

        expected_len = int(total_frames * self.samples_per_frame)
        if expected_len <= 0 or audio.shape[-1] <= expected_len:
            return audio

        return audio[..., :expected_len]

    def _forward_stage1_single(
        self,
        input_ids: torch.Tensor | None,
        req_info: dict[str, Any] | None,
    ) -> OmniOutput:
        """Decode one request's codec codes to waveform."""
        stream_req_id, stream_finished, flush_only = self._resolve_stage1_streaming_info(req_info)

        if stream_req_id is not None and stream_finished and flush_only:
            self._clear_streaming_state(stream_req_id)
            return self._empty_stage1_output()

        if stream_req_id is not None:
            has_codec = input_ids is not None and input_ids.numel() > 0
            has_codec_info = isinstance(req_info, dict) and req_info.get("codec_codes") is not None
            if not has_codec and not has_codec_info:
                return self._empty_stage1_output()

        codec_codes = self._resolve_stage1_codec_codes(input_ids, req_info)

        left_context_size = 0
        if isinstance(req_info, dict):
            left_context_size = int(req_info.get("left_context_size", 0))

        total_frames = int(codec_codes.shape[1]) if codec_codes.ndim >= 2 else 0

        if stream_req_id is not None:
            # Stateless decode per window: streaming KV would double-apply overlap vs Stage-0.
            audio_tensor = self._decode_stage1_audio(codec_codes)
            if stream_finished:
                self._clear_streaming_state(stream_req_id)
            if audio_tensor is None:
                return self._empty_stage1_output()
        else:
            audio_tensor = self._decode_stage1_audio(codec_codes)

        if total_frames > 0 and audio_tensor.numel() > 0:
            audio_tensor = self._trim_right_padding_audio(audio_tensor, total_frames)

        if left_context_size > 0 and audio_tensor.numel() > 0:
            audio_tensor = self._trim_left_context_audio(
                audio_tensor,
                left_context_size,
                total_frames,
            )

        # Drop leading audio from Stage-0 silence bootstrap (first chunk only).
        csf = int(req_info.get("continuation_silence_frames", 0)) if req_info else 0
        if csf > 0 and audio_tensor.numel() > 0:
            samples_per_frame = audio_tensor.shape[-1] // max(1, total_frames - left_context_size)
            trim_samples = csf * samples_per_frame
            if 0 < trim_samples < audio_tensor.shape[-1]:
                audio_tensor = audio_tensor[..., trim_samples:]

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": audio_tensor,
                "sr": torch.tensor(self.sampling_rate, dtype=torch.int, device=audio_tensor.device),
            },
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        del positions, intermediate_tensors, inputs_embeds
        return self._forward_stage1(
            input_ids,
            runtime_additional_information,
            seq_token_counts=kwargs.get("seq_token_counts"),
        )

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | IntermediateTensors | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput | IntermediateTensors:
        del kwargs
        if isinstance(model_outputs, (OmniOutput, IntermediateTensors)):
            return model_outputs

        if isinstance(model_outputs, torch.Tensor):
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": model_outputs,
                    "sr": torch.tensor(self.sampling_rate, dtype=torch.int, device=model_outputs.device),
                },
            )
        raise ValueError(f"Unsupported stage1 output type: {type(model_outputs)!r}")

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> None:
        del hidden_states, sampling_metadata
        return None

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        del logits, sampling_metadata
        return None

    @classmethod
    def get_speech_to_text_config(
        cls,
        model_config: ModelConfig,
        task_type: Literal["transcribe", "translate"],
    ) -> SpeechToTextConfig:
        del task_type
        audio_tokenizer_cfg = cls._resolve_audio_tokenizer_config(model_config.hf_config)
        sampling_rate = int(cfg_get(audio_tokenizer_cfg, "sampling_rate"))
        return SpeechToTextConfig(
            sample_rate=float(sampling_rate),
            max_audio_clip_s=None,
            min_energy_split_window_size=None,
        )

    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
    ) -> int | None:
        audio_tokenizer_cfg = cls._resolve_audio_tokenizer_config(model_config.hf_config)
        frame_rate = cfg_get(audio_tokenizer_cfg, "_frame_rate")
        if frame_rate is None:
            frame_rate = cfg_get(audio_tokenizer_cfg, "frame_rate")
        if frame_rate is None:
            raise ValueError("RaonCode2Wav get_num_audio_tokens requires audio tokenizer frame rate metadata.")
        audio_len_samples = int(round(audio_duration_s * stt_config.sample_rate))
        return compute_num_audio_input_tokens(
            audio_len_samples,
            sampling_rate=int(stt_config.sample_rate),
            frame_rate=float(frame_rate),
        )

    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType:
        del stt_config
        tokenizer = cached_tokenizer_from_config(model_config)
        align_tokenizer(tokenizer)

        if task_type == "translate":
            target_lang = to_language or "en"
            target_name = cls.supported_languages.get(target_lang, target_lang)
            instruction = f"Translate this audio into {target_name}."
        else:
            instruction = "Transcribe this audio."
            if language:
                lang_name = cls.supported_languages.get(language, language)
                instruction = f"Transcribe this audio in {lang_name}."

        if request_prompt:
            instruction = f"{instruction} {request_prompt}"

        from vllm_omni.tokenizers.raon_tokenizer import (
            RaonChatTemplateBuilder,
            TaskType,
        )

        builder = RaonChatTemplateBuilder(tokenizer=tokenizer)
        result = builder.build_prompt(
            task=TaskType.STT,
            audio_count=1,
            instruction=instruction,
        )
        prompt = result.prompt_text

        hf_config = model_config.hf_config
        audio_tokenizer_cfg = cls._resolve_audio_tokenizer_config(hf_config)
        text_model_cfg = cfg_get(hf_config, "text_model_config")
        text_vocab_size = int(cfg_get(text_model_cfg, "vocab_size", 0)) - int(
            cfg_get(audio_tokenizer_cfg, "codebook_size")
        )
        return TokensPrompt(
            prompt_token_ids=tokenizer.encode(prompt),
            # Audio is already at model SR from STT preprocess.
            multi_modal_data={"audio": np.asarray(audio, dtype=np.float32)},
            additional_information={
                "output_mode": ["text_only"],
                "text_vocab_size": [text_vocab_size],
            },
        )

    @classmethod
    def post_process_output(cls, text: str) -> str:
        del cls
        return strip_raon_audio_markers(text)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=list(self._STAGE1_SKIP_PREFIXES),
            ignore_unexpected_suffixes=list(self._IGNORE_WEIGHT_SUFFIXES),
        )
        seen_counts: dict[str, int] = defaultdict(int)

        def _stage1_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for name, tensor in weights:
                seen_counts["__total__"] += 1
                for prefix in self._TRACKED_WEIGHT_PREFIXES:
                    if name.startswith(prefix):
                        seen_counts[prefix] += 1
                        break
                if name.startswith(self._STAGE1_WEIGHT_PREFIX):
                    yield name, tensor

        loaded = loader.load_weights(_stage1_weights())

        loaded_counts: dict[str, int] = defaultdict(int)
        for name in loaded:
            for prefix in self._TRACKED_WEIGHT_PREFIXES:
                if name.startswith(prefix):
                    loaded_counts[prefix] += 1
                    break

        logger.warning(
            "Loaded weights: stage=%s ckpt_total=%d loaded=%d audio_tokenizer=%d/%d",
            self.model_stage,
            int(seen_counts.get("__total__", 0)),
            int(len(loaded)),
            int(loaded_counts.get(self._STAGE1_WEIGHT_PREFIX, 0)),
            int(seen_counts.get(self._STAGE1_WEIGHT_PREFIX, 0)),
        )

        return loaded
