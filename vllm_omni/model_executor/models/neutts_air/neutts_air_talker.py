# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NeuTTS-Air prompt processing backed by vLLM's native Qwen2 model."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.inputs import MultiModalInput as MultiModalInputs
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptUpdate,
    TimingContext,
)

NEUTTS_SPEECH_TOKEN_OFFSET = 151671
NEUTTS_CODEC_VOCAB_SIZE = 65536
NEUTTS_REFERENCE_SAMPLE_RATE = 16_000
DEFAULT_NEUCODEC_REPO = "neuphonic/neucodec"
NEUTTS_ESPEAK_LIBRARY_ENV = "NEUTTS_ESPEAK_LIBRARY"
NEUTTS_ESPEAK_DATA_PATH_ENV = "NEUTTS_ESPEAK_DATA_PATH"
EXPECTED_NEUTTS_ESPEAK_VERSION = (1, 52, 0, 1)

_TEXT_REPLACE = "<|TEXT_REPLACE|>"
_TEXT_PROMPT_START = "<|TEXT_PROMPT_START|>"
_TEXT_PROMPT_END = "<|TEXT_PROMPT_END|>"
_SPEECH_REPLACE = "<|SPEECH_REPLACE|>"
_SPEECH_GENERATION_START = "<|SPEECH_GENERATION_START|>"
_PROMPT_TEMPLATE = "user: Convert the text to speech:<|TEXT_REPLACE|>\nassistant:<|SPEECH_REPLACE|>"

_PHONEMIZER: Any | None = None
_PHONEMIZER_LOCK = Lock()
_REFERENCE_CODECS: dict[str, Any] = {}
_REFERENCE_CODEC_LOCK = Lock()


def _normalize_codes(ref_codes: Any) -> list[int]:
    if isinstance(ref_codes, torch.Tensor):
        values = ref_codes.detach().cpu().reshape(-1).tolist()
    elif isinstance(ref_codes, np.ndarray):
        values = ref_codes.reshape(-1).tolist()
    else:
        values = list(ref_codes)

    codes = [int(value) for value in values]
    invalid = [code for code in codes if not 0 <= code < NEUTTS_CODEC_VOCAB_SIZE]
    if invalid:
        raise ValueError(f"NeuTTS-Air reference codes must be in [0, 65535]; found {invalid[0]}.")
    if not codes:
        raise ValueError("NeuTTS-Air reference codes cannot be empty.")
    return codes


def _configure_explicit_espeak_assets(espeak_wrapper: Any) -> bool:
    """Configure exact NeuTTS eSpeak assets supplied through environment variables."""
    library = os.environ.get(NEUTTS_ESPEAK_LIBRARY_ENV, "").strip()
    data_path = os.environ.get(NEUTTS_ESPEAK_DATA_PATH_ENV, "").strip()

    if bool(library) != bool(data_path):
        raise RuntimeError(f"{NEUTTS_ESPEAK_LIBRARY_ENV} and {NEUTTS_ESPEAK_DATA_PATH_ENV} must be set together.")
    if not library:
        return False

    library_path = Path(library).expanduser()
    data_dir = Path(data_path).expanduser()
    if not library_path.is_file():
        raise RuntimeError(f"NeuTTS-Air eSpeak library does not exist: {library_path}")
    if not data_dir.is_dir():
        raise RuntimeError(f"NeuTTS-Air eSpeak data directory does not exist: {data_dir}")
    if not hasattr(espeak_wrapper, "set_data_path"):
        raise RuntimeError("The installed phonemizer does not support an explicit eSpeak data path.")

    espeak_wrapper.set_library(str(library_path))
    espeak_wrapper.set_data_path(str(data_dir))
    return True


def _get_english_phonemizer() -> Any:
    global _PHONEMIZER
    if _PHONEMIZER is not None:
        return _PHONEMIZER

    with _PHONEMIZER_LOCK:
        if _PHONEMIZER is not None:
            return _PHONEMIZER
        try:
            from phonemizer.backend import EspeakBackend
            from phonemizer.backend.espeak.wrapper import EspeakWrapper

            if not _configure_explicit_espeak_assets(EspeakWrapper):
                try:
                    import espeakng_loader

                    EspeakWrapper.set_library(espeakng_loader.get_library_path())
                    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
                except ImportError:
                    # A system espeak-ng installation is also supported.
                    pass

            phonemizer = EspeakBackend(
                language="en-us",
                preserve_punctuation=True,
                with_stress=True,
                words_mismatch="ignore",
                language_switch="remove-flags",
            )
            version = tuple(phonemizer.version())
            if version != EXPECTED_NEUTTS_ESPEAK_VERSION:
                raise RuntimeError(
                    "NeuTTS-Air requires the official eSpeak assets with version "
                    f"{EXPECTED_NEUTTS_ESPEAK_VERSION}; found {version}. Set "
                    f"{NEUTTS_ESPEAK_LIBRARY_ENV} and {NEUTTS_ESPEAK_DATA_PATH_ENV} "
                    "to the matching library and data directory."
                )
            _PHONEMIZER = phonemizer
        except ImportError as exc:
            raise RuntimeError("NeuTTS-Air text processing requires phonemizer and espeak-ng.") from exc
    return _PHONEMIZER


def phonemize_english(text: str) -> str:
    """Match NeuTTS-Air's English phoneme normalization."""
    phonemes = _get_english_phonemizer().phonemize([text])[0]
    return " ".join(phonemes.split())


def build_neutts_air_prompt_token_ids(
    tokenizer: Any,
    ref_codes: Any,
    ref_text: str,
    target_text: str,
    *,
    phonemize=phonemize_english,
) -> list[int]:
    """Build the exact prompt consumed by the official PyTorch backend."""
    if not ref_text.strip():
        raise ValueError("NeuTTS-Air reference text cannot be empty.")
    if not target_text.strip():
        raise ValueError("NeuTTS-Air target text cannot be empty.")

    codes = _normalize_codes(ref_codes)
    text = f"{phonemize(ref_text)} {phonemize(target_text)}"
    text_ids = tokenizer.encode(text, add_special_tokens=False)

    ids = list(tokenizer.encode(_PROMPT_TEMPLATE))
    text_replace = tokenizer.convert_tokens_to_ids(_TEXT_REPLACE)
    text_start = tokenizer.convert_tokens_to_ids(_TEXT_PROMPT_START)
    text_end = tokenizer.convert_tokens_to_ids(_TEXT_PROMPT_END)
    speech_replace = tokenizer.convert_tokens_to_ids(_SPEECH_REPLACE)
    speech_start = tokenizer.convert_tokens_to_ids(_SPEECH_GENERATION_START)

    text_replace_idx = ids.index(text_replace)
    ids = ids[:text_replace_idx] + [text_start] + list(text_ids) + [text_end] + ids[text_replace_idx + 1 :]

    speech_replace_idx = ids.index(speech_replace)
    speech_ids = [NEUTTS_SPEECH_TOKEN_OFFSET + code for code in codes]
    return ids[:speech_replace_idx] + [speech_start] + speech_ids


class NeuTTSAirProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        del seq_len, mm_counts
        # Reference audio becomes ordinary Qwen2 prompt token IDs, so there is
        # no model-side multimodal tower to profile.
        return {}

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(
            target_sr=NEUTTS_REFERENCE_SAMPLE_RATE,
            target_channels=1,
            expected_hidden_size=self._get_expected_hidden_size(),
        )


class NeuTTSAirDummyInputsBuilder(BaseDummyInputsBuilder[NeuTTSAirProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        del mm_counts
        return "This is a NeuTTS-Air test."

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        del seq_len
        count = mm_counts.get("audio", 0)
        if count <= 0:
            return {}
        overrides = mm_options.get("audio") if mm_options else None
        return {
            "audio": self._get_dummy_audios(
                length=NEUTTS_REFERENCE_SAMPLE_RATE,
                num_audios=count,
                overrides=overrides,
            )
        }

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> ProcessorInputs:
        inputs = super().get_dummy_processor_inputs(seq_len, mm_counts, mm_options)
        inputs.hf_processor_mm_kwargs = {
            "ref_text": "This is the reference voice.",
            "ref_codes": [0],
        }
        return inputs


class NeuTTSAirMultiModalProcessor(BaseMultiModalProcessor[NeuTTSAirProcessingInfo]):
    def _get_mm_fields_config(
        self,
        hf_inputs: Any,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        del hf_inputs, hf_processor_mm_kwargs
        return {}

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        del mm_items, hf_processor_mm_kwargs, out_mm_kwargs
        return []

    @staticmethod
    def _load_reference_codec(repo_id: str) -> Any:
        codec = _REFERENCE_CODECS.get(repo_id)
        if codec is not None:
            return codec

        with _REFERENCE_CODEC_LOCK:
            codec = _REFERENCE_CODECS.get(repo_id)
            if codec is not None:
                return codec
            try:
                from neucodec import NeuCodec
            except (ImportError, ModuleNotFoundError) as exc:
                raise RuntimeError("NeuTTS-Air reference-audio encoding requires NeuCodec.") from exc
            codec = NeuCodec.from_pretrained(repo_id).eval().cpu()
            _REFERENCE_CODECS[repo_id] = codec
            return codec

    def _encode_reference_audio(
        self,
        audio: np.ndarray,
        repo_id: str,
    ) -> list[int]:
        waveform = torch.from_numpy(np.asarray(audio)).float().reshape(1, 1, -1)
        codec = self._load_reference_codec(repo_id)
        with torch.inference_mode():
            codes = codec.encode_code(waveform)
        return _normalize_codes(codes)

    def _resolve_reference_codes(self, inputs: ProcessorInputs) -> list[int]:
        raw_codes = inputs.hf_processor_mm_kwargs.get("ref_codes")
        if raw_codes is not None:
            return _normalize_codes(raw_codes)

        audio_items = inputs.mm_data_items.get("audio")
        if audio_items is None or audio_items.get_count() != 1:
            raise ValueError(
                "NeuTTS-Air requires exactly one reference audio item or mm_processor_kwargs['ref_codes']."
            )
        repo_id = str(inputs.hf_processor_mm_kwargs.get("codec_repo", DEFAULT_NEUCODEC_REPO))
        return self._encode_reference_audio(audio_items.get_all()[0], repo_id)

    def apply(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> MultiModalInputs:
        if not isinstance(inputs.prompt, str):
            raise TypeError("NeuTTS-Air requires a string target-text prompt.")

        ref_text = inputs.hf_processor_mm_kwargs.get("ref_text")
        if not isinstance(ref_text, str):
            raise ValueError("NeuTTS-Air requires mm_processor_kwargs['ref_text'].")

        with timing_ctx.record("encode_reference_audio"):
            ref_codes = self._resolve_reference_codes(inputs)
        with timing_ctx.record("build_neutts_prompt"):
            prompt_ids = build_neutts_air_prompt_token_ids(
                self.info.ctx.tokenizer,
                ref_codes,
                ref_text,
                inputs.prompt,
            )

        # The reference audio has already become ordinary Qwen2 token IDs.
        return MultiModalInputs(
            type="multimodal",
            prompt_token_ids=prompt_ids,
            mm_kwargs=MultiModalKwargsItems({}),
            mm_hashes={},
            mm_placeholders={},
        )


@MULTIMODAL_REGISTRY.register_processor(
    NeuTTSAirMultiModalProcessor,
    info=NeuTTSAirProcessingInfo,
    dummy_inputs=NeuTTSAirDummyInputsBuilder,
)
class NeuTTSAirForCausalLM(Qwen2ForCausalLM, SupportsMultiModal):
    """NeuTTS-Air identity and processor binding over native vLLM Qwen2."""

    supports_multimodal_raw_input_only = True
    supports_multimodal = True
    requires_raw_input_tokens = True

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        """Delegate Omni's unified embedding hook to native vLLM Qwen2."""
        return super().embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> Any:
        """Delegate Omni's unified forward hook to native vLLM Qwen2."""
        return super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> None:
        del modality, i
        return None
