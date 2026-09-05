"""Breeze-TTS-2 request-to-prompt conversion.

This module is intentionally limited to request preparation.  It creates the
real prompt token ids needed by the scheduler and stores the text/audio masks
and reference codes in ``additional_information``.  The stage-0 model later
turns those values into embeddings; this keeps tokenization independent from
the Qwen3/text-encoder implementation and avoids changing sequence length in
the model ``preprocess`` hook.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from vllm_omni.inputs.data import OmniTokensPrompt

from .audio_tokenizer import BreezeReferenceAudioTokenizer

AUDIO_TAG = "<|AUDIO|>"
AUDIO_EOS = "<|audio_eos|>"
INSTRUCTION_BOS = "<ins_bos>"
INSTRUCTION_EOS = "<ins_eos>"


@dataclass(frozen=True)
class _TextSegment:
    text: str


@dataclass(frozen=True)
class _AudioSegment:
    source: Any
    sample_rate: int | None




class BreezeTTS2PromptBuilder:
    """Build one Breeze prompt without depending on model modules.

    ``tokenizer`` is a normal HuggingFace tokenizer.  ``reference_audio_encoder``
    is injected so the builder can use the serving worker's cached audio
    encoder and never reload it per request.  The returned ``OmniTokensPrompt``
    contains scheduler token ids and compact CPU metadata for stage-0
    ``preprocess``.
    """

    def __init__(
        self,
        tokenizer: Any,
        config: Any,
        reference_audio_encoder: BreezeReferenceAudioTokenizer | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.audio_encoder = reference_audio_encoder
        self.audio_token_id = int(config.audio_token_id)
        self.audio_eos_token_id = int(config.audio_eos_token_id)
        # The Breeze tokenizer vocabulary is larger than the Qwen3 backbone
        # vocabulary.  ``prompt_token_ids`` are scheduler bookkeeping only, so
        # keep them in the backbone vocabulary and retain the real Breeze ids
        # in ``additional_information['prompt_ids']`` for the talker.
        self.scheduler_token_id = int(config.pad_token_id)
        scheduler_vocab_size = int(config.backbone_config.vocab_size)
        if self.scheduler_token_id >= scheduler_vocab_size:
            raise ValueError(
                "prompt_pad_token_id must be smaller than the backbone vocabulary: "
                f"{self.scheduler_token_id} >= {scheduler_vocab_size}"
            )
        self.num_codebooks = int(config.num_codebooks)
        self.codebook_size = _resolve_codebook_size(config)
        if self.num_codebooks <= 0 or self.codebook_size <= 0:
            raise ValueError("num_codebooks and codec codebook_size must be positive")
        self._validate_audio_special_tokens()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Any,
        *,
        reference_audio_encoder: BreezeReferenceAudioTokenizer | None = None,
        tokenizer_kwargs: Mapping[str, Any] | None = None,
    ) -> BreezeTTS2PromptBuilder:
        """Load the text tokenizer once and construct a reusable builder."""
        kwargs = dict(tokenizer_kwargs or {})
        kwargs.setdefault("use_fast", True)
        kwargs.setdefault("trust_remote_code", True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        return cls(tokenizer, config, reference_audio_encoder)

    def build(
        self,
        request: Mapping[str, Any],
        *,
        template: str | None = None,
    ) -> OmniTokensPrompt:
        """Convert one request to a vLLM-Omni token prompt.

        Supported templates are ``tts_plain``, ``tts_instruction``,
        ``ref_clone_tata`` and ``ref_edit_tata``.  When omitted, the template
        is inferred from the presence of instruction/reference fields.
        """
        template_name = template or request.get("template")
        if not isinstance(template_name, str) or not template_name:
            template_name = _infer_template(request)
        segments = self._build_segments(template_name, request)

        # Breeze's reference implementation first adds tokenizer special
        # tokens to every text segment, decodes those segments back to text,
        # inserts the audio placeholders, and tokenizes the complete string
        # once.  This matters at segment boundaries: tokenizing each raw text
        # fragment independently can produce different boundary tokens.
        rendered_segments: list[tuple[str, bool, int]] = []
        reference_codes: list[torch.Tensor] = []

        for segment in segments:
            if isinstance(segment, _TextSegment):
                ids = self._encode_one_text(segment.text, add_special_tokens=True)
                if not ids:
                    raise ValueError("Breeze text tokenizer returned an empty segment")
                rendered = str(self.tokenizer.decode(ids, skip_special_tokens=False))
                rendered_segments.append((rendered, True, 0))
                continue

            codes = self._resolve_reference_codes(segment)
            reference_codes.append(codes)
            audio_text = AUDIO_TAG * int(codes.shape[0]) + AUDIO_EOS
            rendered_segments.append((audio_text, False, int(codes.shape[0]) + 1))

        prompt_ids: list[int] = []
        text_mask: list[bool] = []
        text_lengths: list[int] = []
        rendered_prompt = "".join(item[0] for item in rendered_segments)
        prompt_ids = self._encode_one_text(rendered_prompt, add_special_tokens=False)
        offset = 0
        for rendered, is_text, expected_len in rendered_segments:
            segment_ids = self._encode_one_text(rendered, add_special_tokens=False)
            actual_len = len(segment_ids)
            if not is_text and actual_len != expected_len:
                raise ValueError(
                    "Breeze audio placeholder token length changed after rendering: "
                    f"expected {expected_len}, got {actual_len}"
                )
            if not is_text and segment_ids != [
                *([self.audio_token_id] * (expected_len - 1)),
                self.audio_eos_token_id,
            ]:
                raise ValueError("Breeze tokenizer did not preserve audio placeholder ids")
            text_mask.extend([is_text] * actual_len)
            if is_text:
                text_lengths.append(actual_len)
            offset += actual_len
        if offset != len(prompt_ids) or len(text_mask) != len(prompt_ids):
            raise RuntimeError("Breeze rendered prompt length accounting error")
        if not prompt_ids:
            raise ValueError("Breeze prompt is empty")

        input_values = torch.cat(reference_codes, dim=0) if reference_codes else None
        if input_values is not None and int(input_values.shape[0]) != prompt_ids.count(self.audio_token_id):
            raise RuntimeError("reference code count does not match audio placeholder count")

        additional_information: dict[str, Any] = {
            # The model runner may invoke preprocess on a chunk of a long
            # prompt. Keep one compact CPU copy so the talker can reconstruct
            # the full embedding buffer before slicing that chunk.
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(prompt_ids), dtype=torch.bool),
            "text_ids_mask": torch.tensor(text_mask, dtype=torch.bool),
            "text_ids_len": torch.tensor(text_lengths, dtype=torch.long),
            "template": template_name,
        }
        if input_values is not None:
            # ``input_values`` is the name used by the upstream Breeze model.
            # Keep the tensor unbatched here: Omni transports one request at a
            # time and the talker adds the batch dimension after collation.
            additional_information["input_values"] = input_values
            additional_information["ref_code_len"] = int(input_values.shape[0])

        if self.scheduler_token_id < 0:
            raise ValueError("prompt_pad_token_id must be non-negative")

        return OmniTokensPrompt(
            # Never expose Breeze's high text/audio ids to the Qwen3 scheduler
            # or its token-range validation.  The real ids remain available in
            # additional_information and are consumed by ``preprocess``.
            prompt_token_ids=[self.scheduler_token_id] * len(prompt_ids),
            additional_information=additional_information,
        )

    def _build_segments(
        self, template: str, request: Mapping[str, Any]
    ) -> list[_TextSegment | _AudioSegment]:
        speaker = _speaker_prefix(request.get("speaker", "S0"))
        text = _required_text(request, "text")
        instruction = request.get("instruction")
        ref_text = request.get("ref_text")

        if template == "tts_plain":
            return [_TextSegment(f"{speaker}{text}")]
        if template == "tts_instruction":
            if not isinstance(instruction, str) or not instruction:
                raise ValueError("tts_instruction requires a non-empty instruction")
            return [_TextSegment(f"{speaker}{INSTRUCTION_BOS}{instruction}{INSTRUCTION_EOS}{text}")]
        if template in ("ref_clone_tata", "ref_edit_tata"):
            if not isinstance(ref_text, str) or not ref_text:
                raise ValueError(f"{template} requires a non-empty ref_text")
            audio = self._audio_segment(request)
            if template == "ref_clone_tata":
                return [_TextSegment(f"{speaker}{ref_text}"), audio, _TextSegment(f"{speaker}{text}")]
            if not isinstance(instruction, str) or not instruction:
                raise ValueError("ref_edit_tata requires a non-empty instruction")
            return [
                _TextSegment(f"{speaker}{ref_text}"),
                audio,
                _TextSegment(f"{speaker}{INSTRUCTION_BOS}{instruction}{INSTRUCTION_EOS}{text}"),
            ]
        raise ValueError(
            f"unknown Breeze template {template!r}; expected tts_plain, tts_instruction, "
            "ref_clone_tata or ref_edit_tata"
        )

    def _audio_segment(self, request: Mapping[str, Any]) -> _AudioSegment:
        source = request.get("ref_audio")
        if source is None:
            source = request.get("ref_audio_codes")
        if source is None:
            raw_codes = request.get("codes")
            source = raw_codes.get("ref") if isinstance(raw_codes, Mapping) else None
        if source is None:
            raise ValueError("reference template requires ref_audio or ref_audio_codes")
        sample_rate = request.get("ref_audio_sample_rate")
        return _AudioSegment(source, int(sample_rate) if sample_rate is not None else None)

    def _resolve_reference_codes(self, segment: _AudioSegment) -> torch.Tensor:
        source = segment.source
        if _looks_like_codes(source, self.num_codebooks):
            codes = torch.as_tensor(source)
            return _normalize_codes(codes, self.num_codebooks, self.codebook_size)
        if self.audio_encoder is None:
            raise RuntimeError("reference audio was provided but no reference_audio_encoder is configured")
        codes = self.audio_encoder.encode(source, segment.sample_rate)
        return _normalize_codes(codes, self.num_codebooks, self.codebook_size)

    def _encode_one_text(self, text: str, *, add_special_tokens: bool) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=add_special_tokens, padding=False)
        ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return _as_int_list(ids)

    def _validate_audio_special_tokens(self) -> None:
        convert = self.tokenizer.convert_tokens_to_ids
        for token, expected in ((AUDIO_TAG, self.audio_token_id), (AUDIO_EOS, self.audio_eos_token_id)):
            actual = convert(token)
            unk = getattr(self.tokenizer, "unk_token_id", None)
            if actual is None or (unk is not None and actual == unk) or int(actual) != expected:
                raise ValueError(f"tokenizer id for {token!r} is {actual}, expected config value {expected}")

def _infer_template(request: Mapping[str, Any]) -> str:
    has_ref = (
        request.get("ref_audio") is not None
        or request.get("ref_audio_codes") is not None
        or bool(request.get("ref_text"))
    )
    has_instruction = bool(request.get("instruction"))
    if has_ref and has_instruction:
        return "ref_edit_tata"
    if has_ref:
        return "ref_clone_tata"
    return "tts_instruction" if has_instruction else "tts_plain"


def _required_text(request: Mapping[str, Any], key: str) -> str:
    value = request.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Breeze request requires non-empty {key}")
    return value


def _speaker_prefix(value: Any) -> str:
    if value in (None, ""):
        return ""
    value = str(value)
    return value if value.startswith("[") and value.endswith("]") else f"[{value}]"


def _as_int_list(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError("tokenizer input_ids must be a list")
    if value and isinstance(value[0], (list, tuple, np.ndarray, torch.Tensor)):
        if len(value) != 1:
            raise ValueError("tokenizer input_ids contains multiple sequences")
        return _as_int_list(value[0])
    return [int(item) for item in value]


def _looks_like_codes(value: Any, num_codebooks: int) -> bool:
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return value.ndim == 2 and value.shape[-1] == num_codebooks
    return False


def _normalize_codes(codes: Any, num_codebooks: int, codebook_size: int) -> torch.Tensor:
    codes = torch.as_tensor(codes)
    if codes.ndim != 2:
        raise ValueError(f"reference audio codes must be 2D, got {tuple(codes.shape)}")
    if codes.shape[0] == 0:
        raise ValueError("reference audio codes must contain at least one frame")
    if codes.shape[-1] != num_codebooks:
        raise ValueError(f"expected {num_codebooks} codebooks, got {tuple(codes.shape)}")
    codes = codes.to(device="cpu", dtype=torch.long).contiguous()
    if int(codes.min()) < 0 or int(codes.max()) >= codebook_size:
        raise ValueError(f"reference code outside [0, {codebook_size})")
    return codes.to(dtype=torch.int16)


def _resolve_codebook_size(config: Any) -> int:
    return int(config.codec_config["codebook_size"])


__all__ = ["BreezeTTS2PromptBuilder"]
