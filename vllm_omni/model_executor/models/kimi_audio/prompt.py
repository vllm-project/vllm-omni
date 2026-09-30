# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Kimi-Audio message organization and aligned text/audio prompt assembly.

The rules follow MoonshotAI/Kimi-Audio's KimiAPromptManager. Audio encoders
remain worker-owned: callers supply their results by original message index.
This module does not load models, resolve audio resources, or run generation.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from dataclasses import field as dataclass_field
from numbers import Integral
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.tokenizers.kimi_audio import KimiAudioTokenizer


@dataclass(frozen=True)
class KimiAudioSpecialTokens:
    msg_end: int
    media_begin: int
    media_end: int
    kimia_text_blank: int
    kimia_text_eos: int
    kimia_user_msg_start: int
    kimia_assistant_msg_start: int
    kimia_speech_ct_id: int
    kimia_speech_ctd_id: int

    @classmethod
    def from_vocab(cls, vocab: Mapping[str, int]) -> "KimiAudioSpecialTokens":
        """Resolve all required markers from the checkpoint's full vocabulary."""
        values = {}
        for field in fields(cls):
            name = f"<|im_{field.name}|>"
            token_id = vocab.get(name)
            if not isinstance(token_id, Integral) or isinstance(token_id, bool) or token_id < 0:
                raise ValueError(f"Missing or invalid Kimi-Audio special token: {name}")
            values[field.name] = int(token_id)
        if len(set(values.values())) != len(values):
            raise ValueError("Kimi-Audio special tokens must have distinct IDs")
        return cls(**values)


@dataclass(frozen=True)
class KimiAudioEncodedAudio:
    """Worker result for one message, before LLM embedding projection.

    ``codes`` are raw GLM codebook IDs, without ``kimia_token_offset``.
    ``continuous_features`` has shape [num_codes, kimia_adaptor_input_dim].
    It is required for ``audio`` and absent for delayed ``audio-text`` history.
    """

    codes: Sequence[int]
    continuous_features: torch.Tensor | None = None


@dataclass
class KimiAudioPrompt:
    """One unpadded prompt. Feature blocks follow the true mask runs in order.

    Lists are request-local. Feature tensors stay on the encoder's device and
    are borrowed read-only; this builder neither moves nor copies them.
    """

    text_token_ids: list[int]
    audio_token_ids: list[int]
    is_continuous_mask: list[bool]
    continuous_features: list[torch.Tensor]
    # Original message index -> [start, end) of GLM codes, excluding markers
    # and the leading audio-text delay. Used to reserve slots before encoding.
    audio_spans: dict[int, tuple[int, int]] = dataclass_field(default_factory=dict)


class KimiAudioPromptBuilder:
    """Build the two streams from structured messages and encoded audio.

    ``encode_text`` must encode ordinary text without BOS/EOS or interpreting
    marker-looking text as special tokens, matching the official tokenizer.
    It is injected separately from the marker vocabulary because the existing
    vLLM ASR tokenizer also interprets some markers embedded in text.

    Configuration comes from the selected checkpoint: ``audio_token_offset``
    is kimia_token_offset, ``audio_vocab_size`` is vocab_size minus that offset,
    and ``audio_delay`` is kimia_mimo_audiodelaytokens.
    """

    def __init__(
        self,
        encode_text: Callable[[str], Sequence[int]],
        special_tokens: KimiAudioSpecialTokens,
        *,
        audio_token_offset: int,
        audio_vocab_size: int,
        audio_delay: int,
        continuous_feature_size: int,
    ) -> None:
        if audio_token_offset <= 0 or audio_vocab_size <= 0 or continuous_feature_size <= 0 or audio_delay < 0:
            raise ValueError("Invalid Kimi-Audio input configuration")
        self.encode_text = encode_text
        self.tokens = special_tokens
        self.audio_token_offset = audio_token_offset
        self.audio_vocab_size = audio_vocab_size
        self.audio_delay = audio_delay
        self.continuous_feature_size = continuous_feature_size

    @classmethod
    def from_tokenizer(cls, tokenizer: "KimiAudioTokenizer", config: "PretrainedConfig") -> "KimiAudioPromptBuilder":
        """Bind an already-loaded vLLM Kimi tokenizer and checkpoint config.

        vLLM's public encode interprets embedded markers, and its public vocab
        omits some Kimi markers. Use its existing TikToken backend for ordinary
        text and explicit marker lookup; do not reload or modify the tokenizer.
        Keep this dependency on vLLM's private backend confined to this method.
        """
        encoding = tokenizer._tokenizer
        names = [f"<|im_{field.name}|>" for field in fields(KimiAudioSpecialTokens)]
        return cls(
            encode_text=encoding.encode_ordinary,
            special_tokens=KimiAudioSpecialTokens.from_vocab(
                {name: encoding.encode_single_token(name) for name in names}
            ),
            audio_token_offset=config.kimia_token_offset,
            audio_vocab_size=config.vocab_size - config.kimia_token_offset,
            audio_delay=config.kimia_mimo_audiodelaytokens,
            continuous_feature_size=config.kimia_adaptor_input_dim,
        )

    def build(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        audio_inputs: Mapping[int, KimiAudioEncodedAudio] | None = None,
        output_type: Literal["text", "both"] = "text",
        add_assistant_start_msg: bool = True,
    ) -> KimiAudioPrompt:
        """Use official role/message_type/content fields without mutating them.

        For ``audio-text``, content is (audio source, transcript). Audio sources
        are resolved and encoded outside this builder; ``audio_inputs[i]`` is
        the result for messages[i]. Pre-offset ``audio_tokens`` in messages are
        rejected, so the two code-ID conventions cannot be mixed silently.
        """
        if output_type not in ("text", "both"):
            raise ValueError("output_type must be 'text' or 'both'")
        if not messages and not add_assistant_start_msg:
            raise ValueError("An empty prompt requires an assistant start marker")
        audio_inputs = {} if audio_inputs is None else audio_inputs
        audio_indices = set()
        for i, message in enumerate(messages):
            if message.get("role") not in ("user", "assistant"):
                raise ValueError(f"messages[{i}]: role must be 'user' or 'assistant'")
            kind = message.get("message_type")
            if kind not in ("text", "audio", "audio-text"):
                raise ValueError(f"messages[{i}]: unsupported message_type {kind!r}")
            if "audio_tokens" in message:
                raise ValueError(f"messages[{i}]: supply raw GLM codes through audio_inputs instead of audio_tokens")
            if kind in ("audio", "audio-text"):
                audio_indices.add(i)
        if set(audio_inputs) != audio_indices:
            raise ValueError("audio_inputs must contain exactly the audio/audio-text message indices")

        result = KimiAudioPrompt([], [], [], [])
        blank = self.tokens.kimia_text_blank

        def append(audio_ids: Sequence[int], text_ids: Sequence[int], *, continuous: bool = False) -> None:
            if len(audio_ids) != len(text_ids):
                raise ValueError("Kimi-Audio text and audio streams must have equal lengths")
            result.audio_token_ids.extend(audio_ids)
            result.text_token_ids.extend(text_ids)
            result.is_continuous_mask.extend([continuous] * len(audio_ids))

        for i, message in enumerate(messages):
            role = message["role"]
            kind = message["message_type"]
            group_end = i == len(messages) - 1 or messages[i + 1]["role"] != role
            if i == 0 or messages[i - 1]["role"] != role:
                role_token = (
                    self.tokens.kimia_user_msg_start if role == "user" else self.tokens.kimia_assistant_msg_start
                )
                append([role_token], [blank])

            if kind == "text":
                text_ids = self._encode_text(message.get("content"), i)
                append([blank] * len(text_ids), text_ids)
                if role == "assistant":
                    append([blank], [self.tokens.kimia_text_eos])
            else:
                audio = audio_inputs[i]
                audio_ids = self._audio_ids(audio, i)
                if kind == "audio":
                    feature = audio.continuous_features
                    expected_shape = (len(audio_ids), self.continuous_feature_size)
                    if feature is None or tuple(feature.shape) != expected_shape or not feature.is_floating_point():
                        raise ValueError(f"messages[{i}]: continuous_features must be floating point {expected_shape}")
                    append([self.tokens.media_begin], [blank])
                    start = len(result.audio_token_ids)
                    append(audio_ids, [blank] * len(audio_ids), continuous=True)
                    result.audio_spans[i] = (start, len(result.audio_token_ids))
                    append([self.tokens.media_end], [blank])
                    result.continuous_features.append(feature)
                    if group_end:
                        task_token = (
                            self.tokens.kimia_speech_ct_id if output_type == "text" else self.tokens.kimia_speech_ctd_id
                        )
                        append([task_token], [blank])
                else:
                    if audio.continuous_features is not None:
                        raise ValueError(f"messages[{i}]: audio-text history does not use continuous features")
                    content = message.get("content")
                    if not isinstance(content, (tuple, list)) or len(content) != 2:
                        raise ValueError(f"messages[{i}]: audio-text content must be (audio source, transcript)")
                    text_ids = self._encode_text(content[1], i)
                    delayed_audio = [blank] * self.audio_delay + audio_ids
                    if len(text_ids) > len(delayed_audio):
                        raise ValueError(f"messages[{i}]: transcript exceeds the delayed audio stream length")
                    start = len(result.audio_token_ids) + self.audio_delay
                    append(delayed_audio, text_ids + [blank] * (len(delayed_audio) - len(text_ids)))
                    result.audio_spans[i] = (start, len(result.audio_token_ids))
            if group_end:
                append([self.tokens.msg_end], [blank])

        if add_assistant_start_msg:
            append([self.tokens.kimia_assistant_msg_start], [blank])
        return result

    def _encode_text(self, text: object, message_index: int) -> list[int]:
        if not isinstance(text, str):
            raise ValueError(f"messages[{message_index}]: text content must be a string")
        # Preserve the official text tokenizer's chunk boundaries. Even with
        # the same BPE vocabulary, encoding a long run in one call can differ.
        ids = []
        for chunk_start in range(0, len(text), 400_000):
            chunk = text[chunk_start : chunk_start + 400_000]
            start = run_length = 0
            in_whitespace = chunk[0].isspace()
            for i, char in enumerate(chunk):
                is_whitespace = char.isspace()
                run_length = run_length + 1 if is_whitespace == in_whitespace else 1
                in_whitespace = is_whitespace
                if run_length > 25_000:
                    ids.extend(self.encode_text(chunk[start:i]))
                    start, run_length = i, 1
            ids.extend(self.encode_text(chunk[start:]))
        if any(
            not isinstance(token, Integral) or isinstance(token, bool) or not 0 <= token < self.audio_token_offset
            for token in ids
        ):
            raise ValueError(f"messages[{message_index}]: invalid text token IDs")
        return [int(token) for token in ids]

    def _audio_ids(self, audio: KimiAudioEncodedAudio, message_index: int) -> list[int]:
        codes = list(audio.codes)
        if not codes or any(
            not isinstance(code, Integral) or isinstance(code, bool) or not 0 <= code < self.audio_vocab_size
            for code in codes
        ):
            raise ValueError(f"messages[{message_index}]: expected nonempty raw GLM codebook IDs")
        return [int(code) + self.audio_token_offset for code in codes]
