# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
import warnings
from typing import overload

from vllm_omni.model_executor.models.indextts2.utils.common import (
    de_tokenized_by_CJK_char,
)

# Chinese punctuation → ASCII punctuation mapping used by the official
# IndexTTS-2 inference pipeline (indextts/utils/front.py).  The BPE
# vocabulary contains English punctuation tokens (``.``, ``!``, ``?``)
# which the model was trained on; feeding raw Chinese punctuation
# produces incorrect subword tokenization and garbled pronunciation
# (e.g. '。' read as '哦').  Normalise before tokenisation.
_CHAR_REP_MAP = {
    "\u3002": ".",  # 。
    "\uff01": "!",  # ！
    "\uff1f": "?",  # ？
    "\uff1a": ",",  # ：
    "\uff1b": ",",  # ；
    "\uff0c": ",",  # ，
    "\u201c": "'",  # "
    "\u201d": "'",  # "
    "\u2018": "'",  # '
    "\u2019": "'",  # '
    "\uff08": "'",  # （
    "\uff09": "'",  # ）
    "\u300a": "'",  # 《
    "\u300b": "'",  # 》
    "\u3010": "'",  # 【
    "\u3011": "'",  # 】
    "\u2014": " ",  # —
    "\uff5e": " ",  # ～
    "\u00b7": "-",  # ·
    "\u3001": ",",  # 、
    "\n": " ",
    ";": ",",
    ":": ",",
}

_CHAR_REP_PATTERN = re.compile("|".join(re.escape(p) for p in _CHAR_REP_MAP.keys()))


def normalize_text(text: str) -> str:
    """Normalise Chinese punctuation to ASCII equivalents.

    This is a lightweight replacement for the ``TextNormalizer`` used in
    the official index-tts-vllm pipeline.  It strips Chinese punctuation
    marks and replaces them with the ASCII tokens the BPE model was
    trained on.  Full text normalisation (numbers, dates, currency, etc.)
    is intentionally omitted here — it should be handled by the caller
    when needed.
    """
    return _CHAR_REP_PATTERN.sub(lambda m: _CHAR_REP_MAP[m.group()], text)


class TextTokenizer:
    @property
    def vocab_size(self):
        return self.sp_model.GetPieceSize()

    @property
    def unk_token(self):
        return "<unk>"

    @property
    def pad_token(self):
        return None

    @property
    def bos_token(self):
        return "<s>"

    @property
    def eos_token(self):
        return "</s>"

    @property
    def pad_token_id(self):
        return -1

    @property
    def bos_token_id(self):
        return 0

    @property
    def eos_token_id(self):
        return 1

    @property
    def unk_token_id(self):
        return self.sp_model.unk_id()

    @property
    def special_tokens_map(self):
        return {
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        return vocab

    @overload
    def convert_ids_to_tokens(self, ids: int) -> str: ...

    @overload
    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]: ...

    def convert_ids_to_tokens(self, ids: list[int] | int):
        return self.sp_model.IdToPiece(ids)

    def convert_tokens_to_ids(self, tokens: list[str] | str) -> list[int]:
        if isinstance(tokens, str):
            tokens = [tokens]
        return [self.sp_model.PieceToId(token) for token in tokens]

    def tokenize(self, text: str) -> list[str]:
        return self.encode(text, out_type=str)

    def encode(self, text: str, **kwargs):
        if len(text) == 0:
            return []
        if len(text.strip()) == 1:
            return self.sp_model.Encode(text, out_type=kwargs.pop("out_type", int), **kwargs)
        # Normalise Chinese punctuation to ASCII equivalents before
        # tokenisation so that the BPE model sees the same tokens it was
        # trained on.  This prevents mispronunciation of Chinese punctuation
        # marks (e.g. 。→哦) and ensures sentence-segment boundaries are
        # correctly detected.
        text = normalize_text(text)
        for pre_tokenizer in self.pre_tokenizers:
            text = pre_tokenizer(text)
        return self.sp_model.Encode(text, out_type=kwargs.pop("out_type", int), **kwargs)

    def decode(self, ids: list[int] | int, do_lower_case=False, **kwargs):
        if isinstance(ids, int):
            ids = [ids]
        decoded = self.sp_model.Decode(ids, out_type=kwargs.pop("out_type", str), **kwargs)
        return de_tokenized_by_CJK_char(decoded, do_lower_case=do_lower_case)

    @staticmethod
    def split_segments_by_token(
        tokenized_str: list[str],
        split_tokens: list[str],
        max_text_tokens_per_segment: int,
        quick_streaming_tokens: int = 0,
    ) -> list[list[str]]:
        if len(tokenized_str) == 0:
            return []
        segments: list[list[str]] = []
        current_segment = []
        current_segment_tokens_len = 0
        i = 0
        while i < len(tokenized_str):
            token = tokenized_str[i]
            current_segment.append(token)
            current_segment_tokens_len += 1
            if not ("," in split_tokens or "▁," in split_tokens) and (
                "," in current_segment or "▁," in current_segment
            ):
                sub_segments = TextTokenizer.split_segments_by_token(
                    current_segment,
                    [",", "▁,"],
                    max_text_tokens_per_segment=max_text_tokens_per_segment,
                    quick_streaming_tokens=quick_streaming_tokens,
                )
            elif "-" not in split_tokens and "-" in current_segment:
                sub_segments = TextTokenizer.split_segments_by_token(
                    current_segment,
                    ["-"],
                    max_text_tokens_per_segment=max_text_tokens_per_segment,
                    quick_streaming_tokens=quick_streaming_tokens,
                )
            elif current_segment_tokens_len <= max_text_tokens_per_segment:
                if token in split_tokens and current_segment_tokens_len > 2:
                    if i < len(tokenized_str) - 1:
                        if tokenized_str[i + 1] in ["'", "▁'"]:
                            current_segment.append(tokenized_str[i + 1])
                            i += 1
                    segments.append(current_segment)
                    current_segment = []
                    current_segment_tokens_len = 0
                i += 1
                continue
            else:
                sub_segments = []
                for j in range(0, len(current_segment), max_text_tokens_per_segment):
                    if j + max_text_tokens_per_segment < len(current_segment):
                        sub_segments.append(current_segment[j : j + max_text_tokens_per_segment])
                    else:
                        sub_segments.append(current_segment[j:])
                warnings.warn(
                    f"The tokens length of segment exceeds limit: {max_text_tokens_per_segment}, "
                    f"Tokens in segment: {current_segment}."
                    "Maybe unexpected behavior",
                    RuntimeWarning,
                )
            segments.extend(sub_segments)
            current_segment = []
            current_segment_tokens_len = 0
            i += 1
        if current_segment_tokens_len > 0:
            assert current_segment_tokens_len <= max_text_tokens_per_segment
            segments.append(current_segment)
        merged_segments = []
        total_token = 0
        for segment in segments:
            total_token += len(segment)
            if len(segment) == 0:
                continue
            if len(merged_segments) == 0:
                merged_segments.append(segment)
            elif (
                len(merged_segments[-1]) + len(segment) <= max_text_tokens_per_segment
                and total_token > quick_streaming_tokens
            ):
                merged_segments[-1] = merged_segments[-1] + segment
            elif len(merged_segments[-1]) + len(segment) <= max_text_tokens_per_segment / 2:
                merged_segments[-1] = merged_segments[-1] + segment
            else:
                merged_segments.append(segment)
        return merged_segments

    punctuation_marks_tokens = [
        ".",
        "!",
        "?",
        "▁.",
        "▁?",
        "▁...",
    ]

    def split_segments(
        self, tokenized: list[str], max_text_tokens_per_segment=120, quick_streaming_tokens=0
    ) -> list[list[str]]:
        return TextTokenizer.split_segments_by_token(
            tokenized,
            self.punctuation_marks_tokens,
            max_text_tokens_per_segment=max_text_tokens_per_segment,
            quick_streaming_tokens=quick_streaming_tokens,
        )
