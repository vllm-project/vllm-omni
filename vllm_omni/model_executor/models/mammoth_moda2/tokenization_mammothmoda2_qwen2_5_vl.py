# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. on 2025-09-30.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://www.apache.org/licenses/LICENSE-2.0
#
# This modified file is released under the same license.
#
# --- Upstream header preserved below ---
#
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import base64
import os
import unicodedata
from collections.abc import Collection

import tiktoken
from loguru import logger
from transformers import AddedToken, PreTrainedTokenizer

VOCAB_FILES_NAMES = {
    "vocab_file": "mammothu.tiktoken",
    "special_tokens_file": "mammothu_vision_tokens.txt",
}

PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
ENDOFTEXT = "<|endoftext|>"
IMSTART = "<|im_start|>"
IMEND = "<|im_end|>"
# as the default behavior is changed to allow special tokens in
# regular texts, the surface forms of special tokens need to be
# as different as possible to minimize the impact
QWEN_SPECIAL_TOKENS = (
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<tool_call>",
    "</tool_call>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<|repo_name|>",
    "<|file_sep|>",
)

# align to qwen2.5 tokenizer length (151846)
EXTRAS = [f"<|extra_{i}|>" for i in range(181)]  # 205 - 19[len(QWEN_SPECIAL_TOKENS)] - 5
# align to qwen2.5 embedding size (152064)
EXTRAS += [f"<|extra_margin_{i}|>" for i in range(152064 - 151846)]
# append new token in gen embedding range
EXTRAS += ["<|endofline|>", "<|endoffile|>", "<|gen_placeholder|>", "<|useless token|>", "<|beginoftext|>"]
EXTRAS = tuple(EXTRAS)
# changed to use actual index to avoid misconfiguration with vocabulary expansion
SPECIAL_START_ID = 151643


def _load_tiktoken_bpe(tiktoken_bpe_file: str) -> dict[bytes, int]:
    with open(tiktoken_bpe_file, "rb") as f:
        contents = f.read()
    return {
        base64.b64decode(token): int(rank) for token, rank in (line.split() for line in contents.splitlines() if line)
    }


class MammothUTokenizer(PreTrainedTokenizer):
    """MammothU tokenizer."""

    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab_file: str,
        special_tokens_file: str,
        errors: str = "replace",
        bos_token: str = "<|beginoftext|>",
        eos_token: str = "<|endoftext|>",
        pad_token: str = "<|endoftext|>",
        img_token: str = "<|image token|>",
        boi_token: str = "<|image start|>",
        eoi_token: str = "<|image end|>",
        eol_token: str = "<|endofline|>",
        eof_token: str = "<|endoffile|>",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # how to handle errors in decoding UTF-8 byte sequences
        # use ignore if you are in streaming inference
        self.errors = errors
        self.mergeable_ranks = _load_tiktoken_bpe(vocab_file)
        vision_tokens = [t.strip() for t in open(special_tokens_file).readlines() if len(t.strip()) > 0]
        SPECIAL_TOKENS = tuple(
            enumerate(
                (
                    (
                        ENDOFTEXT,
                        IMSTART,
                        IMEND,
                    )
                    + QWEN_SPECIAL_TOKENS
                    + EXTRAS
                    + tuple(vision_tokens)
                ),
                start=SPECIAL_START_ID,
            )
        )

        self.special_tokens = {token: index for index, token in SPECIAL_TOKENS}
        self.special_tokens_set = set(t for _, t in SPECIAL_TOKENS)

        enc = tiktoken.Encoding(
            "mammothu",
            pat_str=PAT_STR,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        assert len(self.mergeable_ranks) + len(self.special_tokens) == enc.n_vocab, (
            f"{len(self.mergeable_ranks) + len(self.special_tokens)} != {enc.n_vocab} in encoding"
        )

        self.decoder = {v: k for k, v in self.mergeable_ranks.items()}
        self.decoder.update({v: k for k, v in self.special_tokens.items()})
        self.tokenizer = enc
        self.eod_id = self.tokenizer.eot_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.img_token = img_token
        self.boi_token = boi_token
        self.eoi_token = eoi_token
        self.eol_token = eol_token
        self.eof_token = eof_token
        self.image_content_token = "<|image_pad|>"  # come from Qwen2.5-VL
        self.gen_image_token = "<|gen_image_pad|>"
        self.gen_image_placeholder_token = "<|gen_placeholder|>"
        self.visual_tokens = ["<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"]
        self.visual_tokens_ids = [self.get_vocab()[token] for token in self.visual_tokens]

        self.vision_range = (self.get_vocab()[self.boi_token], self.tokenizer.n_vocab - 1)
        logger.info(f"MammothUTokeniser Vision range: {self.vision_range}")

    def __getstate__(self):
        # for pickle lovers
        state = self.__dict__.copy()
        del state["tokenizer"]
        return state

    def __setstate__(self, state):
        # tokenizer is not python native; don't pass it; rebuild it
        self.__dict__.update(state)
        enc = tiktoken.Encoding(
            "mammothu",
            pat_str=PAT_STR,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        self.tokenizer = enc

    def __len__(self) -> int:
        return self.tokenizer.n_vocab

    def get_vocab(self) -> dict[bytes | str, int]:
        vocab = self.mergeable_ranks.copy()
        vocab.update(self.special_tokens)
        return vocab

    @property
    def gen_placeholder_id(self):
        return self.get_vocab()[self.gen_image_placeholder_token]

    def convert_tokens_to_ids(self, tokens: bytes | str | list[bytes | str]) -> list[int]:
        if isinstance(tokens, (str, bytes)):
            if tokens in self.special_tokens:
                return self.special_tokens[tokens]
            else:
                return self.mergeable_ranks.get(tokens)

        ids = []
        for token in tokens:
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            else:
                ids.append(self.mergeable_ranks.get(token))
        return ids

    def _add_tokens(
        self,
        new_tokens: list[str] | list[AddedToken],
        special_tokens: bool = False,
    ) -> int:
        if not special_tokens and new_tokens:
            raise ValueError("Adding regular tokens is not supported")

        added_tokens = 0
        for token in new_tokens:
            surface_form = token.content if isinstance(token, AddedToken) else token
            if surface_form in self.special_tokens_set:
                # Token already exists in our special tokens set
                added_tokens += 1
            else:
                logger.warning(f"Token {surface_form} is not in the predefined special tokens set and cannot be added")

        return added_tokens

    def add_special_tokens(self, special_tokens_dict: dict[str, str | AddedToken]) -> int:
        """
        Add special tokens to the tokenizer and update the special tokens mapping.
        Only adds tokens that are already in the special_tokens_set.

        Args:
            special_tokens_dict: dictionary of special tokens to add.
                The key is the token type and the value is the token to add.

        Returns:
            Number of tokens added to the vocabulary.
        """
        added_tokens = 0

        for token_type, token in special_tokens_dict.items():
            if token_type == "additional_special_tokens" and isinstance(token, list):
                added_tokens += self._add_tokens(token, special_tokens=True)
            else:
                token_value = token.content if isinstance(token, AddedToken) else token
                if token_value in self.special_tokens_set:
                    setattr(self, token_type, token_value)
                    added_tokens += 1
                else:
                    logger.warning(
                        f"Token {token_value} is not in the predefined special tokens set and cannot be added"
                    )

        return added_tokens

    def save_vocabulary(self, save_directory: str, **kwargs) -> tuple[str]:
        """
        Save only the vocabulary of the tokenizer (vocabulary).

        Returns:
            `tuple(str)`: Paths to the files saved.
        """
        regular_file_path = os.path.join(save_directory, self.vocab_files_names["vocab_file"])
        with open(regular_file_path, "w", encoding="utf8") as w:
            for k, v in self.mergeable_ranks.items():
                line = base64.b64encode(k).decode("utf8") + " " + str(v) + "\n"
                w.write(line)

        excluded_special_tokens = set(
            (
                ENDOFTEXT,
                IMSTART,
                IMEND,
            )
            + EXTRAS
        )
        special_file_path = os.path.join(save_directory, self.vocab_files_names["special_tokens_file"])
        with open(special_file_path, "w", encoding="utf8") as w:
            for k in self.special_tokens:
                if k not in excluded_special_tokens:
                    print(k, file=w)

        return (regular_file_path, special_file_path)

    def tokenize(
        self,
        text: str,
        allowed_special: set | str = "all",
        disallowed_special: Collection | str = (),
        **kwargs,
    ) -> list[bytes | str]:
        """
        Converts a string in a sequence of tokens.

        Args:
            text (`str`):
                The sequence to be encoded.
            allowed_special (`Literal["all"]` or `set`):
                The surface forms of the tokens to be encoded as special tokens in regular texts.
                Default to "all".
            disallowed_special (`Literal["all"]` or `Collection`):
                The surface forms of the tokens that should not be in regular texts and trigger errors.
                Default to an empty tuple.

            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific encode method.

        Returns:
            `list[bytes|str]`: The list of tokens.
        """
        tokens = []
        text = unicodedata.normalize("NFC", text)

        # this implementation takes a detour: text -> token id -> token surface forms
        for t in self.tokenizer.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special):
            tokens.append(self.decoder[t])

        return tokens

    def convert_tokens_to_string(self, tokens: list[bytes | str]) -> str:
        """
        Converts a sequence of tokens in a single string.
        """
        text = ""
        temp = b""
        for t in tokens:
            if isinstance(t, str):
                if temp:
                    text += temp.decode("utf-8", errors=self.errors)
                    temp = b""
                text += t
            elif isinstance(t, bytes):
                temp += t
            else:
                raise TypeError("token should only be of type types or str")
        if temp:
            text += temp.decode("utf-8", errors=self.errors)
        return text

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.n_vocab

    def _convert_id_to_token(self, index: int) -> bytes | str:
        """Converts an id to a token, special tokens included"""
        if index in self.decoder:
            return self.decoder[index]
        raise ValueError("unknown ids")

    def _convert_token_to_id(self, token: bytes | str) -> int:
        """Converts a token to an id using the vocab, special tokens included"""
        if token in self.special_tokens:
            return self.special_tokens[token]
        if token in self.mergeable_ranks:
            return self.mergeable_ranks[token]
        raise ValueError("unknown token")

    def _decode(
        self,
        token_ids: int | list[int],
        skip_special_tokens: bool = False,
        errors: str | None = None,
        **kwargs,
    ) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        if skip_special_tokens:
            token_ids = [i for i in token_ids if i < self.eod_id]

        return self.tokenizer.decode(token_ids, errors=errors or self.errors)

    def bytes_to_str(self, byte_tokens) -> str:
        """Convert byte tokens to string representation.

        Args:
            byte_tokens: A dictionary where keys are byte objects and values are integers,
                        or a single byte object.

        Returns:
            If input is a dictionary, returns a new dictionary with byte keys converted to strings.
            If input is a single byte object, returns the string representation.
        """
        if isinstance(byte_tokens, dict):
            return {k.decode("utf-8", errors=self.errors): v for k, v in byte_tokens.items() if isinstance(k, bytes)}
        if isinstance(byte_tokens, bytes):
            return byte_tokens.decode("utf-8", errors=self.errors)
        return byte_tokens
