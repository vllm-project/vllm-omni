# SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

# Vendored from NVIDIA-NeMo/Speech (branch nemotron-labs-voicechat)
# nemo/collections/common/tokenizers/tokenizer_spec.py and
# nemo/collections/common/tokenizers/huggingface/auto_tokenizer.py; training-only code removed.
# Only the imports were adjusted (nemo.utils logging -> stdlib shim); the class
# bodies are exact copies.

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Optional

from transformers import AutoTokenizer as AUTOTOKENIZER

from .compat import logging

__all__ = ["TokenizerSpec", "AutoTokenizer"]


class TokenizerSpec(ABC):
    """
    Inherit this class to implement a new tokenizer.
    """

    @abstractmethod
    def text_to_tokens(self, text):
        """Converts text into a list of tokens."""
        pass

    @abstractmethod
    def tokens_to_ids(self, tokens):
        """Converts a list of tokens to their corresponding IDs."""
        pass

    @abstractmethod
    def ids_to_tokens(self, ids):
        """Converts a list of token IDs back to tokens."""
        pass

    @abstractmethod
    def text_to_ids(self, text):
        """Converts text directly to token IDs."""
        pass

    def add_special_tokens(self, special_tokens: List[str]):
        """Adds special tokens (eos, pad, cls...) to vocab."""
        raise NotImplementedError("To be implemented")

    def apply_chat_template(self, *args, **kwargs):
        """Applies chat template and tokenizes results"""
        raise NotImplementedError("To be implemented")

    @property
    def name(self):
        """name of the class"""
        return type(self).__name__

    @property
    def cls(self):
        """Property alias to match MegatronTokenizer; returns cls_id if available."""
        if hasattr(self, "cls_id"):
            return self.cls_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'cls' or 'cls_id'")

    @property
    def sep(self):
        """Property alias to match MegatronTokenizer; returns sep_id if available."""
        if hasattr(self, "sep_id"):
            return self.sep_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'sep' or 'sep_id'")

    @property
    def pad(self):
        """Property alias to match MegatronTokenizer; returns pad_id if available."""
        if hasattr(self, "pad_id"):
            return self.pad_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'pad' or 'pad_id'")

    @property
    def eod(self):
        """Property alias to match MegatronTokenizer; returns eod_id if available."""
        if hasattr(self, "eod_id"):
            return self.eod_id
        if hasattr(self, "eos_id"):
            # Default to end-of-sentence id if end-of-document is not defined.
            return self.eos_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'eod', 'eod_id', 'eos', or 'eos_id'")

    @property
    def bos(self):
        """Property alias to match MegatronTokenizer; returns bos_id if available."""
        if hasattr(self, "bos_id"):
            return self.bos_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'bos' or 'bos_id'")

    @property
    def eos(self):
        """Property alias to match MegatronTokenizer; returns eos_id if available."""
        if hasattr(self, "eos_id"):
            return self.eos_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'eos' or 'eos_id'")

    @property
    def mask(self):
        """Property alias to match MegatronTokenizer; returns mask_id if available."""
        if hasattr(self, "mask_id"):
            return self.mask_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'mask' or 'mask_id'")


class AutoTokenizer(TokenizerSpec):
    """
    Wrapper of HuggingFace AutoTokenizer
    https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer.

    """

    def __init__(
        self,
        pretrained_model_name: str,
        vocab_file: str | None = None,
        merges_file: str | None = None,
        mask_token: str | None = None,
        bos_token: str | None = None,
        eos_token: str | None = None,
        pad_token: str | None = None,
        sep_token: str | None = None,
        cls_token: str | None = None,
        unk_token: str | None = None,
        additional_special_tokens: List | None = [],
        use_fast: bool | None = True,
        trust_remote_code: bool | None = False,
        include_special_tokens: bool = False,
    ):
        """
        Args:
            pretrained_model_name: corresponds to HuggingFace-AutoTokenizer's 'pretrained_model_name_or_path' input
                argument. For more details please refer to the documentation of the `from_pretrained` method here:
                https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer.
                The list of all supported models can be found here: https://huggingface.co/models
            vocab_file: path to file with vocabulary which consists
                of characters separated by newlines.
            mask_token: mask token
            bos_token: the beginning of sequence token
            eos_token: the end of sequence token. Usually equal to sep_token
            pad_token: token to use for padding
            sep_token: token used for separating sequences
            cls_token: class token. Usually equal to bos_token
            unk_token: token to use for unknown tokens
            additional_special_tokens: list of other tokens beside standard special tokens (bos, eos, pad, etc.). For
                example, sentinel tokens for T5 (<extra_id_0>, <extra_id_1>, etc.)
            use_fast: whether to use fast HuggingFace tokenizer
            include_special_tokens: when True, converting text to ids will include special tokens / prompt tokens (if
                any), yielding self.tokenizer(text).input_ids
        """
        try:
            self._initialize_tokenizer(pretrained_model_name, vocab_file, merges_file, use_fast, trust_remote_code)
            assert self.tokenizer, "tokenizer not initialized"
        except Exception:
            try:
                self._initialize_tokenizer(
                    pretrained_model_name, vocab_file, merges_file, not use_fast, trust_remote_code
                )
                assert self.tokenizer, "tokenizer not initialized"
            except Exception as e:
                raise ValueError(
                    f"Unable to instantiate HuggingFace AUTOTOKENIZER for {pretrained_model_name}. Exception: {e}"
                )

        self.include_special_tokens = include_special_tokens
        self.original_vocab_size = len(self.tokenizer)
        special_tokens_dict = {}

        # # setting special tokens, by default the default model's special tokens will be preserved
        # # unless passes new values to the special tokens
        if unk_token is not None:
            special_tokens_dict["unk_token"] = unk_token
        if mask_token is not None:
            special_tokens_dict["mask_token"] = mask_token
        if pad_token is not None:
            special_tokens_dict["pad_token"] = pad_token

        # if the model does not have eos_token but has sep_token,
        # set eos_token = sep_token, and vice versa
        if sep_token is not None:
            special_tokens_dict["sep_token"] = sep_token
        elif self.tokenizer.sep_token is None and self.tokenizer.eos_token:
            special_tokens_dict["sep_token"] = self.tokenizer.eos_token
        if eos_token is not None:
            special_tokens_dict["eos_token"] = eos_token
        elif self.tokenizer.eos_token is None and self.tokenizer.sep_token:
            special_tokens_dict["eos_token"] = self.tokenizer.sep_token

        # if the model does not have bos_token but has cls_token,
        # set bos_token = cls_token, and vice versa
        if bos_token is not None:
            special_tokens_dict["bos_token"] = bos_token
        elif self.tokenizer.bos_token is None and self.tokenizer.cls_token:
            special_tokens_dict["bos_token"] = self.tokenizer.cls_token
        if cls_token is not None:
            special_tokens_dict["cls_token"] = cls_token
        elif self.tokenizer.cls_token is None and self.tokenizer.bos_token:
            special_tokens_dict["cls_token"] = self.tokenizer.bos_token

        # add additional special tokens (not standard special tokens such as bos, eod, sep)
        if additional_special_tokens is not None:
            special_tokens_dict["additional_special_tokens"] = additional_special_tokens

        new_tokens_in_vocab = []
        for token in [mask_token, bos_token, eos_token, pad_token, sep_token, cls_token, unk_token]:
            if token is not None and token not in self.tokenizer.get_vocab():
                new_tokens_in_vocab.append(token)
        for token in additional_special_tokens:
            if token is not None and token not in self.tokenizer.get_vocab():
                new_tokens_in_vocab.append(token)

        if len(new_tokens_in_vocab) > 0:
            """
            Special tokens that were not previously included in the tokenizer's vocabulary file will be added to
            the vocabulary and, as a result, the model should be resized, for example:

            # define your model
            pretrained_model_name = 'roberta-base'
            model = nemo_nlp.modules.get_lm_model(pretrained_model_name=pretrained_model_name)

            # define pretrained tokenizer
            tokenizer_default = nemo_nlp.modules.get_tokenizer(tokenizer_name=pretrained_model_name)

            special_tokens = {'bos_token': '<BOS>',
                              'cls_token': '<CSL>',
                              'additional_special_tokens': ['<MY_NER_TOKEN>', '<ANOTHER_TOKEN>']}
            tokenizer_default.add_special_tokens(special_tokens_dict=special_tokens)

            # resize your model so that the embeddings for newly added tokens are updated during training/finetuning
            model.resize_token_embeddings(tokenizer_default.vocab_size)

            See NLP_Tokenizers.ipynb for more details.
            """
            logging.warning(
                f"{new_tokens_in_vocab} \n will be added to the vocabulary.\n"
                f"Please resize your model accordingly, "
                f"see NLP_Tokenizers.ipynb for more details."
            )
        self.add_special_tokens(special_tokens_dict)
        self.space_sensitive = self.text_to_tokens("x y") != self.text_to_tokens("x") + self.text_to_tokens("y")
        self._inv_vocab_dict = {}

    def _initialize_tokenizer(
        self,
        pretrained_model_name: str,
        vocab_file: str | None = None,
        merges_file: str | None = None,
        use_fast: bool | None = False,
        trust_remote_code: bool | None = False,
    ):
        # this logic deals with different huggingface tokenizers having different positional args
        if vocab_file is None:
            self.tokenizer = AUTOTOKENIZER.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name,
                use_fast=use_fast,
                trust_remote_code=trust_remote_code,
            )
        elif merges_file is None:
            self.tokenizer = AUTOTOKENIZER.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name,
                vocab_file=vocab_file,
                use_fast=use_fast,
                trust_remote_code=trust_remote_code,
            )
        else:
            self.tokenizer = AUTOTOKENIZER.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name,
                vocab_file=vocab_file,
                merges_file=merges_file,
                use_fast=use_fast,
                trust_remote_code=trust_remote_code,
            )

    @property
    def vocab_size(self):
        """
        Returns the size of the tokenizer's vocabulary.

        Returns:
            int: The number of tokens in the vocabulary.
        """
        return len(self.tokenizer)

    def add_special_tokens(self, special_tokens_dict: dict) -> int:
        """
        Adds a dictionary of special tokens (eos, pad, cls...). If special tokens are NOT in the vocabulary, they are
        added to it (indexed starting from the last index of the current vocabulary).

        Args:
            special_tokens_dict: dict of string. Keys should be in the list of predefined special attributes:
                [``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``,
                ``mask_token``, ``additional_special_tokens``].
                Tokens are only added if they are not already in the vocabulary.

        Returns:
            Number of tokens added to the vocabulary.
        """
        num_tokens_added = self.tokenizer.add_special_tokens(special_tokens_dict)

        if num_tokens_added > 0:
            logging.info(f"{num_tokens_added} special tokens added, resize your model accordingly.")
        for k in self.tokenizer.SPECIAL_TOKENS_ATTRIBUTES:
            setattr(self, k, getattr(self.tokenizer, k, None))
        return num_tokens_added

    def text_to_tokens(self, text):
        """
        Converts text into a list of tokens.

        Args:
            text (str): Input text to be tokenized.

        Returns:
            List[str]: List of tokens.
        """
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def tokens_to_ids(self, tokens):
        """
        Converts a list of tokens to their corresponding IDs.

        Args:
            tokens (List[str]): List of tokens to convert.

        Returns:
            List[int]: List of token IDs.
        """
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids

    def ids_to_tokens(self, ids):
        """
        Converts a list of token IDs back to tokens.

        Args:
            ids (List[int]): List of token IDs to convert.

        Returns:
            List[str]: List of tokens.
        """
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return tokens

    def text_to_ids(self, text):
        """
        Converts text directly to token IDs.

        Args:
            text (str): Input text to be converted to IDs.

        Returns:
            List[int]: List of token IDs. If include_special_tokens is True, will include special tokens from the
            tokenizer's configuration.
        """
        if self.include_special_tokens:
            return self.tokenizer(text).input_ids
        tokens = self.text_to_tokens(text)
        ids = self.tokens_to_ids(tokens)
        return ids

    def apply_chat_template(self, *args, **kwargs):
        """Applies chat template and tokenizes results"""
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    @property
    def vocab(self):
        """
        Returns the vocabulary as a list where the index corresponds to the token ID.

        Returns:
            List[str]: List of tokens in the vocabulary.
        """
        id2vocab = {v: k for k, v in self.tokenizer.vocab.items()}
        return [id2vocab[i] for i in range(len(id2vocab))]

    @property
    def pad_id(self):
        """
        Gets the ID of the padding token.

        Returns:
            int or None: The ID of the padding token if it exists, None otherwise.
        """
        if getattr(self, "pad_token") is None:
            return None
        return self.tokens_to_ids([getattr(self, "pad_token")])[0]

    @property
    def bos_id(self):
        """
        Gets the ID of the beginning-of-sequence token.

        Returns:
            int or None: The ID of the BOS token if it exists, None otherwise.
        """
        if getattr(self, "bos_token") is None:
            return None
        return self.tokens_to_ids([getattr(self, "bos_token")])[0]

    @property
    def eos_id(self):
        """
        Gets the ID of the end-of-sequence token.

        Returns:
            int or None: The ID of the EOS token if it exists, None otherwise.
        """
        if getattr(self, "eos_token") is None:
            return None
        return self.tokens_to_ids([getattr(self, "eos_token")])[0]

    @property
    def eod(self):
        """
        Gets the ID of the end-of-document token (same as EOS token). Required for megatron-core compatibility.

        Returns:
            int: The ID of the EOD/EOS token.
        """
        return self.tokens_to_ids([getattr(self, "eos_token")])[0]

    @property
    def sep_id(self):
        """
        Gets the ID of the separator token.

        Returns:
            int or None: The ID of the separator token if it exists, None otherwise.
        """
        if getattr(self, "sep_token") is None:
            return None
        return self.tokens_to_ids([getattr(self, "sep_token")])[0]

    @property
    def cls_id(self):
        """
        Gets the ID of the classifier token.

        Returns:
            int or None: The ID of the classifier token if it exists, None otherwise.
        """
        if getattr(self, "cls_token") is None:
            return None
        return self.tokens_to_ids([getattr(self, "cls_token")])[0]

    @property
    def unk_id(self):
        """
        Gets the ID of the unknown token.

        Returns:
            int or None: The ID of the unknown token if it exists, None otherwise.
        """
        if getattr(self, "unk_token") is None:
            return None
        return self.tokens_to_ids([getattr(self, "unk_token")])[0]

    @property
    def mask_id(self):
        """
        Gets the ID of the mask token.

        Returns:
            int or None: The ID of the mask token if it exists, None otherwise.
        """
        if getattr(self, "mask_token") is None:
            return None
        return self.tokens_to_ids([getattr(self, "mask_token")])[0]

    @property
    def name(self):
        """
        Returns the name of the underlying HuggingFace tokenizer class.

        Returns:
            str: Name of the tokenizer class.
        """
        return type(self.tokenizer).__name__

    def save_pretrained(self, save_directory: str):
        """Saves tokenizer's vocabulary and other artifacts to the specified directory"""
        return self.tokenizer.save_pretrained(save_directory)
