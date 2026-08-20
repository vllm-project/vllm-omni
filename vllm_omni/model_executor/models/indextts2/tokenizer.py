# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

from transformers import PreTrainedTokenizer
from vllm.logger import init_logger

from vllm_omni.model_executor.models.indextts2.utils.front import TextTokenizer

logger = init_logger(__name__)


class IndexTTS2Tokenizer(PreTrainedTokenizer):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        vocab_file = kwargs.pop("vocab_file", None)
        if vocab_file is None:
            vocab_file = os.path.join(pretrained_model_name_or_path, "bpe.model")
        logger.info(f"IndexTTS2Tokenizer.from_pretrained: {pretrained_model_name_or_path}")
        return cls(vocab_file, **kwargs)

    def __init__(self, vocab_file: str, **kwargs):
        logger.info(f"IndexTTS2Tokenizer.__init__: vocab_file={vocab_file}")
        self.vocab_file = vocab_file
        self._tok = TextTokenizer(vocab_file)
        logger.info("IndexTTS2Tokenizer initialized successfully")
        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        return self._tok.vocab_size

    def get_vocab(self):
        return self._tok.get_vocab()

    def _tokenize(self, text):
        return self._tok.tokenize(text)

    def _convert_token_to_id(self, token):
        return self._tok.convert_tokens_to_ids(token)[0]

    def convert_tokens_to_string(self, tokens):
        return self._tok.decode(self._tok.convert_tokens_to_ids(tokens))

    def split_segments(
        self,
        tokenized: list[str],
        max_text_tokens_per_segment: int = 120,
        quick_streaming_tokens: int = 0,
    ) -> list[list[str]]:
        return self._tok.split_segments(
            tokenized,
            max_text_tokens_per_segment=max_text_tokens_per_segment,
            quick_streaming_tokens=quick_streaming_tokens,
        )
