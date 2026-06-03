from __future__ import annotations

import pytest
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer as HFTokenizer

from vllm_omni.diffusion.models.omnivoice.pipeline_omnivoice import (
    _tokenize_with_nonverbal_tags,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _ids(tok: HFTokenizer, s: str) -> list[int]:
    return tok.encode(s).ids


class TestNonVerbalTags:
    @classmethod
    def setup_class(cls):
        tokenizer_path = hf_hub_download(repo_id="k2-fsa/OmniVoice", filename="tokenizer.json")
        cls.tokenizer = HFTokenizer.from_file(tokenizer_path)

    def test_plain_text_fallback(self):
        tok = self.tokenizer
        text = "hello world"
        assert _tokenize_with_nonverbal_tags(text, tok) == _ids(tok, text)

    def test_single_nonverbal_tag_at_beginning(self):
        tok = self.tokenizer
        text = "[laughter]HelloWorld"
        expected = _ids(tok, "[laughter]") + _ids(tok, "Hello") + _ids(tok, "World")
        assert _tokenize_with_nonverbal_tags(text, tok) == expected

    def test_single_nonverbal_tag_in_middle(self):
        tok = self.tokenizer
        text = "Hello[laughter]world"
        expected = _ids(tok, "Hello") + _ids(tok, "[laughter]") + _ids(tok, "world")
        assert _tokenize_with_nonverbal_tags(text, tok) == expected

    def test_multiple_nonverbal_tags(self):
        tok = self.tokenizer
        text = "[sigh]a[question-en]b[surprise-oh]"
        expected = (
            _ids(tok, "[sigh]")
            + _ids(tok, "a")
            + _ids(tok, "[question-en]")
            + _ids(tok, "b")
            + _ids(tok, "[surprise-oh]")
        )
        assert _tokenize_with_nonverbal_tags(text, tok) == expected

    def test_tag_only(self):
        tok = self.tokenizer
        text = "[question-en]"
        assert _tokenize_with_nonverbal_tags(text, tok) == _ids(tok, text)

    def test_unknown_tag_not_special_cased(self):
        tok = self.tokenizer
        text = "x[unknown-tag]y"
        assert _tokenize_with_nonverbal_tags(text, tok) == _ids(tok, text)
