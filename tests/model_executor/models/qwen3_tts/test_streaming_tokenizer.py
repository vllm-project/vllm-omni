# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3-TTS streaming-tokenizer unit tests, without serving or engine state."""

from __future__ import annotations

import pytest
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast
from transformers.models.qwen2.tokenization_qwen2 import PRETOKENIZE_REGEX

from vllm_omni.model_executor.models.qwen3_tts.streaming_tokenizer import (
    Qwen3TTSStreamingTokenizerSpec,
    _PreTokenizer,
    _StreamingBPE,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(scope="module")
def tokenizer():
    """Construct merge rules that exercise English, Chinese, and rank conflicts."""
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    vocab = {token: index for index, token in enumerate(sorted(byte_level.alphabet()))}
    merges = []
    for text in (
        "assistant",
        "bc",
        "ab",
        "Get",
        " the",
        "trust",
        " trust",
        " now",
        "自动",
        "驾",
        "驾驶",
        "自动驾驶",
        "系统",
        "正在",
        "运行",
    ):
        encoded = byte_level.pre_tokenize_str(text)[0][0]
        left = encoded[0]
        for right in encoded[1:]:
            if left + right not in vocab:
                merges.append((left, right))
                vocab[left + right] = len(vocab)
            left += right
    backend = Tokenizer(BPE(vocab=vocab, merges=merges))
    backend.decoder = decoders.ByteLevel()
    backend.normalizer = normalizers.NFC()
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Split(Regex(PRETOKENIZE_REGEX), behavior="isolated"), byte_level]
    )
    backend.add_special_tokens(
        [
            AddedToken(content, normalized=False, special=True)
            for content in (
                "<|im_start|>",
                "<|im_end|>",
                "<tool_call>",
                "</tool_call>",
            )
        ]
    )
    return PreTrainedTokenizerFast(tokenizer_object=backend)


@pytest.fixture(scope="module")
def spec(tokenizer):
    return Qwen3TTSStreamingTokenizerSpec(tokenizer.backend_tokenizer)


@pytest.mark.parametrize(
    "deltas",
    [
        pytest.param(("Get ", "the ", "trust ", "now."), id="english-spaces"),
        pytest.param(tuple("自动驾驶系统正在运行") + ("",), id="chinese"),
        pytest.param(("She's ", "WE'RE ", "done."), id="contractions"),
        pytest.param(("价格25元，记录", "。"), id="number-and-unit"),
        pytest.param(("a  \n ", "\n"), id="whitespace-newlines"),
        pytest.param(("e\u0301", "\u0327", "中", ""), id="nfc-reordering"),
        pytest.param(("ᄀ", "ᅡ", "ᆨ", ""), id="hangul-composition"),
        pytest.param(("\u088f'", "rea"), id="hf-unicode-categories"),
        pytest.param(("前文", "<tool_call>中文", "</tool_call>后文", ""), id="literal-tokens"),
        pytest.param(("前文", "<|im_sta"), id="unfinished-literal-at-eof"),
    ],
)
def test_stream_matches_whole_encoding_across_delivery_batches(tokenizer, spec, deltas):
    """Coalesce frontend-released deltas; only the last delta carries EOF."""
    expected = tokenizer.encode("".join(deltas), add_special_tokens=False)
    deliveries = {
        deltas,
        ("".join(deltas),),
        tuple("".join(deltas[index : index + 2]) for index in range(0, len(deltas), 2)),
    }
    deliveries.update(("".join(deltas[:split]), "".join(deltas[split:])) for split in range(1, len(deltas)))
    for batches in sorted(deliveries):
        stream = spec.create_stream()
        output = []
        for index, text in enumerate(batches):
            output.extend(stream.feed(text, final=index == len(batches) - 1))
            assert output == expected[: len(output)]
        assert output == expected


@pytest.mark.parametrize(
    "deltas, expected",
    [
        pytest.param(
            ("Get ", "the", ""),
            [[("Get", True)], [(" the", False)], [("", True)]],
            id="word-before-space",
        ),
        pytest.param(
            ("!\n", " \n", "X", ""),
            [[("!", False)], [], [("\n", True), (" \n", True), ("X", False)], [("", True)]],
            id="newline-attached-to-punctuation",
        ),
        pytest.param(
            ("a  \n ", "\n", "X", ""),
            [[("a", True)], [], [("  \n \n", True), ("X", False)], [("", True)]],
            id="newlines-join-whitespace",
        ),
        pytest.param(
            ("'rE", "1²Ⅳ", ""),
            [[("'rE", True)], [("1", True), ("²", True), ("Ⅳ", True)], []],
            id="fixed-contraction-and-numbers",
        ),
        pytest.param((" \t", ""), [[], [(" \t", True)]], id="whitespace-at-eof"),
    ],
)
def test_pre_tokenizer_preserves_confirmed_piece_boundaries(spec, deltas, expected):
    pre_tokenizer = _PreTokenizer(spec.splitter)
    for index, text in enumerate(deltas):
        pieces = pre_tokenizer.feed(text, final=index == len(deltas) - 1)
        assert [(piece.text, piece.ends_piece) for piece in pieces] == expected[index]


@pytest.mark.parametrize("prefix_length", [4, 5, 8])
def test_pre_tokenizer_delivers_only_new_text_across_context_compaction(spec, prefix_length):
    pre_tokenizer = _PreTokenizer(spec.splitter)
    prefix = "中" * prefix_length
    pieces = pre_tokenizer.feed(prefix, final=False)
    assert [(piece.text, piece.ends_piece) for piece in pieces] == [(prefix, False)]
    pieces = pre_tokenizer.feed("文1", final=False)
    assert [(piece.text, piece.ends_piece) for piece in pieces] == [("文", True), ("1", True)]


def test_bpe_does_not_commit_a_leaf_token_that_can_be_resegmented(tokenizer, spec):
    bpe = _StreamingBPE(spec)
    assert bpe.feed("ab", ends_piece=False) == []
    result = bpe.feed("c", ends_piece=True)
    assert result == tokenizer.encode("abc", add_special_tokens=False)


def test_open_chinese_piece_can_release_before_punctuation_or_eof(tokenizer, spec):
    stream = spec.create_stream()
    assert stream.feed("自动驾") == []
    released = stream.feed("驶系统正在运行")
    assert released == tokenizer.encode("自动驾驶系统正在", add_special_tokens=False)


@pytest.mark.parametrize(
    "prefix, first_suffix, second_suffix",
    [
        ("自动驾", "驶", ""),
        ("a  \n ", "\nX", "Y"),
        ("e\u0301", "\u0327中", "文"),
        ("ᄀ", "ᅡ", "ᅩ"),
    ],
)
def test_copy_keeps_pending_encoding_independent(tokenizer, spec, prefix, first_suffix, second_suffix):
    first = spec.create_stream()
    released = first.feed(prefix)
    second = first.copy()
    assert released + first.feed(first_suffix, final=True) == tokenizer.encode(
        prefix + first_suffix, add_special_tokens=False
    )
    assert released + second.feed(second_suffix, final=True) == tokenizer.encode(
        prefix + second_suffix, add_special_tokens=False
    )
