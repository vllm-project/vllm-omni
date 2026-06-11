# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for StepAudioReasoningParser.

Step-Audio supports two representations of thinking markers:

1. **Special tokens**: ``<|THINK_START|>`` and ``<|THINK_END|>``
   (single-token IDs in vocab).

2. **Text markers**: ``<think>`` and ``</think>``
   (multi-token sequences for </think>, e.g. [522, 26865]).

These tests use standalone mock objects so they can run without
installing the full vLLM/torch stack.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Bootstrap minimal vLLM stubs so the parser can be imported
# ---------------------------------------------------------------------------


def _install_vllm_stubs() -> dict[str, ModuleType | None]:
    """Temporarily install minimal stub modules for the vllm imports the
    parser needs, so it can be imported without the full vLLM/torch stack.

    Returns a snapshot of the original ``sys.modules`` entries that were
    replaced. The caller MUST pass it to ``_restore_vllm_stubs`` once the
    parser is imported. These stubs are process-global: if they leak, they
    shadow the real vllm modules for every other test in the same worker
    (and any forked server subprocess), e.g. replacing
    ``vllm.entrypoints.openai.engine.protocol`` with a stub that lacks
    ``ErrorResponse`` and breaking unrelated server-startup tests.
    """
    stubs: dict[str, ModuleType] = {}

    # DeltaMessage stub
    delta_mod = ModuleType("vllm.entrypoints.openai.engine.protocol")

    class DeltaMessage:
        __slots__ = ("reasoning", "content")

        def __init__(self, reasoning=None, content=None):
            self.reasoning = reasoning
            self.content = content

    delta_mod.DeltaMessage = DeltaMessage
    stubs["vllm.entrypoints.openai.engine.protocol"] = delta_mod

    # ReasoningParser stub
    reasoning_mod = ModuleType("vllm.reasoning")

    class ReasoningParser:
        def __init__(self, tokenizer, *args, **kwargs):
            self.model_tokenizer = tokenizer

        @property
        def vocab(self):
            if not self.model_tokenizer:
                return {}
            return self.model_tokenizer.get_vocab()

    class _ReasoningParserManager:
        reasoning_parsers = {}
        lazy_parsers = {}

        @classmethod
        def register_lazy_module(cls, name, module_path, class_name):
            cls.lazy_parsers[name] = (module_path, class_name)

    reasoning_mod.ReasoningParser = ReasoningParser
    reasoning_mod.ReasoningParserManager = _ReasoningParserManager
    stubs["vllm.reasoning"] = reasoning_mod
    stubs["vllm.reasoning.abs_reasoning_parsers"] = reasoning_mod

    # TokenizerLike stub
    tokenizers_mod = ModuleType("vllm.tokenizers")
    tokenizers_mod.TokenizerLike = object
    stubs["vllm.tokenizers"] = tokenizers_mod

    # ChatCompletionRequest / ResponsesRequest stubs
    chat_mod = ModuleType("vllm.entrypoints.openai.chat_completion.protocol")
    chat_mod.ChatCompletionRequest = MagicMock
    stubs["vllm.entrypoints.openai.chat_completion.protocol"] = chat_mod
    responses_mod = ModuleType("vllm.entrypoints.openai.responses.protocol")
    responses_mod.ResponsesRequest = MagicMock
    stubs["vllm.entrypoints.openai.responses.protocol"] = responses_mod

    # Parent packages: only stub the ones not already importable, so we never
    # clobber a real (installed) vllm package tree.
    for mod_path in [
        "vllm",
        "vllm.entrypoints",
        "vllm.entrypoints.openai",
        "vllm.entrypoints.openai.engine",
        "vllm.entrypoints.openai.chat_completion",
        "vllm.entrypoints.openai.responses",
    ]:
        if mod_path not in sys.modules and mod_path not in stubs:
            stubs[mod_path] = ModuleType(mod_path)

    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    return saved


def _restore_vllm_stubs(saved: dict[str, ModuleType | None]) -> None:
    """Undo ``_install_vllm_stubs`` so ``sys.modules`` is left untouched."""
    for name, original in saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


# Install stubs only for the duration of the import. The parser binds these
# stub classes into its own module namespace at import time, so it keeps using
# them at runtime even after sys.modules is restored below.
_saved_modules = _install_vllm_stubs()
try:
    from vllm_omni.reasoning.step_audio_reasoning_parser import (  # noqa: E402
        StepAudioReasoningParser,
    )
finally:
    _restore_vllm_stubs(_saved_modules)

# Short aliases for marker constants
TS = StepAudioReasoningParser.THINK_START_TEXT  # <think>
TE = StepAudioReasoningParser.THINK_END_TEXT  # </think>
TSS = StepAudioReasoningParser.THINK_START_SPECIAL  # <|THINK_START|>
TES = StepAudioReasoningParser.THINK_END_SPECIAL  # <|THINK_END|>


# ---------------------------------------------------------------------------
# Mock tokenizers
# ---------------------------------------------------------------------------


class SingleTokenTokenizer:
    """Tokenizer where both ``<think>`` and ``</think>`` are single tokens.

    This exercises the fast (token-ID) path.
    Also includes special-token forms ``<|THINK_START|>`` and ``<|THINK_END|>``.
    """

    def __init__(self):
        self._vocab = {
            TS: 100,  # <think>
            TE: 101,  # </think>
            TSS: 151669,  # <|THINK_START|>
            TES: 151670,  # <|THINK_END|>
            "Hello": 1,
            " world": 2,
            "reasoning": 3,
            " content": 4,
            "\n": 5,
            "</": 6,
            "think>": 7,
            "some text": 8,
        }
        self._reverse = {v: k for k, v in self._vocab.items()}

    def get_vocab(self):
        return dict(self._vocab)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self._reverse.get(i, "") for i in ids)

    def tokenize(self, text):
        if not text:
            return []
        return [text]


class MultiTokenTokenizer:
    """Tokenizer where ``</think>`` is NOT a single token.

    ``<think>`` is in the vocab (ID 100), but ``</think>`` is NOT.
    Instead, ``</think>`` encodes as ["</", "think>"] → [522, 26865].

    Both special tokens ``<|THINK_START|>`` and ``<|THINK_END|>`` ARE
    present as single tokens.

    This exercises the text-based path (with special-token fallback).
    """

    def __init__(self):
        self._vocab = {
            TS: 100,  # <think> - single token
            # </think> is intentionally NOT in the vocab
            "</": 522,
            "think>": 26865,
            TSS: 151669,  # <|THINK_START|>
            TES: 151670,  # <|THINK_END|>
            "Hello": 1,
            " world": 2,
            "reasoning": 3,
            " content": 4,
            "\n": 5,
            "some text": 8,
        }
        self._reverse = {v: k for k, v in self._vocab.items()}

    def get_vocab(self):
        return dict(self._vocab)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self._reverse.get(i, "") for i in ids)

    def tokenize(self, text):
        if not text:
            return []
        return [text]


class TextOnlyTokenizer:
    """Tokenizer with only text-form markers (no special tokens at all).

    Both ``<think>`` and ``</think>`` are single tokens.
    ``<|THINK_START|>`` and ``<|THINK_END|>`` are NOT in the vocab.

    This tests the pure text-based fallback paths.
    """

    def __init__(self):
        self._vocab = {
            TS: 100,  # <think>
            TE: 101,  # </think>
            "Hello": 1,
            " world": 2,
            "reasoning": 3,
            " content": 4,
            "\n": 5,
            "</": 6,
            "think>": 7,
        }
        self._reverse = {v: k for k, v in self._vocab.items()}

    def get_vocab(self):
        return dict(self._vocab)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self._reverse.get(i, "") for i in ids)

    def tokenize(self, text):
        if not text:
            return []
        return [text]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stream_extract(parser, deltas_and_ids):
    """Run streaming extraction over a list of (delta_text, delta_ids) tuples.
    Returns (reasoning_text, content_text)."""
    prev_text = ""
    prev_ids = []
    all_r = []
    all_c = []

    for delta_text, delta_ids in deltas_and_ids:
        cur_text = prev_text + delta_text
        cur_ids = prev_ids + delta_ids
        result = parser.extract_reasoning_streaming(prev_text, cur_text, delta_text, prev_ids, cur_ids, delta_ids)
        if result is not None:
            if result.reasoning:
                all_r.append(result.reasoning)
            if result.content:
                all_c.append(result.content)
        prev_text = cur_text
        prev_ids = cur_ids

    return "".join(all_r), "".join(all_c)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def single_token_parser():
    return StepAudioReasoningParser(SingleTokenTokenizer())


@pytest.fixture
def multi_token_parser():
    return StepAudioReasoningParser(MultiTokenTokenizer())


@pytest.fixture
def text_only_parser():
    return StepAudioReasoningParser(TextOnlyTokenizer())


@pytest.fixture
def request_obj():
    return MagicMock(messages=[], model="test-model")


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestInit:
    def test_single_token_ids(self, single_token_parser):
        """Text-form markers are in vocab as single tokens."""
        assert single_token_parser.think_start_text_id == 100
        assert single_token_parser.think_end_text_id == 101

    def test_single_token_special_ids(self, single_token_parser):
        """Special-token markers are also in vocab."""
        assert single_token_parser.think_start_special_id == 151669
        assert single_token_parser.think_end_special_id == 151670

    def test_single_token_has_ids(self, single_token_parser):
        assert single_token_parser._has_start_token_id is True
        assert single_token_parser._has_end_token_id is True

    def test_multi_token_text_ids(self, multi_token_parser):
        """<think> is a single token; </think> is NOT."""
        assert multi_token_parser.think_start_text_id == 100
        assert multi_token_parser.think_end_text_id == -1

    def test_multi_token_special_ids(self, multi_token_parser):
        """Special tokens are still in vocab."""
        assert multi_token_parser.think_start_special_id == 151669
        assert multi_token_parser.think_end_special_id == 151670

    def test_multi_token_has_ids(self, multi_token_parser):
        """Even without </think> as single token, special tokens provide IDs."""
        assert multi_token_parser._has_start_token_id is True
        assert multi_token_parser._has_end_token_id is True

    def test_text_only_ids(self, text_only_parser):
        """Only text-form markers, no special tokens."""
        assert text_only_parser.think_start_text_id == 100
        assert text_only_parser.think_end_text_id == 101
        assert text_only_parser.think_start_special_id == -1
        assert text_only_parser.think_end_special_id == -1
        assert text_only_parser._has_start_token_id is True
        assert text_only_parser._has_end_token_id is True

    def test_missing_tokenizer_raises(self):
        with pytest.raises(ValueError, match="model tokenizer must be passed"):
            StepAudioReasoningParser(None)


# ---------------------------------------------------------------------------
# extract_reasoning (non-streaming)
# ---------------------------------------------------------------------------


class TestExtractReasoning:
    @pytest.mark.parametrize(
        "parser_name",
        [
            "single_token_parser",
            "multi_token_parser",
            "text_only_parser",
        ],
    )
    def test_basic_extraction(self, request, parser_name, request_obj):
        parser = request.getfixturevalue(parser_name)
        output = f"some reasoning{TE}Hello world"
        reasoning, content = parser.extract_reasoning(output, request_obj)
        assert reasoning == "some reasoning"
        assert content == "Hello world"

    @pytest.mark.parametrize(
        "parser_name",
        [
            "single_token_parser",
            "multi_token_parser",
            "text_only_parser",
        ],
    )
    def test_no_end_token(self, request, parser_name, request_obj):
        parser = request.getfixturevalue(parser_name)
        output = "still thinking..."
        reasoning, content = parser.extract_reasoning(output, request_obj)
        assert reasoning == "still thinking..."
        assert content is None

    @pytest.mark.parametrize(
        "parser_name",
        [
            "single_token_parser",
            "multi_token_parser",
            "text_only_parser",
        ],
    )
    def test_with_start_token_prefix(self, request, parser_name, request_obj):
        parser = request.getfixturevalue(parser_name)
        output = f"{TS}my reasoning{TE}answer"
        reasoning, content = parser.extract_reasoning(output, request_obj)
        assert reasoning == "my reasoning"
        assert content == "answer"

    @pytest.mark.parametrize(
        "parser_name",
        [
            "single_token_parser",
            "multi_token_parser",
            "text_only_parser",
        ],
    )
    def test_strips_newline_after_end(self, request, parser_name, request_obj):
        parser = request.getfixturevalue(parser_name)
        output = f"reasoning{TE}\nHello"
        reasoning, content = parser.extract_reasoning(output, request_obj)
        assert reasoning == "reasoning"
        assert content == "Hello"

    @pytest.mark.parametrize(
        "parser_name",
        [
            "single_token_parser",
            "multi_token_parser",
            "text_only_parser",
        ],
    )
    def test_empty_reasoning(self, request, parser_name, request_obj):
        parser = request.getfixturevalue(parser_name)
        output = f"{TE}Hello"
        reasoning, content = parser.extract_reasoning(output, request_obj)
        assert reasoning is None
        assert content == "Hello"

    @pytest.mark.parametrize(
        "parser_name",
        [
            "single_token_parser",
            "multi_token_parser",
        ],
    )
    def test_special_end_token(self, request, parser_name, request_obj):
        """Test with <|THINK_END|> special token form."""
        parser = request.getfixturevalue(parser_name)
        output = f"reasoning{TES}answer"
        reasoning, content = parser.extract_reasoning(output, request_obj)
        assert reasoning == "reasoning"
        assert content == "answer"

    @pytest.mark.parametrize(
        "parser_name",
        [
            "single_token_parser",
            "multi_token_parser",
        ],
    )
    def test_special_start_token(self, request, parser_name, request_obj):
        """Test with <|THINK_START|> special token form."""
        parser = request.getfixturevalue(parser_name)
        output = f"{TSS}reasoning{TE}answer"
        reasoning, content = parser.extract_reasoning(output, request_obj)
        assert reasoning == "reasoning"
        assert content == "answer"


# ---------------------------------------------------------------------------
# is_reasoning_end
# ---------------------------------------------------------------------------


class TestIsReasoningEnd:
    def test_single_token_text_end(self, single_token_parser):
        assert single_token_parser.is_reasoning_end([1, 2, 101]) is True

    def test_single_token_special_end(self, single_token_parser):
        assert single_token_parser.is_reasoning_end([1, 2, 151670]) is True

    def test_single_token_no_end(self, single_token_parser):
        assert single_token_parser.is_reasoning_end([1, 2, 3]) is False

    def test_multi_token_text_end_decoded(self, multi_token_parser):
        """</think> = [522, 26865] detected via decode."""
        assert multi_token_parser.is_reasoning_end([1, 522, 26865]) is True

    def test_multi_token_special_end(self, multi_token_parser):
        """<|THINK_END|> detected as single token ID."""
        assert multi_token_parser.is_reasoning_end([1, 151670]) is True

    def test_multi_token_no_end(self, multi_token_parser):
        assert multi_token_parser.is_reasoning_end([1, 2, 3]) is False

    def test_multi_token_partial_marker(self, multi_token_parser):
        """Only </ without think> should NOT count."""
        assert multi_token_parser.is_reasoning_end([1, 522, 3]) is False

    def test_multi_token_empty_input(self, multi_token_parser):
        assert multi_token_parser.is_reasoning_end([]) is False

    def test_text_only_end(self, text_only_parser):
        assert text_only_parser.is_reasoning_end([1, 2, 101]) is True

    def test_text_only_no_end(self, text_only_parser):
        assert text_only_parser.is_reasoning_end([1, 2, 3]) is False

    def test_start_after_end_returns_false(self, single_token_parser):
        """Multi-turn prompt: start marker AFTER end marker → reasoning
        is still active (the generation prompt's start marker)."""
        # IDs: [start=100, end=101, start=100] → last marker is start
        assert single_token_parser.is_reasoning_end([100, 101, 100]) is False

    def test_end_is_last_marker_returns_true(self, single_token_parser):
        """End marker is the last marker → reasoning has ended."""
        # IDs: [start=100, end=101] → last marker is end
        assert single_token_parser.is_reasoning_end([100, 101]) is True

    def test_start_only_no_end(self, single_token_parser):
        """Only a start marker, no end marker → reasoning is active."""
        assert single_token_parser.is_reasoning_end([100, 1, 2]) is False

    def test_multi_turn_prompt_start_after_end(self, multi_token_parser):
        """Simulated multi-turn prompt with previous assistant response.

        Prompt contains: ... </think ... <think ...
        The last marker is <think (start), so is_reasoning_end
        should return False.
        """
        # Decode of these IDs: "reasoning" +  "</ " + "think>" + " " + "<think"
        # The last marker is the start marker.
        # Using: [3=reasoning, 522="</", 26865="think>", 100=TS]
        ids = [3, 522, 26865, 100]
        assert multi_token_parser.is_reasoning_end(ids) is False

    def test_multi_turn_prompt_end_is_last(self, multi_token_parser):
        """Previous turn ends with </think, no new start → ended."""
        # [3=reasoning, 522="</", 26865="think>"]
        ids = [3, 522, 26865]
        assert multi_token_parser.is_reasoning_end(ids) is True

    def test_special_start_after_end(self, multi_token_parser):
        """Special token start after end marker → reasoning still active."""
        # [151670=<|THINK_END|>, 151669=<|THINK_START|>]
        ids = [151670, 151669]
        assert multi_token_parser.is_reasoning_end(ids) is False


# ---------------------------------------------------------------------------
# is_reasoning_end_streaming
# ---------------------------------------------------------------------------


class TestIsReasoningEndStreaming:
    def test_single_token_delta_has_end(self, single_token_parser):
        assert single_token_parser.is_reasoning_end_streaming([1, 2], [101]) is True

    def test_single_token_delta_no_end(self, single_token_parser):
        assert single_token_parser.is_reasoning_end_streaming([1, 2], [3]) is False

    def test_multi_token_delta_has_both_parts(self, multi_token_parser):
        """Full </think> in delta → [522, 26865]."""
        assert multi_token_parser.is_reasoning_end_streaming([1], [522, 26865]) is True

    def test_multi_token_marker_spans_boundary(self, multi_token_parser):
        """</ in previous, think> in delta."""
        assert multi_token_parser.is_reasoning_end_streaming([1, 522], [26865]) is True

    def test_multi_token_no_end(self, multi_token_parser):
        assert multi_token_parser.is_reasoning_end_streaming([1, 2], [3]) is False


# ---------------------------------------------------------------------------
# extract_reasoning_streaming
# ---------------------------------------------------------------------------


class TestExtractReasoningStreaming:
    """Test streaming extraction for both single-token and multi-token modes."""

    @pytest.mark.parametrize(
        "parser_name",
        [
            "single_token_parser",
            "multi_token_parser",
            "text_only_parser",
        ],
    )
    def test_simple_streaming(self, request_, parser_name):
        parser = request_.getfixturevalue(parser_name)
        # Simulate: <think> reasoning </think>  final
        if parser_name == "multi_token_parser":
            deltas = [TS, "reasoning", TE, " content"]
            token_ids_per_delta = [
                [100],  # <think>
                [3],  # reasoning
                [522, 26865],  # </think>
                [4],  #  content
            ]
        else:
            deltas = [TS, "reasoning", TE, " content"]
            token_ids_per_delta = [
                [100],  # <think>
                [3],  # reasoning
                [101],  # </think>
                [4],  #  content
            ]

        r, c = _stream_extract(parser, list(zip(deltas, token_ids_per_delta)))
        assert r == "reasoning"
        assert c == " content"

    @pytest.mark.parametrize(
        "parser_name",
        [
            "single_token_parser",
            "multi_token_parser",
            "text_only_parser",
        ],
    )
    def test_no_end_token_streaming(self, request_, parser_name):
        parser = request_.getfixturevalue(parser_name)
        deltas_and_ids = [
            (TS, [100]),
            ("thinking", [3]),
            (" more", [2]),
        ]
        r, c = _stream_extract(parser, deltas_and_ids)
        assert r == "thinking more"
        assert c == ""

    def test_multi_token_end_split_across_chunks(self, multi_token_parser):
        """The key test: </think> is split across streaming chunks.

        Previous chunk ends with "reasoning</" and delta starts with
        "think>\\nfinal answer".  This is the real-world scenario when
        ``</think>`` is tokenized as two separate tokens.
        """
        deltas_and_ids = [
            ("reasoning", [3]),
            ("</", [522]),
            ("think>\nHello", [26865, 1]),
        ]
        r, c = _stream_extract(multi_token_parser, deltas_and_ids)
        assert r == "reasoning"
        assert c == "Hello"

    def test_multi_token_start_token_in_delta(self, multi_token_parser):
        """Test that <think> start token is skipped in streaming output."""
        deltas_and_ids = [
            (TS, [100]),
            ("reasoning", [3]),
            (TE, [522, 26865]),
            ("Hello", [1]),
        ]
        r, c = _stream_extract(multi_token_parser, deltas_and_ids)
        assert r == "reasoning"
        assert c == "Hello"

    def test_multi_token_three_way_split(self, multi_token_parser):
        """</think> split across 3 chunks: </ + thi + nk>."""
        deltas_and_ids = [
            ("some reasoning", [3]),
            ("</", [522]),
            ("thi", [8]),
            ("nk>", [8]),
            ("\nHello", [1]),
        ]
        r, c = _stream_extract(multi_token_parser, deltas_and_ids)
        assert r == "some reasoning"
        assert c == "Hello"

    def test_special_end_token_in_stream(self, multi_token_parser):
        """<|THINK_END|> special token in streaming (decoded as text)."""
        deltas_and_ids = [
            ("reasoning", [3]),
            (TES, [151670]),
            ("answer", [1]),
        ]
        r, c = _stream_extract(multi_token_parser, deltas_and_ids)
        assert r == "reasoning"
        assert c == "answer"

    def test_no_start_token_in_stream(self, multi_token_parser):
        """Model starts directly with reasoning (no <think> prefix)."""
        deltas_and_ids = [
            ("reasoning", [3]),
            (TE, [522, 26865]),
            ("answer", [1]),
        ]
        r, c = _stream_extract(multi_token_parser, deltas_and_ids)
        assert r == "reasoning"
        assert c == "answer"

    def test_empty_reasoning(self, single_token_parser):
        """Start immediately followed by end → empty reasoning."""
        deltas_and_ids = [
            (TS, [100]),
            (TE, [101]),
            ("Hello", [1]),
        ]
        r, c = _stream_extract(single_token_parser, deltas_and_ids)
        assert r in ("", None)
        assert c == "Hello"

    def test_end_marker_no_content_after(self, multi_token_parser):
        """End marker with no content following it."""
        deltas_and_ids = [
            ("reasoning", [3]),
            (TE, [522, 26865]),
        ]
        r, c = _stream_extract(multi_token_parser, deltas_and_ids)
        assert r == "reasoning"
        assert c == ""

    def test_delta_is_just_start_token(self, multi_token_parser):
        """Delta consisting only of the start token should be skipped."""
        deltas_and_ids = [
            (TS, [100]),
        ]
        r, c = _stream_extract(multi_token_parser, deltas_and_ids)
        assert r == ""
        assert c == ""

    @pytest.mark.parametrize(
        "parser_name",
        [
            "single_token_parser",
            "multi_token_parser",
        ],
    )
    def test_only_first_newline_stripped_after_end(self, request_, parser_name):
        """Only the first newline after </think> should be stripped."""
        parser = request_.getfixturevalue(parser_name)
        if parser_name == "multi_token_parser":
            deltas_and_ids = [
                ("reasoning", [3]),
                (TE, [522, 26865]),
                ("\n\nHello", [5, 1]),
            ]
        else:
            deltas_and_ids = [
                ("reasoning", [3]),
                (TE, [101]),
                ("\n\nHello", [5, 1]),
            ]
        r, c = _stream_extract(parser, deltas_and_ids)
        assert r == "reasoning"
        assert c == "\nHello"

    @pytest.mark.parametrize(
        "parser_name",
        [
            "single_token_parser",
            "multi_token_parser",
        ],
    )
    def test_content_after_end_multiple_deltas(self, request_, parser_name):
        """Multiple content deltas after end marker."""
        parser = request_.getfixturevalue(parser_name)
        if parser_name == "multi_token_parser":
            deltas_and_ids = [
                (TS, [100]),
                ("reasoning", [3]),
                (TE, [522, 26865]),
                ("Hello", [1]),
                (" world", [2]),
            ]
        else:
            deltas_and_ids = [
                (TS, [100]),
                ("reasoning", [3]),
                (TE, [101]),
                ("Hello", [1]),
                (" world", [2]),
            ]
        r, c = _stream_extract(parser, deltas_and_ids)
        assert r == "reasoning"
        assert c == "Hello world"

    def test_pending_buffer_resolved_on_non_marker(self, multi_token_parser):
        """When buffered text turns out not to be a marker prefix, it is
        emitted as reasoning on the next call."""
        # "reason<" is buffered because "<" could start a marker.
        # Next delta ">" doesn't form a marker, so the buffer is
        # flushed as reasoning.
        parser = multi_token_parser
        deltas_and_ids = [
            (TS, [100]),
            ("reason", [3]),
            ("<", [8]),  # "<" is buffered (prefix of markers)
            ("> more", [8]),  # ">" doesn't complete any marker
        ]
        r, c = _stream_extract(parser, deltas_and_ids)
        assert r == "reason> more"
        assert c == ""

    def test_multi_token_end_with_reasoning_before_buffer(self, multi_token_parser):
        """Reasoning text before a buffered prefix should be emitted."""
        # "reasoning<" → "reasoning" emitted, "<" buffered
        # "</think>\nanswer" → buffer resolves as end marker
        parser = multi_token_parser
        deltas_and_ids = [
            ("reasoning", [3]),
            ("<", [8]),
            (f"/{TE[2:]}\nanswer", [522, 26865, 1]),
        ]
        r, c = _stream_extract(parser, deltas_and_ids)
        assert r == "reasoning"
        assert c == "answer"

    def test_multi_token_five_way_split(self, multi_token_parser):
        """</think> split across 5 tiny chunks."""
        # Split: each chunk is one character of </think>
        chunks = list(TE)
        deltas_and_ids = [
            ("reasoning", [3]),
        ]
        for ch in chunks:
            deltas_and_ids.append((ch, [8]))
        deltas_and_ids.append(("\nHello", [1]))
        r, c = _stream_extract(multi_token_parser, deltas_and_ids)
        assert r == "reasoning"
        assert c == "Hello"

    def test_single_token_parser_with_pending(self, single_token_parser):
        """Single-token parser: ambiguous '<' prefix buffered then resolved."""
        # Even with single-token markers, the text-based path buffers
        # '<' because it could be a prefix of a marker.
        parser = single_token_parser
        deltas_and_ids = [
            (TS, [100]),
            ("reason", [3]),
            ("<", [8]),
            ("> not a marker", [8]),
        ]
        r, c = _stream_extract(parser, deltas_and_ids)
        assert r == "reason> not a marker"
        assert c == ""


# ---------------------------------------------------------------------------


class TestCountReasoningTokens:
    def test_single_token_basic(self, single_token_parser):
        # IDs: [100=<think>, 3=reasoning, 101=</think>, 4=content]
        assert single_token_parser.count_reasoning_tokens([100, 3, 101, 4]) == 1

    def test_single_token_no_end(self, single_token_parser):
        # All tokens after start are reasoning
        assert single_token_parser.count_reasoning_tokens([100, 3, 2]) == 2

    def test_multi_token_basic(self, multi_token_parser):
        # </think> encodes as [522, 26865]
        # Full sequence: <think> reasoning </think> content
        # IDs: [100, 3, 522, 26865, 4]
        result = multi_token_parser.count_reasoning_tokens([100, 3, 522, 26865, 4])
        assert result > 0

    def test_multi_token_no_end(self, multi_token_parser):
        # No end token → all tokens inside <think> are counted.
        result = multi_token_parser.count_reasoning_tokens([100, 3, 2])
        assert result > 0

    def test_text_only_basic(self, text_only_parser):
        assert text_only_parser.count_reasoning_tokens([100, 3, 101, 4]) == 1

    def test_text_only_no_end(self, text_only_parser):
        assert text_only_parser.count_reasoning_tokens([100, 3, 2]) == 2


# ---------------------------------------------------------------------------
# extract_content_ids
# ---------------------------------------------------------------------------


class TestExtractContentIds:
    def test_single_token_basic(self, single_token_parser):
        # [1, 101=</think>, 2, 3] → content starts after ID 101
        result = single_token_parser.extract_content_ids([1, 101, 2, 3])
        assert result == [2, 3]

    def test_single_token_no_end(self, single_token_parser):
        result = single_token_parser.extract_content_ids([1, 2, 3])
        assert result == []

    def test_multi_token_basic(self, multi_token_parser):
        # </think> = [522, 26865]
        # [3, 522, 26865, 4] → content after </think>
        result = multi_token_parser.extract_content_ids([3, 522, 26865, 4])
        assert 4 in result

    def test_multi_token_no_end(self, multi_token_parser):
        result = multi_token_parser.extract_content_ids([1, 2, 3])
        assert result == []

    def test_text_only_basic(self, text_only_parser):
        result = text_only_parser.extract_content_ids([1, 101, 2, 3])
        assert result == [2, 3]

    def test_text_only_no_end(self, text_only_parser):
        result = text_only_parser.extract_content_ids([1, 2, 3])
        assert result == []


# ---------------------------------------------------------------------------
# Fixture request workaround (pytest "request" fixture shadowed by parametrize)
# ---------------------------------------------------------------------------


@pytest.fixture
def request_(request):
    """Proxy for the pytest ``request`` fixture to avoid name collisions
    with the ``request_obj`` fixture used in extract_reasoning tests."""
    return request
