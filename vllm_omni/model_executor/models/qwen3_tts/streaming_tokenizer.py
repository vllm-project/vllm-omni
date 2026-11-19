# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact incremental text encoding for Qwen3-TTS.

Qwen3TTSStreamingTokenizerSpec shares checkpoint rules between request-local
Qwen3TTSStreamingTokenizer instances. Encoding is independent of transport,
frontend chunk boundaries, and engine submission.
"""

from __future__ import annotations

import copy
import json
from bisect import bisect_left, insort
from dataclasses import dataclass

import regex as re
from tokenizers import Regex, Tokenizer, pre_tokenizers

_CONTRACTION_REGEX = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
_PRETOKENIZE_REGEX = (
    rf"{_CONTRACTION_REGEX}|[^\r\n\p{{L}}\p{{N}}]?\p{{L}}+|\p{{N}}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

_TRAILING_WHITESPACE = pre_tokenizers.Split(Regex(r"\s+\z"), behavior="removed")
_FIXED_PIECE = pre_tokenizers.Split(Regex(rf"\A(?:{_CONTRACTION_REGEX}|\p{{N}})\z"), behavior="removed")
_GRAPHEME = re.compile(r"\X")


@dataclass(frozen=True)
class _Piece:
    """New text and an end signal for one regex piece, not the whole input."""

    text: str
    ends_piece: bool


class _PreTokenizer:
    """Adapt HF's Qwen Split to incremental text without replaying old bytes.

    The context represents one open regex piece already delivered to BPE;
    pending whitespace has not been delivered.
    The longest fixed regex alternative is three characters. Retaining four
    prefix characters preserves its selection or rejection, including the
    optional leading character of a letter/punctuation run. Once a piece is
    longer, its interior is repetition. Retaining its prefix and last
    non-whitespace character preserves future boundaries without keeping a
    whole open Chinese run; trailing whitespace stays in pending instead.

    Buffer all trailing whitespace until non-whitespace or EOF arrives, since
    later text may change how HF groups that suffix. Already determined text
    before it is still delivered, but a piece ending in pending whitespace
    stays open until that whitespace is delivered too.
    All matching, including Unicode classification, uses HF's regex engine.
    """

    _CONTEXT_PREFIX_LENGTH = 4

    def __init__(self, splitter: pre_tokenizers.Split) -> None:
        self._splitter = splitter
        self._context = ""
        self._pending = ""

    def feed(self, text: str, *, final: bool) -> list[_Piece]:
        source = self._context + self._pending + text
        delivered = len(self._context)
        matches = self._splitter.pre_tokenize_str(source)
        self._context = ""
        stable_end = len(source)
        if not final:
            remaining = _TRAILING_WHITESPACE.pre_tokenize_str(source)
            stable_end = remaining[-1][1][1] if remaining else 0
        self._pending = source[stable_end:]

        pieces: list[_Piece] = []
        for index, (matched, (start, end)) in enumerate(matches):
            if start >= stable_end:
                break
            if end < delivered:
                raise AssertionError("HF pre-tokenizer revised an already delivered boundary")
            piece_end = min(end, stable_end)
            delta = source[max(start, delivered) : piece_end]
            ends_piece = end <= stable_end and (
                final or index < len(matches) - 1 or not _FIXED_PIECE.pre_tokenize_str(matched)
            )
            if delta or ends_piece:
                pieces.append(_Piece(delta, ends_piece))
            if not ends_piece:
                context = source[start:piece_end]
                self._context = (
                    context
                    if len(context) <= self._CONTEXT_PREFIX_LENGTH
                    else context[: self._CONTEXT_PREFIX_LENGTH] + context[-1]
                )
        return pieces


class Qwen3TTSStreamingTokenizerSpec:
    """Immutable Qwen tokenizer rules shared by independent input streams.

    Args:
        tokenizer: The already loaded HuggingFace tokenizer backend.

    Raises:
        ValueError: The backend does not implement the supported Qwen encoding.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        config = json.loads(tokenizer.to_str())
        expected_pre_tokenizer = {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": _PRETOKENIZE_REGEX},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
            ],
        }
        if config["normalizer"] != {"type": "NFC"} or config["pre_tokenizer"] != expected_pre_tokenizer:
            raise ValueError("Streaming Qwen3-TTS input requires the checkpoint's NFC/Qwen pre-tokenizer")
        model = config["model"]
        if (
            model["type"] != "BPE"
            or model["dropout"] is not None
            or model["unk_token"] is not None
            or model["continuing_subword_prefix"]
            or model["end_of_word_suffix"]
            or model["byte_fallback"]
            or model.get("ignore_merges", False)
        ):
            raise ValueError("Streaming Qwen3-TTS input requires deterministic byte-level BPE")

        vocab = model["vocab"]
        self.byte_ids: dict[str, int] = {token: token_id for token, token_id in vocab.items() if len(token) == 1}
        self.normalizer = tokenizer.normalizer
        self.splitter = tokenizer.pre_tokenizer[0]
        self.byte_level = tokenizer.pre_tokenizer[1]
        self.rules: list[tuple[int, int, int]] = []
        self.left_ranks: dict[int, list[int]] = {}
        available = set(self.byte_ids)
        if len(available) != 256:
            raise ValueError("Streaming Qwen3-TTS input requires all 256 byte symbols")
        for rank, pair in enumerate(model["merges"]):
            left, right = pair.split() if isinstance(pair, str) else pair
            output = left + right
            if left not in available or right not in available or output in available:
                raise ValueError("Qwen BPE merge operands must be produced by earlier ranks")
            available.add(output)
            left_id, right_id, output_id = vocab[left], vocab[right], vocab[output]
            self.rules.append((left_id, right_id, output_id))
            self.left_ranks.setdefault(left_id, []).append(rank)

        self.added_ids: dict[str, int] = {}
        self.added_prefixes: set[str] = set()
        for token in config["added_tokens"]:
            if token["normalized"] or token["single_word"] or token["lstrip"] or token["rstrip"]:
                raise ValueError("Streaming Qwen3-TTS input requires literal added-token matching")
            content = token["content"]
            if not content:
                raise ValueError("Streaming Qwen3-TTS input does not support empty added tokens")
            self.added_ids[content] = token["id"]
            self.added_prefixes.update(content[:end] for end in range(1, len(content)))
        self.added_initials = {token[0] for token in self.added_ids}
        self.header_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False).ids
        if len(self.header_ids) != 3:
            raise ValueError("Unexpected Qwen3-TTS assistant header")
        self.footer_ids = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False).ids
        if len(self.footer_ids) != 5:
            raise ValueError("Unexpected Qwen3-TTS assistant footer")

    def create_stream(self) -> Qwen3TTSStreamingTokenizer:
        """Create an empty, independent streaming encoder."""
        return Qwen3TTSStreamingTokenizer(self)


class _StreamingBPE:
    """Compose ranked, left-to-right pair replacements with sparse state.

    Spec validation ensures a merge cannot create an earlier rule's operands,
    so ranked replacement passes equal the checkpoint's priority-based BPE.
    Each pass withholds only a possible left operand, forwarding irreversible
    output to later passes. Dispatch visits the next matching rule or pending
    pass, rather than scanning every merge rank for every input byte.
    """

    def __init__(self, spec: Qwen3TTSStreamingTokenizerSpec) -> None:
        self.spec = spec
        self.pending_ranks: list[int] = []

    def feed(self, text: str, *, ends_piece: bool) -> list[int]:
        output: list[int] = []
        for encoded, _offsets in self.spec.byte_level.pre_tokenize_str(text):
            for char in encoded:
                self._push(self.spec.byte_ids[char], 0, output)
        if ends_piece:
            while self.pending_ranks:
                rank = self.pending_ranks.pop(0)
                self._push(self.spec.rules[rank][0], rank + 1, output)
        return output

    def _push(self, token: int, start: int, output: list[int]) -> None:
        work = [(token, start)]
        while work:
            token, start = work.pop()
            ranks = self.spec.left_ranks.get(token, ())
            left_index = bisect_left(ranks, start)
            pending_index = bisect_left(self.pending_ranks, start)
            # Next rule (rank >= start) whose left operand is this token, or the sentinel.
            left_rank = ranks[left_index] if left_index < len(ranks) else len(self.spec.rules)
            # Next pending merge rule (rank >= start) to check against the current token.
            pending_rank = (
                self.pending_ranks[pending_index] if pending_index < len(self.pending_ranks) else len(self.spec.rules)
            )
            rank = min(left_rank, pending_rank)
            if rank == len(self.spec.rules):
                output.append(token)
            elif rank == pending_rank:
                self.pending_ranks.pop(pending_index)
                left, right, merged = self.spec.rules[rank]
                if token == right:
                    # Merged new token. Pushes to `work` to check its rules.
                    work.append((merged, rank + 1))
                else:
                    if token == left:
                        # Keep the new left operand pending; forward the old one below.
                        insort(self.pending_ranks, rank)
                    else:
                        # This token could be the operand of other pending ranks.
                        # Pushes to `work` to check other rules at later ranks.
                        work.append((token, rank + 1))
                    # There may be more rules to check for this left token
                    # Push the old left operand last so it is processed first.
                    work.append((left, rank + 1))
            else:
                insort(self.pending_ranks, rank)


class Qwen3TTSStreamingTokenizer:
    """Append-only encoding equivalent to the fixed checkpoint tokenizer.

    Packet boundaries do not flush normalization, pre-tokenization or BPE.
    Only literal added tokens, confirmed regex boundaries and EOF do so.
    """

    def __init__(self, spec: Qwen3TTSStreamingTokenizerSpec) -> None:
        self.spec = spec
        self._literal_pending = ""
        self._normalization_pending = ""
        self._pre_tokenizer = _PreTokenizer(spec.splitter)
        self._bpe = _StreamingBPE(spec)
        self._finished = False

    def copy(self) -> Qwen3TTSStreamingTokenizer:
        """Fork the small request-local state without copying shared rules."""
        result = copy.copy(self)
        result._pre_tokenizer = copy.copy(self._pre_tokenizer)
        result._bpe = copy.copy(self._bpe)
        result._bpe.pending_ranks = self._bpe.pending_ranks.copy()
        return result

    def feed(self, text: str, *, final: bool = False) -> list[int]:
        """Consume source once and return only irreversible new token IDs.

        Args:
            text: The next source-text delta.
            final: Whether no further source text will arrive.

        Returns:
            Stable token IDs, possibly empty before EOF.

        Raises:
            RuntimeError: Input was already closed.
            TypeError: Text or the final flag has an invalid type.
        """
        if self._finished:
            raise RuntimeError("Streaming tokenizer input is already closed")
        if not isinstance(text, str) or type(final) is not bool:
            raise TypeError("Streaming tokenizer requires string text and boolean final")

        source = self._literal_pending + text
        self._literal_pending = ""
        output: list[int] = []
        cursor = 0
        index = 0
        while index < len(source):
            if source[index] not in self.spec.added_initials:
                index += 1
                continue
            end = index + 1
            matched_end = index
            matched_id = None
            while end <= len(source):
                candidate = source[index:end]
                if candidate in self.spec.added_ids:
                    matched_end, matched_id = end, self.spec.added_ids[candidate]
                if candidate not in self.spec.added_prefixes:
                    break
                end += 1
            if end > len(source) and not final:
                self._literal_pending = source[index:]
                break
            if matched_id is not None:
                output.extend(self._encode_text(source[cursor:index], final=True))
                output.append(matched_id)
                index = matched_end
                cursor = index
            else:
                index += 1
        output.extend(self._encode_text(source[cursor:index], final=final))
        self._finished = final
        return output

    def _encode_text(self, text: str, *, final: bool) -> list[int]:
        """Normalize only complete grapheme clusters before regex processing.

        Retaining the last cluster keeps canonical reordering and composition,
        including Hangul composition, on the uncommitted side of the boundary.
        Literal added tokens terminate normalization independently.
        """
        source = self._normalization_pending + text
        boundary = len(source)
        if not final:
            for grapheme in _GRAPHEME.finditer(source):
                boundary = grapheme.start()
        normalized = self.spec.normalizer.normalize_str(source[:boundary])
        self._normalization_pending = source[boundary:]
        output: list[int] = []
        for piece in self._pre_tokenizer.feed(normalized, final=final):
            output.extend(self._bpe.feed(piece.text, ends_piece=piece.ends_piece))
        return output
