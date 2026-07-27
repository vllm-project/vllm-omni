# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Iterable, Sequence
from itertools import islice
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser
from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


class StepAudioReasoningParser(ReasoningParser):
    """Reasoning parser for Step-Audio models.

    Step-Audio supports two representations of thinking markers:

    1. **Special tokens**: ``<|THINK_START|>`` and ``<|THINK_END|>``
       (single-token IDs, e.g. 151669 and 151670).

    2. **Text markers**: ```` and ``````
       (multi-token sequences, e.g. ``</think`` → [522, 26865],
       ``>`` → [29]).

    The chat template and typical model output use the **text form**
    (multi-token).  The parser therefore always uses **text-based**
    matching for streaming extraction, but also checks single-token
    IDs when available for methods that only receive token IDs
    (``is_reasoning_end``, ``extract_content_ids``, etc.).
    """

    # Text-form think markers (used in chat template & typical model output).
    THINK_START_TEXT = "<think>"
    THINK_END_TEXT = "</think>"

    # Special-token-form think markers (single-token IDs in vocab).
    THINK_START_SPECIAL = "<|THINK_START|>"
    THINK_END_SPECIAL = "<|THINK_END|>"

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser constructor during construction."
            )

        # --- Resolve think markers ----------------------------------------
        # Special-token form (single tokens in vocab for fast ID checks).
        self.think_start_special_id: int = self.vocab.get(self.THINK_START_SPECIAL, -1)
        self.think_end_special_id: int = self.vocab.get(self.THINK_END_SPECIAL, -1)

        # Text-form markers in vocab (usually NOT single tokens for
        # Step-Audio, but checked for completeness).
        self.think_start_text_id: int = self.vocab.get(self.THINK_START_TEXT, -1)
        self.think_end_text_id: int = self.vocab.get(self.THINK_END_TEXT, -1)

        # Set canonical token IDs for the ReasoningParser base contract.
        # Prefer special-token IDs; fall back to text-form IDs.
        self.think_start_token_id: int = (
            self.think_start_special_id if self.think_start_special_id != -1 else self.think_start_text_id
        )
        self.think_end_token_id: int = (
            self.think_end_special_id if self.think_end_special_id != -1 else self.think_end_text_id
        )

        # Whether single-token ID matching is possible for start/end.
        self._has_start_token_id = self.think_start_token_id != -1
        self._has_end_token_id = self.think_end_token_id != -1

        # For text-based matching we must recognise BOTH forms.
        # self.think_start_token / self.think_end_token are used by
        # some helper methods and for the text-based path.
        self.think_start_token = self.THINK_START_TEXT
        self.think_end_token = self.THINK_END_TEXT

        # Tracks whether we have already emitted the end of reasoning
        # in streaming mode.
        self._reasoning_ended = False

        # Tracks whether reasoning JUST ended (within the last call),
        # so we can strip a leading newline from the next content delta.
        self._just_ended = False

        # Buffer for text that might be the start of a multi-token
        # think marker (e.g. "</" could be the beginning of "</think>").
        # We hold this text until the next delta disambiguates it.
        self._pending = ""

        # Complete think marker strings used for ambiguous-prefix detection.
        # When streaming, the tail of delta_text might be the start of one
        # of these markers (e.g. "</" could be the beginning of ATHINK_END_AT).
        # We buffer such ambiguous text to avoid leaking partial markers.
        self._all_markers = (
            self.THINK_START_TEXT,
            self.THINK_END_TEXT,
            self.THINK_START_SPECIAL,
            self.THINK_END_SPECIAL,
        )

    # ------------------------------------------------------------------
    # Token-ID helpers
    # ------------------------------------------------------------------

    def _has_end_token_in_ids(self, token_ids: Sequence[int]) -> bool:
        """Check for end marker in token IDs (single-token forms only)."""
        if self.think_end_special_id != -1 and self.think_end_special_id in token_ids:
            return True
        if self.think_end_text_id != -1 and self.think_end_text_id in token_ids:
            return True
        return False

    def _has_start_token_in_ids(self, token_ids: Sequence[int]) -> bool:
        """Check for start marker in token IDs (single-token forms only)."""
        if self.think_start_special_id != -1 and self.think_start_special_id in token_ids:
            return True
        if self.think_start_text_id != -1 and self.think_start_text_id in token_ids:
            return True
        return False

    # ------------------------------------------------------------------
    # Text-based helpers (handles both text-form and special-form markers)
    # ------------------------------------------------------------------

    def _has_end_token_in_text(self, text: str) -> bool:
        return self.THINK_END_TEXT in text or self.THINK_END_SPECIAL in text

    def _has_start_token_in_text(self, text: str) -> bool:
        return self.THINK_START_TEXT in text or self.THINK_START_SPECIAL in text

    def _find_end_token_in_text(self, text: str) -> tuple[int, int]:
        """Find the first end token in text. Returns (start, end) indices
        such that text[start:end] is the end token.  Returns (-1, -1) if
        not found."""
        idx = text.find(self.THINK_END_TEXT)
        if idx != -1:
            return idx, idx + len(self.THINK_END_TEXT)
        idx = text.find(self.THINK_END_SPECIAL)
        if idx != -1:
            return idx, idx + len(self.THINK_END_SPECIAL)
        return -1, -1

    def _find_start_token_in_text(self, text: str) -> tuple[int, int]:
        """Find the first start token in text. Returns (start, end) indices
        such that text[start:end] is the start token.  Returns (-1, -1) if
        not found."""
        idx = text.find(self.THINK_START_TEXT)
        if idx != -1:
            return idx, idx + len(self.THINK_START_TEXT)
        idx = text.find(self.THINK_START_SPECIAL)
        if idx != -1:
            return idx, idx + len(self.THINK_START_SPECIAL)
        return -1, -1

    def _decode_ids(self, token_ids: Sequence[int]) -> str:
        """Decode a sequence of token IDs to text using the model tokenizer."""
        return self.model_tokenizer.decode(token_ids, skip_special_tokens=False)

    # ------------------------------------------------------------------
    # ReasoningParser interface
    # ------------------------------------------------------------------

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Check if reasoning has ended in the given token sequence.

        When called with **prompt** token IDs (by the serving layer), the
        prompt may contain think markers from previous assistant turns.
        In multi-turn conversations the prompt can include both start
        and end markers, with the *last* marker being a start marker
        (from the generation prompt).  In that case reasoning is NOT
        ended — the model is about to generate inside a new think block.

        We therefore find the **last** think marker (start or end) in
        the decoded text and only return True if it is an end marker.
        """
        try:
            text = self._decode_ids(input_ids)
        except Exception:
            # Fall back to token-ID-only check.
            result = self._has_end_token_in_ids(input_ids)
            logger.debug(
                "StepAudio is_reasoning_end: decode failed, fallback result=%s, num_ids=%d",
                result,
                len(input_ids),
            )
            return result

        # Find the position of the LAST think marker (start or end).
        last_start = max(
            text.rfind(self.THINK_START_TEXT),
            text.rfind(self.THINK_START_SPECIAL),
        )
        last_end = max(
            text.rfind(self.THINK_END_TEXT),
            text.rfind(self.THINK_END_SPECIAL),
        )

        if last_end == -1:
            # No end marker at all → reasoning not ended.
            logger.debug(
                "StepAudio is_reasoning_end: no end marker found, text_tail=%r (len=%d), returning False",
                text[-100:] if len(text) > 100 else text,
                len(text),
            )
            return False

        if last_start > last_end:
            # Last marker is a START marker (e.g. the generation prompt's
            # ``).  Reasoning is still active.
            logger.debug(
                "StepAudio is_reasoning_end: last_start=%d > last_end=%d, reasoning still active, returning False",
                last_start,
                last_end,
            )
            return False

        # Last marker is an END marker → reasoning has ended.
        logger.debug(
            "StepAudio is_reasoning_end: last_end=%d >= last_start=%d, reasoning ended, returning True",
            last_end,
            last_start,
        )
        return True

    def is_reasoning_end_streaming(self, input_ids: Sequence[int], delta_ids: Iterable[int]) -> bool:
        delta_list = tuple(delta_ids)
        # Fast check: single-token markers in delta.
        if self._has_end_token_in_ids(delta_list):
            return True
        # Fallback: decode delta + tail of input_ids and check text.
        # The end marker may span the boundary between previous tokens
        # and new delta tokens when it is encoded as multiple tokens
        # (e.g. "</think" = [522, 26865]).
        try:
            delta_text = self._decode_ids(delta_list)
            if self._has_end_token_in_text(delta_text):
                return True
            overlap_ids = list(input_ids[-8:]) + list(delta_list)
            overlap_text = self._decode_ids(overlap_ids)
            return self._has_end_token_in_text(overlap_text)
        except Exception:
            return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # Fast check: single end-token in IDs.
        if self._has_end_token_id:
            # Check for special-token or text-form single token.
            for end_id in (self.think_end_special_id, self.think_end_text_id):
                if end_id != -1 and end_id in islice(input_ids, 0, max(0, len(input_ids) - 1)):
                    idx = input_ids.index(end_id)
                    return input_ids[idx + 1 :]

        # Fallback: decode to text and find multi-token end marker.
        try:
            text = self._decode_ids(input_ids)
            end_pos, end_len = self._find_end_token_in_text(text)
            if end_pos == -1:
                return []
            content_start = end_len
            if content_start < len(text) and text[content_start] == "\n":
                content_start += 1
            if content_start >= len(text):
                return []
            for i in range(len(input_ids)):
                prefix_text = self._decode_ids(input_ids[: i + 1])
                if len(prefix_text) > content_start:
                    return input_ids[i:]
            return []
        except Exception:
            return []

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        logger.debug(
            "StepAudio extract_reasoning: model_output=%r (len=%d)",
            model_output[:200],
            len(model_output),
        )
        # Strip leading start token if present (text or special form).
        for start_tok in (self.THINK_START_TEXT, self.THINK_START_SPECIAL):
            if model_output.startswith(start_tok):
                model_output = model_output[len(start_tok) :]
                logger.debug("StepAudio extract_reasoning: stripped start_tok=%r", start_tok)
                break

        # Find the end token (text or special form).
        end_pos, end_len = self._find_end_token_in_text(model_output)
        if end_pos == -1:
            # No end token found — everything is reasoning.
            logger.debug(
                "StepAudio extract_reasoning: no end token, reasoning=%r (len=%d), content=None",
                model_output[:200],
                len(model_output),
            )
            return model_output or None, None

        reasoning = model_output[:end_pos]
        content = model_output[end_len:]
        # Strip leading newline that models often emit right after end token.
        content = content.lstrip("\n")
        logger.debug(
            "StepAudio extract_reasoning: end_pos=%d, reasoning=%r (len=%d), content=%r (len=%d)",
            end_pos,
            reasoning[:200],
            len(reasoning),
            content[:200],
            len(content),
        )
        return reasoning or None, content or None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        # Step-Audio typically uses multi-token text-form markers
        # (e.g. "</think" → [522, 26865] + ">" → [29]).
        # We always use the text-based path because it handles both
        # single-token special markers and multi-token text markers.
        # The text-based path is also efficient since previous_text
        # and delta_text are already available.
        return self._extract_streaming_text(
            previous_text,
            delta_text,
        )

    # ------------------------------------------------------------------
    # Streaming: text-based (handles both text and special token forms)
    # ------------------------------------------------------------------

    def _ends_with_marker_prefix(self, text: str) -> str:
        """If text ends with a prefix of any think marker, return the
        ambiguous suffix that should be buffered.

        For example:
        - text="reasoning</" → marker ATHINK_END_AT starts with "</" → return "</"
        - text="reasoning<" → markers start with "<" → return "<"
        - text="reasoning<th" → ATHINK_START starts with "<th" → return "<th"
          (but "</think" also starts with "<", so we buffer from the longest)
        - text="reasoning</thi" → ATHINK_END_AT starts with "</thi" → return "</thi"
        """
        if not text:
            return ""

        # Find the longest suffix of text that is a prefix of any marker.
        best = ""
        for marker in self._all_markers:
            for i in range(1, len(marker) + 1):
                prefix = marker[:i]
                if text.endswith(prefix) and len(prefix) > len(best):
                    best = prefix
        return best

    # Counter for debug logging (to avoid flooding logs)
    _stream_call_count: int = 0

    def _extract_streaming_text(
        self,
        previous_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        # Reset _just_ended flag at the start of each call.
        # It will be set to True if this call ends reasoning.
        just_ended_before = self._just_ended
        self._just_ended = False

        # Debug logging for first few calls
        StepAudioReasoningParser._stream_call_count += 1
        call_num = StepAudioReasoningParser._stream_call_count
        if call_num <= 20:
            logger.debug(
                "StepAudio stream #%d: prev_text=%r (len=%d), delta_text=%r (len=%d), _reasoning_ended=%s, _pending=%r",
                call_num,
                previous_text[:100],
                len(previous_text),
                delta_text[:100],
                len(delta_text),
                self._reasoning_ended,
                self._pending[:50] if self._pending else "",
            )

        # Prepend any pending text from a previous call.
        # The pending text is already part of previous_text (the framework
        # accumulates all generated text), so we must also trim it from
        # previous_text to avoid double-counting when we form `combined`.
        if self._pending:
            n = len(self._pending)
            if previous_text.endswith(self._pending):
                previous_text = previous_text[:-n]
            delta_text = self._pending + delta_text
            self._pending = ""

        combined = previous_text + delta_text
        has_end = self._has_end_token_in_text(combined)
        has_end_in_previous = self._has_end_token_in_text(previous_text)
        has_end_in_delta = self._has_end_token_in_text(delta_text)

        if call_num <= 20:
            logger.debug(
                "StepAudio stream #%d: has_end=%s, has_end_in_previous=%s, has_end_in_delta=%s, combined=%r (len=%d)",
                call_num,
                has_end,
                has_end_in_previous,
                has_end_in_delta,
                combined[:150],
                len(combined),
            )

        if has_end_in_previous or self._reasoning_ended:
            # Already past reasoning — everything is content.
            self._reasoning_ended = True
            # Strip at most one leading newline right after the end marker.
            if delta_text.startswith("\n") and just_ended_before:
                return DeltaMessage(content=delta_text[1:] or None)
            return DeltaMessage(content=delta_text)

        if has_end_in_delta:
            # End token fully contained in the delta — split reasoning
            # and content.
            end_index, end_len = self._find_end_token_in_text(delta_text)
            reasoning = delta_text[:end_index]
            content = delta_text[end_len:]
            # Strip leading newline after end token.
            if content.startswith("\n"):
                content = content[1:]
            self._reasoning_ended = True
            self._just_ended = True
            return DeltaMessage(
                reasoning=reasoning or None,
                content=content or None,
            )

        if has_end and not has_end_in_previous and not has_end_in_delta:
            # End token spans the boundary between previous_text and
            # delta_text (e.g. previous ends with "</" and delta
            # starts with "think>...").  Find the split point in the
            # combined text.
            end_index, end_len = self._find_end_token_in_text(combined)
            marker_start_in_prev = end_index - len(previous_text)
            if marker_start_in_prev > 0:
                # Some of previous_text's trailing chars are part of
                # the marker — they were already emitted as reasoning
                # in a previous call.  The delta begins the content
                # after the marker.
                content_after = combined[end_len:]
                if content_after.startswith("\n"):
                    content_after = content_after[1:]
                self._reasoning_ended = True
                self._just_ended = True
                return DeltaMessage(content=content_after or None)
            else:
                # Edge case: the entire marker is in previous_text
                # but was not detected because of how text was split
                # across streaming chunks.
                self._reasoning_ended = True
                self._just_ended = True
                full_content = combined[end_len:]
                if full_content.startswith("\n"):
                    full_content = full_content[1:]
                return DeltaMessage(content=full_content or None)

        # No end token seen yet — everything is reasoning.
        # Skip the start token prefix if present in delta.
        start_idx, start_len = self._find_start_token_in_text(delta_text)
        if start_idx != -1 and delta_text.strip() == delta_text[start_idx:start_len]:
            # Delta is only the start token — skip it entirely.
            return None
        if start_idx == 0:
            # Delta starts with the start token — skip past it.
            after_start = delta_text[start_len:]
            return DeltaMessage(reasoning=after_start or None)

        # Check if the delta ends with a partial marker prefix (e.g. "</"
        # which could be the beginning of the end marker).  Buffer the
        # ambiguous text to avoid leaking partial markers into reasoning.
        ambiguous_suffix = self._ends_with_marker_prefix(delta_text)
        if ambiguous_suffix:
            if ambiguous_suffix == delta_text:
                # Entire delta is a possible marker prefix — buffer it all.
                self._pending = delta_text
                return None
            else:
                # Emit the non-ambiguous part as reasoning, buffer the rest.
                to_emit = delta_text[: -len(ambiguous_suffix)]
                self._pending = ambiguous_suffix
                return DeltaMessage(reasoning=to_emit) if to_emit else None

        return DeltaMessage(reasoning=delta_text)

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        """Count tokens within thinking spans."""
        # Fast path: if single-token markers are present, count using IDs.
        if self._has_start_token_id and self._has_end_token_id:
            count = 0
            depth = 0
            for token_id in token_ids:
                if token_id == self.think_start_token_id:
                    depth += 1
                    continue
                if token_id == self.think_end_token_id:
                    if depth > 0:
                        depth -= 1
                    continue
                if depth > 0:
                    count += 1
            # Only trust the fast-path result if we actually found markers.
            if depth == 0 and count > 0:
                return count

        # Text-based fallback: handles both text-form multi-token markers
        # and special-token single-token markers in decoded text.
        try:
            text = self._decode_ids(token_ids)
            if not text:
                return 0
            n_tokens = len(token_ids)
            if n_tokens == 0:
                return 0

            total_chars = len(text)
            if total_chars == 0:
                return 0
            chars_per_token = total_chars / n_tokens

            count = 0
            depth = 0
            i = 0
            while i < len(text):
                found_start = False
                for start_tok in (self.THINK_START_TEXT, self.THINK_START_SPECIAL):
                    if text[i:].startswith(start_tok):
                        depth += 1
                        i += len(start_tok)
                        found_start = True
                        break
                if found_start:
                    continue

                found_end = False
                for end_tok in (self.THINK_END_TEXT, self.THINK_END_SPECIAL):
                    if text[i:].startswith(end_tok):
                        if depth > 0:
                            depth -= 1
                        i += len(end_tok)
                        found_end = True
                        break
                if found_end:
                    continue

                if depth > 0:
                    count += 1
                i += 1

            # Scale per-character count to approximate token count.
            if chars_per_token > 1:
                count = max(1, round(count / chars_per_token))

            return count
        except Exception:
            return 0
