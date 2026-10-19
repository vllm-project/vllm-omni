# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Incremental Ming speech text preprocessing.

Only punctuation/newline boundaries (with 12/8 sentence and 12-element comma thresholds),
or end of input, flush the buffer. There is no hard character-count flush.
"""

from collections.abc import Callable

import regex as re

from .front_toolkit import (
    PROTECTED_LEXEME,
    WebMarkupScrubber,
    add_period,
    preprosess_text,
    tokenize_mixed_text_iterator,
)
from .native_text_segment_cut import cut_text_by_semantic_length


class NativeMingTextFrontend:
    def __init__(self, max_length: int = 50, *, scrub_markup: bool = True):
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.max_length = max_length
        self.buffer: list[str] = []
        self.count = 0
        self.finished = False
        self._prefix = ""
        self._prefix_done = False
        # 网页代码清洗（与 sync 路径的 strip_web_markup 等价，流式状态机）：
        # 围栏/标签/实体等标记噪声在进入分句器之前被吞噬，正文与讲解词直通。
        self._scrubber = WebMarkupScrubber() if scrub_markup else None
        self._pending_ascii: list[str] = []
        self._pending_is_url = False
        self._pending_space: list[str] = []

    def _flush_ascii(self):
        """Tokenize a complete lexical run once, preserving URL/abbreviation boundaries."""
        text = "".join(self._pending_ascii)
        self._pending_ascii.clear()
        self._pending_is_url = False
        start = 0
        for match in PROTECTED_LEXEME.finditer(text):
            for token in tokenize_mixed_text_iterator([text[start : match.start()]]):
                yield token, False
            token = match.group()
            # Once whitespace completes a URL, sentence punctuation belongs to
            # the sentence. Keep those characters and process their boundary.
            url = token.rstrip(".!?") if token.lower().startswith(("http://", "https://")) else token
            yield url, True
            for punctuation in token[len(url) :]:
                yield punctuation, False
            start = match.end()
        for token in tokenize_mixed_text_iterator([text[start:]]):
            yield token, False

    def _stable_tokens(self, delta: str, final: bool):
        # A word, decimal or URL may span arbitrary Thinker deltas. Do not count
        # its incomplete pieces as separate elements or commit its internal dot.
        for char in delta:
            if char.isspace() and char != "\n":
                if self._pending_ascii:
                    yield from self._flush_ascii()
                self._pending_space.append(char)
                continue
            if self._pending_space:
                yield "".join(self._pending_space), False
                self._pending_space.clear()
            if (
                self._pending_ascii
                and self._pending_ascii[-1] == ","
                and not self._pending_is_url
                and not char.isdigit()
            ):
                yield from self._flush_ascii()
            if char.isascii() and not char.isspace():
                self._pending_ascii.append(char)
                if char == "/" and "".join(self._pending_ascii[-8:]).lower().endswith(("http://", "https://")):
                    self._pending_is_url = True
                if not self._pending_is_url and (
                    char in "!?"
                    or (char == "," and (len(self._pending_ascii) < 2 or not self._pending_ascii[-2].isdigit()))
                ):
                    yield from self._flush_ascii()
                continue
            if self._pending_ascii:
                yield from self._flush_ascii()
            yield char, False
        if final:
            if self._pending_ascii:
                yield from self._flush_ascii()
            if self._pending_space:
                yield "".join(self._pending_space), False
                self._pending_space.clear()

    def feed(self, delta: str, final: bool = False) -> list[str]:
        if self.finished:
            raise ValueError("Text arrived after the Ming frontend was finalized")
        if self._scrubber is not None:
            delta = self._scrubber.feed(delta, final=final)
        if not self._prefix_done:
            delta = (self._prefix + delta).lstrip()
            if not final and re.fullmatch(r"[A-Za-z]{0,16}", delta):
                self._prefix = delta
                return []
            delta = re.sub(r"^[A-Za-z]{1,16}\t", "", delta, count=1)
            self._prefix = ""
            self._prefix_done = True
        output = []
        for ele, protected in self._stable_tokens(delta, final):
            if len(ele) == 0:
                continue
            if protected:
                self.buffer.append(ele)
                continue
            should_process = False
            if ele[-1] in "！？。!?" and (len(self.buffer) >= 12 or (self.count > 0 and len(self.buffer) >= 8)):
                should_process = True
                self.buffer.append(ele)
            elif ele[-1] in "，," and len(self.buffer) >= 12:
                should_process = True
                self.buffer.append(ele)
            elif (
                ele[-1] == "."
                and (len(self.buffer) >= 12 or (self.count > 0 and len(self.buffer) >= 8))
                and (not bool(re.search("[0-9]", self.buffer[-1][-1])))
            ):
                should_process = True
                self.buffer.append(ele)
            elif ele[-1] == "\n":
                if len(self.buffer) > 0:
                    if re.search("[\\u4e00-\\u9fff]", "".join(self.buffer)):
                        if re.search("[\\u4e00-\\u9fff]", self.buffer[-1][-1]):
                            ele = "，"
                            self.buffer.append(ele)
                    else:
                        if len(ele) > 1 and re.search("[a-zA-Z]", ele[-2]):
                            ele = ele[:-1] + "."
                        else:
                            ele = ele[:-1]
                        self.buffer.append(ele)
                if len(self.buffer) >= 12 or (self.count > 0 and len(self.buffer) >= 8):
                    should_process = True
            else:
                self.buffer.append(ele)
                continue
            if should_process:
                buffered_text = preprosess_text("".join(self.buffer))
                text_list = cut_text_by_semantic_length(buffered_text, self.max_length)
                self.count += len(text_list)
                output.extend(text_list)
                self.buffer = []
        if final:
            if self.buffer and any(char.isalnum() for token in self.buffer for char in token):
                text = preprosess_text("".join(self.buffer))
                # A bare numeric answer is already a complete utterance. An
                # invented sentence dot can change the model's pronunciation.
                if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)[。！？.!?，,；;：:、]*", text.strip()):
                    text = add_period(text)
                output.extend(cut_text_by_semantic_length(text, self.max_length))
            self.buffer = []
            self.finished = True
        return output


def normalize_native_fragment(
    fragment: str,
    prompt: str = "",
    tn_mode: str = "upstream",
    normalizer: Callable[[str], str] | None = None,
) -> str:
    """Apply the requested TN once without resegmenting native fragments."""
    from .front_toolkit import normalize_text_kit
    from .text_processing import is_chinese, normalize_numbers

    text = fragment
    language = re.search(r"Target language:\s*([^\n.]+)", prompt)
    skip_tn = language is not None and language.group(1).strip().lower() not in {
        "zh",
        "zh-cn",
        "chinese",
        "en",
        "english",
    }
    # The zh/en toolkit's character whitelist would erase these letters, even
    # when automatic language selection supplied an English caption.
    skip_tn = skip_tn or any(char.isalpha() and not (char.isascii() or "\u4e00" <= char <= "\u9fff") for char in text)
    if not skip_tn:
        if not is_chinese(text):
            text = normalize_numbers(text)
        if tn_mode == "upstream":
            if normalizer is None:
                raise ValueError("tn_mode='upstream' requires a normalizer loaded from the Talker checkpoint")
            text = normalizer(text)
            text = normalize_text_kit(text)
        text = text.lstrip("，").strip()
    return text
