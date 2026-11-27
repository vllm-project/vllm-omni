# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""What of the thinker's text is spoken, and how (framework-independent).

* Tool calls are never spoken: only the text before the first ``<tool_call>``.
* Markdown symbols and bracketed asides are dropped; the sound tags the talkers
  were trained on are kept whole (``[laughter] [breath] [sigh] [cough]
  [quick_breath]``, ``<laughter>..</laughter>``, ``<strong>..</strong>``).
* While the thinker is still writing, anything that may still change is held
  back, so the spoken text only ever grows.
"""

from __future__ import annotations

import regex as re

TOOL_CALL_OPEN = "<tool_call>"
POINT_TAGS = frozenset({"[laughter]", "[breath]", "[sigh]", "[cough]", "[quick_breath]"})
SPAN_TAGS = frozenset({"laughter", "strong"})
SOUND_TAG_RE = re.compile(r"\[(?:laughter|breath|sigh|cough|quick_breath)\]|</?(?:laughter|strong)>")
_CLOSE = {"[": "]", "<": ">", "(": ")", "（": "）", "【": "】"}
_MAX_TAG_CHARS = 40
_STRONG_END = re.compile(r"[，。！？、,.!?;；:：]")
_SPOKEN_SYMBOLS = {"→": ", so ", "⇒": ", so ", "≥": " at least ", "≤": " at most "}


def speakable_text(text: str, final: bool) -> str:
    """The part of the thinker text that may be spoken: everything before the
    first ``<tool_call>``. While streaming, a trailing prefix of ``<tool_call>``
    is held back until the next characters show whether it is a call."""
    i = text.find(TOOL_CALL_OPEN)
    if i >= 0:
        return text[:i]
    if not final:
        for k in range(len(TOOL_CALL_OPEN) - 1, 0, -1):
            if text.endswith(TOOL_CALL_OPEN[:k]):
                return text[:-k]
    return text


def clean_for_speech(text: str, final: bool) -> str:
    """Drop markdown symbols and bracketed asides, keep whitelisted sound tags.

    A bracket that does not close within 40 characters is not a tag or an aside,
    so its text is spoken; one that may still close is held back. ``<strong>``
    closes at the next punctuation mark. When ``final``, open spans are closed.
    The result only grows as more text arrives."""
    out: list[str] = []
    open_spans: list[str] = []
    buf = text

    def emit_tag(tag: str) -> None:
        if tag in POINT_TAGS:
            out.append(tag)
            return
        m = re.fullmatch(r"<(/?)([a-z_\-]+)>", tag)
        if not m or m.group(2) not in SPAN_TAGS:
            return
        name = m.group(2)
        if not m.group(1):
            if name not in open_spans:
                open_spans.append(name)
                out.append(tag)
        elif name in open_spans:
            open_spans.remove(name)
            out.append(tag)

    def emit_text(value: str) -> None:
        for sym, spoken in _SPOKEN_SYMBOLS.items():
            value = value.replace(sym, spoken)
        if "strong" in open_spans:
            m = _STRONG_END.search(value)
            if m:
                if m.start():
                    out.append(value[: m.start()])
                out.append("</strong>")
                open_spans.remove("strong")
                value = value[m.start() :]
        if value:
            out.append(value)

    while buf:
        hits = [k for k in (buf.find(c) for c in [*_CLOSE, "*", "#", "`"]) if k >= 0]
        i = min(hits) if hits else -1
        if i < 0:
            emit_text(buf)
            buf = ""
            break
        if i > 0:
            emit_text(buf[:i])
            buf = buf[i:]
        c = buf[0]
        if c in "*#`":
            buf = buf[1:]
            continue
        j = buf.find(_CLOSE[c], 1)
        if j < 0 or j > _MAX_TAG_CHARS:
            if j > _MAX_TAG_CHARS or len(buf) > _MAX_TAG_CHARS:
                buf = buf[1:]  # not a tag: speak what follows the bracket
                continue
            break  # may still close: hold it back
        tag, buf = buf[: j + 1], buf[j + 1 :]
        if c in "[<":
            emit_tag(tag)
    if final:
        if buf:
            rest = re.sub(r"[\[<(（【].*$", "", buf)
            if rest:
                emit_text(rest)
        for name in reversed(open_spans):
            out.append(f"</{name}>")
    return "".join(out)


def strip_sound_tags(text: str) -> str:
    """Display text: the reply without sound tags."""
    return SOUND_TAG_RE.sub("", text)
