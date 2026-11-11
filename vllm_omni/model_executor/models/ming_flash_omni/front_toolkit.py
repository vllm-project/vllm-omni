# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from collections.abc import Iterable

import regex as re

TOKENIZE_PATTERN = r"(?:[a-zA-Z]\.)+|[a-zA-Z]+(?:['\-][a-zA-Z]+)*|\d+(?:\.\d+)?|[\u4e00-\u9fff]|\s+|\S"
PROTECTED_LEXEME = re.compile(
    r"https?://[^\s，。！？；、]+|(?:[A-Za-z]\.){2,}|\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr)\.", re.IGNORECASE
)


def tokenize_mixed_text_iterator(text_iterator: Iterable[str]):
    """
    Args:
        text_iterator (Iterable[str]): 一个产出字符串的迭代器。
            例如，一个打开的文件对象，或一个生成器表达式。
    Yields:
        Iterator[str]: 逐一产出文本中的标记（单词、中文、标点、空格等）。
    """
    # 函数本身就是一个生成器，将从 re.finditer 返回的迭代器中逐一产出结果
    for chunk in text_iterator:
        # re.finditer 返回一个迭代器，而不是列表
        for match in re.finditer(TOKENIZE_PATTERN, chunk):
            yield match.group(0)


def clean_text_formatting(text):
    if not text:
        return text

    # 定义开头需要被替换为空格的“杂质标点”
    # 保留了：() {} [] （） 【】 "" '' “” ‘’ 《》 等成对符号
    # 替换掉：， 。 ！ ？ ； ： 、 …… 等容易出现在开头且没意义的标点
    leading_punct_pattern = r"^[，。！？；：、……\.\!\?\-\—]+"

    # 替换开头标点
    text = re.sub(leading_punct_pattern, " ", text)
    # 去除首尾多余空格
    return text.strip()


def chinese_punc_en2cn(text):
    """
    含中文文本清理：包含中文即处理，多个堆叠标点最终只保留一个。
    1. 增加 URL和小数点 保护，防止链接内部符号被误转。
    2. 增强标点优先级逻辑，确保顿号（、）正确去重。
    3. 修复异类标点堆叠的匹配规则。
    4、修复开头符号
    """
    # 仅检查是否包含中文，无中文则直接返回原文本
    if not text or not re.search(r"[\u4e00-\u9fa5]", text):
        return clean_text_formatting(text)

    # URL 占位保护
    urls = re.findall(r"https?://[^\s，。！？]+", text)
    for i, url in enumerate(urls):
        text = text.replace(url, f"URLHOLDER{i}")

    # 保护数字间的小数点，防止被 punc_map 误转为 。
    text = re.sub(r"(?<=\d)\.(?=\d)", "DOTPLACEHOLDER", text)

    # 基础符号转换
    text = text.replace("...", "……")
    punc_map = {
        ",": "，",
        ".": "。",
        ";": "；",
        ":": "：",
        "!": "！",
        "?": "？",
        "(": "（",
        ")": "）",
        "[": "【",
        "]": "】",
        "<": "《",
        ">": "》",
        "_": "——",
    }
    map_pattern = re.compile("|".join(re.escape(k) for k in punc_map.keys()))
    text = map_pattern.sub(lambda m: punc_map[m.group(0)], text)

    # 还原小数点
    text = text.replace("DOTPLACEHOLDER", ".")

    # 修复：引号计数与转换 (改用闭包计数，避免 := 运算符在 re.sub 中的潜在崩溃)
    dq_count = [0]
    sq_count = [0]

    def dq_sub(_):
        dq_count[0] += 1
        return "“" if dq_count[0] % 2 != 0 else "”"

    def sq_sub(_):
        sq_count[0] += 1
        return "‘" if sq_count[0] % 2 != 0 else "’"

    text = re.sub(r'"', dq_sub, text)
    text = re.sub(r"'", sq_sub, text)

    # 彻底消除标点周边的空格
    ch_puncs = r"，。！？；：“”‘’（）【】《》——……、"
    text = re.sub(rf"[ \t\u00A0\u3000]*([{ch_puncs}])[ \t\u00A0\u3000]*", r"\1", text)

    # 标点递归清理
    while True:
        old_text = text

        # Interpolate the punctuation class to preserve repeated Latin letters.
        text = re.sub(rf"([{ch_puncs}])\1+", r"\1", text)

        # 弱标点(，；、) 遇到 强标点(。！？) 时删除弱标点
        text = re.sub(r"[，；、]+([。！？])", r"\1", text)
        text = re.sub(r"([。！？])[，；、]+", r"\1", text)

        # 处理 逗号/分号 与 顿号 的堆叠
        text = re.sub(r"[，；]+[、]|[、][，；]+", "、", text)

        # 强标点间唯一化
        text = re.sub(r"([。！？])[。！？]+", r"\1", text)

        # 剩余异类弱标点堆叠唯一化
        text = re.sub(r"([，；、])[，；、]+", r"\1", text)

        if old_text == text:
            break

    # 压缩汉字/单词间连续空格
    text = re.sub(r"[ \t\u00A0\u3000]+", " ", text)

    # 还原 URL
    for i, url in enumerate(urls):
        text = text.replace(f"URLHOLDER{i}", url)

    return clean_text_formatting(text)


# 弱标点结尾视为未收束，升格为句末标点（替换而非追加：
# 纯英文不经 chinese_punc_en2cn 的标点去重，追加会留下 ",." 漏网）
WEAK_TAIL_PUNCT = "，,、;；"


def add_period(text):
    """尾部无标点则添加句号；汉字结尾加'。'，否则加'.'。弱标点(，,、;；)结尾升格为句末标点"""
    stripped = text.rstrip()
    if not stripped:  # 空串或全空白
        return "."

    last_char = stripped[-1]

    # 弱标点结尾（逗号/顿号/分号）：升格为句末标点
    if last_char in WEAK_TAIL_PUNCT:
        has_chinese = bool(re.search(r"[一-鿿]", stripped))  # U+4E00-U+9FFF 汉字区
        return stripped[:-1] + ("。" if has_chinese else ".")

    # 如果最后一个字符不是字母或数字，则认为它已是标点符号，直接返回
    if not last_char.isalnum():
        return stripped

    # 判断是否为汉字（基本汉字区和扩展A区）
    is_chinese = "\u4e00" <= last_char <= "\u9fff" or "\u3400" <= last_char <= "\u4dbf"
    suffix = "。" if is_chinese else "."
    return stripped + suffix


def fallback_replace_non_alnum(text, replacement="，"):
    """
    兜底方案：将非汉字、英文、数字及基础TTS标点的字符替换为指定符号。
    并合并连续的替换符号。
    保留中文基础标点：，。！？；：、
    保留英文基础标点：,.!?;:
    保留小数点：.和空格等
    """
    # 白名单新增: () （） " “ ” (半角/全角括号 + 半角/中文双引号) 不再被替换成逗号
    text = re.sub(
        r"[^\u4e00-\u9fffA-Za-z0-9，。！？；：、,.!?;:\. \u2019\u2018'\"()（）\u201C\u201D]", replacement, text
    )
    text = re.sub(re.escape(replacement) + "+", replacement, text).strip()
    return text


def normalize_text_kit(text):
    """Normalize unsupported characters, Chinese punctuation, and whitespace."""
    text = fallback_replace_non_alnum(text)
    text = chinese_punc_en2cn(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_markdown_code(text):
    """
    清除 Markdown 行内代码标记（反引号）。
    支持单反引号 `code` 和双反引号 ``code`` 两种格式。
    """
    # 双反引号优先匹配（避免单反引号先匹配到内层）
    text = re.sub(r"``(.+?)``", r"\1", text)
    # 单反引号
    text = re.sub(r"`(.+?)`", r"\1", text)

    # 清除残留的未配对反引号
    text = text.replace("`", "")
    return text


def collapse_markup_residue(text):
    """标记删除后的收尾收敛：多空白合一、连串分隔符（含空格夹杂）折叠为单个逗号。"""
    text = re.sub(r"[\s]{2,}", " ", text)
    return re.sub(r"(?:[ \t]*[,，;；:：][ \t]*){2,}", "，", text)


_BLOCK_TAG_NAMES = ("script", "style", "svg")
_MATH_GATE_CHARS = re.compile(r"[\\^_]")
_MATH_GATE_SYMBOLS = re.compile(r"[∑∏∫∂∇√≈≠≤≥±×÷∝∈∉∀∃∞→←↔·°α-ωΑ-Ω]")
# ASCII attribute syntax avoids treating comparisons such as <b 且 c> as HTML.
# Syntactically valid <b> / <b disabled> remain HTML: that ambiguity cannot be
# resolved without a markup-mode API. Non-ASCII prose in angle brackets is kept.
_TAG_SYNTAX = re.compile(
    r"\s*(/?)[ \t\r\n]*([A-Za-z][A-Za-z0-9:_-]*)"
    r"(?:\s+[A-Za-z_:][A-Za-z0-9:_.-]*"
    r"(?:\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s\"'<>`=]+))?)*\s*/?\s*"
)
_CURRENCY_NUMBER = re.compile(r"\d+(?:[.,]\d+)*[.,]?\s*")
_CURRENCY_UNIT = re.compile(
    r"\d+(?:[.,]\d+)*\s*(?:USD|dollars?|cents?|"
    r"/(?:months?|mo|years?|yr|weeks?|days?|hours?|hr|minutes?|min|seconds?|sec|tokens?|requests?))\s*",
    re.IGNORECASE,
)
_MARKUP_START = re.compile(r"[<&`$\\]")
_ENTITY_PREFIX = re.compile(r"&(?:[A-Za-z][A-Za-z0-9]{0,10}|#\d{0,6})?")
_ENTITY_COMPLETE = re.compile(r"&(?:[A-Za-z][A-Za-z0-9]{0,10}|#\d{1,6});")


def math_like(content: str) -> bool:
    """Conservative math gate; a Windows path is ordinary readable content."""
    if re.match(r"^(?:[A-Za-z]:[\\/]|\\\\)", content):
        return False
    return bool(_MATH_GATE_CHARS.search(content) or _MATH_GATE_SYMBOLS.search(content))


class WebMarkupScrubber:
    """Bounded incremental parser shared by streaming and offline paths.

    Suppress fences, script/style/svg bodies, comments, declarations and math
    ($…$, $$…$$, \\(…\\), \\[…\\]);
    keep ordinary container text, comparisons, prices and inline identifiers.
    Candidate tags/inline code/math are bounded; on overflow they are literal.
    Unfinished comments/blocks/fences suppress through EOF, incomplete ordinary
    tags and inline delimiters stay literal. State is independent of feed sizes.
    """

    _TAG_LIMIT = 600
    _INLINE_LIMIT = 200
    _DISPLAY_LIMIT = 400

    def __init__(self):
        self._out = []
        self.mode = "out"
        self.hold = ""
        self.tag = ""
        self.quote = ""

    def feed(self, delta, final=False):
        # Ordinary text is the hot path. No parser transition is possible
        # without a delimiter, so avoid per-character Python dispatch here.
        if self.mode == "out" and not self.hold and not _MARKUP_START.search(delta):
            self._emit(delta)
        else:
            for ch in delta:
                self._take(ch)
        if final:
            if self.mode in ("fence", "block", "comment"):
                self._emit(" ")
            elif self.mode == "entity" and self.hold.lower() == "&nbsp":
                self._emit(" ")
            else:
                self._emit(self.hold)
            self.mode, self.hold, self.tag, self.quote = "out", "", "", ""
        result, self._out = "".join(self._out), []
        return result

    def _emit(self, text):
        if text:
            self._out.append(text)

    def _literal(self):
        self._emit(self.hold)
        self.mode, self.hold, self.quote = "out", "", ""

    def _take(self, ch):
        if self.mode == "out":
            self._out_mode(ch)
        elif self.mode == "tag":
            self._tag_mode(ch)
        elif self.mode == "block":
            self._block_mode(ch)
        elif self.mode == "fence":
            self._fence_mode(ch)
        elif self.mode == "comment":
            self._comment_mode(ch)
        elif self.mode == "tick":
            self._tick_mode(ch)
        elif self.mode == "dollar":
            self._dollar_mode(ch)
        elif self.mode == "bsl":
            self._bsl_mode(ch)
        elif self.mode == "lmath":
            self._lmath_mode(ch)
        else:
            self._entity_mode(ch)

    def _out_mode(self, ch):
        if ch == "`":
            self.hold += ch
            if len(self.hold) == 3:
                self.mode, self.hold = "fence", ""
            return
        if self.hold:
            if self.hold == "`":
                self.mode = "tick"
                self._tick_mode(ch)
                return
            self._emit(self.hold)  # Double backticks stay literal.
            self.hold = ""
        if ch == "\\":
            self.mode, self.hold = "bsl", "\\"
        elif ch in "<&$":
            self.mode = {"<": "tag", "&": "entity", "$": "dollar"}[ch]
            self.hold = ch
        else:
            self._emit(ch)

    def _tag_mode(self, ch):
        if ch == "<" and not self.quote:
            self._literal()
            self._out_mode(ch)
            return
        self.hold += ch
        if self.hold == "<!--":
            self.mode, self.hold = "comment", ""
            return
        if len(self.hold) == 2 and not (ch.isascii() and ch.isalpha() or ch in "/!? \t\r\n"):
            self._literal()
            return
        if self.quote:
            if ch == self.quote:
                self.quote = ""
        elif ch in "\"'":
            self.quote = ch
        elif ch == ">":
            inner = self.hold[1:-1]
            low = inner.strip().lower()
            match = _TAG_SYNTAX.fullmatch(inner)
            if low.startswith(("!", "?")) or re.match(r"/?\s*(?:https?://|www\.)", low):
                self._emit(" ")
            elif match:
                closing, name = match.group(1), match.group(2).lower()
                if name in _BLOCK_TAG_NAMES and not closing and not inner.rstrip().endswith("/"):
                    self.mode, self.hold, self.tag = "block", "", name
                    return
                self._emit(" ")
            else:
                self._emit(self.hold)
            self.mode, self.hold = "out", ""
            return
        if len(self.hold) > self._TAG_LIMIT:
            self._literal()

    def _block_mode(self, ch):
        # Only a possible closing tag is retained, never the suppressed body.
        if ch == "<":
            self.hold = "<"
        elif self.hold:
            self.hold += ch
            if ch == ">":
                if re.fullmatch(r"<\s*/\s*" + self.tag + r"\s*>", self.hold, re.IGNORECASE):
                    self.mode, self.hold, self.tag = "out", "", ""
                    self._emit(" ")
                else:
                    self.hold = ""
            elif len(self.hold) > self._TAG_LIMIT:
                self.hold = ""

    def _fence_mode(self, ch):
        self.hold = self.hold + ch if ch == "`" else ""
        if self.hold == "```":
            self.mode, self.hold = "out", ""
            self._emit(" ")

    def _comment_mode(self, ch):
        self.hold = (self.hold + ch)[-3:]
        if self.hold == "-->":
            self.mode, self.hold = "out", ""
            self._emit(" ")

    def _tick_mode(self, ch):
        if ch == "`":
            inner = self.hold[1:]
            # Inline identifiers with underscores are readable. Only explicit
            # dollar formulas or TeX-like backslash content opt into math here.
            is_math = ("\\" in inner or "$" in inner) and math_like(inner)
            if is_math:
                self._emit(" ")
            elif "<" in inner or "&" in inner:
                # Backticks do not turn HTML into speakable markup. The inner
                # candidate is bounded and contains no backticks, so this parse
                # cannot recursively enter another inline-code candidate.
                self._emit(WebMarkupScrubber().feed(inner, final=True))
            else:
                self._emit(inner)
            self.mode, self.hold = "out", ""
        elif ch == "\n" or len(self.hold) > self._INLINE_LIMIT:
            self._literal()
            self._out_mode(ch)
        else:
            self.hold += ch

    def _dollar_mode(self, ch):
        display = self.hold.startswith("$$")
        content = self.hold[2:] if display else self.hold[1:]
        # A number alone followed by whitespace is ambiguous: $2 \times 3$
        # and $2 x_1$ must remain buffered until their mathematical close.
        # CJK prose/punctuation or a recognized currency unit disambiguates
        # prices early without releasing numerical formulas as speech. ASCII
        # commas/semicolons/colons alone remain ambiguous ($2, x_1$).
        currency_boundary = ch in "，。！？；：、" or "\u3400" <= ch <= "\u9fff"
        unit_boundary = currency_boundary or ch.isspace() or ch in ".,;:!?"
        amount = _CURRENCY_NUMBER.fullmatch(content) if not display else None
        unit = _CURRENCY_UNIT.fullmatch(content) if not display else None
        if amount and (currency_boundary or content.rstrip().endswith(".") and ch.isspace()) or unit and unit_boundary:
            self._literal()
            self._out_mode(ch)
            return
        if ch == "`" and not display:
            self._literal()
            self._out_mode(ch)
            return
        if ch == "\n" and not display:
            self._literal()
            self._out_mode(ch)
            return
        if len(content) >= (self._DISPLAY_LIMIT if display else self._INLINE_LIMIT) and ch != "$":
            self._literal()
            self._out_mode(ch)
            return
        self.hold += ch
        if self.hold in ("$$",):
            return
        if self.hold == "$$$":
            self._literal()
        elif display and self.hold.endswith("$$"):
            self._emit(" " if math_like(self.hold[2:-2]) else self.hold)
            self.mode, self.hold = "out", ""
        elif not display and ch == "$":
            self._emit(" " if math_like(self.hold[1:-1]) else self.hold)
            self.mode, self.hold = "out", ""

    def _bsl_mode(self, ch):
        # Only explicit TeX openers start math; ordinary paths remain literal.
        if ch in "([":
            self.mode = "lmath"
            self.hold = "\\" + ch
            self.quote = "\\)" if ch == "(" else "\\]"
        else:
            self._emit(self.hold)
            self.mode, self.hold = "out", ""
            self._out_mode(ch)

    def _lmath_mode(self, ch):
        # Explicit TeX pairs may span newlines, including malformed command
        # escapes emitted as a real newline. Keep the existing bounded literal
        # fallback for unfinished or overlong candidates.
        display = self.hold.startswith("\\[")
        closer = self.quote
        if ch == "`":
            self._literal()
            self._out_mode(ch)
            return
        cap = (self._DISPLAY_LIMIT + 4) if display else (self._INLINE_LIMIT + 4)
        if len(self.hold) >= cap:
            self._literal()
            self._out_mode(ch)
            return
        self.hold += ch
        if self.hold.endswith(closer):
            self._emit(" ")
            self.mode, self.hold, self.quote = "out", "", ""

    def _entity_mode(self, ch):
        candidate = self.hold + ch
        if _ENTITY_COMPLETE.fullmatch(candidate):
            self._emit(" ")
            self.mode, self.hold = "out", ""
        elif _ENTITY_PREFIX.fullmatch(candidate):
            self.hold = candidate
        else:
            if self.hold.lower() == "&nbsp":
                self._emit(" ")
                self.mode, self.hold = "out", ""
            else:
                self._literal()
            self._out_mode(ch)


def strip_web_markup(text):
    """Clean a complete text using exactly the incremental parser's rules."""
    if not text:
        return text
    return collapse_markup_residue(WebMarkupScrubber().feed(text, final=True))
