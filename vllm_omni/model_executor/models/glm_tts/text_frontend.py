# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight text frontend aligned with the official GLM-TTS preprocessing."""

from __future__ import annotations

import json
import os
import re

try:
    import emoji
except ImportError:
    emoji = None  # type: ignore[assignment]

try:
    import inflect
except ImportError:
    inflect = None  # type: ignore[assignment]

try:
    from tn.chinese.normalizer import Normalizer as ZhNormalizer
except ImportError:
    ZhNormalizer = None

try:
    from tn.english.normalizer import Normalizer as EnNormalizer
except ImportError:
    EnNormalizer = None


CHINESE_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fff]+")
PUNCTUATION_CHARS = r"。？！；：、.?!;:，,"


def contains_chinese(text: str) -> bool:
    return bool(CHINESE_CHAR_PATTERN.search(text))


def markdown_norm(markdown_text: str) -> str:
    markdown_text = re.sub(r"^(\d+)\. ", r"\1。", markdown_text)
    return markdown_text.replace("\\n", "\n")


def multi_line_process(plain_text: str) -> str:
    lines: list[str] = []
    for line in plain_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line[-1] not in {".", "!", "?", ";", ":", "：", "。", "！", "？", "；", "，"}:
            line = f"{line}{'。' if contains_chinese(line) else '. '}"
        lines.append(line)
    return "".join(lines) if contains_chinese("".join(lines)) else " ".join(lines)


def remove_bracket(text: str, lang: str = "zh") -> str:
    brackets_to_remove = [
        ("(", ")"),
        ("（", "）"),
        ("【", "】"),
        ("「", "」"),
        ("`", "`"),
        ("《", "》"),
        ("『", "』"),
        ("{", "}"),
        ("[", "]"),
    ]
    if lang != "en":
        brackets_to_remove.append(("‘", "’"))
    for left, right in brackets_to_remove:
        text = text.replace(left, "").replace(right, "")
    return text


def number_to_chinese(number: int | float | str) -> str:
    units = ["", "十", "百", "千", "万", "十万", "百万", "千万", "亿", "十亿", "百亿", "千亿"]
    nums = "零一二三四五六七八九"
    if isinstance(number, (float, str)):
        str_num = str(float(number))
        if "." in str_num:
            int_part, decimal_part = str_num.split(".")
            if decimal_part.strip("0") == "":
                return number_to_chinese(int(int_part))
            chinese_int = number_to_chinese(int(int_part)) if int(int_part) != 0 else "零"
            chinese_decimal = "".join(nums[int(d)] for d in decimal_part)
            return f"{chinese_int}点{chinese_decimal}"

    number = int(number)
    if number == 0:
        return "零"
    if number < 0:
        return "负" + number_to_chinese(abs(number))

    result: list[str] = []
    unit_position = 0
    zero_flag = False
    last_unit = ""
    while number > 0:
        num = number % 10
        current_unit = units[unit_position] if unit_position < len(units) else ""
        if num == 0:
            if not zero_flag and result and last_unit not in ["万", "亿"]:
                result.append(nums[num])
                zero_flag = True
        else:
            if current_unit:
                result.append(current_unit)
            result.append(nums[num])
            zero_flag = False
            last_unit = current_unit
        unit_position += 1
        number //= 10

    result.reverse()
    text = "".join(result)
    text = re.sub(r"^一十", "十", text)
    text = re.sub(r"零+", "零", text)
    text = re.sub(r"零$", "", text)
    return text


def tn_scientific_notation(text: str) -> str:
    pattern = re.compile(r"(.*?)(-?\d+(?:\.\d+)?)[*x×](\d+(?:\.\d+)?)\^(-?\d+(?:\.\d+)?)(.*?)$")
    match = pattern.match(text)
    if not match:
        return text
    prefix, base, multiplier, exponent, suffix = match.groups()
    description = f"{number_to_chinese(base)}乘{number_to_chinese(multiplier)}的{number_to_chinese(exponent)}次方"
    return f"{prefix}{description}{suffix}"


def replace_asterisk_with_multiply(text: str, lang: str) -> str:
    if lang == "en":
        replace_name = "multiply"
        replace_number = "0-9"
    else:
        replace_name = "乘"
        replace_number = "一二三四五六七八九十百千万亿"

    rule0 = rf"(?<=[{replace_number}a-zA-Z])\s*\*\s*(?=[{replace_number}a-zA-Z])"
    rule1 = rf"(?<=[{replace_number}a-zA-Z]\))\s*\*\s*(?=[{replace_number}a-zA-Z])"
    rule2 = rf"(?<=[{replace_number}a-zA-Z])\s*\*\s*(?=\([{replace_number}a-zA-Z])"
    rule3 = rf"(?<=[{replace_number}a-zA-Z]\))\s*\*\s*(?=\([{replace_number}a-zA-Z])"
    return re.sub(f"{rule0}|{rule1}|{rule2}|{rule3}", replace_name, text).replace("*", "")


def special_replace(text: str) -> str:
    text = text.replace("\\", "")
    text = f" {text} "
    text = re.sub(r"(?<=\W)额(?=\W)", "呃", text)
    text = re.sub(r"[~～]+", "。", text)
    text = replace_asterisk_with_multiply(text, "zh")
    return text.strip()


def spell_out_number(text: str, inflect_parser: object | None) -> str:
    if inflect_parser is None:
        return text
    new_text: list[str] = []
    start = None
    for idx, char in enumerate(text):
        if not char.isdigit():
            if start is not None:
                new_text.append(inflect_parser.number_to_words(text[start:idx]))
                start = None
            new_text.append(char)
        elif start is None:
            start = idx
    if start is not None and start < len(text):
        new_text.append(inflect_parser.number_to_words(text[start:]))
    return "".join(new_text)


def replace_space(text: str) -> str:
    alphanumeric_pattern = r"[a-zA-Z0-9]"
    punctuation_pattern = r"[.,!?;:]"
    text = re.sub(r" +", " ", text)
    result = ""
    idx = 0
    while idx < len(text):
        current_char = text[idx]
        if current_char != " ":
            result += current_char
            idx += 1
            continue
        prev_char = text[idx - 1] if idx > 0 else ""
        next_char = text[idx + 1] if idx + 1 < len(text) else ""
        if re.match(alphanumeric_pattern, prev_char) and re.match(alphanumeric_pattern, next_char):
            result += " "
        elif re.match(punctuation_pattern, prev_char) and re.match(alphanumeric_pattern, next_char):
            result += " "
        idx += 1
    return result


def normalize_punctuation(text: str, punctuation_chars: str) -> str:
    text = replace_space(text)
    text = re.sub(rf"([{punctuation_chars}])\1+", r"\1", text)
    text = re.sub(rf"([{punctuation_chars}])(?=[{punctuation_chars}])", "", text)
    text = text.replace("#", "")
    text = text.replace("！", "。")
    text = text.replace("!", ".")
    return text


def ensure_proper_ending(text: str) -> str:
    if not text:
        return text
    if text[-1] in "？?":
        return text
    if text[-1] in PUNCTUATION_CHARS:
        return f"{text[:-1]}{'。' if contains_chinese(text) else '.'}"
    return f"{text}{'。' if contains_chinese(text) else '.'}"


class GLMTTSTextFrontend:
    """Subset of the official GLM-TTS text frontend used by the adapter."""

    def __init__(self) -> None:
        self.inflect_parser = inflect.engine() if inflect is not None else None
        self.zh_tn_model = (
            ZhNormalizer(
                remove_erhua=False,
                full_to_half=True,
                remove_interjections=False,
                overwrite_cache=True,
            )
            if ZhNormalizer is not None
            else None
        )
        self.en_tn_model = EnNormalizer() if EnNormalizer is not None else None
        self._custom_replacements = self._load_custom_replacements()

    @staticmethod
    def _load_custom_replacements() -> list[tuple[str, str]]:
        custom_replace_path = os.path.join(os.path.dirname(__file__), "configs", "custom_replace.jsonl")
        if not os.path.exists(custom_replace_path):
            return []
        replacements: list[tuple[str, str]] = []
        with open(custom_replace_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                replacements.append((item["origin"], item["new"]))
        return replacements

    def pre_replace(self, sentence: str) -> str:
        sentence = tn_scientific_notation(sentence)
        sentence = re.sub(r"(?<=\d)\s*-\s*(?=\d)", "减", sentence)
        sentence = sentence.replace("-", "")
        sentence = re.sub(r"咯([" + re.escape(PUNCTUATION_CHARS) + r"])", r"喽\1", sentence)
        for origin, new in self._custom_replacements:
            sentence = sentence.replace(origin, new)
        return sentence

    def post_replace(self, sentence: str) -> str:
        sentence = remove_bracket(sentence)
        sentence = sentence.replace(" - ", "，")
        sentence = sentence.replace("——", "，")
        sentence = re.sub(r"[,:：;；、]+", "，", sentence)
        sentence = re.sub(r"[.…]+", "。", sentence)
        sentence = re.sub(r"[_·]+", "", sentence)
        sentence = re.sub(r"""['"‘’“”|]+""", "", sentence)

        symbol_map = {
            "†": "，",
            "²": "平方",
            "³": "立方",
            "/": "每",
            "~": "到",
            "～": "到",
            "①": "一",
            "②": "二",
            "③": "三",
            "④": "四",
            "⑤": "五",
            "⑥": "六",
            "⑦": "七",
            "⑧": "八",
            "⑨": "九",
            "⑩": "十",
            "α": "阿尔法",
            "β": "贝塔",
            "γ": "伽玛",
            "Γ": "伽玛",
            "δ": "德尔塔",
            "Δ": "德尔塔",
            "△": "德尔塔",
            "ε": "艾普西龙",
            "ζ": "捷塔",
            "η": "依塔",
            "θ": "西塔",
            "Θ": "西塔",
            "ι": "艾欧塔",
            "κ": "喀帕",
            "λ": "拉姆达",
            "Λ": "拉姆达",
            "μ": "缪",
            "ν": "拗",
            "ξ": "克西",
            "Ξ": "克西",
            "ο": "欧米克伦",
            "π": "派",
            "Π": "派",
            "ρ": "肉",
            "ς": "西格玛",
            "Σ": "西格玛",
            "σ": "西格玛",
            "τ": "套",
            "υ": "宇普西龙",
            "φ": "服艾",
            "Φ": "服艾",
            "χ": "器",
            "ψ": "普赛",
            "Ψ": "普赛",
            "ω": "欧米伽",
            "Ω": "欧米伽",
            "□": "方框",
            ">": "大于",
            "<": "小于",
            "∈": "属于",
            "∉": "不属于",
            "∪": "并",
            "∩": "交",
            "⊥": "垂直",
            "∥": "平行",
            "≠": "不等于",
            "∵": "因为",
            "∴": "所以",
            "∅": "空集",
            "⊂": "真包含于",
            "⊃": "包含",
            "⊆": "包含于",
            "⊇": "真包含",
            "⊄": "不属于",
            "⊅": "非超集",
            "⊈": "不属于",
            "⊉": "非超集",
        }
        for old, new in symbol_map.items():
            sentence = sentence.replace(old, new)
        return sentence

    def text_normalize(self, text: str | None) -> str | None:
        if text is None:
            return None
        text = self._preprocess_text(text)
        if contains_chinese(text):
            text = self._normalize_chinese_text(text).lower()
        else:
            text = self._normalize_english_text(text)
        text = normalize_punctuation(text, PUNCTUATION_CHARS)
        return ensure_proper_ending(text)

    def _preprocess_text(self, text: str) -> str:
        text = markdown_norm(text)
        text = multi_line_process(text)
        if emoji is not None:
            text = emoji.replace_emoji(text, replace="")
        return re.sub(r"(?<=[a-zA-Z])-(?=[a-zA-Z])", " ", text)

    def _normalize_chinese_text(self, text: str) -> str:
        text = self.pre_replace(text)
        if self.zh_tn_model is not None:
            text = self.zh_tn_model.normalize(text)
        text = special_replace(text)
        return self.post_replace(text).strip()

    def _normalize_english_text(self, text: str) -> str:
        text = text.replace("'", "’")
        if self.en_tn_model is not None:
            text = self.en_tn_model.normalize(text)
        text = remove_bracket(text, "en")
        text = replace_asterisk_with_multiply(text, "en")
        text = text.replace("—", " ")
        text = text.replace("’", "'")
        text = spell_out_number(text, self.inflect_parser)
        text = re.sub(r"\s+", " ", text)
        keep_punctuation = r"\.,!\?'\:;"
        pattern = rf"[^\w\s{keep_punctuation}]"
        text = re.sub(pattern, "", text)
        text = text.lower()
        text = re.sub(r"\.+", ".", text)
        text = re.sub(r"\,+", ",", text)
        text = re.sub(r"!+", "!", text)
        text = re.sub(r"\?+", "?", text)
        text = re.sub(r"'+", "'", text)
        text = re.sub(r":+", ":", text)
        text = re.sub(r";+", ";", text)
        text = re.sub(r"\s*([.,?!':;])\s*", r"\1 ", text)
        return text.strip()
