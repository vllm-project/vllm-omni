# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from inclusionAI/Ming (MIT); see LICENSE.Ming and NOTICE.
# https://github.com/inclusionAI/Ming/blob/e58533db227031990c5a6864dcf5f08fb53ed0d2/front/text_segment_cut.py
# Positions are omitted because audio needs only fragments.
import regex as re

from .front_toolkit import PROTECTED_LEXEME
from .text_processing import append_text_fragment, get_semantic_length, has_valid_content


def split_long_fragment(text_fragment: str, max_len: int) -> list[str]:
    """
    分割超长的文本片段。优先沿标点(clause边界)切,避免从字中间断开。
    仅当单个 clause 自身仍超 max_len 时,才退化到字/词级切分(保底不卡死)。
    """
    fragment_semantic_len = get_semantic_length(text_fragment)
    if fragment_semantic_len <= max_len:
        return [text_fragment]

    # —— 第一优先级:沿标点切 clause ——
    # 标点(中文，、；。！？ + 英文 ,;!?!) 及列表分隔(空格+[-•·]) 作为切分边界。
    # 用 split 保留分隔符并并回前一 clause, 使子串拼接 == 原文(位置校准仍能 find 到)。
    clause_pattern = r"([，、；。！？,;!?]|\s+[-•·])"
    parts = re.split(clause_pattern, text_fragment)
    merged_clauses = []
    for part in parts:
        if part == "":
            continue
        if merged_clauses and re.fullmatch(clause_pattern, part):
            merged_clauses[-1] += part  # 标点/分隔 并回前一段
        else:
            merged_clauses.append(part)

    fragments = []
    current_fragment = ""

    for clause in merged_clauses:
        clause_semantic_len = get_semantic_length(clause)

        # 单个 clause 自身超 max_len: 保底字/词级切分(避免无限卡死)
        if clause_semantic_len > max_len:
            if current_fragment:
                fragments.append(current_fragment)
                current_fragment = ""
            fragments.extend(_split_by_char_word(clause, max_len))
            continue

        current_semantic_len = get_semantic_length(current_fragment)
        if current_semantic_len + clause_semantic_len <= max_len:
            current_fragment += clause
        else:
            if current_fragment:
                fragments.append(current_fragment)
            current_fragment = clause

    if current_fragment:
        fragments.append(current_fragment)

    return fragments


def _split_by_char_word(text: str, max_len: int) -> list[str]:
    """
    保底切分:按字(汉字)/词(英文)等基本语义单元切,用于单个 clause 仍超长时。
    """
    fragments = []
    current = ""
    tokens = re.findall(r"(\s+|[，、；。！？,;!?]|[\u4e00-\u9fa5]|[a-zA-Z]+|[^a-zA-Z\u4e00-\u9fa5]+)", text)
    for tok in tokens:
        tok_len = get_semantic_length(tok)
        cur_len = get_semantic_length(current)
        if cur_len + tok_len <= max_len:
            current += tok
        else:
            if current:
                fragments.append(current)
            if tok_len > max_len:
                fragments.append(tok)
                current = ""
            else:
                current = tok
    if current:
        fragments.append(current)
    return fragments


def cut_text_by_semantic_length(text: str, max_semantic_length: int = 50, min_tail_length: int = 5) -> list[str]:
    """Split at sentence/clause boundaries and preserve short terminal tails."""
    if not has_valid_content(text):
        return []
    # Atomic alphabetic placeholders keep URLs/abbreviations inside one
    # semantic unit, including when an oversized clause is split by words.
    # Select a prefix absent from input so literal user text cannot collide.
    prefix = "MINGPROTECTEDTOKEN"
    while prefix in text:
        prefix += "X"
    protected: dict[str, str] = {}

    def protect(match: re.Match) -> str:
        value = match.group()
        lexeme = value.rstrip(".!?") if value.lower().startswith(("http://", "https://")) else value
        suffix = format(len(protected), "x").translate(str.maketrans("0123456789", "ABCDEFGHIJ"))
        marker = prefix + suffix + "Z"
        protected[marker] = lexeme
        return marker + value[len(lexeme) :]

    def restore(fragment: str) -> str:
        for marker, lexeme in protected.items():
            fragment = fragment.replace(marker, lexeme)
        return fragment.replace(placeholder, ".")

    placeholder = "##DOT##"
    processed = PROTECTED_LEXEME.sub(protect, text)
    processed = re.sub(r"(\d)\.(\d)", r"\1" + placeholder + r"\2", processed)
    for _ in range(3):
        processed = re.sub(r"([A-Z])\.([A-Z])", r"\1" + placeholder + r"\2", processed)
    processed = processed.replace("\n", " ").replace("。，", "。")
    if get_semantic_length(processed) <= max_semantic_length:
        return [restore(processed)]
    normalized = processed.replace(".", "。").replace("!", "！").replace("?", "？").replace(",", "，")
    sentences = []
    current = ""
    for char in normalized:
        current += char
        if char in "。！？":
            if current.strip():
                sentences.append(current.strip())
            current = ""
    if current:
        tail = current.strip()
        if not tail.endswith(("。", "！", "？")):
            tail += "。"
        sentences.append(tail)
    fragments = []
    for sentence in sentences:
        clauses = []
        current = ""
        for char in sentence:
            current += char
            if char in "，;；":
                clause = current.strip()
                if clause and has_valid_content(clause):
                    clauses.append(clause)
                elif clause and clauses:
                    clauses[-1] += clause
                current = ""
        clause = current.strip()
        if clause and has_valid_content(clause):
            clauses.append(clause)
        elif clause and clauses:
            clauses[-1] += clause
        index = 0
        while index < len(clauses):
            clause = clauses[index]
            length = get_semantic_length(clause)
            if length < min_tail_length and index + 1 < len(clauses):
                combined = clause + clauses[index + 1]
                if get_semantic_length(combined) <= max_semantic_length:
                    append_text_fragment(fragments, combined, max_semantic_length, min_tail_length)
                    index += 2
                    continue
            parts = split_long_fragment(clause, max_semantic_length) if length > max_semantic_length else [clause]
            for part in parts:
                append_text_fragment(fragments, part, max_semantic_length, min_tail_length)
            index += 1
    return [restore(fragment) for fragment in fragments]
