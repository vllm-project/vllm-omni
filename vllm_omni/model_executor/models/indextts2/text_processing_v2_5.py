# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Text preparation shared by IndexTTS 2.5 prompt sizing and model prefill."""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache

from .tokenizer_v2_5 import (
    INDEXTTS25_TOKENIZER_FILE,
    encode_indextts25_text,
    lang_to_token,
    normalize_language_code,
)

PRONUNCIATION_ANNOTATION_PATTERN = re.compile(r"<([^|>\n]+)\|([^>\n]+)>")
_SPECIAL_TOKEN_PATTERN = re.compile(r"<\|([^|]+)\|>")
_PINYIN_TONE_PATTERN = re.compile(
    r"(?<![a-z])"
    r"((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?"
    r"(?:[aeiouüv]|[ae]i|u[aio]|ao|ou|i[aue]|[uüv]e|[uvü]ang?|"
    r"uai|[aeiuv]n|[aeio]ng|ia[no]|i[ao]ng)|ng|er)([1-5])",
    re.IGNORECASE,
)
_NAME_PATTERN = re.compile(r"[\u4e00-\u9fff]+(?:[-·—][\u4e00-\u9fff]+){1,2}")
_TECH_TERM_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+")
_ENGLISH_CONTRACTION_PATTERN = re.compile(
    r"(what|where|who|which|how|t?here|it|s?he|that|this)'s",
    re.IGNORECASE,
)
_EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$")

_CHAR_REPLACEMENTS = {
    "：": ",",
    "；": ",",
    ";": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": " ",
    "·": "-",
    "、": ",",
    "...": "…",
    ",,,": "…",
    "，，，": "…",
    "……": "…",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
    ":": ",",
}
# Deliberately prefer the longest replacement. This diverges from the
# insertion-ordered upstream regex so ``，，，`` becomes one ellipsis, not ``,,,``.
_CLEAN_PATTERN = re.compile("|".join(re.escape(key) for key in sorted(_CHAR_REPLACEMENTS, key=len, reverse=True)))


def _is_kana(text: str) -> bool:
    return bool(re.fullmatch(r"[\u3040-\u309f]+", text) or re.fullmatch(r"[\u30a0-\u30ff]+", text))


def apply_pronunciation_annotations(text: str) -> str:
    """Convert ``<word|pronunciation>`` to the official special-token form."""

    def replace(match: re.Match[str]) -> str:
        word = match.group(1)
        pronunciation = match.group(2).upper()
        if _is_kana(pronunciation):
            return f" {pronunciation} "
        token = "SPECIAL_TOKEN_2" if re.search(r"[\u4e00-\u9fff]", word) else "SPECIAL_TOKEN_1"
        return f"<|{token}|>{pronunciation}<|{token}|>"

    return PRONUNCIATION_ANNOTATION_PATTERN.sub(replace, text)


def clean_indextts25_text(text: str) -> str:
    return _CLEAN_PATTERN.sub(lambda match: _CHAR_REPLACEMENTS[match.group()], text)


@lru_cache(maxsize=2)
def _load_wetext_normalizer(lang: str):
    import platform

    if platform.system() == "Linux":
        try:
            if lang == "zh":
                from tn.chinese.normalizer import Normalizer
            else:
                from tn.english.normalizer import Normalizer
        except ModuleNotFoundError as exc:
            if exc.name != "tn":
                raise
            raise RuntimeError(
                f"IndexTTS 2.5 text normalization backend for {lang!r} is unavailable. "
                "Install it with pip install 'vllm-omni[indextts2]'"
            ) from exc
        if lang == "zh":
            return Normalizer(
                remove_interjections=False,
                remove_erhua=False,
                overwrite_cache=False,
            )
        return Normalizer(overwrite_cache=False)

    try:
        from wetext import Normalizer
    except ModuleNotFoundError as exc:
        if exc.name != "wetext":
            raise
        raise RuntimeError(
            f"IndexTTS 2.5 text normalization backend for {lang!r} is unavailable. "
            "Install it with pip install 'vllm-omni[indextts2]'"
        ) from exc
    if lang == "zh":
        return Normalizer(remove_erhua=False, lang="zh", operator="tn")
    return Normalizer(lang="en", operator="tn")


def _uses_chinese_normalizer(text: str) -> bool:
    has_chinese = bool(re.search(r"[\u4e00-\u9fff]", text))
    has_alpha = bool(re.search(r"[a-zA-Z]", text))
    if has_chinese or not has_alpha or _EMAIL_PATTERN.fullmatch(text):
        return True
    return bool(_PINYIN_TONE_PATTERN.search(text))


def _protect_values(
    text: str,
    values: list[str],
    *,
    prefix: str,
) -> tuple[str, list[str]]:
    unique_values = sorted(set(values), key=len, reverse=True)
    for index, value in enumerate(unique_values):
        marker = f"<{prefix}_{chr(ord('a') + index)}>"
        text = text.replace(value, marker)
    return text, unique_values


def _restore_values(
    text: str,
    values: list[str],
    *,
    prefix: str,
    transform: Callable[[str], str] | None = None,
) -> str:
    for index, value in enumerate(values):
        marker = f"<{prefix}_{chr(ord('a') + index)}>"
        text = text.replace(marker, transform(value) if transform else value)
    return text


def _correct_pinyin(pinyin: str) -> str:
    if not pinyin or pinyin[0] not in "jqxJQX":
        return pinyin.upper()
    return re.sub(
        r"([jqx])[uü](n|e|an)*(\d)",
        r"\g<1>v\g<2>\g<3>",
        pinyin,
        flags=re.IGNORECASE,
    ).upper()


def _normalize_zh_or_en(text: str) -> str:
    normalizer_lang = "zh" if _uses_chinese_normalizer(text) else "en"
    text = _ENGLISH_CONTRACTION_PATTERN.sub(r"\1 is", text)

    tech_terms = _TECH_TERM_PATTERN.findall(text)
    for term in sorted(set(tech_terms), key=len, reverse=True):
        text = text.replace(term, term.replace("-", "<H>"))

    pinyin_values: list[str] = []
    name_values: list[str] = []
    if normalizer_lang == "zh":
        pinyin_values = [match.group(0) for match in _PINYIN_TONE_PATTERN.finditer(text)]
        text, pinyin_values = _protect_values(
            text,
            pinyin_values,
            prefix="pinyin",
        )
        text, name_values = _protect_values(
            text,
            _NAME_PATTERN.findall(text),
            prefix="n",
        )

    result = _load_wetext_normalizer(normalizer_lang).normalize(text)
    result = _restore_values(result, name_values, prefix="n")
    result = _restore_values(
        result,
        pinyin_values,
        prefix="pinyin",
        transform=_correct_pinyin,
    )
    if tech_terms:
        result = re.sub(r"\s*<H>\s*", "-", result)
    if normalizer_lang == "zh":
        result = result.replace("$", ".")
    return clean_indextts25_text(result)


def _normalize_with_official_backend(text: str, lang: str) -> str:
    if lang in {"zh", "en", "zhen"}:
        return _normalize_zh_or_en(text)
    if lang == "es":
        # NeMo is optional in the upstream implementation and falls back to
        # raw text when its grammar package is unavailable.
        try:
            from nemo_text_processing.text_normalization.normalize import (
                Normalizer,
            )

            normalizer = Normalizer(input_case="cased", lang="es")
            return normalizer.normalize(text, verbose=False)
        except ModuleNotFoundError as exc:
            if exc.name != "nemo_text_processing":
                raise
            return text
    return text


@lru_cache(maxsize=1)
def _load_japanese_tagger():
    try:
        import fugashi
        import unidic_lite
    except ModuleNotFoundError as exc:
        if exc.name not in {"fugashi", "unidic_lite"}:
            raise
        raise RuntimeError(
            "IndexTTS 2.5 Japanese preprocessing requires fugashi and unidic-lite. "
            "Install them with pip install 'vllm-omni[indextts2]'"
        ) from exc
    _ = unidic_lite
    return fugashi.Tagger()


def _process_japanese_text(text: str) -> str:
    """Tokenize Japanese like the official g2p_ratio=0 frontend."""

    def process_segment(segment: str) -> str:
        return " ".join(token.surface for token in _load_japanese_tagger()(segment))

    parts = re.split(r"( +)", text)
    return "".join(process_segment(part) if part.strip() else part for part in parts)


def normalize_indextts25_text(
    text: str,
    *,
    lang: str,
    text_normalization: bool = True,
    normalizer: Callable[[str, str], str] | None = None,
) -> str:
    """Apply the deterministic portion of the official preprocessing order.

    A caller-provided normalizer is used for number/date expansion. Keeping it
    injectable lets prompt sizing and Talker prefill use the same normalizer
    instance and avoids constructing heavyweight TN grammars per request.
    """
    lang = normalize_language_code(lang)
    text = clean_indextts25_text(text)
    if text_normalization:
        normalize = normalizer or _normalize_with_official_backend
        text = normalize(text, lang)
    if lang in {"ja", "zh", "en", "zhen"}:
        text = text.lower()
    elif lang == "es":
        text = text.upper()
    text = apply_pronunciation_annotations(text)
    if lang == "ja":
        text = _process_japanese_text(text)
    return _SPECIAL_TOKEN_PATTERN.sub(
        lambda match: f"<|{match.group(1).upper()}|>",
        text,
    )


def prepare_indextts25_text(
    text: str,
    *,
    lang: str,
    model_dir: str,
    tokenizer_file: str = INDEXTTS25_TOKENIZER_FILE,
    text_normalization: bool = True,
    normalizer: Callable[[str, str], str] | None = None,
) -> tuple[list[int], int]:
    if not text or not text.strip():
        raise ValueError("IndexTTS 2.5 text cannot be empty")
    lang_code = normalize_language_code(lang)
    # When no heavyweight TN callback is installed, punctuation/casing and
    # annotations are still deterministic. The Talker installs the official
    # language-specific normalizer before enabling text_normalization.
    normalized = normalize_indextts25_text(
        text,
        lang=lang_code,
        text_normalization=text_normalization,
        normalizer=normalizer,
    )
    # ``zhen`` is not a registered special token. Keep its official literal
    # prefix so tiktoken encodes it through existing mergeable ranks; remapping
    # it to ``common`` would change both the text ABI and preprocessing route.
    prefixed = f"<|{lang_code}|> {normalized}"
    token_ids = encode_indextts25_text(
        prefixed,
        model_dir=model_dir,
        tokenizer_file=tokenizer_file,
    )
    return token_ids, lang_to_token(lang_code)
