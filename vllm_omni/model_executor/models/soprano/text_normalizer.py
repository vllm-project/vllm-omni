# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
Normalize input text to a format that Soprano recognizes.
Adapted from https://github.com/neonbjb/tortoise-tts/blob/main/tortoise/utils/tokenizer.py
"""

from functools import lru_cache

import regex as re


@lru_cache(maxsize=1)
def _number_engine():
    try:
        import inflect
    except ImportError as exc:
        raise ImportError("Soprano text normalization requires vllm-omni[soprano]") from exc
    return inflect.engine()


####################################################################################################
# Abbreviations

_abbreviations = [
    (re.compile(rf"\b{x[0]}\.", re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("ms", "miss"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]
_cased_abbreviations = [
    (re.compile(rf"\b{x[0]}\b"), x[1])
    for x in [
        ("Hz", "hertz"),
        ("kHz", "kilohertz"),
        ("KBs", "kilobytes"),
        ("KB", "kilobyte"),
        ("MBs", "megabytes"),
        ("MB", "megabyte"),
        ("GBs", "gigabytes"),
        ("GB", "gigabyte"),
        ("TBs", "terabytes"),
        ("TB", "terabyte"),
        ("APIs", "a p i's"),
        ("API", "a p i"),
        ("CLIs", "c l i's"),
        ("CLI", "c l i"),
        ("CPUs", "c p u's"),
        ("CPU", "c p u"),
        ("GPUs", "g p u's"),
        ("GPU", "g p u"),
        ("Ave", "avenue"),
        ("etc", "et cetera"),
        ("Mon", "monday"),
        ("Tues", "tuesday"),
        ("Wed", "wednesday"),
        ("Thurs", "thursday"),
        ("Fri", "friday"),
        ("Sat", "saturday"),
        ("Jan", "january"),
        ("Feb", "february"),
        ("Mar", "march"),
        ("Apr", "april"),
        ("Aug", "august"),
        ("Sept", "september"),
        ("Oct", "october"),
        ("Nov", "november"),
        ("Dec", "december"),
        ("and/or", "and or"),
    ]
]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations + _cased_abbreviations:
        text = re.sub(regex, replacement, text)
    return text


####################################################################################################
# Numbers

_num_prefix_re = re.compile(r"#\d")
_num_suffix_re = re.compile(r"\b\d+(K|M|B|T)\b", re.IGNORECASE)
_num_letter_split_re = re.compile(r"(\d[a-z]|[a-z]\d)", re.IGNORECASE)

_comma_number_re = re.compile(r"(\d[\d\,]+\d)")
_date_re = re.compile(r"(^|[^/])(\d\d?[/-]\d\d?[/-]\d\d(?:\d\d)?)($|[^/])")
_phone_number_re = re.compile(r"(\(?\d{3}\)?[-.\s]\d{3}[-.\s]?\d{4})")
_time_re = re.compile(r"(\d\d?:\d\d(?::\d\d)?)")
_pounds_re = re.compile(r"£([\d\,]*\d+)")
_dollars_re = re.compile(r"\$([\d\.\,]*\d+)")
_decimal_number_re = re.compile(r"(\d+(?:\.\d+)+)")
_multiply_re = re.compile(r"(\d\s?\*\s?\d)")
_divide_re = re.compile(r"(\d\s?/\s?\d)")
_add_re = re.compile(r"(\d\s?\+\s?\d)")
_subtract_re = re.compile(r"(\d?\s?-\s?\d)")  # also does negative numbers
_fraction_re = re.compile(r"(\d+(?:/\d+)+)")
_ordinal_re = re.compile(r"\d+(st|nd|rd|th)")
_number_re = re.compile(r"\d+")


def _expand_num_prefix(m):
    match = m.group(0)
    return f"number {match[1]}"


def _expand_num_suffix(m):
    match = m.group(0)
    if match[-1].upper() == "K":
        return f"{match[:-1]} thousand"
    elif match[-1].upper() == "M":
        return f"{match[:-1]} million"
    elif match[-1].upper() == "B":
        return f"{match[:-1]} billion"
    elif match[-1].upper() == "T":
        return f"{match[:-1]} trillion"
    return match  # unexpected format


def _split_alphanumeric(m):
    match = m.group(1)
    return f"{match[0]} {match[1]}"


def _remove_commas(m):
    return m.group(1).replace(",", "")


def _expand_date(m):
    match = m.group(2)
    match = re.split("[./-]", match)
    return m.group(1) + " dash ".join(match) + m.group(3)


def _expand_phone_number(m):
    match = m.group(1)
    match = re.sub(r"\D", "", match)
    assert len(match) == 10
    match = f"{' '.join(list(match[:3]))}, {' '.join(list(match[3:6]))}, {' '.join(list(match[6:]))}"
    return match


def _expand_time(m):
    match = m.group(1)
    match = match.split(":")
    if len(match) == 2:
        hours, minutes = match
        if minutes == "00":
            if int(hours) == 0:
                return "0"
            elif int(hours) > 12:
                return f"{hours} minutes"
            return f"{hours} o'clock"
        elif minutes.startswith("0"):
            minutes = f"oh {minutes[1:]}"
        return f"{hours} {minutes}"
    else:
        hours, minutes, seconds = match
        if int(hours) != 0:
            # Preserve upstream's formatting (including its set representation
            # for a nonzero, non-leading-zero minute in this branch).
            spoken_minutes = "oh oh" if minutes == "00" else f"oh {minutes}" if minutes.startswith("0") else {minutes}
            spoken_seconds = "" if seconds == "00" else f"oh {seconds}" if seconds.startswith("0") else seconds
            return f"{hours} {spoken_minutes} {spoken_seconds}"
        elif minutes != "00":
            return (
                f"{minutes} {'oh oh' if seconds == '00' else f'oh {seconds}' if seconds.startswith('0') else seconds}"
            )
        else:
            return seconds


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return f"{dollars} {dollar_unit}, {cents} {cent_unit}"
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return f"{dollars} {dollar_unit}"
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return f"{cents} {cent_unit}"
    else:
        return "zero dollars"


def _expand_decimal_point(m):
    match = m.group(1)
    match = match.split(".")
    return match[0] + " point " + " point ".join(" ".join(list(match[i])) for i in range(1, len(match)))


def _expand_fraction(m):
    match = m.group(1)
    match = match.split("/")
    return " over ".join(match) if len(match) == 2 else " slash ".join(match)


def _expand_multiply(m):
    return " times ".join(m.group(1).split("*"))


def _expand_divide(m):
    return " over ".join(m.group(1).split("/"))


def _expand_add(m):
    return " plus ".join(m.group(1).split("+"))


def _expand_subtract(m):
    return " minus ".join(m.group(1).split("-"))


def _expand_ordinal(m):
    return _number_engine().number_to_words(m.group(0), andword="")


def _expand_number(m):
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + _number_engine().number_to_words(num % 100)
        elif num % 100 == 0:
            return _number_engine().number_to_words(num // 100) + " hundred"
        else:
            return _number_engine().number_to_words(num, andword="", zero="oh", group=2).replace(", ", " ")
    else:
        return _number_engine().number_to_words(num, andword="")


def normalize_numbers(text):
    text = re.sub(_num_prefix_re, _expand_num_prefix, text)
    text = re.sub(_num_suffix_re, _expand_num_suffix, text)
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_date_re, _expand_date, text)
    text = re.sub(_phone_number_re, _expand_phone_number, text)
    text = re.sub(_time_re, _expand_time, text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_multiply_re, _expand_multiply, text)
    text = re.sub(_divide_re, _expand_divide, text)
    text = re.sub(_add_re, _expand_add, text)
    text = re.sub(_subtract_re, _expand_subtract, text)

    text = re.sub(_fraction_re, _expand_fraction, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    for _ in range(2):  # need to do this twice to find all matches
        text = re.sub(_num_letter_split_re, _split_alphanumeric, text)
    text = re.sub(_number_re, _expand_number, text)
    return text


####################################################################################################
# Special characters & other patterns

_preunicode_special_characters = [
    (re.compile(x[0]), x[1])
    for x in [
        ("—", " - "),
    ]
]
_special_characters = [
    (re.compile(x[0]), x[1])
    for x in [
        ("@", " at "),
        ("&", " and "),
        ("%", " percent "),
        (":", "."),
        (";", ","),
        (r"\+", " plus "),
        (r"\\", " backslash "),
        ("~", " about "),
        ("(^| )<3", " heart "),
        ("<=", " less than or equal to "),
        (">=", " greater than or equal to "),
        ("<", " less than "),
        (">", " greater than "),
        ("=", " equals "),
        ("/", " slash "),
        ("_", " "),
        (r"\*", " "),
    ]
]
_link_header_re = re.compile(r"(https?://)")
_dash_re = re.compile(r"(. - .)")
_dot_re = re.compile(r"([A-Z]\.[A-Z])", re.IGNORECASE)
_parentheses_re = re.compile(r"[\(\[\{].*[\)\]\}](.|$)")
_camelcase_re = re.compile(r"\b([A-Z][a-z]*)+\b")


def expand_preunicode_special_characters(text):
    for regex, replacement in _preunicode_special_characters:
        text = re.sub(regex, replacement, text)
    return text


def expand_special_characters(text):
    for regex, replacement in _special_characters:
        text = re.sub(regex, replacement, text)
    return text


def _expand_link_header(m):
    return "h t t p s colon slash slash "


def _expand_dash(m):
    match = m.group(0)
    return f"{match[0]}, {match[4]}"


def _expand_dot(m):
    match = m.group(0)
    return f"{match[0]} dot {match[2]}"


def _expand_parentheses(m):
    match = m.group(0)
    match = re.sub(r"[\(\[\{]", ", ", match)
    match = re.sub(r"[\)\]\}][^$.!?,]", ", ", match)
    match = re.sub(r"[\)\]\}]", "", match)
    return match


def _split_mixedcase(m):
    match = m.group(0)
    matches = re.findall("[A-Z][a-z]*", match)
    if len(matches) == 1:
        return match  # Single capital word
    if len(matches) == len(match):
        return match  # All uppercase
    if len(matches) == len(match) - 1 and match[-1] == "s":
        return f"{match[:-1]}'s"  # plural uppercase word
    return " ".join(matches)


def normalize_special(text):
    text = re.sub(_link_header_re, _expand_link_header, text)
    text = re.sub(_dash_re, _expand_dash, text)
    text = re.sub(_dot_re, _expand_dot, text)
    text = re.sub(_parentheses_re, _expand_parentheses, text)
    return text


def normalize_mixedcase(text):
    text = re.sub(_camelcase_re, _split_mixedcase, text)
    return text


####################################################################################################
# Misc


def lowercase(text):
    return text.lower()


def convert_to_ascii(text):
    try:
        from unidecode import unidecode
    except ImportError as exc:
        raise ImportError("Soprano text normalization requires vllm-omni[soprano]") from exc
    return unidecode(text)


def normalize_newlines(text):
    text = text.split("\n")
    for i in range(len(text)):
        text[i] = text[i].strip()
        if not text[i]:
            continue
        if text[i][-1] not in ".!?":
            text[i] = f"{text[i]}."
    return " ".join(text)


def remove_unknown_characters(text):
    text = re.sub(r"[^A-Za-z !\$%&'\*\+,-./0123456789<>\?_]", "", text)
    text = re.sub(r"[<>/_+]", "", text)
    return text


def collapse_whitespace(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r" [.\?!,]", lambda m: m.group(0)[1], text)
    return text.strip()


def dedup_punctuation(text):
    text = re.sub(r"\.\.\.+", "[ELLIPSIS]", text)
    text = re.sub(r",+", ",", text)
    text = re.sub(r"[\.,]*\.[\.,]*", ".", text)
    text = re.sub(r"[\.,!]*![\.,!]*", "!", text)
    text = re.sub(r"[\.,!\?]*\?[\.,!\?]*", "?", text)
    text = re.sub(r"\[ELLIPSIS\]", "...", text)
    return text


def collapse_triple_letters(text):
    text = re.sub(r"(\w)\1{2,}", lambda m: m.group(0)[:2], text)
    return text


def clean_text(text):
    text = expand_preunicode_special_characters(text)
    text = convert_to_ascii(text)
    text = normalize_newlines(text)
    text = normalize_numbers(text)
    text = normalize_special(text)
    text = expand_abbreviations(text)
    text = normalize_mixedcase(text)
    text = expand_special_characters(text)
    text = lowercase(text)
    text = remove_unknown_characters(text)
    text = collapse_whitespace(text)
    text = dedup_punctuation(text)
    text = collapse_triple_letters(text)
    return text
