# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Pure lexical normalization; callers bound query and card text lengths."""

from unicodedata import normalize

_STOP_WORDS = frozenset(
    "a an the is are was were i my me you your user self what which where when do does did in of to and please".split()
)


def lexemes(text: str) -> frozenset[str]:
    words = "".join(char if char.isalnum() else " " for char in normalize("NFKC", text).casefold()).split()
    terms = set(words).difference(_STOP_WORDS)
    for word in words:
        terms.update(
            word[index : index + 2]
            for index in range(len(word) - 1)
            if all("\u3400" <= char <= "\u9fff" for char in word[index : index + 2])
        )
    return frozenset(terms)
