# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Kokoro text frontend: grapheme-to-phoneme conversion, chunking and voices.

Ported from ``KPipeline`` in the reference ``kokoro`` package (v0.9.4) so that
text reaches the network as the same phoneme chunks. The one deliberate
difference: where ``KPipeline`` truncates a non-English chunk past the phoneme
budget, dropping its tail, this frontend splits the chunk instead. G2P comes
from ``misaki`` (the optional ``vllm-omni[kokoro]`` extra); Japanese and
Mandarin need its ``ja`` / ``zh`` extras on top.
"""

from __future__ import annotations

import functools
import importlib
import os
from collections.abc import Iterator
from typing import Any

import regex as re
from vllm.logger import init_logger

logger = init_logger(__name__)

# Phonemes per network pass, excluding the two boundary pad tokens.
MAX_PHONEMES = 510

DEFAULT_VOICE = "af_heart"

# Lang code -> language. A voice's first letter is its lang code (af_heart -> a).
LANG_CODES: dict[str, str] = {
    "a": "American English",
    "b": "British English",
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "p": "pt-br",
    "j": "Japanese",
    "z": "Mandarin Chinese",
}

# Accepted ``language`` request values -> lang code.
LANGUAGE_ALIASES: dict[str, str] = {
    "en-us": "a",
    "en-gb": "b",
    "es": "e",
    "fr-fr": "f",
    "hi": "h",
    "it": "i",
    "pt-br": "p",
    "ja": "j",
    "zh": "z",
}

_INSTALL_HINT = "Kokoro needs the misaki G2P package: pip install 'vllm-omni[kokoro]'"

# G2P the ``kokoro`` extra does not install: lang code -> (module, package).
_EXTRA_G2P: dict[str, tuple[str, str]] = {
    "j": ("misaki.ja", "misaki[ja]"),
    "z": ("misaki.zh", "misaki[zh]"),
}


@functools.cache
def missing_g2p_package(lang_code: str) -> str | None:
    """Return the package ``lang_code``'s G2P still needs, or None if it imports.

    The base ``kokoro`` extra is checked at load time (``load_weights`` warms
    up the English G2P), so only the optional Japanese / Mandarin extras can
    be missing on a running server.
    """
    extra = _EXTRA_G2P.get(lang_code)
    if extra is None:
        return None
    module, package = extra
    try:
        importlib.import_module(module)
    except ImportError:
        return package
    return None


def resolve_lang_code(voice: str, language: str | None = None) -> str:
    """Return the G2P lang code for a request; ``language`` overrides the voice's."""
    if language:
        key = language.lower()
        code = LANGUAGE_ALIASES.get(key, key)
        if code not in LANG_CODES:
            raise ValueError(f"Unsupported language {language!r}. Supported: {', '.join(sorted(LANGUAGE_ALIASES))}")
        return code
    code = voice[:1].lower()
    if code not in LANG_CODES:
        raise ValueError(f"Cannot infer a language from voice {voice!r}")
    return code


def resolve_voices_dir(model: str, revision: str | None = None) -> str:
    """Return a local directory holding the checkpoint's ``voices/*.pt`` packs."""
    if os.path.isdir(model):
        return os.path.join(model, "voices")
    from vllm_omni.transformers_utils.repo_utils import hf_api

    snapshot = hf_api().snapshot_download(model, revision=revision, allow_patterns=["voices/*.pt"])
    return os.path.join(snapshot, "voices")


def list_voices(voices_dir: str) -> list[str]:
    if not os.path.isdir(voices_dir):
        return []
    return sorted(name[: -len(".pt")] for name in os.listdir(voices_dir) if name.endswith(".pt"))


def _tokens_to_ps(tokens: list[Any]) -> str:
    return "".join(t.phonemes + (" " if t.whitespace else "") for t in tokens).strip()


def _waterfall_last(
    tokens: list[Any],
    next_count: int,
    waterfall: tuple[str, ...] = ("!.?…", ":;", ",—"),
    bumps: tuple[str, ...] = (")", "”"),
) -> int:
    """Index after the last strong-enough break that keeps the chunk in budget."""
    for w in waterfall:
        z = next((i for i, t in reversed(list(enumerate(tokens))) if t.phonemes in set(w)), None)
        if z is None:
            continue
        z += 1
        if z < len(tokens) and tokens[z].phonemes in bumps:
            z += 1
        if next_count - len(_tokens_to_ps(tokens[:z])) <= MAX_PHONEMES:
            return z
    return len(tokens)


def _en_chunks(tokens: list[Any]) -> Iterator[str]:
    tks: list[Any] = []
    pcount = 0
    for t in tokens:
        t.phonemes = "" if t.phonemes is None else t.phonemes
        next_ps = t.phonemes + (" " if t.whitespace else "")
        next_pcount = pcount + len(next_ps.rstrip())
        if next_pcount > MAX_PHONEMES:
            z = _waterfall_last(tks, next_pcount)
            yield _tokens_to_ps(tks[:z])
            tks = tks[z:]
            pcount = len(_tokens_to_ps(tks))
            if not tks:
                next_ps = next_ps.lstrip()
        tks.append(t)
        pcount += len(next_ps)
    if tks:
        yield _tokens_to_ps(tks).strip()


def _sentence_chunks(text: str, chunk_size: int = 400) -> list[str]:
    """Group sentences into ~``chunk_size``-character chunks (non-English path)."""
    sentences = re.split(r"([.!?]+)", text)
    chunks: list[str] = []
    current = ""
    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        if i + 1 < len(sentences):
            sentence += sentences[i + 1]
        if len(current) + len(sentence) <= chunk_size:
            current += sentence
        else:
            if current:
                chunks.append(current.strip())
            current = sentence
    if current:
        chunks.append(current.strip())
    if not chunks:
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks


# Break points for re-splitting an overflowing non-English chunk, strongest
# first: sentence ends (ASCII and full-width), clause marks, then whitespace.
_BREAKS = (r"[.!?。！？…]+", r"[,;:，、；：]+", r"\s+")


def _split_in_two(text: str) -> tuple[str, str]:
    """Split ``text`` at its strongest break, taking the one nearest the middle."""
    middle = len(text) // 2
    for pattern in _BREAKS:
        cuts = [m.end() for m in re.finditer(pattern, text) if 0 < m.end() < len(text)]
        if cuts:
            cut = min(cuts, key=lambda c: abs(c - middle))
            return text[:cut], text[cut:]
    return text[:middle], text[middle:]


def _phonemize_within_budget(g2p: Any, text: str) -> list[str]:
    """Phonemize a non-English chunk, re-splitting it while it overflows a pass.

    A chunk that fits is phonemized whole, as ``KPipeline`` does; one that
    does not is split and each part phonemized, instead of truncated.
    """
    ps, _ = g2p(text)
    if not ps or len(ps) <= MAX_PHONEMES or len(text) < 2:
        return [ps] if ps else []
    head, tail = _split_in_two(text)
    return [*_phonemize_within_budget(g2p, head.strip()), *_phonemize_within_budget(g2p, tail.strip())]


class KokoroTextFrontend:
    """Per-language G2P instances, built lazily and reused across requests."""

    def __init__(self) -> None:
        self._g2p: dict[str, Any] = {}

    def _get_g2p(self, lang_code: str) -> Any:
        g2p = self._g2p.get(lang_code)
        if g2p is not None:
            return g2p
        try:
            from misaki import espeak
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        if lang_code in "ab":
            from misaki import en

            british = lang_code == "b"
            try:
                fallback = espeak.EspeakFallback(british=british)
            except Exception as exc:
                logger.warning("Kokoro espeak fallback unavailable, out-of-dictionary words will be skipped: %s", exc)
                fallback = None
            g2p = en.G2P(trf=False, british=british, fallback=fallback, unk="")
        elif lang_code == "j":
            try:
                from misaki import ja
            except ImportError as exc:
                raise ImportError("Kokoro Japanese voices need: pip install 'misaki[ja]'") from exc
            g2p = ja.JAG2P()
        elif lang_code == "z":
            try:
                from misaki import zh
            except ImportError as exc:
                raise ImportError("Kokoro Mandarin voices need: pip install 'misaki[zh]'") from exc
            g2p = zh.ZHG2P(version=None)
        else:
            g2p = espeak.EspeakG2P(language=LANG_CODES[lang_code])
        self._g2p[lang_code] = g2p
        return g2p

    def warmup(self, lang_code: str = "a") -> None:
        self.phoneme_chunks("Hello.", lang_code)

    def phoneme_chunks(self, text: str, lang_code: str) -> list[str]:
        """Split ``text`` into phoneme strings of at most ``MAX_PHONEMES`` each.

        Paragraphs (newline runs) never share a chunk, matching the reference.
        """
        g2p = self._get_g2p(lang_code)
        chunks: list[str] = []
        for paragraph in re.split(r"\n+", text.strip()):
            if not paragraph.strip():
                continue
            if lang_code in "ab":
                _, tokens = g2p(paragraph)
                pieces = list(_en_chunks(tokens))
            else:
                pieces = []
                for chunk in _sentence_chunks(paragraph):
                    if chunk.strip():
                        pieces.extend(_phonemize_within_budget(g2p, chunk))
            for ps in pieces:
                if not ps:
                    continue
                if len(ps) > MAX_PHONEMES:
                    logger.warning("Kokoro truncating a %d-phoneme chunk to %d", len(ps), MAX_PHONEMES)
                    ps = ps[:MAX_PHONEMES]
                chunks.append(ps)
        return chunks


def phonemes_to_ids(phonemes: str, vocab: dict[str, int]) -> list[int]:
    """Map a phoneme string to network input ids, with boundary pads (id 0)."""
    return [0, *(vocab[p] for p in phonemes if p in vocab), 0]
