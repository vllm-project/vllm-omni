# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Lazy NeMo r1.2.0 forward TN matching the frozen ZONOS2 frontend."""

from __future__ import annotations

import os
import threading
from typing import Any

import regex as re

SERVER_TO_NEMO_LANG = {
    "en_us": "en",
    "en_gb": "en",
    "fr_fr": "fr",
    "de": "de",
    "es": "es",
    "it": "it",
    "pt_br": "pt",
    "ja": "ja",
    "cmn": "zh",
    "ko": "ko",
}
_DIGIT_PUNCT = re.compile(r"(\d)([.!?,;:])(?=\s|$)")
_SPACE_PUNCT = re.compile(r" +([.!?,;:])(?=\s|$)")


class Zonos2TextNormalizer:
    def __init__(self, cache_dir: str | None = None):
        self.cache_dir: str = (
            cache_dir
            or os.environ.get(
                "VLLM_ZONOS2_TN_CACHE_DIR",
                os.path.join(
                    os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "vllm-omni", "zonos2-tn"
                ),
            )
            or os.path.join(os.path.expanduser("~/.cache"), "vllm-omni", "zonos2-tn")
        )
        self._normalizers: dict[str, Any] = {}
        self._locks: dict[str, Any] = {}
        self._lock = threading.Lock()

    def _language_lock(self, lang: str):
        with self._lock:
            return self._locks.setdefault(lang, threading.Lock())

    def _build(self, lang: str):
        try:
            from importlib.metadata import version

            from nemo_text_processing.text_normalization.normalize import Normalizer
        except ImportError as exc:
            raise ImportError(
                "ZONOS2 text normalization requires nemo_text_processing==1.2.0, "
                "pynini==2.1.6.post1 and sacremoses (install NeMo's dependencies)."
            ) from exc
        if version("nemo_text_processing") != "1.2.0":
            raise RuntimeError("ZONOS2 frozen frontend requires nemo_text_processing==1.2.0")
        case = "lower_cased" if lang == "ko" else "cased"
        cache = os.path.join(self.cache_dir, f"{lang}_{case}")
        os.makedirs(cache, exist_ok=True)
        normalizer = Normalizer(input_case=case, lang=lang, cache_dir=cache, overwrite_cache=False)
        # NeMo's zh/ja verbalizers read FARs without writing them.
        if lang in ("zh", "ja"):
            prefix = "jp" if lang == "ja" else "zh"
            far = os.path.join(cache, f"{prefix}_tn_True_deterministic_verbalizer.far")
            if not os.path.exists(far):
                from nemo_text_processing.text_normalization.en.graph_utils import generator_main

                generator_main(far, {"verbalize": normalizer.verbalizer.fst})
        return normalizer

    def normalize(self, text: str, language: str = "en_us") -> str:
        language = language.strip().lower().replace("-", "_")
        if language not in SERVER_TO_NEMO_LANG:
            raise ValueError(f"Unsupported ZONOS2 language: {language!r}")
        if not text.strip():
            raise ValueError("Input text cannot be empty")
        lang = SERVER_TO_NEMO_LANG[language]
        with self._language_lock(lang):
            if lang not in self._normalizers:
                self._normalizers[lang] = self._build(lang)
            prepared = _DIGIT_PUNCT.sub(r"\1 \2", text)
            result = self._normalizers[lang].normalize(prepared, punct_post_process=lang in ("en", "zh", "ja", "ko"))
        if not isinstance(result, str) or not result.strip():
            raise ValueError(f"ZONOS2 text normalization returned empty text for {language}")
        return _SPACE_PUNCT.sub(r"\1", result)
