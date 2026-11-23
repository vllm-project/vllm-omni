# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Author: wanren
# Date: 2025/10/16
import logging
from pathlib import Path

import regex as re

from .talker_pre_processor import TalkerPreProcessor
from .talker_re import TalkerRE


class TalkerTN:
    def __init__(self, resource_dir: str | Path):
        resource_dir = Path(resource_dir)
        required = [f"{lang}_tn_{kind}.fst" for lang in ("zh", "en") for kind in ("tagger", "verbalizer")]
        required.append("talker_re.json")
        missing = [name for name in required if not (resource_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing Ming text normalization resources in {resource_dir}: {', '.join(missing)}. "
                "Download the complete talker/talker_tn directory from the selected model revision."
            )
        try:
            from .normalizer import Normalizer
        except ModuleNotFoundError as exc:
            if exc.name != "pynini":
                raise
            raise ModuleNotFoundError(
                "Ming upstream text normalization requires pip install 'vllm-omni[ming-tn]'"
            ) from exc

        self.talker_pre_processor = TalkerPreProcessor()
        self.talker_re = TalkerRE()
        self.talker_re.update(resource_dir / "talker_re.json")
        self.re_contains_chinese = re.compile(r"[一-鿿]")
        self.re_digit_hyphen = re.compile(r"(?<=\d)-(?=\d)")
        self.tn_zh = Normalizer(str(resource_dir / "zh_tn"), ordertype="tn")
        self.tn_en = Normalizer(str(resource_dir / "en_tn"), ordertype="en_tn")

    def __call__(self, text):
        logging.debug(f"[TalkerTN] input: {text!r}")
        text = self.talker_pre_processor(text)
        logging.debug(f"[TalkerTN] after pre_processor: {text!r}")
        if len(text) == 0:
            return text

        is_chinese = bool(self.re_contains_chinese.search(text))
        logging.debug(f"[TalkerTN] is_chinese={is_chinese}")
        if is_chinese:
            bak_text = text
            try:
                text = self.talker_re(text)
                logging.debug(f"[TalkerTN] after talker_re: {text!r}")
            except Exception as e:
                logging.debug(f"[TalkerTN] talker_re raised {e!r}, fallback to bak_text")
                text = bak_text
            text = self.re_digit_hyphen.sub("，", text)
            out = self.tn_zh.normalize(text)
            logging.debug(f"[TalkerTN] after tn_zh.normalize: {out!r}")
            return out
        out = self.tn_en.normalize(text)
        logging.debug(f"[TalkerTN] after tn_en.normalize: {out!r}")
        return out

    def normalize(self, text):
        return self(text)
