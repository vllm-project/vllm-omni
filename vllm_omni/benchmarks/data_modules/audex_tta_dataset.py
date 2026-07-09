# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audex text-to-audio caption dataset for ``vllm bench serve``.

Drives the ``nemotron_labs_audex_tta`` deployment through
``/v1/audio/speech`` (``--backend openai-audio-speech``) with caption-only
rows; the online ``audex_tta`` adapter builds the TTA prompt and applies
the default guidance (cfg_scale 3.0) server-side.

Rows come from ``captions.lst`` under ``<root>/<locale>/``::

    utt_id|caption text

RTF/throughput metrics only: the output is general audio (not speech), so
the seed-tts WER/SIM evaluation does not apply — ``seed_tts_ref_wav_path``
stays empty and the eval pipeline skips those metrics.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any

from vllm.benchmarks.datasets import SampleRequest
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.hf import get_cached_tokenizer

from vllm_omni.benchmarks.data_modules.seed_tts_dataset import (
    SeedTTSDataset,
    SeedTTSSampleRequest,
)

logger = logging.getLogger(__name__)


@dataclass
class _CaptionRow:
    utterance_id: str
    caption: str


def _parse_caption_line(line: str) -> _CaptionRow | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split("|", 1)
    if len(parts) < 2:
        logger.warning("Skipping malformed caption line (need 'utt_id|caption'): %r", line[:120])
        return None
    caption = parts[1].strip()
    if not caption:
        return None
    return _CaptionRow(utterance_id=parts[0].strip(), caption=caption)


class AudexTTADataset(SeedTTSDataset):
    """Caption rows for Audex text-to-audio benchmarking (RTF/throughput only)."""

    # The TTA decode cap is 500 frames (~10 s); 4 codec tokens per frame
    # plus the end marker.
    DEFAULT_OUTPUT_LEN = 4200

    def load_data(self) -> None:
        meta = self._root / self.locale / "captions.lst"
        if not meta.is_file():
            raise FileNotFoundError(
                f"Audex TTA captions not found: {meta} (expected {self._root}/{self.locale}/captions.lst)"
            )
        rows: list[_CaptionRow] = []
        for line in meta.read_text(encoding="utf-8").splitlines():
            row = _parse_caption_line(line)
            if row is not None:
                rows.append(row)
        if not rows:
            raise ValueError(f"No valid caption rows in {meta}")
        if not self.disable_shuffle:
            rng = random.Random(self.random_seed)
            rng.shuffle(rows)
        self._caption_rows = rows
        self._rows = []
        self.data = self._caption_rows
        logger.info("Loaded Audex TTA captions: root=%s locale=%s rows=%d", self._root, self.locale, len(rows))

    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        output_len: int | None = None,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs: Any,
    ) -> list[SampleRequest]:
        if output_len is None:
            output_len = self.DEFAULT_OUTPUT_LEN
        tok = get_cached_tokenizer(tokenizer)
        out: list[SampleRequest] = []
        for i, row in enumerate(self._caption_rows):
            if len(out) >= num_requests:
                break
            out.append(
                SeedTTSSampleRequest(
                    prompt=row.caption,
                    prompt_len=max(1, len(tok.encode(row.caption))),
                    expected_output_len=output_len,
                    multi_modal_data=None,
                    request_id=f"{request_id_prefix}{i}",
                    seed_tts_speech_extra={"max_new_tokens": output_len},
                    seed_tts_utterance_id=row.utterance_id,
                    seed_tts_locale=self.locale,
                    seed_tts_system_prompt=self._system_prompt,
                    seed_tts_ref_wav_path="",  # non-speech output: WER/SIM skipped
                )
            )
        logger.info("Audex TTA: built %d requests (asked %d)", len(out), num_requests)
        self.maybe_oversample_requests(out, num_requests, request_id_prefix, no_oversample)
        return out
