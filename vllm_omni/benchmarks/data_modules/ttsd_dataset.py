"""MOSS-TTSD dialogue dataset for ``vllm bench serve``.

Loads ``meta.lst`` rows of the form::

    utt_id|ref1_wav_rel|ref2_wav_rel|dialogue_text

where ``dialogue_text`` contains ``[S1]`` / ``[S2]`` speaker tags. Both ref
wavs live under ``{root}/<locale>/prompt-wavs/``. Builds one
``SampleRequest`` per row with ``ref_audio`` (speaker 1) and ``ref_audio_2``
(speaker 2) inlined as ``data:`` URLs.

Companion to ``seed_tts_dataset.py``; same on-disk layout convention so the
existing ``--dataset-path`` flag works.
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
    _ref_audio_payload,
)

logger = logging.getLogger(__name__)


@dataclass
class _TTSDRow:
    utterance_id: str
    ref1_rel: str
    ref2_rel: str
    target_text: str


def _parse_ttsd_line(line: str) -> _TTSDRow | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split("|")
    if len(parts) < 4:
        logger.warning("Skipping malformed TTSD meta line (need 4 '|'-fields): %r", line[:120])
        return None
    return _TTSDRow(
        utterance_id=parts[0].strip(),
        ref1_rel=parts[1].strip(),
        ref2_rel=parts[2].strip(),
        target_text=parts[3].strip(),
    )


class TTSDDataset(SeedTTSDataset):
    """Two-speaker dialogue prompts for MOSS-TTSD-v1.0.

    Reuses the SeedTTS root resolver so users can point ``--dataset-path`` at
    a local directory or a HF repo with the same layout convention.
    """

    DEFAULT_OUTPUT_LEN = 4096

    def load_data(self) -> None:
        meta = self._root / self.locale / "meta.lst"
        if not meta.is_file():
            raise FileNotFoundError(
                f"TTSD meta not found: {meta} (expected layout: {self._root}/{self.locale}/meta.lst)"
            )
        text = meta.read_text(encoding="utf-8")
        rows: list[_TTSDRow] = []
        for line in text.splitlines():
            r = _parse_ttsd_line(line)
            if r is not None:
                rows.append(r)
        if not rows:
            raise ValueError(f"No valid rows in {meta}")
        if not self.disable_shuffle:
            rng = random.Random(self.random_seed)
            rng.shuffle(rows)
        self._ttsd_rows = rows
        # Parent's self._rows stays empty; sample() is overridden.
        self._rows = []
        self.data = self._ttsd_rows
        logger.info("Loaded TTSD: root=%s locale=%s rows=%d", self._root, self.locale, len(rows))

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
        for i, row in enumerate(self._ttsd_rows):
            if len(out) >= num_requests:
                break
            wav1 = (self._root / self.locale / row.ref1_rel).resolve()
            wav2 = (self._root / self.locale / row.ref2_rel).resolve()
            if not wav1.is_file() or not wav2.is_file():
                logger.warning("Missing TTSD refs for %s: %s / %s", row.utterance_id, wav1, wav2)
                continue
            ref1_uri = _ref_audio_payload(wav1, inline=self.inline_ref_audio)
            ref2_uri = _ref_audio_payload(wav2, inline=self.inline_ref_audio)
            speech_extra: dict[str, Any] = {
                "ref_audio": ref1_uri,
                "ref_audio_2": ref2_uri,
                "max_new_tokens": output_len,
            }
            target = row.target_text
            prompt_len = len(tok.encode(target))
            out.append(
                SeedTTSSampleRequest(
                    prompt=target,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=None,
                    request_id=f"{request_id_prefix}{i}",
                    seed_tts_speech_extra=speech_extra,
                    seed_tts_utterance_id=row.utterance_id,
                    seed_tts_locale=self.locale,
                    seed_tts_system_prompt=self._system_prompt,
                    # SIM computed against ref1 (speaker 1) for a stable signal;
                    # per-speaker SIM is left for offline eval.
                    seed_tts_ref_wav_path=str(wav1),
                )
            )
        logger.info("TTSD: built %d requests (asked %d)", len(out), num_requests)
        self.maybe_oversample_requests(out, num_requests, request_id_prefix, no_oversample)
        return out
