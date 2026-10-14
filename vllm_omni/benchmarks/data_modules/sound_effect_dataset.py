"""MOSS-SoundEffect dataset for ``vllm bench serve``.

Loads ``meta.lst`` rows of the form::

    utt_id|ambient_sound|duration_seconds

No reference audio. Builds one SampleRequest per row sending
``ambient_sound`` + ``duration_seconds`` to ``/v1/audio/speech``. Speaker
similarity (SIM) is not applicable; the ``seed_tts_ref_wav_path`` field is
left empty so the seed-tts-eval pipeline skips it.
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
class _SoundEffectRow:
    utterance_id: str
    ambient_sound: str
    duration_seconds: float


def _parse_se_line(line: str) -> _SoundEffectRow | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split("|")
    if len(parts) < 3:
        logger.warning("Skipping malformed sound-effect meta line (need 3 '|'-fields): %r", line[:120])
        return None
    try:
        duration = float(parts[2].strip())
    except ValueError:
        logger.warning("Bad duration_seconds %r in: %s", parts[2], line[:120])
        return None
    text = parts[1].strip()
    if not text:
        return None
    return _SoundEffectRow(utterance_id=parts[0].strip(), ambient_sound=text, duration_seconds=duration)


class SoundEffectDataset(SeedTTSDataset):
    """Ambient-sound prompts for MOSS-SoundEffect.

    Reuses the SeedTTS root resolver. The ``input`` text in each request is
    set to an empty string — SoundEffect ignores it; ``ambient_sound`` is the
    actual conditioning signal.
    """

    DEFAULT_OUTPUT_LEN = 512  # ~40 s of audio at 12.5 fps

    def load_data(self) -> None:
        meta = self._root / self.locale / "meta.lst"
        if not meta.is_file():
            raise FileNotFoundError(
                f"Sound-effect meta not found: {meta} (expected {self._root}/{self.locale}/meta.lst)"
            )
        text = meta.read_text(encoding="utf-8")
        rows: list[_SoundEffectRow] = []
        for line in text.splitlines():
            r = _parse_se_line(line)
            if r is not None:
                rows.append(r)
        if not rows:
            raise ValueError(f"No valid rows in {meta}")
        if not self.disable_shuffle:
            rng = random.Random(self.random_seed)
            rng.shuffle(rows)
        self._se_rows = rows
        self._rows = []
        self.data = self._se_rows
        logger.info("Loaded SoundEffect: root=%s locale=%s rows=%d", self._root, self.locale, len(rows))

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
        for i, row in enumerate(self._se_rows):
            if len(out) >= num_requests:
                break
            speech_extra: dict[str, Any] = {
                "ambient_sound": row.ambient_sound,
                "duration_seconds": row.duration_seconds,
                "max_new_tokens": output_len,
            }
            prompt_len = max(1, len(tok.encode(row.ambient_sound)))
            out.append(
                SeedTTSSampleRequest(
                    prompt="",  # SoundEffect ignores text
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=None,
                    request_id=f"{request_id_prefix}{i}",
                    seed_tts_speech_extra=speech_extra,
                    seed_tts_utterance_id=row.utterance_id,
                    seed_tts_locale=self.locale,
                    seed_tts_system_prompt=self._system_prompt,
                    seed_tts_ref_wav_path="",  # SIM not applicable
                )
            )
        logger.info("SoundEffect: built %d requests (asked %d)", len(out), num_requests)
        self.maybe_oversample_requests(out, num_requests, request_id_prefix, no_oversample)
        return out
