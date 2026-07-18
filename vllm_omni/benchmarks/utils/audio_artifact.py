# SPDX-License-Identifier: Apache-2.0
"""Audio artifact collector for TTS/Omni benchmark runs.

Collects a curated subset of generated audio clips during a benchmark run and
writes them as compressed audio files plus an `index.json` for offline
review. Designed so a small CI artifact (typically <10 MB per nightly bench)
lets reviewers listen to representative outputs without re-running the bench.

Usage:
    collector = get_collector()  # singleton, no-op when SAVE_AUDIO_SAMPLES unset
    collector.add(utterance_id="zh_001", wav_f32=arr, sample_rate=16000,
                  ref_text="hello world")
    collector.flush()  # called once at end of bench
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_ENV_NUM_SAMPLES = "SAVE_AUDIO_SAMPLES"
_ENV_OUTPUT_DIR = "AUDIO_SAMPLES_DIR"
_DEFAULT_OUTPUT_DIR = "tests/dfx/perf/results/audio_samples"


@dataclass
class _Item:
    utterance_id: str
    wav_f32: np.ndarray
    sample_rate: int
    ref_text: str = ""
    locale: str = ""
    duration_sec: float = 0.0


@dataclass
class AudioArtifactCollector:
    """In-memory buffer for representative audio clips from a bench run.

    The collector keeps every clip the bench adds, and `flush()` picks K
    representative ones (first, last, plus evenly-spaced middle) to write to
    disk. This biases samples toward run-start (warmup) and run-end (steady
    state) so reviewers see both regimes.
    """

    num_samples: int = 0
    output_dir: str = _DEFAULT_OUTPUT_DIR
    label: str = "bench"
    _buffer: list[_Item] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return self.num_samples > 0

    def add(
        self,
        utterance_id: str,
        wav_f32: np.ndarray,
        sample_rate: int,
        ref_text: str = "",
        locale: str = "",
    ) -> None:
        """Buffer a clip. No-op when collector is disabled."""
        if not self.enabled:
            return
        if wav_f32 is None or len(wav_f32) == 0:
            return
        duration = float(len(wav_f32)) / float(max(sample_rate, 1))
        self._buffer.append(
            _Item(
                utterance_id=str(utterance_id),
                wav_f32=np.asarray(wav_f32, dtype=np.float32),
                sample_rate=int(sample_rate),
                ref_text=ref_text,
                locale=locale,
                duration_sec=duration,
            )
        )

    def _pick_indices(self, n_buffered: int) -> list[int]:
        """Pick up to `num_samples` representative indices.

        Always include first and last; fill remaining slots evenly spaced.
        """
        k = min(self.num_samples, n_buffered)
        if k <= 0:
            return []
        if k == 1:
            return [0]
        if k >= n_buffered:
            return list(range(n_buffered))
        # Linear spacing across the run, inclusive of both ends.
        step = (n_buffered - 1) / (k - 1)
        idxs = sorted({int(round(i * step)) for i in range(k)})
        # Edge case: dedup may shrink set; pad with random middle indices.
        while len(idxs) < k and len(idxs) < n_buffered:
            for candidate in range(n_buffered):
                if candidate not in idxs:
                    idxs.append(candidate)
                    break
            idxs = sorted(idxs)
        return idxs[:k]

    def flush(self) -> str | None:
        """Write picked clips and index.json to `output_dir/label/`.

        Returns the output directory path when files were written, else None.
        Best-effort: never raises out of the bench loop.
        """
        if not self.enabled or not self._buffer:
            return None
        try:
            import soundfile as sf
        except ImportError:
            logger.warning(
                "soundfile not available; skipping audio artifact flush (install with `pip install soundfile`)."
            )
            return None

        out_root = Path(self.output_dir) / self.label
        try:
            out_root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning("Cannot create audio artifact dir %s: %s", out_root, e)
            return None

        idxs = self._pick_indices(len(self._buffer))
        manifest: list[dict] = []
        written = 0
        for slot, buf_idx in enumerate(idxs):
            item = self._buffer[buf_idx]
            # FLAC: lossless, ~50% of raw WAV, plays in most native players.
            # Plain wav is fallback if soundfile lacks FLAC support.
            safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in item.utterance_id)
            stem = f"{slot:02d}_{safe_id}"
            fname = f"{stem}.flac"
            target = out_root / fname
            try:
                sf.write(str(target), item.wav_f32, item.sample_rate, format="FLAC")
            except (RuntimeError, ValueError):
                # FLAC unavailable (rare; libsndfile config), fall back to WAV.
                fname = f"{stem}.wav"
                target = out_root / fname
                try:
                    sf.write(str(target), item.wav_f32, item.sample_rate, subtype="PCM_16")
                except Exception:
                    logger.exception("Failed to write audio sample %s", target)
                    continue
            written += 1
            manifest.append(
                {
                    "slot": slot,
                    "buffer_index": buf_idx,
                    "total_buffered": len(self._buffer),
                    "utterance_id": item.utterance_id,
                    "locale": item.locale,
                    "ref_text": item.ref_text,
                    "sample_rate": item.sample_rate,
                    "duration_sec": round(item.duration_sec, 3),
                    "file": fname,
                }
            )

        if written == 0:
            logger.warning("Audio artifact flush wrote 0 files to %s", out_root)
            return None

        index_path = out_root / "index.json"
        try:
            with open(index_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {"label": self.label, "items": manifest},
                    fh,
                    ensure_ascii=False,
                    indent=2,
                )
        except OSError:
            logger.exception("Failed to write audio index at %s", index_path)

        logger.info(
            "Audio artifact: wrote %d/%d clips to %s",
            written,
            len(self._buffer),
            out_root,
        )
        return str(out_root)


# Process-global singleton. Bench code reads env vars on first access; resets
# happen at the bench boundary via `reset_collector` so multi-bench runs in
# one process get separate output directories.
_collector: AudioArtifactCollector | None = None


def _label_from_env() -> str:
    label = os.environ.get("AUDIO_SAMPLES_LABEL", "").strip()
    if label:
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
    return "bench"


def get_collector() -> AudioArtifactCollector:
    """Return the process-wide collector, constructing on first call.

    When `SAVE_AUDIO_SAMPLES` is unset or non-positive, returns a disabled
    collector whose `add()`/`flush()` are no-ops, so call sites can stay
    unconditional.
    """
    global _collector
    if _collector is not None:
        return _collector
    try:
        n = int(os.environ.get(_ENV_NUM_SAMPLES, "0"))
    except ValueError:
        n = 0
    n = max(0, n)
    output_dir = os.environ.get(_ENV_OUTPUT_DIR, _DEFAULT_OUTPUT_DIR)
    _collector = AudioArtifactCollector(
        num_samples=n,
        output_dir=output_dir,
        label=_label_from_env(),
    )
    return _collector


def reset_collector() -> None:
    """Drop the cached collector. Used between bench invocations in one process."""
    global _collector
    _collector = None
