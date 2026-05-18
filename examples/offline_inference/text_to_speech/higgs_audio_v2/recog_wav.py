# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ASR-based sanity check for synthesized WAV files.

Loads an English offline transducer (sherpa-onnx-nemo-fast-conformer) and
transcribes the input WAV files. Compares the transcript against an expected
prompt (when supplied) using a simple normalized word-level WER.

Usage:

    python examples/offline_inference/text_to_speech/higgs_audio_v2/recog_wav.py \\
        --wav /tmp/higgs_stage1_isolation/hello_world.wav \\
              /tmp/higgs_stage1_isolation/the_quick_brown_fox_jumps_over_the_lazy_dog.wav

    # With expected text (one per --wav, in the same order):
    python recog_wav.py --wav a.wav b.wav --expected "Hello world." "Mary had a little lamb"

    # Glob over a directory:
    python recog_wav.py --glob "/tmp/higgs_stage1_isolation/*.wav"
"""

from __future__ import annotations

import argparse
import glob as _glob
import os
import re
import sys
import wave
from pathlib import Path

import numpy as np


def _resolve_model_dir(user_path: str | None) -> str:
    """Return a sherpa-onnx-nemo-fast-conformer-transducer-en-24500 dir.

    Tries: ``user_path`` (if set) → ``HF_HUB_CACHE`` snapshot →
    ``~/.cache/huggingface``.
    """
    if user_path and os.path.isdir(user_path):
        return user_path
    try:
        from huggingface_hub import snapshot_download

        # local_files_only=True: don't redownload; just resolve the cached path.
        return snapshot_download(
            repo_id="csukuangfj/sherpa-onnx-nemo-fast-conformer-transducer-en-24500",
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not resolve the ASR model directory. Either pass --model-dir "
            "explicitly or download via `huggingface_hub.snapshot_download("
            "'csukuangfj/sherpa-onnx-nemo-fast-conformer-transducer-en-24500')`."
        ) from exc


def _load_wav_mono_f32(path: str) -> tuple[np.ndarray, int]:
    """Return (mono_pcm_float32 in [-1, 1], sample_rate)."""
    with wave.open(path, "rb") as w:
        nch = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
        nfr = w.getnframes()
        raw = w.readframes(nfr)
    if sw != 2:
        raise ValueError(f"{path}: only int16 WAV supported (got sample_width={sw})")
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        pcm = pcm.reshape(-1, nch).mean(axis=1)
    return pcm, sr


def _normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return " ".join(s.split())


def _word_error_rate(ref: str, hyp: str) -> float:
    """Standard Levenshtein-based WER on whitespace-tokenized strings."""
    ref_t = _normalize_text(ref).split()
    hyp_t = _normalize_text(hyp).split()
    if not ref_t:
        return 0.0 if not hyp_t else 1.0
    # DP edit distance
    n, m = len(ref_t), len(hyp_t)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_t[i - 1] == hyp_t[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m] / n


def _build_recognizer(model_dir: str, num_threads: int = 2):
    import sherpa_onnx

    encoder = os.path.join(model_dir, "encoder.onnx")
    decoder = os.path.join(model_dir, "decoder.onnx")
    joiner = os.path.join(model_dir, "joiner.onnx")
    tokens = os.path.join(model_dir, "tokens.txt")
    for f in (encoder, decoder, joiner, tokens):
        if not os.path.isfile(f):
            raise FileNotFoundError(f"missing model file: {f}")
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        tokens=tokens,
        num_threads=num_threads,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        model_type="nemo_transducer",
    )


def _transcribe(recog, wav_path: str) -> str:
    pcm, sr = _load_wav_mono_f32(wav_path)
    if sr != 16000:
        # Resample to 16kHz via simple polyphase (sherpa-onnx wants 16k input).
        try:
            from scipy.signal import resample_poly

            from math import gcd

            g = gcd(sr, 16000)
            pcm = resample_poly(pcm, 16000 // g, sr // g).astype(np.float32)
        except ImportError:
            # Fall back to linear interpolation if scipy isn't available.
            x_old = np.arange(len(pcm))
            new_len = int(round(len(pcm) * 16000 / sr))
            x_new = np.linspace(0, len(pcm) - 1, new_len)
            pcm = np.interp(x_new, x_old, pcm).astype(np.float32)
        sr = 16000
    stream = recog.create_stream()
    stream.accept_waveform(sr, pcm)
    recog.decode_stream(stream)
    return stream.result.text.strip()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wav", nargs="*", default=[], help="One or more WAV paths.")
    p.add_argument("--glob", default=None, help="Glob pattern to expand into WAV paths.")
    p.add_argument("--expected", nargs="*", default=[], help="Expected text per WAV, in order.")
    p.add_argument("--model-dir", default=None, help="sherpa-onnx model directory.")
    p.add_argument("--num-threads", type=int, default=2)
    args = p.parse_args()

    wavs = list(args.wav)
    if args.glob:
        wavs.extend(sorted(_glob.glob(args.glob)))
    if not wavs:
        print("no wav files provided (use --wav or --glob)", file=sys.stderr)
        return 2

    expected_map: dict[str, str] = {}
    for path, expected in zip(args.wav, args.expected, strict=False):
        expected_map[os.path.abspath(path)] = expected

    recog = _build_recognizer(_resolve_model_dir(args.model_dir), num_threads=args.num_threads)

    overall_wers: list[float] = []
    for wav in wavs:
        try:
            hyp = _transcribe(recog, wav)
        except Exception as exc:
            print(f"[FAIL] {wav}: {exc}", file=sys.stderr)
            continue
        ref = expected_map.get(os.path.abspath(wav))
        if ref is None:
            ref = _infer_expected_from_basename(wav)
        wer = _word_error_rate(ref, hyp) if ref else None
        label = Path(wav).name
        wer_str = f" WER={wer * 100:.1f}%" if wer is not None else ""
        ref_str = f" | expected={ref!r}" if ref else ""
        print(f"[asr] {label}: hyp={hyp!r}{ref_str}{wer_str}")
        if wer is not None:
            overall_wers.append(wer)

    if overall_wers:
        avg = sum(overall_wers) / len(overall_wers)
        print(f"\naverage WER over {len(overall_wers)} files: {avg * 100:.1f}%")
    return 0


def _infer_expected_from_basename(wav_path: str) -> str | None:
    """Recover the original prompt from a slugified WAV filename.

    Stage-1 isolation WAVs follow the slug convention from
    ``batch_speech_client._slug`` / the reference fixtures: lowercase, words
    joined by underscores, trailing punctuation stripped. We invert the slug
    by replacing underscores with spaces (lossy on capitalisation/punctuation
    but good enough for WER scoring).
    """
    stem = Path(wav_path).stem
    if not stem:
        return None
    return stem.replace("_", " ")


if __name__ == "__main__":
    sys.exit(main())
