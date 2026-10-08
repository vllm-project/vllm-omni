# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Frozen ZONOS2 P6 inputs, operating points and statistical definitions."""

from __future__ import annotations

import unicodedata

import numpy as np

CASES = (
    ("en_01", "Hello, this is the first Zonos2 baseline sample.", "en_us", False),
    ("en_02", "The quick brown fox jumps over the lazy dog.", "en_us", False),
    ("en_03", "Artificial intelligence is transforming the world of computing.", "en_us", False),
    ("zh_01", "你好，这是第一条中文基线样本。", "cmn", False),
    ("zh_02", "今天天气不错，我们一起去公园散步吧。", "cmn", False),
    ("zh_03", "人工智能技术正在改变我们的生活方式。", "cmn", False),
    ("num_01", "The meeting is on March 3rd, 2026, at 3:30 PM.", "en_us", False),
    ("num_02", "Order number 45821 costs 199 dollars and 99 cents.", "en_us", False),
    ("vc_01", "This sentence should sound like the reference speaker.", "en_us", True),
    ("vc_02", "Voice cloning transfers the timbre of a speaker.", "en_us", True),
)
PARAMS = {
    "temperature": 1.15,
    "top_k": 106,
    "top_p": 1.0,
    "min_p": 0.18,
    "repetition_penalty": 1.2,
    "repetition_window": 50,
    "repetition_codebooks": 8,
    "max_tokens": 1024,
    "seed": 42,
    "emotion_cfg_scale": 1.0,
}
THRESHOLDS = {
    "en_wer": 0.15,
    "zh_cer": 0.15,
    "speaker_cosine": 0.5,
    "utmos": 3.0,
    "duration_min_s": 0.25,
    "duration_max_s": 15.0,
}


def quantiles_ci(values, *, samples: int = 2000, seed: int = 42) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Statistics require nonempty finite values")
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(samples, len(values)))]
    result = {"n": len(values), "mean": float(values.mean())}
    for q in (50, 95):
        distribution = np.percentile(draws, q, axis=1)
        result[f"p{q}"] = float(np.percentile(values, q))
        result[f"p{q}_ci95"] = np.percentile(distribution, [2.5, 97.5]).tolist()
    return result


def normalize_asr(text: str, language: str):
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(c for c in text if not unicodedata.category(c).startswith("P") or c == "'")
    text = " ".join(text.split())
    return list(text.replace(" ", "")) if language == "cmn" else text.split()


def edit_counts(reference, hypothesis) -> dict:
    """Unit-cost edit counts with deterministic tie-breaking, no metric clamping."""
    table = [[(0, 0, 0, 0)] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
    for i in range(1, len(reference) + 1):
        table[i][0] = (i, 0, i, 0)
    for j in range(1, len(hypothesis) + 1):
        table[0][j] = (j, 0, 0, j)
    for i, truth in enumerate(reference, 1):
        for j, predicted in enumerate(hypothesis, 1):
            cost, s, d, ins = table[i - 1][j - 1]
            diagonal = (cost + int(truth != predicted), s + int(truth != predicted), d, ins)
            cost, s, d, ins = table[i - 1][j]
            delete = (cost + 1, s, d + 1, ins)
            cost, s, d, ins = table[i][j - 1]
            insert = (cost + 1, s, d, ins + 1)
            table[i][j] = min((diagonal, delete, insert), key=lambda row: row[0])
    errors, substitutions, deletions, insertions = table[-1][-1]
    if not reference:
        raise ValueError("Empty ASR reference")
    return {
        "reference_units": len(reference),
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "errors": errors,
        "rate": errors / len(reference),
    }


def failures(row: dict) -> list[str]:
    failed = []
    rate_key = "zh_cer" if row["language"] == "cmn" else "en_wer"
    if row[rate_key] > THRESHOLDS[rate_key]:
        failed.append(rate_key)
    if row["utmos"] < THRESHOLDS["utmos"]:
        failed.append("utmos")
    cosine = row.get("speaker_cosine")
    if cosine is not None and cosine < THRESHOLDS["speaker_cosine"]:
        failed.append("speaker_cosine")
    if not THRESHOLDS["duration_min_s"] <= row["duration_s"] <= THRESHOLDS["duration_max_s"]:
        failed.append("duration")
    if row.get("reached_cap"):
        failed.append("token_cap")
    return failed
