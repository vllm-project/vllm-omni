# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU metric/statistical oracles for the fixed P6 evaluation protocol."""

import numpy as np
import pytest

from benchmarks.zonos2.protocol import edit_counts, failures, normalize_asr, quantiles_ci

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    "truth,hypothesis,expected",
    [
        (["a", "b"], ["a", "b"], (0, 0, 0)),
        (["a", "b"], ["a", "c"], (1, 0, 0)),
        (["a", "b"], ["a"], (0, 1, 0)),
        (["a"], ["a", "b", "c"], (0, 0, 2)),
    ],
)
def test_edit_counts(truth, hypothesis, expected):
    row = edit_counts(truth, hypothesis)
    assert (row["substitutions"], row["deletions"], row["insertions"]) == expected
    assert row["rate"] == sum(expected) / len(truth)


def test_empty_reference_rejected_and_wer_not_clamped():
    with pytest.raises(ValueError):
        edit_counts([], [])
    assert edit_counts(["a"], ["a", "b", "c"])["rate"] == 2


def test_asr_units_and_punctuation():
    assert normalize_asr("Hello, WORLD!", "en_us") == ["hello", "world"]
    assert normalize_asr("你 好，世界！", "cmn") == list("你好世界")
    assert normalize_asr("It's a test.", "en_us") == ["it's", "a", "test"]


def test_constant_quantiles_and_ci():
    row = quantiles_ci([3.0] * 8)
    assert row["p50"] == row["p95"] == 3
    assert row["p50_ci95"] == row["p95_ci95"] == [3, 3]


def test_bootstrap_reproducible_and_bounds_ordered():
    a = quantiles_ci(list(range(30)))
    assert a == quantiles_ci(list(range(30)))
    assert a["p50_ci95"][0] <= a["p50"] <= a["p50_ci95"][1]
    assert a["p95_ci95"][0] <= a["p95"] <= a["p95_ci95"][1]


@pytest.mark.parametrize("values", [[], [np.nan], [np.inf]])
def test_invalid_statistics_rejected(values):
    with pytest.raises(ValueError):
        quantiles_ci(values)


def test_failures_preserve_each_bad_metric():
    row = {"language": "cmn", "zh_cer": 0.4, "utmos": 2, "speaker_cosine": 0.2, "duration_s": 20, "reached_cap": True}
    assert set(failures(row)) == {"zh_cer", "utmos", "speaker_cosine", "duration", "token_cap"}
    row.update(zh_cer=0, utmos=4, speaker_cosine=None, duration_s=3, reached_cap=False)
    assert failures(row) == []


def test_kernel_intervals_union_without_double_counting():
    from benchmarks.zonos2.profile_report import union_us

    assert union_us([]) == 0
    assert union_us([(0, 10), (3, 8), (8, 15), (20, 25)]) == 20


def test_quality_noninferiority_gate_rejects_regression():
    from benchmarks.zonos2.gates import quality_gate

    baseline = {"en_wer": 0.02, "zh_cer": 0.0, "utmos_mean": 3.5, "speaker_cosine_mean": 0.95}
    assert quality_gate(baseline, baseline)["pass"]
    worse = {**baseline, "zh_cer": 0.05}
    result = quality_gate(baseline, worse)
    assert not result["pass"] and not result["checks"]["zh_cer"]
