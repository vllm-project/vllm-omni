# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Keep paired statistics and validation failure reporting honest."""

import pytest

from benchmarks.tts.benchmark_cosyvoice3_b1 import paired_summary
from benchmarks.tts.validate_cosyvoice3_b1 import require_executed_tests

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_summary_preserves_pairs_under_large_common_drift():
    pairs = [
        {"baseline": {"completed_ms": latency}, "head": {"completed_ms": latency - 0.5}}
        for latency in (1.0, 10.0, 100.0, 1000.0)
    ]
    summary = paired_summary(pairs)
    assert summary["paired_mean_saved_ms"] == 0.5
    assert summary["paired_mean_saved_ci95_ms"] == [0.5, 0.5]
    assert summary["positive_pairs"] == 4
    reversed_pairs = [{"baseline": p["head"], "head": p["baseline"]} for p in pairs]
    assert paired_summary(reversed_pairs)["paired_mean_saved_ci95_ms"] == [-0.5, -0.5]


@pytest.mark.parametrize("body", ["", '<testcase><skipped message="no CUDA"/></testcase>'])
def test_empty_or_skipped_suite_is_not_a_success(tmp_path, body):
    report = tmp_path / "tests.xml"
    report.write_text(f"<testsuites><testsuite>{body}</testsuite></testsuites>", encoding="utf-8")
    with pytest.raises(RuntimeError, match="absent or skipped"):
        require_executed_tests(report)
    report.write_text("<testsuites><testsuite><testcase/></testsuite></testsuites>", encoding="utf-8")
    assert require_executed_tests(report) == 1
