# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".buildkite" / "common" / "scripts"))

from skip_ci import (  # noqa: E402
    CiDecision,
    _diff_only_contains_skip_mark_changes,
    resolve_ci_decision,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _decision(changed_files: list[str]) -> CiDecision:
    return resolve_ci_decision(changed_files)


def _is_yaml_gated(decision: CiDecision) -> bool:
    return decision.skip_l2_l3 and not decision.skip_all


def _all_l23_skipped(decision: CiDecision) -> bool:
    checks = (
        ("cuda", "l2"),
        ("cuda", "l3"),
        ("amd", "l2"),
        ("amd", "l3"),
        ("intel", "l2"),
        ("npu", "l2"),
    )
    return all(not decision.is_run(platform, level) for platform, level in checks)


def test_docs_only_change() -> None:
    assert _decision(["docs/foo.md", "README.md", "mkdocs.yml"]).skip_all


def test_docs_only_rejects_product_code() -> None:
    assert not _decision(["docs/foo.md", "vllm_omni/foo.py"]).skip_all


def test_skip_mark_diff_only() -> None:
    diff = """\
diff --git a/tests/foo/test_bar.py b/tests/foo/test_bar.py
--- a/tests/foo/test_bar.py
+++ b/tests/foo/test_bar.py
@@ -1,3 +1,4 @@
 import pytest
+
+@pytest.mark.skip(reason="temp")
 def test_x():
"""
    assert _diff_only_contains_skip_mark_changes(diff)


def test_skip_mark_diff_rejects_assertion_change() -> None:
    diff = """\
diff --git a/tests/foo/test_bar.py b/tests/foo/test_bar.py
--- a/tests/foo/test_bar.py
+++ b/tests/foo/test_bar.py
@@ -1,3 +1,3 @@
 def test_x():
-    assert 1 == 1
+    assert 2 == 2
"""
    assert not _diff_only_contains_skip_mark_changes(diff)


def test_cuda_ready_only_runs_cuda_l2() -> None:
    decision = _decision([".buildkite/cuda/test-ready.yml"])
    assert _is_yaml_gated(decision)
    assert decision.is_run("cuda", "l2")
    assert not decision.is_run("cuda", "l3")
    assert _all_l23_skipped(decision) is False


def test_cuda_merge_only_runs_cuda_l3() -> None:
    decision = _decision([".buildkite/cuda/test-merge.yml"])
    assert _is_yaml_gated(decision)
    assert not decision.is_run("cuda", "l2")
    assert decision.is_run("cuda", "l3")
    assert not decision.is_run("amd", "l3")


def test_l4_only_skips_all_l23() -> None:
    decision = _decision([".buildkite/cuda/test-nightly.yml"])
    assert _is_yaml_gated(decision)
    assert _all_l23_skipped(decision)
    assert not decision.is_run("cuda", "l2")
    assert not decision.is_run("cuda", "l3")


def test_weekly_only_skips_all_l23() -> None:
    decision = _decision([".buildkite/cuda/test-weekly.yml"])
    assert _is_yaml_gated(decision)
    assert _all_l23_skipped(decision)


def test_npu_nightly_only_skips_all_l23() -> None:
    decision = _decision([".buildkite/npu/test-npu-nightly.yml"])
    assert _is_yaml_gated(decision)
    assert _all_l23_skipped(decision)


def test_nightly_and_weekly_skips_all_l23() -> None:
    decision = _decision(
        [
            ".buildkite/cuda/test-nightly.yml",
            ".buildkite/cuda/test-weekly.yml",
        ],
    )
    assert _is_yaml_gated(decision)
    assert _all_l23_skipped(decision)


def test_l2_and_l4_skips_l3() -> None:
    files = [
        ".buildkite/cuda/test-ready.yml",
        ".buildkite/cuda/test-nightly.yml",
    ]
    decision = _decision(files)
    assert _is_yaml_gated(decision)
    assert decision.is_run("cuda", "l2")
    assert not decision.is_run("cuda", "l3")


def test_l3_and_l4_skips_l2() -> None:
    files = [
        ".buildkite/cuda/test-merge.yml",
        ".buildkite/cuda/test-nightly.yml",
    ]
    decision = _decision(files)
    assert _is_yaml_gated(decision)
    assert not decision.is_run("cuda", "l2")
    assert decision.is_run("cuda", "l3")


def test_ready_and_merge_runs_cuda_l2_l3() -> None:
    decision = _decision(
        [
            ".buildkite/cuda/test-ready.yml",
            ".buildkite/cuda/test-merge.yml",
        ],
    )
    assert _is_yaml_gated(decision)
    assert decision.is_run("cuda", "l2")
    assert decision.is_run("cuda", "l3")


def test_amd_ready_only_runs_amd_l2() -> None:
    decision = _decision([".buildkite/amd/test-amd-ready.yml"])
    assert _is_yaml_gated(decision)
    assert decision.is_run("amd", "l2")
    assert not decision.is_run("cuda", "l2")


def test_intel_pipeline_runs_intel_l2() -> None:
    decision = _decision([".buildkite/intel/pipeline-intel.yml"])
    assert _is_yaml_gated(decision)
    assert decision.is_run("intel", "l2")
    assert not decision.is_run("cuda", "l2")


def test_non_whitelist_buildkite_runs_normal_ci() -> None:
    assert not _is_yaml_gated(_decision([".buildkite/common/scripts/skip_ci.py"]))
    assert not _is_yaml_gated(_decision([".buildkite/cuda/pipeline.yml"]))


def test_whitelist_plus_non_whitelist_buildkite_runs_normal_ci() -> None:
    assert not _is_yaml_gated(
        _decision(
            [
                ".buildkite/cuda/test-ready.yml",
                ".buildkite/cuda/pipeline.yml",
            ],
        ),
    )


def test_mixed_product_code_disables_targeting() -> None:
    assert not _is_yaml_gated(
        _decision([".buildkite/cuda/test-ready.yml", "vllm_omni/x.py"]),
    )


def test_non_whitelist_path_disables_targeting() -> None:
    assert not _is_yaml_gated(_decision([".buildkite/unknown/foo.yml"]))
    assert not _is_yaml_gated(_decision(["vllm_omni/x.py"]))


def test_is_run_l23_when_not_yaml_gated() -> None:
    decision = _decision(["vllm_omni/x.py"])
    assert decision.is_run("cuda", "l2")
    assert decision.is_run("amd", "l3")


def test_is_run_l23_respects_decision() -> None:
    decision = _decision([".buildkite/cuda/test-ready.yml"])
    assert _is_yaml_gated(decision)
    assert decision.is_run("cuda", "l2")
    assert not decision.is_run("cuda", "l3")
    assert not decision.is_run("amd", "l2")


def test_docs_or_skip_mark_only_without_diff_range_helper() -> None:
    assert _decision(["docs/a.md"]).skip_all
    assert not _decision(["vllm_omni/a.py"]).skip_all


def test_decision_serializes_to_dict() -> None:
    decision = _decision([".buildkite/cuda/test-ready.yml"])
    assert _is_yaml_gated(decision)
    payload = asdict(decision)
    assert payload["skip_l2_l3"] is True
    assert payload["device"]["skip_cuda"] is False


def test_yaml_gated_message_lists_changed_and_targets() -> None:
    decision = _decision([".buildkite/cuda/test-ready.yml"])
    assert decision.message.startswith("only CI config YAML changed")
    assert "changed: L2=[.buildkite/cuda/test-ready.yml]" in decision.message
    assert "run: cuda/l2" in decision.message
    assert "skip: cuda/l3" in decision.message
    assert "amd/l2" in decision.message
