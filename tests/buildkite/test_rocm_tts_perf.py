# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
RUNNER = Path(".buildkite/amd/scripts/run_rocm_tts_perf.py").resolve()


@pytest.mark.parametrize(
    ("filenames", "content", "guard_status", "expected"),
    [
        (["result_model_a_0", "result_model_b_0"], "{}", 0, 0),
        (["result_model_a_0"], "{}", 0, 5),
        (["result_model_a_0", "result_model_a_1"], "{}", 0, 5),
        (["result_model_a_0", "result_other_0"], "{}", 0, 5),
        (["result_model_a_0", "result_model_b_0"], "bad", 0, 5),
        ([], "{}", 1, 1),
        ([], "{}", 124, 124),
    ],
)
def test_tts_perf_result_evidence(tmp_path, filenames, content, guard_status, expected):
    artifact_dir = tmp_path / "artifacts" / "tts-test"
    old = artifact_dir / "benchmark-results" / "old"
    old.mkdir(parents=True)
    (old / "result_model_b_old.json").write_text("{}", encoding="utf-8")
    guard = tmp_path / ".buildkite/amd/scripts/run-rocm-diffusion-nightly.sh"
    guard.parent.mkdir(parents=True)
    guard.write_text(
        "set -eu\n"
        '[[ "$1 $2 $3 $4" == "2 2 165m tts-test" ]]\n'
        '[[ "$5" == "tests/dfx/perf/scripts/run_benchmark.py::test_performance_benchmark[case-a]" ]]\n'
        '[[ "$6" == "tests/dfx/perf/scripts/run_benchmark.py::test_performance_benchmark[case-b]" ]]\n'
        '[[ "$7 $8 $9" == "--test-config-file tests/dfx/perf/tests/test_tts.json --require-baseline" ]]\n'
        '[[ "${10} ${11}" == "-m full_model and tts and rocm and MI325 and cards_2" ]]\n'
        'for filename in $RESULT_FILES; do printf "%s" "$CONTENT" > "$BENCHMARK_DIR/$filename.json"; done\n'
        'exit "$GUARD_STATUS"\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--gpus",
            "2",
            "--artifact-name",
            "tts-test",
            "--expect-results",
            "model_a",
            "1",
            "--expect-results",
            "model_b",
            "1",
            "--cases",
            "case-a",
            "case-b",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "BUILDKITE_BUILD_CHECKOUT_PATH": str(tmp_path),
            "RESULT_FILES": " ".join(filenames),
            "CONTENT": content,
            "GUARD_STATUS": str(guard_status),
        },
        check=False,
    )
    assert result.returncode == expected
    manifest = (artifact_dir / "result-files.txt").read_text(encoding="utf-8").splitlines()
    assert {Path(path).stem for path in manifest} == set(filenames)
