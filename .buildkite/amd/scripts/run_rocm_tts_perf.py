# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Run selected AMD TTS perf cases and validate their top-level sweep artifacts."""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, deque
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--expect-results", nargs=2, action="append", metavar=("MODEL", "COUNT"), required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9-]+", args.artifact_name):
        parser.error("invalid artifact name")
    expected = {model: int(count) for model, count in args.expect_results}

    artifact_dir = Path(os.environ["BUILDKITE_BUILD_CHECKOUT_PATH"]) / "artifacts" / args.artifact_name
    result_root = artifact_dir / "benchmark-results"
    result_root.mkdir(parents=True, exist_ok=True)
    # A retry or a reused checkout must only count this invocation's measurements.
    benchmark_dir = Path(tempfile.mkdtemp(prefix="run-", dir=result_root))
    nodes = [f"tests/dfx/perf/scripts/run_benchmark.py::test_performance_benchmark[{case}]" for case in args.cases]
    run = subprocess.run(
        [
            "bash",
            ".buildkite/amd/scripts/run-rocm-diffusion-nightly.sh",
            str(args.gpus),
            str(len(nodes)),
            "165m",
            args.artifact_name,
            *nodes,
            "--test-config-file",
            "tests/dfx/perf/tests/test_tts.json",
            "--require-baseline",
            "-m",
            f"full_model and tts and rocm and MI325 and cards_{args.gpus}",
        ],
        env={**os.environ, "BENCHMARK_DIR": str(benchmark_dir)},
        check=False,
    )
    log = artifact_dir / "pytest.log"
    if log.is_file():
        with log.open(encoding="utf-8", errors="replace") as stream:
            (artifact_dir / "pytest-summary.txt").write_text("".join(deque(stream, maxlen=100)), encoding="utf-8")
    results = sorted(benchmark_dir.glob("result_*.json"))
    (artifact_dir / "result-files.txt").write_text("".join(f"{path}\n" for path in results), encoding="utf-8")
    if run.returncode:
        return run.returncode

    counts: Counter[str] = Counter()
    for path in results:
        models = [name for name in expected if path.name.startswith(f"result_{name}_")]
        if len(models) != 1:
            print(f"Unexpected TTS result filename: {path.name}", file=sys.stderr)
            return 5
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"Unreadable TTS result JSON {path}: {exc}", file=sys.stderr)
            return 5
        counts[models[0]] += 1
    if counts != expected:
        print(f"Expected TTS result counts {expected}, found {dict(counts)}", file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
