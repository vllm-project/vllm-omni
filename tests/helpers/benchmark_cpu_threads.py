# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Benchmark unchanged CPU pytest cases in fresh processes at 1/2/4 threads.

Run inside the CI image with its CPU/memory quota. Arguments after ``--`` are
passed to pytest unchanged. Every warmup and measured run must pass.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from importlib.metadata import version
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory for logs, JUnit and results")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    pytest_args = args.pytest_args[1:] if args.pytest_args[:1] == ["--"] else args.pytest_args
    if args.warmups < 0 or args.repeats < 2 or not pytest_args:
        parser.error("Need nonnegative warmups, at least two measured runs, and pytest arguments")
    args.output.mkdir(parents=True, exist_ok=False)
    results = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {package: version(package) for package in ("torch", "vllm", "vllm-omni", "pytest")},
        "pytest_args": pytest_args,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "runs": [],
    }
    # Set the environment before importing torch in each independent process.
    driver = (
        "import os, sys, torch, pytest; "
        "assert torch.get_num_threads() == int(os.environ['OMP_NUM_THREADS']); "
        "print('torch_threads=', torch.get_num_threads(), flush=True); "
        "sys.exit(pytest.main(sys.argv[1:]))"
    )
    for iteration in range(args.warmups + args.repeats):
        for threads in (1, 2, 4):
            stem = args.output / f"threads-{threads}-run-{iteration}"
            env = dict(os.environ)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[name] = str(threads)
            started = time.perf_counter()
            with stem.with_suffix(".log").open("w", encoding="utf-8") as log:
                run = subprocess.run(
                    [sys.executable, "-c", driver, *pytest_args, f"--junitxml={stem}.xml", "--durations=50"],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=3600,
                )
            elapsed = time.perf_counter() - started
            if run.returncode:
                print(f"Failed run ({run.returncode}); inspect {stem}.log", file=sys.stderr)
                return run.returncode
            cases = ET.parse(stem.with_suffix(".xml")).findall(".//testcase")
            if not cases or any(case.find("skipped") is not None for case in cases):
                print(f"Benchmark must execute every requested case: {stem}.xml", file=sys.stderr)
                return 1
            results["runs"].append(
                {
                    "threads": threads,
                    "iteration": iteration,
                    "warmup": iteration < args.warmups,
                    "wall_seconds": elapsed,
                    "cases": [dict(case.attrib) for case in cases],
                }
            )
            (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
            print(f"threads={threads} iteration={iteration} wall={elapsed:.2f}s", flush=True)
    for threads in (1, 2, 4):
        values = [run["wall_seconds"] for run in results["runs"] if run["threads"] == threads and not run["warmup"]]
        print(f"threads={threads}: mean={statistics.mean(values):.2f}s range={min(values):.2f}..{max(values):.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
