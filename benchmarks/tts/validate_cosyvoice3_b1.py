# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One-command CUDA validation of B1: environment, real tests, paired replay, traces.

Requires a working Linux CUDA vLLM-Omni development environment. Does not install
dependencies, download model weights, change drivers, or launch a TTS server.
"""

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = "tests/model_executor/models/cosyvoice3"


def run(command, log_path, env):
    print("Running:", " ".join(command), flush=True)
    print("Log:", log_path, flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def require_executed_tests(path):
    cases = ET.parse(path).getroot().findall(".//testcase")
    if not cases or any(case.find("skipped") is not None for case in cases):
        raise RuntimeError(f"Tests were absent or skipped; inspect {path}")
    return len(cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("b1-results"))
    parser.add_argument("--quick", action="store_true", help="smaller replay; tests still run in full")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    # A fresh directory prevents stale reports being mistaken for this run.
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "scope": "sampler + runner tests; not Stage-0/E2E", "completed": []}
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        import torch

        from benchmarks.tts.benchmark_cosyvoice3_b1 import environment, load_runtime

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this interpreter; no CPU substitution is allowed")
        report["environment"] = environment(load_runtime(), "cuda")
        (output / "environment.json").write_text(json.dumps(report["environment"], indent=2), encoding="utf-8")
        groups = {
            "benchmark-harness": ["tests/benchmarks/test_cosyvoice3_b1.py"],
            "cpu-and-runner": [
                f"{TEST_DIR}/test_cosyvoice3_model_helpers.py",
                f"{TEST_DIR}/test_cosyvoice3_sampling_policy.py",
                "-k",
                "sample or ras or host_policy",
            ],
            "cuda": [f"{TEST_DIR}/test_cosyvoice3_sampling_cuda.py"],
        }
        for name, tests in groups.items():
            junit = output / f"{name}.xml"
            command = [sys.executable, "-m", "pytest", "-q", "-o", "addopts=", f"--junitxml={junit}", *tests]
            run(command, output / f"{name}.log", env)
            report["completed"].append({"step": name, "executed_tests": require_executed_tests(junit)})
        command = [
            sys.executable,
            "-m",
            "benchmarks.tts.benchmark_cosyvoice3_b1",
            "--device",
            "cuda",
            "--trace",
            "--output",
            str(output / "sampler.json"),
        ]
        if args.quick:
            command.append("--quick")
        run(command, output / "sampler.log", env)
        report["completed"].append({"step": "sampler", "report": str(output / "sampler.json")})
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        (output / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("Validation:", report["status"], "—", output, flush=True)


if __name__ == "__main__":
    main()
