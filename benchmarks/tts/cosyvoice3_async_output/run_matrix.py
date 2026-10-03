# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Run the recorded B2 workload against two fixed source checkouts."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pynvml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--patched", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    manifest = json.loads((scripts / "manifest.json").read_text())
    sources = {"base": args.baseline.resolve(), "head": args.patched.resolve()}
    if sources["base"] == sources["head"]:
        parser.error("baseline and patched must be separate checkouts")
    for variant, source in sources.items():
        hashes = manifest["baseline_files" if variant == "base" else "measurement_files"]
        for name, expected in hashes.items():
            actual = hashlib.sha256((source / name).read_bytes()).hexdigest()
            if actual != expected:
                parser.error(f"{variant}: {name} does not match the recorded source")
    root = args.validation_dir.resolve()
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    labels = [f"b2-v028-{variant}-c1-r{round_id}" for variant in ("base", "head") for round_id in (1, 2)]
    if any((results / label).exists() or (results / f"{label}.log").exists() for label in labels):
        parser.error("result paths already exist; choose a fresh validation directory")

    pynvml.nvmlInit()
    try:
        device = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(device)
        if isinstance(name, bytes):
            name = name.decode()
        record = {"manifest": manifest, "gpu": name, "order": [], "memory": {}}
        for variant, round_id in [("base", 1), ("head", 1), ("head", 2), ("base", 2)]:
            label = f"b2-v028-{variant}-c1-r{round_id}"
            source = sources[variant]
            env = dict(os.environ, PYTHONPATH=str(source), COSYVOICE3_TRT="0")
            env.setdefault("XDG_CACHE_HOME", str(root / "cache"))
            command = [
                sys.executable,
                "-u",
                str(scripts / "benchmark.py"),
                "--validation-dir",
                str(root),
                "--label",
                label,
                "--requests",
                "8",
                "--warmups",
                "2",
                "--concurrency",
                "1",
            ]
            print(f"RUN {label}, source={source}", flush=True)
            memory = [pynvml.nvmlDeviceGetMemoryInfo(device).used]
            start = time.monotonic()
            with (results / f"{label}.log").open("w") as log:
                with subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT) as process:
                    while process.poll() is None:
                        memory.append(pynvml.nvmlDeviceGetMemoryInfo(device).used)
                        time.sleep(0.2)
            record["order"].append(label)
            record["memory"][label] = {
                "idle_device_mib": memory[0] / 2**20,
                "sampled_peak_device_mib": max(memory) / 2**20,
                "samples": len(memory),
                "sampling_interval_s": 0.2,
                "wall_s": time.monotonic() - start,
                "exit_code": process.returncode,
                "command": command,
                "cwd": str(source),
            }
            (results / "matrix.json").write_text(json.dumps(record, indent=2) + "\n")
            print(f"FINISHED {label}: exit={process.returncode}", flush=True)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        for round_id in (1, 2):
            command = [
                sys.executable,
                str(scripts / "compare.py"),
                f"b2-v028-base-c1-r{round_id}",
                f"b2-v028-head-c1-r{round_id}",
                "--results-dir",
                str(results),
                "--warmups",
                "2",
            ]
            subprocess.run(command, check=True)
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
