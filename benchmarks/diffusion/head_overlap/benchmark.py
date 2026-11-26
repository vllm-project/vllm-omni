# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Configured, interleaved Omni.generate comparisons; hashing is outside timing."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def arrays(outputs):
    result = []
    for output in outputs:
        for value in output.images or []:
            if isinstance(value, list):
                result.extend(np.asarray(frame) for frame in value)
            else:
                result.append(np.asarray(value))
    if not result or any(not frame.size for frame in result):
        raise AssertionError("Empty image/video output")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--case", required=True, help="Case ID in the workload JSON")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config["cases"] = [case for case in config["cases"] if case["id"] == args.case]
    if len(config["cases"]) != 1 or args.rounds < 1:
        parser.error("Choose one case from the workload and a positive number of rounds")
    if os.environ.get("VLLM_OMNI_PACKED_QKV") != "1":
        parser.error("Set VLLM_OMNI_PACKED_QKV=1 for the native baseline")
    case = config["cases"][0]
    args.out.mkdir(parents=True, exist_ok=True)
    prompt = {**config["prompt"], **case.get("prompt", {})}
    if "media" in case:
        media = {}
        for name, paths in case["media"].items():
            if isinstance(paths, str):
                media[name] = Image.open(paths).convert("RGB")
            else:
                media[name] = [Image.open(path).convert("RGB") for path in paths]
        prompt["multi_modal_data"] = media
    records, expected = [], {}
    for mode in (0, "auto"):
        # Use the public engine configuration for both arms. Neither engine
        # imports a benchmark extension or receives per-request routing flags.
        omni = Omni(model=args.model, **config["engine"], ulysses_overlap_chunks=mode)
        try:
            runs = [("first", 42), *[("steady", 42)] * args.rounds, ("correctness", 43), ("correctness", 44)]
            for phase, seed in runs:
                params = OmniDiffusionSamplingParams(**case["sampling"], seed=seed)
                started = time.perf_counter()
                outputs = omni.generate(prompt, sampling_params_list=[params])
                elapsed_ms = (time.perf_counter() - started) * 1000
                values = arrays(outputs)
                signature = [(list(v.shape), str(v.dtype), hashlib.sha256(v.tobytes()).hexdigest()) for v in values]
                if not all(np.isfinite(v).all() for v in values) or signature != expected.setdefault(seed, signature):
                    raise AssertionError(f"Output mismatch for {case['id']} / {mode} / seed {seed}")
                row = dict(case=case["id"], mode=mode, phase=phase, seed=seed, e2e_ms=elapsed_ms, outputs=signature)
                records.append(row)
                print("HEAD_OVERLAP_RESULT " + json.dumps(row), flush=True)
                temporary = args.out / "results.tmp"
                temporary.write_text(json.dumps(records, indent=2))
                temporary.replace(args.out / "results.json")
        finally:
            omni.close()


if __name__ == "__main__":
    main()
