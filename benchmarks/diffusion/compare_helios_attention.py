# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare raw outputs from benchmark_helios_attention.py (requires scikit-image).

Usage: python -m benchmarks.diffusion.compare_helios_attention helios-attention
This reports numerical alignment, not a perceptual-quality pass/fail threshold.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BACKENDS = ("TORCH_SDPA", "FLASH_ATTN", "CUDNN_ATTN")


def compare_metrics(left: np.ndarray, right: np.ndarray) -> dict:
    if left.shape != right.shape or left.ndim != 4 or left.shape[-1] != 3:
        raise ValueError(f"Expected matching FHWC RGB videos: {left.shape}, {right.shape}")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Non-finite video output")
    delta = left.astype(np.float64) - right
    mse = float(np.mean(delta**2))
    return {
        "exact": bool(np.array_equal(left, right)),
        "mae": float(np.mean(np.abs(delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(mse)),
        # Exact equality has infinite PSNR; represent it without invalid JSON.
        "psnr_db": None if mse == 0 else float(-10 * np.log10(mse)),
        "temporal_delta_mae": float(np.mean(np.abs(np.diff(left, axis=0) - np.diff(right, axis=0)))),
    }


def compare(left: np.ndarray, right: np.ndarray) -> dict:
    metrics = compare_metrics(left, right)
    from skimage.metrics import structural_similarity

    metrics["ssim_mean"] = float(
        np.mean([structural_similarity(a, b, data_range=1.0, channel_axis=-1) for a, b in zip(left, right)])
    )
    return metrics


def validate_runs(runs: dict) -> dict:
    records = {}
    for name, data in runs.items():
        rows = [row for row in data["records"] if not row["warmup"]]
        keyed = {(row["num_frames"], row["seed"], row["repeat"]): row for row in rows}
        metadata = data["metadata"]
        expected = {
            (frames, seed, repeat)
            for frames in metadata["frames"]
            for seed in metadata["seeds"]
            for repeat in range(metadata["repeats"])
        }
        if set(keyed) != expected or len(keyed) != len(rows):
            raise ValueError(f"Incomplete or duplicate measurements: {name}")
        records[name] = keyed
    baseline = records["TORCH_SDPA"]
    reference_metadata = runs["TORCH_SDPA"]["metadata"]
    # Startup latency and the selector itself may differ; workload and software may not.
    ignored_fields = {"backend", "engine_startup_ms", "attention_implementations"}
    reference_contract = {key: value for key, value in reference_metadata.items() if key not in ignored_fields}
    for name, keyed in records.items():
        contract = {key: value for key, value in runs[name]["metadata"].items() if key not in ignored_fields}
        if contract != reference_contract or keyed.keys() != baseline.keys():
            raise ValueError(f"Workload or environment differs from TORCH_SDPA: {name}")
        for (frames, seed, repeat), row in keyed.items():
            if row["prompt"] != baseline[(frames, seed, repeat)]["prompt"]:
                raise ValueError(f"Prompt mismatch: {name}, {frames}, {seed}, {repeat}")
            if row["prompt"] != keyed[(frames, seed, 0)]["prompt"]:
                raise ValueError(f"Prompt changed between repetitions: {name}")
    return records


def load_verified_array(root: Path, backend: str, row: dict) -> np.ndarray:
    path = root / backend / row["array"]
    output = np.load(path, allow_pickle=False)
    digest = hashlib.sha256(np.ascontiguousarray(output, dtype=np.float32).tobytes()).hexdigest()
    if digest != row["sha256"]:
        raise ValueError(f"npy hash mismatch: {path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    runs = {name: json.loads((args.root / name / "results.json").read_text()) for name in BACKENDS}
    records = validate_runs(runs)
    baseline = records["TORCH_SDPA"]
    report = {"alignment_vs_torch_sdpa": {}, "self_variance": {}}
    for name, keyed in records.items():
        report["alignment_vs_torch_sdpa"][name] = []
        report["self_variance"][name] = []
        for (frames, seed, repeat), row in keyed.items():
            reference = baseline[(frames, seed, repeat)]
            output = load_verified_array(args.root, name, row)
            if repeat == 0:
                other = load_verified_array(args.root, "TORCH_SDPA", reference)
                group = "alignment_vs_torch_sdpa"
            else:
                first = keyed[(frames, seed, 0)]
                other = load_verified_array(args.root, name, first)
                group = "self_variance"
            report[group][name].append({"num_frames": frames, "seed": seed, "repeat": repeat, **compare(other, output)})
    output_path = args.root / "alignment.json"
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(output_path)


if __name__ == "__main__":
    main()
