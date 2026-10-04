# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Export a compact summary from backend results.json files and alignment.json.

Provenance is supplied separately: script hashes must describe the measured run,
not whichever scripts happen to be present when exporting historical evidence.
"""

import argparse
import json
from pathlib import Path

from benchmarks.diffusion.compare_helios_attention import BACKENDS, validate_runs

PROVENANCE_KEYS = (
    "date",
    "runtime_commit",
    "benchmark_script_sha256",
    "comparison_script_sha256",
    "model",
    "model_revision",
    "checkpoint_verification",
    "scope",
)
RECORD_KEYS = (
    "num_frames",
    "seed",
    "repeat",
    "wall_ms",
    "transformer_gpu_ms",
    "transformer_forward_count",
    "mean_transformer_forward_ms",
    "worker_peak_reserved_mib",
    "sha256",
)


def build_artifact(runs: dict, alignment: dict, provenance: dict) -> dict:
    records = validate_runs(runs)
    excluded = {"model", "backend", "engine_startup_ms", "attention_implementations"}
    artifact = {key: provenance[key] for key in PROVENANCE_KEYS}
    artifact.update(
        metadata={key: value for key, value in runs["TORCH_SDPA"]["metadata"].items() if key not in excluded},
        prompts_by_seed={str(row["seed"]): row["prompt"] for row in records["TORCH_SDPA"].values()},
        backends={},
        alignment=alignment,
    )
    for name, data in runs.items():
        artifact["backends"][name] = {
            "attention_implementations": data["metadata"]["attention_implementations"],
            "summary": data["summary"],
            "measurements": [{key: row[key] for key in RECORD_KEYS} for row in records[name].values()],
        }
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--provenance", type=Path, required=True, help="Run provenance JSON (or a previous export)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = {name: json.loads((args.root / name / "results.json").read_text()) for name in BACKENDS}
    alignment = json.loads((args.root / "alignment.json").read_text())
    artifact = build_artifact(runs, alignment, json.loads(args.provenance.read_text()))
    args.output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
