# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Write an explicit local-tokenizer A/B config without editing checked-in YAML."""

import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("full", "step"), required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_num_seqs < 1:
        parser.error("max-num-seqs must be positive")
    if not (args.tokenizer / "tokenizer_config.json").is_file():
        parser.error("tokenizer must be a local directory containing tokenizer_config.json")
    root = Path(__file__).resolve().parents[3]
    source = root / "vllm_omni/deploy/pi05_step.yaml"
    config = yaml.safe_load(source.read_text())
    config["dtype"] = args.dtype
    stage = config["stages"][0]
    stage["step_execution"] = args.backend == "step"
    stage["max_num_seqs"] = args.max_num_seqs if args.backend == "step" else 1
    stage["custom_pipeline_args"] = {"tokenizer": str(args.tokenizer.resolve())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        yaml.safe_dump(config, output, sort_keys=False)


if __name__ == "__main__":
    main()
