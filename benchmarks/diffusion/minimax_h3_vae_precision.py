# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare complete decoder outputs, including floating-point bit patterns."""

import argparse
import json
from pathlib import Path

import torch


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, object]:
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError("Decoder output shape or dtype changed")
    a, b = reference.reshape(-1), candidate.reshape(-1)
    different_values = different_bytes = 0
    max_abs = error_energy = 0.0
    for begin in range(0, a.numel(), 2_000_000):
        x, y = a[begin : begin + 2_000_000], b[begin : begin + 2_000_000]
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("Decoder output contains a non-finite value")
        different_values += int(torch.count_nonzero(x != y))
        # Floating comparison treats -0 and +0 as equal; compare raw bytes too.
        xb, yb = x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)
        different_bytes += int(torch.count_nonzero(xb != yb))
        delta = y.double() - x.double()
        max_abs = max(max_abs, delta.abs().max().item())
        error_energy += delta.square().sum().item()
    return {
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "elements": reference.numel(),
        "different_values": different_values,
        "different_bytes": different_bytes,
        "max_abs": max_abs,
        "rmse": (error_energy / reference.numel()) ** 0.5,
        "bit_exact": different_bytes == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    results = {}
    for filename in ("video-float.pt", "video-rgb.pt", "video-plain-rgb.pt", "audio-float.pt"):
        reference = torch.load(args.baseline_dir / filename, map_location="cpu", weights_only=True, mmap=True)
        candidate = torch.load(args.candidate_dir / filename, map_location="cpu", weights_only=True, mmap=True)
        results[filename] = compare(reference, candidate)
    text = json.dumps(results, indent=2) + "\n"
    print(text, end="")
    if args.output_json:
        args.output_json.write_text(text)
    if not all(result["bit_exact"] for result in results.values()):
        raise SystemExit("Decoder outputs are not bit-exact")


if __name__ == "__main__":
    main()
