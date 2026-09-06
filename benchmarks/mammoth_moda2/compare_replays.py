# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Compare immutable native-engine replay artifacts without changing outputs."""

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file


def metrics(reference, candidate):
    assert reference.shape == candidate.shape
    # Float64 reductions keep large identical tensors' cosine near one.
    a, b = reference.double(), candidate.double()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    delta = (a - b).abs()
    norm_product = a.square().sum().sqrt() * b.square().sum().sqrt()
    return {
        "shape": list(a.shape),
        "exact": torch.equal(reference, candidate),
        "max_abs": delta.max().item(),
        "mean_abs": delta.mean().item(),
        "rms_relative": (delta.square().mean() / a.square().mean().clamp_min(1e-30)).sqrt().item(),
        "cosine": (a.mul(b).sum() / norm_product).item() if norm_product > 0 else None,
    }


def image_metrics(reference, candidate):
    a = (reference.float() / 2 + 0.5).clamp(0, 1)
    b = (candidate.float() / 2 + 0.5).clamp(0, 1)
    mse = (a - b).square().mean().item()
    axis = torch.arange(11, dtype=torch.float32) - 5
    gaussian = torch.exp(-(axis.square()) / (2 * 1.5**2))
    gaussian /= gaussian.sum()
    kernel = (gaussian[:, None] * gaussian[None, :]).expand(3, 1, 11, 11)

    def average(value):
        return torch.nn.functional.conv2d(value, kernel, groups=3)

    mean_a, mean_b = average(a), average(b)
    var_a = average(a.square()) - mean_a.square()
    var_b = average(b.square()) - mean_b.square()
    covariance = average(a * b) - mean_a * mean_b
    ssim = ((2 * mean_a * mean_b + 0.01**2) * (2 * covariance + 0.03**2)) / (
        (mean_a.square() + mean_b.square() + 0.01**2) * (var_a + var_b + 0.03**2)
    )
    return {
        "normalized_rgb_mae": (a - b).abs().mean().item(),
        "psnr_db": -10 * math.log10(mse) if mse else "infinite",
        "ssim_11x11_sigma1_5": ssim.mean().item(),
    }


def compare(root_a, root_b):
    outputs_a = [load_file(str(root_a / f"decoded-{i}.safetensors"))["decoded"] for i in range(4)]
    outputs_b = [load_file(str(root_b / f"decoded-{i}.safetensors"))["decoded"] for i in range(4)]
    records = {
        "reference": str(root_a),
        "candidate": str(root_b),
        "reference_self_replay": [metrics(outputs_a[0], value) for value in outputs_a[1:]],
        "candidate_self_replay": [metrics(outputs_b[0], value) for value in outputs_b[1:]],
        "decoded": metrics(outputs_a[0], outputs_b[0]),
        "image": image_metrics(outputs_a[0], outputs_b[0]),
        "observations": {},
    }
    a_dir = root_a / "warmup-observation/rank-0"
    b_dir = root_b / "warmup-observation/rank-0"
    for source in sorted(a_dir.glob("*.safetensors")):
        left, right = load_file(str(source)), load_file(str(b_dir / source.name))
        assert left.keys() == right.keys()
        records["observations"][source.name] = {name: metrics(left[name], right[name]) for name in left}
    rank_one = root_b / "warmup-observation/rank-1"
    if rank_one.exists():
        records["candidate_cross_rank"] = {}
        for source in sorted(b_dir.glob("*.safetensors")):
            left, right = load_file(str(source)), load_file(str(rank_one / source.name))
            records["candidate_cross_rank"][source.name] = {name: metrics(left[name], right[name]) for name in left}
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = compare(Path(args.reference), Path(args.candidate))
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in ("decoded", "image")}, indent=2))


if __name__ == "__main__":
    main()
