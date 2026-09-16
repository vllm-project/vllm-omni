#!/usr/bin/env python3
"""Reconstruct CP RainFusion masks and compare them with full-Q trace masks."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch


def _load(directory: Path):
    traces = []
    for path in sorted(directory.glob("rainfusion_mask_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["path"] = path
        traces.append(payload)
    return traces


def _key(trace):
    return trace["step"], trace["layer"]


def _reconstruct_cp(traces):
    first = traces[0]
    block_size = first["block_size"]
    global_blocks = (first["kv_valid_len"] + block_size - 1) // block_size
    mask = torch.zeros(
        (*first["mask"].shape[:2], global_blocks, first["mask"].shape[-1]), dtype=torch.int8
    )
    filled = torch.zeros(global_blocks, dtype=torch.bool)
    for trace in traces:
        if trace["block_size"] != block_size or trace["kv_valid_len"] != first["kv_valid_len"]:
            raise ValueError("CP trace group has inconsistent block geometry")
        start = trace["q_global_start"] // block_size
        local_blocks = trace["mask"].shape[-2]
        end = start + local_blocks
        if end > global_blocks or filled[start:end].any():
            raise ValueError(f"invalid or overlapping CP rows in {trace['path']}")
        mask[:, :, start:end, :] = trace["mask"]
        filled[start:end] = True
    if not filled.all():
        missing = torch.where(~filled)[0].tolist()
        raise ValueError(f"CP trace group is missing global Q blocks: {missing}")
    return mask


def _reconstruct_full_q(traces):
    """Restore USP's head-sharded full-Q mask to the original head order."""
    ordered = sorted(traces, key=lambda trace: trace["rank"])
    first = ordered[0]
    expected_ranks = int(first["world_size"])
    if len(ordered) != expected_ranks:
        ranks = [trace["rank"] for trace in ordered]
        raise ValueError(
            f"full-Q trace is missing USP ranks: got {ranks}, expected {expected_ranks} ranks"
        )
    if [trace["rank"] for trace in ordered] != list(range(expected_ranks)):
        raise ValueError("full-Q trace ranks must be contiguous and start at zero")
    for trace in ordered[1:]:
        if (
            trace["block_size"] != first["block_size"]
            or trace["kv_valid_len"] != first["kv_valid_len"]
            or trace["mask"].shape[0] != first["mask"].shape[0]
            or trace["mask"].shape[2:] != first["mask"].shape[2:]
        ):
            raise ValueError("full-Q trace group has inconsistent block geometry")
    return torch.cat([trace["mask"] for trace in ordered], dim=1)


def _save_mask_plot(reference, reconstructed, key, output_dir, batch, head):
    """Draw the full-Q, reconstructed CP, and XOR masks for one batch/head."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "--plot-dir requires matplotlib; install it in the trace environment"
        ) from exc
    if not 0 <= batch < reference.shape[0]:
        raise ValueError(f"batch={batch} is outside [0, {reference.shape[0]})")
    if not 0 <= head < reference.shape[1]:
        raise ValueError(f"head={head} is outside [0, {reference.shape[1]})")
    output_dir.mkdir(parents=True, exist_ok=True)
    full_q = reference[batch, head].numpy()
    cp = reconstructed[batch, head].numpy()
    diff = full_q != cp
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for axis, image, title, cmap in (
        (axes[0], full_q, "USP/full-Q mask", "Greys"),
        (axes[1], cp, "CP reconstructed mask", "Greys"),
        (axes[2], diff, "XOR difference", "Reds"),
    ):
        axis.imshow(image, interpolation="nearest", aspect="auto", cmap=cmap, vmin=0, vmax=1)
        axis.set_title(title)
        axis.set_xlabel("global KV block")
        axis.set_ylabel("global Q block")
    figure.suptitle(f"RainFusion mask: step={key[0]}, layer={key[1]}, batch={batch}, head={head}")
    filename = f"rainfusion_mask_step{key[0]:04d}_layer{key[1]:04d}_head{head:04d}.png"
    figure.savefig(output_dir / filename, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-q-dir", type=Path, required=True, help="USP/full-Q trace directory")
    parser.add_argument("--cp-dir", type=Path, required=True, help="AllGather-KV CP trace directory")
    parser.add_argument("--plot-dir", type=Path, help="write per-step/layer mask PNGs here")
    parser.add_argument("--batch", type=int, default=0, help="batch index to draw, default: 0")
    parser.add_argument("--head", type=int, default=0, help="global head index to draw, default: 0")
    args = parser.parse_args()
    full_groups = defaultdict(list)
    cp_groups = defaultdict(list)
    for trace in _load(args.full_q_dir):
        full_groups[_key(trace)].append(trace)
    for trace in _load(args.cp_dir):
        cp_groups[_key(trace)].append(trace)
    common = sorted(full_groups.keys() & cp_groups.keys())
    if not common:
        raise ValueError("no common (step, layer) trace files")
    for key in common:
        reference = _reconstruct_full_q(full_groups[key])
        reconstructed = _reconstruct_cp(cp_groups[key])
        if reference.shape != reconstructed.shape:
            raise ValueError(f"step={key[0]} layer={key[1]} shape mismatch: {reference.shape} vs {reconstructed.shape}")
        diff = reference != reconstructed
        if args.plot_dir is not None:
            _save_mask_plot(reference, reconstructed, key, args.plot_dir, args.batch, args.head)
        changed_rows = torch.where(diff.any(dim=(0, 1, 3)))[0].tolist()
        print(
            f"step={key[0]} layer={key[1]} diff={int(diff.sum())}/{diff.numel()} "
            f"ratio={diff.float().mean().item():.8f} changed_q_blocks={changed_rows}"
        )


if __name__ == "__main__":
    main()
