# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Build an exact AdaLN sidecar from local native weights and a fixed adapter."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm_omni.diffusion.models.minimax_h3.adaln_cache import (
    FORMAT_VERSION,
    MODES,
    MiniMaxH3AdalnCache,
    build_cache,
    canonical_json,
    file_digest,
    input_names,
    schedule_contract,
)
from vllm_omni.diffusion.models.minimax_h3.fasth3 import FastH3WeightFusion
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTArchConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-variant", choices=("fl2va", "ref2va"), required=True)
    parser.add_argument("--mode", choices=tuple(MODES), default="t2va")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--audio-flow-shift", type=float, default=3.0)
    parser.add_argument("--base-schedule", type=float, nargs="+")
    parser.add_argument("--fasth3-adapter", type=Path)
    parser.add_argument("--device", default="cuda", help="Use the serving device type; cpu is for CPU-only fixtures")
    args = parser.parse_args()
    root = args.transformer_path.expanduser().resolve()
    if args.output.exists():
        raise ValueError(f"Refusing to overwrite {args.output}")
    config = json.loads((root / "config.json").read_text())
    arch = MiniMaxH3DiTArchConfig.from_mapping(config)
    variant = "ref2va" if args.mode.startswith("ref2va") else "fl2va"
    if args.model_variant != variant:
        raise ValueError("Cache mode does not match --model-variant")
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("The builder supports CPU fixtures and CUDA serving")
    fusion = None
    if args.fasth3_adapter is not None:
        if args.mode != "t2va" or args.model_variant != "fl2va":
            raise ValueError("FastH3 Preview adapters support T2VA only")
        fusion = FastH3WeightFusion.from_path(
            args.fasth3_adapter,
            head_dim=arch.attention_head_dim,
            num_blocks=arch.num_layers,
            num_refiner_blocks=arch.token_refiner_num_layers,
        )
        if fusion is None:
            raise ValueError("Not a supported fixed FastH3 adapter")
        if args.base_schedule is not None and tuple(args.base_schedule) != fusion.base_schedule:
            raise ValueError("Explicit schedule does not match the FastH3 adapter")
        args.base_schedule = fusion.base_schedule
    num_steps = args.num_inference_steps
    if num_steps is None:
        num_steps = len(args.base_schedule) - 1 if args.base_schedule is not None else 50
    contract = schedule_contract(
        mode=args.mode,
        num_steps=num_steps,
        base_schedule=args.base_schedule,
        flow_shift=args.flow_shift,
        audio_flow_shift=args.audio_flow_shift,
    )
    adapter_sha = file_digest(fusion.source) if fusion is not None else None
    index = root / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        single = root / "model.safetensors"
        with safe_open(str(single), framework="pt", device="cpu") as source:
            weight_map = {name: single.name for name in source.keys()}
    required = input_names(arch.num_layers)
    if missing := required - set(weight_map):
        raise ValueError(f"Native checkpoint is missing AdaLN inputs: {sorted(missing)}")

    def weights():
        ordered = sorted(required, key=lambda name: (not name.startswith("time_embedder."), name))
        for name in ordered:
            filename = (root / weight_map[name]).resolve()
            if not filename.is_relative_to(root):
                # HF snapshot shards may be symlinks to blobs; use the lexical
                # index path for containment, while allowing those symlinks.
                relative = Path(weight_map[name])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Checkpoint index points outside the transformer directory")
            with safe_open(str(filename), framework="pt", device="cpu") as source:
                weight = source.get_tensor(name)
                if fusion is not None:
                    weight = fusion.fuse(name, weight.to(device))
                yield name, weight

    payload, manifest = build_cache(
        arch,
        weights(),
        contract=contract,
        model_variant=args.model_variant,
        adapter_sha256=adapter_sha,
        device=device,
    )
    if fusion is not None and file_digest(fusion.source) != adapter_sha:
        raise ValueError("Adapter changed during cache construction")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".adaln-", suffix=".safetensors", dir=args.output.parent)
    os.close(descriptor)
    try:
        save_file(payload, temporary, metadata={"format_version": FORMAT_VERSION, "manifest": canonical_json(manifest)})
        MiniMaxH3AdalnCache(arch, path=temporary, model_variant=args.model_variant)
        # Atomic no-clobber publication on the same filesystem.
        os.link(temporary, args.output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f"Wrote weight-bound AdaLN sidecar: {args.output}")


if __name__ == "__main__":
    main()
