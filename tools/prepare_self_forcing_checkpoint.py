# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Assemble a vLLM-Omni model directory for the Self-Forcing causal DMD generator.

The Self-Forcing release ships only the distilled generator
(``checkpoints/self_forcing_dmd.pt``); the UMT5 text encoder, tokenizer and Wan
VAE come from Wan2.1-T2V-1.3B-Diffusers. This script links those components,
converts the generator's EMA weights to safetensors under ``transformer/`` and
writes the ``model_index.json`` that selects ``SelfForcingPipeline``.

    python tools/prepare_self_forcing_checkpoint.py --output-dir ./self-forcing-dmd
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

BASE_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
SELF_FORCING_REPO = "gdhe17/Self-Forcing"
SELF_FORCING_CHECKPOINT = "checkpoints/self_forcing_dmd.pt"
BASE_COMPONENTS = ("tokenizer", "text_encoder", "vae")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, help="Directory to write the assembled model to.")
    parser.add_argument(
        "--base-model",
        default=BASE_REPO,
        help="Wan2.1-T2V-1.3B-Diffusers repo ID or a local copy of it.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=f"Local Self-Forcing checkpoint; downloaded from {SELF_FORCING_REPO} when omitted.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy the base components instead of symlinking them (about 11 GiB).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Heavy imports stay below argument parsing so --help needs no torch install.
    import torch
    from safetensors.torch import save_file

    from vllm_omni.diffusion.models.self_forcing.transformer import (
        SELF_FORCING_CHECKPOINT_CONTRACT,
        SELF_FORCING_ROLLOUT_DEFAULTS,
    )
    from vllm_omni.transformers_utils.repo_utils import hf_api

    api = hf_api()
    base_dir = Path(args.base_model)
    if not base_dir.is_dir():
        base_dir = Path(
            api.snapshot_download(
                args.base_model,
                allow_patterns=["model_index.json", *(f"{name}/*" for name in BASE_COMPONENTS)],
            )
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Checked before the checkpoint is fetched, so a bad pair of paths costs nothing:
    # assembling into the base model directory would delete the components it reads from,
    # since each target below names the same path as its source and stale targets go first.
    if output_dir.resolve() == base_dir.resolve():
        raise ValueError(
            f"--output-dir must not be the base model directory ({base_dir}). Assembling in place "
            f"would delete the base components this script reads from."
        )

    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path(api.hf_hub_download(SELF_FORCING_REPO, SELF_FORCING_CHECKPOINT))
    )

    for name in BASE_COMPONENTS:
        source = (base_dir / name).resolve()
        target = output_dir / name
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
        if args.copy:
            shutil.copytree(source, target)
        else:
            target.symlink_to(source, target_is_directory=True)

    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)["generator_ema"]
    if not all(name.startswith("model.") for name in state_dict):
        raise ValueError(f"{checkpoint} does not hold a Self-Forcing generator under the 'model.' prefix.")
    weights = {name.removeprefix("model."): tensor.contiguous() for name, tensor in state_dict.items()}
    transformer_dir = output_dir / "transformer"
    transformer_dir.mkdir(exist_ok=True)
    save_file(weights, str(transformer_dir / "diffusion_pytorch_model.safetensors"))
    transformer_config = {
        "_class_name": "SelfForcingTransformer3DModel",
        **{
            name: list(value) if isinstance(value, tuple) else value
            for name, value in SELF_FORCING_CHECKPOINT_CONTRACT.items()
        },
        **SELF_FORCING_ROLLOUT_DEFAULTS,
    }
    (transformer_dir / "config.json").write_text(json.dumps(transformer_config, indent=2) + "\n")

    base_index = json.loads((base_dir / "model_index.json").read_text())
    model_index = {
        "_class_name": "SelfForcingPipeline",
        **{name: base_index[name] for name in BASE_COMPONENTS},
        "transformer": ["vllm_omni", "SelfForcingTransformer3DModel"],
        "self_forcing_checkpoint": f"{SELF_FORCING_REPO}/{SELF_FORCING_CHECKPOINT}",
    }
    (output_dir / "model_index.json").write_text(json.dumps(model_index, indent=2) + "\n")
    print(f"Assembled Self-Forcing model at {output_dir.resolve()}")


if __name__ == "__main__":
    main()
