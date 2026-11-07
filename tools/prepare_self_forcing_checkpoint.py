# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Assemble a vLLM-Omni model directory for a Self-Forcing-family causal DMD generator.

The releases ship only the distilled generator; the UMT5 text encoder,
tokenizer and Wan VAE come from Wan2.1-T2V-1.3B-Diffusers. This script links
those components, converts the generator weights to safetensors under
``transformer/`` and writes the ``model_index.json`` that selects the pipeline.

    python tools/prepare_self_forcing_checkpoint.py --output-dir ./self-forcing-dmd
    python tools/prepare_self_forcing_checkpoint.py --release minwm-action2v --output-dir ./minwm-action2v-dmd
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

BASE_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
BASE_COMPONENTS = ("tokenizer", "text_encoder", "vae")


@dataclass(frozen=True)
class Release:
    repo: str
    checkpoint: str
    # Generator state dicts to try in order; the first one present is converted.
    state_keys: tuple[str, ...]
    rollout: dict[str, int | bool]
    pipeline: str


RELEASES = {
    "self-forcing": Release(
        repo="gdhe17/Self-Forcing",
        checkpoint="checkpoints/self_forcing_dmd.pt",
        state_keys=("generator_ema",),
        rollout={"num_frames_per_block": 3, "window_frames": 21, "sink_frames": 0, "use_prope": False},
        pipeline="SelfForcingPipeline",
    ),
    "minwm-action2v": Release(
        repo="MIN-Lab/minWM",
        checkpoint="Wan21/Action2V/dmd/model.pt",
        state_keys=("generator_ema", "generator"),
        rollout={"num_frames_per_block": 4, "window_frames": 20, "sink_frames": 0, "use_prope": True},
        pipeline="MinWMAction2VPipeline",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, help="Directory to write the assembled model to.")
    parser.add_argument("--release", choices=sorted(RELEASES), default="self-forcing", help="Generator to assemble.")
    parser.add_argument(
        "--base-model",
        default=BASE_REPO,
        help="Wan2.1-T2V-1.3B-Diffusers repo ID or a local copy of it.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Local generator checkpoint; downloaded from the release repo when omitted.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy the base components instead of symlinking them (about 11 GiB).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release = RELEASES[args.release]
    # Heavy imports stay below argument parsing so --help needs no torch install.
    import torch
    from safetensors.torch import save_file

    from vllm_omni.diffusion.models.self_forcing.transformer import SELF_FORCING_CHECKPOINT_CONTRACT
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
        Path(args.checkpoint) if args.checkpoint else Path(api.hf_hub_download(release.repo, release.checkpoint))
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

    states = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_key = next((key for key in release.state_keys if key in states), None)
    if state_key is None:
        raise ValueError(f"{checkpoint} holds none of the generator state dicts {release.state_keys}.")
    state_dict = states[state_key]
    if not all(name.startswith("model.") for name in state_dict):
        raise ValueError(f"{checkpoint} does not hold a generator under the 'model.' prefix.")
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
        **release.rollout,
    }
    (transformer_dir / "config.json").write_text(json.dumps(transformer_config, indent=2) + "\n")

    base_index = json.loads((base_dir / "model_index.json").read_text())
    model_index = {
        "_class_name": release.pipeline,
        **{name: base_index[name] for name in BASE_COMPONENTS},
        "transformer": ["vllm_omni", "SelfForcingTransformer3DModel"],
        "generator_checkpoint": f"{release.repo}/{release.checkpoint}:{state_key}",
    }
    (output_dir / "model_index.json").write_text(json.dumps(model_index, indent=2) + "\n")
    print(f"Assembled {release.pipeline} model at {output_dir.resolve()}")


if __name__ == "__main__":
    main()
