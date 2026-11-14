# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Assemble a servable Boundless-World-Model (BWM) directory.

BWM (https://huggingface.co/BLM-Lab/Boundless-World-Model) releases a single
Wan-native-format checkpoint (fine-tuned Wan2.2-TI2V-5B DiT + action
encoder). vLLM-Omni's Wan transformer loads diffusers-format weights, so this
script:

1. downloads the diffusers-format Wan2.2-TI2V-5B base
   (``Wan-AI/Wan2.2-TI2V-5B-Diffusers``: transformer config + weights, VAE);
2. downloads the BWM checkpoint (``step-12000.safetensors``);
3. converts the fine-tuned DiT weights to diffusers naming and overlays them
   on the base transformer state (the text pathway keeps base weights; BWM
   runs with text conditioning disabled);
4. splits the action-encoder weights into an ``action_encoder/`` component;
5. writes a ``model_index.json`` with ``_class_name:
   BoundlessWorldModelPipeline``.

Usage:
    python examples/offline_inference/bwm/download_bwm.py \
        --output-dir models/BWM [--base-dir <existing diffusers base>]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# Wan-native -> diffusers weight-name fragments (the standard Wan conversion
# map used by diffusers' convert_wan_to_diffusers.py, restricted to keys that
# appear in the BWM checkpoint).
TOP_LEVEL_RENAMES = {
    "time_embedding.0.": "condition_embedder.time_embedder.linear_1.",
    "time_embedding.2.": "condition_embedder.time_embedder.linear_2.",
    "time_projection.1.": "condition_embedder.time_proj.",
    "head.head.": "proj_out.",
    "head.modulation": "scale_shift_table",
}
BLOCK_RENAMES = {
    ".self_attn.q.": ".attn1.to_q.",
    ".self_attn.k.": ".attn1.to_k.",
    ".self_attn.v.": ".attn1.to_v.",
    ".self_attn.o.": ".attn1.to_out.0.",
    ".self_attn.norm_q.": ".attn1.norm_q.",
    ".self_attn.norm_k.": ".attn1.norm_k.",
    ".cross_attn.q.": ".attn2.to_q.",
    ".cross_attn.k.": ".attn2.to_k.",
    ".cross_attn.v.": ".attn2.to_v.",
    ".cross_attn.o.": ".attn2.to_out.0.",
    ".cross_attn.norm_q.": ".attn2.norm_q.",
    ".cross_attn.norm_k.": ".attn2.norm_k.",
    ".ffn.0.": ".ffn.net.0.proj.",
    ".ffn.2.": ".ffn.net.2.",
    ".norm3.": ".norm2.",
    ".modulation": ".scale_shift_table",
}
ACTION_PREFIX = "pipe.action_encoder."


def convert_dit_key(key: str) -> str:
    for old, new in TOP_LEVEL_RENAMES.items():
        if key.startswith(old) or key == old:
            return key.replace(old, new, 1)
    if key.startswith("blocks."):
        for old, new in BLOCK_RENAMES.items():
            if old in key:
                return key.replace(old, new, 1)
    return key  # e.g. patch_embedding.*


def load_base_transformer_state(transformer_dir: Path) -> dict[str, torch.Tensor]:
    index_file = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    if index_file.exists():
        index = json.loads(index_file.read_text())
        shards = sorted(set(index["weight_map"].values()))
        state: dict[str, torch.Tensor] = {}
        for shard in shards:
            state.update(load_file(str(transformer_dir / shard)))
        return state
    return load_file(str(transformer_dir / "diffusion_pytorch_model.safetensors"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-repo",
        default="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        help="Diffusers-format Wan2.2-TI2V-5B repo (transformer + vae)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Existing local diffusers-format base (skips base download)",
    )
    parser.add_argument("--bwm-repo", default="BLM-Lab/Boundless-World-Model")
    parser.add_argument("--bwm-file", default="step-12000.safetensors")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download, snapshot_download

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    if args.base_dir is not None:
        base = args.base_dir
    else:
        base = Path(
            snapshot_download(
                args.base_repo,
                allow_patterns=["transformer/*", "vae/*", "model_index.json"],
            )
        )

    print(f"[1/4] Copying VAE from {base}")
    if not (out / "vae").exists():
        shutil.copytree(base / "vae", out / "vae")

    print(f"[2/4] Downloading BWM checkpoint {args.bwm_repo}/{args.bwm_file}")
    ckpt_path = hf_hub_download(args.bwm_repo, args.bwm_file)
    bwm_state = load_file(ckpt_path)

    print("[3/4] Converting DiT weights to diffusers naming")
    base_state = load_base_transformer_state(base / "transformer")
    converted, action_state, unmatched = {}, {}, []
    for key, tensor in bwm_state.items():
        if key.startswith(ACTION_PREFIX):
            action_state[key[len(ACTION_PREFIX) :]] = tensor
            continue
        new_key = convert_dit_key(key)
        if new_key not in base_state:
            unmatched.append((key, new_key))
            continue
        converted[new_key] = tensor
    if unmatched:
        raise RuntimeError(f"Unmapped checkpoint keys (conversion map incomplete): {unmatched[:10]}")

    merged = dict(base_state)
    merged.update(converted)
    print(
        f"  transformer: {len(converted)} fine-tuned keys over {len(base_state)} base keys; "
        f"action encoder: {len(action_state)} keys"
    )

    tdir = out / "transformer"
    tdir.mkdir(exist_ok=True)
    shutil.copy(base / "transformer" / "config.json", tdir / "config.json")
    save_file(merged, str(tdir / "diffusion_pytorch_model.safetensors"))

    adir = out / "action_encoder"
    adir.mkdir(exist_ok=True)
    save_file(action_state, str(adir / "diffusion_pytorch_model.safetensors"))

    print("[4/4] Writing model_index.json")
    (out / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "BoundlessWorldModelPipeline",
                "transformer": ["diffusers", "WanTransformer3DModel"],
                "vae": ["diffusers", "AutoencoderKLWan"],
                "action_encoder": ["vllm_omni", "BWMActionEncoder"],
                "bwm": {"action_dim": 14, "action_type": "eef_abs"},
            },
            indent=2,
        )
    )
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
