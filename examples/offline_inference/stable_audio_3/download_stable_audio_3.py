# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Download Stable Audio 3 (medium) and prepare it for vLLM-Omni.

The HuggingFace repo ``stabilityai/stable-audio-3-medium`` ships Stability's own
``model_config.json`` (architecture) plus the weights, but it does NOT ship the
two discovery files vLLM-Omni's engine needs:

  * ``model_index.json``     — tells the engine which registered pipeline class to
                               build (``_class_name`` -> registry key, resolved in
                               ``vllm_omni/entrypoints/utils.py:resolve_model_config_path``).
  * ``transformer/config.json`` — read by ``OmniDiffusionConfig.enrich_config``;
                               an empty stub is enough because StableAudio3Pipeline
                               configures itself from ``model_config.json``.

This script downloads the repo and writes both files, mirroring the pattern used
by other custom (non-diffusers) models such as DreamID-Omni.

Note: ``stabilityai/stable-audio-3-medium`` is a gated model. Before running:
  1. Accept the license at https://huggingface.co/stabilityai/stable-audio-3-medium
  2. Log in with ``hf auth login`` (or pass ``--token``).
"""

import argparse
import json
import os

from huggingface_hub import snapshot_download

REPO_ID = "stabilityai/stable-audio-3-medium"
PIPELINE_CLASS = "StableAudio3Pipeline"


def main(output_dir: str, token: str | None) -> None:
    os.makedirs(output_dir, exist_ok=True)

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=output_dir,
        token=token,
    )

    # 1. Discovery: which pipeline class to instantiate.
    model_index_path = os.path.join(output_dir, "model_index.json")
    with open(model_index_path, "w", encoding="utf-8") as f:
        json.dump({"_class_name": PIPELINE_CLASS}, f, indent=2)
    print(f"model_index.json created at {model_index_path}")

    # 2. Empty transformer config stub — StableAudio3Pipeline reads its real
    #    architecture from model_config.json, so an empty dict is sufficient to
    #    satisfy enrich_config's transformer/config.json lookup.
    transformer_dir = os.path.join(output_dir, "transformer")
    os.makedirs(transformer_dir, exist_ok=True)
    transformer_config_path = os.path.join(transformer_dir, "config.json")
    with open(transformer_config_path, "w", encoding="utf-8") as f:
        json.dump({}, f)
    print(f"transformer/config.json created at {transformer_config_path}")

    print(
        f"\nDone. Run inference with:\n"
        f"  python ../text_to_audio/text_to_audio.py --model {output_dir} "
        f'--prompt "An ambient drone with shimmering overtones" --audio-length 30.0'
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare Stable Audio 3 medium.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./stable-audio-3-medium",
        help="Directory to download the model into.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace access token (defaults to the cached login).",
    )
    args = parser.parse_args()
    main(args.output_dir, args.token)
