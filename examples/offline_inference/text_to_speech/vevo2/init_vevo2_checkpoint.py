# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-time setup helper for Vevo2 checkpoints.

The published ``RMSnow/Vevo2`` checkpoint ships its real configs in
sub-folders (``contentstyle_modeling/posttrained/config.json`` etc.) but
not at the root. Three root artifacts have to exist before vLLM-Omni can
load it:

``config.json``
    vLLM-Omni's stage-config factory dispatches by reading the root
    ``config.json``'s ``model_type``; without it the loader fails with
    ``ValueError: Could not determine model_type for model: <path>``.

``model.safetensors``
    vLLM's ``default_loader`` enumerates root weight files and aborts with
    ``Cannot find any model weights with <path>`` if none exist. The real
    Vevo2 weights live in sub-folders (the upstream
    ``Vevo2InferencePipeline`` loads them itself in
    :meth:`Vevo2ForCausalLM.load_weights`), so a 0-tensor placeholder is
    enough to satisfy the enumeration step.

tokenizer files
    vLLM constructs ``AutoTokenizer.from_pretrained(<root>)`` for its
    structured-output manager; it must find tokenizer files at the root or
    fall back to a model-type lookup, which fails for the custom ``vevo2``
    ``model_type``.

This script is **idempotent and repairs in place**: it writes whatever is
missing, then verifies every required artifact before reporting success. A
run interrupted half-way is fixed by simply running it again, and any
failure is fatal rather than a printed warning -- a checkpoint that reports
success must actually load.

Usage:
    python init_vevo2_checkpoint.py /path/to/Vevo2 [--overwrite]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from vllm_omni.model_executor.models.vevo2.configuration_vevo2 import Vevo2Config

# Tokenizer files mirrored from the AR sub-checkpoint to the root, in the
# order HF looks for them.
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "generation_config.json",
)

# ``AutoTokenizer`` needs a vocabulary; a fast ``tokenizer.json`` or the slow
# ``vocab.json`` pair both satisfy it, so require at least one rather than
# pinning a layout the upstream checkpoint may change.
TOKENIZER_ALTERNATIVES = ("tokenizer.json", "vocab.json")


def missing_artifacts(ckpt: Path) -> list[str]:
    """Return the root artifacts that still need to be created."""
    missing = [name for name in ("config.json", "model.safetensors") if not (ckpt / name).is_file()]
    if not any((ckpt / name).is_file() for name in TOKENIZER_ALTERNATIVES):
        missing.append(" or ".join(TOKENIZER_ALTERNATIVES))
    return missing


def write_root_config(ckpt: Path, cfg: Vevo2Config, overwrite: bool) -> None:
    root_cfg = ckpt / "config.json"
    if root_cfg.is_file() and not overwrite:
        return
    cfg.architectures = ["Vevo2ForCausalLM"]
    root_cfg.write_text(cfg.to_json_string())
    print(f"Wrote {root_cfg}  (model_type=vevo2)")


def write_placeholder_weights(ckpt: Path, overwrite: bool) -> None:
    placeholder = ckpt / "model.safetensors"
    if placeholder.is_file() and not overwrite:
        return
    # Any failure here is fatal: the placeholder is required for the engine to
    # start, so degrading it to a warning would leave a checkpoint that
    # reports success and then fails later with an opaque loader error.
    import torch
    from safetensors.torch import save_file

    save_file({"_vevo2_placeholder": torch.zeros(0, dtype=torch.float32)}, placeholder)
    print(f"Wrote {placeholder}  (placeholder for vLLM weight enumeration)")


def copy_tokenizer_files(ckpt: Path, cfg: Vevo2Config, overwrite: bool) -> None:
    ar_tok_dir = ckpt / cfg.ar_subdir
    if not ar_tok_dir.is_dir():
        raise SystemExit(
            f"Missing AR sub-checkpoint: {ar_tok_dir}\n"
            f"The tokenizer files are mirrored from there, so {ckpt} does not look like a "
            f"Vevo2 checkpoint. Download it first:\n"
            f"    hf download RMSnow/Vevo2 --local-dir {ckpt}"
        )

    copied = []
    for fname in TOKENIZER_FILES:
        src = ar_tok_dir / fname
        dst = ckpt / fname
        if src.is_file() and (not dst.exists() or overwrite):
            shutil.copy2(src, dst)
            copied.append(fname)
    if copied:
        print(f"Copied tokenizer files from {ar_tok_dir} -> root: {', '.join(copied)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "checkpoint_dir",
        help="Local Vevo2 checkpoint directory (e.g. ./ckpts/Vevo2).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate every root artifact even if it already exists.",
    )
    args = parser.parse_args()

    ckpt = Path(args.checkpoint_dir).resolve()
    if not ckpt.is_dir():
        raise SystemExit(f"Not a directory: {ckpt}")

    missing = missing_artifacts(ckpt)
    if not missing and not args.overwrite:
        print(f"Already initialised: {ckpt}  (re-run with --overwrite to regenerate)")
        return
    if missing and (ckpt / "config.json").is_file():
        # A previous run was interrupted, or wrote the config and then failed.
        print(f"Repairing partially initialised checkpoint; missing: {', '.join(missing)}")

    cfg = Vevo2Config()
    write_root_config(ckpt, cfg, args.overwrite)
    write_placeholder_weights(ckpt, args.overwrite)
    copy_tokenizer_files(ckpt, cfg, args.overwrite)

    # Only claim success once every required artifact is actually present.
    still_missing = missing_artifacts(ckpt)
    if still_missing:
        raise SystemExit(f"Initialisation incomplete for {ckpt}; still missing: {', '.join(still_missing)}")
    print(f"Checkpoint initialised: {ckpt}")


if __name__ == "__main__":
    main()
