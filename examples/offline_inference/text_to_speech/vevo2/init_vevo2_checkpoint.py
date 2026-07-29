# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-time setup helper for Vevo2 checkpoints.

The published ``RMSnow/Vevo2`` checkpoint ships its real configs in
sub-folders (``contentstyle_modeling/posttrained/config.json`` etc.) but
not at the root. vLLM-Omni's stage-config factory dispatches by reading
the root ``config.json``'s ``model_type``, so without this file the
loader fails with::

    ValueError: Could not determine model_type for model: <path>

Run this script once per local checkpoint to write a minimal root
``config.json`` whose ``model_type`` is ``vevo2`` and whose
``architectures`` is ``["Vevo2ForCausalLM"]``. The checkpoint's actual
sub-component configs are still consumed by
``Vevo2InferencePipeline.__init__`` at load time and are unaffected.

Usage:
    python init_vevo2_checkpoint.py /path/to/Vevo2 [--overwrite]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vllm_omni.model_executor.models.vevo2.configuration_vevo2 import Vevo2Config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "checkpoint_dir",
        help="Local Vevo2 checkpoint directory (e.g. ./ckpts/Vevo2).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing root config.json. Otherwise the script is a no-op.",
    )
    args = parser.parse_args()

    ckpt = Path(args.checkpoint_dir).resolve()
    if not ckpt.is_dir():
        raise SystemExit(f"Not a directory: {ckpt}")

    root_cfg = ckpt / "config.json"
    if root_cfg.exists() and not args.overwrite:
        print(f"Already present: {root_cfg}  (re-run with --overwrite to regenerate)")
        return

    cfg = Vevo2Config()
    cfg.architectures = ["Vevo2ForCausalLM"]
    root_cfg.write_text(cfg.to_json_string())
    print(f"Wrote {root_cfg}  (model_type=vevo2)")

    # vLLM's default_loader enumerates ``model.safetensors`` /
    # ``pytorch_model.bin`` at the root and aborts with
    # ``Cannot find any model weights with <path>`` if absent. The real
    # Vevo2 weights live in sub-folders (the upstream
    # ``Vevo2InferencePipeline`` loads them itself in
    # :meth:`Vevo2ForCausalLM.load_weights`), so the wrapper exhausts
    # vLLM's iterator and discards it. Write a 0-tensor placeholder so
    # the enumeration step succeeds.
    placeholder = ckpt / "model.safetensors"
    if not placeholder.exists() or args.overwrite:
        try:
            import torch as _torch
            from safetensors.torch import save_file

            save_file({"_vevo2_placeholder": _torch.zeros(0, dtype=_torch.float32)}, placeholder)
            print(f"Wrote {placeholder}  (placeholder for vLLM weight enumeration)")
        except Exception as exc:  # pragma: no cover - rare envs
            print(f"Warning: could not write {placeholder}: {exc}")

    # vLLM constructs ``AutoTokenizer.from_pretrained(<root>)`` for its
    # structured-output manager; it must find tokenizer files at the
    # root or fall back to a model-type lookup which fails for the
    # custom ``vevo2`` ``model_type``. Mirror the AR's tokenizer files
    # to the root.
    import shutil

    ar_tok_dir = ckpt / cfg.ar_subdir
    if ar_tok_dir.is_dir():
        tokenizer_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
            "special_tokens_map.json",
            "generation_config.json",
        ]
        copied = []
        for fname in tokenizer_files:
            src = ar_tok_dir / fname
            dst = ckpt / fname
            if src.is_file() and (not dst.exists() or args.overwrite):
                shutil.copy2(src, dst)
                copied.append(fname)
        if copied:
            print(f"Copied tokenizer files from {ar_tok_dir} -> root: {', '.join(copied)}")


if __name__ == "__main__":
    main()
