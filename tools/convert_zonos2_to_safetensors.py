#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Convert the official ZONOS2 checkpoint (model.pth pickle + params.json) to safetensors.

This tool is the ONLY place in the repository allowed to read the pickle
container (``torch.load(..., weights_only=False)``). Runtime model code must
consume the safetensors output and must never touch ``model.pth``.

The key normalization mirrors the official loader
(``zonos2/models/weight.py::_normalize_zonos2_state_dict``):
  * ``.parametrizations.X.original`` -> ``.X``        (weight-norm unwrap)
  * drop ``.router.ent_denom`` / ``.router.normalized_entropy`` (training-only)

Outputs written to ``--output-dir``:
  model.safetensors   every tensor, original dtype (bf16), normalized keys
  config.json         params.json + architectures/model_type/torch_dtype
  manifest.json       per-tensor {src_key, key, shape, dtype, sha256}, the
                      removed/renamed lists, and totals — the L0 weight
                      completeness baseline ("no tensor left behind").

Usage:
  python tools/convert_zonos2_to_safetensors.py \
      --input Zyphra/ZONOS2 --output-dir /path/to/zonos2-safetensors

``--input`` accepts an HF repo id, a local directory containing model.pth +
params.json, or a direct path to a .pth/.pt file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

ARCHITECTURES = ["Zonos2ForConditionalGeneration"]
MODEL_TYPE = "zonos2"

# Keys dropped by the official loader (training-only router entropy stats).
_DROP_SUBSTRINGS = (".router.ent_denom", ".router.normalized_entropy")


def _resolve_input(input_path: str) -> tuple[Path, Path]:
    """Return (checkpoint_file, params_json) for a dir / file / HF repo id."""
    p = Path(input_path)
    if p.is_dir():
        ckpt = p / "model.pth"
        params = p / "params.json"
    elif p.is_file() and p.suffix in (".pth", ".pt"):
        ckpt, params = p, p.parent / "params.json"
    else:
        from huggingface_hub import snapshot_download

        snap = Path(snapshot_download(input_path))
        ckpt, params = snap / "model.pth", snap / "params.json"
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    if not params.is_file():
        raise FileNotFoundError(f"params.json not found: {params}")
    return ckpt, params


def _normalize_state_dict(
    sd: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[tuple[str, str]], list[str]]:
    """Mirror the official key normalization. Returns (sd, renamed, removed)."""
    renamed: list[tuple[str, str]] = []
    removed: list[str] = []
    out: dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        if any(sub in key for sub in _DROP_SUBSTRINGS):
            removed.append(key)
            continue
        new_key = key
        if ".parametrizations." in key and key.endswith(".original"):
            new_key = key.replace(".parametrizations.", ".").removesuffix(".original")
        if new_key != key:
            renamed.append((key, new_key))
        if new_key in out:
            raise ValueError(f"key collision after normalization: {new_key}")
        out[new_key] = value
    return out, renamed, removed


def _tensor_sha256(t: torch.Tensor) -> str:
    t = t.detach().cpu().contiguous()
    try:
        buf = t.numpy().tobytes()
    except (TypeError, RuntimeError):
        # numpy has no bf16; reinterpret raw bytes instead (no value change).
        buf = t.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(buf).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="HF repo id / dir with model.pth / direct .pth path")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    ckpt_path, params_path = _resolve_input(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] loading pickle checkpoint: {ckpt_path}", flush=True)
    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    for wrap in ("model", "state_dict"):
        if isinstance(raw, dict) and wrap in raw and isinstance(raw[wrap], dict):
            print(f"      unwrapped top-level key: {wrap!r}", flush=True)
            raw = raw[wrap]
            break
    if not isinstance(raw, dict) or not all(isinstance(v, torch.Tensor) for v in raw.values()):
        raise ValueError("unexpected checkpoint structure: not a flat tensor dict")

    print(f"[2/5] normalizing keys ({len(raw)} tensors)", flush=True)
    sd, renamed, removed = _normalize_state_dict(raw)
    print(f"      renamed: {len(renamed)}, removed (training-only): {len(removed)}", flush=True)

    print("[3/5] hashing tensors", flush=True)
    entries = []
    total_params = 0
    for key, t in sd.items():
        entries.append(
            {
                "key": key,
                "src_key": next((o for o, n in renamed if n == key), key),
                "shape": list(t.shape),
                "dtype": str(t.dtype).removeprefix("torch."),
                "numel": t.numel(),
                "sha256": _tensor_sha256(t),
            }
        )
        total_params += t.numel()

    print("[4/5] writing model.safetensors", flush=True)
    st_path = out_dir / "model.safetensors"
    sd = {k: (t if t.is_contiguous() else t.contiguous()) for k, t in sd.items()}
    save_file(sd, str(st_path), metadata={"format": "pt"})

    print("[5/5] writing config.json + manifest.json", flush=True)
    with open(params_path) as f:
        config = json.load(f)
    config["architectures"] = ARCHITECTURES
    config["model_type"] = MODEL_TYPE
    config["torch_dtype"] = "bfloat16"
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Source-file digests make the manifest self-contained for provenance.
    def _file_sha256(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as fp:
            for chunk in iter(lambda: fp.read(1 << 26), b""):
                h.update(chunk)
        return h.hexdigest()

    print("      hashing source files for provenance", flush=True)
    groups: dict[str, int] = {}
    for e in entries:
        top = e["key"].split(".")[0]
        groups[top] = groups.get(top, 0) + 1

    manifest = {
        "source_checkpoint": str(ckpt_path),
        "source_params": str(params_path),
        "source_sha256": {
            ckpt_path.name: _file_sha256(ckpt_path),
            params_path.name: _file_sha256(params_path),
        },
        "tensor_count": len(entries),
        "total_params": total_params,
        "groups": groups,
        "renamed": [{"src": o, "dst": n} for o, n in renamed],
        "removed_training_only": removed,
        "tensors": entries,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)

    print(f"DONE: {len(entries)} tensors, {total_params / 1e9:.2f}B params -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
