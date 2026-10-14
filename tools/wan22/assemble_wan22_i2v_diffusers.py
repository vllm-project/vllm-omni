#!/usr/bin/env python3
"""
Assemble a Wan2.2-I2V-A14B-Diffusers-style model directory using a Diffusers
skeleton and optional replacement transformer checkpoints.

This tool does NOT run any external conversion step. You can use it in two
ways:
- keep the original weights from the Diffusers skeleton
- replace transformer/transformer_2 with converted checkpoints such as
  LightX2V outputs
- use legacy LightX2V arg names (--high-noise-weight/--low-noise-weight),
  which are accepted as aliases

Typical use:
  python tools/wan22/assemble_wan22_i2v_diffusers.py \
    --diffusers-skeleton /path/to/Wan2.2-I2V-A14B-Diffusers \
    --transformer-weight /path/to/high_noise_out/diffusion_pytorch_model.safetensors \
    --transformer-2-weight /path/to/low_noise_out/diffusion_pytorch_model.safetensors \
    --output-dir /path/to/Wan2.2-I2V-A14B-Custom-Diffusers
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

WEIGHT_CANDIDATES = (
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.bin",
    "diffusion_pytorch_model.pt",
    "model.safetensors",
    "pytorch_model.bin",
    "model.pt",
)
WEIGHT_INDEX_CANDIDATES = (
    "diffusion_pytorch_model.safetensors.index.json",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)

ROOT_REQUIRED_FILES = ("model_index.json",)
ROOT_REQUIRED_DIRS = ("tokenizer", "text_encoder", "vae", "transformer", "transformer_2")
OPTIONAL_DIRS = ("image_encoder", "image_processor", "scheduler", "feature_extractor")


class AssembleError(RuntimeError):
    pass


@dataclass(frozen=True)
class WeightSpec:
    kind: str  # "single" | "sharded"
    single_file: Path | None = None
    index_file: Path | None = None
    shard_files: tuple[Path, ...] = ()


def _load_shard_files_from_index(index_file: Path, role: str) -> tuple[Path, ...]:
    try:
        with index_file.open(encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        raise AssembleError(f"Failed to parse {role} index file: {index_file}. error={exc}") from exc

    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise AssembleError(f"Invalid {role} index file (missing/empty weight_map): {index_file}")

    shard_names = sorted({str(v) for v in weight_map.values()})
    shard_paths: list[Path] = []
    missing: list[str] = []
    for shard_name in shard_names:
        shard_path = index_file.parent / shard_name
        if not shard_path.is_file():
            missing.append(str(shard_path))
        else:
            shard_paths.append(shard_path)

    if missing:
        raise AssembleError(f"{role} index references missing shard file(s): " + ", ".join(missing))

    if not shard_paths:
        raise AssembleError(f"No shard files referenced by {role} index: {index_file}")

    return tuple(shard_paths)


def _resolve_weight_spec(path: Path, role: str) -> WeightSpec:
    if path.is_file():
        return WeightSpec(kind="single", single_file=path)

    if path.is_dir():
        for name in WEIGHT_CANDIDATES:
            candidate = path / name
            if candidate.is_file():
                return WeightSpec(kind="single", single_file=candidate)

        for index_name in WEIGHT_INDEX_CANDIDATES:
            index_file = path / index_name
            if not index_file.is_file():
                continue
            shard_files = _load_shard_files_from_index(index_file, role=role)
            return WeightSpec(
                kind="sharded",
                index_file=index_file,
                shard_files=shard_files,
            )

        shard_candidates = sorted(path.glob("diffusion_pytorch_model-*.safetensors"))
        if shard_candidates:
            raise AssembleError(
                f"Detected sharded {role} files under {path}, but index json is missing. "
                f"Expected one of: {', '.join(WEIGHT_INDEX_CANDIDATES)}"
            )

        raise AssembleError(
            f"Cannot find {role} weight under directory: {path}. "
            f"Expected one of single files [{', '.join(WEIGHT_CANDIDATES)}] "
            f"or sharded index files [{', '.join(WEIGHT_INDEX_CANDIDATES)}]."
        )

    raise AssembleError(f"{role} path does not exist: {path}")


def _canonical_weight_name(weight_file: Path) -> str:
    suffix = weight_file.suffix.lower()
    if suffix == ".safetensors":
        return "diffusion_pytorch_model.safetensors"
    if suffix == ".bin":
        return "diffusion_pytorch_model.bin"
    if suffix == ".pt":
        return "diffusion_pytorch_model.pt"
    return weight_file.name


def _validate_skeleton(skeleton: Path) -> None:
    if not skeleton.is_dir():
        raise AssembleError(f"--diffusers-skeleton is not a directory: {skeleton}")

    for file_name in ROOT_REQUIRED_FILES:
        if not (skeleton / file_name).is_file():
            raise AssembleError(f"Missing required file in skeleton: {skeleton / file_name}")

    for dir_name in ROOT_REQUIRED_DIRS:
        if not (skeleton / dir_name).is_dir():
            raise AssembleError(f"Missing required directory in skeleton: {skeleton / dir_name}")

    if not (skeleton / "transformer" / "config.json").is_file():
        raise AssembleError(f"Missing transformer config: {skeleton / 'transformer/config.json'}")

    if not (skeleton / "transformer_2" / "config.json").is_file():
        raise AssembleError(f"Missing transformer_2 config: {skeleton / 'transformer_2/config.json'}")


def _ensure_clean_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise AssembleError(
                f"Output directory already exists: {output_dir}. Use --overwrite to remove and recreate it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)


def _copy_or_link_dir(src: Path, dst: Path, asset_mode: str) -> None:
    if asset_mode == "copy":
        shutil.copytree(src, dst)
    elif asset_mode == "symlink":
        dst.symlink_to(src, target_is_directory=True)
    else:
        raise AssembleError(f"Unknown asset mode: {asset_mode}")


def _materialize_weight(weight: WeightSpec, dst_dir: Path, role: str) -> tuple[Path, ...]:
    if weight.kind == "single":
        assert weight.single_file is not None
        dst = dst_dir / _canonical_weight_name(weight.single_file)
        shutil.copy2(weight.single_file, dst)
        return (dst,)

    if weight.kind == "sharded":
        assert weight.index_file is not None
        copied: list[Path] = []
        index_dst = dst_dir / weight.index_file.name
        shutil.copy2(weight.index_file, index_dst)
        copied.append(index_dst)
        for shard_file in weight.shard_files:
            shard_dst = dst_dir / shard_file.name
            shutil.copy2(shard_file, shard_dst)
            copied.append(shard_dst)
        return tuple(copied)

    raise AssembleError(f"Unknown {role} weight kind: {weight.kind}")


def _assemble(
    skeleton: Path,
    output_dir: Path,
    transformer_weight: WeightSpec,
    transformer_2_weight: WeightSpec,
    asset_mode: str,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    shutil.copy2(skeleton / "model_index.json", output_dir / "model_index.json")

    for dir_name in ROOT_REQUIRED_DIRS:
        if dir_name in ("transformer", "transformer_2"):
            continue
        _copy_or_link_dir(skeleton / dir_name, output_dir / dir_name, asset_mode)

    for dir_name in OPTIONAL_DIRS:
        src_dir = skeleton / dir_name
        if src_dir.is_dir():
            _copy_or_link_dir(src_dir, output_dir / dir_name, asset_mode)

    (output_dir / "transformer").mkdir(parents=True, exist_ok=True)
    (output_dir / "transformer_2").mkdir(parents=True, exist_ok=True)

    shutil.copy2(skeleton / "transformer" / "config.json", output_dir / "transformer" / "config.json")
    shutil.copy2(skeleton / "transformer_2" / "config.json", output_dir / "transformer_2" / "config.json")

    transformer_copied = _materialize_weight(transformer_weight, output_dir / "transformer", role="transformer")
    transformer_2_copied = _materialize_weight(
        transformer_2_weight,
        output_dir / "transformer_2",
        role="transformer_2",
    )

    return transformer_copied, transformer_2_copied


def _validate_output(
    output_dir: Path,
    transformer_copied: tuple[Path, ...],
    transformer_2_copied: tuple[Path, ...],
) -> None:
    if not (output_dir / "model_index.json").is_file():
        raise AssembleError("Output validation failed: model_index.json missing")

    required_paths = (
        output_dir / "tokenizer",
        output_dir / "text_encoder",
        output_dir / "vae",
        output_dir / "transformer" / "config.json",
        output_dir / "transformer_2" / "config.json",
        *transformer_copied,
        *transformer_2_copied,
    )
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise AssembleError("Output validation failed, missing: " + ", ".join(missing))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble a Wan2.2-I2V-A14B-Diffusers directory while optionally "
            "replacing transformer and transformer_2 weights."
        )
    )
    parser.add_argument(
        "--diffusers-skeleton",
        type=Path,
        required=True,
        help="Path to a local Wan-AI/Wan2.2-I2V-A14B-Diffusers directory.",
    )
    parser.add_argument(
        "--transformer-weight",
        type=Path,
        help=(
            "Optional checkpoint file, or directory containing either a single-file "
            "weight or sharded index+shards for transformer/. If omitted, keep the "
            "skeleton's original transformer weights."
        ),
    )
    parser.add_argument(
        "--transformer-2-weight",
        type=Path,
        help=(
            "Optional checkpoint file, or directory containing either a single-file "
            "weight or sharded index+shards for transformer_2/. If omitted, keep the "
            "skeleton's original transformer_2 weights."
        ),
    )
    parser.add_argument(
        "--high-noise-weight",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--low-noise-weight",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the assembled model.",
    )
    parser.add_argument(
        "--asset-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help=(
            "How to materialize non-transformer assets (tokenizer/text_encoder/vae/optional dirs). "
            "symlink saves disk and is default."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output-dir if it exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    skeleton = args.diffusers_skeleton.resolve()
    output_dir = args.output_dir.resolve()

    if args.transformer_weight is not None and args.high_noise_weight is not None:
        print(
            "[ERROR] --transformer-weight and --high-noise-weight are aliases; please provide only one.",
            file=sys.stderr,
        )
        return 2
    if args.transformer_2_weight is not None and args.low_noise_weight is not None:
        print(
            "[ERROR] --transformer-2-weight and --low-noise-weight are aliases; please provide only one.",
            file=sys.stderr,
        )
        return 2

    transformer_weight_arg = args.transformer_weight if args.transformer_weight is not None else args.high_noise_weight
    transformer_2_weight_arg = (
        args.transformer_2_weight if args.transformer_2_weight is not None else args.low_noise_weight
    )

    transformer_input = (
        transformer_weight_arg.resolve() if transformer_weight_arg is not None else skeleton / "transformer"
    )
    transformer_2_input = (
        transformer_2_weight_arg.resolve() if transformer_2_weight_arg is not None else skeleton / "transformer_2"
    )

    try:
        _validate_skeleton(skeleton)
        transformer_weight = _resolve_weight_spec(transformer_input, role="transformer")
        transformer_2_weight = _resolve_weight_spec(transformer_2_input, role="transformer_2")

        _ensure_clean_output(output_dir, overwrite=args.overwrite)
        transformer_copied, transformer_2_copied = _assemble(
            skeleton=skeleton,
            output_dir=output_dir,
            transformer_weight=transformer_weight,
            transformer_2_weight=transformer_2_weight,
            asset_mode=args.asset_mode,
        )
        _validate_output(output_dir, transformer_copied, transformer_2_copied)
    except AssembleError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    def _weight_summary(copied: tuple[Path, ...]) -> str:
        if len(copied) == 1:
            return copied[0].name
        return f"{copied[0].name} + {len(copied) - 1} shard files"

    print("[OK] Assembled Wan2.2 I2V Diffusers directory:")
    print(f"  output_dir: {output_dir}")
    print(f"  transformer weight: {_weight_summary(transformer_copied)}")
    print(f"  transformer_2 weight: {_weight_summary(transformer_2_copied)}")
    print("\nUse it with vLLM-Omni, for example:")
    print(f"  vllm serve {output_dir} --omni --port 8091")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
