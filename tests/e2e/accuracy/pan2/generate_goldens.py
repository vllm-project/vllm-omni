# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Generate frozen PAN2 transformer and pipeline reference artifacts from the tiny random-weight checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import ModularPipeline
from diffusers.utils import export_to_video
from PIL import Image
from safetensors.torch import save_file

MODEL = "wuqing157/tiny-pan2-modular-pipe"
# Pin the immutable commit once the tiny checkpoint is published on the Hub.
REVISION: str | None = "3ff3130b4527f61aaab7bde01465c9884dc84cdd"
PROMPT = "A red fox trots through fresh snow in a pine forest at dawn."
NEGATIVE_PROMPT = "blurry, low quality, distorted"
SCHEMA_VERSION = 1
SEED = 42
HEIGHT = 128
WIDTH = 224
NUM_FRAMES = 313
FPS = 24
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 3.0
TRANSFORMER_CASE_SHAPE = (1, 97, 3, 4, 6)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _directory_fingerprint(root: Path) -> dict[str, Any]:
    """Fingerprint every local checkpoint file without recording host paths."""
    if not root.is_dir():
        raise ValueError(f"Local checkpoint must be a directory: {root}")
    files = []
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        file_sha256 = _sha256(path)
        size = path.stat().st_size
        digest.update(f"{relative}\0{size}\0{file_sha256}\n".encode())
        files.append({"path": relative, "size": size, "sha256": file_sha256})
    if not files:
        raise ValueError(f"Local checkpoint contains no files: {root}")
    return {
        "algorithm": "sha256-tree-v1",
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "total_size": sum(entry["size"] for entry in files),
    }


def _repository_provenance() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[4]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {
        "repository": "vllm-project/vllm-omni",
        "revision": revision,
        "dirty": dirty,
        "generator_sha256": _sha256(Path(__file__)),
    }


def _cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().contiguous()


def _normalize_input_image(source: Path, destination: Path) -> Image.Image:
    if not source.is_file():
        raise ValueError(f"I2V input image does not exist: {source}")
    with Image.open(source) as image:
        normalized = image.convert("RGB")
        normalized.load()
    normalized.save(destination, format="PNG", optimize=False)
    return normalized


def _frames_to_uint8(frames: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(frames))
    if tensor.is_floating_point():
        # Match diffusers.utils.export_to_video, which truncates rather than
        # rounds when converting float frames to the encoder's uint8 input.
        tensor = tensor.clamp(0, 1) * 255
    return tensor.to(torch.uint8).contiguous()


def _model_provenance(model: str | None) -> tuple[dict[str, Any], bool]:
    if model is None:
        if REVISION is None:
            raise RuntimeError("Pin REVISION to the published tiny checkpoint commit, or pass --model.")
        return {"kind": "huggingface_hub", "model_id": MODEL, "revision": REVISION}, True

    # A directory hash proves which local bytes were used, but not that those
    # bytes match the Hub checkpoint. Such output is for development comparisons
    # and must never be uploaded as the canonical frozen reference.
    provenance = {
        "kind": "local_checkpoint",
        "checkpoint_fingerprint": _directory_fingerprint(Path(model).expanduser().resolve()),
        "verification": "unverified-local-copy",
    }
    return provenance, False


def generate(task: str, model: str | None, input_image: Path | None, output_root: Path) -> Path:
    model_provenance, publishable = _model_provenance(model)
    repository_provenance = _repository_provenance()
    if publishable and repository_provenance["dirty"]:
        raise RuntimeError("Refusing to generate publishable golden assets from a dirty vllm-omni worktree.")
    model_path, revision = (MODEL, REVISION) if model is None else (str(Path(model).expanduser().resolve()), None)

    output_dir = output_root / task
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_image: Image.Image | None = None
    input_path: Path | None = None
    if task == "i2v":
        if input_image is None:
            raise ValueError("--input-image is required for task=i2v.")
        input_path = output_dir / "input.png"
        normalized_image = _normalize_input_image(input_image, input_path)
    elif input_image is not None:
        raise ValueError("--input-image is only valid for task=i2v.")

    pipe = ModularPipeline.from_pretrained(model_path, revision=revision)
    pipe.load_components(dtype=torch.bfloat16)
    pipe.to("cuda")

    torch.manual_seed(1234)
    hidden_states = torch.randn(TRANSFORMER_CASE_SHAPE, dtype=torch.bfloat16, device="cuda")
    encoder_hidden_states = torch.randn(
        1, 8, pipe.transformer.config.text_embed_dim, dtype=torch.bfloat16, device="cuda"
    )
    timestep = torch.tensor([500.0], device="cuda")
    with torch.inference_mode():
        transformer_output = pipe.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]

    transformer_path = output_dir / "transformer_case.safetensors"
    save_file(
        {
            "hidden_states": _cpu(hidden_states),
            "encoder_hidden_states": _cpu(encoder_hidden_states),
            "timestep": _cpu(timestep),
            "output": _cpu(transformer_output),
        },
        transformer_path,
    )

    torch.accelerator.reset_peak_memory_stats()
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    generation_kwargs = {
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "height": HEIGHT,
        "width": WIDTH,
        "num_frames": NUM_FRAMES,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "generator": generator,
        "output_type": "np",
    }
    if normalized_image is not None:
        generation_kwargs["image"] = normalized_image
    if pipe.guider.guidance_scale != GUIDANCE_SCALE:
        raise RuntimeError(f"Expected the default guidance scale {GUIDANCE_SCALE}, got {pipe.guider.guidance_scale}.")

    torch.accelerator.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = pipe(**generation_kwargs, output=["videos", "latents"])
    torch.accelerator.synchronize()
    generation_seconds = time.perf_counter() - started
    frames = outputs["videos"][0]

    video_path = output_dir / "pipeline.mp4"
    export_to_video(frames, video_path, fps=FPS)
    reference_path = output_dir / "pipeline_reference.safetensors"
    save_file(
        {
            "final_latents": _cpu(outputs["latents"]),
            "decoded_frames_uint8": _frames_to_uint8(frames),
        },
        reference_path,
    )

    scheduler_config = dict(pipe.scheduler.config)
    input_metadata = None
    if input_path is not None:
        input_metadata = {
            "filename": input_path.name,
            "sha256": _sha256(input_path),
            "size": input_path.stat().st_size,
            "format": "png-rgb",
        }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "publishable": publishable,
        "reference_implementation": {
            "library": "diffusers",
            "version": __import__("diffusers").__version__,
            "pipeline_class": type(pipe).__name__,
            "torch_version": torch.__version__,
        },
        "model_provenance": model_provenance,
        "generator_provenance": repository_provenance,
        "scheduler": {
            "class": type(pipe.scheduler).__name__,
            "config_sha256": _json_sha256(scheduler_config),
            "config": scheduler_config,
        },
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "seed": SEED,
        "generator_device": "cuda",
        "dtype": "bfloat16",
        "height": HEIGHT,
        "width": WIDTH,
        "num_frames": NUM_FRAMES,
        "fps": FPS,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "input_image": input_metadata,
        "generation_seconds": generation_seconds,
        "peak_reserved_gib": torch.accelerator.max_memory_reserved() / 1024**3,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n")

    artifact_paths = [transformer_path, video_path, reference_path, metadata_path]
    if input_path is not None:
        artifact_paths.append(input_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "publishable": publishable,
        "model_provenance": model_provenance,
        "metadata_sha256": _sha256(metadata_path),
        "files": {path.name: {"sha256": _sha256(path), "size": path.stat().st_size} for path in artifact_paths},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("t2v", "i2v"), required=True)
    parser.add_argument(
        "--model",
        help=(
            "Optional local checkpoint. Local output is fingerprinted and marked development-only; "
            "it cannot be published as the canonical golden."
        ),
    )
    parser.add_argument("--input-image", type=Path, help="Fixed local RGB input for task=i2v.")
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/pan2-goldens-v1"))
    args = parser.parse_args()
    print(generate(args.task, args.model, args.input_image, args.output_root))


if __name__ == "__main__":
    main()
