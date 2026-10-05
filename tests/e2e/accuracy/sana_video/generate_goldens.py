# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Generate frozen SANA-Video transformer and pipeline reference artifacts."""

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
from diffusers import SanaImageToVideoPipeline, SanaVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image
from safetensors.torch import save_file

PROMPT = "A cat walking on the grass, facing the camera. motion score: 30."
NEGATIVE_PROMPT = (
    "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden disappearance, jump cuts, "
    "jerky movements, rapid shot changes, frames out of sync, inconsistent character shapes, temporal artifacts, "
    "jitter, and ghosting effects, creating a disorienting visual experience."
)
SCHEMA_VERSION = 2
SEED = 42
NUM_FRAMES = 81
FPS = 16
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 6.0
VARIANTS = {
    "480p": {
        "model": "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
        "revision": "fed3bce411c58a0f688a31afe8f52e61acc2b15f",
        "height": 480,
        "width": 832,
        "vae_dtype": torch.float32,
        "transformer_case_shape": (1, 16, 3, 8, 8),
    },
    "720p": {
        "model": "Efficient-Large-Model/SANA-Video_2B_720p_diffusers",
        "revision": "8bda5e623d0f48cd6da3b387b10ca35d15cf1c4e",
        "height": 704,
        "width": 1280,
        "vae_dtype": torch.bfloat16,
        "transformer_case_shape": (1, 128, 2, 4, 4),
    },
}


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
        entry = f"{relative}\0{size}\0{file_sha256}\n".encode()
        digest.update(entry)
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


def _model_provenance(
    config: dict[str, Any],
    model: str | None,
    local_source_model: str | None,
    local_source_revision: str | None,
) -> tuple[dict[str, Any], bool]:
    if model is None:
        return (
            {
                "kind": "huggingface_hub",
                "model_id": config["model"],
                "revision": config["revision"],
            },
            True,
        )

    local_path = Path(model).expanduser().resolve()
    if bool(local_source_model) != bool(local_source_revision):
        raise ValueError("--local-source-model and --local-source-revision must be specified together.")
    provenance = {
        "kind": "local_checkpoint",
        "checkpoint_fingerprint": _directory_fingerprint(local_path),
        "claimed_source_model": local_source_model,
        "claimed_source_revision": local_source_revision,
        "verification": "unverified-local-copy",
    }
    # A directory hash proves which local bytes were used, but not that those
    # bytes match a claimed Hub commit. Such output is useful for development
    # comparisons but must never be uploaded as the canonical frozen reference.
    return provenance, False


def generate(
    variant: str,
    task: str,
    model: str | None,
    input_image: Path | None,
    output_root: Path,
    local_source_model: str | None = None,
    local_source_revision: str | None = None,
) -> Path:
    config = dict(VARIANTS[variant])
    model_provenance, publishable = _model_provenance(
        config,
        model,
        local_source_model,
        local_source_revision,
    )
    repository_provenance = _repository_provenance()
    if publishable and repository_provenance["dirty"]:
        raise RuntimeError("Refusing to generate publishable golden assets from a dirty vllm-omni worktree.")
    if model is not None:
        config["model"] = str(Path(model).expanduser().resolve())
        config["revision"] = None

    output_dir = output_root / variant / task
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

    pipeline_class = SanaImageToVideoPipeline if task == "i2v" else SanaVideoPipeline
    pipe = pipeline_class.from_pretrained(
        config["model"],
        revision=config["revision"],
        torch_dtype=torch.bfloat16,
    )
    pipe.text_encoder.to(torch.bfloat16)
    pipe.vae.to(config["vae_dtype"])
    pipe.to("cuda")

    torch.manual_seed(1234)
    hidden_states = torch.randn(config["transformer_case_shape"], dtype=torch.bfloat16, device="cuda")
    encoder_hidden_states = torch.randn(1, 8, 2304, dtype=torch.bfloat16, device="cuda")
    encoder_attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]], device="cuda")
    timestep = torch.tensor([500.0], device="cuda")
    with torch.inference_mode():
        transformer_output = pipe.transformer(
            hidden_states,
            encoder_hidden_states,
            timestep,
            encoder_attention_mask=encoder_attention_mask,
        ).sample

    transformer_path = output_dir / "transformer_case.safetensors"
    save_file(
        {
            "hidden_states": _cpu(hidden_states),
            "encoder_hidden_states": _cpu(encoder_hidden_states),
            "encoder_attention_mask": _cpu(encoder_attention_mask),
            "timestep": _cpu(timestep),
            "output": _cpu(transformer_output),
        },
        transformer_path,
    )

    final_latents: dict[str, torch.Tensor] = {}

    def capture_final_latents(_pipe, step, _timestep, callback_kwargs):
        if step == NUM_INFERENCE_STEPS - 1:
            final_latents["latents"] = _cpu(callback_kwargs["latents"])
        return callback_kwargs

    torch.accelerator.reset_peak_memory_stats()
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    generation_kwargs = {
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "height": config["height"],
        "width": config["width"],
        "frames": NUM_FRAMES,
        "guidance_scale": GUIDANCE_SCALE,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "generator": generator,
        "output_type": "np",
        "callback_on_step_end": capture_final_latents,
        "callback_on_step_end_tensor_inputs": ["latents"],
    }
    if normalized_image is not None:
        generation_kwargs["image"] = normalized_image

    torch.accelerator.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        frames = pipe(**generation_kwargs).frames[0]
    torch.accelerator.synchronize()
    generation_seconds = time.perf_counter() - started
    if "latents" not in final_latents:
        raise RuntimeError("Diffusers did not expose the final denoised latents to the generation callback.")

    video_path = output_dir / "pipeline.mp4"
    export_to_video(frames, video_path, fps=FPS)
    reference_path = output_dir / "pipeline_reference.safetensors"
    save_file(
        {
            "final_latents": final_latents["latents"],
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
        "variant": variant,
        "task": task,
        "publishable": publishable,
        "reference_implementation": {
            "library": "diffusers",
            "version": __import__("diffusers").__version__,
            "pipeline_class": pipeline_class.__name__,
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
        "vae_dtype": str(config["vae_dtype"]).removeprefix("torch."),
        "height": config["height"],
        "width": config["width"],
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
        "variant": variant,
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
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--task", choices=("t2v", "i2v"), required=True)
    parser.add_argument(
        "--model",
        help=(
            "Optional local checkpoint. Local output is fingerprinted and marked development-only; "
            "it cannot be published as the canonical golden."
        ),
    )
    parser.add_argument("--input-image", type=Path, help="Fixed local RGB input for task=i2v.")
    parser.add_argument("--local-source-model", help="Claimed upstream model ID for local checkpoint audit records.")
    parser.add_argument("--local-source-revision", help="Claimed upstream commit for local checkpoint audit records.")
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/sana-video-goldens-v2"))
    args = parser.parse_args()
    print(
        generate(
            args.variant,
            args.task,
            args.model,
            args.input_image,
            args.output_root,
            args.local_source_model,
            args.local_source_revision,
        )
    )


if __name__ == "__main__":
    main()
