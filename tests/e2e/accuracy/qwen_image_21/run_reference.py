# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Isolated, pinned Diffusers worker with import/asset preflight and repeatability."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    root = args.reference_root.resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    assert revision == request["reference_revision"], revision

    # These imports must succeed with the worker's real interpreter and PYTHONPATH.
    import diffusers
    import torch
    from diffusers import QwenImage21Pipeline
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21AttnProcessor
    from safetensors import safe_open
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    pipeline_source = Path(inspect.getfile(QwenImage21Pipeline)).resolve()
    assert pipeline_source.is_relative_to(root / "src"), pipeline_source
    assert Path(diffusers.__file__).resolve().is_relative_to(root / "src"), diffusers.__file__
    assert "dummy_" not in QwenImage21Pipeline.__module__, QwenImage21Pipeline
    assert torch.cuda.is_available(), "Reference requires CUDA"
    model = Path(request["model"])
    index = json.loads((model / "model_index.json").read_text())
    assert index["_class_name"] == "QwenImage21Pipeline", index
    shards = {}
    for component in ("transformer", "text_encoder", "vae"):
        component_root = model / component
        assert json.loads((component_root / "config.json").read_text()), component
        indexes = list(component_root.glob("*.safetensors.index.json"))
        if indexes:
            names = {name for path in indexes for name in json.loads(path.read_text())["weight_map"].values()}
            files = [component_root / name for name in sorted(names)]
        else:
            files = sorted(component_root.glob("*.safetensors"))
        assert files, f"No weights for {component}"
        for path in files:
            assert path.is_file() and path.stat().st_size > 0, f"Missing/empty weight: {path}"
            with safe_open(str(path), framework="pt", device="cpu") as weights:
                assert list(weights.keys()), f"Empty safetensors header: {path}"
            shards[str(path.relative_to(model))] = path.stat().st_size
    assert json.loads((model / "scheduler/scheduler_config.json").read_text())
    # Validate tokenizer/processor loading without loading model weights.
    processor = Qwen3VLProcessor.from_pretrained(model / "processor", local_files_only=True)
    assert isinstance(processor, Qwen3VLProcessor)
    assert Qwen3VLForConditionalGeneration is not None
    # Download-manager metadata under .cache is not a model input and may have
    # different ownership. Hash only the actual components consumed by both paths.
    config_files = [model / "model_index.json"] + [
        path
        for component in ("transformer", "text_encoder", "vae", "processor", "scheduler")
        for path in (model / component).glob("*.json")
    ]
    metadata = {
        "reference_revision": revision,
        "python": sys.executable,
        "diffusers_source": diffusers.__file__,
        "diffusers_source_version": diffusers.__version__,
        "pipeline_source": str(pipeline_source),
        "packages": {name: version(name) for name in ("torch", "diffusers", "transformers", "accelerate", "torchao")},
        "device": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "weight_files_bytes": shards,
        "config_sha256": {
            str(path.relative_to(model)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(config_files)
        },
    }
    (args.output_dir / "preflight.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    if args.preflight:
        return

    pipe = QwenImage21Pipeline.from_pretrained(model, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda")
    try:
        pipe.transformer.set_attn_processor(QwenImage21AttnProcessor())
        pipe.transformer.set_attention_backend("native")
        original_forward = pipe.transformer.forward

        def cudnn_forward(*forward_args, **forward_kwargs):
            # Upstream prefill passes backend=None even after set_attention_backend.
            # Restrict only DiT attention; preserve the text encoder/VAE numerical paths.
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                return original_forward(*forward_args, **forward_kwargs)

        pipe.transformer.forward = cudnn_forward
        generation = {
            key: value for key, value in request["generation"].items() if key not in ("seed", "generator_device")
        }
        for filename in ("reference.png", "reference_repeat.png"):
            result = pipe(
                **generation,
                use_kv_cache=True,
                generator=torch.Generator(device=request["generation"]["generator_device"]).manual_seed(
                    request["generation"]["seed"]
                ),
            )
            assert len(result.images) == 1 and result.images[0].mode == "RGBA", result.images
            result.images[0].save(args.output_dir / filename)
        metadata.update(
            {
                "component_dtypes": {
                    name: str(getattr(pipe, name).dtype) for name in ("transformer", "text_encoder", "vae")
                },
                "attention_processor": "QwenImage21AttnProcessor",
                "transformer_sdpa": "CUDNN_ATTENTION",
                "text_encoder_attention": pipe.text_encoder.config._attn_implementation,
                "use_kv_cache": True,
                "scheduler_class": type(pipe.scheduler).__name__,
                "scheduler_config": dict(pipe.scheduler.config),
                "timesteps": pipe.scheduler.timesteps.cpu().tolist(),
                "sigmas": pipe.scheduler.sigmas.cpu().tolist(),
            }
        )
        (args.output_dir / "reference_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    finally:
        pipe.maybe_free_model_hooks()
        del pipe
        # Process exit releases all reference weights before the native server starts.


if __name__ == "__main__":
    main()
