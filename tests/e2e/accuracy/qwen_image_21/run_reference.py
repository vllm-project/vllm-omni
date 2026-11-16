# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Run only in the isolated, pinned Diffusers environment; never imports vLLM."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import sys
from contextlib import ExitStack
from importlib.metadata import distribution, version
from pathlib import Path

from contract import (
    EXECUTION_SETTINGS,
    REFERENCE_REVISION,
    REFERENCE_TORCHAO_VERSION,
    device_preflight,
    sha256,
    validate_model,
    write_json,
)


def preflight(request: dict) -> dict:
    device = device_preflight(request["device_type"])
    for key, value in EXECUTION_SETTINGS[request["device_type"]].items():
        assert request[key] == value, f"Mismatched {key}: {request[key]} != {value}"
    import diffusers
    import huggingface_hub
    import torch

    assert version("torchao") == REFERENCE_TORCHAO_VERSION, "Run setup_reference.sh to install reference torchao"
    from diffusers import AutoencoderKLQwenImage21, QwenImage21Pipeline, QwenImage21Transformer2DModel
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21AttnProcessor
    from PIL import Image
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    dist = distribution("diffusers")
    source = json.loads(dist.read_text("direct_url.json") or "{}")
    assert source.get("vcs_info", {}).get("commit_id") == REFERENCE_REVISION, source
    package = Path(str(dist.locate_file("diffusers"))).resolve()
    assert Path(diffusers.__file__).resolve().parent == package, "PYTHONPATH shadows the pinned reference"
    classes = (QwenImage21Pipeline, AutoencoderKLQwenImage21, QwenImage21Transformer2DModel, QwenImage21AttnProcessor)
    for cls in classes:
        assert Path(inspect.getfile(cls)).resolve().is_relative_to(package), inspect.getfile(cls)
    assert "use_kv_cache" in inspect.signature(QwenImage21Pipeline.__call__).parameters
    assert "dummy_" not in Qwen3VLForConditionalGeneration.__module__
    model = Path(request["model"])
    assets = validate_model(model)
    processor = Qwen3VLProcessor.from_pretrained(str(model), subfolder="processor", local_files_only=True)
    assert processor.tokenizer is not None and processor.image_processor is not None
    del processor
    for entry in request["inputs"]:
        path = Path(entry["path"])
        assert sha256(path) == entry["sha256"], path
        with Image.open(path) as image:
            image.load()
            assert list(image.size) == entry["size"] and image.mode == entry["mode"]
    packages = ["diffusers", "torchao", "torch", "transformers", "accelerate", "Pillow", "huggingface-hub"]
    if request["device_type"] == "npu":
        packages.append("torch-npu")
    return {
        "python": sys.executable,
        "reference_revision": REFERENCE_REVISION,
        "pipeline_path": inspect.getfile(QwenImage21Pipeline),
        "versions": {name: version(name) for name in packages},
        "huggingface_hub_path": str(Path(huggingface_hub.__file__).resolve()),
        "cuda": torch.version.cuda,
        **device,
        "assets": assets,
    }


def generate(request: dict, output: Path, repeats: int) -> None:
    import torch
    from diffusers import QwenImage21Pipeline
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21AttnProcessor
    from PIL import Image
    from torch.nn.attention import SDPBackend, sdpa_kernel

    images = []
    pipe = None
    adapters = ExitStack()
    is_cuda = request["device_type"] == "cuda"
    is_rocm = request["device_type"] == "rocm"
    torch_device = "cuda" if is_cuda or is_rocm else request["device_type"]
    cudnn_sdp_enabled = torch.backends.cuda.cudnn_sdp_enabled() if is_cuda else None
    try:
        if is_cuda:
            # vLLM's CUDA platform disables cuDNN SDPA globally. Match that policy
            # in this isolated reference worker; the DiT override below is separate.
            torch.backends.cuda.enable_cudnn_sdp(False)
        elif not is_rocm:
            # NPUOmniPlatform.set_device enables internal formats.
            # torch_npu exposes a write-only setter. This worker exits after
            # generation, so the setting cannot leak into another comparison.
            torch.npu.config.allow_internal_format = True
        for entry in request["inputs"]:
            with Image.open(entry["path"]) as image:
                image.load()
                images.append(image.copy())
        pipe = QwenImage21Pipeline.from_pretrained(
            request["model"],
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to(torch_device)
        # Native constructs the encoder under the loader's accelerator context.
        # Recreate only non-checkpoint RoPE buffers using the reference's own
        # constructors on that device, without copying any native tensors.
        vision = pipe.text_encoder.model.visual.rotary_pos_emb
        text = pipe.text_encoder.model.language_model.rotary_emb
        device = pipe.text_encoder.device
        with torch.device(device):
            if hasattr(vision, "config"):
                fresh_vision = type(vision)(vision.config)
            else:
                fresh_vision = type(vision)(vision.dim, vision.theta)
            fresh_text = type(text)(text.config, device=device)
        for target, source in ((vision, fresh_vision), (text, fresh_text)):
            for name, value in source.named_buffers():
                buffer = target.get_buffer(name)
                assert buffer.dtype == value.dtype == torch.float32
                buffer.copy_(value)
        pipe.transformer.set_attn_processor(QwenImage21AttnProcessor())
        pipe.transformer.set_attention_backend("native")
        if not is_cuda and not is_rocm:
            from npu_reference import real_rope_on_npu

            adapters.enter_context(real_rope_on_npu(pipe.transformer))
        original_forward = pipe.transformer.forward

        def cudnn_forward(*args, **kwargs):
            # Prefill uses backend=None; constrain both phases at the SDPA dispatcher.
            # Encoder and VAE retain the cuDNN-disabled policy outside this scope.
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                return original_forward(*args, **kwargs)

        if is_cuda:
            pipe.transformer.forward = cudnn_forward
        metadata = {
            "comparison_kind": "aligned_reference_parity",
            "controlled_adaptations": {
                "encoder_rope": (
                    f"Rebuild non-checkpoint FP32 RoPE buffers on {device} with upstream constructors "
                    "to match native default-loader initialization, without copying native tensors."
                ),
                "transformer_attention": (
                    "Use upstream QwenImage21AttnProcessor with the native SDPA backend "
                    "to compare against the native SDPA execution path."
                ),
                "rng": (
                    f"Use a seeded {request['generator_device']} generator to match the native request's RNG device."
                ),
            },
            "component_dtypes": {
                name: str(getattr(pipe, name).dtype) for name in ("transformer", "text_encoder", "vae")
            },
            "text_attention": pipe.text_encoder.config._attn_implementation,
            "encoder_rope_initialization": str(device),
            "transformer_attention": type(pipe.transformer.transformer_blocks[0].attn.processor).__name__,
            "transformer_sdpa": (
                "CUDNN_ATTENTION (prefill and decode)"
                if is_cuda
                else "PyTorch SDPA"
                if is_rocm
                else "PyTorch NPU SDPA dispatcher"
            ),
            "scheduler": dict(pipe.scheduler.config),
            "use_kv_cache": True,
            "request": request,
        }
        if is_rocm:
            metadata["controlled_adaptations"]["transformer_sdpa"] = (
                "Keep upstream native SDPA. ROCm has no cuDNN attention backend."
            )
        elif is_cuda:
            metadata["controlled_adaptations"].update(
                {
                    "sdpa_outside_transformer": (
                        "Disable cuDNN SDPA eligibility to match vLLM's CUDA platform policy for the encoder and VAE."
                    ),
                    "transformer_sdpa": (
                        "Constrain prefill and decode to CUDNN_ATTENTION to match the requested native CUDNN_ATTN backend."
                    ),
                }
            )
            metadata["sdpa_enabled_outside_transformer"] = {
                "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
                "flash": torch.backends.cuda.flash_sdp_enabled(),
                "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "math": torch.backends.cuda.math_sdp_enabled(),
            }
            metadata["cudnn"] = torch.backends.cudnn.version()
        else:
            metadata["controlled_adaptations"].update(
                {
                    "internal_formats": "Enable NPU internal formats to match NPUOmniPlatform.set_device.",
                    "transformer_rope": (
                        "Keep upstream position/frequency lookup on CPU and use upstream real-valued rotation on NPU "
                        "to avoid unsupported complex operations; preserve upstream attention segmentation."
                    ),
                }
            )
            metadata["known_differences"] = {
                "transformer_prefill": (
                    "Reference retains segmented NPU SDPA; native TORCH_SDPA uses dense block-causal prefill. "
                    "See thresholds.json for IT2I observations and calibration limits."
                ),
            }
            metadata["npu_allow_internal_format"] = True
            metadata["transformer_rope"] = "upstream CPU position lookup + upstream real-valued rotation on NPU"
        write_json(output / "effective_reference.json", metadata)
        assert all(dtype == "torch.bfloat16" for dtype in metadata["component_dtypes"].values()), metadata
        for repeat in range(repeats):
            result = pipe(
                prompt=request["prompt"],
                image=images or None,
                negative_prompt=request["negative_prompt"],
                true_cfg_scale=request["true_cfg_scale"],
                height=request["height"],
                width=request["width"],
                num_inference_steps=request["num_inference_steps"],
                num_images_per_prompt=1,
                generator=torch.Generator(device=request["generator_device"]).manual_seed(request["seed"]),
                output_type="pil",
                output_resolution=1024,
                use_kv_cache=True,
            )
            assert isinstance(result.images, list) and len(result.images) == 1
            image = result.images[0]
            assert isinstance(image, Image.Image)
            image.save(output / f"reference_{repeat}.png")
            assert image.mode == "RGBA", f"Expected four VAE channels, got {image.mode}"
            assert image.size == (request["width"], request["height"]), image.size
            write_json(output / f"schedule_{repeat}.json", {"timesteps": pipe.scheduler.timesteps.tolist()})
    finally:
        adapters.close()
        if is_cuda:
            torch.backends.cuda.enable_cudnn_sdp(cudnn_sdp_enabled)
        if pipe is not None:
            pipe.maybe_free_model_hooks()
        del pipe
        for image in images:
            image.close()
        gc.collect()
        getattr(torch, torch_device).empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    assert args.repeats >= 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    provenance = preflight(request)
    write_json(args.output_dir / "reference_preflight.json", provenance)
    if not args.preflight_only:
        generate(request, args.output_dir, args.repeats)


if __name__ == "__main__":
    main()
