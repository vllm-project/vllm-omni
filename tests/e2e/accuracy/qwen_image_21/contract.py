# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Small, model-specific request and provenance contract shared with the worker."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen-Image-2.1"
REFERENCE_REVISION = "6256aa7666cedd47443adc8f82da9a10e110b09c"
REFERENCE_TORCHAO_VERSION = "0.16.0"
ROOT = Path(__file__).resolve().parents[4]
EXECUTION_SETTINGS = {
    "cuda": {"generator_device": "cuda", "attention_backend": "CUDNN_ATTN"},
    # Match CPU RNG on NPU; preserve native dense and reference segmented prefill.
    # See thresholds.json for the resulting IT2I calibration scope.
    "npu": {"generator_device": "cpu", "attention_backend": "TORCH_SDPA"},
    # ROCm reports the CUDA device namespace, but cuDNN attention is not available.
    "rocm": {"generator_device": "cuda", "attention_backend": "TORCH_SDPA"},
}
INPUTS = (
    ("qwen_image_edit_2511_test1.png", "ae23dbfe1723f0cae27a4152fa21feea83180437c079dd32d142112c18095b1d"),
    ("qwen_image_edit_2511_test2.png", "b13355115f5e5da9e3d146c4ad942eeba4e1c35005f786f44c0ea07f0298d024"),
)
CASES: dict[str, dict[str, Any]] = {
    "it2i": {
        "prompt": (
            "Replace the cat's head inside the green circle in the first image with a 3D cartoon version "
            "of the dog in the second image. Preserve the dog's fur colors and ears, the pink costume "
            "and hat from the first image. Remove the green circle. Use a transparent background."
        ),
        "negative_prompt": "blurry, distorted, extra heads, green circle",
        "true_cfg_scale": 4.0,
        "width": 768,
        "height": 1024,
    },
    "t2i": {
        "prompt": "A ceramic teapot on a wooden table, warm natural window light, detailed photograph.",
        "negative_prompt": None,
        "true_cfg_scale": 1.0,
        "width": 1024,
        "height": 1024,
    },
}


def write_json(path: Path, value: object) -> None:
    """Stdlib-only writer for the isolated reference worker (no serving imports)."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def device_preflight(device_type: str) -> dict:
    """Check the requested accelerator in either worker without importing vLLM."""
    import torch

    assert device_type in EXECUTION_SETTINGS, f"Unsupported accuracy device: {device_type}"
    if device_type == "npu":
        import torch_npu  # noqa: F401 -- register the NPU device in the independent reference

    torch_device = "cuda" if device_type == "rocm" else device_type
    accelerator = getattr(torch, torch_device)
    assert accelerator.is_available(), f"{device_type} is required; accuracy preflight must not skip"
    assert accelerator.is_bf16_supported(), f"BF16 {device_type} hardware is required"
    properties = accelerator.get_device_properties(0)
    memory = properties.total_memory
    device_name = accelerator.get_device_name(0)
    device_arch = str(getattr(properties, "gcnArchName", "")) if device_type == "rocm" else ""
    # Some ROCm CI images return an empty marketing name. Preserve a stable
    # architecture identity so calibration cannot silently fall back to CUDA's
    # default thresholds.
    if not device_name and device_arch:
        device_name = device_arch
    if device_type in {"cuda", "rocm"}:
        assert memory >= 70 * 1024**3, "Use one >=80 GB GPU"
    result = {"device_type": device_type, "device": device_name, "memory_bytes": memory}
    if device_arch:
        result["device_arch"] = device_arch
    return result


def validate_model(model: Path) -> dict:
    """Check configs, tokenizers and every indexed shard before allocating weights."""
    files = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "transformer/config.json",
        "text_encoder/config.json",
        "vae/config.json",
        "processor/tokenizer.json",
        "processor/tokenizer_config.json",
        "processor/preprocessor_config.json",
        "processor/chat_template.jinja",
        "vae/diffusion_pytorch_model.safetensors",
    ]
    for name in (
        "text_encoder/model.safetensors.index.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
    ):
        index = model / name
        weights = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        assert weights, f"Empty weight index: {index}"
        files.append(name)
        files.extend(str(Path(name).parent / shard) for shard in sorted(set(weights.values())))
    result: dict[str, dict[str, int | str]] = {}
    for name in files:
        path = model / name
        assert path.is_file() and path.stat().st_size > 0, f"Missing or empty asset: {path}"
        result[name] = {"bytes": path.stat().st_size}
        if path.suffix != ".safetensors":
            result[name]["sha256"] = sha256(path)
    model_index = json.loads((model / "model_index.json").read_text(encoding="utf-8"))
    assert model_index["_class_name"] == "QwenImage21Pipeline", model_index
    config = json.loads((model / "transformer/config.json").read_text(encoding="utf-8"))
    assert config["causal_condition"] is True, "This contract requires prefix KV caching on both sides"
    return {"path": str(model), "files": result}
