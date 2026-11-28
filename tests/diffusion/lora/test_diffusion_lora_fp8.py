# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Z-Image runtime PEFT LoRA state on top of online FP8 quantization."""

import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from safetensors.torch import save_file

from tests.helpers.mark import hardware_test
from tests.helpers.monitor import DeviceMemoryMonitor
from tests.helpers.runtime import OmniRunner, get_model_prefix
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest
from vllm_omni.lora.utils import stable_lora_int_id
from vllm_omni.platforms import current_omni_platform

MODEL = "Tongyi-MAI/Z-Image-Turbo"
PROMPT = "a photo of a cat sitting on a laptop keyboard"
DIM = 3840
FFN_DIM = 10240
ADALN_DIM = 256

# Adapter B uses the per-projection names of the Diffusers checkpoint, so the
# manager must pack Q/K/V into the FP8 ``to_qkv`` and W1/W3 into the FP8
# ``w13``. It also covers the FP8 row-parallel projections and the
# modulation linear, which Z-Image keeps unquantized.
ADAPTER_B_MODULES = {
    "attention.to_q": (DIM, DIM),
    "attention.to_k": (DIM, DIM),
    "attention.to_v": (DIM, DIM),
    "attention.to_out.0": (DIM, DIM),
    "feed_forward.w1": (DIM, FFN_DIM),
    "feed_forward.w3": (DIM, FFN_DIM),
    "feed_forward.w2": (FFN_DIM, DIM),
    "adaLN_modulation.0": (ADALN_DIM, 4 * DIM),
}


def _save_adapter(adapter_dir: Path, rank: int, weights: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> LoRARequest:
    adapter_dir.mkdir()
    tensors = {}
    for module_name, (lora_a, lora_b) in weights.items():
        tensors[f"base_model.model.{module_name}.lora_A.weight"] = lora_a
        tensors[f"base_model.model.{module_name}.lora_B.weight"] = lora_b
    save_file(tensors, str(adapter_dir / "adapter_model.safetensors"))
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"r": rank, "lora_alpha": rank, "target_modules": sorted(weights)}),
        encoding="utf-8",
    )
    return LoRARequest(
        lora_name=adapter_dir.name,
        lora_int_id=stable_lora_int_id(str(adapter_dir)),
        lora_path=str(adapter_dir),
    )


def _write_fused_qkv_adapter(adapter_dir: Path, rank: int, strength: float) -> LoRARequest:
    """Rank-``rank`` adapter on the fused QKV projection of the first block."""
    lora_a = torch.zeros((rank, DIM), dtype=torch.float32)
    lora_a[0, 0] = 1.0
    lora_b = torch.zeros((3 * DIM, rank), dtype=torch.float32)
    # Perturb V non-uniformly so the adapter has an observable output.
    lora_b[2 * DIM :, 0] = torch.linspace(-strength, strength, DIM)
    return _save_adapter(adapter_dir, rank, {"transformer.layers.0.attention.to_qkv": (lora_a, lora_b)})


def _write_split_projection_adapter(adapter_dir: Path, rank: int, strength: float) -> LoRARequest:
    """Dense random adapter on every linear of the first block."""
    generator = torch.Generator().manual_seed(0)
    weights = {}
    for suffix, (in_dim, out_dim) in ADAPTER_B_MODULES.items():
        lora_a = torch.randn((rank, in_dim), generator=generator) / in_dim**0.5
        lora_b = torch.randn((out_dim, rank), generator=generator) * strength / rank**0.5
        weights[f"transformer.layers.0.{suffix}"] = (lora_a, lora_b)
    return _save_adapter(adapter_dir, rank, weights)


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_zimage_lora_online_fp8(tmp_path: Path) -> None:
    adapter_a = _write_fused_qkv_adapter(tmp_path / "adapter_a", rank=8, strength=0.1)
    adapter_b = _write_split_projection_adapter(tmp_path / "adapter_b", rank=16, strength=0.05)
    sequence = (
        ("base", None, 1.0),
        ("a", adapter_a, 1.0),
        ("b", adapter_b, 1.0),
        ("a_again", adapter_a, 1.0),
        ("zero_scale", adapter_a, 0.0),
        ("a_after_zero", adapter_a, 1.0),
        ("base_again", None, 1.0),
        ("b_after_base", adapter_b, 1.0),
    )
    images: dict[str, NDArray[np.uint8]] = {}
    timings: dict[str, list[float]] = {}
    with OmniRunner(
        get_model_prefix() + MODEL,
        enforce_eager=True,
        dtype="bfloat16",
        quantization="fp8",
        diffusion_attention_config={"default": {"backend": "TORCH_SDPA"}},
        lora_backend="peft",
        lora_application="runtime",
        max_cpu_loras=2,
    ) as runner:
        # The first request runs before any adapter is loaded, so the later
        # base requests also check that LoRA wrapping of the FP8 layers keeps
        # the quantized base output. B is loaded with a larger rank to
        # exercise buffer reallocation.
        monitor = DeviceMemoryMonitor(current_omni_platform.current_device(), interval=0.02)
        try:
            for repetition in range(2):
                if repetition == 1:
                    current_omni_platform.reset_peak_memory_stats()
                    monitor.start()
                for name, adapter, scale in sequence:
                    started = time.perf_counter()
                    outputs = runner.omni.generate(
                        PROMPT,
                        OmniDiffusionSamplingParams(
                            height=256,
                            width=256,
                            num_inference_steps=2,
                            guidance_scale=0.0,
                            seed=42,
                            lora_request=adapter,
                            lora_scale=scale,
                        ),
                    )
                    elapsed = time.perf_counter() - started
                    if repetition > 0:
                        timings.setdefault(name, []).append(elapsed)
                    assert len(outputs) == 1
                    assert outputs[0].final_output_type == "image"
                    assert len(outputs[0].images) == 1
                    assert outputs[0].images[0].size == (256, 256)
                    pixels = np.asarray(outputs[0].images[0])
                    if name in images:
                        np.testing.assert_array_equal(pixels, images[name], err_msg=f"Repeated {name} differs")
                    images[name] = pixels
        finally:
            monitor.stop()

        # Device-wide polling includes child worker allocations. Timing and
        # memory are diagnostic, not thresholds.
        print(
            json.dumps(
                {"peak_device_memory_mib": monitor.peak_used_mb, "latency_seconds": timings},
                sort_keys=True,
            ),
            flush=True,
        )

    # Nonzero controls prevent a silently ignored adapter from passing.
    assert not np.array_equal(images["a"], images["base"]), "Adapter A had no effect"
    assert not np.array_equal(images["b"], images["base"]), "Adapter B had no effect"
    assert not np.array_equal(images["a"], images["b"]), "Different adapters produced identical images"
    for name in ("a_again", "a_after_zero"):
        np.testing.assert_array_equal(images[name], images["a"], err_msg=f"Adapter A was not restored: {name}")
    np.testing.assert_array_equal(images["b_after_base"], images["b"], err_msg="Adapter B was not restored")
    for name in ("zero_scale", "base_again"):
        np.testing.assert_array_equal(images[name], images["base"], err_msg=f"Adapter state leaked into {name}")
