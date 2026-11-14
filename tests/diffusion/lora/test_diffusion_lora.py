# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Z-Image runtime LoRA state and parity with module-level CPU offload."""

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


def _write_adapter(adapter_dir: Path, rank: int, strength: float) -> LoRARequest:
    adapter_dir.mkdir()
    # Exercise the fused QKV projection used by the actual Z-Image model.
    dim = 3840
    module_name = "transformer.layers.0.attention.to_qkv"
    lora_a = torch.zeros((rank, dim), dtype=torch.float32)
    lora_a[0, 0] = 1.0
    lora_b = torch.zeros((3 * dim, rank), dtype=torch.float32)
    # Perturb V non-uniformly so distinct adapters have observable outputs.
    lora_b[2 * dim :, 0] = torch.linspace(-strength, strength, dim)
    save_file(
        {
            f"base_model.model.{module_name}.lora_A.weight": lora_a,
            f"base_model.model.{module_name}.lora_B.weight": lora_b,
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"r": rank, "lora_alpha": rank, "target_modules": [module_name]}),
        encoding="utf-8",
    )
    return LoRARequest(
        lora_name=adapter_dir.name,
        lora_int_id=stable_lora_int_id(str(adapter_dir)),
        lora_path=str(adapter_dir),
    )


def _run_sequence(
    adapter_a: LoRARequest,
    adapter_b: LoRARequest,
    *,
    offload: bool,
) -> dict[str, NDArray[np.uint8]]:
    offload_config = {"mode": "module", "components": ["dit", "text_encoder"]} if offload else None
    images: dict[str, NDArray[np.uint8]] = {}
    timings: dict[str, list[float]] = {}
    sequence = (
        ("base", None, 1.0),
        ("a", adapter_a, 1.0),
        ("b", adapter_b, 1.0),
        ("a_again", adapter_a, 1.0),
        ("zero_scale", adapter_a, 0.0),
        ("a_after_zero", adapter_a, 1.0),
        ("base_again", None, 1.0),
        ("a_after_base", adapter_a, 1.0),
    )
    with OmniRunner(
        get_model_prefix() + MODEL,
        enforce_eager=True,
        dtype="bfloat16",
        diffusion_attention_config={"default": {"backend": "TORCH_SDPA"}},
        diffusion_offload_config=offload_config,
        lora_backend="peft",
        lora_application="runtime",
        lora_path=adapter_a.lora_path if offload else None,
        max_cpu_loras=2,
    ) as runner:
        # The reference starts with no adapters, so its first base output
        # also checks that subsequent LoRA wrapping preserves the base model.
        # The offloaded run preloads A while the DiT starts on CPU. B is
        # loaded on demand with a larger rank to exercise reallocation.
        monitor = DeviceMemoryMonitor(current_omni_platform.current_device(), interval=0.02)
        try:
            # Validate the cold sequence too, but measure only after every
            # adapter/rank has been exercised once.
            for repetition in range(4):
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
                        np.testing.assert_array_equal(pixels, images[name], err_msg=f"Repeated {name}, {offload=}")
                    images[name] = pixels
        finally:
            monitor.stop()

        # Device-wide polling includes child worker allocations. Timing and
        # memory are diagnostic, not thresholds that depend on GPU/interconnect.
        print(
            json.dumps(
                {"offload": offload, "peak_device_memory_mib": monitor.peak_used_mb, "latency_seconds": timings},
                sort_keys=True,
            ),
            flush=True,
        )

    # Nonzero controls prevent a silently ignored adapter from passing parity.
    assert not np.array_equal(images["a"], images["base"]), "Adapter A had no effect"
    assert not np.array_equal(images["b"], images["base"]), "Adapter B had no effect"
    assert not np.array_equal(images["a"], images["b"]), "Different adapters produced identical images"
    for name in ("a_again", "a_after_zero", "a_after_base"):
        np.testing.assert_array_equal(images[name], images["a"], err_msg=f"Adapter A was not restored: {name}")
    for name in ("zero_scale", "base_again"):
        np.testing.assert_array_equal(images[name], images["base"], err_msg=f"Adapter state leaked into {name}")
    return images


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_zimage_lora_module_offload(tmp_path: Path) -> None:
    adapter_a = _write_adapter(tmp_path / "adapter_a", rank=8, strength=0.1)
    adapter_b = _write_adapter(tmp_path / "adapter_b", rank=16, strength=-0.1)
    baseline = _run_sequence(adapter_a, adapter_b, offload=False)
    offloaded = _run_sequence(adapter_a, adapter_b, offload=True)
    for name, image in baseline.items():
        np.testing.assert_array_equal(offloaded[name], image, err_msg=f"Offload changed the output for {name}")
