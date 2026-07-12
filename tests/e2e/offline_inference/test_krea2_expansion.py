# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""L4 functionality tests for the Krea 2 text-to-image diffusion pipeline (``Krea2Pipeline``).

Runs against the public few-step distilled checkpoint ``krea/Krea-2-Turbo`` by default (override with the
``KREA2_MODEL`` environment variable, e.g. ``krea/Krea-2-Raw`` for the Raw checkpoint or a local diffusers
directory). They cover a basic functional smoke plus the layerwise-CPU-offload path that the pipeline declares
via ``SupportsComponentDiscovery`` / ``_layerwise_offload_blocks_attrs``.
"""

import os

import numpy as np
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunnerHandler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest
from vllm_omni.lora.utils import stable_lora_int_id

MODEL = os.environ.get("KREA2_MODEL", "krea/Krea-2-Turbo")
# vLLM-Omni-compatible PEFT repackaging of krea/Krea-2-LoRA-darkbrush (264 modules, r=alpha=32).
LORA = os.environ.get("KREA2_LORA", "NagaSaiAbhinay/Krea-2-vllm-darkbrush-LoRA")
PROMPT = "a fox in the snow, photorealistic"
# darkbrush is a trigger-word LoRA: the trained monochrome ink-wash style only manifests when
# its trigger phrase is present, so the LoRA test prompt appends it for a strong, stable signal.
LORA_TRIGGER = "monochrome ink wash style"
LORA_PROMPT = f"a fox in the snow, {LORA_TRIGGER}"

pytestmark = [
    pytest.mark.diffusion,
    pytest.mark.full_model,
]


def _sampling() -> OmniDiffusionSamplingParams:
    # Small resolution + few steps to keep the L4 case light. guidance_scale resolution is checkpoint-aware inside
    # the pipeline (distilled -> no-CFG, Raw -> CFG), so this stays agnostic to which checkpoint KREA2_MODEL points at.
    return OmniDiffusionSamplingParams(
        height=512,
        width=512,
        num_inference_steps=8,
        guidance_scale=0.0,
        seed=42,
    )


@hardware_test(res={"cuda": "H100"})
@pytest.mark.parametrize("omni_runner", [(MODEL, None)], indirect=True)
def test_krea2_text_to_image_001(omni_runner_handler: OmniRunnerHandler) -> None:
    """Basic functional smoke: a single prompt produces a decoded image."""
    omni_runner_handler.send_diffusion_request({"model": MODEL, "prompt": PROMPT, "sampling_params": _sampling()})


@hardware_test(res={"cuda": "H100"})
@pytest.mark.parametrize(
    "omni_runner",
    [(MODEL, None, {"enable_layerwise_offload": True})],
    indirect=True,
)
def test_krea2_layerwise_offload(omni_runner_handler: OmniRunnerHandler) -> None:
    """Exercise layerwise CPU offload on the DiT (SupportsComponentDiscovery + _layerwise_offload_blocks_attrs)."""
    omni_runner_handler.send_diffusion_request({"model": MODEL, "prompt": PROMPT, "sampling_params": _sampling()})


@hardware_test(res={"cuda": "H100"})
@pytest.mark.cache
@pytest.mark.parametrize("omni_runner", [(MODEL, None, {"cache_backend": "cache_dit"})], indirect=True)
def test_krea2_cache_dit(omni_runner_handler: OmniRunnerHandler) -> None:
    """Exercise Cache-DiT on Krea 2 via the custom Krea2Pipeline enabler (``enable_cache_for_krea2``).

    Validates the docs' Cache-DiT support claim for Krea 2. ``has_separate_cfg`` is checkpoint-aware
    (False for the distilled Turbo no-CFG path, True for the Raw CFG path); the default Turbo checkpoint
    exercises the no-CFG branch.
    """
    omni_runner_handler.send_diffusion_request({"model": MODEL, "prompt": PROMPT, "sampling_params": _sampling()})


@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize(
    "omni_runner",
    # Standalone HSDP: use_hsdp=True with an explicit hsdp_shard_size shards the DiT weights
    # across 2 ranks (replicate_size defaults to 1 -> world_size = 1 * 2 = 2). HSDP must keep
    # tensor_parallel_size/data_parallel_size at 1 (see vllm_omni/diffusion/data.py __post_init__).
    # Also validated locally at 8 GPUs (hsdp_shard_size=8); CI shard provisions 2.
    [(MODEL, None, {"use_hsdp": True, "hsdp_shard_size": 2})],
    indirect=True,
)
def test_krea2_hsdp(omni_runner_handler: OmniRunnerHandler) -> None:
    """Exercise Hybrid Sharded Data Parallel (HSDP) weight sharding on the DiT across 2 GPUs.

    Validates the docs' HSDP-support claim for Krea 2. Scheduled on the multi-GPU (distributed_cuda)
    shard by ``num_cards=2``; skipped automatically on boxes with fewer than 2 CUDA devices.
    """
    omni_runner_handler.send_diffusion_request({"model": MODEL, "prompt": PROMPT, "sampling_params": _sampling()})


@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize(
    "omni_runner",
    # VAE patch-parallel decode: partition the VAE decode across the TP world. Mirrors the online
    # "tp_vae_patch" case (tensor_parallel_size=2 + vae_patch_parallel_size=2) from the wan2.2/z-image
    # expansion suites, so vae_patch_parallel_size must equal the world size established by TP.
    # Also validated locally at 8 GPUs (tp=8, vae_patch=8); CI shard provisions 2.
    [(MODEL, None, {"tensor_parallel_size": 2, "vae_patch_parallel_size": 2})],
    indirect=True,
)
def test_krea2_vae_patch_parallel(omni_runner_handler: OmniRunnerHandler) -> None:
    """Exercise VAE patch-parallel (decode) across 2 GPUs.

    Validates the docs' VAE-patch-parallel (decode) support claim for Krea 2. Scheduled on the
    multi-GPU (distributed_cuda) shard by ``num_cards=2``; skipped automatically on boxes with
    fewer than 2 CUDA devices.
    """
    omni_runner_handler.send_diffusion_request({"model": MODEL, "prompt": PROMPT, "sampling_params": _sampling()})


def _generate(handler: OmniRunnerHandler, lora_request: LoRARequest | None, lora_scale: float = 1.0) -> np.ndarray:
    sp = _sampling()
    sp.lora_request = lora_request
    sp.lora_scale = lora_scale
    resp = handler.send_diffusion_request({"model": MODEL, "prompt": LORA_PROMPT, "sampling_params": sp})
    return np.asarray(resp.images[0].convert("RGB"), dtype=np.int16)


@hardware_test(res={"cuda": "H100"})
@pytest.mark.parametrize("omni_runner", [(MODEL, None)], indirect=True)
def test_krea2_lora(omni_runner_handler: OmniRunnerHandler) -> None:
    """Validate diffusion LoRA on Krea 2: visible effect, scale sensitivity, clean deactivation.

    Uses ``LORA`` (a vLLM-Omni-compatible PEFT repackaging of ``krea/Krea-2-LoRA-darkbrush``,
    264 modules matched via the ReplicatedLinear projections) passed per-request through
    ``OmniDiffusionSamplingParams.lora_request`` / ``lora_scale`` — the same fields the
    diffusion LoRA manager reads to activate/deactivate the adapter.
    """
    lora_request = LoRARequest(lora_name="darkbrush", lora_int_id=stable_lora_int_id(LORA), lora_path=LORA)

    baseline = _generate(omni_runner_handler, None)
    img_1x = _generate(omni_runner_handler, lora_request, lora_scale=1.0)
    img_2x = _generate(omni_runner_handler, lora_request, lora_scale=2.0)
    restored = _generate(omni_runner_handler, None)

    diff_1x = np.abs(baseline - img_1x).mean()
    diff_2x = np.abs(baseline - img_2x).mean()
    diff_restored = np.abs(baseline - restored).mean()

    # (a) Adapter has a visible effect at both scales.
    assert diff_1x > 0.5, f"LoRA scale=1.0 had no visible effect: diff={diff_1x}"
    assert diff_2x > 0.5, f"LoRA scale=2.0 had no visible effect: diff={diff_2x}"
    # (b) Scale changes the output (stronger adapter -> larger perturbation).
    assert not np.isclose(diff_1x, diff_2x, atol=1.0), f"LoRA scale had no effect: 1x={diff_1x:.2f}, 2x={diff_2x:.2f}"
    # (c) Passing lora_request=None cleanly deactivates: same seed -> back to baseline (modulo fp drift).
    assert diff_restored < 5.0, f"LoRA did not deactivate cleanly: diff_restored={diff_restored:.2f}"
