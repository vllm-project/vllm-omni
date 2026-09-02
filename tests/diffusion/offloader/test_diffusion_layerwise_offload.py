# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import concurrent.futures
import gc
import multiprocessing as mp
import os
from typing import Any

import numpy as np
import pytest
import torch
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

from tests.helpers.mark import hardware_test
from tests.helpers.monitor import DeviceMemoryMonitor
from tests.helpers.runtime import OmniRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

AUDIO_MODEL: dict[str, dict[str, int | None]] = {
    # The inference peak includes backend workspaces as well as model weights.
    # On ROCm, large MIOpen workspaces and allocator fragmentation can mask the
    # resident-weight reduction, so use a conservative floor that still catches
    # a disabled/no-op layerwise offloader.
    "stabilityai/stable-audio-open-1.0": {"cuda": 1500, "rocm": 512},
}

IMAGE_VIDEO_MODELS: dict[str, dict[str, int | None]] = {
    "riverclouds/qwen_image_random": {"cuda": 4500, "rocm": None},
    # "Wan-AI/Wan2.2-T2V-A14B-Diffusers": {"cuda": 45000, "rocm": None},
}

MODELS: dict[str, dict[str, int | None]] = {**AUDIO_MODEL, **IMAGE_VIDEO_MODELS}

MODEL_MARKS = {
    "riverclouds/qwen_image_random": pytest.mark.core_model,
    "stabilityai/stable-audio-open-1.0": pytest.mark.full_model,
}

AUDIO_MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "runner_params": {},
    "sampler_params": {},
}

IMAGE_VIDEO_MODELS_PARAMS: dict[str, dict[str, Any]] = {
    "runner_params": {"boundary_ratio": 0.875, "flow_shift": 5.0},
    "sampler_params": {"height": 480, "width": 640, "num_frames": 5},
}

_OFFLOAD_STATE_PROBE = "vllm_omni_layerwise_offload_test"
# A second CI build can reverse the same exact-head measurement without a
# source edit; the default keeps the existing baseline-first workload.
_MEASUREMENT_ORDER_ENV = "VLLM_OMNI_OFFLOAD_TEST_ORDER"
_MEASUREMENT_ORDERS: dict[str, tuple[bool, bool]] = {
    "baseline-first": (False, True),
    "offload-first": (True, False),
}


class OffloadStateProbe:
    """Test-only worker extension exposing the configured offloader state."""

    model_runner: Any

    def get_offload_state_for_test(self) -> dict[str, Any]:
        backend = getattr(self.model_runner, "offload_backend", None)
        block_groups = getattr(backend, "_blocks", ())
        group_sizes = [len(group) for group in block_groups] if isinstance(block_groups, (list, tuple)) else []

        return {
            "probe": _OFFLOAD_STATE_PROBE,
            "backend_type": (
                None if backend is None else f"{backend.__class__.__module__}.{backend.__class__.__qualname__}"
            ),
            "requested": bool(getattr(self.model_runner.od_config, "enable_layerwise_offload", False)),
            "enabled": bool(backend is not None and backend.is_enabled()),
            "block_group_count": len(group_sizes),
            "block_group_sizes": group_sizes,
            "block_count": sum(group_sizes),
        }


def check_audio_determinism(audio1: np.ndarray, audio2: np.ndarray, atol: float = 1e-2) -> bool:
    if not np.allclose(audio1, audio2, atol=atol):
        diff = np.abs(audio1 - audio2)
        print(f"Max difference: {diff.max()}")
        print(f"Mean difference: {diff.mean()}")
        raise AssertionError(f"Audio outputs differ beyond tolerance atol={atol}")
    return True


def _device_used_mb(device_index: int) -> float:
    current_omni_platform.synchronize()
    with current_omni_platform.device(device_index):
        free_bytes, total_bytes = current_omni_platform.mem_get_info()
    return (total_bytes - free_bytes) / (1024**2)


def _extract_audio(output: Any) -> np.ndarray | None:
    if not output:
        return None
    multimodal_output = getattr(output[0], "multimodal_output", None)
    if not isinstance(multimodal_output, dict):
        return None
    audio = multimodal_output.get("audio")
    if audio is None:
        return None
    if isinstance(audio, torch.Tensor):
        return audio.detach().cpu().numpy()
    return np.asarray(audio)


def _collect_offload_states(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value] if value.get("probe") == _OFFLOAD_STATE_PROBE else []
    if isinstance(value, (list, tuple)):
        states: list[dict[str, Any]] = []
        for item in value:
            states.extend(_collect_offload_states(item))
        return states
    return []


def _measurement_order() -> tuple[bool, bool]:
    """Return the requested order without changing the default CI workload."""
    configured = os.environ.get(_MEASUREMENT_ORDER_ENV, "baseline-first")
    try:
        return _MEASUREMENT_ORDERS[configured]
    except KeyError as exc:
        allowed = ", ".join(_MEASUREMENT_ORDERS)
        raise ValueError(f"{_MEASUREMENT_ORDER_ENV} must be one of: {allowed}; got {configured!r}") from exc


def run_inference(
    model_name: str,
    layerwise_offload: bool = False,
    num_inference_steps: int = 3,
) -> dict[str, Any]:
    current_omni_platform.empty_cache()
    device_index = current_omni_platform.current_device()
    initial_used_mb = _device_used_mb(device_index)

    if model_name in AUDIO_MODEL:
        params = AUDIO_MODEL_PARAMS
    else:
        params = IMAGE_VIDEO_MODELS_PARAMS

    with OmniRunner(
        model_name,
        enable_layerwise_offload=layerwise_offload,
        worker_extension_cls=f"{OffloadStateProbe.__module__}.{OffloadStateProbe.__qualname__}",
        # TODO: we might want to add overlapped feature e2e tests
        # cache_backend="cache_dit",
        **params["runner_params"],
    ) as runner:
        offload_states = _collect_offload_states(
            runner.omni.engine.collective_rpc(method="get_offload_state_for_test", timeout=60)
        )
        if not offload_states:
            raise AssertionError("The offload-state worker probe returned no diffusion worker results")

        # Measure steady-state inference memory, not model construction. Enabling
        # layerwise offload first loads the model and then replaces each block's
        # device storage with CPU-backed weights.  Monitoring that transition
        # captures both the original model and temporary staging allocations,
        # which is not representative of layerwise-offloaded inference.
        monitor = DeviceMemoryMonitor(device_index=device_index, interval=0.02)
        current_omni_platform.reset_peak_memory_stats()
        monitor.start()

        try:
            # Refer to tests/e2e/offline_inference/test_wan22.py
            # Use minimal settings for testing
            output = runner.omni.generate(
                "A cat sitting on a table",
                OmniDiffusionSamplingParams(
                    generator=torch.Generator(device=current_omni_platform.device_type).manual_seed(42),
                    guidance_scale=1.0,
                    num_inference_steps=num_inference_steps,
                    **params["sampler_params"],
                ),
            )
        finally:
            monitor.stop()

        audio = _extract_audio(output)
        del output

    # DeviceMemoryMonitor reports absolute device usage. Subtract this run's
    # starting usage. Each mode runs in its own spawned process below, so model
    # objects and process-local allocator/compiler/backend workspace state cannot
    # carry into the other measurement. Shared filesystem caches can persist;
    # _MEASUREMENT_ORDER_ENV enables a reversed-order validation run for that.
    peak_used_mb = monitor.peak_used_mb
    incremental_peak_mb = max(0.0, peak_used_mb - initial_used_mb)

    del runner
    gc.collect()
    cleanup_dist_env_and_memory()
    current_omni_platform.empty_cache()
    post_cleanup_used_mb = _device_used_mb(device_index)

    return {
        "initial_used_mb": initial_used_mb,
        "peak_used_mb": peak_used_mb,
        "incremental_peak_mb": incremental_peak_mb,
        "post_cleanup_used_mb": post_cleanup_used_mb,
        "audio": audio,
        "offload_states": offload_states,
    }


def run_inference_isolated(
    model_name: str,
    layerwise_offload: bool = False,
    num_inference_steps: int = 3,
) -> dict[str, Any]:
    """Run one measurement in a fresh interpreter and accelerator context."""
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as executor:
        return executor.submit(run_inference, model_name, layerwise_offload, num_inference_steps).result()


def _assert_offload_state(measurement: dict[str, Any], *, expected_enabled: bool) -> None:
    states = measurement["offload_states"]
    assert states, "Expected at least one offload-state result"
    for state in states:
        assert state["requested"] is expected_enabled, f"Unexpected offload request state: {state}"
        assert state["enabled"] is expected_enabled, f"Unexpected offload state: {state}"
        if expected_enabled:
            assert state["backend_type"], f"Enabled offloader has no backend type: {state}"
            assert state["block_group_count"] > 0, f"Enabled offloader has no block groups: {state}"
            assert state["block_count"] > 0, f"Enabled offloader has no managed blocks: {state}"
        else:
            assert state["block_group_count"] == 0, f"Disabled offloader retained block groups: {state}"
            assert state["block_count"] == 0, f"Disabled offloader retained managed blocks: {state}"


def _print_measurement(label: str, measurement: dict[str, Any]) -> None:
    print(
        f"{label}: initial={measurement['initial_used_mb']:.1f} MB, "
        f"peak={measurement['peak_used_mb']:.1f} MB, "
        f"incremental_peak={measurement['incremental_peak_mb']:.1f} MB, "
        f"post_cleanup={measurement['post_cleanup_used_mb']:.1f} MB"
    )
    print(f"{label} offload state: {measurement['offload_states']}")


@pytest.mark.diffusion
@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, (False, True)),
        ("baseline-first", (False, True)),
        ("offload-first", (True, False)),
    ],
)
def test_measurement_order(monkeypatch, configured: str | None, expected: tuple[bool, bool]) -> None:
    if configured is None:
        monkeypatch.delenv(_MEASUREMENT_ORDER_ENV, raising=False)
    else:
        monkeypatch.setenv(_MEASUREMENT_ORDER_ENV, configured)

    assert _measurement_order() == expected


@pytest.mark.diffusion
@pytest.mark.core_model
@pytest.mark.cpu
def test_measurement_order_rejects_unknown_value(monkeypatch) -> None:
    monkeypatch.setenv(_MEASUREMENT_ORDER_ENV, "unknown")

    with pytest.raises(ValueError, match=_MEASUREMENT_ORDER_ENV):
        _measurement_order()


@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4", "rocm": "MI325"})
@pytest.mark.parametrize(
    "model_name",
    [pytest.param(name, marks=MODEL_MARKS[name]) for name in MODELS],
)
def test_layerwise_offload_diffusion_model(model_name: str):
    """Test that layerwise offloading reduces GPU memory usage.

    This test verifies that layerwise offloading significantly reduces peak
    GPU memory usage compared to loading the entire model on GPU. The layerwise
    offloader keeps only a single transformer block on GPU at a time, with
    prefetching for compute-memory overlap.
    """
    measurements: dict[bool, dict[str, Any]] = {}
    try:
        measurement_order = _measurement_order()
        print(
            "Measurement order: "
            + " -> ".join("layerwise-offload" if enabled else "baseline" for enabled in measurement_order)
        )
        for enabled in measurement_order:
            measurements[enabled] = run_inference_isolated(model_name, layerwise_offload=enabled)
    except ValueError as exc:
        # omni_snapshot_download wraps GatedRepoError in a ValueError; skip instead of failing.
        if "Access to model" in str(exc) and "is restricted" in str(exc):
            pytest.skip(
                f"Skipping: gated HF repo {model_name!r} inaccessible "
                f"({exc}). See docs/contributing/ci/hf_credentials.md."
            )
        raise

    no_offload = measurements[False]
    layerwise_offload = measurements[True]
    _assert_offload_state(no_offload, expected_enabled=False)
    _assert_offload_state(layerwise_offload, expected_enabled=True)
    _print_measurement("No offload", no_offload)
    _print_measurement("Layerwise offload", layerwise_offload)

    audio_no_offload = no_offload["audio"]
    audio_offload = layerwise_offload["audio"]
    if audio_no_offload is not None or audio_offload is not None:
        assert audio_no_offload is not None and audio_offload is not None, (
            "Both isolated modes must return audio when either mode returns audio"
        )
        # Match the sibling cpu-offload test's tolerance: layerwise offload moves
        # blocks across the PCIe bus on a side stream, which can perturb cuBLAS
        # algorithm selection and produce ~ULP-level drift larger than 1e-3.
        check_audio_determinism(audio_offload, audio_no_offload, atol=1e-2)

    is_rocm = torch.version.hip is not None
    platform = "rocm" if is_rocm else "cuda"
    expected_saved_memory = MODELS[model_name][platform]

    if expected_saved_memory is None:
        pytest.skip(f"Threshold not defined for {platform} on {model_name}")
    assert expected_saved_memory is not None

    # Verify that layerwise offloading significantly reduces memory usage
    # Passes only if the actual savings meets the expected savings
    no_offload_peak_memory = no_offload["incremental_peak_mb"]
    layerwise_offload_peak_memory = layerwise_offload["incremental_peak_mb"]
    actual_saved_memory = no_offload_peak_memory - layerwise_offload_peak_memory
    assert layerwise_offload_peak_memory + expected_saved_memory <= no_offload_peak_memory, (
        f"Layerwise offload peak memory {layerwise_offload_peak_memory} MB "
        f"should be at least {expected_saved_memory} MB less than no offload peak memory "
        f"{no_offload_peak_memory} MB (actual savings: {actual_saved_memory} MB)"
    )
