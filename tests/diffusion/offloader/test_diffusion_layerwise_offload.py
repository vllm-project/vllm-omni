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

from tests.helpers.mark import hardware_marks
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
    "riverclouds/qwen_image_random": [
        pytest.mark.core_model,
        *hardware_marks(res={"cuda": "L4", "rocm": "MI325"}),
    ],
    "stabilityai/stable-audio-open-1.0": [
        pytest.mark.full_model,
        *hardware_marks(res={"cuda": ["L4", "B200"], "rocm": "MI325"}),
    ],
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
_OFFLOAD_MEMORY_PROBE = "vllm_omni_layerwise_offload_memory_test"
# A second CI build can reverse the same exact-head measurement without a
# source edit; the default keeps the existing baseline-first workload.
_MEASUREMENT_ORDER_ENV = "VLLM_OMNI_OFFLOAD_TEST_ORDER"
_MEASUREMENT_ORDERS: dict[str, tuple[bool, bool]] = {
    "baseline-first": (False, True),
    "offload-first": (True, False),
}


class OffloadStateProbe:
    """Test-only worker extension exposing offloader and allocator state."""

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

    def reset_peak_memory_for_test(self) -> dict[str, str]:
        """Start a process-local inference peak after model initialization."""
        current_omni_platform.synchronize()
        torch.accelerator.reset_peak_memory_stats()
        return {"probe": _OFFLOAD_MEMORY_PROBE, "operation": "reset"}

    def get_peak_memory_for_test(self) -> dict[str, str | float]:
        """Return this worker's allocator peak without counting other jobs."""
        current_omni_platform.synchronize()
        return {
            "probe": _OFFLOAD_MEMORY_PROBE,
            "operation": "snapshot",
            "peak_allocated_mb": torch.accelerator.max_memory_allocated() / (1024**2),
        }


def check_audio_determinism(audio1: np.ndarray, audio2: np.ndarray, atol: float = 1e-2) -> bool:
    if not np.allclose(audio1, audio2, atol=atol):
        diff = np.abs(audio1 - audio2)
        print(f"Max difference: {diff.max()}")
        print(f"Mean difference: {diff.mean()}")
        raise AssertionError(f"Audio outputs differ beyond tolerance atol={atol}")
    return True


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


def _collect_probe_results(value: Any, probe: str) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value] if value.get("probe") == probe else []
    if isinstance(value, (list, tuple)):
        results: list[dict[str, Any]] = []
        for item in value:
            results.extend(_collect_probe_results(item, probe))
        return results
    return []


def _collect_offload_states(value: Any) -> list[dict[str, Any]]:
    return _collect_probe_results(value, _OFFLOAD_STATE_PROBE)


def _collect_memory_results(value: Any, operation: str) -> list[dict[str, Any]]:
    return [
        result
        for result in _collect_probe_results(value, _OFFLOAD_MEMORY_PROBE)
        if result.get("operation") == operation
    ]


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

        # Reset and read allocator peaks in the diffusion worker process. A
        # device-wide mem_get_info() sample includes unrelated sibling jobs on
        # the same GPU and produced negative "savings" in shared ROCm CI.
        reset_results = _collect_memory_results(
            runner.omni.engine.collective_rpc(method="reset_peak_memory_for_test", timeout=60),
            "reset",
        )
        if not reset_results:
            raise AssertionError("The worker peak-memory probe returned no reset acknowledgements")

        # Refer to tests/e2e/offline_inference/test_wan22.py.
        # Use minimal settings for testing.
        output = runner.omni.generate(
            "A cat sitting on a table",
            OmniDiffusionSamplingParams(
                generator=torch.Generator(device=current_omni_platform.device_type).manual_seed(42),
                guidance_scale=1.0,
                num_inference_steps=num_inference_steps,
                **params["sampler_params"],
            ),
        )
        memory_results = _collect_memory_results(
            runner.omni.engine.collective_rpc(method="get_peak_memory_for_test", timeout=60),
            "snapshot",
        )
        if not memory_results:
            raise AssertionError("The worker peak-memory probe returned no snapshots")

        audio = _extract_audio(output)
        del output

    del runner
    gc.collect()
    cleanup_dist_env_and_memory()
    current_omni_platform.empty_cache()

    return {
        "peak_allocated_mb": max(float(result["peak_allocated_mb"]) for result in memory_results),
        "memory_results": memory_results,
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


def _assert_audio_outputs(
    model_name: str,
    audio_no_offload: np.ndarray | None,
    audio_offload: np.ndarray | None,
) -> None:
    if model_name not in AUDIO_MODEL:
        return

    assert audio_no_offload is not None, "Baseline Stable Audio inference returned no audio"
    assert audio_offload is not None, "Layerwise-offloaded Stable Audio inference returned no audio"
    # Match the sibling cpu-offload test's tolerance: layerwise offload moves
    # blocks across the PCIe bus on a side stream, which can perturb cuBLAS
    # algorithm selection and produce ~ULP-level drift larger than 1e-3.
    check_audio_determinism(audio_offload, audio_no_offload, atol=1e-2)


def _print_measurement(label: str, measurement: dict[str, Any]) -> None:
    print(f"{label}: worker_peak_allocated={measurement['peak_allocated_mb']:.1f} MB")
    print(f"{label} worker memory: {measurement['memory_results']}")
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
@pytest.mark.core_model
@pytest.mark.cpu
def test_collect_memory_results_filters_nested_worker_responses() -> None:
    value = [
        None,
        [
            {"probe": _OFFLOAD_MEMORY_PROBE, "operation": "reset"},
            {
                "probe": _OFFLOAD_MEMORY_PROBE,
                "operation": "snapshot",
                "peak_allocated_mb": 123.0,
            },
        ],
        {"probe": _OFFLOAD_STATE_PROBE, "operation": "snapshot"},
    ]

    assert _collect_memory_results(value, "reset") == [{"probe": _OFFLOAD_MEMORY_PROBE, "operation": "reset"}]
    assert _collect_memory_results(value, "snapshot") == [
        {
            "probe": _OFFLOAD_MEMORY_PROBE,
            "operation": "snapshot",
            "peak_allocated_mb": 123.0,
        }
    ]


@pytest.mark.diffusion
@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.parametrize(
    ("audio_no_offload", "audio_offload", "message"),
    [
        (None, None, "Baseline Stable Audio inference returned no audio"),
        (None, np.zeros(1), "Baseline Stable Audio inference returned no audio"),
        (np.zeros(1), None, "Layerwise-offloaded Stable Audio inference returned no audio"),
    ],
)
def test_stable_audio_requires_both_outputs(audio_no_offload, audio_offload, message) -> None:
    with pytest.raises(AssertionError, match=message):
        _assert_audio_outputs("stabilityai/stable-audio-open-1.0", audio_no_offload, audio_offload)


@pytest.mark.diffusion
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
    _assert_audio_outputs(model_name, audio_no_offload, audio_offload)

    is_rocm = torch.version.hip is not None
    platform = "rocm" if is_rocm else "cuda"
    expected_saved_memory = MODELS[model_name][platform]

    if expected_saved_memory is None:
        pytest.skip(f"Threshold not defined for {platform} on {model_name}")
    assert expected_saved_memory is not None

    # Verify that layerwise offloading significantly reduces memory usage
    # Passes only if the actual savings meets the expected savings
    no_offload_peak_memory = no_offload["peak_allocated_mb"]
    layerwise_offload_peak_memory = layerwise_offload["peak_allocated_mb"]
    actual_saved_memory = no_offload_peak_memory - layerwise_offload_peak_memory
    assert layerwise_offload_peak_memory + expected_saved_memory <= no_offload_peak_memory, (
        f"Layerwise offload peak memory {layerwise_offload_peak_memory} MB "
        f"should be at least {expected_saved_memory} MB less than no offload peak memory "
        f"{no_offload_peak_memory} MB (actual savings: {actual_saved_memory} MB)"
    )
