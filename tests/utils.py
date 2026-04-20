# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Some functions are copied from vllm/tests/utils.py
import functools
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path
from typing import Any, Literal

import cloudpickle
import pytest
import torch
from typing_extensions import ParamSpec
from vllm.platforms import current_platform
from vllm.utils.torch_utils import cuda_device_count_stateless

from vllm_omni.platforms import current_omni_platform

_P = ParamSpec("_P")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEPLOY_DIR = _REPO_ROOT / "vllm_omni" / "deploy"
_CI_GENERATED_DIR = _REPO_ROOT / "tests" / ".ci_generated"


# CI overlays as Python dicts (LSP-friendly). Materialized on demand to
# tests/.ci_generated/<model>.yaml via get_deploy_config_path("ci/<name>.yaml").
_CI_OVERLAYS: dict[str, dict[str, Any]] = {
    "qwen2_5_omni": {
        "base_config": "qwen2_5_omni.yaml",
        "async_chunk": False,
        "stages": [
            {
                "stage_id": 0,
                "max_model_len": 16384,
                "max_num_batched_tokens": 16384,
                "max_num_seqs": 1,
                "gpu_memory_utilization": 0.9,
                "skip_mm_profiling": True,
                "load_format": "dummy",
                "default_sampling_params": {"max_tokens": 128},
            },
            {
                "stage_id": 1,
                "max_model_len": 16384,
                "max_num_batched_tokens": 16384,
                "max_num_seqs": 1,
                "gpu_memory_utilization": 0.4,
                "skip_mm_profiling": True,
                "load_format": "dummy",
                "default_sampling_params": {"max_tokens": 4096},
            },
            {
                "stage_id": 2,
                "max_num_seqs": 1,
                "gpu_memory_utilization": 0.5,
                "max_num_batched_tokens": 8192,
                "max_model_len": 8192,
                "load_format": "dummy",
                "devices": "2",
                "default_sampling_params": {"max_tokens": 8192},
            },
        ],
        "platforms": {
            "rocm": {
                "stages": [
                    {"stage_id": 0, "gpu_memory_utilization": 0.9},
                    {"stage_id": 1, "gpu_memory_utilization": 0.4},
                    {"stage_id": 2, "gpu_memory_utilization": 0.5, "devices": "2"},
                ],
            },
            "xpu": {
                "stages": [
                    {
                        "stage_id": 0,
                        "gpu_memory_utilization": 0.9,
                        "max_num_batched_tokens": 16384,
                        "max_model_len": 16384,
                    },
                    {"stage_id": 1, "gpu_memory_utilization": 0.5},
                    {
                        "stage_id": 2,
                        "gpu_memory_utilization": 0.3,
                        "max_num_batched_tokens": 4096,
                        "max_model_len": 4096,
                        "devices": "2",
                    },
                ],
            },
        },
    },
    "qwen3_omni_moe": {
        "base_config": "qwen3_omni_moe.yaml",
        "async_chunk": False,
        "stages": [
            {
                "stage_id": 0,
                "max_num_seqs": 5,
                "max_model_len": 32768,
                "mm_processor_cache_gb": 0,
                "load_format": "dummy",
                "default_sampling_params": {"max_tokens": 150, "ignore_eos": False},
            },
            {
                "stage_id": 1,
                "gpu_memory_utilization": 0.5,
                "max_num_seqs": 5,
                "max_model_len": 32768,
                "load_format": "dummy",
                "default_sampling_params": {"max_tokens": 1000},
            },
            {
                "stage_id": 2,
                "max_num_seqs": 5,
                "max_num_batched_tokens": 100000,
                "load_format": "dummy",
                "default_sampling_params": {"max_tokens": 2000},
            },
        ],
        "platforms": {
            "rocm": {
                "stages": [
                    {"stage_id": 0, "max_num_seqs": 1, "default_sampling_params": {"max_tokens": 100}},
                    {
                        "stage_id": 1,
                        "max_num_seqs": 1,
                        "enforce_eager": True,
                        "default_sampling_params": {"max_tokens": 100},
                    },
                    {
                        "stage_id": 2,
                        "max_num_seqs": 1,
                        "max_num_batched_tokens": 1000000,
                        "default_sampling_params": {"max_tokens": 200},
                    },
                ],
            },
            "xpu": {
                "stages": [
                    {
                        "stage_id": 0,
                        "gpu_memory_utilization": 0.85,
                        "max_num_seqs": 1,
                        "tensor_parallel_size": 4,
                        "enforce_eager": True,
                        "max_num_batched_tokens": 4096,
                        "max_model_len": 4096,
                        "max_cudagraph_capture_size": 0,
                        "skip_mm_profiling": True,
                        "devices": "0,1,2,3",
                        "default_sampling_params": {"max_tokens": 100, "ignore_eos": False},
                    },
                    {
                        "stage_id": 1,
                        "gpu_memory_utilization": 0.6,
                        "max_num_seqs": 1,
                        "enforce_eager": True,
                        "max_num_batched_tokens": 4096,
                        "max_model_len": 4096,
                        "max_cudagraph_capture_size": 0,
                        "skip_mm_profiling": True,
                        "devices": "4",
                    },
                    {
                        "stage_id": 2,
                        "gpu_memory_utilization": 0.3,
                        "max_num_seqs": 1,
                        "max_num_batched_tokens": 100000,
                        "max_cudagraph_capture_size": 0,
                        "skip_mm_profiling": True,
                        "devices": "5",
                        "default_sampling_params": {"max_tokens": 2000},
                    },
                ],
            },
        },
    },
    # Single-stage thinker-only topology for the abort test.
    "qwen2_5_omni_thinker_only": {
        "async_chunk": False,
        "pipeline": "qwen2_5_omni_thinker_only",
        "stages": [
            {
                "stage_id": 0,
                "max_num_seqs": 1,
                "gpu_memory_utilization": 0.9,
                "enforce_eager": True,
                "max_num_batched_tokens": 16384,
                "max_model_len": 16384,
                "skip_mm_profiling": True,
                "mm_processor_cache_gb": 0,
                "load_format": "dummy",
                "devices": "0",
                "default_sampling_params": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_tokens": 128,
                    "seed": 42,
                    "repetition_penalty": 1.1,
                },
            },
        ],
    },
}


def _materialize_ci_overlay(model_type: str) -> Path:
    import yaml

    if model_type not in _CI_OVERLAYS:
        raise KeyError(f"No CI overlay registered for {model_type!r}. Available: {sorted(_CI_OVERLAYS)}")

    _CI_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    out = _CI_GENERATED_DIR / f"{model_type}.yaml"

    overlay = {**_CI_OVERLAYS[model_type]}
    base = overlay.get("base_config")
    if base:
        overlay["base_config"] = str(_DEPLOY_DIR / base)

    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(overlay, f, sort_keys=False)
    return out


def get_deploy_config_path(rel_path: str) -> str:
    """Resolve a deploy yaml; ``ci/<model>.yaml`` materializes from ``_CI_OVERLAYS``."""
    if rel_path.startswith("ci/") and rel_path.endswith(".yaml"):
        model_type = rel_path[len("ci/") : -len(".yaml")]
        if model_type in _CI_OVERLAYS:
            return str(_materialize_ci_overlay(model_type))
    return str(_DEPLOY_DIR / rel_path)


if current_platform.is_rocm():
    from amdsmi import (
        amdsmi_get_gpu_vram_usage,
        amdsmi_get_processor_handles,
        amdsmi_init,
        amdsmi_shut_down,
    )

    @contextmanager
    def _nvml():
        try:
            amdsmi_init()
            yield
        finally:
            amdsmi_shut_down()
elif current_platform.is_cuda():
    from vllm.third_party.pynvml import (
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlInit,
        nvmlShutdown,
    )

    @contextmanager
    def _nvml():
        try:
            nvmlInit()
            yield
        finally:
            nvmlShutdown()
else:

    @contextmanager
    def _nvml():
        yield


def get_physical_device_indices(devices):
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        return devices

    visible_indices = [int(x) for x in visible_devices.split(",")]
    index_mapping = {i: physical for i, physical in enumerate(visible_indices)}
    return [index_mapping[i] for i in devices if i in index_mapping]


@_nvml()
def wait_for_gpu_memory_to_clear(
    *,
    devices: list[int],
    threshold_bytes: int | None = None,
    threshold_ratio: float | None = None,
    timeout_s: float = 120,
) -> None:
    import gc

    assert threshold_bytes is not None or threshold_ratio is not None
    # Use nvml instead of pytorch to reduce measurement error from torch cuda
    # context.
    devices = get_physical_device_indices(devices)
    start_time = time.time()

    # Print waiting start information
    device_list = ", ".join(str(d) for d in devices)
    if threshold_bytes is not None:
        threshold_str = f"{threshold_bytes / 2**30:.2f} GiB"
        condition_str = f"Memory usage ≤ {threshold_str}"
    else:
        threshold_percent = threshold_ratio * 100
        threshold_str = f"{threshold_percent:.1f}%"
        condition_str = f"Memory usage ratio ≤ {threshold_str}"

    print(f"[GPU Memory Monitor] Waiting for GPU {device_list} to free memory, Condition: {condition_str}")

    # Define the is_free function based on threshold type
    if threshold_bytes is not None:

        def is_free(used, total):
            return used <= threshold_bytes / 2**30
    else:

        def is_free(used, total):
            return used / total <= threshold_ratio

    while True:
        output: dict[int, str] = {}
        output_raw: dict[int, tuple[float, float]] = {}
        for device in devices:
            if current_platform.is_rocm():
                dev_handle = amdsmi_get_processor_handles()[device]
                mem_info = amdsmi_get_gpu_vram_usage(dev_handle)
                gb_used = mem_info["vram_used"] / 2**10
                gb_total = mem_info["vram_total"] / 2**10
            else:
                dev_handle = nvmlDeviceGetHandleByIndex(device)
                mem_info = nvmlDeviceGetMemoryInfo(dev_handle)
                gb_used = mem_info.used / 2**30
                gb_total = mem_info.total / 2**30
            output_raw[device] = (gb_used, gb_total)
            # Format to more readable form
            usage_percent = (gb_used / gb_total) * 100 if gb_total > 0 else 0
            output[device] = f"{gb_used:.1f}GiB/{gb_total:.1f}GiB ({usage_percent:.1f}%)"

        # Optimized GPU memory status print
        print("[GPU Memory Status] Current usage:")
        for device_id, mem_info in output.items():
            print(f"  GPU {device_id}: {mem_info}")

        # Calculate waiting duration
        dur_s = time.time() - start_time
        elapsed_minutes = dur_s / 60

        # Check if all devices meet the condition
        if all(is_free(used, total) for used, total in output_raw.values()):
            # Optimized completion message
            print(f"[GPU Memory Freed] Devices {device_list} meet memory condition")
            print(f"   Condition: {condition_str}")
            print(f"   Wait time: {dur_s:.1f} seconds ({elapsed_minutes:.1f} minutes)")
            print("   Final status:")
            for device_id, mem_info in output.items():
                print(f"     GPU {device_id}: {mem_info}")
            break

        # Check timeout
        if dur_s >= timeout_s:
            raise ValueError(
                f"[GPU Memory Timeout] Devices {device_list} still don't meet memory condition after {dur_s:.1f} seconds\n"
                f"Condition: {condition_str}\n"
                f"Current status:\n" + "\n".join(f"  GPU {device}: {output[device]}" for device in devices)
            )

        # Add waiting hint (optional)
        if dur_s > 10 and int(dur_s) % 10 == 0:  # Show hint every 10 seconds
            print(f"Waiting... Already waited {dur_s:.1f} seconds ({elapsed_minutes:.1f} minutes)")

        gc.collect()
        torch.cuda.empty_cache()

        time.sleep(5)


def fork_new_process_for_each_test(func: Callable[_P, None]) -> Callable[_P, None]:
    """Decorator to fork a new process for each test function.
    See https://github.com/vllm-project/vllm/issues/7053 for more details.
    """

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> None:
        # Make the process the leader of its own process group
        # to avoid sending SIGTERM to the parent process
        os.setpgrp()
        from _pytest.outcomes import Skipped

        # Create a unique temporary file to store exception info from child
        # process. Use test function name and process ID to avoid collisions.
        with (
            tempfile.NamedTemporaryFile(
                delete=False, mode="w+b", prefix=f"vllm_test_{func.__name__}_{os.getpid()}_", suffix=".exc"
            ) as exc_file,
            ExitStack() as delete_after,
        ):
            exc_file_path = exc_file.name
            delete_after.callback(os.remove, exc_file_path)

            pid = os.fork()
            print(f"Fork a new process to run a test {pid}")
            if pid == 0:
                # Parent process responsible for deleting, don't delete
                # in child.
                delete_after.pop_all()
                try:
                    func(*args, **kwargs)
                except Skipped as e:
                    # convert Skipped to exit code 0
                    print(str(e))
                    os._exit(0)
                except Exception as e:
                    import traceback

                    tb_string = traceback.format_exc()

                    # Try to serialize the exception object first
                    exc_to_serialize: dict[str, Any]
                    try:
                        # First, try to pickle the actual exception with
                        # its traceback.
                        exc_to_serialize = {"pickled_exception": e}
                        # Test if it can be pickled
                        cloudpickle.dumps(exc_to_serialize)
                    except (Exception, KeyboardInterrupt):
                        # Fall back to string-based approach.
                        exc_to_serialize = {
                            "exception_type": type(e).__name__,
                            "exception_msg": str(e),
                            "traceback": tb_string,
                        }
                    try:
                        with open(exc_file_path, "wb") as f:
                            cloudpickle.dump(exc_to_serialize, f)
                    except Exception:
                        # Fallback: just print the traceback.
                        print(tb_string)
                    os._exit(1)
                else:
                    os._exit(0)
            else:
                pgid = os.getpgid(pid)
                _pid, _exitcode = os.waitpid(pid, 0)
                # ignore SIGTERM signal itself
                old_signal_handler = signal.signal(signal.SIGTERM, signal.SIG_IGN)
                # kill all child processes
                os.killpg(pgid, signal.SIGTERM)
                # restore the signal handler
                signal.signal(signal.SIGTERM, old_signal_handler)
                if _exitcode != 0:
                    # Try to read the exception from the child process
                    exc_info = {}
                    if os.path.exists(exc_file_path):
                        with suppress(Exception), open(exc_file_path, "rb") as f:
                            exc_info = cloudpickle.load(f)

                    if (original_exception := exc_info.get("pickled_exception")) is not None:
                        # Re-raise the actual exception object if it was
                        # successfully pickled.
                        assert isinstance(original_exception, Exception)
                        raise original_exception

                    if (original_tb := exc_info.get("traceback")) is not None:
                        # Use string-based traceback for fallback case
                        raise AssertionError(
                            f"Test {func.__name__} failed when called with"
                            f" args {args} and kwargs {kwargs}"
                            f" (exit code: {_exitcode}):\n{original_tb}"
                        ) from None

                    # Fallback to the original generic error
                    raise AssertionError(
                        f"function {func.__name__} failed when called with"
                        f" args {args} and kwargs {kwargs}"
                        f" (exit code: {_exitcode})"
                    ) from None

    return wrapper


def spawn_new_process_for_each_test(f: Callable[_P, None]) -> Callable[_P, None]:
    """Decorator to spawn a new process for each test function."""

    @functools.wraps(f)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> None:
        # Check if we're already in a subprocess
        if os.environ.get("RUNNING_IN_SUBPROCESS") == "1":
            # If we are, just run the function directly
            return f(*args, **kwargs)

        import torch.multiprocessing as mp

        with suppress(RuntimeError):
            mp.set_start_method("spawn")

        # Get the module
        module_name = f.__module__

        # Create a process with environment variable set
        env = os.environ.copy()
        env["RUNNING_IN_SUBPROCESS"] = "1"

        with tempfile.TemporaryDirectory() as tempdir:
            output_filepath = os.path.join(tempdir, "new_process.tmp")

            # `cloudpickle` allows pickling complex functions directly
            input_bytes = cloudpickle.dumps((f, output_filepath))

            cmd = [sys.executable, "-m", f"{module_name}"]

            returned = subprocess.run(cmd, input=input_bytes, capture_output=True, env=env)

            # check if the subprocess is successful
            try:
                returned.check_returncode()
            except Exception as e:
                # wrap raised exception to provide more information
                raise RuntimeError(f"Error raised in subprocess:\n{returned.stderr.decode()}") from e

    return wrapper


def create_new_process_for_each_test(
    method: Literal["spawn", "fork"] | None = None,
) -> Callable[[Callable[_P, None]], Callable[_P, None]]:
    """Creates a decorator that runs each test function in a new process.

    Args:
        method: The process creation method. Can be either "spawn" or "fork".
               If not specified, it defaults to "spawn" on ROCm and XPU
               platforms and "fork" otherwise.

    Returns:
        A decorator to run test functions in separate processes.
    """
    if method is None:
        # TODO: Spawn is not working correctly on ROCm
        # The test content will not run and tests passed immediately.
        # For now, using `fork` for ROCm as it can run with `fork`
        # and tests are running correctly.
        use_spawn = current_platform.is_xpu()
        method = "spawn" if use_spawn else "fork"

    assert method in ["spawn", "fork"], "Method must be either 'spawn' or 'fork'"

    if method == "fork":
        return fork_new_process_for_each_test

    return spawn_new_process_for_each_test


def cuda_marks(*, res: str, num_cards: int):
    """
    Get a collection of pytest marks to apply for `@cuda_test`.

    Args:
        res: Resource type, e.g., "L4" or "H100".
        num_cards: Number of GPU cards required.

    Returns:
        List of pytest marks to apply.
    """
    test_platform_detail = pytest.mark.cuda

    if res == "L4":
        test_resource = pytest.mark.L4
    elif res == "H100":
        test_resource = pytest.mark.H100
    else:
        raise ValueError(f"Invalid CUDA resource type: {res}. Supported: L4, H100")

    marks = [test_resource, test_platform_detail]

    if num_cards == 1:
        return marks
    else:
        test_distributed = pytest.mark.distributed_cuda(num_cards=num_cards)
        test_skipif = pytest.mark.skipif_cuda(
            cuda_device_count_stateless() < num_cards,
            reason=f"Need at least {num_cards} CUDA GPUs to run the test.",
        )
        return marks + [test_distributed, test_skipif]


def rocm_marks(*, res: str, num_cards: int):
    """
    Get a collection of pytest marks to apply for `@rocm_test`.

    Args:
        res: Resource type, e.g., "MI325".
        num_cards: Number of GPU cards required.

    Returns:
        List of pytest marks to apply.
    """
    test_platform_detail = pytest.mark.rocm

    if res == "MI325":
        test_resource = pytest.mark.MI325
    else:
        raise ValueError(f"Invalid ROCm resource type: {res}. Supported: MI325")

    marks = [test_resource, test_platform_detail]

    if num_cards == 1:
        return marks
    else:
        test_distributed = pytest.mark.distributed_rocm(num_cards=num_cards)
        # TODO: add ROCm support for `skipif_rocm` marker
        return marks + [test_distributed]


def xpu_marks(*, res: str, num_cards: int):
    """
    Get a collection of pytest marks to apply for `@xpu_test`.

    Args:
        res: Resource type, e.g., "B60".
        num_cards: Number of GPU cards required.

    Returns:
        List of pytest marks to apply.
    """
    test_platform_detail = pytest.mark.xpu

    if res == "B60":
        test_resource = pytest.mark.B60
    else:
        raise ValueError(f"Invalid XPU resource type: {res}. Supported: B60")

    marks = [test_resource, test_platform_detail]

    if num_cards == 1:
        return marks
    else:
        test_distributed = pytest.mark.distributed_rocm(num_cards=num_cards)
        # TODO: add XPU support for `skipif_xpu` marker
        return marks + [test_distributed]


def musa_marks(*, res: str, num_cards: int):
    """
    Get a collection of pytest marks to apply for `@musa_test`.

    Args:
        res: Resource type, e.g., "S5000".
        num_cards: Number of GPU cards required.

    Returns:
        List of pytest marks to apply.
    """
    test_platform_detail = pytest.mark.musa

    if res == "S5000":
        test_resource = pytest.mark.S5000
    else:
        raise ValueError(f"Invalid MUSA resource type: {res}. Supported: S5000")

    marks = [test_resource, test_platform_detail]

    if num_cards == 1:
        return marks
    else:
        test_distributed = pytest.mark.distributed_musa(num_cards=num_cards)
        # TODO: add MUSA support for `skipif_musa` marker
        return marks + [test_distributed]


def gpu_marks(*, res: str, num_cards: int):
    """
    Get a collection of pytest marks to apply for `@gpu_test`.
    Platform is automatically determined based on resource type.

    Args:
        res: Resource type, e.g., "L4", "H100" for CUDA, or "MI325" for ROCm, or "B60" for XPU, or "S5000" for MUSA.
        num_cards: Number of GPU cards required.

    Returns:
        List of pytest marks to apply.
    """
    test_platform = pytest.mark.gpu
    if res in ("L4", "H100"):
        return [test_platform] + cuda_marks(res=res, num_cards=num_cards)
    if res == "MI325":
        return [test_platform] + rocm_marks(res=res, num_cards=num_cards)
    if res == "B60":
        return [test_platform] + xpu_marks(res=res, num_cards=num_cards)
    if res == "S5000":
        return [test_platform] + musa_marks(res=res, num_cards=num_cards)
    raise ValueError(f"Invalid resource type: {res}. Supported: L4, H100, MI325, B60, S5000")


def npu_marks(*, res: str, num_cards: int):
    """Get a collection of pytest marks to apply for `@npu_test`."""
    test_platform = pytest.mark.npu
    if res == "A2":
        test_resource = pytest.mark.A2
    elif res == "A3":
        test_resource = pytest.mark.A3
    else:
        # TODO: Currently we don't have various NPU card types defined
        # Use None to skip resource-specific marking for unknown types
        test_resource = None

    if num_cards == 1:
        return [mark for mark in [test_platform, test_resource] if mark is not None]
    else:
        # Multiple cards scenario needs distributed_npu mark
        test_distributed = pytest.mark.distributed_npu(num_cards=num_cards)
        # TODO: add NPU support for `skipif_npu` marker
        return [mark for mark in [test_platform, test_resource, test_distributed] if mark is not None]


def hardware_marks(*, res: dict[str, str], num_cards: int | dict[str, int] = 1):
    """
    Get a collection of pytest marks to apply for `@hardware_test`,
    including CUDA, ROCm, XPU, NPU, and MUSA,
    based on the specified platforms and resources.
    """
    # Validate platforms
    # Don't validate platform details in this decorator
    for platform, _ in res.items():
        if platform not in ("cuda", "rocm", "xpu", "npu", "musa"):
            raise ValueError(f"Unsupported platform: {platform}")

    # Normalize num_cards
    if isinstance(num_cards, int):
        num_cards_dict = {platform: num_cards for platform in res.keys()}
    else:
        num_cards_dict = num_cards
        for platform in num_cards_dict.keys():
            if platform not in res:
                raise ValueError(
                    f"Platform '{platform}' in num_cards but not in res. Available platforms: {list(res.keys())}"
                )
        for platform in res.keys():
            if platform not in num_cards_dict:
                num_cards_dict[platform] = 1

    # Collect marks from all platforms
    all_marks: list[pytest.MarkDecorator] = []
    for platform, resource in res.items():
        cards = num_cards_dict[platform]
        if platform == "cuda" or platform == "rocm" or platform == "xpu":
            marks = gpu_marks(res=resource, num_cards=cards)
        elif platform == "musa":
            marks = musa_marks(res=resource, num_cards=cards)
        elif platform == "npu":
            marks = npu_marks(res=resource, num_cards=cards)
        else:
            raise ValueError(f"Unsupported platform: {platform}")
        all_marks.extend(marks)
    return all_marks


def hardware_test(*, res: dict[str, str], num_cards: int | dict[str, int] = 1):
    """
    Decorate a test for multiple hardware platforms with a single call.
    Automatically wraps the test with @create_new_process_for_each_test() for distributed tests.

    Args:
        res: Mapping from platform to resource type. Supported platforms/resources:
            - cuda: L4, H100
            - rocm: MI325
            - xpu: B60
            - npu: A2, A3
            - musa: S5000
        num_cards: Number of cards required. Can be:
            - int: same card count for all platforms (default: 1)
            - dict: per-platform card count, e.g., {"cuda": 2, "rocm": 2}

    Example:
        @hardware_test(
            res={"cuda": "L4", "rocm": "MI325", "npu": "A2", "musa": "S5000"},
            num_cards={"cuda": 2, "rocm": 2, "npu": 2, "musa": 2},
        )
        def test_multi_platform():
            ...
    """
    all_marks = hardware_marks(res=res, num_cards=num_cards)

    def wrapper(f: Callable[_P, None]) -> Callable[_P, None]:
        func = f
        for mark in reversed(all_marks):
            func = mark(func)
        return func

    return wrapper


class DeviceMemoryMonitor:
    """Poll global device memory usage."""

    def __init__(self, device_index: int, interval: float = 0.05):
        self.device_index = device_index
        self.interval = interval
        self._peak_used_mb = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def monitor_loop() -> None:
            while not self._stop_event.is_set():
                try:
                    with current_omni_platform.device(self.device_index):
                        free_bytes, total_bytes = current_omni_platform.mem_get_info()
                    used_mb = (total_bytes - free_bytes) / (1024**2)
                    self._peak_used_mb = max(self._peak_used_mb, used_mb)
                except Exception:
                    pass
                time.sleep(self.interval)

        self._thread = threading.Thread(target=monitor_loop, daemon=False)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    @property
    def peak_used_mb(self) -> float:
        fallback_alloc = current_omni_platform.max_memory_allocated(device=self.device_index) / (1024**2)
        fallback_reserved = current_omni_platform.max_memory_reserved(device=self.device_index) / (1024**2)
        return max(self._peak_used_mb, fallback_alloc, fallback_reserved)

    def __del__(self):
        self.stop()
