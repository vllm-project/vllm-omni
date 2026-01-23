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
from typing import Any, Literal

import cloudpickle
import pytest
import torch
from typing_extensions import ParamSpec
from vllm.platforms import current_platform
from vllm.utils.torch_utils import cuda_device_count_stateless

_P = ParamSpec("_P")

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
    assert threshold_bytes is not None or threshold_ratio is not None
    # Use nvml instead of pytorch to reduce measurement error from torch cuda
    # context.
    devices = get_physical_device_indices(devices)
    start_time = time.time()
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
            output[device] = f"{gb_used:.02f}/{gb_total:.02f}"

        print("gpu memory used/total (GiB): ", end="")
        for k, v in output.items():
            print(f"{k}={v}; ", end="")
        print("")

        if threshold_bytes is not None:

            def is_free(used, total):
                return used <= threshold_bytes / 2**30  # noqa E731

            threshold = f"{threshold_bytes / 2**30} GiB"
        else:

            def is_free(used, total):
                return used / total <= threshold_ratio  # noqa E731

            threshold = f"{threshold_ratio:.2f}"

        dur_s = time.time() - start_time
        if all(is_free(used, total) for used, total in output_raw.values()):
            print(f"Done waiting for free GPU memory on devices {devices=} ({threshold=}) {dur_s=:.02f}")
            break

        if dur_s >= timeout_s:
            raise ValueError(f"Memory of devices {devices=} not free after {dur_s=:.02f} ({threshold=})")

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


def gpu_marks(*, res: str, num_cards: int):
    """
    Get a collection of pytest marks to apply for `@gpu_test`.
    Platform is automatically determined based on resource type.

    Args:
        res: Resource type, e.g., "L4", "H100" for CUDA, or "MI325" for ROCm.
        num_cards: Number of GPU cards required.

    Returns:
        List of pytest marks to apply.
    """
    test_platform = pytest.mark.gpu
    if res in ("L4", "H100"):
        return [test_platform] + cuda_marks(res=res, num_cards=num_cards)
    if res == "MI325":
        return [test_platform] + rocm_marks(res=res, num_cards=num_cards)
    raise ValueError(f"Invalid resource type: {res}. Supported: L4, H100, MI325")


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


def hardware_test(*, res: dict[str, str], num_cards: int | dict[str, int] = 1):
    """
    Decorate a test for multiple hardware platforms with a single call.
    Automatically wraps the test with @create_new_process_for_each_test() for distributed tests.

    Args:
        res: Mapping from platform to resource type. Supported platforms/resources:
            - cuda: L4, H100
            - rocm: MI325
            - npu: A2, A3
        num_cards: Number of cards required. Can be:
            - int: same card count for all platforms (default: 1)
            - dict: per-platform card count, e.g., {"cuda": 2, "rocm": 2}

    Example:
        @hardware_test(
            res={"cuda": "L4", "rocm": "MI325", "npu": "A2"},
            num_cards={"cuda": 2, "rocm": 2, "npu": 2},
        )
        def test_multi_platform():
            ...
    """
    # Validate platforms
    # Don't validate platform details in this decorator
    for platform, _ in res.items():
        if platform not in ("cuda", "rocm", "npu"):
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
    all_marks: list[Callable[[Callable[_P, None]], Callable[_P, None]]] = []
    for platform, resource in res.items():
        cards = num_cards_dict[platform]
        if platform == "cuda" or platform == "rocm":
            marks = gpu_marks(res=resource, num_cards=cards)
        elif platform == "npu":
            marks = npu_marks(res=resource, num_cards=cards)
        else:
            raise ValueError(f"Unsupported platform: {platform}")
        all_marks.extend(marks)

    create_new_process_flag = False
    for cards in num_cards_dict.values():
        if cards > 1:
            create_new_process_flag = True
            break

    def wrapper(f: Callable[_P, None]) -> Callable[_P, None]:
        if create_new_process_flag:
            # only for distributed tests
            func = create_new_process_for_each_test()(f)
        else:
            func = f
        for mark in reversed(all_marks):
            func = mark(func)
        return func

    return wrapper


class GPUMemoryMonitor:
    """Poll global device memory usage via CUDA APIs."""

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
                    with torch.cuda.device(self.device_index):
                        free_bytes, total_bytes = torch.cuda.mem_get_info()
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
        fallback_alloc = torch.cuda.max_memory_allocated(device=self.device_index) / (1024**2)
        fallback_reserved = torch.cuda.max_memory_reserved(device=self.device_index) / (1024**2)
        return max(self._peak_used_mb, fallback_alloc, fallback_reserved)

    def __del__(self):
        self.stop()
