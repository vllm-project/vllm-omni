# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import functools
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable
from contextlib import ExitStack, suppress
from pathlib import Path
from typing import Any, Literal

import cloudpickle
from typing_extensions import ParamSpec
from vllm.platforms import current_platform

VLLM_PATH = Path(__file__).parent.parent.parent
"""Path to root of the vLLM repository."""


_P = ParamSpec("_P")


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
                delete=False,
                mode="w+b",
                prefix=f"vllm_test_{func.__name__}_{os.getpid()}_",
                suffix=".exc",
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
                        with (
                            contextlib.suppress(Exception),
                            open(exc_file_path, "rb") as f,
                        ):
                            exc_info = cloudpickle.load(f)

                    original_exception = exc_info.get("pickled_exception")
                    if original_exception is not None and isinstance(original_exception, Exception):
                        # Re-raise the actual exception object if it was
                        # successfully pickled.
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

            repo_root = str(VLLM_PATH.resolve())

            env = dict(env or os.environ)
            env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

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
        # TODO: Find out why spawn is not working correctly on ROCm
        # The test content will not run and tests passed immediately.
        # For now, using `fork` for ROCm as it can run with `fork`
        # and tests are running correctly.
        use_spawn = current_platform.is_xpu()
        method = "spawn" if use_spawn else "fork"

    assert method in ["spawn", "fork"], "Method must be either 'spawn' or 'fork'"

    if method == "fork":
        return fork_new_process_for_each_test

    return spawn_new_process_for_each_test
