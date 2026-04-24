# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable
from contextlib import ExitStack, suppress
from typing import Any, Literal

import cloudpickle
from typing_extensions import ParamSpec
from vllm.platforms import current_platform

_P = ParamSpec("_P")


def fork_new_process_for_each_test(func: Callable[_P, None]) -> Callable[_P, None]:
    """Decorator to fork a new process for each test function."""

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> None:
        os.setpgrp()
        from _pytest.outcomes import Skipped

        with (
            tempfile.NamedTemporaryFile(
                delete=False, mode="w+b", prefix=f"vllm_test_{func.__name__}_{os.getpid()}_", suffix=".exc"
            ) as exc_file,
            ExitStack() as delete_after,
        ):
            exc_file_path = exc_file.name
            delete_after.callback(os.remove, exc_file_path)
            pid = os.fork()
            if pid == 0:
                delete_after.pop_all()
                try:
                    func(*args, **kwargs)
                except Skipped as e:
                    print(str(e))
                    os._exit(0)
                except Exception as e:
                    import traceback

                    tb_string = traceback.format_exc()
                    exc_to_serialize: dict[str, Any]
                    try:
                        exc_to_serialize = {"pickled_exception": e}
                        cloudpickle.dumps(exc_to_serialize)
                    except (Exception, KeyboardInterrupt):
                        exc_to_serialize = {
                            "exception_type": type(e).__name__,
                            "exception_msg": str(e),
                            "traceback": tb_string,
                        }
                    try:
                        with open(exc_file_path, "wb") as f:
                            cloudpickle.dump(exc_to_serialize, f)
                    except Exception:
                        print(tb_string)
                    os._exit(1)
                else:
                    os._exit(0)
            else:
                pgid = os.getpgid(pid)
                _pid, _exitcode = os.waitpid(pid, 0)
                old_signal_handler = signal.signal(signal.SIGTERM, signal.SIG_IGN)
                os.killpg(pgid, signal.SIGTERM)
                signal.signal(signal.SIGTERM, old_signal_handler)
                if _exitcode != 0:
                    exc_info = {}
                    if os.path.exists(exc_file_path):
                        with suppress(Exception), open(exc_file_path, "rb") as f:
                            exc_info = cloudpickle.load(f)
                    if (original_exception := exc_info.get("pickled_exception")) is not None:
                        assert isinstance(original_exception, Exception)
                        raise original_exception
                    if (original_tb := exc_info.get("traceback")) is not None:
                        raise AssertionError(
                            f"Test {func.__name__} failed when called with args {args} and kwargs {kwargs}"
                            f" (exit code: {_exitcode}):\n{original_tb}"
                        ) from None
                    raise AssertionError(
                        f"function {func.__name__} failed when called with args {args} and kwargs {kwargs}"
                        f" (exit code: {_exitcode})"
                    ) from None

    return wrapper


def spawn_new_process_for_each_test(f: Callable[_P, None]) -> Callable[_P, None]:
    """Decorator to spawn a new process for each test function."""

    @functools.wraps(f)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> None:
        if os.environ.get("RUNNING_IN_SUBPROCESS") == "1":
            return f(*args, **kwargs)

        import torch.multiprocessing as mp

        with suppress(RuntimeError):
            mp.set_start_method("spawn")

        module_name = f.__module__
        env = os.environ.copy()
        env["RUNNING_IN_SUBPROCESS"] = "1"

        with tempfile.TemporaryDirectory() as tempdir:
            output_filepath = os.path.join(tempdir, "new_process.tmp")
            input_bytes = cloudpickle.dumps((f, output_filepath))
            cmd = [sys.executable, "-m", f"{module_name}"]
            returned = subprocess.run(cmd, input=input_bytes, capture_output=True, env=env)
            try:
                returned.check_returncode()
            except Exception as e:
                raise RuntimeError(f"Error raised in subprocess:\n{returned.stderr.decode()}") from e

    return wrapper


def create_new_process_for_each_test(
    method: Literal["spawn", "fork"] | None = None,
) -> Callable[[Callable[_P, None]], Callable[_P, None]]:
    """Creates a decorator that runs each test function in a new process."""
    if method is None:
        use_spawn = current_platform.is_xpu()
        method = "spawn" if use_spawn else "fork"
    assert method in ["spawn", "fork"], "Method must be either 'spawn' or 'fork'"
    if method == "fork":
        return fork_new_process_for_each_test
    return spawn_new_process_for_each_test
