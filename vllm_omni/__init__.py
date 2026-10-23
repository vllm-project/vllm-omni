# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
vLLM-Omni: Multi-modality models inference and serving with
non-autoregressive structures.

This package extends vLLM beyond traditional text-based, autoregressive
generation to support multi-modality models with non-autoregressive
structures and non-textual outputs.

Architecture:
- 🟡 Modified: vLLM components modified for multimodal support
- 🔴 Added: New components for multimodal and non-autoregressive
  processing
"""

import atexit
import os
import sys

# We import version early, because it will warn if vLLM / vLLM Omni
# are not using the same major + minor version (if vLLM is installed).
# We should do this before applying patch, because vLLM imports might
# throw in patch if the versions differ.
from .version import __version__, __version_tuple__  # isort:skip # noqa: F401

try:
    from . import patch  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    if exc.name != "vllm":
        raise
    # Allow importing vllm_omni without vllm (e.g., documentation builds)
    patch = None  # type: ignore

# Register custom configs (AutoConfig, AutoTokenizer) as early as possible.
try:
    from vllm_omni.transformers_utils import configs as _configs  # noqa: F401, E402
    from vllm_omni.transformers_utils import parsers as _parsers  # noqa: F401, E402
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    if exc.name != "vllm":
        raise
    # Allow importing vllm_omni without vllm (e.g., documentation builds)
    pass

from .config import OmniModelConfig

# Apply the process-wide default at vllm_omni import; later backend policies
# take precedence.
from . import env_override as _env_override  # isort:skip # noqa: F401, E402


def __getattr__(name: str):
    # Lazy import for AsyncOmni and Omni to avoid pulling in heavy
    # dependencies (vllm model_loader → fused_moe → pynvml) at package
    # import time.  This prevents crashes in lightweight subprocesses
    # (e.g. model-architecture inspection) that lack a CUDA context.
    # See: https://github.com/vllm-project/vllm-omni/issues/1793
    if name == "AsyncOmni":
        from .entrypoints.async_omni import AsyncOmni

        return AsyncOmni
    if name == "Omni":
        from .entrypoints.omni import Omni

        return Omni
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "__version_tuple__",
    # Main components
    "Omni",
    "AsyncOmni",
    # Configuration
    "OmniModelConfig",
    # All other components are available through their respective modules
    # processors.*, schedulers.*, executors.*, etc.
]


def _install_registry_exit_bypass() -> None:
    """Bypass interpreter finalization in vLLM's model-inspection subprocess.

    vLLM 0.32 spawns ``python3 -m vllm.model_executor.models.registry`` to
    inspect model architectures; that child imports ``vllm_omni`` through
    ``load_general_plugins()``. On NPU the child's interpreter aborts while
    finalizing (SIGABRT, "corrupted size vs. prev_size", exit code 134), and
    vLLM 0.32 treats that exit code as a hard failure ("Model architectures
    ... failed to be inspected"), refusing to start the engine.

    The child's whole job is ``_run()`` in
    ``vllm/model_executor/models/registry.py``::

        def _run() -> None:
            load_general_plugins()                 # imports vllm_omni (here)
            fn, output_file = pickle.loads(sys.stdin.buffer.read())
            result = fn()                          # architecture check
            with open(output_file, "wb") as f:     # registry serialization
                f.write(pickle.dumps(result))

    ``atexit`` handlers run only after ``_run()`` has either raised or
    returned, which gives exactly two exit paths:

    * **success** -- ``_run()`` returned, so the check passed and the registry
      is already serialized and closed on disk. The handler calls
      :func:`os._exit` with status 0, skipping the finalization that aborts.
      stdout/stderr are flushed first so whatever the child printed survives.
    * **failure** -- ``_run()`` raised, or the child asked for a non-zero
      status. The handler returns without interfering and the interpreter
      exits normally: non-zero status plus the original traceback on stderr,
      which vLLM re-raises as a ``RuntimeError``. Swallowing that status here
      would hide the real error behind a confusing "output file missing"
      failure in the parent.
    """
    failed = False

    original_excepthook = sys.excepthook

    def _excepthook(exc_type, exc_value, traceback):
        nonlocal failed
        failed = True
        original_excepthook(exc_type, exc_value, traceback)

    sys.excepthook = _excepthook

    # ``SystemExit`` is handled before ``sys.excepthook`` runs, so an explicit
    # non-zero ``sys.exit(...)`` has to be recorded separately.
    original_exit = sys.exit

    def _exit(code=None):
        nonlocal failed
        if code not in (None, 0):
            failed = True
        original_exit(code)

    sys.exit = _exit

    def _bypass_finalization():
        if failed:
            return
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:  # pragma: no cover - already closed/broken
                pass
        os._exit(0)

    atexit.register(_bypass_finalization)


if sys.argv and sys.argv[0].endswith("models/registry.py"):
    try:
        import torch as _torch

        _on_npu = hasattr(_torch, "npu") and _torch.npu.is_available()
    except Exception:  # pragma: no cover - torch is always importable here
        _on_npu = False
    if _on_npu:
        _install_registry_exit_bypass()
