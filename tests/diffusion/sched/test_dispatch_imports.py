# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Keep head-side dispatch imports independent of backend execution imports."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]
REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_isolated(script: str) -> None:
    # Skip only the application bootstrap; load the real diffusion packages.
    bootstrap = """
import sys
import types
from pathlib import Path

package = types.ModuleType("vllm_omni")
package.__path__ = [str(Path.cwd() / "vllm_omni")]
sys.modules[package.__name__] = package
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(bootstrap) + textwrap.dedent(script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_dispatch_imports_do_not_load_backend_dependencies() -> None:
    _run_isolated(
        """
from vllm_omni.diffusion.sched.dispatch.config import CentralSchedulingConfig
from vllm_omni.diffusion.sched.dispatch.controller import CentralScheduler

assert CentralSchedulingConfig.__module__.endswith("dispatch.config")
assert CentralScheduler.__module__.endswith("dispatch.controller")
for prefix in (
    "torch",
    "vllm",
    "vllm_omni.diffusion.worker",
    "vllm_omni.diffusion.sched.base_scheduler",
    "vllm_omni.diffusion.sched.interface",
    "vllm_omni.diffusion.sched.request_scheduler",
    "vllm_omni.diffusion.sched.step_scheduler",
    "vllm_omni.diffusion.sched.sigma_schedule",
):
    loaded = [name for name in sys.modules if name == prefix or name.startswith(prefix + ".")]
    assert not loaded, loaded
"""
    )


def test_scheduler_public_exports_keep_their_original_identity() -> None:
    _run_isolated(
        """
import vllm_omni.diffusion.sched as sched

# Only backend leaf modules are stand-ins; exercise the real package exports.
exports = {
    "base_scheduler": ("BaseScheduler", "SchedulerInterface"),
    "interface": (
        "CachedRequestData", "DiffusionRequestStatus", "DiffusionSchedulerOutput",
        "KVPrefetchJob", "NewRequestData", "SchedulerRequestState", "StepBatchSamplingParamsKey",
    ),
    "request_scheduler": ("RequestScheduler",),
    "sigma_schedule": ("BASE_SCHEDULE_KEY", "DMD2SigmaSchedule"),
    "step_scheduler": ("StepScheduler",),
}
expected = {}
for module_name, names in exports.items():
    module = types.ModuleType(f"{sched.__name__}.{module_name}")
    sys.modules[module.__name__] = module
    for name in names:
        value = object()
        setattr(module, name, value)
        expected[name] = value
expected["Scheduler"] = expected["RequestScheduler"]

assert set(sched.__all__) == set(expected)
assert set(expected) <= set(dir(sched))
for name, value in expected.items():
    assert name not in vars(sched)
    assert getattr(sched, name) is value, name
    assert vars(sched)[name] is value, name
assert sched.Scheduler is sched.RequestScheduler
assert not hasattr(sched, "missing_scheduler_export")

namespace = {}
exec("from vllm_omni.diffusion.sched import *", namespace)
assert all(namespace[name] is value for name, value in expected.items())
"""
    )
