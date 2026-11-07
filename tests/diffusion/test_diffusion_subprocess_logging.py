# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

MARKER = "subprocess-logging-test"

# Runs in a fresh interpreter, like a ``spawn`` child: only the inherited env carries logging config.
# Plugin loading and the stage/worker constructors are stubbed; logging setup and emission are real.
CHILD = """
import sys
from types import SimpleNamespace

import vllm_omni.plugins


class Initialized(Exception):
    pass


def stub_init(self, *args, **kwargs):
    module_logger.info("reached initialization")
    raise Initialized


vllm_omni.plugins.load_omni_general_plugins = lambda: None
if sys.argv[1] == "stage":
    import vllm_omni.diffusion.stage_diffusion_proc as module
    from vllm_omni.diffusion.stage_diffusion_proc import StageDiffusionProc as proc_cls

    def run():
        proc_cls.run_diffusion_proc("stub-model", None, "ipc://unused", local_client=True, headless=False)
else:
    import vllm_omni.diffusion.worker.diffusion_worker as module
    from vllm_omni.diffusion.worker.diffusion_worker import WorkerProc as proc_cls

    parallel = SimpleNamespace(enable_expert_parallel=False, use_hsdp=False, hsdp_replicate_size=1)

    def run():
        proc_cls.worker_main(0, SimpleNamespace(parallel_config=parallel), None, None, None)

module_logger = module.logger
proc_cls.__init__ = stub_init
try:
    run()
except Initialized:
    sys.exit(0)
sys.exit("entrypoint returned without reaching initialization")
"""


@pytest.mark.parametrize(
    ("entrypoint", "logger_name"),
    [
        ("stage", "vllm_omni.diffusion.stage_diffusion_proc"),
        ("worker", "vllm_omni.diffusion.worker.diffusion_worker"),
    ],
)
def test_spawned_entrypoint_applies_inherited_logging_config(tmp_path, entrypoint, logger_name):
    """StageDiffusionProc.run_diffusion_proc / WorkerProc.worker_main must apply the inherited VLLM_* logging
    config before initialization; otherwise their INFO records are dropped in the child."""
    config_file = tmp_path / "logging.json"
    config_file.write_text(
        json.dumps(
            {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {"f": {"format": f"%(levelname)s [{MARKER}] %(name)s: %(message)s"}},
                "handlers": {"h": {"class": "logging.StreamHandler", "formatter": "f", "stream": "ext://sys.stderr"}},
                "loggers": {"vllm": {"handlers": ["h"], "level": "INFO", "propagate": False}},
            }
        )
    )
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(sys.path),
        "VLLM_CONFIGURE_LOGGING": "1",
        "VLLM_LOGGING_LEVEL": "INFO",
        "VLLM_LOGGING_CONFIG_PATH": str(config_file),
    }

    result = subprocess.run(
        [sys.executable, "-c", CHILD, entrypoint], env=env, capture_output=True, text=True, timeout=300
    )

    assert result.returncode == 0, result.stderr[-4000:]
    assert f"INFO [{MARKER}] {logger_name}: reached initialization" in result.stderr, result.stderr[-4000:]
