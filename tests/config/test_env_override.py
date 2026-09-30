# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_IMPORT_SCRIPT = """
import sys
import types
import torch._dynamo.config as dynamo_config

# Isolate package startup from unrelated vLLM patches and model registration.
version = types.ModuleType("vllm_omni.version")
version.__version__ = "dev"
version.__version_tuple__ = (0, 0, "dev")
sys.modules[version.__name__] = version
for name in ("vllm_omni.patch", "vllm_omni.transformers_utils.configs",
             "vllm_omni.transformers_utils.parsers"):
    sys.modules[name] = types.ModuleType(name)
config = types.ModuleType("vllm_omni.config")
config.OmniModelConfig = object
sys.modules[config.__name__] = config

dynamo_config.recompile_limit = 11
import vllm_omni
assert dynamo_config.recompile_limit == int(sys.argv[1])
"""


@pytest.mark.parametrize("value", [None, "2", "64", "0", "-1", "not-an-int"])
def test_package_import_configures_recompile_limit(value: str | None) -> None:
    env = os.environ.copy()
    name = "VLLM_OMNI_TORCH_DYNAMO_RECOMPILE_LIMIT"
    env.pop(name, None)
    if value is not None:
        env[name] = value
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_SCRIPT, value or "11"],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    if value in {"0", "-1", "not-an-int"}:
        assert result.returncode != 0
        assert f"ValueError: {name} must be" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
