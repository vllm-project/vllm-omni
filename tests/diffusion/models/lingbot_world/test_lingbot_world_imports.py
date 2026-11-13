# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Keep model-local output helpers usable from shared decoder/collective code."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PACKAGE_ROOT = _REPO_ROOT / "vllm_omni/diffusion/models/lingbot_world"


def test_output_helpers_do_not_import_pipeline_or_transformer():
    # A fresh interpreter exercises Python's real parent-package imports. A
    # source-only helper load would miss an eager package __init__ cycle.
    script = """
import importlib
import sys

package = "vllm_omni.diffusion.models.lingbot_world"
for helper in ("failure", "pixels", "pixel_output", "leased_post_attention"):
    importlib.import_module(f"{package}.{helper}")
    for heavy in ("actions", "pipeline", "transformer"):
        assert f"{package}.{heavy}" not in sys.modules, (helper, heavy)
"""
    environment = dict(os.environ, PYTHONPATH=str(_REPO_ROOT), TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_package_exports_keep_existing_names(monkeypatch):
    package_name = "_lingbot_world_exports_test"
    spec = importlib.util.spec_from_file_location(
        package_name, _PACKAGE_ROOT / "__init__.py", submodule_search_locations=[str(_PACKAGE_ROOT)]
    )
    assert spec is not None and spec.loader is not None
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, package_name, package)
    spec.loader.exec_module(package)

    exports = {
        "CausalLingBotWorldTransformer3DModel": "transformer",
        "LingBotCameraControlReducer": "actions",
        "LingBotWorldCausalDMDPipeline": "pipeline",
        "get_lingbot_world_post_process_func": "pipeline",
        "get_lingbot_world_pre_process_func": "pipeline",
    }
    sentinels = {name: object() for name in exports}
    for module_name in set(exports.values()):
        module = ModuleType(f"{package_name}.{module_name}")
        for name, provider in exports.items():
            if provider == module_name:
                setattr(module, name, sentinels[name])
        monkeypatch.setitem(sys.modules, module.__name__, module)

    assert package.__all__ == list(exports)
    for name, expected in sentinels.items():
        assert getattr(package, name) is expected
    with pytest.raises(AttributeError, match="no attribute 'unknown_component'"):
        getattr(package, "unknown_component")
