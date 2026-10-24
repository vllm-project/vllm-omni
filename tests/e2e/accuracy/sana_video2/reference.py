# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Load unchanged pinned upstream inference modules, excluding training imports.

The upstream package initializers eagerly import unrelated model families and
training datasets. Namespace packages bypass only those initializers. The sole
replacement module is builder: its MODELS is a real mmcv Registry, used only by
the model registration decorators. No attention, conditioning or solver math is
replaced. Run this in a dedicated process (it owns the `diffusion` namespace).
"""

import importlib
import subprocess
import sys
import types
from pathlib import Path

REVISION = "e93c883e10730ee5a4a6edf1cbcf501dc4ef753b"


def load_reference(root):
    from mmcv import Registry

    root = Path(root).resolve()
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        raise ValueError(f"Expected Sana@{REVISION}, got {actual}")
    subprocess.run(["git", "-C", str(root), "diff", "--exit-code", "HEAD"], check=True)
    for name in (
        "diffusion",
        "diffusion.model",
        "diffusion.model.nets",
        "diffusion.utils",
        "diffusion.scheduler",
        "diffusion.guiders",
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(root.joinpath(*name.split(".")))]
        sys.modules[name] = module
    builder = types.ModuleType("diffusion.model.builder")
    setattr(builder, "MODELS", Registry("models"))
    sys.modules[builder.__name__] = builder
    model = importlib.import_module("diffusion.model.nets.sana_video2")
    dpm = importlib.import_module("diffusion.scheduler.dpm_solver")
    euler = importlib.import_module("diffusion.scheduler.flow_euler_sampler")
    return model, dpm, euler
