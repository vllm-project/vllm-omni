# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Check the installed Diffusers loader's revision and precision policy on CPU."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(scope="module")
def loader():
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        pytest.skip("Run in the pinned SGLang comparison environment")
    assert spec is not None and spec.origin is not None
    source = Path(spec.origin).parent / "multimodal_gen/runtime/pipelines/diffusers_pipeline.py"
    tree = ast.parse(source.read_text())
    pipeline = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DiffusersPipeline")
    return next(n for n in pipeline.body if isinstance(n, ast.FunctionDef) and n.name == "_load_diffusers_pipeline")


def test_revision_reaches_snapshot_resolution(loader):
    assignment = next(
        n
        for n in loader.body
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "maybe_download_model"
    )
    seen: dict[str, Any] = {}

    def resolve(model, **kwargs):
        seen.update(model=model, **kwargs)
        return "/cached/pinned-snapshot"

    namespace = {
        "model_path": "model/repo",
        "server_args": SimpleNamespace(revision="pinned-sha"),
        "maybe_download_model": resolve,
    }
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "diffusers_loader", "exec"), namespace)
    assert seen == {"model": "model/repo", "force_diffusers_model": True, "revision": "pinned-sha"}
    assert namespace["model_path"] == "/cached/pinned-snapshot"


def test_diffusers_components_share_the_requested_loading_dtype(loader):
    assignment = next(
        n
        for n in loader.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "load_kwargs" for t in n.targets)
    )
    namespace: dict[str, Any] = {
        "dtype": "bf16",
        "server_args": SimpleNamespace(trust_remote_code=False, revision="pinned-sha"),
    }
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "diffusers_loader", "exec"), namespace)
    assert namespace["load_kwargs"]["torch_dtype"] == "bf16"
