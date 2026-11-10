# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Execute the NPU forward adapter without importing its device-only bases."""

import ast
from enum import Enum
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Mode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


class _Output:
    pass


def _forward(context):
    source = Path(__file__).resolve().parents[3] / "vllm_omni/platforms/npu/worker/npu_model_runner.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OmniNPUModelRunner")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_model_forward")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    namespace = {
        "partial": partial,
        "get_forward_context": lambda: context,
        "CUDAGraphMode": _Mode,
        "OmniOutput": _Output,
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["_model_forward"]


@pytest.mark.parametrize("mode", list(_Mode))
@pytest.mark.parametrize("enable_enpu", [False, True])
def test_full_forward_supplies_backend_before_replay(mode, enable_enpu):
    events = []
    backend = object()
    result = _Output()
    context = SimpleNamespace(cudagraph_runtime_mode=mode)

    class GraphWrapper:
        attn_backend = None

        def set_attn_backend(self, value):
            self.attn_backend = value
            events.append("backend")

        def __call__(self, **kwargs):
            # Without the backend, Ascend's use_updatable_graph returns False
            # and the wrapper cannot update dynamic attention tasks on replay.
            if mode == _Mode.FULL:
                assert self.attn_backend is backend
            assert kwargs["inputs_embeds"] is embeds
            assert kwargs["runtime_additional_information"] == {"request_id": "new"}
            events.append("replay")
            return result

    embeds = object()
    runner = SimpleNamespace(
        model=GraphWrapper(),
        attn_backend=backend,
        enable_enpu=enable_enpu,
        _build_model_kwargs_extra=lambda: {"runtime_additional_information": {"request_id": "new"}},
        _update_full_graph_params_if_needed=lambda ctx, n: events.append("update"),
    )
    assert _forward(context)(runner, 4, inputs_embeds=embeds) is result
    assert runner._omni_last_model_output is result
    expected = ["update", "replay"] if enable_enpu else ["replay", "update"]
    assert events == (["backend"] if mode == _Mode.FULL else []) + expected
