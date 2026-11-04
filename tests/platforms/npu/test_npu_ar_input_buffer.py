# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise NPU AR buffer allocation without importing device-only dependencies."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("input_width", [896, 3584])
def test_input_embedding_buffer_uses_model_input_width(monkeypatch, input_width):
    source = Path(__file__).resolve().parents[3] / "vllm_omni/platforms/npu/worker/npu_ar_model_runner.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUARModelRunner")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    cls.bases = [ast.Name(id="Base", ctx=ast.Load())]
    cls.body = [init]
    module = ast.Module(body=[cls], type_ignores=[])

    attention = ModuleType("vllm_omni.platforms.npu.attention")
    attention.static_shape_decode = SimpleNamespace(install_into_ascend_backend=lambda: None)
    monkeypatch.setitem(sys.modules, attention.__name__, attention)

    class Base:
        def __init__(self):
            self.model_config = SimpleNamespace(
                hf_text_config=SimpleNamespace(hidden_size=896),
                get_inputs_embeds_size=lambda: input_width,
            )
            self.max_num_tokens = 16
            self.dtype = "model-dtype"
            self.vllm_config = object()

        def _make_buffer(self, *shape, **kwargs):
            return (shape, kwargs)

        def _init_duplex_sampling_state(self):
            pass

    namespace = {
        "Base": Base,
        "torch": SimpleNamespace(int32="int32"),
        "OmniKVTransferManager": SimpleNamespace(from_vllm_config=lambda *args: None),
        "needs_omni_connector": lambda config: False,
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    runner = namespace["NPUARModelRunner"]()
    assert runner.hidden_size == 896
    assert runner.inputs_embeds == ((16, input_width), {"dtype": "model-dtype", "numpy": False})
