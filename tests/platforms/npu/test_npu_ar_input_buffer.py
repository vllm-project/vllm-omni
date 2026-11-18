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
@pytest.mark.parametrize("model_type", ["hunyuan_image_3_moe", "other_model"])
def test_input_embedding_buffer_uses_model_input_width(monkeypatch, input_width, model_type):
    source = Path(__file__).resolve().parents[3] / "vllm_omni/platforms/npu/worker/npu_ar_model_runner.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUARModelRunner")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    cls.bases = [ast.Name(id="Base", ctx=ast.Load())]
    cls.body = [init]
    module = ast.Module(body=[cls], type_ignores=[])

    attention = ModuleType("vllm_omni.platforms.npu.attention")
    install_calls = []
    attention.static_shape_decode = SimpleNamespace(install_into_ascend_backend=lambda: install_calls.append(True))
    monkeypatch.setitem(sys.modules, attention.__name__, attention)

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type=model_type),
            hf_text_config=SimpleNamespace(hidden_size=896),
            get_inputs_embeds_size=lambda: input_width,
        )
    )
    device = object()

    class Base:
        def __init__(self, vllm_config, device):
            self.model_config = vllm_config.model_config
            self.max_num_tokens = 16
            self.dtype = "model-dtype"
            self.vllm_config = vllm_config
            self.device = device

        def _make_buffer(self, *shape, **kwargs):
            return (shape, kwargs)

        def _init_duplex_sampling_state(self):
            pass

    namespace = {
        "Base": Base,
        "VllmConfig": SimpleNamespace,
        "torch": SimpleNamespace(int32="int32", device=object),
        "OmniKVTransferManager": SimpleNamespace(from_vllm_config=lambda *args: None),
        "needs_omni_connector": lambda config: False,
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    runner = namespace["NPUARModelRunner"](vllm_config, device)
    assert runner.vllm_config is vllm_config
    assert runner.device is device
    static_decode_enabled = model_type != "hunyuan_image_3_moe"
    assert runner._omni_static_decode_enabled is static_decode_enabled
    assert install_calls == ([True] if static_decode_enabled else [])
    assert runner.hidden_size == 896
    assert runner.inputs_embeds == ((16, input_width), {"dtype": "model-dtype", "numpy": False})
