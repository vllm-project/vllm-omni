# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]
ROOT = Path(__file__).parents[4] / "vllm_omni/platforms/npu/models/lingbot_world"


def load(name):
    spec = importlib.util.spec_from_file_location("_lingbot_" + name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_online_mxfp8_selects_only_attention_and_ffn(monkeypatch):
    module = load("quantization")
    provider = ModuleType("vllm_omni.quantization.mxfp8_config")
    seen = []
    setattr(provider, "DiffusionMXFP8Config", lambda **kw: seen.append(kw) or kw)
    monkeypatch.setitem(sys.modules, provider.__name__, provider)
    result = module.online_mxfp8_config("mxfp8", None, num_layers=40)
    assert result["is_checkpoint_mxfp8_serialized"] is False
    ignored = result["ignored_layers"]
    assert len(ignored) == 82 and len(set(ignored)) == 82
    assert "transformer.blocks.39.cam_injector_layer2" in ignored
    assert all("self_attn" not in x and "cross_attn" not in x and ".ffn" not in x for x in ignored)
    existing = object()
    assert module.online_mxfp8_config(None, existing, num_layers=40) is existing
    with pytest.raises(ValueError, match="not both"):
        module.online_mxfp8_config("mxfp8", existing, num_layers=40)


@pytest.mark.parametrize("mode", ["int8", "fp8", True, ""])
def test_online_quant_rejects_non_mxfp8(mode):
    with pytest.raises(ValueError, match="mxfp8"):
        load("quantization").online_mxfp8_config(mode, None, num_layers=40)
