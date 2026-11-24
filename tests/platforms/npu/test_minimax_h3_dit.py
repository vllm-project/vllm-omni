# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import runpy
import sys
from types import SimpleNamespace

import pytest
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    ("soc", "dtype", "registered", "binary", "expected"),
    [
        pytest.param("ascend950", "bfloat16", True, True, True, id="active-950-bf16"),
        pytest.param("ascend910b", "bfloat16", True, True, False, id="910b-only"),
        pytest.param("ascend950", "float16", True, True, False, id="fp16-only"),
        pytest.param("ascend950", "bfloat16", True, False, False, id="missing-binary"),
        pytest.param("ascend950", "bfloat16", False, True, False, id="inactive-vendor"),
    ],
)
def test_a5_qkv_requires_registered_bf16_kernel(tmp_path, monkeypatch, soc, dtype, registered, binary, expected):
    from vllm_omni.platforms.npu.models.minimax_h3_dit import _supports_eagle_qkv_on_a5

    sdk = tmp_path / "mindiesd"
    sdk.mkdir()
    (sdk / "__init__.py").touch()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "mindiesd", raising=False)

    vendor = sdk / "ops/vendors/aie_ascendc"
    kernels = vendor / "op_impl/ai_core/tbe/kernel"
    descriptor = kernels / soc / "eagle_split_norm_rope_concat/EagleSplitNormRopeConcat_0.json"
    descriptor.parent.mkdir(parents=True)
    descriptor.write_text("{}", encoding="utf-8")
    if binary:
        descriptor.with_suffix(".o").touch()
    manifest = kernels / "config" / soc / "eagle_split_norm_rope_concat.json"
    manifest.parent.mkdir(parents=True)
    entry = {"inputs": [{"dtype": dtype}], "binInfo": {"jsonFilePath": descriptor.relative_to(kernels).as_posix()}}
    manifest.write_text(json.dumps({"binList": [entry]}), encoding="utf-8")
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", str(vendor if registered else tmp_path / "other_vendor"))

    assert _supports_eagle_qkv_on_a5() is expected


def test_npu_platform_installs_h3_dit_fusion(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_blocks import MiniMaxH3Attention
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.platforms.npu.models import minimax_h3_dit

    # Load the real hook without importing the vllm-ascend plugin on CPU.
    monkeypatch.setitem(sys.modules, "vllm_ascend.platform", SimpleNamespace(NPUPlatform=object))
    platform = runpy.run_module("vllm_omni.platforms.npu.platform")["NPUOmniPlatform"]
    monkeypatch.setattr(minimax_h3_dit, "_ascend_device_name", lambda: "A2")
    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    nn.Module.__init__(pipeline)
    attention = MiniMaxH3Attention.__new__(MiniMaxH3Attention)
    nn.Module.__init__(attention)
    pipeline.transformer = nn.Sequential(attention)

    platform.apply_diffusion_model_patches(pipeline, None)

    assert attention._split_norm_rope.__func__ is minimax_h3_dit._npu_split_norm_rope
