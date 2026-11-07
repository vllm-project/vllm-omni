# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise GPU/NPU load hooks with hardware initialization replaced."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_omni.model_executor.mm_request_rng import RandomDrawSpec, RequestRNGProvider
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(params=["gpu", "npu", "npu310"])
def runner_class(request, monkeypatch, mocker):
    if request.param == "gpu":
        return OmniGPUModelRunner

    class HardwareRunner(GPUModelRunner):
        pass

    modules = {
        "vllm_ascend.ascend_forward_context": {
            "get_forward_context": mocker.Mock(),
            "set_ascend_forward_context": mocker.Mock(),
        },
        "vllm_ascend.attention.attention_v1": {"AscendAttentionState": mocker.Mock()},
        "vllm_ascend.attention.utils": {"using_paged_attention": mocker.Mock()},
        "vllm_ascend.ops.rotary_embedding": {"update_cos_sin": mocker.Mock()},
        "vllm_ascend.utils": {"enable_sp": mocker.Mock(), "lmhead_tp_enable": mocker.Mock()},
        "vllm_ascend.worker.v1.model_runner": {
            "SEQ_LEN_WITH_MAX_PA_WORKSPACE": 0,
            "NPUModelRunner": HardwareRunner,
        },
        "vllm_ascend.worker.v1._310p.model_runner_310p": {"NPUModelRunner310": HardwareRunner},
        "vllm_omni.platforms.npu._310p": {"is_310p": lambda: request.param == "npu310"},
        "vllm_omni.platforms.npu._310p.patch": {"apply_model_patches": mocker.Mock()},
        "vllm_omni.platforms.npu.native_rejection_sampler": {"restore_native_rejection_sampler": mocker.Mock()},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    source = Path(__file__).resolve().parents[2] / "vllm_omni/platforms/npu/worker/npu_model_runner.py"
    spec = importlib.util.spec_from_file_location("_npu_request_rng_wiring", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OmniNPUModelRunner


@pytest.mark.parametrize("enabled", [False, True])
def test_load_model_binds_request_rng_only_for_opted_in_models(runner_class, mocker, enabled):
    model = torch.nn.Identity()
    model.supports_mm_request_rng = enabled
    model.bind_mm_rng_provider = mocker.Mock()
    model.advance_mm_rng = mocker.Mock(side_effect=AssertionError("Loading must not draw request RNG"))
    mocker.patch.object(GPUModelRunner, "load_model", lambda instance: setattr(instance, "model", model))
    runner = object.__new__(runner_class)
    runner.vllm_config = runner.model_config = None
    for hook in (
        "_maybe_enable_output_token_ids_for_model_sampler",
        "_init_talker_mtp",
        "_prewarm_attention_capture_workspaces",
        "_report_model_local_kv",
        "_warn_unexposed_stage_hooks",
    ):
        mocker.patch.object(runner, hook)
    runner.load_model()
    if enabled:
        assert isinstance(runner._mm_rng_provider, RequestRNGProvider)
        model.bind_mm_rng_provider.assert_called_once_with(runner._mm_rng_provider)
        assert not runner._mm_rng_provider.streams
    else:
        assert runner._mm_rng_provider is None
        model.bind_mm_rng_provider.assert_not_called()
    model.advance_mm_rng.assert_not_called()
    provider = runner._mm_rng_provider
    if provider is not None:
        spec = RandomDrawSpec((1,), torch.float32, torch.device("cpu"))
        for req_id in ("A", "B"):
            provider.register(req_id, 42, {0: spec})
    mocker.patch.object(runner, "_shutdown_omni_prefix_cache")
    mocker.patch.object(GPUModelRunner, "shutdown")
    runner.shutdown()
    runner.shutdown()
    if provider is not None:
        assert not provider.handles and not provider.streams
