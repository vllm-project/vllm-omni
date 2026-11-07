# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise the production NPU load hook with hardware dependencies replaced."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_omni.model_executor.mm_request_rng import RandomDrawSpec, RequestRNGProvider

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _RNGModel(torch.nn.Module):
    supports_mm_request_rng = True
    mm_rng_seed_field = "vae_generator_seed"

    def bind_mm_rng_provider(self, provider: RequestRNGProvider) -> None:
        self.provider = provider

    def advance_mm_rng(self, generator: torch.Generator, spec: RandomDrawSpec) -> None:
        raise AssertionError("Loading a model must not consume a request's random stream")


@pytest.mark.parametrize("is_310p", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_npu_load_model_binds_request_rng(monkeypatch, mocker, is_310p, enabled):
    model = _RNGModel() if enabled else torch.nn.Identity()

    class HardwareRunner(GPUModelRunner):
        def load_model(self):
            self.model = model

    # Import the actual Omni class and execute its complete load_model method.
    # Only Ascend-specific imports, patching and device setup are substituted.
    hardware_modules = {
        "vllm_ascend.ascend_forward_context": {
            "get_forward_context": mocker.Mock(),
            "set_ascend_forward_context": mocker.Mock(),
        },
        "vllm_ascend.attention.attention_v1": {"AscendAttentionState": mocker.Mock()},
        "vllm_ascend.attention.utils": {"using_paged_attention": mocker.Mock()},
        "vllm_ascend.ops.rotary_embedding": {"update_cos_sin": mocker.Mock()},
        "vllm_ascend.utils": {"enable_sp": mocker.Mock(), "lmhead_tp_enable": mocker.Mock()},
        "vllm_ascend.worker.model_runner_v1": {
            "SEQ_LEN_WITH_MAX_PA_WORKSPACE": 0,
            "NPUModelRunner": HardwareRunner,
        },
        "vllm_ascend._310p.model_runner_310p": {"NPUModelRunner310": HardwareRunner},
        "vllm_omni.platforms.npu._310p": {"is_310p": lambda: is_310p},
        "vllm_omni.platforms.npu._310p.patch": {"apply_model_patches": mocker.Mock()},
        "vllm_omni.platforms.npu.native_rejection_sampler": {"restore_native_rejection_sampler": mocker.Mock()},
    }
    for name, attrs in hardware_modules.items():
        hardware_module = ModuleType(name)
        hardware_module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, hardware_module)
    source = Path(__file__).resolve().parents[3] / "vllm_omni/platforms/npu/worker/npu_model_runner.py"
    spec = importlib.util.spec_from_file_location("_npu_request_rng_wiring", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runner = object.__new__(module.OmniNPUModelRunner)
    runner.vllm_config = runner.model_config = None
    for hook in (
        "_maybe_enable_output_token_ids_for_model_sampler",
        "_init_talker_mtp",
        "_prewarm_attention_capture_workspaces",
    ):
        mocker.patch.object(runner, hook)

    runner.load_model()

    if enabled:
        assert isinstance(runner._mm_rng_provider, RequestRNGProvider)
        assert model.provider is runner._mm_rng_provider
        assert runner._mm_rng_provider.streams == {}
    else:
        assert runner._mm_rng_provider is None
