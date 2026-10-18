# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regressions for worker configuration and lifecycle input boundaries."""

from unittest.mock import Mock

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.lora.manager import LoRABackend
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("method", ["handle_sleep_task", "handle_wake_task"])
def test_malformed_lifecycle_task_returns_error_with_original_id(method, mocker):
    mocker.patch("torch.distributed.is_initialized", return_value=False)
    worker = object.__new__(DiffusionWorker)
    worker.rank = 0
    ack = getattr(worker, method)({"task_id": "bad-task", "unexpected": True})
    assert ack.status == "ERROR"
    assert ack.task_id == "bad-task"
    assert "unexpected" in ack.error_msg


@pytest.mark.parametrize("path,scale", [("adapter", 0.5), (["adapter"], [0.5])])
def test_peft_single_adapter_normalizes_scalar_and_list_config(path, scale, mocker):
    manager = mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.DiffusionLoRAManager")
    worker = object.__new__(DiffusionWorker)
    worker.od_config = OmniDiffusionConfig(lora_backend=LoRABackend.PEFT, lora_path=path, lora_scale=scale)
    worker.model_runner = Mock()
    worker.model_runner.pipeline.lora_is_fused = False
    worker.device = "cpu"
    worker.init_lora_manager()
    assert manager.call_args.kwargs["lora_path"] == "adapter"
    assert manager.call_args.kwargs["lora_scale"] == 0.5
    assert manager.call_args.kwargs["max_cached_adapters"] == 1


@pytest.mark.parametrize("path,scale", [(["a", "b"], 1.0), ("a", [0.5, 1.0])])
def test_dynamic_peft_rejects_multiple_static_adapters(path, scale, mocker):
    manager = mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.DiffusionLoRAManager")
    worker = object.__new__(DiffusionWorker)
    worker.od_config = OmniDiffusionConfig(lora_backend=LoRABackend.PEFT, lora_path=path, lora_scale=scale)
    worker.model_runner = Mock()
    worker.model_runner.pipeline.lora_is_fused = False
    worker.device = "cpu"
    with pytest.raises(ValueError, match="single adapter"):
        worker.init_lora_manager()
    manager.assert_not_called()
