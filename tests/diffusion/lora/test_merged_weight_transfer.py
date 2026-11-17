# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from vllm.config import VllmConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.ipc_engine import IPCWeightTransferEngine

from tests.diffusion.lora.test_merged_manager import _request
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.lora.weight_transfer import MergedLoRAIPCWeightTransferEngine

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
pytest_plugins = ("tests.diffusion.lora.test_merged_manager",)


def _engine(model, monkeypatch):
    manager = DiffusionLoRAManager(model, torch.device("cpu"), torch.bfloat16, lora_application="merged")
    engine = MergedLoRAIPCWeightTransferEngine(
        WeightTransferConfig(backend="omni_lora_ipc"), VllmConfig(), torch.device("cpu"), model
    )
    engine.lora_manager = manager
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    return engine


def _payload(request, names):
    return {
        "names": names,
        "dtype_names": ["float32"] * len(names),
        "shapes": [list(request.lora_tensors[name].shape) for name in names],
        "ipc_handles": [{} for _ in names],
        "peft_config": request.peft_config,
        "lora_int_id": request.lora_int_id,
    }


def test_chunked_session_owns_received_tensors_and_only_commits_at_finish(pipeline, monkeypatch):
    model, _, _ = pipeline
    engine = _engine(model, monkeypatch)
    request = _request()
    base = model.transformer.gate_up.weight.detach().clone()
    names = list(request.lora_tensors)

    def receive(receiver, info):
        receiver.model.load_weights((name, request.lora_tensors[name]) for name in info.names)

    monkeypatch.setattr(IPCWeightTransferEngine, "receive_weights", receive)
    engine.start_weight_update()
    for name in names:
        engine.update_weights(_payload(request, [name]))
        torch.testing.assert_close(model.transformer.gate_up.weight, base, rtol=0, atol=0)
        assert engine._staged.tensors[name].data_ptr() != request.lora_tensors[name].data_ptr()
    for tensor in request.lora_tensors.values():
        tensor.zero_()
    engine.finish_weight_update()
    assert not torch.equal(model.transformer.gate_up.weight, base)
    assert not engine._staged.tensors
    assert engine.model is model


def test_bad_session_preserves_policy_and_next_session_can_succeed(pipeline, monkeypatch):
    model, _, _ = pipeline
    engine = _engine(model, monkeypatch)
    request = _request()
    engine.lora_manager.merge_adapter(request)
    previous = model.transformer.gate_up.weight.detach().clone()

    def receive(receiver, info):
        receiver.model.load_weights((name, request.lora_tensors[name]) for name in info.names)

    monkeypatch.setattr(IPCWeightTransferEngine, "receive_weights", receive)
    engine.start_weight_update()
    engine.update_weights(_payload(request, [next(iter(request.lora_tensors))]))
    with pytest.raises(ValueError, match="Missing A/B"):
        engine.finish_weight_update()
    torch.testing.assert_close(model.transformer.gate_up.weight, previous, rtol=0, atol=0)
    engine.start_weight_update()
    engine.update_weights(_payload(request, list(request.lora_tensors)))
    engine.finish_weight_update()
    torch.testing.assert_close(model.transformer.gate_up.weight, previous, rtol=0, atol=0)


def test_duplicate_chunks_abort_without_mutating_policy(pipeline, monkeypatch):
    model, _, _ = pipeline
    engine = _engine(model, monkeypatch)
    request = _request()
    base = model.transformer.gate_up.weight.detach().clone()

    def receive(receiver, info):
        receiver.model.load_weights((name, request.lora_tensors[name]) for name in info.names)

    monkeypatch.setattr(IPCWeightTransferEngine, "receive_weights", receive)
    engine.start_weight_update()
    payload = _payload(request, list(request.lora_tensors))
    engine.update_weights(payload)
    with pytest.raises(ValueError, match="Duplicate LoRA"):
        engine.update_weights(payload)
    assert not engine._staged.tensors
    torch.testing.assert_close(model.transformer.gate_up.weight, base, rtol=0, atol=0)


def test_trainer_disconnect_preserves_policy_and_allows_another_session(pipeline, monkeypatch):
    model, _, _ = pipeline
    engine = _engine(model, monkeypatch)
    request = _request()
    engine.lora_manager.merge_adapter(request)
    previous = model.transformer.gate_up.weight.detach().clone()

    def receive(receiver, info):
        receiver.model.load_weights((name, request.lora_tensors[name]) for name in info.names)

    monkeypatch.setattr(IPCWeightTransferEngine, "receive_weights", receive)
    engine.shutdown()
    torch.testing.assert_close(model.transformer.gate_up.weight, previous, rtol=0, atol=0)
    assert engine.lora_manager.list_adapters() == [1]
    engine.start_weight_update()
    engine.update_weights(_payload(request, list(request.lora_tensors)))
    engine.finish_weight_update()
    torch.testing.assert_close(model.transformer.gate_up.weight, previous, rtol=0, atol=0)
