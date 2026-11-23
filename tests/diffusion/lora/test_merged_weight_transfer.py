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


def test_allocator_guard_warns_once_per_worker_unless_expandable_segments_enabled(monkeypatch, caplog):
    from vllm_omni.lora.weight_transfer import (
        _reset_expandable_segments_warning,
        has_expandable_segments_enabled,
    )

    _reset_expandable_segments_warning()
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    assert not has_expandable_segments_enabled()

    model = torch.nn.Linear(2, 2)
    with caplog.at_level("WARNING"):
        MergedLoRAIPCWeightTransferEngine(
            WeightTransferConfig(backend="omni_lora_ipc"), VllmConfig(), torch.device("cuda", 0), model
        )
        MergedLoRAIPCWeightTransferEngine(
            WeightTransferConfig(backend="omni_lora_ipc"), VllmConfig(), torch.device("cuda", 0), model
        )
    warnings = [rec.message for rec in caplog.records if "expandable_segments:True" in rec.message]
    assert len(warnings) == 1
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in warnings[0]
    assert "fragmentation" in warnings[0]

    # Setting expandable_segments suppresses the warning.
    _reset_expandable_segments_warning()
    caplog.clear()
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")
    assert has_expandable_segments_enabled()
    with caplog.at_level("WARNING"):
        MergedLoRAIPCWeightTransferEngine(
            WeightTransferConfig(backend="omni_lora_ipc"), VllmConfig(), torch.device("cuda", 0), model
        )
    assert not [rec for rec in caplog.records if "expandable_segments:True" in rec.message]

    # PYTORCH_ALLOC_CONF is also recognized.
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:true")
    assert has_expandable_segments_enabled()


def test_failed_session_cleanup_logs_warning_with_exc_info_without_masking_root_error(pipeline, monkeypatch, caplog):
    model, _, _ = pipeline
    engine = _engine(model, monkeypatch)
    request = _request()

    def receive(receiver, info):
        receiver.model.load_weights((name, request.lora_tensors[name]) for name in info.names)

    monkeypatch.setattr(IPCWeightTransferEngine, "receive_weights", receive)
    monkeypatch.setattr(
        engine._packed_importer,
        "close",
        lambda: (_ for _ in ()).throw(RuntimeError("cleanup close boom")),
    )

    engine.start_weight_update()
    engine.update_weights(_payload(request, [next(iter(request.lora_tensors))]))
    with caplog.at_level("WARNING"):
        with pytest.raises(ValueError, match="Missing A/B"):
            engine.finish_weight_update()

    cleanup_logs = [rec for rec in caplog.records if "cleanup close boom" in rec.message]
    assert len(cleanup_logs) == 1
    assert cleanup_logs[0].exc_info is not None
    assert engine.model is model
    assert not engine._staged.tensors


def test_post_commit_cleanup_failure_logs_warning_and_keeps_committed_policy_usable(pipeline, monkeypatch, caplog):
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

    model, _, _ = pipeline
    engine = _engine(model, monkeypatch)
    request = _request()
    base = model.transformer.gate_up.weight.detach().clone()

    def receive(receiver, info):
        receiver.model.load_weights((name, request.lora_tensors[name]) for name in info.names)

    monkeypatch.setattr(IPCWeightTransferEngine, "receive_weights", receive)
    worker = object.__new__(DiffusionWorker)
    worker.weight_transfer_engine = engine
    worker._weight_update_active = False

    with caplog.at_level("WARNING"):
        worker.start_weight_update()
        worker.update_weights(_payload(request, list(request.lora_tensors)))

        # Arm cleanup failures right before finish_weight_update so they trigger
        # strictly after commit_merged_adapter succeeds.
        fail_once = {"close": True, "reset": True}
        orig_reset = engine.reset_weight_update_target

        def flaky_close():
            if fail_once["close"]:
                fail_once["close"] = False
                raise RuntimeError("post-commit close boom")

        def flaky_reset():
            orig_reset()
            if fail_once["reset"]:
                fail_once["reset"] = False
                raise RuntimeError("post-commit worker reset boom")

        monkeypatch.setattr(engine._packed_importer, "close", flaky_close)
        monkeypatch.setattr(engine, "reset_weight_update_target", flaky_reset)

        # Committed policy must not be falsely reported as aborted on cleanup failure.
        worker.finish_weight_update()

    assert worker._weight_update_active is False
    assert not torch.equal(model.transformer.gate_up.weight, base)
    assert any("post-commit close boom" in rec.message and rec.exc_info is not None for rec in caplog.records)
    assert any("post-commit worker reset boom" in rec.message and rec.exc_info is not None for rec in caplog.records)

    # Engine and worker remain usable for the next session.
    worker.start_weight_update()
    worker.update_weights(_payload(request, list(request.lora_tensors)))
    worker.finish_weight_update()
    assert worker._weight_update_active is False
