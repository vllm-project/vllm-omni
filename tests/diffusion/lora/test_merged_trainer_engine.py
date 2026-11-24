# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU unit tests for the fused omni_lora_ipc trainer engine and wire codec."""

from __future__ import annotations

import socket
import weakref
from typing import Any

import msgspec
import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import VLLMWeightSyncClient
from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory
from vllm.distributed.weight_transfer.ipc_engine import IPCWeightTransferEngine
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear

from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.lora.ipc_wire import (
    IPC_WIRE_TAG,
    decode_ipc_handle_args,
    decode_ipc_handles,
    encode_ipc_handle_args,
    encode_ipc_handles,
)
from vllm_omni.lora.weight_transfer import (
    LoRAAdapterWeightSource,
    MergedLoRAIPCTrainerInitInfo,
    MergedLoRAIPCTrainerWeightTransferEngine,
    MergedLoRAIPCUpdateInfo,
    MergedLoRAIPCWeightTransferEngine,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _fake_cuda_ipc_args(
    tensor: torch.Tensor,
    *,
    tensor_cls: type = torch.Tensor,
    storage_cls: type = torch.UntypedStorage,
    storage_handle: bytes | None = b"ipc-handle-0123456789abcdef",
    ref_counter_handle: bytes | None = b"ref-counter-0123",
    event_handle: bytes | None = None,
) -> tuple[Any, ...]:
    return (
        tensor_cls,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        0,
        storage_cls,
        tensor.dtype,
        0,
        storage_handle,
        tensor.numel() * tensor.element_size(),
        0,
        False,
        ref_counter_handle,
        0,
        event_handle,
        False,
    )


@pytest.mark.parametrize("tensor_cls", [torch.Tensor, torch.nn.Parameter])
@pytest.mark.parametrize("storage_cls", [torch.UntypedStorage, torch.storage.TypedStorage])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_ipc_wire_codec_round_trips_through_msgspec_without_pickle(tensor_cls, storage_cls, dtype):
    sample = torch.zeros((4, 2), dtype=dtype)
    raw_args = _fake_cuda_ipc_args(sample, tensor_cls=tensor_cls, storage_cls=storage_cls)
    encoded = encode_ipc_handle_args(raw_args)
    assert IPC_WIRE_TAG in encoded

    # Must survive strict msgpack serialization without pickle.
    wire_bytes = msgspec.msgpack.encode(encoded)
    wire_decoded = msgspec.msgpack.decode(wire_bytes)
    restored = decode_ipc_handle_args(wire_decoded)
    assert restored == raw_args


def test_ipc_wire_codec_rejects_disallowed_classes_and_malformed_handles():
    sample = torch.zeros((2, 2), dtype=torch.float32)

    class EvilTensor:
        pass

    bad_tensor_cls = _fake_cuda_ipc_args(sample, tensor_cls=EvilTensor)
    with pytest.raises(ValueError, match="Unsupported tensor class"):
        encode_ipc_handle_args(bad_tensor_cls)

    bad_storage_cls = _fake_cuda_ipc_args(sample, storage_cls=EvilTensor)
    with pytest.raises(ValueError, match="Unsupported storage class"):
        encode_ipc_handle_args(bad_storage_cls)

    with pytest.raises(ValueError, match="Expected 15"):
        encode_ipc_handle_args((torch.Tensor,))

    encoded = encode_ipc_handle_args(_fake_cuda_ipc_args(sample))
    encoded[IPC_WIRE_TAG][0] = "builtins.eval"
    with pytest.raises(ValueError, match="Unsupported tensor class tag"):
        decode_ipc_handle_args(encoded)

    encoded_dtype = encode_ipc_handle_args(_fake_cuda_ipc_args(sample))
    encoded_dtype[IPC_WIRE_TAG][5] = "not_a_dtype"
    with pytest.raises(ValueError, match="Unsupported dtype tag"):
        decode_ipc_handle_args(encoded_dtype)


def test_update_info_accepts_both_encoded_and_legacy_raw_handles():
    sample = torch.zeros((2, 4), dtype=torch.float32)
    raw = _fake_cuda_ipc_args(sample)

    # New msgspec-encoded payload
    encoded_handles = msgspec.msgpack.decode(msgspec.msgpack.encode(encode_ipc_handles([{"gpu-0": raw}])))
    info_encoded = MergedLoRAIPCUpdateInfo(
        names=["a"],
        dtype_names=["float32"],
        shapes=[[2, 4]],
        ipc_handles=encoded_handles,
        peft_config={"r": 2, "lora_alpha": 4, "target_modules": ["gate"]},
        lora_int_id=7,
    )
    assert info_encoded.ipc_handles == [{"gpu-0": raw}]

    # Legacy raw-tuple payload remains accepted unchanged.
    info_legacy = MergedLoRAIPCUpdateInfo(
        names=["a"],
        dtype_names=["float32"],
        shapes=[[2, 4]],
        ipc_handles=[{"gpu-0": raw}],
        peft_config={"r": 2, "lora_alpha": 4, "target_modules": ["gate"]},
        lora_int_id=7,
    )
    assert info_legacy.ipc_handles == [{"gpu-0": raw}]

    # Direct decode_ipc_handles preserves raw tuples too.
    assert decode_ipc_handles([{"gpu-0": raw}]) == [{"gpu-0": raw}]


def test_lora_adapter_weight_source_sorts_names_and_normalizes_peft_config():
    t_b = torch.randn(6, 2)
    t_a = torch.randn(2, 4)
    source = LoRAAdapterWeightSource(
        {"z.lora_B.weight": t_b, "a.lora_A.weight": t_a},
        peft_config={"r": 2, "lora_alpha": 4, "target_modules": {"to_v", "to_q"}},
        lora_int_id=3,
    )
    assert [m.name for m in source.metadata] == ["a.lora_A.weight", "z.lora_B.weight"]
    assert [name for name, _ in source.iter_full()] == ["a.lora_A.weight", "z.lora_B.weight"]
    assert source.peft_config["target_modules"] == ["to_q", "to_v"]

    # Updating refreshes metadata.
    source.update({"b.lora_A.weight": t_a}, lora_int_id=4)
    assert [m.name for m in source.metadata] == ["b.lora_A.weight"]
    assert source.lora_int_id == 4

    with pytest.raises(ValueError, match="empty set"):
        LoRAAdapterWeightSource({"a": t_a}, peft_config={"r": 2, "lora_alpha": 4, "target_modules": set()})


def test_trainer_engine_rejects_packed_mode():
    with pytest.raises(ValueError, match="packed=False"):
        MergedLoRAIPCTrainerInitInfo(rank=0, packed=True)


def test_trainer_to_worker_end_to_end_cpu_with_msgspec_and_ref_lifetime(monkeypatch):
    """Drive trainer factory -> msgspec wire -> worker engine -> merged weights on CPU."""
    for mod in (linear, parameter):
        monkeypatch.setattr(mod, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(mod, "get_tensor_model_parallel_rank", lambda: 0)

    with set_current_vllm_config(VllmConfig()):
        model = torch.nn.Module()
        model.transformer = torch.nn.Module()
        model.transformer.stacked_params_mapping = [("gate_up", "gate", 0), ("gate_up", "up", 1)]
        model.transformer.gate_up = linear.MergedColumnParallelLinear(
            4, [6, 6], bias=False, params_dtype=torch.bfloat16
        )
        with torch.no_grad():
            model.transformer.gate_up.weight.fill_(0.25)
        base = model.transformer.gate_up.weight.detach().clone()
        pointer = model.transformer.gate_up.weight.data_ptr()

        manager = DiffusionLoRAManager(model, torch.device("cpu"), torch.bfloat16, lora_application="merged")
        worker_engine = MergedLoRAIPCWeightTransferEngine(
            WeightTransferConfig(backend="omni_lora_ipc"), VllmConfig(), torch.device("cpu"), model
        )
        worker_engine.lora_manager = manager
        monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

        # Fake IPC registry keyed by storage_handle bytes so the worker rebuilds
        # the exact staged tensor (verifying staging refs stay alive through finish).
        exported_tensors: dict[bytes, weakref.ReferenceType[torch.Tensor]] = {}
        events: list[str] = []

        def fake_reduce_tensor(t: torch.Tensor):
            handle = f"handle-{len(exported_tensors)}".encode()
            exported_tensors[handle] = weakref.ref(t)
            return (None, _fake_cuda_ipc_args(t, storage_handle=handle))

        def fake_receive(receiver, info: MergedLoRAIPCUpdateInfo):
            assert isinstance(info.ipc_handles, list)
            rebuilt = []
            for name, handle_map in zip(info.names, info.ipc_handles):
                args = handle_map["gpu-uuid-0"]
                # Decoded back to real PyTorch class and dtype objects!
                assert args[0] is torch.Tensor
                assert args[4] is torch.UntypedStorage
                assert isinstance(args[5], torch.dtype)
                live_tensor = exported_tensors[args[7]]()
                assert live_tensor is not None, "staging tensor freed before worker read it"
                rebuilt.append((name, live_tensor))
            receiver.model.load_weights(rebuilt)

        monkeypatch.setattr(IPCWeightTransferEngine, "receive_weights", fake_receive)

        class InProcessMsgpackClient:
            def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None:
                events.append("init")
                worker_engine.init_transfer_engine(worker_engine.parse_init_info(init_info))

            def start_weight_update(self) -> None:
                events.append("start")
                worker_engine.start_weight_update()

            def update_weights(self, update_info: dict[str, Any]) -> None:
                events.append("update")
                # Force every payload field through real msgspec msgpack round-trip.
                wire = msgspec.msgpack.decode(msgspec.msgpack.encode(update_info))
                # Names must arrive sorted.
                assert wire["names"] == sorted(wire["names"])
                worker_engine.update_weights(wire)

            def finish_weight_update(self, weight_version: str | None = None) -> None:
                events.append("finish")
                # All staging refs must still be alive when finish_weight_update runs.
                assert all(ref() is not None for ref in exported_tensors.values())
                worker_engine.finish_weight_update()

        client = InProcessMsgpackClient()
        assert isinstance(client, VLLMWeightSyncClient)

        generator = torch.Generator().manual_seed(41)
        lora_a = torch.randn(2, 4, generator=generator)
        lora_b = torch.randn(6, 2, generator=generator)
        # Deliberately unsorted insertion order to verify sorted-name alignment.
        source = LoRAAdapterWeightSource(
            {
                "base_model.model.transformer.gate.lora_B.weight": lora_b,
                "base_model.model.transformer.gate.lora_A.weight": lora_a,
            },
            peft_config={"r": 2, "lora_alpha": 3, "target_modules": {"gate"}},
            lora_int_id=5,
        )

        def fake_ipc_collect():
            events.append("ipc_collect")
            # Staging refs must already be cleared before ipc_collect runs.
            assert all(ref() is None for ref in exported_tensors.values())

        trainer = WeightTransferTrainerFactory.trainer_init(
            MergedLoRAIPCTrainerInitInfo(
                rank=0,
                encode_handles=True,
                gpu_uuid="gpu-uuid-0",
                reduce_tensor_fn=fake_reduce_tensor,
                stage_tensor_fn=lambda t: t.detach().clone().contiguous(),
                ipc_collect_fn=fake_ipc_collect,
                synchronize_fn=lambda: None,
            ),
            client=client,
            source=source,
        )
        assert isinstance(trainer, MergedLoRAIPCTrainerWeightTransferEngine)

        trainer.send_weights()
        assert events == ["init", "start", "update", "finish", "ipc_collect"]

        expected = base.float()
        expected[:6] += (lora_b @ lora_a) * 1.5
        torch.testing.assert_close(model.transformer.gate_up.weight, expected.bfloat16(), rtol=0, atol=0)
        assert model.transformer.gate_up.weight.data_ptr() == pointer
        assert manager.list_adapters() == [5]


def _two_rank_all_gather_worker(rank: int, port: int, fail_sender_rpc: bool) -> None:
    torch.distributed.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    # Use an explicit caller-supplied subgroup (distinct from default WORLD).
    subgroup = torch.distributed.new_group(ranks=[0, 1])
    try:
        received_payloads: list[dict[str, Any]] = []
        ipc_collect_calls = 0

        class RecordingClient:
            def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None:
                assert rank == 0

            def start_weight_update(self) -> None:
                assert rank == 0

            def update_weights(self, update_info: dict[str, Any]) -> None:
                assert rank == 0
                received_payloads.append(update_info)
                if fail_sender_rpc:
                    raise RuntimeError("injected sender rpc failure")

            def finish_weight_update(self, weight_version: str | None = None) -> None:
                assert rank == 0

        def fake_reduce(t: torch.Tensor):
            return (None, _fake_cuda_ipc_args(t, storage_handle=f"rank-{rank}".encode()))

        def on_ipc_collect():
            nonlocal ipc_collect_calls
            ipc_collect_calls += 1

        # Insert keys in opposite orders on rank 0 vs rank 1 to prove sorted-name alignment.
        tensors = (
            {"b.lora_B.weight": torch.ones(4, 2), "a.lora_A.weight": torch.ones(2, 4)}
            if rank == 0
            else {"a.lora_A.weight": torch.ones(2, 4), "b.lora_B.weight": torch.ones(4, 2)}
        )
        source = LoRAAdapterWeightSource(
            tensors,
            peft_config={"r": 2, "lora_alpha": 2, "target_modules": {"gate"}},
            lora_int_id=9,
        )
        trainer = WeightTransferTrainerFactory.trainer_init(
            MergedLoRAIPCTrainerInitInfo(
                rank=rank,
                process_group=subgroup,
                encode_handles=True,
                gpu_uuid=f"gpu-uuid-rank-{rank}",
                reduce_tensor_fn=fake_reduce,
                stage_tensor_fn=lambda t: t.detach().clone().contiguous(),
                ipc_collect_fn=on_ipc_collect,
                synchronize_fn=lambda: None,
            ),
            client=RecordingClient(),
            source=source,
        )

        if fail_sender_rpc:
            expected_exc = RuntimeError
            expected_msg = "injected sender rpc failure" if rank == 0 else "weight sync failed on rank 0"
            with pytest.raises(expected_exc, match=expected_msg):
                trainer.send_weights()
        else:
            trainer.send_weights()

        # Both ranks must reach outcome exchange and ipc_collect without deadlocking.
        assert ipc_collect_calls == 1
        if rank == 0:
            if fail_sender_rpc:
                assert len(received_payloads) == 2
                assert received_payloads[1] == {
                    "names": [],
                    "dtype_names": [],
                    "shapes": [],
                    "abort": True,
                }
            else:
                assert len(received_payloads) == 1
            payload = received_payloads[0]
            assert payload["names"] == ["a.lora_A.weight", "b.lora_B.weight"]
            decoded = decode_ipc_handles(payload["ipc_handles"])
            assert isinstance(decoded, list) and len(decoded) == 2
            for entry in decoded:
                assert set(entry.keys()) == {"gpu-uuid-rank-0", "gpu-uuid-rank-1"}
                assert entry["gpu-uuid-rank-0"][7] == b"rank-0"
                assert entry["gpu-uuid-rank-1"][7] == b"rank-1"
        else:
            assert received_payloads == []
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("fail_sender_rpc", [False, True])
def test_trainer_multi_rank_uuid_all_gather_over_caller_process_group(fail_sender_rpc):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    torch.multiprocessing.spawn(_two_rank_all_gather_worker, args=(port, fail_sender_rpc), nprocs=2)


def _rank_failure_phase_worker(rank: int, port: int, mode: str) -> None:
    torch.distributed.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:
        ipc_collect_calls = 0
        reduce_calls = 0

        def on_ipc_collect():
            nonlocal ipc_collect_calls
            ipc_collect_calls += 1

        class NoopClient:
            def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None:
                pass

            def start_weight_update(self) -> None:
                pass

            def update_weights(self, update_info: dict[str, Any]) -> None:
                pass

            def finish_weight_update(self, weight_version: str | None = None) -> None:
                pass

        def stage_fn(t: torch.Tensor) -> torch.Tensor:
            if mode == "prep_oom_rank1" and rank == 1:
                raise RuntimeError("injected rank1 staging OOM")
            return t.detach().clone().contiguous()

        def sync_fn() -> None:
            if mode == "sync_fail_rank1" and rank == 1:
                raise RuntimeError("injected rank1 async CUDA sync error")

        def reduce_fn(t: torch.Tensor):
            nonlocal reduce_calls
            reduce_calls += 1
            if mode == "sender_encode_fail" and rank == 0:
                # Picklable built-in class rejected by encode_ipc_handles on sender AFTER all-gather.
                return (None, _fake_cuda_ipc_args(t, tensor_cls=object))
            return (None, _fake_cuda_ipc_args(t))

        source = LoRAAdapterWeightSource(
            {"a.lora_A.weight": torch.ones(2, 4)},
            peft_config={"r": 2, "lora_alpha": 2, "target_modules": ["gate"]},
        )
        trainer = WeightTransferTrainerFactory.trainer_init(
            MergedLoRAIPCTrainerInitInfo(
                rank=rank,
                encode_handles=True,
                gpu_uuid=f"gpu-{rank}",
                reduce_tensor_fn=reduce_fn,
                stage_tensor_fn=stage_fn,
                ipc_collect_fn=on_ipc_collect,
                synchronize_fn=sync_fn,
            ),
            client=NoopClient(),
            source=source,
        )

        if mode == "prep_oom_rank1":
            match = "injected rank1 staging OOM" if rank == 1 else "trainer preparation failed on rank 1"
            with pytest.raises(RuntimeError, match=match):
                trainer.send_weights()
            # No rank may export CUDA IPC handles when preparation fails.
            assert reduce_calls == 0
            assert ipc_collect_calls == 0
        elif mode == "sync_fail_rank1":
            match = "injected rank1 async CUDA sync error" if rank == 1 else "trainer preparation failed on rank 1"
            with pytest.raises(RuntimeError, match=match):
                trainer.send_weights()
            # Local synchronize failure is coordinated before any IPC export.
            assert reduce_calls == 0
            assert ipc_collect_calls == 0
        else:
            if rank == 0:
                with pytest.raises(ValueError, match="Unsupported tensor class"):
                    trainer.send_weights()
            else:
                with pytest.raises(RuntimeError, match="weight sync failed on rank 0"):
                    trainer.send_weights()
            assert reduce_calls == 1
            assert ipc_collect_calls == 1
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("mode", ["prep_oom_rank1", "sync_fail_rank1", "sender_encode_fail"])
def test_trainer_multi_rank_failures_before_and_after_gather_do_not_deadlock(mode):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    torch.multiprocessing.spawn(_rank_failure_phase_worker, args=(port, mode), nprocs=2)


@pytest.mark.parametrize(
    "failure_mode",
    [
        "start_reply_lost",
        "update_before_worker",
        "update_reply_lost_after_staging",
        "finish_before_worker",
        "finish_reply_lost_after_commit",
    ],
)
@pytest.mark.parametrize("client_has_abort_method", [False, True])
def test_trainer_aborts_remote_worker_without_committing_on_interrupted_rpc(
    monkeypatch,
    failure_mode,
    client_has_abort_method,
):
    """Interrupted start, update, or finish must keep worker state consistent and usable.

    Covers:
    - start_weight_update reaching the worker with only the reply lost
    - update_weights failing before reaching the worker
    - update_weights reaching the worker (staging the full adapter) with only the reply lost
    - finish_weight_update failing before reaching the worker
    - finish_weight_update committing on the worker with only the reply lost
    for both a standard 4-method VLLMWeightSyncClient (abort sentinel) and a client
    exposing an explicit abort_weight_update() method, with an existing merged adapter live.
    """
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

    for mod in (linear, parameter):
        monkeypatch.setattr(mod, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(mod, "get_tensor_model_parallel_rank", lambda: 0)

    def _expected_merged(base_weight: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor) -> torch.Tensor:
        expected = base_weight.detach().clone()
        delta = torch.mm(lora_b.float(), lora_a.float())
        expected[:6].copy_((base_weight[:6].float() + delta).to(base_weight.dtype))
        return expected

    with set_current_vllm_config(VllmConfig()):
        model = torch.nn.Module()
        model.transformer = torch.nn.Module()
        model.transformer.stacked_params_mapping = [("gate_up", "gate", 0), ("gate_up", "up", 1)]
        model.transformer.gate_up = linear.MergedColumnParallelLinear(
            4, [6, 6], bias=False, params_dtype=torch.bfloat16
        )
        with torch.no_grad():
            model.transformer.gate_up.weight.fill_(0.25)
        base = model.transformer.gate_up.weight.detach().clone()

        manager = DiffusionLoRAManager(model, torch.device("cpu"), torch.bfloat16, lora_application="merged")
        worker_engine = MergedLoRAIPCWeightTransferEngine(
            WeightTransferConfig(backend="omni_lora_ipc"), VllmConfig(), torch.device("cpu"), model
        )
        worker_engine.lora_manager = manager
        monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

        worker = object.__new__(DiffusionWorker)
        worker.weight_transfer_engine = worker_engine
        worker._weight_update_active = False

        exported: dict[bytes, torch.Tensor] = {}

        def fake_reduce(t: torch.Tensor):
            handle = f"h-{len(exported)}".encode()
            exported[handle] = t
            return (None, _fake_cuda_ipc_args(t, storage_handle=handle))

        def fake_receive(receiver, info: MergedLoRAIPCUpdateInfo):
            assert isinstance(info.ipc_handles, list)
            receiver.model.load_weights(
                [(name, exported[handle_map["gpu-0"][7]]) for name, handle_map in zip(info.names, info.ipc_handles)]
            )

        monkeypatch.setattr(IPCWeightTransferEngine, "receive_weights", fake_receive)

        fault_armed = {"armed": False}
        staged_before_reply_loss = {"count": 0}

        class FourMethodFlakyClient:
            def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None:
                worker.init_weight_transfer_engine(init_info)

            def start_weight_update(self) -> None:
                if fault_armed["armed"] and failure_mode == "start_reply_lost":
                    fault_armed["armed"] = False
                    worker.start_weight_update()
                    raise ConnectionError("simulated start reply loss after worker started session")
                worker.start_weight_update()

            def update_weights(self, update_info: dict[str, Any]) -> None:
                if fault_armed["armed"] and not update_info.get("abort"):
                    if failure_mode == "update_before_worker":
                        fault_armed["armed"] = False
                        raise ConnectionError("simulated update transport drop")
                    if failure_mode == "update_reply_lost_after_staging":
                        fault_armed["armed"] = False
                        # Worker receives and stages the FULL adapter, then only the reply is lost!
                        worker.update_weights(update_info)
                        staged_before_reply_loss["count"] = len(worker_engine._staged.tensors)
                        raise ConnectionError("simulated update reply loss after worker staged adapter")
                worker.update_weights(update_info)

            def finish_weight_update(self, weight_version: str | None = None) -> None:
                if fault_armed["armed"]:
                    if failure_mode == "finish_before_worker":
                        fault_armed["armed"] = False
                        # Fails BEFORE reaching worker.finish_weight_update!
                        raise ConnectionError("simulated finish transport drop")
                    if failure_mode == "finish_reply_lost_after_commit":
                        fault_armed["armed"] = False
                        worker.finish_weight_update()
                        raise ConnectionError("simulated finish reply loss after worker committed")
                worker.finish_weight_update()

        class AbortCapableFlakyClient(FourMethodFlakyClient):
            def abort_weight_update(self) -> None:
                worker.abort_weight_update()

        client = AbortCapableFlakyClient() if client_has_abort_method else FourMethodFlakyClient()

        generator = torch.Generator().manual_seed(53)
        a10 = torch.randn(2, 4, generator=generator)
        b10 = torch.randn(6, 2, generator=generator)
        a11 = torch.randn(2, 4, generator=generator)
        b11 = torch.randn(6, 2, generator=generator)
        a12 = torch.randn(2, 4, generator=generator)
        b12 = torch.randn(6, 2, generator=generator)
        expected_10 = _expected_merged(base, a10, b10)
        expected_11 = _expected_merged(base, a11, b11)
        expected_12 = _expected_merged(base, a12, b12)

        source = LoRAAdapterWeightSource(
            {
                "base_model.model.transformer.gate.lora_A.weight": a10,
                "base_model.model.transformer.gate.lora_B.weight": b10,
            },
            peft_config={"r": 2, "lora_alpha": 2, "target_modules": ["gate"]},
            lora_int_id=10,
        )
        trainer = WeightTransferTrainerFactory.trainer_init(
            MergedLoRAIPCTrainerInitInfo(
                rank=0,
                gpu_uuid="gpu-0",
                reduce_tensor_fn=fake_reduce,
                stage_tensor_fn=lambda t: t.detach().clone().contiguous(),
                ipc_collect_fn=lambda: None,
                synchronize_fn=lambda: None,
            ),
            client=client,
            source=source,
        )

        # Commit an initial adapter (id=10) so aborts are verified against a live merged policy.
        trainer.send_weights()
        assert manager.list_adapters() == [10]
        torch.testing.assert_close(model.transformer.gate_up.weight, expected_10, rtol=0, atol=0)
        torch.testing.assert_close(model.transformer.gate_up.weight[6:], base[6:], rtol=0, atol=0)

        # Attempt adapter id=11 with the simulated RPC fault armed.
        source.update(
            {
                "base_model.model.transformer.gate.lora_A.weight": a11,
                "base_model.model.transformer.gate.lora_B.weight": b11,
            },
            lora_int_id=11,
        )
        fault_armed["armed"] = True
        with pytest.raises(ConnectionError, match="simulated"):
            trainer.send_weights()

        if failure_mode == "update_reply_lost_after_staging":
            assert staged_before_reply_loss["count"] == 2

        assert worker._weight_update_active is False
        assert not worker_engine._staged.tensors
        if failure_mode == "finish_reply_lost_after_commit":
            # Worker already committed before reply loss; live state must remain consistent.
            assert manager.list_adapters() == [11]
            torch.testing.assert_close(model.transformer.gate_up.weight, expected_11, rtol=0, atol=0)
            torch.testing.assert_close(model.transformer.gate_up.weight[6:], base[6:], rtol=0, atol=0)
            # Subsequent sync with adapter id=12 succeeds cleanly.
            source.update(
                {
                    "base_model.model.transformer.gate.lora_A.weight": a12,
                    "base_model.model.transformer.gate.lora_B.weight": b12,
                },
                lora_int_id=12,
            )
            trainer.send_weights()
            assert worker._weight_update_active is False
            assert manager.list_adapters() == [12]
            torch.testing.assert_close(model.transformer.gate_up.weight, expected_12, rtol=0, atol=0)
            torch.testing.assert_close(model.transformer.gate_up.weight[6:], base[6:], rtol=0, atol=0)
        else:
            # Must fail closed: previous live adapter (id=10) survives untouched, nothing new committed.
            assert manager.list_adapters() == [10]
            torch.testing.assert_close(model.transformer.gate_up.weight, expected_10, rtol=0, atol=0)
            torch.testing.assert_close(model.transformer.gate_up.weight[6:], base[6:], rtol=0, atol=0)
            # Subsequent retry of adapter id=11 succeeds cleanly with exact merged weights.
            trainer.send_weights()
            assert worker._weight_update_active is False
            assert manager.list_adapters() == [11]
            torch.testing.assert_close(model.transformer.gate_up.weight, expected_11, rtol=0, atol=0)
            torch.testing.assert_close(model.transformer.gate_up.weight[6:], base[6:], rtol=0, atol=0)


def test_trainer_does_not_abort_unrelated_already_active_session(monkeypatch):
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

    for mod in (linear, parameter):
        monkeypatch.setattr(mod, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(mod, "get_tensor_model_parallel_rank", lambda: 0)

    with set_current_vllm_config(VllmConfig()):
        model = torch.nn.Module()
        model.transformer = torch.nn.Module()
        model.transformer.stacked_params_mapping = [("gate_up", "gate", 0), ("gate_up", "up", 1)]
        model.transformer.gate_up = linear.MergedColumnParallelLinear(
            4, [6, 6], bias=False, params_dtype=torch.bfloat16
        )
        manager = DiffusionLoRAManager(model, torch.device("cpu"), torch.bfloat16, lora_application="merged")
        worker_engine = MergedLoRAIPCWeightTransferEngine(
            WeightTransferConfig(backend="omni_lora_ipc"), VllmConfig(), torch.device("cpu"), model
        )
        worker_engine.lora_manager = manager
        worker = object.__new__(DiffusionWorker)
        worker.weight_transfer_engine = worker_engine
        worker._weight_update_active = False

        # Pre-existing active session owned by another caller.
        worker.start_weight_update()
        worker_engine._staged.tensors["existing"] = torch.ones(2, 4)

        class WorkerClient:
            def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None:
                worker.init_weight_transfer_engine(init_info)

            def start_weight_update(self) -> None:
                worker.start_weight_update()

            def update_weights(self, update_info: dict[str, Any]) -> None:
                worker.update_weights(update_info)

            def finish_weight_update(self, weight_version: str | None = None) -> None:
                worker.finish_weight_update()

        source = LoRAAdapterWeightSource(
            {"base_model.model.transformer.gate.lora_A.weight": torch.ones(2, 4)},
            peft_config={"r": 2, "lora_alpha": 2, "target_modules": ["gate"]},
            lora_int_id=12,
        )
        trainer = WeightTransferTrainerFactory.trainer_init(
            MergedLoRAIPCTrainerInitInfo(
                rank=0,
                gpu_uuid="gpu-0",
                reduce_tensor_fn=lambda t: (None, _fake_cuda_ipc_args(t)),
                stage_tensor_fn=lambda t: t.detach().clone().contiguous(),
                ipc_collect_fn=lambda: None,
                synchronize_fn=lambda: None,
            ),
            client=WorkerClient(),
            source=source,
        )

        with pytest.raises(RuntimeError, match="already active"):
            trainer.send_weights()

        # Pre-existing active session must remain untouched.
        assert worker._weight_update_active is True
        assert "existing" in worker_engine._staged.tensors


def test_abort_update_info_rejects_non_empty_tensor_payload():
    with pytest.raises(ValueError, match="Abort update info must not carry tensor payloads"):
        MergedLoRAIPCUpdateInfo(
            names=["a"],
            dtype_names=["float32"],
            shapes=[[1]],
            ipc_handles=[{}],
            abort=True,
        )


def test_ar_worker_abort_weight_update_drops_staged_without_committing(monkeypatch):
    from vllm_omni.lora.weight_transfer import _ABORT_UPDATE_PAYLOAD, MergedLoRASessionAbortedError
    from vllm_omni.worker.base import OmniGPUWorkerBase

    for mod in (linear, parameter):
        monkeypatch.setattr(mod, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(mod, "get_tensor_model_parallel_rank", lambda: 0)

    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        model = torch.nn.Module()
        model.transformer = torch.nn.Module()
        model.transformer.stacked_params_mapping = [("gate_up", "gate", 0), ("gate_up", "up", 1)]
        model.transformer.gate_up = linear.MergedColumnParallelLinear(
            4, [6, 6], bias=False, params_dtype=torch.bfloat16
        )
        with torch.no_grad():
            model.transformer.gate_up.weight.fill_(0.25)
        base = model.transformer.gate_up.weight.detach().clone()

        manager = DiffusionLoRAManager(model, torch.device("cpu"), torch.bfloat16, lora_application="merged")
        engine = MergedLoRAIPCWeightTransferEngine(
            WeightTransferConfig(backend="omni_lora_ipc"), vllm_config, torch.device("cpu"), model
        )
        engine.lora_manager = manager
        engine.start_weight_update()
        engine._staged.tensors["dummy"] = torch.ones(2, 4)

        close_count = {"count": 0}

        class _Loan:
            def close(self) -> None:
                close_count["count"] += 1

        worker = object.__new__(OmniGPUWorkerBase)
        worker.vllm_config = vllm_config
        worker.weight_transfer_engine = engine
        worker._weight_update_active = True
        worker._weight_update_is_draft = False
        worker._weight_update_kv_cache_loan = _Loan()

        # Repeated abort_weight_update calls must be idempotent and close the KV loan once.
        worker.abort_weight_update()
        worker.abort_weight_update()
        assert worker._weight_update_active is False
        assert close_count["count"] == 1
        assert not engine._staged.tensors
        assert manager.list_adapters() == []
        torch.testing.assert_close(model.transformer.gate_up.weight, base, rtol=0, atol=0)

        # Subsequent session aborted via the 4-method GPUWorker.update_weights sentinel path.
        engine.start_weight_update()
        engine._staged.tensors["dummy2"] = torch.ones(2, 4)
        worker._weight_update_active = True
        with pytest.raises(MergedLoRASessionAbortedError, match="aborted by trainer"):
            worker.update_weights(dict(_ABORT_UPDATE_PAYLOAD))
        assert worker._weight_update_active is False
        assert close_count["count"] == 2
        assert not engine._staged.tensors
        assert manager.list_adapters() == []
        torch.testing.assert_close(model.transformer.gate_up.weight, base, rtol=0, atol=0)


def _mismatched_names_worker(rank: int, port: int) -> None:
    torch.distributed.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:

        class NoopClient:
            def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None:
                pass

            def start_weight_update(self) -> None:
                pass

            def update_weights(self, update_info: dict[str, Any]) -> None:
                pass

            def finish_weight_update(self, weight_version: str | None = None) -> None:
                pass

        tensors = {"a.lora_A.weight": torch.ones(2, 4)} if rank == 0 else {"b.lora_A.weight": torch.ones(2, 4)}
        reduce_calls = 0
        ipc_collect_calls = 0

        def counting_reduce(t: torch.Tensor):
            nonlocal reduce_calls
            reduce_calls += 1
            return (None, _fake_cuda_ipc_args(t))

        def counting_ipc_collect() -> None:
            nonlocal ipc_collect_calls
            ipc_collect_calls += 1

        source = LoRAAdapterWeightSource(tensors, peft_config={"r": 2, "lora_alpha": 2, "target_modules": ["gate"]})
        trainer = WeightTransferTrainerFactory.trainer_init(
            MergedLoRAIPCTrainerInitInfo(
                rank=rank,
                gpu_uuid=f"gpu-{rank}",
                reduce_tensor_fn=counting_reduce,
                stage_tensor_fn=lambda t: t.detach().clone().contiguous(),
                ipc_collect_fn=counting_ipc_collect,
                synchronize_fn=lambda: None,
            ),
            client=NoopClient(),
            source=source,
        )
        with pytest.raises(ValueError, match="parameter name mismatch"):
            trainer.send_weights()
        assert reduce_calls == 0
        assert ipc_collect_calls == 0
    finally:
        torch.distributed.destroy_process_group()


def test_trainer_rejects_mismatched_parameter_names_across_ranks():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    torch.multiprocessing.spawn(_mismatched_names_worker, args=(port,), nprocs=2)
