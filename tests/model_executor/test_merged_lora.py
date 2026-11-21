# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import socket

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear
from vllm.model_executor.models.utils import WeightsMapper

from vllm_omni.lora.manager import LoRAManager
from vllm_omni.lora.merged_manager import ARMergedLoRAManager
from vllm_omni.lora.merged_weights import check_merge_error
from vllm_omni.lora.request import TensorLoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _manager_prepare_failure_rank(rank: int, port: int) -> None:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear
    from vllm.model_executor.models.utils import WeightsMapper

    from vllm_omni.lora.merged_manager import ARMergedLoRAManager
    from vllm_omni.lora.request import TensorLoRARequest

    torch.distributed.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:
        # Keep this CPU integration check independent of a vLLM TP group.
        for module in (linear, parameter):
            module.get_tensor_model_parallel_world_size = lambda: 1
            module.get_tensor_model_parallel_rank = lambda: 0
        with set_current_vllm_config(VllmConfig()):
            model = torch.nn.Module()
            model.language_model = torch.nn.Module()
            model.language_model.model = torch.nn.Module()
            model.language_model.model.qkv_proj = linear.QKVParallelLinear(
                4, 2, 4, total_num_kv_heads=2, bias=False, params_dtype=torch.bfloat16
            )
            model.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
            model.hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"thinker.model.": "language_model.model."})
            layer = model.language_model.model.qkv_proj
            with torch.no_grad():
                layer.weight.fill_(0.5)
            previous = layer.weight.detach().clone()
            generator = torch.Generator().manual_seed(29)
            tensors = {}
            for name, rows in (("q_proj", 8), ("k_proj", 4), ("v_proj", 4)):
                tensors[f"base_model.model.thinker.model.{name}.lora_A.weight"] = torch.randn(2, 4, generator=generator)
                tensors[f"base_model.model.thinker.model.{name}.lora_B.weight"] = torch.randn(
                    rows, 2, generator=generator
                )
            if rank == 1:
                key = "base_model.model.thinker.model.q_proj.lora_B.weight"
                tensors[key] = tensors[key][:-1]
            request = TensorLoRARequest(
                "policy",
                1,
                "tensor",
                lora_tensors=tensors,
                peft_config={"r": 2, "lora_alpha": 3, "target_modules": ["q_proj", "k_proj", "v_proj"]},
            )
            manager = ARMergedLoRAManager(model, torch.device("cpu"), torch.bfloat16, lora_application="merged")
            error = None
            try:
                manager.prepare_merged_adapter(request)
            except Exception as exc:
                error = exc
            with pytest.raises((ValueError, RuntimeError)):
                check_merge_error(error, torch.distributed.group.WORLD)
            torch.testing.assert_close(layer.weight, previous, rtol=0, atol=0)
            manager.close()
    finally:
        torch.distributed.destroy_process_group()


def test_manager_prepare_failure_aborts_both_ranks_without_mutating_policy():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    torch.multiprocessing.spawn(_manager_prepare_failure_rank, args=(port,), nprocs=2)


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 0), (2, 1), (4, 0), (4, 3)])
def test_ar_qkv_mapping_and_replicated_kv_heads(monkeypatch, tp_size, tp_rank):
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: tp_size)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: tp_rank)
    with set_current_vllm_config(VllmConfig()):
        model = torch.nn.Module()
        model.language_model = torch.nn.Module()
        model.language_model.model = torch.nn.Module()
        model.language_model.model.qkv_proj = linear.QKVParallelLinear(
            4, 2, 4, total_num_kv_heads=2, bias=False, params_dtype=torch.bfloat16
        )
        model.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
        model.hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"thinker.model.": "language_model.model."})
        layer = model.language_model.model.qkv_proj
        with torch.no_grad():
            layer.weight.fill_(0.5)
        base = layer.weight.detach().clone()
        pointer = layer.weight.data_ptr()
        generator = torch.Generator().manual_seed(29)
        tensors = {}
        for name, rows in (("q_proj", 8), ("k_proj", 4), ("v_proj", 4)):
            tensors[f"base_model.model.thinker.model.{name}.lora_A.weight"] = torch.randn(2, 4, generator=generator)
            tensors[f"base_model.model.thinker.model.{name}.lora_B.weight"] = torch.randn(rows, 2, generator=generator)
        request = TensorLoRARequest(
            "policy",
            1,
            "tensor",
            lora_tensors=tensors,
            peft_config={"r": 2, "lora_alpha": 3, "target_modules": ["q_proj", "k_proj", "v_proj"]},
        )
        manager = ARMergedLoRAManager(model, torch.device("cpu"), torch.bfloat16, lora_application="merged")
        manager.merge_adapter(request)
        expected_slices = []
        for name, rows in (("q_proj", 8), ("k_proj", 4), ("v_proj", 4)):
            prefix = f"base_model.model.thinker.model.{name}"
            delta = tensors[f"{prefix}.lora_B.weight"] @ tensors[f"{prefix}.lora_A.weight"] * 1.5
            local_rows = rows // min(tp_size, 4 if name == "q_proj" else 2)
            shard_rank = tp_rank if name == "q_proj" else tp_rank // max(1, tp_size // 2)
            expected_slices.append(delta[shard_rank * local_rows : (shard_rank + 1) * local_rows])
        torch.testing.assert_close(layer.weight, (base.float() + torch.cat(expected_slices)).bfloat16(), rtol=0, atol=0)
        assert model.language_model.model.qkv_proj is layer
        assert layer.weight.data_ptr() == pointer
        assert not any("base_layer" in name for name, _ in model.named_parameters())
        manager.close()
        torch.testing.assert_close(layer.weight, base, rtol=0, atol=0)


def test_ar_merged_manager_uses_shared_lora_base():
    assert ARMergedLoRAManager.__bases__ == (LoRAManager,)
    assert LoRAManager.__module__ == "vllm_omni.lora.manager"
