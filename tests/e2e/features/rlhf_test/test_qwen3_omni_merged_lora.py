# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise real Qwen3-Omni generation around adapter-only IPC updates."""

import time
from contextlib import ExitStack

import pytest
import torch

from tests.e2e.features.rlhf_test.test_weight_transfer_changes_output import _spawn_ipc_exporter
from tests.helpers.mark import hardware_test
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni.entrypoints.async_omni import AsyncOmni

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
IPC_INIT_INFO = {"packed": False}


async def _generate_text(engine: AsyncOmni, request_id: str) -> tuple[int, ...]:
    tokens: tuple[int, ...] = ()
    async for output in engine.generate(
        "What is the capital of China?", request_id=request_id, output_modalities=["text"]
    ):
        if output.final_output_type == "text" and output.outputs:
            tokens = tuple(output.outputs[0].token_ids)
    assert tokens, "Qwen3-Omni produced no thinker text"
    return tokens


async def _publish_adapter(engine: AsyncOmni, factors: list[tuple[str, torch.Tensor]]) -> None:
    payload, exporter = _spawn_ipc_exporter(factors)
    payload.update(
        peft_config={"r": 1, "lora_alpha": 4, "target_modules": ["q_proj"]},
        lora_int_id=1,
    )
    try:
        await engine.start_weight_update()
        await engine.update_weights(payload)
        await engine.finish_weight_update()
    finally:
        del payload
        exporter.terminate()
        exporter.join(timeout=10)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.asyncio
async def test_qwen3_omni_merged_lora_ipc_preserves_graph_storage_and_restores_output(monkeypatch):
    # Raw CUDA IPC handles contain Python classes. The AR engine-core RPC
    # crosses a msgpack boundary, unlike the in-process diffusion worker.
    # Opt in only for this test's trusted, locally spawned IPC exporter.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    deploy = modify_stage_config(
        get_deploy_config_path("ci/qwen3_omni_moe.yaml"),
        updates={"async_chunk": False, "stages": {0: {"weight_transfer_config": {"backend": "omni_lora_ipc"}}}},
    )
    with ExitStack() as after:
        engine = AsyncOmni(
            model=MODEL,
            deploy_config=deploy,
            worker_extension_cls="tests.e2e.features.rlhf_test.qwen3_omni_merge_probe.Qwen3OmniMergeProbe",
        )
        after.callback(engine.shutdown)
        baseline_tokens = await _generate_text(engine, "baseline")
        (before,) = await engine.collective_rpc("merged_q_projection", stage_ids=[0])
        (projection,) = before
        name = projection["name"]
        assert name.startswith("thinker.language_model.model.")
        checkpoint_name = name.replace("thinker.language_model.model.", "thinker.model.", 1).replace(
            ".qkv_proj", ".q_proj"
        )
        prefix = f"base_model.model.{checkpoint_name}"
        a = torch.ones(1, projection["input_size"], dtype=torch.float16)
        b = torch.ones(projection["q_rows"], 1, dtype=torch.float16)
        await engine.init_weight_transfer_engine(dict(IPC_INIT_INFO))
        memory_before = await engine.collective_rpc("merged_memory_probe", kwargs={"reset": True}, stage_ids=[0])

        update_started = time.perf_counter()
        await _publish_adapter(engine, [(f"{prefix}.lora_A.weight", a), (f"{prefix}.lora_B.weight", b)])
        update_seconds = time.perf_counter() - update_started
        memory_after = await engine.collective_rpc("merged_memory_probe", stage_ids=[0])
        print(
            f"Qwen3-Omni merged IPC HBM before={memory_before} after={memory_after} "
            f"update_seconds={update_seconds:.3f}",
            flush=True,
        )
        (after_merge,) = await engine.collective_rpc("merged_q_projection", stage_ids=[0])
        assert after_merge[0]["data_ptr"] == projection["data_ptr"]
        assert after_merge[0]["sample"] != projection["sample"]
        assert await _generate_text(engine, "merged") != baseline_tokens

        await _publish_adapter(
            engine, [(f"{prefix}.lora_A.weight", a), (f"{prefix}.lora_B.weight", torch.zeros_like(b))]
        )
        (restored,) = await engine.collective_rpc("merged_q_projection", stage_ids=[0])
        assert restored[0]["data_ptr"] == projection["data_ptr"]
        assert restored[0]["sample"] == projection["sample"]
        assert await _generate_text(engine, "restored") == baseline_tokens
