# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Real AR worker sleep recovery with a local, randomly initialized checkpoint."""

from __future__ import annotations

import asyncio
import threading
import uuid

import pytest
import torch
import yaml
from transformers import LlamaConfig, LlamaForCausalLM
from vllm import SamplingParams

from tests.helpers.mark import hardware_test
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig
from vllm_omni.entrypoints.async_omni import AsyncOmni

pytestmark = [pytest.mark.core_model, pytest.mark.omni]


async def _generate(engine: AsyncOmni):
    final = None
    async for output in engine.generate(
        {"prompt_token_ids": [1, 19, 41, 88]},
        request_id=uuid.uuid4().hex,
        sampling_params_list=[
            SamplingParams(temperature=0, max_tokens=8, ignore_eos=True, detokenize=False, logprobs=3)
        ],
        output_modalities=["text"],
    ):
        final = output
    assert final is not None and final.finished and final.error is None
    assert len(final.outputs) == 1
    completion = final.outputs[0]
    assert len(completion.token_ids) == 8
    assert completion.logprobs is not None and len(completion.logprobs) == 8
    return list(completion.token_ids), completion.logprobs


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_cancelled_ar_sleep_restores_real_model(tmp_path, monkeypatch):
    checkpoint = tmp_path / "tiny-ar"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        config = LlamaConfig(
            vocab_size=256,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=8,
            max_position_embeddings=128,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=False,
        )
        LlamaForCausalLM(config).save_pretrained(checkpoint)

    pipeline_name = "ar_sleep_recovery_test"
    monkeypatch.setitem(
        OMNI_PIPELINES,
        pipeline_name,
        PipelineConfig(
            model_type=pipeline_name,
            # This backend accepts the Omni runner's extra forward metadata
            # while executing real Llama layers, attention and sampling.
            model_arch="TransformersForCausalLM",
            stages=(
                StagePipelineConfig(
                    stage_id=0,
                    model_stage="text",
                    execution_type=StageExecutionType.LLM_AR,
                    final_output=True,
                    final_output_type="text",
                    owns_tokenizer=True,
                    engine_output_type="text",
                    sampling_constraints={"detokenize": False},
                ),
            ),
        ),
    )
    deploy = tmp_path / "deploy.yaml"
    deploy.write_text(
        yaml.safe_dump(
            {
                "pipeline": pipeline_name,
                "async_chunk": False,
                "dtype": "bfloat16",
                "enable_prefix_caching": False,
                "distributed_executor_backend": "uni",
                "stages": [
                    {
                        "stage_id": 0,
                        "devices": "0",
                        "enable_sleep_mode": True,
                        "gpu_memory_utilization": 0.015,
                        "max_model_len": 128,
                        "max_num_seqs": 2,
                        "max_num_batched_tokens": 128,
                        "enforce_eager": True,
                        "async_scheduling": False,
                        "engine_extras": {"skip_tokenizer_init": True, "kv_cache_memory_bytes": 32 << 20},
                    }
                ],
            }
        )
    )

    async def run():
        engine = AsyncOmni(model=str(checkpoint), deploy_config=str(deploy), log_stats=False)
        try:
            before = await asyncio.wait_for(_generate(engine), timeout=60)
            checksums = await engine.collective_rpc("compute_weight_checksums", timeout=30)
            assert len(checksums) == 1 and len(checksums[0]) == 1
            assert checksums[0][0] and all(len(digest) == 64 for digest in checksums[0][0].values())

            # Establish that the same worker can perform an ordinary cycle.
            await engine.sleep(level=1)
            assert len(await engine.wake_up()) == 1
            assert await engine.is_paused()
            await engine.resume_generation()
            assert await asyncio.wait_for(_generate(engine), timeout=60) == before
            assert await engine.collective_rpc("compute_weight_checksums", timeout=30) == checksums

            pool = engine.engine.stage_pools[0]
            original_rpc = pool.collective_rpc
            for _ in range(3):
                slept = threading.Event()
                release = threading.Event()

                async def delay_sleep_ack(replica_id, method, timeout=None, args=(), kwargs=None):
                    result = await original_rpc(replica_id, method, timeout, args, kwargs)
                    if method == "sleep":
                        # Real GPU offload has finished. Only its acknowledgement
                        # is delayed so caller cancellation is deterministic.
                        slept.set()
                        assert await asyncio.to_thread(release.wait, 60)
                    return result

                with monkeypatch.context() as patch:
                    patch.setattr(pool, "collective_rpc", delay_sleep_ack)
                    task = asyncio.create_task(engine.sleep(level=1))
                    try:
                        assert await asyncio.to_thread(slept.wait, 60)
                        task.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await task
                        release.set()
                        acks = await engine.wake_up()
                        assert len(acks) == 1, "wake RPC was skipped after cancelled sleep"
                        assert await engine.is_paused()
                        await engine.resume_generation()
                        assert await asyncio.wait_for(_generate(engine), timeout=60) == before
                        assert await engine.collective_rpc("compute_weight_checksums", timeout=30) == checksums
                    finally:
                        release.set()
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
        finally:
            engine.shutdown()

    asyncio.run(run())
