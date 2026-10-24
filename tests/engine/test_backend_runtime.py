# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""AR deploy/worker acceptance using generated config and random tiny weights.

No checkpoint or tokenizer download is needed. This validates backend routing
and real prefill/decode, not trained-model quality or a full multimodal model.
"""

import json
import os

import pytest
import yaml
from transformers import Qwen3MoeConfig

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import AsyncOmniRunner
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

pytestmark = [pytest.mark.core_model, pytest.mark.omni]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.asyncio
async def test_ar_moe_deploy_backends_execute_prefill_and_decode(tmp_path, monkeypatch):
    model_path = tmp_path / "tiny-qwen3-moe"
    Qwen3MoeConfig(
        architectures=["Qwen3MoeForCausalLM"],
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_experts=8,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    ).save_pretrained(model_path)
    # The external-pipeline registry is consulted in the parent before stages
    # start. Register topology only; execution uses the real stock Qwen3 MoE.
    monkeypatch.setitem(
        OMNI_PIPELINES,
        "qwen3_moe",
        PipelineConfig(
            model_type="qwen3_moe",
            model_arch="Qwen3MoeForCausalLM",
            stages=(
                StagePipelineConfig(
                    stage_id=0,
                    model_stage="model",
                    execution_type=StageExecutionType.LLM_AR,
                    final_output=True,
                    final_output_type="text",
                    engine_output_type="text",
                ),
            ),
        ),
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    # The acceptance target is AR backend execution.  This local vLLM wheel
    # has no nvcc, so leave FlashInfer's unrelated sampler JIT out of startup.
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
    deploy = tmp_path / "deploy.yaml"
    deploy.write_text(
        yaml.safe_dump(
            {
                "async_chunk": False,
                "dtype": "bfloat16",
                "enable_prefix_caching": False,
                "enable_chunked_prefill": True,
                "stages": [
                    {
                        "stage_id": 0,
                        "devices": "0",
                        "linear_backend": "torch",
                        "attention_backend": "TRITON_ATTN",
                        "moe_backend": "triton",
                        "load_format": "dummy",
                        "enforce_eager": True,
                        "async_scheduling": False,
                        "gpu_memory_utilization": 0.2,
                        "max_model_len": 128,
                        "max_num_batched_tokens": 16,
                        "max_num_seqs": 1,
                        "engine_extras": {
                            "skip_tokenizer_init": True,
                            "kv_cache_memory_bytes": 32 * 1024 * 1024,
                            "worker_cls": "tests.engine.backend_runtime_worker.BackendRuntimeWorker",
                        },
                    },
                ],
            }
        )
    )
    with AsyncOmniRunner(str(model_path), deploy_config=str(deploy)) as runner:
        startup = await runner.collective_rpc("start_backend_runtime_probe", stage_ids=[0])
        output = await runner.send_token_ids_request(
            {"prompt_token_ids": list(range(10, 27)), "max_tokens": 4, "request_id": "backend-runtime"}
        )
        probe = (await runner.collective_rpc("get_backend_runtime_probe", stage_ids=[0]))[0][0]

    evidence = {"startup": startup, "probe": probe, "output_token_ids": list(output.outputs[0].token_ids)}
    evidence_path = tmp_path / "ar-backend-runtime.json"
    evidence_path.write_text(json.dumps(evidence, indent=2))
    print(f"AR backend runtime evidence: {evidence_path}")
    assert probe["linear_backend"] == "torch"
    assert probe["attention_backend"] == "TRITON_ATTN"
    assert probe["moe_backend"] == "triton"
    assert probe["model_class"] == "Qwen3MoeForCausalLM"
    assert probe["worker_pid"] != os.getpid()
    assert probe["model_dtype"] == "torch.bfloat16"
    assert probe["skip_tokenizer_init"]
    implementations = probe["implementations"]
    assert all(name.endswith("default_unquantized_gemm") for name in implementations["linear"].values())
    assert all(name.endswith("TritonAttentionImpl.forward") for name in implementations["attention"].values())
    assert all(name.endswith("TritonExperts.apply") for name in implementations["moe"].values())
    events = probe["events"]
    # A one-token prefill tail is still prefill: distinguish by scheduler
    # computed tokens, never by query tensor length alone.
    assert [(event["num_computed_tokens"], event["num_scheduled_tokens"], event["phase"]) for event in events] == [
        (0, 16, "prefill"),
        (16, 1, "prefill"),
        (17, 1, "decode"),
        (18, 1, "decode"),
        (19, 1, "decode"),
    ]
    for event in events:
        assert event["prompt_tokens"] == 17
        assert event["finite_hidden_states"]
        assert all(event["calls"][category] > 0 for category in ("linear", "attention", "moe"))
