# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");

"""
E2E test for vLLMOmniHttpServer generate flow.

This is a 1:1 replica of
``verl-omni/tests/workers/rollout/rollout_vllm/test_vllm_omni_generate.py``.
Every ``verl`` / ``verl_omni`` symbol used by the reference test \u2014
including ``vLLMHttpServer``, ``vLLMOmniHttpServer``, ``RolloutMode``,
``DiffusionOutput``, ``normalize_token_ids``, ``DiffusionRolloutConfig``,
``DiffusionModelConfig``, ``VllmOmniPipelineBase`` \u2014 is inlined in the
sibling file ``_inlined_server.py``. The test imports **nothing** from
``verl`` or ``verl_omni``.

Usage:
    pytest tests/e2e/offline_inference/rlhf_test/test_ray_async_omni_qwen_image_generate.py -v -s
"""

import os
from uuid import uuid4

import pytest
import ray
import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from tests.e2e.offline_inference.rlhf_test.rlhf_test_utils import (
    DiffusionOutput,
    RolloutMode,
    VllmOmniPipelineBase,
    normalize_token_ids,
    vLLMOmniHttpServer,
)

# HF repo IDs (auto-downloaded on first use; cached under HF_HOME).
MODEL = os.environ.get("VLLM_OMNI_TEST_MODEL", "tiny-random/Qwen-Image")
TOKENIZER_MODEL = os.environ.get("VLLM_OMNI_TEST_TOKENIZER", "Qwen/Qwen2-1.5B-Instruct")


def _resolve_model_path(repo_id: str) -> str:
    """Resolve an HF repo ID to a local snapshot path, downloading if needed."""
    # Allow overriding with a pre-existing local path (skips download).
    if os.path.isdir(repo_id):
        return repo_id
    return snapshot_download(repo_id=repo_id)


class vLLMOmniHttpServerForTest(vLLMOmniHttpServer):
    async def run_server(self, args):
        from tests.e2e.offline_inference.custom_pipeline.qwen_image_pipeline_with_logprob import (
            QwenImagePipelineWithLogProbForTest,
        )

        arch = getattr(self.model_config, "architecture", None)
        if arch and arch not in VllmOmniPipelineBase._registry:
            VllmOmniPipelineBase._registry[arch] = QwenImagePipelineWithLogProbForTest

        await super().run_server(args)


# ---------------------------------------------------------------------
#                \U0001f447 Test Helper Functions & Fixtures \U0001f447
# ---------------------------------------------------------------------

_MIN_PROMPT_TOKENS = 35


def _tokenize_prompt(text: str) -> list[int]:
    """Tokenize a text prompt into valid token IDs for the model."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, trust_remote_code=True)
    messages = [{"role": "user", "content": text}]
    token_ids = normalize_token_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    assert len(token_ids) > _MIN_PROMPT_TOKENS, (
        f"Prompt too short ({len(token_ids)} tokens, need >{_MIN_PROMPT_TOKENS}). "
        f"The pipeline drops the first 34 chat\u2011template prefix tokens; "
        f"use a longer prompt so content tokens remain after the drop."
    )
    return token_ids


@pytest.fixture
def init_server():
    """Create and launch a vLLMOmniHttpServer Ray actor with Qwen/Qwen-Image."""
    model_path = _resolve_model_path(MODEL)

    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
            }
        },
        ignore_reinit_error=True,
    )

    rollout_cfg = OmegaConf.create(
        {
            "_target_": "verl_omni.workers.config.diffusion.DiffusionRolloutConfig",
            "name": "vllm_omni",
            "mode": "async",
            "tensor_model_parallel_size": 1,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "gpu_memory_utilization": 0.8,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 256,
            "max_model_len": 1058,
            "dtype": "bfloat16",
            "load_format": "auto",
            "enforce_eager": True,
            "enable_chunked_prefill": False,
            "enable_prefix_caching": False,
            "enable_sleep_mode": False,
            "free_cache_engine": True,
            "disable_log_stats": True,
            "n": 4,
            "pipeline": {
                "_target_": "verl_omni.workers.config.diffusion.rollout.DiffusionPipelineConfig",
                "height": 512,
                "width": 512,
                "num_inference_steps": 10,
            },
        }
    )

    model_cfg = OmegaConf.create(
        {
            "_target_": "verl_omni.workers.config.diffusion.DiffusionModelConfig",
            "path": str(model_path),
            "tokenizer_path": TOKENIZER_MODEL,
            "trust_remote_code": True,
            "load_tokenizer": True,
        }
    )

    ServerCls = ray.remote(vLLMOmniHttpServerForTest)
    server = ServerCls.options(
        runtime_env={
            "env_vars": {
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                "NCCL_CUMEM_ENABLE": "0",
            }
        },
        max_concurrency=10,
    ).remote(
        config=rollout_cfg,
        model_config=model_cfg,
        rollout_mode=RolloutMode.STANDALONE,
        workers=[],
        replica_rank=0,
        node_rank=0,
        gpus_per_node=1,
        nnodes=1,
        cuda_visible_devices="0",
    )

    ray.get(server.launch_server.remote())

    yield server

    ray.shutdown()


def test_generate(init_server):
    """Concurrent generate() calls covering basic output, logprobs, and multi-request correctness."""
    server = init_server

    prompts = [
        "a beautiful sunset over the ocean with vibrant orange and purple clouds "
        "reflecting on the calm water surface near a rocky coastline",
        "a fluffy orange cat sitting on a wooden windowsill looking outside at "
        "a garden full of colorful flowers on a bright sunny afternoon",
        "a majestic mountain landscape covered with fresh white snow under a "
        "clear blue sky with pine trees in the foreground and a frozen lake",
        "a futuristic city at night with neon lights glowing on tall glass "
        "skyscrapers and flying vehicles soaring between the buildings",
    ]

    refs = []
    for i, prompt in enumerate(prompts):
        rid = f"test_{i}_{uuid4().hex[:8]}"
        ref = server.generate.remote(
            prompt_ids=_tokenize_prompt(prompt),
            sampling_params={
                "num_inference_steps": 10,
                "true_cfg_scale": 4.0,
                "height": 512,
                "width": 512,
                "logprobs": i == 0,  # first request includes logprobs
            },
            request_id=rid,
        )
        refs.append(ref)

    results = ray.get(refs, timeout=600)

    for i, output in enumerate(results):
        assert isinstance(output, DiffusionOutput), f"Request {i}: expected DiffusionOutput"
        assert len(output.diffusion_output) == 3, f"Request {i}: expected 3 channels (CHW)"
        h, w = len(output.diffusion_output[0]), len(output.diffusion_output[0][0])
        assert h > 0 and w > 0, f"Request {i}: image dimensions must be positive"
        assert output.stop_reason in ("completed", "aborted", None), f"Request {i}: unexpected stop_reason"
        assert 0.0 <= output.diffusion_output[0][0][0] <= 1.0, f"Request {i}: pixel values must be in [0, 1]"

    print(f"All {len(prompts)} concurrent requests returned valid DiffusionOutput")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])