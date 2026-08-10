# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E test that follows the EXACT same flow as
``verl-omni/tests/workers/rollout/rollout_vllm/test_vllm_omni_generate.py``
but inlines all the ``verl`` / ``verl_omni`` symbols the flow depends on
(stripped down to only what the diffusion-mode test actually exercises).

Flow (identical to the verl-omni test):

1. Build two ``OmegaConf`` configs (``DiffusionRolloutConfig`` /
   ``DiffusionModelConfig`` shape).
2. ``ServerCls = ray.remote(vLLMOmniHttpServerLocal)`` then
   ``ServerCls.options(...).remote(config=..., model_config=...,
   rollout_mode=RolloutMode.STANDALONE, workers=[], replica_rank=0,
   node_rank=0, gpus_per_node=1, nnodes=1, cuda_visible_devices="0")``.
3. ``ray.get(server.launch_server.remote())``.
4. ``server.generate.remote(prompt_ids=..., sampling_params=..., request_id=...)``
   and assert the returned ``DiffusionOutput``.

The Ray actor lives in ``tests.e2e.features.helpers.verl_omni_server`` so
workers can import it after directory migration (pytest test modules are not
importable as ``test_verl_omni_e2e`` in Ray child processes).

Usage:
    pytest tests/e2e/features/rlhf_test/test_verl_omni_e2e.py -v -s
"""

from __future__ import annotations

import os
from functools import lru_cache
from uuid import uuid4

import pytest
import ray
import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from tests.e2e.features.helpers.verl_omni_server import (
    DiffusionOutput,
    RolloutMode,
    normalize_token_ids,
    vLLMOmniHttpServerLocal,
)
from tests.helpers.mark import hardware_test

MODEL = "tiny-random/Qwen-Image"
TOKENIZER_MODEL = "Qwen/Qwen2-1.5B-Instruct"
_MIN_PROMPT_TOKENS = 35


def _resolve_model_path(repo_id: str) -> str:
    if os.path.isdir(repo_id):
        return repo_id
    return snapshot_download(repo_id=repo_id)


@lru_cache(maxsize=1)
def _get_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_MODEL, trust_remote_code=True)


def _tokenize_prompt(text: str) -> list[int]:
    tokenizer = _get_tokenizer()
    messages = [{"role": "user", "content": text}]
    token_ids = normalize_token_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    assert len(token_ids) > _MIN_PROMPT_TOKENS, (
        f"Prompt too short ({len(token_ids)} tokens, need >{_MIN_PROMPT_TOKENS}). "
        "The pipeline drops the first 34 chat-template prefix tokens; "
        "use a longer prompt so content tokens remain after the drop."
    )
    return token_ids


@pytest.fixture
def init_server():
    """Create and launch a vLLMOmniHttpServerLocal Ray actor with Qwen-Image."""
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
            "name": "vllm_omni",
            "mode": "async",
            "tensor_model_parallel_size": 1,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "gpu_memory_utilization": 0.8,
            "max_num_batched_tokens": 8192,
            # QwenImagePipeline (and the logprob custom subclass) only supports
            # serial request execution; max_num_seqs>1 enables request-level batching.
            "max_num_seqs": 1,
            "max_model_len": 1058,
            "dtype": "bfloat16",
            "load_format": "auto",
            "enforce_eager": True,
            "enable_chunked_prefill": False,
            "enable_prefix_caching": False,
            "enable_sleep_mode": False,
            "free_cache_engine": True,
            "disable_log_stats": True,
            "logprobs_mode": "processed_logprobs",
            "scheduling_policy": "fcfs",
            "seed": 0,
            "n": 4,
            "pipeline": {
                "height": 512,
                "width": 512,
                "num_inference_steps": 10,
            },
        }
    )

    model_cfg = OmegaConf.create(
        {
            "path": str(model_path),
            "local_path": str(model_path),
            "tokenizer_path": TOKENIZER_MODEL,
            "trust_remote_code": True,
            "load_tokenizer": True,
        }
    )

    ServerCls = ray.remote(vLLMOmniHttpServerLocal)
    server = ServerCls.options(
        runtime_env={
            "env_vars": {
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                "NCCL_CUMEM_ENABLE": "0",
            }
        },
        max_concurrency=10,
        num_gpus=1,
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


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
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
                "logprobs": i == 0,
            },
            request_id=rid,
        )
        refs.append(ref)

    results = ray.get(refs, timeout=600)

    for i, output in enumerate(results):
        assert isinstance(output, DiffusionOutput), f"Request {i}: expected DiffusionOutput"
        img = output.diffusion_output
        assert isinstance(img, torch.Tensor) and img.ndim == 3, (
            f"Request {i}: expected CHW torch.Tensor, got {type(img).__name__} ndim={getattr(img, 'ndim', None)}"
        )
        c, h, w = img.shape
        assert c == 3, f"Request {i}: expected 3 channels (CHW), got {c}"
        assert h > 0 and w > 0, f"Request {i}: image dimensions must be positive"
        assert output.stop_reason in ("completed", "aborted", None), (
            f"Request {i}: unexpected stop_reason {output.stop_reason!r}"
        )
        assert 0.0 <= float(img[0, 0, 0]) <= 1.0, f"Request {i}: pixel values must be in [0, 1]"

    print(f"All {len(prompts)} concurrent requests returned valid DiffusionOutput")
