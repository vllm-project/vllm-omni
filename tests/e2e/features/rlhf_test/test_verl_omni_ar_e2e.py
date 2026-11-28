# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""E2E test for the verl-omni AR rollout on the worker-v2 runner.

AR counterpart of ``test_verl_omni_e2e.py``: it follows the flow ``verl_omni``
drives for thinker rollouts, with the ``verl`` / ``verl_omni`` symbols inlined
(``tests.e2e.features.helpers.verl_omni_ar_server``), so the test needs
neither package.

Flow:

1. Build the rollout / model configs verl passes to ``vLLMOmniHttpServer``.
2. ``ray.remote(vLLMOmniHttpServerARLocal)`` then
   ``ray.get(server.launch_server.remote())``.
3. ``server.generate.remote(prompt_ids=..., sampling_params=..., request_id=...)``
   and assert the returned ``TokenOutput``.

Usage:
    pytest tests/e2e/features/rlhf_test/test_verl_omni_ar_e2e.py -v -s
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from functools import partial
from typing import Any
from uuid import uuid4

import pytest
import ray
import torch
from omegaconf import OmegaConf

from tests.e2e.features.helpers.verl_omni_ar_server import (
    TEACHER_HIDDEN_STATES_KEY,
    RolloutMode,
    TokenOutput,
    vLLMOmniHttpServerARLocal,
)
from tests.e2e.features.helpers.verl_omni_ar_worker import tensor_from_bytes
from tests.helpers.mark import hardware_test
from vllm_omni.transformers_utils.repo_utils import hf_api

MODEL = "tiny-random/Qwen3-Omni"
WORKER_V2_AR_RUNNER = "vllm_omni.worker_v2.omni_ar_model_runner.OmniARModelRunner"
# Every prompt fits one prefill step, as verl-omni's teacher engines require.
MAX_MODEL_LEN = 1024


def _resolve_model_path(repo_id: str) -> str:
    if os.path.isdir(repo_id):
        return repo_id
    return hf_api().snapshot_download(repo_id=repo_id)


def _text_config(model_path: str) -> dict:
    with open(os.path.join(model_path, "config.json")) as handle:
        return json.load(handle)["thinker_config"]["text_config"]


def _token_ids(text_config: dict, seed: int, length: int) -> list[int]:
    """Deterministic prompt of ``length`` ids that shares its first block with no other seed."""
    vocab_size = text_config["vocab_size"]
    return [3 + (seed * 7919 + index * 31) % (vocab_size - 3) for index in range(length)]


def _generate(server, prompt_ids: list[int], sampling_params: dict[str, Any], label: str) -> tuple[str, Any]:
    """Submit the way verl does; returns the external request id and the ObjectRef."""
    request_id = f"{label}_{uuid4().hex[:8]}"
    return request_id, server.generate.remote(
        prompt_ids=prompt_ids,
        sampling_params=sampling_params,
        request_id=request_id,
    )


def _is_logprob(value: float) -> bool:
    return math.isfinite(value) and value <= 1e-6


def _prefill_rows(steps: list[dict[str, Any]], request_id_prefix: str, sequence: list[int]) -> torch.Tensor:
    """The request's prefill rows in the recorded step that completed its prefill."""
    hits = [
        (step, req_id)
        for step in steps
        for req_id in step["req_ids"]
        if req_id.startswith(request_id_prefix) and step["num_computed_prefill"][step["req_ids"].index(req_id)] == 0
    ]
    ((step, req_id), *extra) = hits
    assert not extra, "an opted-in request must complete its prefill in exactly one step"
    index = step["req_ids"].index(req_id)
    assert step["prefill_len"][index] == len(sequence)
    assert step["num_scheduled"][index] >= len(sequence)
    start = step["query_start_loc"][index]
    return tensor_from_bytes(step["hidden"])[start : start + len(sequence)]


@pytest.fixture(scope="module")
def init_server(request: pytest.FixtureRequest):
    """Create and launch a vLLMOmniHttpServerARLocal Ray actor with the thinker-only pipeline."""
    model_path = _resolve_model_path(MODEL)
    # The native AMD runner nests TMPDIR under a Buildkite job UUID. Ray adds
    # its session and socket suffixes, which can exceed Linux's 107-byte
    # AF_UNIX path limit before the test reaches the GPU. Keep Ray's root
    # short, unique, and scoped to this fixture.
    ray_temp_dir = tempfile.mkdtemp(prefix="ray-", dir="/tmp")
    request.addfinalizer(partial(shutil.rmtree, ray_temp_dir, ignore_errors=True))

    ray.init(
        _temp_dir=ray_temp_dir,
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
            "gpu_memory_utilization": 0.5,
            "max_num_batched_tokens": MAX_MODEL_LEN,
            "max_num_seqs": 8,
            "max_model_len": MAX_MODEL_LEN,
            "prompt_length": 512,
            "response_length": 64,
            "dtype": "bfloat16",
            "load_format": "auto",
            "disable_log_stats": True,
            "enable_chunked_prefill": False,
            "enable_prefix_caching": False,
            # The RL loop sleeps the rollout engine between phases.
            "enable_sleep_mode": True,
            "logprobs_mode": "processed_logprobs",
            "scheduling_policy": "fcfs",
            "seed": 0,
            "engine_kwargs": {
                "vllm_omni": {
                    "output_mode": "ar",
                    "pipeline_name": "qwen3_omni_moe",
                    "pipeline_mode": "thinker_only",
                    # tiny-random/Qwen3-Omni keeps a 14px vision config while the
                    # Qwen3-Omni vision tower patches at 16px, which only the dummy
                    # multimodal profiling trips. These rollouts are token-only.
                    "skip_mm_profiling": True,
                }
            },
        }
    )
    model_cfg = OmegaConf.create({"path": str(model_path), "local_path": str(model_path), "trust_remote_code": True})

    ServerCls = ray.remote(vLLMOmniHttpServerARLocal)
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

    try:
        ray.get(server.launch_server.remote(), timeout=1200)
        # The contracts under test are the worker-v2 ones; fail fast if another runner serves the stage.
        (runner_classes,) = ray.get(server.collective_rpc.remote("runner_class"), timeout=60)
        assert runner_classes == [WORKER_V2_AR_RUNNER]
        yield server
    finally:
        try:
            ray.get(server.shutdown.remote(), timeout=60)
        finally:
            try:
                ray.kill(server, no_restart=True)
            finally:
                ray.shutdown()


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_ar_rollout_returns_tokens_and_logprobs(init_server):
    """Concurrent student rollouts: token ids, one finite logprob per token, verl stop reason."""
    text_config = _text_config(_resolve_model_path(MODEL))
    refs = [
        _generate(
            init_server,
            _token_ids(text_config, seed, 24 + 8 * seed),
            {"max_tokens": 8, "temperature": 1.0, "logprobs": True},
            f"rollout_{seed}",
        )[1]
        for seed in range(4)
    ]

    outputs = ray.get(refs, timeout=600)

    for output in outputs:
        assert isinstance(output, TokenOutput)
        assert 1 <= len(output.token_ids) <= 8
        assert output.log_probs is not None and len(output.log_probs) == len(output.token_ids)
        assert all(_is_logprob(logprob) for logprob in output.log_probs)
        assert output.stop_reason == "completed"
        assert output.extra_fields["global_steps"] == 0
        assert TEACHER_HIDDEN_STATES_KEY not in output.extra_fields


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_ar_teacher_scores_prompt_logprobs(init_server):
    """Teacher logprob mode: ``max_tokens=1, prompt_logprobs=0`` scores the sequence it was given."""
    text_config = _text_config(_resolve_model_path(MODEL))
    sequence = _token_ids(text_config, 11, 40)

    _, ref = _generate(init_server, sequence, {"max_tokens": 1, "prompt_logprobs": 0}, "teacher_logprobs")
    output = ray.get(ref, timeout=600)

    assert output.log_probs is None
    scored = output.extra_fields
    # One row per scored token plus the dummy row verl appends for the last token.
    assert scored["prompt_ids"][:-1] == [[token_id] for token_id in sequence[1:]]
    assert scored["prompt_ids"][-1] == [0]
    assert scored["prompt_logprobs"][-1] == [0.0]
    assert all(_is_logprob(row[0]) for row in scored["prompt_logprobs"][:-1])


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_ar_teacher_hidden_states_match_the_forward_pass(init_server):
    """Teacher hidden-state mode: opted-in requests get their prefill rows once; plain ones get none.

    The expected rows come from a probe that records each step's forward
    output and batch layout before sampling, independently of the opt-in code.
    """
    text_config = _text_config(_resolve_model_path(MODEL))
    hidden_size = text_config["hidden_size"]
    (acked,) = ray.get(init_server.collective_rpc.remote("start_hidden_state_recording"), timeout=60)
    assert acked == [True]

    opt_in = {"max_tokens": 1, "temperature": 1.0, "return_hidden_states": True}
    single_step_ids = _token_ids(text_config, 23, 48)
    decode_ids = _token_ids(text_config, 31, 20)
    single_step_id, single_step = _generate(init_server, single_step_ids, opt_in, "teacher_hidden")
    # An opted-in request that decodes after its prefill must not re-emit or fail.
    decode_id, with_decodes = _generate(init_server, decode_ids, {**opt_in, "max_tokens": 4}, "teacher_hidden_decodes")
    _, plain = _generate(init_server, _token_ids(text_config, 37, 17), {"max_tokens": 4, "temperature": 1.0}, "plain")
    single_output, decode_output, plain_output = ray.get([single_step, with_decodes, plain], timeout=600)

    for output, sequence in ((single_output, single_step_ids), (decode_output, decode_ids)):
        hidden = output.extra_fields[TEACHER_HIDDEN_STATES_KEY]
        assert isinstance(hidden, torch.Tensor) and hidden.device.type == "cpu"
        assert hidden.shape == (len(sequence), hidden_size)
        assert bool(torch.isfinite(hidden).all())
        assert output.stop_reason == "completed"
    assert TEACHER_HIDDEN_STATES_KEY not in plain_output.extra_fields

    # collective_rpc returns one list per stage, holding each rank's result.
    ((steps,),) = ray.get(init_server.collective_rpc.remote("take_hidden_state_recording"), timeout=60)
    for output, request_id, sequence in (
        (single_output, single_step_id, single_step_ids),
        (decode_output, decode_id, decode_ids),
    ):
        torch.testing.assert_close(
            output.extra_fields[TEACHER_HIDDEN_STATES_KEY],
            _prefill_rows(steps, request_id, sequence),
            rtol=0,
            atol=0,
        )


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_ar_sleep_wake_keeps_serving_the_rollout(init_server):
    """RL loop phases on worker v2: rollout, sleep, wake, rollout again."""
    text_config = _text_config(_resolve_model_path(MODEL))
    sequence = _token_ids(text_config, 41, 32)
    sampling_params = {"max_tokens": 6, "temperature": 0.0}

    _, before_ref = _generate(init_server, sequence, sampling_params, "before_sleep")
    before = ray.get(before_ref, timeout=600)
    ray.get(init_server.sleep.remote(), timeout=300)
    ray.get(init_server.wake_up.remote(), timeout=300)
    _, after_ref = _generate(init_server, sequence, sampling_params, "after_wake")
    after = ray.get(after_ref, timeout=600)

    assert before.token_ids and after.token_ids == before.token_ids
    assert after.log_probs is None and after.stop_reason == "completed"
