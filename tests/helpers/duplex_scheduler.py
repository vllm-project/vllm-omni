# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""A real Stage 0 duplex scheduler and step helpers for PersonaPlex-shaped sessions."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import LlamaConfig

# vllm_omni must patch vllm.v1.request before Request is bound here.
# isort: off
from vllm_omni.config import OmniModelConfig
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.engine import EngineCoreOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, SlidingWindowSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

# isort: on

# The Stage 0 context PersonaPlex served when the 3001 > 3000 crash was
# reported: max_model_len equal to the temporal transformer's 3000-frame window.
DUPLEX_MAX_MODEL_LEN = 3000
DUPLEX_BLOCK_SIZE = 16
DUPLEX_MAX_NUM_SEQS = 4


def make_engine_duplex_scheduler(
    tmp_path: Path,
    *,
    max_model_len: int = DUPLEX_MAX_MODEL_LEN,
    sliding_window: int = DUPLEX_MAX_MODEL_LEN,
) -> OmniARScheduler:
    """A Stage-0 duplex scheduler built from real vLLM configs with a real KV
    cache manager, so admission, chunked prefill, the resumable stop and the
    streaming update all run production code. The KV spec mirrors Helium's
    per-layer sliding window; the pool holds ``DUPLEX_MAX_NUM_SEQS`` full contexts."""
    LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=max_model_len,
    ).save_pretrained(tmp_path)
    model_config = ModelConfig(
        model=str(tmp_path), skip_tokenizer_init=True, dtype="float16", max_model_len=max_model_len
    )
    vllm_config = VllmConfig(
        model_config=OmniModelConfig.from_vllm_model_config(model_config, stage_id=0, session_mode="duplex"),
        # PersonaPlex Stage 0: synchronous scheduling, 512-token budget.
        scheduler_config=SchedulerConfig(
            max_num_seqs=DUPLEX_MAX_NUM_SEQS,
            max_num_batched_tokens=512,
            max_model_len=max_model_len,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            watermark=0.0,
        ),
        cache_config=CacheConfig(block_size=DUPLEX_BLOCK_SIZE, enable_prefix_caching=False),
    )
    num_blocks = DUPLEX_MAX_NUM_SEQS * (max_model_len // DUPLEX_BLOCK_SIZE + 2)
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    spec = SlidingWindowSpec(
        block_size=DUPLEX_BLOCK_SIZE,
        num_kv_heads=2,
        head_size=32,
        dtype=torch.float16,
        sliding_window=sliding_window,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layers.0.attn"], spec)],
    )
    # vLLM fills in its built-in KV managers only while the spec registry is
    # empty, and a test module may already have imported a model package that
    # registers its own spec. Registering the built-ins again is idempotent.
    register_all_kvcache_specs(vllm_config)
    return OmniARScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=DUPLEX_BLOCK_SIZE,
        log_stats=False,
    )


def duplex_append(request_id: str, num_tokens: int) -> Request:
    """One PersonaPlex append: ``num_tokens`` placeholder slots (the join also
    carries the persona prefill) and one greedy token per frame."""
    return Request(
        request_id=request_id,
        prompt_token_ids=[0] * num_tokens,
        sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
        pooling_params=None,
        arrival_time=0.0,
        block_hasher=None,
        resumable=True,
    )


def engine_step(sched: OmniARScheduler) -> tuple[dict[str, int], dict[str, EngineCoreOutput]]:
    """Run one step against a worker that books samples like the GPU runner.

    A request whose scheduled span completes its prompt samples one token at
    position ``num_tokens + 1``, which ``GPUModelRunner._bookkeeping_sync``
    asserts is within ``max_model_len``; the assertion text is the one a
    session past the context hits on a real engine. Returns each sampling
    request's prompt length and the step's outputs by request id."""
    scheduler_output = sched.schedule()
    req_ids = list(scheduler_output.num_scheduled_tokens)
    sampled_token_ids: list[list[int]] = []
    sampled_at: dict[str, int] = {}
    for req_id in req_ids:
        request = sched.requests[req_id]
        if request.num_computed_tokens < request.num_tokens:
            sampled_token_ids.append([])  # a prefill chunk short of the prompt end
            continue
        end_idx = request.num_tokens + 1
        assert end_idx <= sched.max_model_len, (
            "Sampled token IDs exceed the max model length. "
            f"Total number of tokens: {end_idx} > max_model_len: {sched.max_model_len}"
        )
        sampled_at[req_id] = request.num_tokens
        sampled_token_ids.append([3])
    model_runner_output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
        sampled_token_ids=sampled_token_ids,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    engine_core_outputs = sched.update_from_output(scheduler_output, model_runner_output)
    outputs = {output.request_id: output for eco in engine_core_outputs.values() for output in eco.outputs}
    return sampled_at, outputs


def join_duplex_sessions(sched: OmniARScheduler, joins: dict[str, int]) -> None:
    """Admit each session's first append and step until every one has sampled."""
    for request_id, num_tokens in joins.items():
        sched.add_request(duplex_append(request_id, num_tokens))
    pending = set(joins)
    while pending:
        sampled_at, _ = engine_step(sched)
        pending -= set(sampled_at)


def append_duplex_frame(sched: OmniARScheduler) -> tuple[dict[str, int], dict[str, EngineCoreOutput]]:
    """Every live session appends one frame; the step that follows runs them."""
    for request_id, request in list(sched.requests.items()):
        if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            sched.add_request(duplex_append(request_id, 1))
    return engine_step(sched)
