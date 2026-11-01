# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Dense decode sampler graphs captured and parity-checked during startup.

Capture compares eager and captured state transitions with identical noise.
Serving only replays warmed shapes and preserves request torch RNG draws. CPU payload staging
remains outside capture. This module does not handle mixed prefill batches.
"""

import copy
import time

import torch
from vllm.logger import init_logger
from vllm.v1.outputs import SamplerOutput

logger = init_logger(__name__)

_STATE = (
    "_decode_last_codes",
    "_decode_has_codes",
    "_decode_delay_count",
    "_decode_eoc_countdown",
    "_decode_generation_done",
)


_BUCKETS = (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 96, 128)
_FIELDS = ("temperature", "top_k", "top_p")


def _graph_key(model, batch, hidden, force_audio_inputs, metadata):
    return (
        batch,
        hidden.dtype,
        hidden.device,
        bool(force_audio_inputs),
        bool(metadata.all_greedy),
        tuple(getattr(metadata, f) is None for f in _FIELDS),
        tuple(getattr(model, name).data_ptr() for name in _STATE),
    )


def _replay_captured(model, hidden, metadata, force_audio_inputs):
    """Replay an already captured (bucket) graph writing inputs in place.

    The capture path below pads through temporaries on every call. Once a
    graph exists, only the live rows are written into its static inputs:
    padded rows keep finite values from earlier steps and own no request
    state, forced-audio inputs are constants, sampling parameters are copied
    only when the cached metadata changes, and request noise is drawn directly
    into the graph's noise input.
    """
    cache = getattr(model, "_dense_sample_graphs", None)
    if not cache:
        return None
    batch = hidden.shape[0]
    bucket = batch
    if getattr(model.config, "audio_mrv2_sample_graph_buckets", False):
        bucket = next((n for n in _BUCKETS if n >= batch), batch)
    model._ensure_decode_state_capacity(bucket, hidden.device)
    key = _graph_key(model, bucket, hidden, force_audio_inputs, metadata)
    entry = cache.get(key)
    if entry is None:
        return None
    graph, static_hidden, static_input_ids, static_noise, static_metadata, output, staging, keepalive = entry
    books = model.num_codebooks
    static_hidden[:batch].copy_(hidden)
    if not force_audio_inputs:
        static_input_ids[:batch].copy_(model._last_step_input_ids)
        if batch < bucket:
            static_input_ids[batch:].fill_(model._audio_continuation_id)
    if metadata.generators:
        from vllm_omni.utils.seeded_exponential import fill_exponential_rows

        fill_exponential_rows(static_noise[: batch * books], [metadata.generators.get(i) for i in range(batch)])
    else:
        static_noise[: batch * books].exponential_()
    sources = getattr(model, "_dense_sample_metadata_sources", None)
    if sources is None:
        sources = model._dense_sample_metadata_sources = {}
    if sources.get(key) is not metadata:
        for name in _FIELDS:
            value = getattr(metadata, name)
            if value is not None:
                getattr(static_metadata, name)[:batch].copy_(value)
        sources[key] = metadata
    graph.replay()
    model._finish_audio_staging(staging[:batch], batch, hidden.device, batch)
    return SamplerOutput(sampled_token_ids=output.sampled_token_ids[:batch].clone(), logprobs_tensors=None)


def run_dense_sample(model, hidden, logits, metadata, force_audio_inputs=False, active_batch=None):
    batch = hidden.shape[0]
    if active_batch is None:
        replayed = _replay_captured(model, hidden, metadata, force_audio_inputs)
        if replayed is not None:
            return replayed
    # Serving never captures a new shape. Unsupported shapes/metadata use the
    # regular sampler; startup owns capture and its parity checks.
    if not getattr(model, "_warming_sample_graphs", False):
        return None
    if active_batch is None and getattr(model.config, "audio_mrv2_sample_graph_buckets", False):
        bucket = next((n for n in (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 96, 128) if n >= batch), batch)
        if bucket != batch:
            # Dummy rows own no request state. The batch-state gather replaces
            # these rows before any later request can use them.
            model._ensure_decode_state_capacity(bucket, hidden.device)
            padded_hidden = hidden.new_zeros((bucket, hidden.shape[1]))
            padded_hidden[:batch].copy_(hidden)
            padded_metadata = copy.copy(metadata)
            for name in ("temperature", "top_k", "top_p"):
                value = getattr(metadata, name)
                if value is not None:
                    padded = value.new_ones(bucket)
                    padded[:batch].copy_(value)
                    setattr(padded_metadata, name, padded)
            original_ids = model._last_step_input_ids
            padded_ids = original_ids.new_full((bucket,), model._audio_continuation_id)
            if not force_audio_inputs:
                padded_ids[:batch].copy_(original_ids)
            model._last_step_input_ids = padded_ids
            try:
                output = run_dense_sample(
                    model,
                    padded_hidden,
                    logits.new_empty((bucket, logits.shape[1])),
                    padded_metadata,
                    force_audio_inputs=force_audio_inputs,
                    active_batch=batch,
                )
            finally:
                model._last_step_input_ids = original_ids
            return SamplerOutput(sampled_token_ids=output.sampled_token_ids[:batch], logprobs_tensors=None)
    actual = batch if active_batch is None else active_batch
    effective_ids = (
        torch.full((batch,), model._audio_continuation_id, dtype=model._last_step_input_ids.dtype, device=hidden.device)
        if force_audio_inputs
        else model._last_step_input_ids
    )
    books = model.num_codebooks
    vocab = model.modality_head.vocab_size
    noise = torch.empty((batch * books, vocab), dtype=torch.float32, device=hidden.device)
    if actual < batch:
        noise[actual * books :].fill_(1)
    if metadata.generators:
        from vllm_omni.utils.seeded_exponential import fill_exponential_rows

        fill_exponential_rows(noise[: actual * books], [metadata.generators.get(i) for i in range(actual)])
    else:
        noise[: actual * books].exponential_()
    fields = _FIELDS
    # force_audio_inputs controls capture-time Python branches (including
    # the mixed-tail assertion). Never reuse that graph for real EOS inputs.
    key = _graph_key(model, batch, hidden, force_audio_inputs, metadata)
    cache = getattr(model, "_dense_sample_graphs", None)
    if cache is None:
        cache = model._dense_sample_graphs = {}
    entry = cache.get(key)
    if entry is None:
        static_hidden = hidden.clone()
        static_input_ids = effective_ids.clone()
        static_noise = noise.clone()
        static_metadata = copy.copy(metadata)
        for name in fields:
            value = getattr(metadata, name)
            setattr(static_metadata, name, None if value is None else value.clone())
        # Ensure the reusable staging exists before capture. Each graph keeps
        # its own reference if a later larger batch grows the model buffer.
        staging = model._get_audio_gpu_staging_buffer(batch, hidden.device)
        state = {name: getattr(model, name).clone() for name in _STATE}
        original_ids = model._last_step_input_ids
        original_qsl = model._last_step_query_start_loc
        original_mode_rows = getattr(model, "_step_audio_mode_rows", 0)
        original_tail_rows = getattr(model, "_step_audio_tail_rows", 0)
        original_direct_rows = getattr(model, "_fast_audio_direct_rows", 0)
        # Dense decode has one token per row. Own this input instead of
        # capturing a view of the runner's resizable staging buffer.
        static_qsl = torch.arange(batch + 1, dtype=torch.int32, device=hidden.device)

        def restore():
            for name, value in state.items():
                getattr(model, name).copy_(value)

        try:
            model._in_audio_sample_graph = True
            # Outer eligibility and real-input validation already ran. The
            # owned graph inputs represent exactly one audio-state row each.
            # Normalize capture-time host dispatch; never capture an outer
            # mixed-batch assertion or a stock-sampler branch into this graph.
            model._step_audio_mode_rows = 0
            model._step_audio_tail_rows = 0
            model._fast_audio_direct_rows = batch
            model._audio_graph_noise = static_noise
            model._last_step_input_ids = static_input_ids
            model._last_step_query_start_loc = static_qsl
            model._last_logits_hidden = static_hidden
            reference = model.sample(logits, static_metadata)
            expected_ids = reference.sampled_token_ids.clone()
            expected_staging = staging.clone()
            expected_state = {name: getattr(model, name).clone() for name in _STATE}
            restore()
            model._last_logits_hidden = static_hidden
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = model.sample(logits, static_metadata)
            # CUDA capture records work; it does not execute the transition.
            # Validate only after replay from the preserved initial state.
            restore()
            graph.replay()
            torch.testing.assert_close(output.sampled_token_ids, expected_ids, atol=0, rtol=0)
            torch.testing.assert_close(staging, expected_staging, atol=0, rtol=0)
            for name, expected in expected_state.items():
                torch.testing.assert_close(getattr(model, name), expected, atol=0, rtol=0)
            restore()
            logger.debug("Higgs sample graph parity checked for batch=%d", batch)
        finally:
            model._in_audio_sample_graph = False
            model._step_audio_mode_rows = original_mode_rows
            model._step_audio_tail_rows = original_tail_rows
            model._fast_audio_direct_rows = original_direct_rows
            model._audio_graph_noise = None
            model._last_step_input_ids = original_ids
            model._last_step_query_start_loc = original_qsl
            model._last_logits_hidden = None
        # Graphs reference storage, not Python owners. Model caches and state
        # can grow after capture; keep their original allocations alive.
        keepalive = [getattr(model, name) for name in _STATE]
        for name in ("_row_index_cache", "_codebook_index_cache", "_boc_frame_cache"):
            keepalive.extend(getattr(model, name).values())
        keepalive.append(static_qsl)
        entry = (graph, static_hidden, static_input_ids, static_noise, static_metadata, output, staging, keepalive)
        cache[key] = entry
    graph, static_hidden, static_input_ids, static_noise, static_metadata, output, staging, keepalive = entry
    static_hidden.copy_(hidden)
    static_input_ids.copy_(effective_ids)
    static_noise.copy_(noise)
    for name in fields:
        value = getattr(metadata, name)
        if value is not None:
            getattr(static_metadata, name).copy_(value)
    graph.replay()
    model._finish_audio_staging(staging[:actual], actual, hidden.device, actual)
    return SamplerOutput(sampled_token_ids=output.sampled_token_ids.clone(), logprobs_tensors=None)


@torch.inference_mode()
def capture_sample_graphs(model, max_num_reqs):
    """Warm the MRv2 sampler without consuming request RNG or decode state."""
    if not model._sample_graph_enabled or getattr(model, "_sample_graphs_ready", False):
        return
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.metadata import SamplingMetadata

    device = model.model.embed_tokens.weight.device
    if device.type != "cuda":
        return
    start = time.perf_counter()
    memory_before = torch.accelerator.memory_allocated(device)
    reserved_before = torch.accelerator.memory_reserved(device)
    model._resolve_token_ids()
    buckets = (
        sorted({next((b for b in _BUCKETS if b >= n), n) for n in range(1, max_num_reqs + 1)})
        if getattr(model.config, "audio_mrv2_sample_graph_buckets", False)
        else range(1, max_num_reqs + 1)
    )
    model._ensure_decode_state_capacity(max(buckets), device)
    model._get_audio_gpu_staging_buffer(max(buckets), device)
    saved_state = {name: getattr(model, name).clone() for name in _STATE}
    fields = (
        "_last_step_input_ids",
        "_last_step_query_start_loc",
        "_last_logits_hidden",
        "_step_audio_mode_rows",
        "_step_audio_tail_rows",
        "_fast_audio_direct_rows",
    )
    saved = {name: getattr(model, name, None) for name in fields}
    try:
        model._warming_sample_graphs = True
        with torch.random.fork_rng(devices=[device.index]):
            for batch in buckets:
                hidden = model.model.embed_tokens.weight.new_zeros((batch, model.model.embed_tokens.weight.shape[1]))
                logits = hidden.new_empty((batch, model.config.text_config.vocab_size))
                model._last_step_input_ids = torch.full(
                    (batch,), model._audio_continuation_id, device=device, dtype=torch.long
                )
                model._last_step_query_start_loc = torch.arange(batch + 1, device=device, dtype=torch.int32)
                ones = torch.ones(batch, device=device)
                zeros = torch.zeros_like(ones)
                for greedy in (False, True):
                    metadata = SamplingMetadata(
                        temperature=zeros if greedy else ones,
                        all_greedy=greedy,
                        all_random=not greedy,
                        top_p=ones,
                        top_k=torch.full((batch,), -1, device=device, dtype=torch.long),
                        generators={},
                        max_num_logprobs=None,
                        no_penalties=True,
                        prompt_token_ids=None,
                        frequency_penalties=zeros,
                        presence_penalties=zeros,
                        repetition_penalties=ones,
                        output_token_ids=[[] for _ in range(batch)],
                        allowed_token_ids_mask=None,
                        bad_words_token_ids={},
                        logitsprocs=LogitsProcessors(),
                    )
                    for forced in (False, True):
                        run_dense_sample(model, hidden, logits, metadata, force_audio_inputs=forced)
    finally:
        for name, value in saved_state.items():
            getattr(model, name).copy_(value)
        for name, value in saved.items():
            setattr(model, name, value)
        model._warming_sample_graphs = False
        model._clear_last_audio_outputs()
        model._async_audio_gpu_staging = None
        model._dense_sample_metadata_sources = {}
    torch.accelerator.synchronize()
    model._sample_graphs_ready = True
    logger.info(
        "Higgs sampler captured %d CUDA graphs in %.2f s "
        "(allocated delta %.1f MiB, reserved delta %.1f MiB, reserved total %.1f MiB)",
        len(model._dense_sample_graphs),
        time.perf_counter() - start,
        (torch.accelerator.memory_allocated(device) - memory_before) / 2**20,
        (torch.accelerator.memory_reserved(device) - reserved_before) / 2**20,
        torch.accelerator.memory_reserved(device) / 2**20,
    )
