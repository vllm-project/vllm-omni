# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""KDA layers of Ming3 steps that carry a prefill.

vLLM's Bailing-MoE-v3 KDA layer runs every row of such a step through the
chunked kernels, launched from Python in every layer. At high concurrency the
step is mostly decode rows, and the launches take longer on the host than the
kernels take on the GPU. Here:

- As vLLM's Qwen GDN layer does, the leading decode rows advance their state
  in place with the recurrent kernel of decode-only steps; only the prefill
  rows' states are gathered, chunked and scattered back. Decode rows get the
  same kernel whether or not a prefill shares their step.
- Two host waits are gone. New sequences' zero initial state is selected on
  the device instead of zeroing rows found with a boolean mask, and FLA's
  ``chunk_kda`` takes the chunk indices the attention metadata built on the
  host instead of deriving them from the device ``cu_seqlens`` with
  ``.tolist()``.
- The chunk kernels are launched from Python once per step and replayed as a
  CUDA graph by the step's other KDA layers (``_StepChunks``).

Every change keeps the kernels and their inputs, so results are unchanged up
to the decode rows' kernel. Decode-only and speculative steps keep the
upstream path. The override is bound to this model's layer instances only.
"""

from __future__ import annotations

import inspect
import types

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.models import bailing_moe_v3 as upstream
from vllm.third_party.flash_linear_attention.ops import (
    chunk_delta_h,
    chunk_o,
    chunk_scaled_dot_kkt,
    cumsum,
    index,
    kda,
    solve_tril,
    wy_fast,
)
from vllm.third_party.flash_linear_attention.ops.kda import chunk_kda, fused_recurrent_kda
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

logger = init_logger(__name__)

_UPSTREAM_FORWARD = upstream.BailingMoeV3KimiDeltaAttention._forward
# The override reproduces the upstream prefill branch around this statement.
_UPSTREAM_MARKER = "zero_idx = state_indices[~has_initial_state]"

# FLA modules that derive chunk indices from ``cu_seqlens`` when not given them.
_CHUNK_INDEX_USERS = (kda, chunk_delta_h, chunk_o, chunk_scaled_dot_kkt, cumsum, solve_tril, wy_fast)
_derive_chunk_indices = index.prepare_chunk_indices
# (cu_seqlens, chunk indices) of the KDA call in flight; matched by identity.
_known_chunk_indices: tuple[torch.Tensor, torch.Tensor] | None = None


def _prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    known = _known_chunk_indices
    if known is not None and cu_seqlens is known[0] and chunk_size == FLA_CHUNK_SIZE:
        return known[1]
    return _derive_chunk_indices(cu_seqlens, chunk_size)


def _recurrent_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    attn_metadata,
    core_attn_out: torch.Tensor,
    step: object,
    **gate_options,
) -> None:
    """Delta-rule attention of a non-speculative step with prefills, decode rows first.

    ``step`` identifies the model step (its forward context): the KDA layers
    of one step can belong to different KV cache groups, each with its own
    metadata object.
    """
    num_actual_tokens = q.shape[1]
    num_decodes = attn_metadata.num_decodes
    if num_decodes > 0:
        decode_out, _ = fused_recurrent_kda(
            q=q[:, :num_decodes],
            k=k[:, :num_decodes],
            v=v[:, :num_decodes],
            g=g[:, :num_decodes],
            beta=beta[:, :num_decodes],
            initial_state=recurrent_state,
            use_qk_l2norm_in_kernel=True,
            **gate_options,
            cu_seqlens=attn_metadata.non_spec_query_start_loc[: num_decodes + 1],
            ssm_state_indices=attn_metadata.non_spec_state_indices_tensor[:num_decodes],
        )
        core_attn_out[0, :num_decodes] = decode_out[0]
        q, k, v, g, beta = (x[:, num_decodes:] for x in (q, k, v, g, beta))

    # The prefill rows' metadata is rebased past the decode rows.
    state_indices = attn_metadata.prefill_state_indices
    # New sequences start from a zero state: select it on the device instead
    # of zeroing rows found with a boolean mask.
    initial_state = torch.where(
        attn_metadata.prefill_has_initial_state[:, None, None, None],
        recurrent_state[state_indices],
        0,
    )
    global _step_chunks
    if _step_chunks is None or _step_chunks.step is not step:
        _step_chunks = _StepChunks(step, gate_options)
    out, last_state = _step_chunks.run([q, k, v, g, beta], initial_state, attn_metadata)
    recurrent_state[state_indices] = last_state
    core_attn_out[0, num_decodes:num_actual_tokens] = out[0]


def _chunk(q, k, v, g, beta, initial_state, cu_seqlens, chunk_indices, gate_options):
    global _known_chunk_indices
    _known_chunk_indices = None if chunk_indices is None else (cu_seqlens, chunk_indices)
    try:
        return chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            **gate_options,
            cu_seqlens=cu_seqlens,
        )
    finally:
        _known_chunk_indices = None


class _StepChunks:
    """The chunk kernels of one step's prefill rows, recorded once and replayed per KDA layer.

    Every KDA layer of a step runs the same ~11 Triton kernels on the same
    rows, and launching them from Python costs far more host time than the
    GPU spends on them. The first layer runs them eagerly (its launches also
    settle the autotuners), the second records them in a CUDA graph and the
    rest replay it, with their inputs and initial states copied into the
    recording's buffers. The kernels and their inputs are the same, so the
    results are those of eager launches. Inside the recording only the rows'
    layout is read (``cu_seqlens`` and chunk indices), which every KV cache
    group of the step shares; state slots are gathered and scattered per layer.
    """

    _stream: torch.cuda.Stream | None = None
    # Every step records into one pool, reusing the blocks of the step before
    # instead of leaving a released private pool behind per step. The
    # allocators reuse a pool only while a graph in it is alive, so the last
    # recording is released once the next one is made.
    _pool = None
    _last: _StepChunks | None = None
    _failed = False

    def __init__(self, step, gate_options):
        self.step = step
        self.gate_options = gate_options
        # The recording reads this metadata's tensors, so it is kept alive.
        self.metadata = None
        self.layers = 0
        self.graph: torch.cuda.CUDAGraph | None = None
        self.inputs: list[torch.Tensor] = []
        self.outputs: tuple[torch.Tensor, torch.Tensor] | None = None

    def _eager(self, inputs, initial_state, metadata):
        return _chunk(
            *inputs,
            initial_state.contiguous(),
            metadata.prefill_query_start_loc,
            metadata.chunk_indices,
            self.gate_options,
        )

    def run(self, inputs: list[torch.Tensor], initial_state: torch.Tensor, metadata):
        self.layers += 1
        if self.layers == 1 or _StepChunks._failed or not inputs[0].is_cuda:
            return self._eager(inputs, initial_state, metadata)
        values = [*inputs, initial_state]
        if self.graph is None:
            if self.layers > 2:
                return self._eager(inputs, initial_state, metadata)
            self._record(values, metadata)
            if self.graph is None:
                return self._eager(inputs, initial_state, metadata)
        elif any(buffer.shape != value.shape for buffer, value in zip(self.inputs, values, strict=True)):
            return self._eager(inputs, initial_state, metadata)
        else:
            for buffer, value in zip(self.inputs, values, strict=True):
                buffer.copy_(value)
        self.graph.replay()
        return self.outputs

    def _record(self, values, metadata):
        self.inputs = [x.clone(memory_format=torch.contiguous_format) for x in values]
        self.metadata = metadata
        if _StepChunks._stream is None:
            _StepChunks._stream = torch.cuda.Stream()
            _StepChunks._pool = torch.cuda.graph_pool_handle()
        stream = _StepChunks._stream
        graph = torch.cuda.CUDAGraph()
        # Capture directly: torch.cuda.graph() would synchronize the device
        # and collect garbage at every step.
        stream.wait_stream(torch.cuda.current_stream())
        try:
            with torch.cuda.stream(stream):
                graph.capture_begin(pool=_StepChunks._pool, capture_error_mode="thread_local")
                try:
                    outputs = self._eager(self.inputs[:-1], self.inputs[-1], metadata)
                finally:
                    graph.capture_end()
        except Exception:
            _StepChunks._failed = True
            logger.warning("Ming3: recording the KDA chunk kernels failed; launching them per layer", exc_info=True)
            return
        finally:
            torch.cuda.current_stream().wait_stream(stream)
        self.graph, self.outputs = graph, outputs
        _StepChunks._last = self


_step_chunks: _StepChunks | None = None


def _kda_forward(
    self,
    q_proj_states: torch.Tensor,
    k_proj_states: torch.Tensor,
    v_proj_states: torch.Tensor,
    g1: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
) -> None:
    forward_context = upstream.get_forward_context()
    attn_metadata_map = forward_context.attn_metadata
    attn_metadata = attn_metadata_map.get(self.prefix) if isinstance(attn_metadata_map, dict) else None
    if attn_metadata is None or attn_metadata.num_prefills == 0 or attn_metadata.spec_sequence_masks is not None:
        return _UPSTREAM_FORWARD(self, q_proj_states, k_proj_states, v_proj_states, g1, beta, core_attn_out)

    has_initial_state = attn_metadata.has_initial_state
    query_start_loc = attn_metadata.non_spec_query_start_loc
    state_indices = attn_metadata.non_spec_state_indices_tensor
    num_actual_tokens = attn_metadata.num_actual_tokens
    conv_state, recurrent_state = self.kv_cache
    recurrent_state_active = recurrent_state[..., : self.head_dim]
    if not upstream.is_conv_state_dim_first():
        conv_state = conv_state.transpose(-1, -2)
    conv_states = conv_state.chunk(3, dim=-2)

    projections = []
    for states, conv, conv_state_part in zip(
        (q_proj_states, k_proj_states, v_proj_states),
        (self.q_conv1d, self.k_conv1d, self.v_conv1d),
        conv_states,
        strict=True,
    ):
        weights = conv.weight.view(conv.weight.size(0), conv.weight.size(2))
        mixed = upstream.causal_conv1d_fn(
            states[:num_actual_tokens].transpose(0, 1),
            weights,
            conv.bias,
            activation="silu",
            conv_states=conv_state_part,
            has_initial_state=has_initial_state,
            cache_indices=state_indices,
            query_start_loc=query_start_loc,
            metadata=attn_metadata,
        ).transpose(0, 1)
        projections.append(upstream.rearrange(mixed, "n (h d) -> 1 n h d", d=self.head_dim))
    q, k, v = projections
    _recurrent_step(
        q,
        k,
        v,
        g1[:, :num_actual_tokens],
        beta[:, :num_actual_tokens],
        recurrent_state_active,
        attn_metadata,
        core_attn_out,
        forward_context,
        safe_gate=self.safe_gate,
        lower_bound=self.lower_bound,
    )


def install_sync_free_kda_prefill(backbone: nn.Module) -> int:
    """Bind the sync-free prefill to every KDA layer of ``backbone``; return the layer count."""
    try:
        source = inspect.getsource(_UPSTREAM_FORWARD)
    except (OSError, TypeError):
        source = ""
    if _UPSTREAM_MARKER not in source:
        logger.info("Ming3: upstream KDA prefill changed; keeping the upstream path")
        return 0
    layers = [module for module in backbone.modules() if isinstance(module, upstream.BailingMoeV3KimiDeltaAttention)]
    for layer in layers:
        layer._forward = types.MethodType(_kda_forward, layer)
    if layers:
        # Other callers pass other cu_seqlens tensors and keep deriving their indices.
        for module in _CHUNK_INDEX_USERS:
            if getattr(module, "prepare_chunk_indices", None) in (_derive_chunk_indices, _prepare_chunk_indices):
                module.prepare_chunk_indices = _prepare_chunk_indices
    return len(layers)


__all__ = ["install_sync_free_kda_prefill"]
