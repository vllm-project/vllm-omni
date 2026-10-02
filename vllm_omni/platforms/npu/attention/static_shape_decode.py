#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Static-shape decode attention: full-graph replay without per-step rebinding.

Vendored from vllm-ascend (Apache-2.0) so that vLLM-Omni can deliver it
without patching vllm-ascend: everything this module needs from vllm-ascend is
reachable through factories vLLM-Omni already overrides, so the rest of the
change lives in ``static_shape_backend.py`` and in ``NPUOmniPlatform``.

``FULL_DECODE_ONLY`` captures a whole decode step, but
``npu_fused_infer_attention_score`` takes ``actual_seq_lengths_kv`` as a host
``SymInt[]``, so its tiling is baked into the captured task. Because the KV length
grows by one every step, vllm-ascend has to re-issue the op for every layer on
every step (``update_full_graph_params`` -> ``graph_task_update_begin/end``). On a
short-context decoder that rebind is the step: measured 3.68 ms per step for the
20-layer MiniCPM-o Talker on 910B3, against 1.21 ms for the replay itself.

This module removes the rebind by making every host-side argument constant. The op
declares a *fixed* KV capacity, and the real sequence length is carried instead by
a device-resident ``pse_shift`` bias: ``0`` for valid positions and ``-inf`` past
the end of each sequence. The captured graph refreshes that bias itself from the
live ``seq_lens`` tensor, so a replay needs nothing from the host at all.

The cost is that attention reads the whole declared capacity every step instead of
the live prefix. Capacity *buckets* keep that honest: several graphs are captured
per batch size, one per candidate capacity, and each step replays the smallest
bucket that covers it. With one bucket equal to the model's whole block table this
degenerates to a fixed capacity, which is only sensible for short contexts.

There is no switch: the numbers below are the measured-stable ones, and the only
thing that decides whether a stage engages is its own ``max_model_len`` (see
:func:`capacity_for`), so a long-context stage keeps the stock path untouched.
"""

from contextlib import contextmanager

import torch
from vllm.logger import logger

# Padding rows of a cudagraph batch carry seq_len 0. A row whose bias is -inf
# everywhere makes softmax produce NaN, so clamp the mask to keep one live slot.
_MIN_SEQ_LEN = 1

_pse_cache: dict[tuple, torch.Tensor] = {}
_arange_cache: dict[tuple, torch.Tensor] = {}
_scalar_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

# Per (graph size, capacity), the tensors that capture baked in. Replay reads them
# directly, so the runtime metadata has to keep handing us the same buffers; see
# ``sync_captured_inputs``.
_captured_inputs: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
_captured_slot_mappings: dict[tuple[int, int], torch.Tensor] = {}

# Scratch the captured tasks write into. Capture-time allocations come out of the
# graph's private pool, so anything that goes out of scope here can be recycled
# into the next capture while a live graph still points at it.
_graph_scratch: dict[tuple[int, int], list[torch.Tensor]] = {}

# One FIA workspace per attention shape, shared by every layer and every graph
# that matches it. Keyed by shape rather than by graph so the measuring call
# never runs more than once per shape; see ``graph_workspace``.
_workspace_cache: dict[tuple, torch.Tensor] = {}

_warned_moved_inputs = False

# Which capacity the current step runs at. During capture a bucket is forced, so
# that one pass over the batch sizes produces the graphs for exactly that bucket.
_capture_bucket: int | None = None
_runtime_bucket: int | None = None


# Candidate capacities to capture graphs for, smallest first. A bucket that just
# covers the sequence is worth having: attention reads the declared capacity every
# step, and the Talker's sequence is ~135 (a ~15-token condition plus ~118 codec
# frames) against a 4096-token table. The model's own capacity is always added as
# the top bucket, so a longer sequence still has a graph to land on.
_CANDIDATE_BUCKETS = "512"

# Only a small decoder has fixed-capacity graphs worth capturing; a long-context
# stage (a 32768-token Thinker) keeps the stock path. The ceiling is a property of
# the shapes that get captured, not a per-deployment preference, so it is a code
# constant: the stage's own ``max_model_len`` is what selects the outcome.
_MAX_CONTEXT = 8192


class _CapacityEntries(dict):
    """The wrapper's entries dict, plus one dict per static-shape capacity.

    Sleep mode's ``reset_all_graph_params`` clears whatever dict is installed on
    the wrapper, so the per-capacity dicts hang off it rather than off the
    wrapper: otherwise a reset would leave stale graphs behind them.
    """

    def __init__(self, existing: dict | None = None) -> None:
        super().__init__(existing or {})
        self.by_capacity: dict[int, dict] = {}

    def clear(self) -> None:
        super().clear()
        for entries in self.by_capacity.values():
            entries.clear()
        self.by_capacity.clear()


def install_into_ascend_aclgraph() -> None:
    """Give the image's ACLGraphWrapper a per-capacity key and drop a dead barrier.

    Two things the stock wrapper does that are wrong for static-shape decode:

    * It keys captured graphs on ``batch_descriptor`` alone. The KV capacity is
      baked into the captured tasks, so one batch size holds one graph *per
      bucket*. Without that, the second capture pass finds the first bucket's
      entry, replays it instead of capturing, and hangs on the device
      synchronize that closes the pass.
    * It synchronizes the stream before every FULL replay, to order
      ``update_attn_params`` against the previous one. This path issues no
      updates, so the barrier is pure cost. ``enable_enpu`` gates that one line
      and nothing else in the stock ``__call__``, which is what lets us borrow it
      instead of copying the whole wrapper.

    This wraps ``__call__`` rather than subclassing ``ACLGraphWrapper``, for the
    same reason :func:`install_into_ascend_backend` patches the backend class:
    ``current_platform`` is vllm-ascend's ``NPUPlatform``, so
    ``NPUOmniPlatform.get_graph_wrapper_cls`` is never consulted.

    It deliberately does **not** touch ``vllm_ascend.attention.fixed_kv_decode``:
    that module only exists in vllm-ascend builds carrying the out-of-tree
    static-shape patch, and leaning on it would leave this change dependent on the
    patch it exists to replace.
    """
    from vllm_ascend.compilation import acl_graph

    wrapper_cls = acl_graph.ACLGraphWrapper
    if getattr(wrapper_cls, "_omni_static_shape_call", False):
        return
    original_call = wrapper_cls.__call__

    def _call(self, *args, **kwargs):
        capacity = current_capacity()
        if capacity is None:
            return original_call(self, *args, **kwargs)
        entries = self.concrete_aclgraph_entries
        if not isinstance(entries, _CapacityEntries):
            entries = _CapacityEntries(entries)
            self.concrete_aclgraph_entries = entries
        saved_entries = self.concrete_aclgraph_entries
        saved_enpu = self.enable_enpu
        self.concrete_aclgraph_entries = entries.by_capacity.setdefault(capacity, {})
        # Only once the capture has recorded this (size, capacity) is there a
        # fixed graph to replay with nothing to order against; during the
        # capture itself the barrier still has to stand.
        if not saved_enpu and _is_static_shape_graph_here(capacity):
            self.enable_enpu = True
        try:
            return original_call(self, *args, **kwargs)
        finally:
            self.concrete_aclgraph_entries = saved_entries
            self.enable_enpu = saved_enpu

    wrapper_cls.__call__ = _call
    wrapper_cls._omni_static_shape_call = True
    logger.info("[minicpmo] static-shape: ACLGraphWrapper keyed per KV capacity, pre-replay barrier dropped")


def _is_static_shape_graph_here(capacity: int) -> bool:
    """``is_static_shape_graph`` for the batch size the forward context is about to run."""
    from vllm.forward_context import get_forward_context

    descriptor = getattr(get_forward_context(), "batch_descriptor", None)
    if descriptor is None:
        return False
    return (descriptor.num_tokens, capacity) in _captured_inputs


def install_into_ascend_backend() -> None:
    """Put the omni impl/builder on the class vLLM's selector actually uses.

    ``current_platform.get_attn_backend_cls`` is vllm-ascend's ``NPUPlatform``,
    not ``NPUOmniPlatform``. Without this the extra buckets are captured but the
    stock builder still runs for them, so runtime looks up ``(batch, None)``.
    Patch the stock backend class in this process instead.
    """
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackend

    from vllm_omni.platforms.npu.attention.static_shape_backend import (
        OmniStaticShapeAttentionBackendImpl,
        OmniStaticShapeMetadataBuilder,
    )

    if getattr(AscendAttentionBackend, "_omni_static_shape_installed", False):
        return
    AscendAttentionBackend.get_impl_cls = staticmethod(lambda: OmniStaticShapeAttentionBackendImpl)
    AscendAttentionBackend.get_builder_cls = staticmethod(lambda: OmniStaticShapeMetadataBuilder)
    AscendAttentionBackend._omni_static_shape_installed = True
    install_into_ascend_aclgraph()
    logger.info("[minicpmo] installed omni static-shape impl/builder onto AscendAttentionBackend")


def capacity_for(max_model_len: int, block_size: int) -> int:
    """Full KV capacity this decoder would run at, or 0 when it should not engage."""
    if max_model_len > _MAX_CONTEXT:
        return 0
    # Round to the block table, not to max_model_len: the op indexes whole blocks.
    return -(-int(max_model_len) // int(block_size)) * int(block_size)


def buckets_for(capacity: int, block_size: int) -> tuple[int, ...]:
    """Capacities to capture graphs for, smallest first.

    Candidates at or above this model's own capacity are dropped: the full
    capacity is always the top bucket and always covers the model.
    """
    if not capacity:
        return ()
    raw = _CANDIDATE_BUCKETS
    candidates = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            continue
        rounded = -(-value // int(block_size)) * int(block_size)
        if rounded < capacity:
            candidates.add(rounded)
    candidates.add(int(capacity))
    return tuple(sorted(candidates))


def select_bucket(max_seq_len: int, buckets: tuple[int, ...]) -> int:
    """Smallest captured capacity that covers this step."""
    for bucket in buckets:
        if max_seq_len <= bucket:
            return bucket
    return buckets[-1]


def set_runtime_bucket(capacity: int | None) -> None:
    global _runtime_bucket
    _runtime_bucket = capacity


@contextmanager
def capturing_bucket(capacity: int):
    """Force one capacity for a whole capture pass over the batch sizes."""
    global _capture_bucket
    previous = _capture_bucket
    _capture_bucket = capacity
    try:
        yield
    finally:
        _capture_bucket = previous


def current_capacity() -> int | None:
    return _capture_bucket if _capture_bucket is not None else _runtime_bucket


def graph_key() -> int | None:
    """Extra ACL graph dispatch key, so one batch size can hold several buckets.

    Read by vllm-ascend's ``ACLGraphWrapper`` in builds that carry the static-shape
    hook; nothing in this package calls it directly.
    """
    return current_capacity()


def get_pse_buffer(
    max_rows: int,
    num_heads: int,
    capacity: int,
    dtype: torch.dtype,
    device: torch.device,
    q_len: int = 1,
) -> torch.Tensor:
    key = (max_rows, num_heads, capacity, dtype, device, q_len)
    buf = _pse_cache.get(key)
    if buf is None:
        buf = torch.zeros(max_rows, num_heads, q_len, capacity, dtype=dtype, device=device)
        _pse_cache[key] = buf
    return buf


def _get_arange(num_heads: int, capacity: int, device: torch.device, q_len: int = 1) -> torch.Tensor:
    """``ar[.., i, j] = j - i`` -- the KV index measured from query ``i``.

    With one query per sequence this is a plain arange and the comparison
    against the sequence length is the whole mask. With ``q_len`` queries the
    shift is what makes the mask causal *among* them: query ``i`` of the step
    owns KV position ``seq_len - q_len + i``, so it may read ``j`` only while
    ``j - i < seq_len - q_len + 1``, which is one comparison against a single
    per-row length again.
    """
    key = (num_heads, capacity, device, q_len)
    ar = _arange_cache.get(key)
    if ar is None:
        columns = torch.arange(capacity, dtype=torch.int32, device=device).view(1, 1, 1, capacity)
        queries = torch.arange(q_len, dtype=torch.int32, device=device).view(1, 1, q_len, 1)
        ar = (columns - queries).expand(1, num_heads, q_len, capacity).contiguous()
        _arange_cache[key] = ar
    return ar


def _get_scalars(dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (dtype, device)
    pair = _scalar_cache.get(key)
    if pair is None:
        pair = (
            torch.tensor(float("-inf"), dtype=dtype, device=device),
            torch.tensor(0.0, dtype=dtype, device=device),
        )
        _scalar_cache[key] = pair
    return pair


def prewarm(
    max_rows: int,
    num_heads: int,
    buckets: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    q_lens: tuple[int, ...] = (1,),
) -> None:
    """Materialize every buffer the captured steps will reference.

    Called at model load. Anything allocated inside `torch.npu.graph` comes out of
    that graph's private pool instead, so the buffers have to exist first.
    """
    for capacity in buckets:
        for q_len in q_lens:
            get_pse_buffer(max_rows, num_heads, capacity, dtype, device, q_len)
            _get_arange(num_heads, capacity, device, q_len)
    _get_scalars(dtype, device)


def emit_mask_refresh(
    pse: torch.Tensor,
    seq_lens_device: torch.Tensor,
    num_heads: int,
    capacity: int,
    q_len: int = 1,
) -> None:
    """Record the bias refresh into the graph being captured.

    Runs once per captured step, ahead of the attention layers. On replay it
    re-reads ``seq_lens_device`` in place, which is how a captured step learns
    that the sequence grew without any host involvement.

    ``seq_lens`` counts the whole step, so with ``q_len`` queries the first of
    them ends ``q_len - 1`` positions earlier; the shifted arange from
    ``_get_arange`` turns that back into one comparison per row.
    """
    ar = _get_arange(num_heads, capacity, pse.device, q_len)
    neg, zero = _get_scalars(pse.dtype, pse.device)
    lens = (seq_lens_device - (q_len - 1)).clamp(min=_MIN_SEQ_LEN).to(torch.int32).view(-1, 1, 1, 1)
    torch.where(ar >= lens, neg, zero, out=pse)


def graph_workspace(num_tokens: int, capacity: int, shape_key: tuple, factory) -> torch.Tensor:
    """Hand out one workspace per attention shape and keep it alive for good.

    Layers of one graph share a workspace because their shapes agree. That has
    to be decided *before* asking for the workspace, not after:
    ``_npu_fused_infer_attention_score_get_max_workspace`` allocates as it
    measures, and an allocation made while a graph is capturing comes out of
    that graph's private pool, where dropping the reference does not hand the
    memory back. Measuring per layer therefore strands one workspace per layer
    in the pool (Talker: 20 layers x 4 graphs x 106 MiB = 8.3 GiB). Measure
    once per shape instead and give the same buffer to every layer that matches.
    """
    ws = _workspace_cache.get(shape_key)
    if ws is None:
        ws = factory()
        _workspace_cache[shape_key] = ws
    return ws


def keep_alive(num_tokens: int, capacity: int, tensor: torch.Tensor) -> torch.Tensor:
    _graph_scratch.setdefault((num_tokens, capacity), []).append(tensor)
    return tensor


def remember_captured_inputs(
    num_tokens: int,
    capacity: int,
    seq_lens_device: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor | None = None,
) -> None:
    _captured_inputs[(num_tokens, capacity)] = (seq_lens_device, block_table)
    if slot_mapping is not None:
        _captured_slot_mappings[(num_tokens, capacity)] = slot_mapping


def captured_slot_mapping(num_tokens: int) -> torch.Tensor | None:
    """The slot mapping this size's graph writes KV through, if captured.

    Usually the same persistent buffer the runtime fills, in which case a caller
    that writes here writes exactly where the step would have. When it is not --
    the FIA batch-padding path materializes fresh tensors, which is why
    :func:`sync_captured_inputs` exists -- this is the one the replay reads, and
    the other one is not.
    """
    capacity = current_capacity()
    return _captured_slot_mappings.get((num_tokens, capacity))


def captured_seq_lens(num_tokens: int) -> torch.Tensor | None:
    """The device sequence-length buffer this size's graph reads, if captured.

    A replay takes its live sequence length from here (see the module
    docstring), so a caller that advances the length itself -- the multi-frame
    Talker loop, once per codec frame -- writes into this tensor rather than
    rebuilding attention metadata.
    """
    capacity = current_capacity()
    captured = _captured_inputs.get((num_tokens, capacity))
    return None if captured is None else captured[0]


def is_static_shape_graph(num_tokens: int) -> bool:
    capacity = current_capacity()
    return capacity is not None and (num_tokens, capacity) in _captured_inputs


def sync_captured_inputs(num_tokens: int, seq_lens_device: torch.Tensor, block_table: torch.Tensor) -> None:
    """Keep replay reading live values when the runtime hands us new buffers.

    vLLM's ``seq_lens`` and block table are persistent buffers sliced from offset
    0, so in the common case these are the very tensors the capture baked in and
    this is two pointer comparisons. If a build ever materializes a fresh tensor
    (the FIA batch-padding path does), copy the live values into the captured
    buffers rather than silently replaying stale ones.
    """
    global _warned_moved_inputs
    capacity = current_capacity()
    captured = _captured_inputs.get((num_tokens, capacity))
    if captured is None:
        return
    cap_seq_lens, cap_block_table = captured
    moved = False
    if seq_lens_device is not None and seq_lens_device.data_ptr() != cap_seq_lens.data_ptr():
        rows = min(cap_seq_lens.shape[0], seq_lens_device.shape[0])
        cap_seq_lens[:rows].copy_(seq_lens_device[:rows])
        moved = True
    if block_table is not None and block_table.data_ptr() != cap_block_table.data_ptr():
        rows = min(cap_block_table.shape[0], block_table.shape[0])
        cols = min(cap_block_table.shape[1], block_table.shape[1])
        cap_block_table[:rows, :cols].copy_(block_table[:rows, :cols])
        moved = True
    if moved and not _warned_moved_inputs:
        _warned_moved_inputs = True
        logger.warning(
            "Fixed-KV decode: attention metadata moved off the captured buffers; "
            "copying seq_lens/block_table every step. Replay stays correct but "
            "costs two extra device copies per step."
        )


def reset() -> None:
    global _capture_bucket, _runtime_bucket
    _graph_scratch.clear()
    _workspace_cache.clear()
    _captured_inputs.clear()
    _captured_slot_mappings.clear()
    _pse_cache.clear()
    _arange_cache.clear()
    _scalar_cache.clear()
    _capture_bucket = None
    _runtime_bucket = None
