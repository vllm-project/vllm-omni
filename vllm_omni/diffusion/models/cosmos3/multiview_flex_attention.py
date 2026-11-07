# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sparse multiview attention for Cosmos3.

This module is an inference-only implementation of the Multiview-AV visibility
rules.  It intentionally depends only on the public PyTorch FlexAttention API;
the training implementation is used as a behavioral oracle, not as source.
"""

from __future__ import annotations

import math
from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import flex_attention as torch_flex_attention

SPARSE_Q_BLOCK_SIZE = 64
SPARSE_KV_BLOCK_SIZE = 64

# The default SM100 BF16/FP16 head_dim=128 FlexAttention configuration is
# 128x64 with three stages and eight warps. The multiview mask_mod adds enough
# state that the default first exceeded shared-memory capacity and, after only
# shrinking BLOCK_N, produced an illegal access at launch. Use the smallest
# square tile supported by the forward autotuner, remove software pipelining,
# and keep TMA disabled. Aligning the sparse mask blocks with the compute tile
# also avoids sub-block address arithmetic in the generated kernel.
TRITON_Q_BLOCK_SIZE = 64
TRITON_KV_BLOCK_SIZE = 64
TRITON_NUM_STAGES = 1
TRITON_NUM_WARPS = 4

# The UND stream is padded to a fixed capacity rather than to the nearest block
# above each prompt's real length.  A pad that tracks the prompt changes the
# packed key tensor's sequence dimension, so the block-mask shapes, packing
# buffers and compiled flex guards would all vary with every prompt.  A fixed
# capacity keeps them identical across prompts of one request geometry.
#
# Padding to the capacity is numerically free: the extra keys carry
# ``sample_id == -1`` and are therefore already excluded from every real query by
# the predicate's ``same_sample`` term, so the output is bit-identical.  It is
# near-free in compute too, because the fully-padded blocks are visible only to
# the padded query rows in the final Q block.
#
# The default is the Cosmos3 prompt truncation cap plus the ``eos`` and
# ``vision_start`` framing tokens the tokenizer appends after truncating.
DEFAULT_MAX_UND_TOKENS = 4096 + 2

# ``maskless`` is not a sparse kernel: it runs the same visibility rules as a
# few unmasked varlen FlashAttention passes merged by log-sum-exp (see
# ``multiview_maskless_attention``), so it has no block geometry.
MULTIVIEW_BACKENDS: tuple[str, ...] = ("maskless", "triton")


def validate_multiview_backend(backend: str) -> str:
    """Reject an unknown backend name.

    Exposed so callers can fail at load time; ``MultiviewLayout`` is only built
    once a request arrives, which would otherwise defer a config typo to the
    first generation.
    """
    if backend not in MULTIVIEW_BACKENDS:
        raise ValueError(
            f"Cosmos3 multiview attention backend must be one of {list(MULTIVIEW_BACKENDS)}, got {backend!r}."
        )
    return backend


@dataclass(frozen=True)
class MaskItem:
    """Semantic description of one packed vision item.

    ``token_shape`` is ``(latent_frames, patch_height, patch_width)``.  Frames
    are camera-major: all frames of view zero, followed by all frames of view
    one. ``seconds_per_frame`` is the positive wall-clock duration of one
    latent frame and must agree for items sharing the view grid.
    """

    token_shape: tuple[int, int, int]
    num_views: int
    view_offset: int = 0
    is_control: bool = False
    seconds_per_frame: float = 1.0
    is_lidar: bool = False

    @property
    def num_tokens(self) -> int:
        return math.prod(self.token_shape)


@dataclass(frozen=True)
class MultiviewLayout:
    """Explicit packed camera/LiDAR streams and request-invariant attention geometry.

    Items own all sensor geometry, including optional controls and independent
    camera/LiDAR shapes and frame rates.
    """

    items: tuple[MaskItem, ...]
    cross_view_past_window_seconds: float
    backend: str = "triton"
    #: Sparse UND padding capacity, independent of any one prompt's
    #: length, so the compiled attention sees a single shape.  See
    #: ``DEFAULT_MAX_UND_TOKENS``.
    max_und_tokens: int = DEFAULT_MAX_UND_TOKENS
    caption_lengths: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        # The pipeline validates the backend, window and geometry once per
        # request. ``dataclasses.replace`` reruns this on every forward, so keep
        # only the maskless op registration and the single-rate-per-view invariant
        # that semantic run grouping depends on.
        if self.backend == "maskless":
            # Register the custom op before the regionally compiled GEN layers.
            from . import multiview_maskless_attention  # noqa: F401
        rates_by_view_offset: dict[int, float] = {}
        for item in self.items:
            expected = rates_by_view_offset.setdefault(item.view_offset, item.seconds_per_frame)
            if not math.isclose(item.seconds_per_frame, expected):
                raise ValueError(
                    "Cosmos3 multiview items sharing a view offset must use the same seconds_per_frame: "
                    f"offset={item.view_offset}, got {item.seconds_per_frame}, expected {expected}."
                )

    @property
    def gen_tokens(self) -> int:
        return sum(item.num_tokens for item in self.items)

    @property
    def block_sizes(self) -> tuple[int, int]:
        """The ``(q, kv)`` sparse block granularity this backend demands."""
        if self.backend != "triton":
            raise RuntimeError(f"Cosmos3 multiview backend {self.backend!r} has no sparse block geometry.")
        return SPARSE_Q_BLOCK_SIZE, SPARSE_KV_BLOCK_SIZE


@dataclass(frozen=True)
class MultiviewFlexMetadata:
    """Per-token metadata for rectangular GEN-to-[UND|GEN] attention."""

    sample_id: torch.Tensor
    frame_id: torch.Tensor
    view_id: torch.Tensor
    is_control: torch.Tensor
    is_und: torch.Tensor
    timestamp: torch.Tensor
    query_start: int
    cross_view_past_window_seconds: float

    @property
    def kv_len(self) -> int:
        return int(self.sample_id.numel())

    @property
    def q_len(self) -> int:
        return self.kv_len - self.query_start

    def query_vectors(self) -> tuple[torch.Tensor, ...]:
        query_slice = slice(self.query_start, None)
        return (
            self.sample_id[query_slice],
            self.frame_id[query_slice],
            self.view_id[query_slice],
            self.is_control[query_slice],
            self.is_und[query_slice],
            self.timestamp[query_slice],
        )

    def key_vectors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.sample_id,
            self.frame_id,
            self.view_id,
            self.is_control,
            self.is_und,
            self.timestamp,
        )

    def query_grouping_vectors(self) -> tuple[torch.Tensor, ...]:
        """Discrete query fields that define semantic runs.

        Timestamp is deliberately excluded: under the validated single-rate
        camera layout it is a function of ``(view_id, frame_id)``. UND and
        padding runs use the fixed ``-1.0`` sentinel.
        """
        return self.query_vectors()[:-1]

    def key_grouping_vectors(self) -> tuple[torch.Tensor, ...]:
        """Discrete key fields that define semantic runs; see query variant."""
        return self.key_vectors()[:-1]


@dataclass(frozen=True)
class MultiviewAttentionContext:
    """Runtime wrapper that keeps the request-local caches on the transformer."""

    layout: MultiviewLayout
    #: Triton caches a ``BlockMask``; maskless caches its ``MultiviewMasklessPlan``
    #: under its own key.
    mask_cache: MutableMapping[tuple[Any, ...], Any]
    buffer_cache: MutableMapping[tuple[Any, ...], torch.Tensor] = field(default_factory=dict)
    #: ``MasklessRuntime`` attached by ``prepare_maskless_context``; ``None`` for Triton.
    maskless: Any = None


@dataclass(frozen=True)
class PaddedAttentionGeometry:
    real_q_len: int
    padded_q_len: int
    real_und_len: int
    padded_und_len: int


def expand_multiview_condition_frame_indexes(
    indexes: Sequence[int] | int | None,
    num_views: int,
    latent_t: int,
) -> list[int]:
    """Expand per-view-local latent frame indexes into camera-major indexes."""
    if num_views <= 0 or latent_t <= 0 or latent_t % num_views:
        raise ValueError(
            "Cosmos3 multiview expansion requires latent_t divisible by num_views: "
            f"latent_t={latent_t}, num_views={num_views}."
        )
    if indexes is None:
        local_indexes: Sequence[int] = ()
    elif isinstance(indexes, int):
        local_indexes = (indexes,)
    else:
        local_indexes = indexes
    frames_per_view = latent_t // num_views
    filtered = sorted({int(index) for index in local_indexes if 0 <= int(index) < frames_per_view})
    return [view * frames_per_view + frame for view in range(num_views) for frame in filtered]


def build_multiview_flex_metadata(
    layout: MultiviewLayout,
    geometry: PaddedAttentionGeometry,
    device: torch.device | str,
) -> MultiviewFlexMetadata:
    """Derive single-sample token metadata from the streams and their padding."""
    device = torch.device(device)
    num_und = geometry.real_und_len
    if (
        geometry.real_q_len != layout.gen_tokens
        or geometry.padded_q_len < geometry.real_q_len
        or not 0 <= num_und <= min(geometry.padded_und_len, layout.max_und_tokens)
    ):
        raise ValueError(f"Invalid Cosmos3 multiview padding geometry: {geometry}.")
    seq_len = geometry.padded_und_len + geometry.padded_q_len

    sample_id = torch.full((seq_len,), -1, dtype=torch.int64, device=device)
    frame_id = torch.full_like(sample_id, -1)
    view_id = torch.full_like(sample_id, -1)
    is_control = torch.zeros(seq_len, dtype=torch.bool, device=device)
    is_und = torch.zeros_like(is_control)
    timestamp = torch.full((seq_len,), -1.0, dtype=torch.float32, device=device)
    sample_id[:num_und] = 0
    is_und[:num_und] = True
    camera_views = {
        view
        for item in layout.items
        if not item.is_lidar
        for view in range(item.view_offset, item.view_offset + item.num_views)
    }
    if (
        camera_views != set(range(len(layout.caption_lengths)))
        or not layout.caption_lengths
        or sum(layout.caption_lengths) != num_und
        or any(length <= 0 for length in layout.caption_lengths)
    ):
        raise ValueError("Caption boundaries must partition the real text tokens with one segment per camera.")
    start = 0
    for view, length in enumerate(layout.caption_lengths):
        view_id[start : start + length] = view
        start += length

    start = geometry.padded_und_len
    for item in layout.items:
        end = start + item.num_tokens
        latent_t, patch_h, patch_w = item.token_shape
        spatial_tokens = patch_h * patch_w
        frames_per_view = latent_t // item.num_views
        item_frames = torch.arange(frames_per_view, dtype=torch.int64, device=device)
        item_frames = item_frames.repeat(item.num_views).repeat_interleave(spatial_tokens)
        item_views = torch.arange(
            item.view_offset,
            item.view_offset + item.num_views,
            dtype=torch.int64,
            device=device,
        ).repeat_interleave(frames_per_view * spatial_tokens)
        item_timestamps = item_frames.to(torch.float32) * item.seconds_per_frame
        sample_id[start:end] = 0
        frame_id[start:end] = item_frames
        # Negative sensor IDs distinguish LiDAR from every camera while
        # retaining the six-vector metadata used by both attention backends.
        view_id[start:end] = -2 if item.is_lidar else item_views
        is_control[start:end] = item.is_control
        timestamp[start:end] = item_timestamps
        start = end

    return MultiviewFlexMetadata(
        sample_id=sample_id,
        frame_id=frame_id,
        view_id=view_id,
        is_control=is_control,
        is_und=is_und,
        timestamp=timestamp,
        query_start=geometry.padded_und_len,
        cross_view_past_window_seconds=layout.cross_view_past_window_seconds,
    )


def _make_pair_allowed(
    q_vectors: tuple[torch.Tensor, ...],
    k_vectors: tuple[torch.Tensor, ...],
    cross_view_past_window_seconds: float,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build the exact pair predicate with tensor-only traced configuration.

    The returned closure is stored on Triton's ``BlockMask`` and traced by
    Inductor. Materialize scalar options here, outside that closure, so Python
    strings, booleans, and floats cannot become dynamic captured scalars.
    """
    device = q_vectors[2].device
    temporal_window = torch.tensor(cross_view_past_window_seconds, dtype=torch.float32, device=device)
    temporal_window_eps = torch.tensor(1e-4, dtype=torch.float32, device=device)

    def pair_allowed(q_index: torch.Tensor, kv_index: torch.Tensor) -> torch.Tensor:
        q_sample = q_vectors[0][q_index]
        q_view = q_vectors[2][q_index]
        q_control = q_vectors[3][q_index]
        q_timestamp = q_vectors[5][q_index]
        k_sample = k_vectors[0][kv_index]
        k_view = k_vectors[2][kv_index]
        k_control = k_vectors[3][kv_index]
        k_und = k_vectors[4][kv_index]
        k_timestamp = k_vectors[5][kv_index]

        # Sentinel equality deliberately isolates padding from real tokens
        # while giving every padded query at least one padded key.
        same_sample = q_sample == k_sample
        same_view = q_view == k_view
        timestamp_gap = q_timestamp - k_timestamp
        within_temporal_window = (timestamp_gap >= -temporal_window_eps) & (
            timestamp_gap <= temporal_window + temporal_window_eps
        )
        in_scope = same_view | within_temporal_window

        sensor_to_sensor = (~q_control) & (~k_control) & in_scope
        sensor_to_control = (~q_control) & k_control & same_view
        control_to_control = q_control & k_control & same_view
        control_to_sensor = q_control & (~k_control) & same_view
        reads_caption = k_und & ((q_view == -2) | same_view)
        return same_sample & (
            reads_caption | (~k_und & (sensor_to_sensor | sensor_to_control | control_to_control | control_to_sensor))
        )

    return pair_allowed


def _semantic_groups(vectors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return run IDs and first-token indexes without converting field dtypes."""
    if not vectors:
        raise ValueError("Cosmos3 multiview semantic grouping requires at least one field.")
    seq_len = vectors[0].numel()
    if any(vector.numel() != seq_len for vector in vectors):
        raise ValueError("Cosmos3 multiview semantic grouping fields must have equal lengths.")
    changed = torch.zeros(seq_len, dtype=torch.bool, device=vectors[0].device)
    # Start run zero even when metadata fields contain negative padding or
    # LiDAR sentinels. The cumulative count therefore produces non-negative IDs.
    changed[:1] = True
    for vector in vectors:
        changed[1:] |= vector[1:] != vector[:-1]
    group_ids = changed.to(torch.int64).cumsum(0) - 1
    representatives = torch.nonzero(changed, as_tuple=False).flatten()
    return group_ids, representatives


def _block_group_presence(group_ids: torch.Tensor, block_size: int, num_groups: int) -> torch.Tensor:
    if group_ids.numel() % block_size:
        raise ValueError(
            f"Cosmos3 multiview metadata length {group_ids.numel()} is not aligned to block size {block_size}."
        )
    blocks = group_ids.view(-1, block_size)
    presence = torch.zeros((blocks.shape[0], num_groups), dtype=torch.bool, device=group_ids.device)
    presence.scatter_(1, blocks, True)
    return presence


def _block_indices(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Keep the full KV-block width instead of trimming to the densest row.
    # ``create_block_mask`` always produces full-width contiguous indices and
    # both the Triton template and ``BlockMask`` helpers assume that layout
    # (see pytorch/pytorch#153344); a trimmed column slice is also
    # non-contiguous, which no upstream code path ever exercises. Full width
    # additionally keeps the mask shapes identical across CFG branches, so
    # both share one compiled kernel, and avoids a device sync here.
    counts = mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.argsort(mask.to(torch.int8), dim=-1, descending=True, stable=True)
    return counts, indices.to(torch.int32)


def build_multiview_block_mask(
    metadata: MultiviewFlexMetadata,
    *,
    q_block_size: int = SPARSE_Q_BLOCK_SIZE,
    kv_block_size: int = SPARSE_KV_BLOCK_SIZE,
) -> BlockMask:
    """Compress semantic runs into a Triton FlexAttention block mask.

    The projection works at semantic-run and sparse-block granularity.  Its
    largest dense intermediates are block-grid sized (about 10.4M entries for
    the released 11-view geometry at 64x64), never the roughly 42B-token dense
    mask.  The emitted counts/indices use the same full-width contiguous layout
    that ``create_block_mask`` produces, which is the only layout the Triton
    template and the ``BlockMask`` utilities are exercised with upstream.
    """
    q_vectors = metadata.query_vectors()
    k_vectors = metadata.key_vectors()
    q_group_ids, q_representatives = _semantic_groups(metadata.query_grouping_vectors())
    k_group_ids, k_representatives = _semantic_groups(metadata.key_grouping_vectors())

    pair_allowed = _make_pair_allowed(
        q_vectors,
        k_vectors,
        metadata.cross_view_past_window_seconds,
    )
    group_allowed = pair_allowed(q_representatives[:, None], k_representatives[None, :])
    q_presence = _block_group_presence(q_group_ids, q_block_size, q_representatives.numel())
    k_presence = _block_group_presence(k_group_ids, kv_block_size, k_representatives.numel())

    # Float16 represents these tiny integer overlap counts exactly and gives a
    # fast tensor-core projection on CUDA. CPU tests use float32 matmul.
    projection_dtype = torch.float16 if q_presence.device.type == "cuda" else torch.float32
    q_projection = q_presence.to(projection_dtype)
    k_projection = k_presence.to(projection_dtype)
    visible_blocks = (q_projection @ group_allowed.to(projection_dtype) @ k_projection.T) > 0
    forbidden_blocks = (q_projection @ (~group_allowed).to(projection_dtype) @ k_projection.T) > 0
    full_blocks = visible_blocks & (~forbidden_blocks)
    partial_blocks = visible_blocks & (~full_blocks)

    partial_counts, partial_indices = _block_indices(partial_blocks)
    full_counts, full_indices = _block_indices(full_blocks)

    def mask_mod(batch: torch.Tensor, head: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        del batch, head
        return pair_allowed(q_idx, kv_idx)

    return BlockMask.from_kv_blocks(
        partial_counts[None, None],
        partial_indices[None, None],
        full_counts[None, None],
        full_indices[None, None],
        BLOCK_SIZE=(q_block_size, kv_block_size),
        mask_mod=mask_mod,
        seq_lengths=(metadata.q_len, metadata.kv_len),
        compute_q_blocks=False,
    )


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def get_multiview_attention_plan(
    context: MultiviewAttentionContext,
    *,
    real_und_len: int,
    real_q_len: int,
    device: torch.device,
) -> tuple[BlockMask, PaddedAttentionGeometry]:
    """Build or retrieve the request-local mask for one CFG text length.

    Both padded lengths are pure functions of the layout, never of this call's
    ``real_und_len``: prompts of different lengths produce different masks but
    identically shaped tensors, so they share one compiled kernel.
    """
    layout = context.layout
    q_block_size, kv_block_size = layout.block_sizes
    padded_q_len = _round_up(real_q_len, q_block_size)
    padded_und_len = _round_up(layout.max_und_tokens, kv_block_size)
    geometry = PaddedAttentionGeometry(real_q_len, padded_q_len, real_und_len, padded_und_len)
    key = (layout, real_und_len, device)
    cached = context.mask_cache.get(key)
    if cached is not None:
        return cached, geometry

    metadata = build_multiview_flex_metadata(layout, geometry, device)
    plan = build_multiview_block_mask(
        metadata,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
    )
    context.mask_cache[key] = plan
    return plan, geometry


def _packing_buffer(
    cache: MutableMapping[tuple[Any, ...], torch.Tensor] | None,
    slot: str,
    reference: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Return a zeroed packing buffer, reused across layers when a cache is given.

    The packed q/k/v layouts are rebuilt in all 36 GEN layers of all 70
    forwards, and for the released 11-view geometry the query buffer alone is
    ~1.7 GiB, so allocating and zeroing one per layer costs terabytes of
    pointless memset per run.  Only the real token rows are ever written, so the
    padding rows keep the zeros from the initial allocation and the buffer stays
    reusable for any later call with the same slot, shape, dtype, and device.
    """
    if cache is None:
        return reference.new_zeros(shape)
    key = (slot, shape, reference.dtype, reference.device.type, reference.device.index)
    buffer = cache.get(key)
    if buffer is None:
        buffer = reference.new_zeros(shape)
        cache[key] = buffer
    return buffer


def _validate_parts(parts: tuple[tuple[torch.Tensor, int], ...]) -> tuple[torch.Tensor, int, int, int, int]:
    if not parts:
        raise ValueError("Cosmos3 multiview attention requires at least one sequence part.")
    reference = parts[0][0]
    batch, _, heads, head_dim = reference.shape
    total_len = sum(target_len for _, target_len in parts)
    for tensor, target_len in parts:
        if tensor.ndim != 4 or tensor.shape[0] != batch or tensor.shape[2:] != (heads, head_dim):
            raise ValueError(
                "Cosmos3 multiview attention sequence parts must share [B, H, D]: "
                f"reference={tuple(reference.shape)}, part={tuple(tensor.shape)}."
            )
        if tensor.shape[1] > target_len:
            raise ValueError(f"Cannot pad sequence length {tensor.shape[1]} down to {target_len}.")
    return reference, batch, heads, head_dim, total_len


def _pack_padded_bhsd(
    *parts: tuple[torch.Tensor, int],
    buffer_cache: MutableMapping[tuple[Any, ...], torch.Tensor] | None = None,
    slot: str = "",
) -> torch.Tensor:
    """Pack ``[B, S, H, D]`` parts directly into contiguous ``[B, H, S, D]``."""
    reference, batch, heads, head_dim, total_len = _validate_parts(parts)
    packed = _packing_buffer(buffer_cache, f"bhsd:{slot}", reference, (batch, heads, total_len, head_dim))
    offset = 0
    for tensor, target_len in parts:
        packed[:, :, offset : offset + tensor.shape[1]].copy_(tensor.transpose(1, 2))
        offset += target_len
    return packed


_compiled_flex_attention = None


def _compile_flex_attention() -> Callable[..., torch.Tensor]:
    # The GEN length follows the request's views, clip length and resolution,
    # so a static compile recompiles per request geometry.  Dynamo's recompile
    # limit (8 by default) then drops to eager FlexAttention, which materializes
    # the full fp32 score matrix (~5.4 TB at the released 11-view geometry).
    # Dynamic shapes serve every geometry from one kernel.  The settings are
    # patched explicitly because this runs inside a regionally compiled GEN
    # layer, whose ``dynamic=False`` config stays in effect for nested compiles.
    # ``fullgraph=True`` turns an exhausted recompile budget into an error
    # instead of the eager fallback.
    compiled = torch.compile(torch_flex_attention, dynamic=True, fullgraph=True)

    def run(*args: Any, **kwargs: Any) -> torch.Tensor:
        with torch._dynamo.config.patch(automatic_dynamic_shapes=True, assume_static_by_default=False):
            return compiled(*args, **kwargs)

    return run


def flex_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_mask: BlockMask,
) -> torch.Tensor:
    """Run pinned Triton FlexAttention on contiguous ``[B, H, S, D]`` tensors."""
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("Cosmos3 multiview FlexAttention requires contiguous [B, H, S, D] inputs.")
    kernel_options = {
        "BACKEND": "TRITON",
        "BLOCK_M": TRITON_Q_BLOCK_SIZE,
        "BLOCK_N": TRITON_KV_BLOCK_SIZE,
        "num_stages": TRITON_NUM_STAGES,
        "num_warps": TRITON_NUM_WARPS,
        "USE_TMA": False,
    }
    if q.device.type == "cuda":
        global _compiled_flex_attention
        if _compiled_flex_attention is None:
            _compiled_flex_attention = _compile_flex_attention()
        output = _compiled_flex_attention(
            q,
            k,
            v,
            block_mask=block_mask,
            enable_gqa=True,
            kernel_options=kernel_options,
        )
    else:
        # Eager CPU support is useful for tiny correctness tests; production
        # multiview inference is admitted only on the Triton/CUDA path.
        output = torch_flex_attention(
            q,
            k,
            v,
            block_mask=block_mask,
            enable_gqa=True,
            kernel_options=kernel_options,
        )
    return output


@torch.compiler.disable
def padded_multiview_triton_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_und: torch.Tensor,
    v_und: torch.Tensor,
    context: MultiviewAttentionContext,
) -> torch.Tensor:
    """``padded_multiview_flex_attention`` kept out of the regionally compiled GEN layer.

    Traced inline, the mask plan, packing buffers and kernel specialize on the
    request geometry and the prompt length, so the layer graph exhausts its
    recompile budget within a few requests and the attention falls back to
    eager. Here it always runs through the dedicated dynamic-shape compile in
    ``flex_attention``.
    """
    return padded_multiview_flex_attention(q, k, v, k_und, v_und, context)


def padded_multiview_flex_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_und: torch.Tensor,
    v_und: torch.Tensor,
    context: MultiviewAttentionContext,
) -> torch.Tensor:
    """Pad UND and GEN independently, attend once, then trim GEN rows."""
    plan, geometry = get_multiview_attention_plan(
        context,
        real_und_len=k_und.shape[1],
        real_q_len=q.shape[1],
        device=q.device,
    )
    # Every GEN layer packs the same three shapes, and each layer's attention
    # has consumed the previous one before the next overwrites them, so the
    # buffers are reused for the whole request instead of being re-zeroed.
    buffers = context.buffer_cache
    q_padded = _pack_padded_bhsd((q, geometry.padded_q_len), buffer_cache=buffers, slot="q")
    k_all = _pack_padded_bhsd(
        (k_und, geometry.padded_und_len),
        (k, geometry.padded_q_len),
        buffer_cache=buffers,
        slot="k",
    )
    v_all = _pack_padded_bhsd(
        (v_und, geometry.padded_und_len),
        (v, geometry.padded_q_len),
        buffer_cache=buffers,
        slot="v",
    )
    output = flex_attention(q_padded, k_all, v_all, block_mask=plan)
    return output[:, :, : geometry.real_q_len].transpose(1, 2)
