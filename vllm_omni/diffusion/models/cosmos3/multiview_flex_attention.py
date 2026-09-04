# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse multiview attention for Cosmos3.

This module is an inference-only implementation of the Multiview-AV visibility
rules.  It intentionally depends only on the public PyTorch FlexAttention API;
the training implementation is used as a behavioral oracle, not as source.
"""

from __future__ import annotations

import math
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import torch
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import flex_attention as torch_flex_attention

AttentionScope = Literal["all_views", "same_view", "same_view_or_frame"]

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

# FlashAttention-4 runs a fixed 128x128 forward tile on SM100 and stages two Q
# tiles per CTA whenever the query length exceeds one tile, so the sparse block
# map it consumes must be (2 * tile_m, tile_n).  These are not tunable: the
# kernel derives the same numbers from its own heuristic and rejects metadata
# that disagrees.  See flash_attn/cute/interface.py::_get_fwd_config and
# flash_attn/cute/block_sparsity.py::normalize_block_sparse_config.
FA4_SPARSE_Q_BLOCK_SIZE = 256
FA4_SPARSE_KV_BLOCK_SIZE = 128

# Coarser blocks admit more disallowed pairs into partially-masked tiles, where
# the mask_mod resolves them exactly.  Measured on the released 11-view / 21-frame
# / 30x52 geometry, moving from 64x64 to 256x128 raises visited tiles from 10.66%
# to 11.43% of the dense rectangle (+7% attention MACs) and raises the partial
# tile share from 6% to 19%.
_BACKEND_BLOCK_SIZES: dict[str, tuple[int, int]] = {
    "triton": (SPARSE_Q_BLOCK_SIZE, SPARSE_KV_BLOCK_SIZE),
    "fa4": (FA4_SPARSE_Q_BLOCK_SIZE, FA4_SPARSE_KV_BLOCK_SIZE),
}

# The UND stream is padded to a fixed capacity rather than to the nearest block
# above each prompt's real length.  A pad that tracks the prompt changes the
# packed key tensor's sequence dimension, and the flex kernel is compiled with
# dynamic=False, so every distinct prompt-length bucket is a fresh recompile
# ("tensor 'key' size mismatch at index 2").  Dynamo's default limit of eight is
# reached after a handful of prompts, after which the frame falls back to eager
# FlexAttention -- which materializes a ~2.7 TB score matrix at the released
# 11-view geometry.  A fixed capacity gives one compiled kernel for the life of
# the process.
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

_VALID_ATTENTION_SCOPES = frozenset({"all_views", "same_view", "same_view_or_frame"})


MULTIVIEW_BACKENDS: tuple[str, ...] = tuple(sorted(_BACKEND_BLOCK_SIZES))


def validate_multiview_backend(backend: str) -> str:
    """Reject an unknown backend name.

    Exposed so callers can fail at load time; ``MultiviewLayout`` is only built
    once a request arrives, which would otherwise defer a config typo to the
    first generation.
    """
    if backend not in _BACKEND_BLOCK_SIZES:
        raise ValueError(
            f"Cosmos3 multiview attention backend must be one of {list(MULTIVIEW_BACKENDS)}, got {backend!r}."
        )
    return backend


def _validate_attention_scope(attention_scope: str) -> AttentionScope:
    if attention_scope not in _VALID_ATTENTION_SCOPES:
        raise ValueError(
            "Cosmos3 multiview attention_scope must be one of "
            f"{sorted(_VALID_ATTENTION_SCOPES)}, got {attention_scope!r}."
        )
    return attention_scope  # type: ignore[return-value]


@dataclass(frozen=True)
class MaskItem:
    """Semantic description of one packed vision item.

    ``token_shape`` is ``(latent_frames, patch_height, patch_width)``.  Frames
    are camera-major: all frames of view zero, followed by all frames of view
    one.  ``condition_mask`` is true for clean frames.
    """

    token_shape: tuple[int, int, int]
    condition_mask: torch.Tensor
    num_views: int
    view_offset: int = 0
    is_control: bool = False

    def __post_init__(self) -> None:
        latent_t, patch_h, patch_w = self.token_shape
        if latent_t <= 0 or patch_h <= 0 or patch_w <= 0:
            raise ValueError(f"Cosmos3 multiview token_shape must be positive, got {self.token_shape}.")
        if self.num_views <= 0 or latent_t % self.num_views:
            raise ValueError(
                "Cosmos3 multiview latent frames must be divisible by num_views: "
                f"latent_t={latent_t}, num_views={self.num_views}."
            )
        if self.condition_mask.ndim != 1 or self.condition_mask.numel() != latent_t:
            raise ValueError(
                "Cosmos3 multiview condition_mask must have one value per latent frame: "
                f"shape={tuple(self.condition_mask.shape)}, latent_t={latent_t}."
            )
        if self.condition_mask.dtype is not torch.bool:
            raise TypeError(f"Cosmos3 multiview condition_mask must have dtype bool, got {self.condition_mask.dtype}.")

    @property
    def num_tokens(self) -> int:
        return math.prod(self.token_shape)


@dataclass(frozen=True)
class MultiviewLayout:
    """Request-invariant geometry passed from the pipeline to the transformer."""

    num_views: int
    latent_frames: int
    patch_height: int
    patch_width: int
    condition_frame_indexes: tuple[int, ...] = ()
    attention_scope: AttentionScope = "same_view_or_frame"
    backend: str = "triton"
    #: Capacity the UND stream is padded to, independent of any one prompt's
    #: length, so the compiled attention sees a single shape.  See
    #: ``DEFAULT_MAX_UND_TOKENS``.
    max_und_tokens: int = DEFAULT_MAX_UND_TOKENS

    #: v1 always packs one fully-clean WSM control item then one RGB target.
    NUM_ITEMS: ClassVar[int] = 2

    def __post_init__(self) -> None:
        _validate_attention_scope(self.attention_scope)
        validate_multiview_backend(self.backend)
        if self.backend == "fa4":
            # Importing here registers vllm_omni::cosmos3_multiview_fa4 while we
            # are still host-side.  The attention call site imports the module
            # lazily too, but that site is inside the regionally compiled GEN
            # block, and registering a torch.library op from under Dynamo is not
            # something to rely on.  The module itself defers every CuTe/CUTLASS
            # import to _load_fa4, so this stays safe on CPU-only hosts.
            from . import multiview_fa4  # noqa: F401
        if self.max_und_tokens <= 0:
            raise ValueError(f"Cosmos3 multiview max_und_tokens must be positive, got {self.max_und_tokens}.")
        if self.num_views <= 0 or self.latent_frames <= 0:
            raise ValueError("Cosmos3 multiview num_views and latent_frames must be positive.")
        if self.latent_frames % self.num_views:
            raise ValueError(
                "Cosmos3 multiview latent_frames must be camera-major and divisible by num_views: "
                f"latent_frames={self.latent_frames}, num_views={self.num_views}."
            )
        if self.patch_height <= 0 or self.patch_width <= 0:
            raise ValueError("Cosmos3 multiview patch dimensions must be positive.")
        invalid = [index for index in self.condition_frame_indexes if not 0 <= index < self.latent_frames]
        if invalid:
            raise ValueError(
                f"Cosmos3 multiview condition frame indexes are outside the packed latent stream: {invalid}."
            )
        if tuple(sorted(set(self.condition_frame_indexes))) != self.condition_frame_indexes:
            raise ValueError("Cosmos3 multiview condition frame indexes must be sorted and unique.")

    @property
    def item_tokens(self) -> int:
        return self.latent_frames * self.patch_height * self.patch_width

    @property
    def gen_tokens(self) -> int:
        return self.item_tokens * self.NUM_ITEMS

    @property
    def block_sizes(self) -> tuple[int, int]:
        """The ``(q, kv)`` sparse block granularity this backend demands."""
        return _BACKEND_BLOCK_SIZES[self.backend]

    def mask_items(self, device: torch.device) -> tuple[MaskItem, ...]:
        """Build the packed items, resolving clean frames by packed position.

        Both the control/target role and the set of clean frames follow the
        same positional rule -- every item but the last is a fully clean
        control -- so both come from ``resolve_item_condition_frames``.
        """
        shape = (self.latent_frames, self.patch_height, self.patch_width)
        items = []
        for item_index in range(self.NUM_ITEMS):
            clean_frames = resolve_item_condition_frames(
                item_index,
                self.NUM_ITEMS,
                self.condition_frame_indexes,
                self.latent_frames,
            )
            condition_mask = torch.zeros(self.latent_frames, dtype=torch.bool, device=device)
            if clean_frames:
                condition_mask[torch.tensor(clean_frames, dtype=torch.int64, device=device)] = True
            is_control = item_index < self.NUM_ITEMS - 1
            items.append(MaskItem(shape, condition_mask, self.num_views, is_control=is_control))
        return tuple(items)

    def cache_key(self) -> tuple[Any, ...]:
        return (
            self.num_views,
            self.latent_frames,
            self.patch_height,
            self.patch_width,
            self.condition_frame_indexes,
            self.attention_scope,
            self.backend,
            self.max_und_tokens,
        )


@dataclass(frozen=True)
class MultiviewFlexMetadata:
    """Per-token metadata for rectangular GEN-to-[UND|GEN] attention."""

    sample_id: torch.Tensor
    frame_id: torch.Tensor
    view_id: torch.Tensor
    is_noisy: torch.Tensor
    is_control: torch.Tensor
    is_und: torch.Tensor
    query_start: int
    attention_scope: AttentionScope

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
            self.is_noisy[query_slice],
            self.is_control[query_slice],
            self.is_und[query_slice],
        )

    def key_vectors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.sample_id,
            self.frame_id,
            self.view_id,
            self.is_noisy,
            self.is_control,
            self.is_und,
        )


@dataclass(frozen=True)
class MultiviewAttentionContext:
    """Runtime wrapper that keeps the request-local caches on the transformer."""

    layout: MultiviewLayout
    mask_cache: MutableMapping[tuple[Any, ...], BlockMask]
    buffer_cache: MutableMapping[tuple[Any, ...], torch.Tensor] = field(default_factory=dict)


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


def resolve_item_condition_frames(
    item_index: int,
    num_items: int,
    condition_frame_indexes: Sequence[int],
    latent_t: int,
) -> list[int]:
    """Controls are fully clean; only the final target uses request indexes."""
    if not 0 <= item_index < num_items:
        raise IndexError(f"Cosmos3 multiview item_index={item_index} outside num_items={num_items}.")
    if item_index < num_items - 1:
        return list(range(latent_t))
    return sorted({int(index) for index in condition_frame_indexes if 0 <= int(index) < latent_t})


def build_multiview_flex_metadata(
    seq_len: int,
    full_q_offsets: Sequence[int],
    items_per_sample: Sequence[Sequence[MaskItem]] | Sequence[MaskItem],
    device: torch.device | str,
    num_und: int,
    attention_scope: AttentionScope = "same_view_or_frame",
) -> MultiviewFlexMetadata:
    """Build metadata without ever materializing a token-by-token dense mask.

    ``full_q_offsets`` contains the start of every packed vision item plus the
    end of the final item.  The first offset is also the query start and may be
    larger than ``num_und`` because UND is padded independently.
    """
    attention_scope = _validate_attention_scope(attention_scope)
    device = torch.device(device)
    if seq_len <= 0 or num_und < 0 or num_und > seq_len:
        raise ValueError(f"Invalid Cosmos3 multiview sequence geometry: seq_len={seq_len}, num_und={num_und}.")
    if items_per_sample and isinstance(items_per_sample[0], MaskItem):  # type: ignore[index]
        samples: list[list[MaskItem]] = [list(items_per_sample)]  # type: ignore[arg-type]
    else:
        samples = [list(items) for items in items_per_sample]  # type: ignore[arg-type]
    if len(samples) != 1:
        raise ValueError("Cosmos3 multiview v1 supports exactly one sample per request.")
    items = samples[0]
    if not items:
        raise ValueError("Cosmos3 multiview metadata requires at least one vision item.")
    if len(full_q_offsets) != len(items) + 1:
        raise ValueError(
            "Cosmos3 multiview full_q_offsets must contain one boundary per item plus the end: "
            f"offsets={list(full_q_offsets)}, items={len(items)}."
        )
    offsets = tuple(int(offset) for offset in full_q_offsets)
    if offsets[0] < num_und or offsets[-1] > seq_len or any(a > b for a, b in zip(offsets, offsets[1:])):
        raise ValueError(f"Invalid Cosmos3 multiview item offsets: {offsets} for seq_len={seq_len}.")

    sample_id = torch.full((seq_len,), -1, dtype=torch.int64, device=device)
    frame_id = torch.full_like(sample_id, -1)
    view_id = torch.full_like(sample_id, -1)
    is_noisy = torch.zeros(seq_len, dtype=torch.bool, device=device)
    is_control = torch.zeros_like(is_noisy)
    is_und = torch.zeros_like(is_noisy)
    sample_id[:num_und] = 0
    is_und[:num_und] = True

    view_offsets = {item.view_offset for item in items}
    if attention_scope == "same_view_or_frame" and len(view_offsets) > 1:
        raise ValueError(
            "Cosmos3 same_view_or_frame attention does not support mixed view offsets (for example LiDAR)."
        )

    for item_index, (item, start, end) in enumerate(zip(items, offsets[:-1], offsets[1:], strict=True)):
        if end - start != item.num_tokens:
            raise ValueError(
                f"Cosmos3 multiview item {item_index} occupies {end - start} tokens, expected {item.num_tokens}."
            )
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
        noisy_frames = (~item.condition_mask.to(device=device)).repeat_interleave(spatial_tokens)
        sample_id[start:end] = 0
        frame_id[start:end] = item_frames
        view_id[start:end] = item_views
        is_noisy[start:end] = noisy_frames
        is_control[start:end] = item.is_control

    return MultiviewFlexMetadata(
        sample_id=sample_id,
        frame_id=frame_id,
        view_id=view_id,
        is_noisy=is_noisy,
        is_control=is_control,
        is_und=is_und,
        query_start=offsets[0],
        attention_scope=attention_scope,
    )


def _pair_allowed(
    q_sample: torch.Tensor,
    q_frame: torch.Tensor,
    q_view: torch.Tensor,
    q_noisy: torch.Tensor,
    q_control: torch.Tensor,
    k_sample: torch.Tensor,
    k_frame: torch.Tensor,
    k_view: torch.Tensor,
    k_noisy: torch.Tensor,
    k_control: torch.Tensor,
    k_und: torch.Tensor,
    attention_scope: AttentionScope,
) -> torch.Tensor:
    # Sentinel equality deliberately isolates padding from real tokens while
    # giving every padded query at least one padded key. This avoids relying on
    # backend-specific handling of an empty softmax row.
    same_sample = q_sample == k_sample
    same_view = q_view == k_view
    same_frame = q_frame == k_frame
    if attention_scope == "all_views":
        in_scope = torch.ones_like(same_view)
    elif attention_scope == "same_view":
        in_scope = same_view
    else:
        in_scope = same_view | same_frame

    rgb_pair = (~q_control) & (~k_control) & in_scope
    clean_rgb_pair = rgb_pair & (~q_noisy) & (~k_noisy)
    noisy_rgb_query = rgb_pair & q_noisy
    rgb_to_control = (~q_control) & k_control & same_view
    control_to_control = q_control & k_control & same_view
    return same_sample & (k_und | clean_rgb_pair | noisy_rgb_query | rgb_to_control | control_to_control)


def multiview_pair_predicate(
    metadata: MultiviewFlexMetadata,
    q_index: torch.Tensor,
    kv_index: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the exact token visibility predicate (also used as mask_mod)."""
    q_vectors = metadata.query_vectors()
    k_vectors = metadata.key_vectors()
    return _pair_allowed(
        q_vectors[0][q_index],
        q_vectors[1][q_index],
        q_vectors[2][q_index],
        q_vectors[3][q_index],
        q_vectors[4][q_index],
        k_vectors[0][kv_index],
        k_vectors[1][kv_index],
        k_vectors[2][kv_index],
        k_vectors[3][kv_index],
        k_vectors[4][kv_index],
        k_vectors[5][kv_index],
        metadata.attention_scope,
    )


def _semantic_groups(vectors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    stacked = torch.stack([vector.to(torch.int64) for vector in vectors], dim=0)
    changed = torch.ones(stacked.shape[1], dtype=torch.bool, device=stacked.device)
    if stacked.shape[1] > 1:
        changed[1:] = torch.any(stacked[:, 1:] != stacked[:, :-1], dim=0)
    group_ids = changed.to(torch.int64).cumsum(0) - 1
    representatives = torch.nonzero(changed, as_tuple=False).flatten()
    return group_ids, tuple(vector[representatives] for vector in vectors)


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


def _pack_allowed_bits(group_allowed: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Pack the ``[q_group, k_group]`` truth table into int32 bit words.

    The kernel-side mask_mod reads one word and tests one bit, so the whole
    visibility rule set collapses to a few hundred bytes that stay resident in
    cache.  Bit ``g_k`` of word ``g_q * words_per_row + g_k // 32`` is the
    answer for that group pair.  Words hold the unsigned 32-bit pattern stored
    in ``int32``, which is what the CuTe kernel reinterprets.
    """
    num_q_groups, num_k_groups = group_allowed.shape
    words_per_row = (num_k_groups + 31) // 32
    padded = group_allowed.new_zeros((num_q_groups, words_per_row * 32))
    padded[:, :num_k_groups] = group_allowed
    weights = torch.arange(32, device=group_allowed.device, dtype=torch.int64)
    words = (padded.view(num_q_groups, words_per_row, 32).to(torch.int64) << weights).sum(-1)
    words = torch.where(words >= 2**31, words - 2**32, words)
    return words.reshape(-1).to(torch.int32).contiguous(), words_per_row


# eq=False: the fields are tensors, so a generated __eq__ would return a tensor
# rather than a bool. Instances are cache values compared by identity.
@dataclass(frozen=True, eq=False)
class MultiviewBlockSparsity:
    """Backend-neutral sparse block map plus the run-compressed mask table.

    ``partial_*``/``full_*`` are the FlexAttention KV-block layout, which
    FlashAttention-4 consumes unchanged as ``BlockSparseTensorsTorch``.  The
    remaining fields are the exact per-element fallback used inside partially
    masked tiles: a token-to-run id for each side plus the packed truth table
    over run pairs.  Together they encode ``multiview_pair_predicate`` without
    the kernel knowing anything about views, frames, or noise levels.
    """

    partial_counts: torch.Tensor
    partial_indices: torch.Tensor
    full_counts: torch.Tensor
    full_indices: torch.Tensor
    q_word_base: torch.Tensor
    k_group_ids: torch.Tensor
    allowed_words: torch.Tensor
    group_allowed: torch.Tensor
    words_per_row: int
    q_block_size: int
    kv_block_size: int
    metadata: MultiviewFlexMetadata

    @property
    def q_len(self) -> int:
        return self.metadata.q_len

    @property
    def kv_len(self) -> int:
        return self.metadata.kv_len

    def aux_tensors(self) -> list[torch.Tensor]:
        """The mask_mod auxiliary tensors, in the order the kernel indexes them."""
        return [self.q_word_base, self.k_group_ids, self.allowed_words]

    def to_block_mask(self) -> BlockMask:
        metadata = self.metadata

        def mask_mod(
            batch: torch.Tensor, head: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ) -> torch.Tensor:
            del batch, head
            return multiview_pair_predicate(metadata, q_idx, kv_idx)

        return BlockMask.from_kv_blocks(
            self.partial_counts[None, None],
            self.partial_indices[None, None],
            self.full_counts[None, None],
            self.full_indices[None, None],
            BLOCK_SIZE=(self.q_block_size, self.kv_block_size),
            mask_mod=mask_mod,
            seq_lengths=(metadata.q_len, metadata.kv_len),
            compute_q_blocks=False,
        )


def build_multiview_block_sparsity(
    metadata: MultiviewFlexMetadata,
    *,
    q_block_size: int = SPARSE_Q_BLOCK_SIZE,
    kv_block_size: int = SPARSE_KV_BLOCK_SIZE,
) -> MultiviewBlockSparsity:
    """Compress semantic runs into a sparse block map and a mask lookup table.

    The projection works at semantic-run and sparse-block granularity.  Its
    largest dense intermediates are block-grid sized (about 10.4M entries for
    the released 11-view geometry at 64x64), never the roughly 42B-token dense
    mask.  The emitted counts/indices use the same full-width contiguous layout
    that ``create_block_mask`` produces, which is the only layout the Triton
    template and the ``BlockMask`` utilities are exercised with upstream, and
    the layout FlashAttention-4 validates its own block sparsity against.
    """
    q_vectors = metadata.query_vectors()
    k_vectors = metadata.key_vectors()
    q_group_ids, q_reps = _semantic_groups(q_vectors)
    k_group_ids, k_reps = _semantic_groups(k_vectors)

    group_allowed = _pair_allowed(
        q_reps[0][:, None],
        q_reps[1][:, None],
        q_reps[2][:, None],
        q_reps[3][:, None],
        q_reps[4][:, None],
        k_reps[0][None, :],
        k_reps[1][None, :],
        k_reps[2][None, :],
        k_reps[3][None, :],
        k_reps[4][None, :],
        k_reps[5][None, :],
        metadata.attention_scope,
    )
    q_presence = _block_group_presence(q_group_ids, q_block_size, len(q_reps[0]))
    k_presence = _block_group_presence(k_group_ids, kv_block_size, len(k_reps[0]))

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

    # Fold the table row stride into the query-side id so the kernel needs no
    # compile-time shape constant and one compiled mask_mod serves every layout.
    allowed_words, words_per_row = _pack_allowed_bits(group_allowed)
    q_word_base = (q_group_ids * words_per_row).to(torch.int32).contiguous()

    return MultiviewBlockSparsity(
        partial_counts=partial_counts,
        partial_indices=partial_indices,
        full_counts=full_counts,
        full_indices=full_indices,
        q_word_base=q_word_base,
        k_group_ids=k_group_ids.to(torch.int32).contiguous(),
        allowed_words=allowed_words,
        group_allowed=group_allowed,
        words_per_row=words_per_row,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
        metadata=metadata,
    )


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def get_multiview_attention_plan(
    context: MultiviewAttentionContext,
    *,
    real_und_len: int,
    real_q_len: int,
    device: torch.device,
) -> tuple[BlockMask | MultiviewBlockSparsity, PaddedAttentionGeometry]:
    """Build or retrieve the request-local mask for one CFG text length.

    Returns whichever mask representation the layout's backend consumes: a
    ``BlockMask`` for Triton FlexAttention, or a ``MultiviewBlockSparsity`` for
    FlashAttention-4.  Both are built from the same run-level projection, so the
    two backends never disagree about which pairs are visible.

    Both padded lengths are pure functions of the layout, never of this call's
    ``real_und_len``: prompts of different lengths produce different masks but
    identically shaped tensors, so they share one compiled kernel.
    """
    layout = context.layout
    if real_q_len != layout.gen_tokens:
        raise ValueError(
            "Cosmos3 multiview packed GEN length does not match the request layout: "
            f"attention={real_q_len}, layout={layout.gen_tokens}."
        )
    if real_und_len > layout.max_und_tokens:
        raise ValueError(
            "Cosmos3 multiview UND stream exceeds the layout capacity the attention was sized for: "
            f"tokens={real_und_len}, max_und_tokens={layout.max_und_tokens}."
        )
    q_block_size, kv_block_size = layout.block_sizes
    padded_q_len = _round_up(real_q_len, q_block_size)
    padded_und_len = _round_up(layout.max_und_tokens, kv_block_size)
    geometry = PaddedAttentionGeometry(real_q_len, padded_q_len, real_und_len, padded_und_len)
    key = (
        layout.cache_key(),
        real_und_len,
        padded_und_len,
        real_q_len,
        padded_q_len,
        q_block_size,
        kv_block_size,
        device.type,
        device.index,
    )
    cached = context.mask_cache.get(key)
    if cached is not None:
        return cached, geometry

    items = layout.mask_items(device)
    item_tokens = layout.item_tokens
    item_offsets = tuple(padded_und_len + index * item_tokens for index in range(len(items) + 1))
    metadata = build_multiview_flex_metadata(
        seq_len=padded_und_len + padded_q_len,
        full_q_offsets=item_offsets,
        items_per_sample=items,
        device=device,
        num_und=real_und_len,
        attention_scope=layout.attention_scope,
    )
    sparsity = build_multiview_block_sparsity(
        metadata,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
    )
    plan = sparsity if layout.backend == "fa4" else sparsity.to_block_mask()
    context.mask_cache[key] = plan
    return plan, geometry


def get_multiview_block_mask(
    context: MultiviewAttentionContext,
    *,
    real_und_len: int,
    real_q_len: int,
    device: torch.device,
) -> tuple[BlockMask, PaddedAttentionGeometry]:
    """Request-local ``BlockMask`` for the Triton FlexAttention backend."""
    plan, geometry = get_multiview_attention_plan(
        context,
        real_und_len=real_und_len,
        real_q_len=real_q_len,
        device=device,
    )
    if not isinstance(plan, BlockMask):
        raise TypeError(
            "Cosmos3 multiview get_multiview_block_mask requires backend='triton', got "
            f"{context.layout.backend!r}; use get_multiview_attention_plan instead."
        )
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


def _pack_padded_bshd(
    *parts: tuple[torch.Tensor, int],
    buffer_cache: MutableMapping[tuple[Any, ...], torch.Tensor] | None = None,
    slot: str = "",
) -> torch.Tensor:
    """Concatenate and pad ``[B, S, H, D]`` parts, keeping the native layout.

    FlashAttention-4 consumes ``[B, S, H, D]`` directly, so unlike the Triton
    path this never transposes; the copy is the concatenation the packed layout
    needs anyway.
    """
    reference, batch, heads, head_dim, total_len = _validate_parts(parts)
    packed = _packing_buffer(buffer_cache, f"bshd:{slot}", reference, (batch, total_len, heads, head_dim))
    offset = 0
    for tensor, target_len in parts:
        packed[:, offset : offset + tensor.shape[1]].copy_(tensor)
        offset += target_len
    return packed


_compiled_flex_attention = None


def flex_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_mask: BlockMask,
    backend: str,
) -> torch.Tensor:
    """Run pinned Triton FlexAttention on contiguous ``[B, H, S, D]`` tensors."""
    if backend != "triton":
        raise ValueError(f"Cosmos3 multiview v1 supports only backend='triton', got {backend!r}.")
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
            _compiled_flex_attention = torch.compile(torch_flex_attention, dynamic=False)
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


def padded_multiview_flex_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_und: torch.Tensor,
    v_und: torch.Tensor,
    context: MultiviewAttentionContext,
) -> torch.Tensor:
    """Pad UND and GEN independently, attend once, then trim GEN rows."""
    if q.shape[:2] != k.shape[:2] or k.shape != v.shape:
        raise ValueError(
            "Cosmos3 multiview q/k/v sequence geometry must match before GQA: "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}."
        )
    if k_und.shape != v_und.shape or k_und.shape[0] != q.shape[0]:
        raise ValueError(
            "Cosmos3 multiview UND key/value geometry mismatch: "
            f"k_und={tuple(k_und.shape)}, v_und={tuple(v_und.shape)}."
        )
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
    if context.layout.backend == "fa4":
        # Imported lazily: the CuTe/CUTLASS stack is an optional, Blackwell-only
        # dependency, and this module must stay importable on CPU-only hosts.
        from .multiview_fa4 import multiview_fa4_attention

        q_padded = _pack_padded_bshd((q, geometry.padded_q_len), buffer_cache=buffers, slot="q")
        k_all = _pack_padded_bshd(
            (k_und, geometry.padded_und_len),
            (k, geometry.padded_q_len),
            buffer_cache=buffers,
            slot="k",
        )
        v_all = _pack_padded_bshd(
            (v_und, geometry.padded_und_len),
            (v, geometry.padded_q_len),
            buffer_cache=buffers,
            slot="v",
        )
        output = multiview_fa4_attention(q_padded, k_all, v_all, plan)
        return output[:, : geometry.real_q_len]

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
    output = flex_attention(q_padded, k_all, v_all, block_mask=plan, backend=context.layout.backend)
    return output[:, :, : geometry.real_q_len].transpose(1, 2)
