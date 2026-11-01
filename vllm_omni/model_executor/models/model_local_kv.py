# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Declaration protocol for attention KV kept outside the paged manager.

Several models hold their own attention KV as HuggingFace ``transformers`` cache
objects rather than using the engine's paged KV manager. That memory is
allocated after the profiling run that decides how much KV the engine may claim,
so no per-stage footprint number accounts for it, and nothing catches a cache
that silently stops working.

This module lets a model *declare* what it holds. It describes; it never
allocates. Allocation for the diffusion path is owned by the DiT KV manager
(RFC #5244 / PR #6094) and this protocol must not grow into a second allocator.

Why a runtime query rather than a static table: geometry is frequently not
knowable until weights are loaded. ``ming_flash_omni``'s talker builds its cache
from a ``Qwen2Config`` that comes from the checkpoint, so layers, kv-heads,
head-dim and dtype simply do not exist in this repo. A table would have a hole
in it; a post-load query does not.

Why capacity is a resolved number rather than a formula: the four known caches
are each bounded by a different mechanism -- a sliding window, a decode-loop
trip count, an encoder frame limit, a hardcoded constant -- and not one of them
is ``max_model_len``. Only the model knows its own bound.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "HasModelLocalKV",
    "ModelLocalKVScope",
    "ModelLocalKVSpec",
    "RowDriver",
    "collect_model_local_kv_specs",
    "spec_from_hf_config",
]


class ModelLocalKVScope(str, Enum):
    """How long one allocation lives.

    Diagnostic only. Lifetime and size are independent: a per-call cache can be
    the widest thing a model owns, and a process-lifetime one can be a single
    row. ``ModelLocalKVSpec.rows`` carries the size; this carries only the
    reader's mental model of when the memory appears and goes.
    """

    INVOCATION = "invocation"
    """Dies when the call returns (e.g. a per-step working copy)."""

    REQUEST = "request"
    """Retained across steps of one request, released when it finishes."""

    SESSION = "session"
    """Outlives a request; belongs to a duplex/streaming session."""

    MODEL = "model"
    """Lives as long as the model (e.g. captured into a CUDA-graph pool)."""


class RowDriver(str, Enum):
    """What sets how many rows of a cache are live at peak.

    Two values, because two are all the known caches need. A third belongs
    here only when a model actually has a cache the engine widens by some
    other number -- adding one speculatively costs a branch everywhere and
    buys nothing.
    """

    FIXED = "fixed"
    """A count the model controls: a captured-bucket sum, a serialized call path."""

    MAX_NUM_SEQS = "max_num_seqs"
    """One row per in-flight sequence. Only the engine knows the value."""


@dataclass(frozen=True)
class ModelLocalKVSpec:
    """One declared KV allocation.

    A model returns one entry per *distinct lifetime*, not per cache object: if
    the same geometry exists as both a retained per-request cache and a
    short-lived working copy, that is two entries.
    """

    name: str
    layers: int
    kv_heads: int
    head_dim: int
    dtype: torch.dtype

    physical_capacity_positions: int
    """Positions the tensor can physically hold.

    Not the logical sequence position. A sliding-window cache truncates on every
    write, so a stream of thousands of tokens may only ever occupy
    ``sliding_window - 1`` slots.
    """

    capacity_source: str
    """Free-text note on where the bound comes from, for diagnostics only.

    Never branch on this. It exists so a reader can tell "71 because sliding
    window" from "2048 because someone typed 2048", which is otherwise
    invisible.
    """

    scope: ModelLocalKVScope
    """How long one allocation lives. Diagnostic only.

    Deliberately not the multiplier. An earlier revision derived scaling from
    scope, which made every non-``MODEL`` cache multiply by ``max_num_seqs``.
    That over-reported ming by that factor (its calls are serialized) and used
    the wrong driver entirely for MiniCPM-o (duplex sessions, not sequences).
    Lifetime does not imply replication topology; ``rows`` says what does.
    """

    rows: RowDriver
    """What sets how many rows of this shape are live at peak.

    A row is one batch entry's worth of the geometry above. Deliberately does
    not distinguish "one allocation N rows wide" from "N allocations of one
    row": those cost the same, and an earlier revision that tried to carry both
    got the object topology wrong in three of four declarers. Bytes are what
    this protocol reports; describe the layout in ``allocation_note``.

    Required, because defaulting it silently understates every cache whose
    multiplicity is not 1.
    """

    rows_fixed: int = 1
    """The count, when ``rows`` is ``FIXED``. Ignored otherwise."""

    rows_reason: str = ""
    """Why ``rows_fixed`` is that number. Required when ``rows`` is ``FIXED``.

    A bare constant cannot be reviewed. ``rows_fixed=1`` on a cache that looks
    per-request is exactly the claim a reader has to be able to check.
    """

    allocation_note: str | None = None
    """Optional detail such as "rebuilt per text segment". Diagnostic only."""

    def __post_init__(self) -> None:
        if self.rows is RowDriver.FIXED:
            if self.rows_fixed < 1:
                raise ValueError(f"rows_fixed must be >= 1, got {self.rows_fixed}")
            if not self.rows_reason.strip():
                raise ValueError("a FIXED row count needs rows_reason; an unexplained constant cannot be reviewed")

    @property
    def bytes_per_row(self) -> int:
        """Bytes for a single batch row of this cache."""
        return (
            2  # K and V
            * self.layers
            * self.kv_heads
            * self.head_dim
            * self.physical_capacity_positions
            * torch.empty((), dtype=self.dtype).element_size()
        )

    def row_count(self, max_num_seqs: int = 1) -> int:
        """Rows live at peak, given the engine's concurrency."""
        if self.rows is RowDriver.FIXED:
            return self.rows_fixed
        return max(1, max_num_seqs)

    def peak_bytes(self, max_num_seqs: int = 1) -> int:
        """Peak bytes once the engine's concurrency is known.

        The model supplies geometry and names the driver; only the engine knows
        the number behind it.
        """
        return self.bytes_per_row * self.row_count(max_num_seqs)


def _resolve(config: object, candidates: Sequence[str], what: str) -> int:
    for attr in candidates:
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    raise ValueError(
        f"Cannot determine {what} from {type(config).__name__}: tried "
        f"{', '.join(candidates)}. Pass it explicitly to spec_from_hf_config()."
    )


def spec_from_hf_config(
    config: object,
    *,
    name: str,
    physical_capacity_positions: int,
    capacity_source: str,
    scope: ModelLocalKVScope,
    dtype: torch.dtype,
    rows: RowDriver,
    rows_fixed: int = 1,
    rows_reason: str = "",
    allocation_note: str | None = None,
    layers: int | None = None,
    kv_heads: int | None = None,
    head_dim: int | None = None,
) -> ModelLocalKVSpec:
    """Build a spec from the same config object the cache is built from.

    Every known model-local cache is constructed from an HF config, so the
    geometry half of a declaration is the same three lookups each time. Pass
    ``config`` and this derives them; pass ``layers``/``kv_heads``/``head_dim``
    explicitly to override when the config uses encoder-decoder naming or when
    the built module is a more truthful source than the config.

    Raises rather than guessing when an attribute is absent: a silently wrong
    geometry would under-report, which is the failure this protocol exists to
    prevent. ``rows`` is keyword-required for the same reason -- an earlier
    revision defaulted the batch extent to 1 here while documenting it as
    required on the dataclass, so every declarer that used this helper got the
    default and the guarantee was decorative.
    """
    if layers is None:
        layers = _resolve(config, ("num_hidden_layers", "encoder_layers", "num_layers"), "layer count")
    if kv_heads is None:
        kv_heads = _resolve(
            config,
            ("num_key_value_heads", "num_attention_heads", "encoder_attention_heads"),
            "kv head count",
        )
    if head_dim is None:
        explicit = getattr(config, "head_dim", None)
        if explicit is not None:
            head_dim = int(explicit)
        else:
            hidden = _resolve(config, ("hidden_size", "d_model"), "hidden size")
            heads = _resolve(config, ("num_attention_heads", "encoder_attention_heads"), "attention head count")
            head_dim = hidden // heads

    return ModelLocalKVSpec(
        name=name,
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        physical_capacity_positions=physical_capacity_positions,
        capacity_source=capacity_source,
        scope=scope,
        rows=rows,
        rows_fixed=rows_fixed,
        rows_reason=rows_reason,
        allocation_note=allocation_note,
    )


@runtime_checkable
class HasModelLocalKV(Protocol):
    """Implemented by whichever object owns the cache.

    Implement it on the owner, not on the registered model. The owner is
    usually several levels down (a codec decoder, a talker backbone), and
    ``collect_model_local_kv_specs`` finds it by walking the module tree, so
    no intermediate class has to forward anything.
    """

    def model_local_kv_specs(self) -> Sequence[ModelLocalKVSpec]: ...


def _unwrap_to_module(model: object) -> object:
    """Unwrap runner-side wrappers down to the real module tree.

    ``vllm.compilation.cuda_graph.CUDAGraphWrapper`` is a plain callable, not
    an ``nn.Module``, but its ``__getattr__`` forwards anything the runnable
    has -- including ``named_modules``. So a check for ``named_modules`` does
    not detect the wrapper, and the wrapper and its runnable both answer to the
    same attributes. That matters twice over: it means walking the wrapper
    happens to work, and it means a naive walk can enter the tree as both the
    wrapper and the root module and count a root-level declarer twice.

    Prefer the supported ``unwrap()`` accessor, which returns the runnable
    directly.
    """
    seen: set[int] = set()
    current = model
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        unwrap = getattr(current, "unwrap", None)
        if not callable(unwrap):
            return current
        try:
            nxt = unwrap()
        except Exception:
            return current
        if nxt is None or nxt is current:
            return current
        current = nxt
    return model


def collect_model_local_kv_specs(model: object) -> list[tuple[str, ModelLocalKVSpec]]:
    """Collect every declaration in a loaded model, with the owner's path.

    Walks ``named_modules()`` because the cache owner is an inner module in all
    known cases. Requiring each registered model to forward the call would put
    the burden on classes that do not own a cache and would silently report
    zero the moment someone forgot -- exactly the failure mode this is meant to
    surface.

    A raising declaration is logged and skipped rather than propagated:
    reporting memory must not be able to break model load.
    """
    root = _unwrap_to_module(model)

    owners: list[tuple[str, object]] = []
    seen_ids: set[int] = set()
    if isinstance(root, HasModelLocalKV):
        owners.append(("", root))
        seen_ids.add(id(root))

    named_modules = getattr(root, "named_modules", None)
    if callable(named_modules):
        for path, module in named_modules():
            if id(module) in seen_ids or not isinstance(module, HasModelLocalKV):
                continue
            seen_ids.add(id(module))
            owners.append((path, module))

    collected: list[tuple[str, ModelLocalKVSpec]] = []
    for path, owner in owners:
        try:
            specs = owner.model_local_kv_specs()
        except Exception:
            logger.warning(
                "model_local_kv_specs() raised on %s; skipping its declaration",
                type(owner).__name__,
                exc_info=True,
            )
            continue
        collected.extend((path, spec) for spec in specs)
    return collected
