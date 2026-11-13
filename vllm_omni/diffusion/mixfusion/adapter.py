# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MixFusion adapter protocol and attention-stream layout types.

The root hook packs and restores a forward call through an object
implementing :class:`MixFusionAdapter`. Models with a declarative
``_mixfusion_plan`` get the framework's
:class:`~vllm_omni.diffusion.mixfusion.chunked.ChunkedJointAdapter`; an
explicit ``mixfusion_adapter`` is the escape hatch for a layout family the
declarative plan cannot express.

The hooks never interpret the payloads in :class:`PackedInputs`
(``boundary_outputs`` and ``adapter_state``); they only move them between the
boundaries the plan declares.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from torch import nn

from vllm_omni.diffusion.mixfusion.execution_plan import PackedExecutionPlan

#: How a stream's tokens are laid out across the packed rows. ``"chunked"``
#: streams arrive as one row per (request, chunk) and carry one chunk per
#: position in the second dimension; ``"request"`` streams arrive as one row
#: per request with the real token count in ``request_stream_lens``.
StreamKind = Literal["chunked", "request"]


@dataclass(frozen=True)
class StreamSpec:
    """One attention stream of a packed model call.

    Stream order in the adapter's :class:`AttentionLayout` is the token order
    inside every packed request.
    """

    name: str
    kind: StreamKind


@dataclass(frozen=True)
class AttentionLayout:
    """The model's attention streams, in per-request token order.

    Exactly one stream is ``"chunked"``: it is the positional Q/K/V of the
    packed attention call. The adapter stores the layout on the plan.
    """

    streams: tuple[StreamSpec, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.streams)


@dataclass
class PackedInputs:
    """Rewritten forward call produced by :meth:`MixFusionAdapter.pack_inputs`.

    ``boundary_outputs`` replaces the outputs of boundaries only the adapter
    can compute; ``adapter_state`` carries adapter-private reuse data for the
    next step (step-invariant layout only — it must not reference the input
    latents, text embeddings or masks, which live on the active step). Both
    are opaque to the framework.
    """

    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    plan: PackedExecutionPlan
    boundary_outputs: dict[str, Any] = field(default_factory=dict)
    adapter_state: Any = None


@runtime_checkable
class MixFusionAdapter(Protocol):
    """Model-side contract for the MixFusion packed path.

    Scheduler admission precedes packing. Worker validation checks only that
    the actual inputs satisfy the model contract; it never chooses a fallback.
    """

    def validate_inputs(self, module: nn.Module, input_batch: Any) -> None:
        """Raise when runtime inputs disagree with the scheduled packed contract."""
        ...

    def pack_inputs(self, module: nn.Module, bound: Any, *, adapter_state: Any | None) -> PackedInputs:
        """Rewrite the bound forward call into the packed layout.

        ``bound`` is an :class:`inspect.BoundArguments` over the model's
        forward. ``adapter_state`` is this branch's value from the previous
        step's ``PackedInputs`` (or ``None``); it holds only step-invariant
        layout data, so reusing it must never keep inputs alive.
        """
        ...

    def restore_output(self, module: nn.Module, output: Any, plan: PackedExecutionPlan) -> list[Any]:
        """Turn the packed forward output back into one output per request."""
        ...
