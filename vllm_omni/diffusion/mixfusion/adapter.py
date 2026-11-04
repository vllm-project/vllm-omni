# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-side MixFusion adapter protocol.

MixFusion has three layers: a common driver (admission and step lifecycle), a
common layout/hook layer (:class:`PackedExecutionPlan` and the module-boundary
hooks), and a model-side adapter implementing this protocol, which alone knows
the model's forward signature, packed-call shape and output layout.

The framework never interprets the model-specific payloads in
:class:`PackedInputs` (``boundary_outputs`` and ``adapter_state``); adding a
model needs a new adapter and plan declaration, not driver or hook changes.
"""

from __future__ import annotations

from collections.abc import Hashable
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

    Declared through :meth:`MixFusionAdapter.build_attention_layout`; stream
    order is the token order inside every packed request.
    """

    name: str
    kind: StreamKind

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("StreamSpec.name must be non-empty")
        if self.kind not in ("chunked", "request"):
            raise ValueError(f"StreamSpec {self.name!r} has unknown kind {self.kind!r}")


@dataclass(frozen=True)
class AttentionLayout:
    """The model's declared attention streams, in per-request token order."""

    streams: tuple[StreamSpec, ...]

    def __post_init__(self) -> None:
        if not self.streams:
            raise ValueError("AttentionLayout needs at least one stream")
        names = [spec.name for spec in self.streams]
        if len(set(names)) != len(names):
            raise ValueError(f"AttentionLayout stream names must be unique, got {names}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.streams)


@dataclass
class PackedInputs:
    """Rewritten forward call plus the layout it runs under.

    Produced by :meth:`MixFusionAdapter.pack_inputs`. ``boundary_outputs``
    replaces the outputs of boundaries only the adapter can compute;
    ``adapter_state`` carries adapter-private reuse data for the next step.
    """

    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    plan: PackedExecutionPlan
    # boundary key -> value the framework returns instead of running that
    # module's forward. Opaque to the framework.
    boundary_outputs: dict[str, Any] = field(default_factory=dict)
    # Step-invariant reuse key chosen by the adapter.
    cache_key: Hashable | None = None
    # Adapter-private bundle handed back to ``pack_inputs(..., cache=...)`` on
    # the next step. Opaque to the framework.
    adapter_state: dict[str, Any] | None = None


@runtime_checkable
class MixFusionAdapter(Protocol):
    """Model-side contract for the MixFusion packed path.

    The root hook binds the forward call, asks the adapter whether it can
    pack, delegates packing, and restores the per-request output list.
    """

    def build_packed_call(self, input_batch: Any) -> dict[str, Any]:
        """Build this model's packed forward keyword arguments.

        The driver calls ``transformer(**build_packed_call(input_batch))``
        under an armed step; the root hook turns that call into the packed
        layout. Only the adapter knows the model's forward contract.
        """
        ...

    def can_pack(self, module: nn.Module, bound: Any) -> tuple[bool, str]:
        """Whether this call is packable; return ``(False, reason)`` to stay serial.

        ``bound`` is an :class:`inspect.BoundArguments` over the model's
        forward. The driver calls this during admission to choose the path;
        the root hook calls it again on the real call as an assertion (a False
        answer there aborts the packed step), so implementations must be
        deterministic for equivalent arguments. Raise ``TypeError`` when the
        call lacks a required argument (e.g. ``hidden_states``); admission
        treats that as a refusal.
        """
        ...

    def build_attention_layout(self, plan: PackedExecutionPlan | None = None) -> AttentionLayout:
        """Declare the attention streams this model packs.

        The layout is step-invariant, so ``plan`` may be ``None`` while the
        adapter is still building it. The packed-attention entry point reads
        it off the armed step to split QKV and reassemble outputs.
        """
        ...

    def pack_inputs(
        self, module: nn.Module, bound: Any, *, step: Any, cache: PackedInputs | None
    ) -> PackedInputs | None:
        """Rewrite the bound forward call into the packed layout.

        ``cache`` is the bundle from the previous step (or ``None``); use it to
        skip step-invariant work. Returning ``None`` keeps the serial path.
        """
        ...

    def prepare_position_embeddings(self, module: nn.Module, plan: Any, cache: Any = None) -> Any:
        """Build the step-invariant position embeddings for a packed step.

        Only the adapter's own ``pack_inputs`` path calls and implements it;
        the protocol entry keeps that model-side responsibility explicit and
        testable. ``plan``, ``cache`` and the return value are opaque payloads
        the framework only carries.
        """
        ...

    def restore_output(self, module: nn.Module, output: Any, plan: PackedExecutionPlan) -> list[Any]:
        """Turn the packed forward output back into one output per request."""
        ...
