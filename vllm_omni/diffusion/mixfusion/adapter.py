# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-side MixFusion adapter protocol.

MixFusion is split into three layers:

- a common driver that decides *which* requests may be packed and owns the
  step lifecycle (forward context, fallback to the serial path);
- common hooks and :class:`~vllm_omni.diffusion.mixfusion.execution_plan.PackedExecutionPlan`
  that implement the chunk-stacked layout at declared module boundaries;
- a model-side adapter implementing this protocol, which alone knows the
  model's forward signature, its packed-call shape and its output layout.

The framework never interprets the model-specific payloads carried by
:class:`PackedInputs` (``boundary_outputs`` and ``adapter_state``); it only
moves them between the hooks that a model's ``_mixfusion_plan`` declares.
Adding a model therefore needs a new adapter file and plan declaration, not
changes in the driver or hooks.
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

    A model declares its streams through
    :meth:`MixFusionAdapter.build_attention_layout`. Stream order is the token
    order inside every packed request (e.g. text then image chunks).
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

    Produced by :meth:`MixFusionAdapter.pack_inputs` from the caller's bound
    forward arguments. ``args`` / ``kwargs`` are what the model forward is
    actually invoked with; ``boundary_outputs`` replaces the outputs of
    boundaries that only the adapter can produce (keyed by the names its plan
    declares), and ``adapter_state`` carries whatever the adapter wants to
    reuse on the next denoise step (e.g. step-invariant RoPE tables).
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

    The common root hook binds the model forward call, asks the adapter
    whether it can pack, delegates packing to it, and asks it to restore the
    per-request output list afterwards. Everything model-specific lives
    behind these methods.
    """

    def can_pack(self, module: nn.Module, bound: Any) -> tuple[bool, str]:
        """Whether this call is packable.

        ``bound`` is an :class:`inspect.BoundArguments` over the model's
        forward. Return ``(False, reason)`` to keep the serial path; the
        reason is used in diagnostics.
        """
        ...

    def build_attention_layout(self, plan: PackedExecutionPlan | None = None) -> AttentionLayout:
        """Declare the attention streams this model packs.

        The layout is step-invariant, so ``plan`` may be ``None`` while the
        adapter is still building it. The packed-attention entry point reads
        the layout off the armed step to know how to split the QKV inputs and
        reassemble the outputs.
        """
        ...

    def pack_inputs(
        self, module: nn.Module, bound: Any, *, step: Any, cache: PackedInputs | None
    ) -> PackedInputs | None:
        """Rewrite the bound forward call into the packed layout.

        ``cache`` is the bundle this adapter returned on the previous step (or
        ``None``); use it to skip step-invariant work. Returning ``None``
        keeps the serial path.
        """
        ...

    def restore_output(self, module: nn.Module, output: Any, plan: PackedExecutionPlan) -> list[Any]:
        """Turn the packed forward output back into one output per request."""
        ...
