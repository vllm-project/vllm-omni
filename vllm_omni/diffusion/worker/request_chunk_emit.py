# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bind the request-mode chunk emitter for the duration of one forward.

The worker installs the callable around ``execute_model``. Pipelines read it
with :func:`get_emit_chunk`. ``None`` means the non-streaming path: decode may
still overlap the forward, but pixels stay on the worker until the final
output. The emitter never talks to the client; it only enqueues worker-local
``CHUNK_READY`` work.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar, Token

from vllm_omni.diffusion.data import DiffusionOutput

EmitChunkFn = Callable[[DiffusionOutput], None]

_emit_chunk: ContextVar[EmitChunkFn | None] = ContextVar(
    "vllm_omni_request_emit_chunk",
    default=None,
)


def bind_emit_chunk(fn: EmitChunkFn | None) -> Token[EmitChunkFn | None]:
    """Install ``fn`` for the current context. Pass ``None`` to suppress emits."""
    return _emit_chunk.set(fn)


def get_emit_chunk() -> EmitChunkFn | None:
    """Return the emitter bound for this context, or ``None``."""
    return _emit_chunk.get()


def reset_emit_chunk(token: Token[EmitChunkFn | None]) -> None:
    """Restore the emitter installed before :func:`bind_emit_chunk`."""
    _emit_chunk.reset(token)
