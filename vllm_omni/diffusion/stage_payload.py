# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch-free transport helpers for disaggregated diffusion stage payloads.

RFC #4590 §2.3 requires a small typed envelope that carries the *data* one
diffusion stage produces for the next, kept strictly separate from the mutable
runner-local ``DiffusionRequestState``. A ``DiffusionRequestState`` must never
cross a process boundary; a stage payload may.

The envelope type itself is now
:class:`~vllm_omni.diffusion.models.interface.StagePayload` (the #4948
``DiffusionV2Atoms`` contract). This module remains the **torch-free transport
leaf** every stage layer imports, and is the single source of truth for:

* the two payload-key constants (:data:`STAGE_PAYLOAD_OUTPUT_KEY` /
  :data:`STAGE_PAYLOAD_PROMPT_KEY`) and the schema version;
* tensor sanitization at the export boundary
  (:func:`sanitize_transport_tensor`: detach -> host -> contiguous -> clone);
* rejection of non-transportable values — modules, generators, CUDA streams,
  schedulers, process-local cache objects, device pointers
  (:func:`_looks_non_transportable`, :func:`validate_transport_scalar`).

``StagePayload`` (in ``interface.py``, which stays torch-free) calls these free
functions so the transport safety lives here without pulling torch into the
capability-probe module. The first implementation uses the existing host-based
(msgpack over ZMQ, or in-process) stage transport; host-to-device movement
happens once at the *restore* boundary inside the owning pipeline.
"""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


#: Bump when the envelope's wire shape changes incompatibly. Consumers reject
#: payloads whose ``schema_version`` they do not understand.
DIFFUSION_STAGE_PAYLOAD_SCHEMA_VERSION = 1

#: The two channels a :class:`StagePayload` travels between stages, kept
#: here (the torch-free transport leaf every stage layer already imports) as the
#: single source of truth so the runner, the generic transition processor, and
#: the AR-Diffusion runner cannot drift:
#:
#: * :data:`STAGE_PAYLOAD_OUTPUT_KEY` — key under which the *producer* stage's
#:   ``DiffusionOutput.custom_output`` carries the outgoing payload.
#: * :data:`STAGE_PAYLOAD_PROMPT_KEY` — key under which the *consumer* stage's
#:   request prompt ``extra`` dict carries the incoming payload.
STAGE_PAYLOAD_OUTPUT_KEY = "__diffusion_stage_payload__"
STAGE_PAYLOAD_PROMPT_KEY = "diffusion_stage_payload"

#: Metadata values are restricted to these plain, transportable Python types
#: (recursively for containers). Everything else is rejected at validation.
#: ``bytes`` is allowed for opaque model-private blobs; numpy scalars/arrays are
#: allowed because the existing msgpack transport already supports them.
_ALLOWED_METADATA_SCALARS = (str, bool, bytes, numbers.Number, type(None))


class StagePayloadError(ValueError):
    """Raised when a stage payload fails validation."""


class NonTransportableValueError(StagePayloadError):
    """Raised when payload metadata/tensors contain a non-transportable value.

    The message names the offending key path and type so the failure points at
    the model code that packed it, not at the generic transport layer.
    """


def _is_tensor(value: Any) -> bool:
    """Duck-typed torch.Tensor check that does not import torch eagerly."""
    cls = type(value)
    # torch.Tensor instances report a module of "torch"; checking the MRO names
    # avoids importing torch in environments (config-only tools, tests) that do
    # not have it while still recognizing real tensors at runtime.
    return any(base.__module__ == "torch" and base.__name__ == "Tensor" for base in cls.__mro__)


def _looks_non_transportable(value: Any) -> str | None:
    """Return a reason string if ``value`` is clearly non-transportable.

    Catches the specific hazards RFC #4590 §2.3 enumerates: live modules,
    generators, CUDA streams, schedulers, and other process-local objects. This
    is a *positive* guard against obviously-unsafe types; the metadata validator
    additionally enforces an allow-list, so novel unsafe types are still caught.
    """
    cls = type(value)
    module = cls.__module__ or ""
    name = cls.__name__

    if callable(value) and (module or name):
        # Functions, methods, lambdas, and class objects are not data.
        if module == "builtins" and name in ("function", "builtin_function_or_method"):
            return "callable"
        if cls.__name__ in ("function", "method", "builtin_function_or_method", "type"):
            return "callable"

    # torch process-local objects.
    if module.startswith("torch"):
        if name in ("Generator", "Stream", "Event", "device", "dtype"):
            return f"torch.{name}"
        if "Module" in name or name.endswith("Module"):
            return f"torch module ({name})"

    # Diffusion scheduler instances and runner-local state objects.
    lowered = name.lower()
    if "scheduler" in lowered:
        return f"scheduler object ({name})"
    if name in ("DiffusionRequestState", "DreamZeroState", "ARDiffusionKVState", "ARDiffusionKVCache"):
        return f"process-local state object ({name})"
    if hasattr(value, "__next__") and hasattr(value, "__iter__") and name == "generator":
        return "generator"

    return None


def sanitize_transport_tensor(tensor: "torch.Tensor") -> "torch.Tensor":
    """Return a host-resident, detached, contiguous copy of ``tensor``.

    This is the single explicit device->host transport boundary (RFC #4590 §7):

    * ``detach()`` drops autograd linkage;
    * ``.cpu()`` performs the one device-to-host move;
    * ``.contiguous()`` makes the buffer serialization-safe;
    * ``.clone()`` guarantees the exported tensor never aliases a buffer the
      producer may reuse (a stage-boundary copy is acceptable for the first
      correctness implementation).

    dtype and shape are preserved exactly.
    """
    detached = tensor.detach()
    host = detached.cpu()
    contiguous = host.contiguous()
    # clone() only when .cpu()/.contiguous() returned a view onto live memory.
    if contiguous.data_ptr() == tensor.data_ptr():
        contiguous = contiguous.clone()
    return contiguous


def validate_transport_scalar(value: Any, *, path: str) -> None:
    """Raise :class:`NonTransportableValueError` if ``value`` is not transportable.

    Recursively enforces the metadata/scalar allow-list a stage payload carries
    across a process boundary: allow-listed scalars (str/bool/bytes/number/None),
    nested list/tuple/dict of those, and numpy arrays/scalars; everything else —
    tensors (which must ride the dedicated tensor channels), schedulers,
    generators, torch modules, and other process-local objects — is rejected.

    Extracted from the former ``DiffusionStagePayload._validate_metadata_value``
    so :class:`~vllm_omni.diffusion.models.interface.StagePayload` can reuse it
    without importing torch (this module is the torch-free transport leaf).
    """
    # Tensors do not belong in the scalar channels — they must be transported
    # via the dedicated tensor dicts so a future connector can back them with
    # handles without rewriting scalar packing.
    if _is_tensor(value):
        raise NonTransportableValueError(
            f"{path} is a torch.Tensor; tensors must be placed in a payload tensor dict, not a scalar dict."
        )
    reason = _looks_non_transportable(value)
    if reason is not None:
        raise NonTransportableValueError(f"{path} is non-transportable ({reason}).")

    if isinstance(value, _ALLOWED_METADATA_SCALARS):
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            validate_transport_scalar(item, path=f"{path}[{i}]")
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise StagePayloadError(f"{path} has non-string key {k!r}.")
            validate_transport_scalar(v, path=f"{path}[{k!r}]")
        return
    # numpy arrays/scalars are transportable via the existing msgpack codec.
    module = type(value).__module__ or ""
    if module.startswith("numpy"):
        return
    raise NonTransportableValueError(
        f"{path} has non-transportable type {type(value).__name__}. Allowed: "
        "str/bool/bytes/number/None, nested list/tuple/dict of those, numpy arrays, "
        "or a torch.Tensor placed in a payload tensor dict."
    )
