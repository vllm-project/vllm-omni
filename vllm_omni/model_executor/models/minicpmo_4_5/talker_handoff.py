# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Thinker-to-Talker hidden-state handoff over the Talker's stage connector.

``llm2tts`` runs in the orchestrator process and the Talker consumes its
handoff inside the stage-1 engine process. By default the Thinker hidden
states are converted to nested float lists and ride the request's
``model_intermediate_buffer`` through msgpack; for a typical reply that
conversion alone costs ~100 ms of orchestrator time per request.

When the Talker's stage connector opts in (``thinker_talker_handoff: true``
in its ``extra``), the producer instead ``put()``s the tensor on a connector
built from that same spec and the request carries only a small marker under
``hidden_states.tts``. The Talker resolves the marker with ``get()`` at its
first prefill. Nothing else changes: the request classes, the scheduler and
the default (list) path are untouched, and a deploy config without the
option never executes this module's transport code.

Ownership follows the SHM connector family: a payload is claimed by the first
reader (unlink), the producer unlinks by key on ``cleanup()``/``close()``, and
a handoff nobody consumed within ``thinker_talker_handoff_ttl_s`` is reaped
on the producer's next ``put()``.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import OrderedDict
from typing import Any

import regex as re
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

#: Marker key placed under ``model_intermediate_buffer["hidden_states"]["tts"]``.
HANDOFF_MARKER = "__minicpmo45_connector_handoff__"
#: Connector ``extra`` option that opts a deployment into the connector handoff.
HANDOFF_OPTION = "thinker_talker_handoff"
#: Connector ``extra`` option: seconds an unconsumed handoff stays alive.
HANDOFF_TTL_OPTION = "thinker_talker_handoff_ttl_s"
_DEFAULT_TTL_S = 600.0
_THINKER_STAGE = "0"
_TALKER_STAGE = "1"
_KEY_PREFIX = "mcpo45_handoff"
_UNSAFE_KEY_CHARS = re.compile(r"[^0-9A-Za-z_.-]")


def handoff_connector_spec(model_config: Any) -> tuple[str, dict[str, Any]] | None:
    """Return ``(name, extra)`` of the Talker stage connector if it opts in.

    ``model_config`` is the stage-1 ``OmniModelConfig``; both the orchestrator
    (through ``target_model_config``) and the Talker worker see the same
    ``stage_connector_config``, so both sides derive the same connector.
    """
    connector_config = getattr(model_config, "stage_connector_config", None)
    if isinstance(connector_config, dict):
        name, extra = connector_config.get("name"), connector_config.get("extra")
    else:
        name, extra = getattr(connector_config, "name", None), getattr(connector_config, "extra", None)
    if not isinstance(name, str) or not name.strip() or not isinstance(extra, dict):
        return None
    if not extra.get(HANDOFF_OPTION):
        return None
    return name.strip(), dict(extra)


def create_handoff_connector(spec: tuple[str, dict[str, Any]]) -> Any:
    """Instantiate the handoff connector from a ``handoff_connector_spec``."""
    from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
    from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec

    name, extra = spec
    return OmniConnectorFactory.create_connector(ConnectorSpec(name=name, extra=extra))


def handoff_ttl_s(spec: tuple[str, dict[str, Any]]) -> float:
    try:
        ttl = float(spec[1].get(HANDOFF_TTL_OPTION, _DEFAULT_TTL_S))
    except (TypeError, ValueError):
        return _DEFAULT_TTL_S
    return ttl if ttl > 0 else _DEFAULT_TTL_S


def is_handoff_marker(value: Any) -> bool:
    return isinstance(value, dict) and HANDOFF_MARKER in value


class TalkerHandoffProducer:
    """Orchestrator-side owner of unconsumed Thinker handoffs."""

    def __init__(self, connector: Any, *, ttl_s: float = _DEFAULT_TTL_S):
        self._connector = connector
        self._ttl_s = float(ttl_s)
        self._counter = itertools.count()
        # key -> (request_id, put time); insertion order is put order.
        self._pending: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def pending_keys(self) -> list[str]:
        with self._lock:
            return list(self._pending)

    def put(self, request_id: str, hidden_states: torch.Tensor) -> dict[str, Any] | None:
        """Store ``hidden_states`` for ``request_id`` and return its marker.

        Returns ``None`` when the connector refused the payload so the caller
        can fall back to the list handoff. Each put gets a fresh key: the
        duplex path re-puts per Talker condition and the previous payload may
        still be unread.
        """
        self._expire()
        key = f"{_KEY_PREFIX}_{_UNSAFE_KEY_CHARS.sub('_', str(request_id))}_{next(self._counter)}"
        try:
            ok, _size, metadata = self._connector.put(
                _THINKER_STAGE,
                _TALKER_STAGE,
                key,
                hidden_states.detach().cpu().contiguous(),
            )
        except Exception:
            logger.warning(
                "Thinker->Talker handoff put failed for %s; using the list handoff",
                request_id,
                exc_info=True,
            )
            return None
        if not ok:
            return None
        with self._lock:
            self._pending[key] = (str(request_id), time.monotonic())
        return {HANDOFF_MARKER: {"key": key, "metadata": metadata}}

    def cleanup(self, request_id: str) -> None:
        """Unlink every unconsumed handoff of ``request_id``."""
        request_id = str(request_id)
        with self._lock:
            keys = [key for key, (owner, _) in self._pending.items() if owner == request_id]
            for key in keys:
                del self._pending[key]
        for key in keys:
            self._cleanup_key(key)

    def close(self) -> None:
        with self._lock:
            keys = list(self._pending)
            self._pending.clear()
        for key in keys:
            self._cleanup_key(key)
        self._connector.close()

    def _cleanup_key(self, key: str) -> None:
        try:
            self._connector.cleanup(key)
        except Exception:
            logger.debug("Thinker->Talker handoff cleanup failed for %s", key, exc_info=True)

    def _expire(self) -> None:
        """Reap handoffs older than the TTL; a consumed key is a no-op unlink."""
        deadline = time.monotonic() - self._ttl_s
        expired: list[str] = []
        with self._lock:
            for key, (_, put_time) in self._pending.items():
                if put_time > deadline:
                    break
                expired.append(key)
            for key in expired:
                del self._pending[key]
        for key in expired:
            self._cleanup_key(key)


_producers: dict[tuple[str, str], TalkerHandoffProducer] = {}
_producers_lock = threading.Lock()


def get_talker_handoff_producer(model_config: Any) -> TalkerHandoffProducer | None:
    """Return the process-wide producer for the Talker's connector spec, or ``None``.

    ``None`` means the deployment did not opt in (or passed no stage config)
    and the caller must keep today's list handoff.
    """
    spec = handoff_connector_spec(model_config)
    if spec is None:
        return None
    name, extra = spec
    cache_key = (name, repr(sorted((key, repr(value)) for key, value in extra.items())))
    with _producers_lock:
        producer = _producers.get(cache_key)
        if producer is None:
            producer = TalkerHandoffProducer(create_handoff_connector(spec), ttl_s=handoff_ttl_s(spec))
            _producers[cache_key] = producer
    return producer


def resolve_talker_handoff(info: dict[str, Any], connector: Any) -> torch.Tensor | None:
    """Replace the marker under ``hidden_states.tts`` with the tensor it names.

    The replacement is made in place so the runner's per-request buffer keeps
    the tensor for later prefill chunks; the segment can only be claimed once.
    Returns ``None`` when the buffer carries no marker.
    """
    hidden_info = info.get("hidden_states")
    marker = hidden_info.get("tts") if isinstance(hidden_info, dict) else None
    if not is_handoff_marker(marker):
        return None
    ref = marker[HANDOFF_MARKER]
    key = ref.get("key") if isinstance(ref, dict) else None
    if not isinstance(key, str):
        raise ValueError("MiniCPM-o Talker connector handoff marker has no key")
    result = connector.get(_THINKER_STAGE, _TALKER_STAGE, key, ref.get("metadata"))
    if result is None:
        raise ValueError(
            f"MiniCPM-o Talker connector handoff {key!r} is unavailable: it was already consumed, "
            "cleaned up after the request was aborted, or expired after "
            f"{HANDOFF_TTL_OPTION} seconds"
        )
    tensor = result[0] if isinstance(result, tuple) else result
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"MiniCPM-o Talker connector handoff {key!r} did not hold a tensor: {type(tensor).__name__}")
    hidden_info["tts"] = tensor
    return tensor
