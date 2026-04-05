"""Monkey-patch vLLM's MooncakeConnector to fix request-ID mismatch in PD disaggregation.

vLLM's InputProcessor appends a random suffix to each request ID. The prefill
engine stores KV under its suffix, but the decode engine generates a different
suffix. This patch threads ``remote_request_id`` through ``kv_transfer_params``
so the decode side references the correct KV entry.
"""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_patched: bool = False


@dataclass
class PatchedRecvReqMeta:
    """Receive-request metadata carrying the prefill engine's request ID."""

    request_id: str
    remote_request_id: str
    local_block_ids: list[int]
    kv_transfer_params: dict[str, Any]


def _import_mooncake_module():
    """Import MooncakeConnector module, supporting both vLLM >=0.16 and older."""
    for mod_path in (
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector",
    ):
        try:
            return importlib.import_module(mod_path)
        except (ImportError, ModuleNotFoundError):
            continue
    return None


def _create_patched_mooncake_connector():
    """Return a subclass of MooncakeConnector with remote_request_id support."""
    _mc_mod = _import_mooncake_module()
    if _mc_mod is None:
        raise ImportError("Cannot import MooncakeConnector from upstream vLLM")
    _OriginalMooncakeConnector = _mc_mod.MooncakeConnector

    class PatchedMooncakeConnector(_OriginalMooncakeConnector):
        """Fixes request-ID mismatch in PD disaggregation by injecting
        remote_request_id on the prefill side and using it for KV lookup
        on the decode side.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.remote_to_local_req: dict[str, str] = {}
            logger.info("[PatchedMooncakeConnector] Initialized")

        def request_finished(
            self,
            request: Any,
            block_ids: list[int],
        ) -> tuple[bool, dict[str, Any] | None]:
            result = super().request_finished(request, block_ids)

            if isinstance(result, tuple) and len(result) == 2:
                delay_free, kv_params = result
            else:
                delay_free, kv_params = False, result

            # Normalise _reqs_need_send values
            req_id = getattr(request, "request_id", None)
            if req_id and hasattr(self, "_reqs_need_send"):
                entry = self._reqs_need_send.get(req_id)
                if isinstance(entry, tuple) and len(entry) == 2:
                    self._reqs_need_send[req_id] = entry[1]

            # Inject remote_request_id into kv_transfer_params
            if kv_params is not None and isinstance(kv_params, dict):
                kv_params["remote_request_id"] = req_id or "NOT_SET"
                if hasattr(self, "side_channel_host"):
                    kv_params.setdefault("remote_host", self.side_channel_host)
                if hasattr(self, "side_channel_port"):
                    kv_params.setdefault("remote_port", self.side_channel_port)

            return delay_free, kv_params

        def add_new_req(
            self,
            request_id: str,
            local_block_ids: list[int],
            kv_transfer_params: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            super().add_new_req(request_id, local_block_ids, kv_transfer_params, **kwargs)

            kv_transfer_params = kv_transfer_params or {}
            load_remote_cache = kv_transfer_params.get(
                "do_remote_prefill",
                kv_transfer_params.get("load_remote_cache", False),
            )

            if load_remote_cache:
                remote_request_id = kv_transfer_params.get("remote_request_id", request_id)
                meta = PatchedRecvReqMeta(
                    request_id=request_id,
                    remote_request_id=remote_request_id,
                    local_block_ids=local_block_ids,
                    kv_transfer_params=kv_transfer_params,
                )
                if not hasattr(self, "_reqs_need_recv"):
                    self._reqs_need_recv = {}
                self._reqs_need_recv[request_id] = meta

        def group_kv_pull(self, metadata: Any | None = None) -> None:
            """Use remote_request_id as ZMQ lookup key via save-patch-restore."""
            if not hasattr(self, "_reqs_need_recv") or not self._reqs_need_recv:
                return

            original_recv = self._reqs_need_recv.copy()
            patched_recv: dict[str, Any] = {}

            for local_id, meta in original_recv.items():
                if isinstance(meta, PatchedRecvReqMeta):
                    remote_id = meta.remote_request_id
                    self.remote_to_local_req[remote_id] = local_id
                    patched_meta = type(meta)(
                        request_id=remote_id,
                        remote_request_id=remote_id,
                        local_block_ids=meta.local_block_ids,
                        kv_transfer_params=meta.kv_transfer_params,
                    )
                    patched_recv[remote_id] = patched_meta
                else:
                    patched_recv[local_id] = meta

            self._reqs_need_recv = patched_recv
            super().group_kv_pull(metadata)

            # Restore unconsumed entries to original local keys
            for remote_id, local_id in list(self.remote_to_local_req.items()):
                if remote_id in self._reqs_need_recv:
                    entry = self._reqs_need_recv.pop(remote_id)
                    self._reqs_need_recv[local_id] = original_recv.get(local_id, entry)

        def receive_kv(self, path: Any = None, req_blocks: Any = None) -> Any:
            result = super().receive_kv(path, req_blocks)

            if self.remote_to_local_req:
                completed = [
                    rid
                    for rid, lid in self.remote_to_local_req.items()
                    if not hasattr(self, "_reqs_need_recv") or lid not in self._reqs_need_recv
                ]
                for remote_id in completed:
                    self.remote_to_local_req.pop(remote_id, None)

            return result

    PatchedMooncakeConnector.__qualname__ = _OriginalMooncakeConnector.__qualname__

    return PatchedMooncakeConnector


def apply_mooncake_connector_patch() -> bool:
    """Replace vLLM's MooncakeConnector with the patched version."""
    global _patched
    if _patched:
        return True

    _mc_module = _import_mooncake_module()
    if _mc_module is None:
        logger.warning("[monkey_patch] Cannot import MooncakeConnector — patch NOT applied.")
        return False

    _OriginalClass = _mc_module.MooncakeConnector

    PatchedClass = _create_patched_mooncake_connector()

    _mc_module.MooncakeConnector = PatchedClass
    for _, module in sys.modules.items():
        if hasattr(module, "MooncakeConnector") and module.MooncakeConnector is _OriginalClass:
            module.MooncakeConnector = PatchedClass

    _patched = True
    logger.info("[monkey_patch] MooncakeConnector patch applied")
    return True
