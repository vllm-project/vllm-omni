# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Hidden-state store/restore over the LMCache multiprocess cache server.

The in-process path reaches into ``engine.hidden_state_store`` and files one
entry per (chunk, layer). MP has no in-process engine: the cache lives in
another process and the worker talks to it over RPC, so the layers a request
needs travel together as one object per chunk and the layer set is declared
once at registration.
"""

import pickle
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# The RPCs are small and synchronous. A worker that cannot reach the cache
# server within this budget should fall back to recompute, not stall a step.
_RPC_TIMEOUT_S = 10.0


def layer_order(mm_keys: tuple[str, ...]) -> tuple[str, ...]:
    """Return the layer keys in the order they are packed along dim 1.

    Store and restore both derive the order from the same config, so the
    index of a layer in the stacked tensor is stable for a given engine.
    """
    return (*mm_keys, "hidden")


def stack_layers(
    layers: dict[str, torch.Tensor], mm_keys: tuple[str, ...]
) -> torch.Tensor | None:
    """Pack ``{layer_key: [rows, hidden]}`` into ``[rows, num_layers, hidden]``.

    Returns ``None`` when a layer is missing or the layers disagree on width.
    One object per chunk means a single shape for every layer, which the MP
    server is told once at registration.
    """
    order = layer_order(mm_keys)
    if set(order) != set(layers):
        logger.error(
            "LMCache MP: layer set mismatch (expected=%s, got=%s); skipping",
            order,
            sorted(layers),
        )
        return None
    widths = {int(layers[key].shape[-1]) for key in order}
    if len(widths) != 1:
        logger.error(
            "LMCache MP: layers disagree on hidden size (%s); the MP cache "
            "stores one object per chunk covering every layer, which needs a "
            "single width. Use the in-process path for this model.",
            {key: tuple(layers[key].shape) for key in order},
        )
        return None
    return torch.stack([layers[key] for key in order], dim=1)


def unstack_layers(
    stacked: torch.Tensor, mm_keys: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    """Split ``[rows, num_layers, hidden]`` back into per-layer tensors."""
    order = layer_order(mm_keys)
    return {key: stacked[:, idx, :] for idx, key in enumerate(order)}


class MPHiddenStateBackend:
    """Store and retrieve hidden states through the MP cache server.

    Args:
        adapter: The ``LMCacheMPWorkerAdapter`` the KV connector already owns.
            Reusing it keeps the hidden states on the same server connection,
            instance id, and chunk size as the KV they accompany.
    """

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter
        self._registered_shape: tuple[int, int, str] | None = None

    @property
    def chunk_size(self) -> int:
        """Token count per cache chunk, as the server reported it."""
        return int(self._adapter.lmcache_tokens_per_chunk)

    def ensure_registered(self, stacked: torch.Tensor) -> bool:
        """Declare the layout on first use, from the first tensor to be stored.

        The layer count and hidden size are a property of the model, not of a
        request, so this runs once. A later tensor of a different shape is a
        bug upstream rather than something to re-register.
        """
        num_layers = int(stacked.shape[1])
        hidden_size = int(stacked.shape[2])
        dtype_name = str(stacked.dtype).split(".")[-1]
        shape = (num_layers, hidden_size, dtype_name)
        if self._registered_shape == shape:
            return True
        if self._registered_shape is not None:
            logger.error(
                "LMCache MP: hidden-state shape changed (%s -> %s); not storing",
                self._registered_shape,
                shape,
            )
            return False
        try:
            ok = self._adapter.req_client.register_hidden_state(
                self._adapter.instance_id,
                self._adapter.model_name,
                self._adapter.world_size,
                num_layers,
                hidden_size,
                dtype_name,
            ).result(timeout=_RPC_TIMEOUT_S)
        except Exception:
            logger.exception("LMCache MP: register_hidden_state failed")
            return False
        if not ok:
            logger.error("LMCache MP: server refused the hidden-state layout %s", shape)
            return False
        self._registered_shape = shape
        logger.info(
            "LMCache MP hidden-state layout registered (layers=%d, hidden=%d, "
            "dtype=%s, chunk=%d)",
            num_layers,
            hidden_size,
            dtype_name,
            self.chunk_size,
        )
        return True

    def store(
        self,
        token_ids: list[int],
        stacked: torch.Tensor,
        token_offset: int,
        request_id: str,
    ) -> bool:
        """Store ``stacked`` as the chunks covering ``[token_offset, end)``.

        Args:
            token_ids: Keyed token ids of the whole prefix up to the end of
                the range; the server hashes them to address the chunks.
            stacked: ``[rows, num_layers, hidden]``, chunk-aligned.
            token_offset: First token index the rows cover.
            request_id: The vLLM request id.

        Returns:
            Whether every chunk was stored.
        """
        if not self.ensure_registered(stacked):
            return False
        rows = int(stacked.shape[0])
        chunks = list(stacked.split(self.chunk_size, dim=0))
        key = self._adapter._create_key(
            token_ids, token_offset, token_offset + rows, request_id
        )
        try:
            return bool(
                self._adapter.req_client.store_hidden_state(
                    key, self._adapter.instance_id, pickle.dumps(chunks)
                ).result(timeout=_RPC_TIMEOUT_S)
            )
        except Exception:
            logger.exception(
                "LMCache MP: store_hidden_state failed (req_id=%s)", request_id
            )
            return False

    def retrieve(self, token_ids: list[int], request_id: str) -> torch.Tensor | None:
        """Return the cached ``[tokens, num_layers, hidden]`` prefix, or None.

        The returned prefix may be shorter than ``token_ids``; the caller
        decides whether that is enough.
        """
        key = self._adapter._create_key(token_ids, 0, len(token_ids), request_id)
        try:
            data, num_tokens = self._adapter.req_client.retrieve_hidden_state(
                key, self._adapter.instance_id
            ).result(timeout=_RPC_TIMEOUT_S)
        except Exception:
            logger.exception(
                "LMCache MP: retrieve_hidden_state failed (req_id=%s)", request_id
            )
            return None
        if not num_tokens or not data:
            return None
        return torch.cat(pickle.loads(data), dim=0)

    def lookup(self, token_ids: list[int], request_id: str) -> int:
        """Return how many leading tokens have hidden states cached."""
        key = self._adapter._create_key(token_ids, 0, len(token_ids), request_id)
        try:
            return int(
                self._adapter.req_client.lookup_hidden_state(
                    key, self._adapter.instance_id
                ).result(timeout=_RPC_TIMEOUT_S)
            )
        except Exception:
            logger.exception(
                "LMCache MP: lookup_hidden_state failed (req_id=%s)", request_id
            )
            return 0


def find_mp_backend() -> MPHiddenStateBackend | None:
    """Return a backend bound to the MP worker adapter, if one is in use.

    Mirrors how the in-process path finds its adapter: walk the KV transfer
    group, including a MultiConnector's children.
    """
    try:
        from vllm.distributed.kv_transfer import (
            get_kv_transfer_group,
            has_kv_transfer_group,
        )

        if not has_kv_transfer_group():
            return None
        connector = get_kv_transfer_group()
        candidates = getattr(connector, "_connectors", None) or [connector]
        for candidate in candidates:
            adapter = getattr(candidate, "worker_adapter", None)
            if adapter is not None and hasattr(adapter, "req_client"):
                return MPHiddenStateBackend(adapter)
    except Exception:
        logger.debug("LMCache MP: no MP worker adapter found", exc_info=True)
    return None
