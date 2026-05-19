# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lifecycle wrapper around :class:`OmniCoordinator`.

``OmniCoordinatorRuntime`` is the single-purpose owner of the head-side
coordinator process artifacts: it picks two free TCP ports, constructs an
:class:`OmniCoordinator` bound to them, exposes the resulting addresses, and
provides a single ``close()`` method to tear everything down.

The ROUTER address is later handed to :class:`OmniMasterServer` so it can be
published to registering replicas; the PUB address is handed to the
``Orchestrator``, which constructs its :class:`OmniCoordClientForHub` against
it.
"""

from __future__ import annotations

import logging

from vllm.utils.network_utils import get_open_ports_list

from .omni_coordinator import OmniCoordinator

logger = logging.getLogger(__name__)


class OmniCoordinatorRuntime:
    """Own one :class:`OmniCoordinator` and the two ports it binds.

    Constructor binds; :meth:`close` tears down. The class deliberately does
    not expose the coordinator instance — callers should consume the
    coordinator only via its wire protocol through
    :class:`OmniCoordClientForStage` and :class:`OmniCoordClientForHub`.
    """

    def __init__(
        self,
        *,
        host: str,
        heartbeat_timeout: float,
    ) -> None:
        if not host:
            raise ValueError("host must be a non-empty string")
        if heartbeat_timeout <= 0:
            raise ValueError("heartbeat_timeout must be positive")

        router_port, pub_port = get_open_ports_list(count=2)
        self.router_address: str = f"tcp://{host}:{router_port}"
        self.pub_address: str = f"tcp://{host}:{pub_port}"

        self._closed = False
        self._coordinator = OmniCoordinator(
            router_zmq_addr=self.router_address,
            pub_zmq_addr=self.pub_address,
            heartbeat_timeout=heartbeat_timeout,
        )

        logger.info(
            "[OmniCoordinatorRuntime] Started (router=%s pub=%s heartbeat_timeout=%.1fs)",
            self.router_address,
            self.pub_address,
            heartbeat_timeout,
        )

    def close(self) -> None:
        """Tear down the underlying coordinator. Idempotent."""
        if self._closed:
            return
        self._closed = True
        try:
            self._coordinator.close()
        except Exception:
            logger.exception("[OmniCoordinatorRuntime] coordinator close failed")
