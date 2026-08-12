# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import torch

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)

_SCHEMA_VERSION = 1
_KIND_TENSORS = "tensors"
_KIND_STRUCTURED = "structured"
_KIND_OBJECT = "object"
_INIT_AGENT = "NIXL_INIT_AGENT"
_TENSOR_MARKER = "__nixl_tensor_index__"
_TUPLE_MARKER = "__nixl_tuple__"


class NixlConnector(OmniConnectorBase):
    """OmniConnector backed by vLLM's native NIXL wrapper.

    This connector intentionally depends on vLLM's optional NIXL integration
    (``vllm.distributed.nixl_utils``). It transfers raw tensor payloads
    directly through NIXL READ operations. Non-tensor Python
    payloads are serialized with OmniSerializer, packed into a uint8 CPU tensor,
    and moved through the same NIXL path.
    """

    supports_raw_data: bool = True

    def __init__(self, config: dict[str, Any]):
        self.config = dict(config or {})
        self._closed = True
        self._registered_descs: list[Any] = []
        self._pending: dict[str, tuple[list[torch.Tensor], list[Any], float]] = {}
        self._remote_agents: list[str] = []
        self._metrics: dict[str, int] = {
            "puts": 0,
            "gets": 0,
            "errors": 0,
            "bytes_transferred": 0,
        }

        from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config

        if NixlWrapper is None:
            raise RuntimeError("NIXL is not available. Install the optional nixl/rixl package to use NixlConnector.")

        backends = self.config.get("backends", ["UCX"])
        self._backends = list(backends) if isinstance(backends, (list, tuple)) else [str(backends)]
        if nixl_agent_config is None:
            agent_config = None
        else:
            non_ucx_backends = [backend for backend in self._backends if backend != "UCX"]
            if non_ucx_backends:
                agent_config = nixl_agent_config(backends=self._backends, capture_telemetry=False)
            else:
                num_threads = int(self.config.get("num_threads", 4))
                agent_config = nixl_agent_config(num_threads=num_threads, capture_telemetry=False)

        self._agent = NixlWrapper(str(self.config.get("agent_name", uuid.uuid4())), agent_config)
        self._receive_device = self._parse_device(self.config.get("receive_device"))
        self._default_memory_type = self.config.get("memory_type")
        # The lease bounds how long a ``put`` payload stays registered while it
        # waits to be read. A consumer stage may legitimately queue for a long
        # time (e.g. video diffusion with concurrency above the replica count),
        # so the default must be far larger than any plausible queueing delay.
        # The environment variables intentionally take precedence over the
        # deployment YAML, which ships a fixed 300s lease that is too short for
        # long-queueing pipelines and cannot be overridden per deployment.
        self._lease_seconds = float(
            os.environ.get("VLLM_OMNI_NIXL_LEASE_S") or self.config.get("lease_seconds", 3600.0)
        )
        self._transfer_timeout_s = float(
            os.environ.get("VLLM_OMNI_NIXL_XFER_TIMEOUT_S") or self.config.get("transfer_timeout_s", 300.0)
        )
        self._poll_interval_s = float(self.config.get("poll_interval_s", 0.001))
        self._closed = False

    def put(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: Any,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        if self._closed:
            raise RuntimeError("Cannot put data: NixlConnector is closed")

        try:
            self._cleanup_expired_pending()
            kind = _KIND_TENSORS
            if self._is_tensor_payload(data):
                tensors, tensor_specs = self._normalize_tensor_payload(data)
            elif self._contains_tensor(data):
                skeleton, payload_tensors = self._extract_tensor_leaves(data)
                header = self.serialize_obj(skeleton)
                header_tensor = torch.frombuffer(header, dtype=torch.uint8).clone().contiguous()
                tensors, tensor_specs = self._normalize_tensor_payload([header_tensor, *payload_tensors])
                kind = _KIND_STRUCTURED
            else:
                payload = self.serialize_obj(data)
                tensor = torch.frombuffer(payload, dtype=torch.uint8).clone().contiguous()
                tensors, tensor_specs = self._normalize_tensor_payload(tensor)
                kind = _KIND_OBJECT

            memory_type = self._resolve_memory_type(tensors[0])
            regions = [self._tensor_region(tensor) for tensor in tensors]
            reg_descs = self._agent.get_reg_descs(regions, memory_type)
            self._agent.register_memory(reg_descs, backends=self._backends)
            self._registered_descs.append(reg_descs)

            self._pending[put_key] = (tensors, [reg_descs], time.monotonic() + self._lease_seconds)
            size = sum(spec["size"] for spec in tensor_specs)
            metadata = {
                "schema_version": _SCHEMA_VERSION,
                "kind": kind,
                "agent_metadata": self._agent.get_agent_metadata(),
                "memory_type": memory_type,
                "regions": regions,
                "tensor_specs": tensor_specs,
                "size": size,
            }
            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += size
            logger.debug("NixlConnector put %s->%s key=%s size=%d", from_stage, to_stage, put_key, size)
            return True, size, metadata
        except Exception:
            self._metrics["errors"] += 1
            logger.error("NixlConnector put failed for %s", put_key, exc_info=True)
            return False, 0, None

    def get(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, int] | None:
        if self._closed:
            raise RuntimeError("Cannot get data: NixlConnector is closed")

        remote_agent = None
        local_reg_descs = None
        local_dlist = None
        remote_dlist = None
        xfer_handle = None
        try:
            if not isinstance(metadata, dict) or metadata.get("schema_version") != _SCHEMA_VERSION:
                logger.error("NixlConnector get has invalid metadata for %s", get_key)
                return None

            tensor_specs = metadata.get("tensor_specs")
            regions = metadata.get("regions")
            if not isinstance(tensor_specs, list) or not isinstance(regions, list):
                raise RuntimeError(f"Invalid NIXL metadata for {get_key}: missing tensor_specs/regions")

            # Metadata may be serialized (e.g. msgpack over ZMQ) between ``put``
            # and ``get``, which turns the region descriptor tuples into lists.
            # Restore tuples and strip to the 3-tuple ``(addr, len, dev)`` form
            # that NIXL's ``get_xfer_descs`` requires; the 4th reg-desc metadata
            # field is only valid for ``get_reg_descs`` (registration).
            remote_xfer_regions = [tuple(region)[:3] for region in regions]

            local_tensors = [self._allocate_tensor_from_spec(spec, metadata.get("kind")) for spec in tensor_specs]
            local_memory_type = self._resolve_memory_type(local_tensors[0])
            local_regions = [self._tensor_region(tensor) for tensor in local_tensors]
            local_reg_descs = self._agent.get_reg_descs(local_regions, local_memory_type)
            self._agent.register_memory(local_reg_descs, backends=self._backends)

            remote_agent = self._agent.add_remote_agent(metadata["agent_metadata"])
            self._remote_agents.append(remote_agent)
            remote_descs = self._agent.get_xfer_descs(remote_xfer_regions, metadata.get("memory_type", "DRAM"))
            local_xfer_regions = [tuple(region)[:3] for region in local_regions]
            local_descs = self._agent.get_xfer_descs(local_xfer_regions, local_memory_type)
            remote_dlist = self._agent.prep_xfer_dlist(remote_agent, remote_descs)
            local_dlist = self._agent.prep_xfer_dlist(_INIT_AGENT, local_descs)

            desc_ids = list(range(len(local_tensors)))
            xfer_handle = self._agent.make_prepped_xfer(
                "READ",
                local_dlist,
                desc_ids,
                remote_dlist,
                desc_ids,
            )
            self._agent.transfer(xfer_handle)
            self._wait_for_transfer(xfer_handle, get_key)
            self._agent.release_xfer_handle(xfer_handle)
            xfer_handle = None

            size = int(metadata.get("size", sum(spec.get("size", 0) for spec in tensor_specs)))
            if metadata.get("kind") == _KIND_OBJECT:
                raw = local_tensors[0].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                payload = self.deserialize_obj(raw)
            elif metadata.get("kind") == _KIND_STRUCTURED:
                raw = local_tensors[0].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                skeleton = self.deserialize_obj(raw)
                payload = self._restore_tensor_leaves(skeleton, local_tensors[1:])
            else:
                payload = local_tensors[0] if len(local_tensors) == 1 else local_tensors
            self._metrics["gets"] += 1
            self._metrics["bytes_transferred"] += size
            logger.debug("NixlConnector get %s->%s key=%s size=%d", from_stage, to_stage, get_key, size)
            return payload, size
        except Exception:
            self._metrics["errors"] += 1
            logger.error("NixlConnector get failed for %s", get_key, exc_info=True)
            return None
        finally:
            if xfer_handle is not None:
                self._safe_call(self._agent.release_xfer_handle, xfer_handle)
            if local_dlist is not None:
                self._safe_call(self._agent.release_dlist_handle, local_dlist)
            if remote_dlist is not None:
                self._safe_call(self._agent.release_dlist_handle, remote_dlist)
            if remote_agent is not None:
                self._safe_call(self._agent.remove_remote_agent, remote_agent)
                if remote_agent in self._remote_agents:
                    self._remote_agents.remove(remote_agent)
            if local_reg_descs is not None:
                self._safe_call(self._agent.deregister_memory, local_reg_descs)

    def cleanup(self, request_id: str) -> None:
        pending = self._pending.pop(request_id, None)
        if pending is None:
            return
        _, reg_descs_list, _ = pending
        for reg_descs in reg_descs_list:
            self._safe_call(self._agent.deregister_memory, reg_descs)
            if reg_descs in self._registered_descs:
                self._registered_descs.remove(reg_descs)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for request_id in list(self._pending):
            self.cleanup(request_id)
        for agent_name in list(self._remote_agents):
            self._safe_call(self._agent.remove_remote_agent, agent_name)
        self._remote_agents.clear()
        for reg_descs in list(self._registered_descs):
            self._safe_call(self._agent.deregister_memory, reg_descs)
        self._registered_descs.clear()

    def health(self) -> dict[str, Any]:
        return {
            "status": "unhealthy" if self._closed else "healthy",
            "pending_requests": len(self._pending),
            **self._metrics,
        }

    def _wait_for_transfer(self, handle: int, request_id: str) -> None:
        deadline = time.monotonic() + self._transfer_timeout_s
        while True:
            state = self._agent.check_xfer_state(handle)
            if state == "DONE":
                return
            if state != "PROC":
                raise RuntimeError(f"NIXL transfer for {request_id} failed with state={state}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"NIXL transfer for {request_id} timed out")
            time.sleep(self._poll_interval_s)

    def _cleanup_expired_pending(self) -> None:
        now = time.monotonic()
        for request_id, (_, _, deadline) in list(self._pending.items()):
            if now >= deadline:
                logger.warning(
                    "NixlConnector lease expired for request %s after %.0fs; its buffer is "
                    "being reclaimed while a consumer may still read it, which yields "
                    "corrupt data. Raise VLLM_OMNI_NIXL_LEASE_S above the maximum "
                    "producer-to-consumer queueing delay.",
                    request_id,
                    self._lease_seconds,
                )
                self.cleanup(request_id)

    def _resolve_memory_type(self, tensor: torch.Tensor) -> str:
        if self._default_memory_type is not None:
            return str(self._default_memory_type)
        if tensor.device.type == "cpu":
            return "DRAM"
        return "VRAM"

    @staticmethod
    def _tensor_region(tensor: torch.Tensor) -> tuple[int, int, int, str]:
        device_id = max(tensor.get_device(), 0) if tensor.device.type != "cpu" else 0
        return (tensor.data_ptr(), tensor.numel() * tensor.element_size(), device_id, "")

    @staticmethod
    def _is_tensor_payload(payload: Any) -> bool:
        return isinstance(payload, torch.Tensor) or (
            isinstance(payload, (list, tuple))
            and bool(payload)
            and all(isinstance(item, torch.Tensor) for item in payload)
        )

    @classmethod
    def _contains_tensor(cls, payload: Any) -> bool:
        if isinstance(payload, torch.Tensor):
            return True
        if isinstance(payload, dict):
            return any(cls._contains_tensor(value) for value in payload.values())
        if isinstance(payload, (list, tuple)):
            return any(cls._contains_tensor(value) for value in payload)
        return False

    @classmethod
    def _extract_tensor_leaves(cls, payload: Any) -> tuple[Any, list[torch.Tensor]]:
        tensors: list[torch.Tensor] = []

        def visit(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                index = len(tensors)
                tensors.append(value)
                return {_TENSOR_MARKER: index}
            if isinstance(value, dict):
                return {key: visit(item) for key, item in value.items()}
            if isinstance(value, list):
                return [visit(item) for item in value]
            if isinstance(value, tuple):
                return {_TUPLE_MARKER: [visit(item) for item in value]}
            return value

        return visit(payload), tensors

    @classmethod
    def _restore_tensor_leaves(cls, skeleton: Any, tensors: list[torch.Tensor]) -> Any:
        if isinstance(skeleton, dict):
            if set(skeleton) == {_TENSOR_MARKER}:
                return tensors[int(skeleton[_TENSOR_MARKER])]
            if set(skeleton) == {_TUPLE_MARKER}:
                return tuple(cls._restore_tensor_leaves(item, tensors) for item in skeleton[_TUPLE_MARKER])
            return {key: cls._restore_tensor_leaves(value, tensors) for key, value in skeleton.items()}
        if isinstance(skeleton, list):
            return [cls._restore_tensor_leaves(item, tensors) for item in skeleton]
        return skeleton

    @staticmethod
    def _normalize_tensor_payload(payload: Any) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
        tensors = [payload] if isinstance(payload, torch.Tensor) else list(payload)
        normalized: list[torch.Tensor] = []
        specs: list[dict[str, Any]] = []
        for tensor in tensors:
            contiguous = tensor.detach().contiguous()
            normalized.append(contiguous)
            specs.append(
                {
                    "shape": list(contiguous.shape),
                    "dtype": str(contiguous.dtype),
                    "device": str(contiguous.device),
                    "size": contiguous.numel() * contiguous.element_size(),
                }
            )
        return normalized, specs

    def _allocate_tensor_from_spec(self, spec: dict[str, Any], kind: str | None) -> torch.Tensor:
        shape = spec.get("shape")
        if not isinstance(shape, list):
            raise RuntimeError(f"Invalid NIXL tensor shape: {shape!r}")
        dtype_name = str(spec.get("dtype", "")).removeprefix("torch.")
        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            raise RuntimeError(f"Unsupported NIXL tensor dtype: {spec.get('dtype')!r}")
        device = (
            torch.device("cpu")
            if kind == _KIND_OBJECT
            else self._receive_device or self._parse_device(spec.get("device")) or torch.device("cpu")
        )
        return torch.empty(tuple(int(dim) for dim in shape), dtype=dtype, device=device)

    @staticmethod
    def _parse_device(device_like: Any) -> torch.device | None:
        if device_like is None:
            return None
        try:
            return torch.device(device_like)
        except Exception as exc:
            raise RuntimeError(f"Invalid NIXL receive device: {device_like!r}") from exc

    @staticmethod
    def _safe_call(func: Any, *args: Any) -> None:
        try:
            func(*args)
        except Exception:
            logger.debug("Ignoring NIXL cleanup failure", exc_info=True)
