# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified OmniConnector and KV cache transfer management."""

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch
from vllm.logger import init_logger

from .factory import OmniConnectorFactory
from .utils.config import ConnectorSpec

logger = init_logger(__name__)


@dataclass
class OmniKVCacheConfig:
    """Configuration for OmniKVTransferManager."""

    connector_config: dict[str, Any] | None = None
    from_stage: str | None = None
    to_stage: str | None = None
    stage_id: str | int | None = None
    engine_input_source: list[str | int] | None = None
    need_recv_cache: bool = False
    need_send_cache: bool = False
    recv_timeout: float = 30.0


@dataclass
class KVCacheTransferData:
    """Container for KV cache transfer data."""

    request_id: str
    layer_blocks: dict[str, Any]
    block_ids: list[int]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class OmniKVTransferManager:
    """Unified management for OmniConnector and KV cache transfer.

    This class encapsulates all KV cache related operations:
    - Connector initialization and lazy creation
    - KV cache extraction from GPU blocks
    - KV cache transfer with retry logic
    - KV cache receiving with timeout
    """

    def __init__(self, config: OmniKVCacheConfig):
        self.config = config
        self._connector = None

        # Pre-calculate send stages (from_stage, to_stage)
        self.send_stages = (
            (str(config.from_stage), str(config.to_stage)) if config.from_stage and config.to_stage else (None, None)
        )

        # Pre-calculate receive stages (from_stage, to_stage)
        recv_from = config.from_stage
        if config.engine_input_source:
            recv_from = config.engine_input_source[0]
        elif isinstance(config.stage_id, int):
            recv_from = config.stage_id - 1

        self.recv_stages = (
            (str(recv_from), str(config.stage_id))
            if recv_from is not None and config.stage_id is not None
            else (None, None)
        )

    @classmethod
    def _create(cls, cfg: dict | None) -> "OmniKVTransferManager":
        """Create manager from raw config dict."""
        if not cfg or not isinstance(cfg, dict):
            return cls(OmniKVCacheConfig())
        return cls(
            OmniKVCacheConfig(
                connector_config=cfg.get("connector_config"),
                from_stage=cfg.get("omni_from_stage"),
                to_stage=cfg.get("omni_to_stage"),
                stage_id=cfg.get("stage_id"),
                engine_input_source=cfg.get("engine_input_source", []),
                need_recv_cache=cfg.get("need_recv_cache", False),
                need_send_cache=cfg.get("need_send_cache", False),
                recv_timeout=cfg.get("recv_timeout", 30.0),
            )
        )

    @classmethod
    def from_model_config(cls, config: Any) -> "OmniKVTransferManager":
        """Create from model config (for AR model runner)."""
        return cls._create(getattr(config, "omni_kv_config", None))

    @classmethod
    def from_od_config(cls, config: Any) -> "OmniKVTransferManager":
        """Create from OmniDiffusion config (for diffusion runner)."""
        return cls._create(getattr(config, "omni_kv_config", None))

    @classmethod
    def from_vllm_config(cls, vllm_config: Any, model_config: Any) -> "OmniKVTransferManager":
        """Create from vllm config with fallback to kv_transfer_config."""
        # Primary: omni_kv_config from model_config
        omni_kv = getattr(model_config, "omni_kv_config", None)
        if isinstance(omni_kv, dict):
            return cls._create(omni_kv)

        # Fallback: check kv_transfer_config
        kv_cfg = getattr(vllm_config, "kv_transfer_config", None)
        if kv_cfg:
            direct = getattr(kv_cfg, "omni_connector_config", None)
            if isinstance(direct, dict) and direct:
                return cls._create({"connector_config": direct})
            extra = getattr(kv_cfg, "kv_connector_extra_config", None)
            if isinstance(extra, dict):
                omni = extra.get("omni_connector_config")
                if isinstance(omni, dict) and omni:
                    return cls._create({"connector_config": omni})

        return cls(OmniKVCacheConfig())

    @property
    def connector(self):
        """Lazy initialization of connector."""
        # If a previous initialization attempt failed, don't retry on every access.
        if self._connector is False:
            return None

        if self._connector is None:
            cfg = self.config.connector_config
            if cfg and (c_type := cfg.get("type")):
                try:
                    logger.info(f"Initializing OmniConnector with config: {cfg}")
                    c_extra = {k: v for k, v in cfg.items() if k != "type"}
                    self._connector = OmniConnectorFactory.create_connector(ConnectorSpec(name=c_type, extra=c_extra))
                except Exception as e:
                    logger.error(f"Failed to initialize OmniConnector: {e}")
                    import traceback

                    traceback.print_exc()
                    # Cache failure sentinel to avoid repeated initialization attempts in hot paths.
                    self._connector = False

        return self._connector if self._connector else None

    def get_connector(self):
        """Get connector (compatibility wrapper for existing code)."""
        return self.connector

    def handle_finished_requests_kv_transfer(
        self,
        finished_reqs: dict[str, dict[str, Any]],
        kv_caches: list[torch.Tensor],
        block_size: int,
        cache_dtype: str,
        request_id_resolver: Callable[[str], str] | None = None,
    ) -> list[str]:
        """Handle KV cache transfer for finished requests.

        This method extracts KV cache from GPU blocks and transfers them
        to the downstream stage via the connector.

        Args:
            finished_reqs: Dict mapping request_id to {block_ids, seq_len}
            kv_caches: List of KV cache tensors per layer
            block_size: Size of each cache block
            cache_dtype: Data type of the cache
            request_id_resolver: Optional function to resolve global request ID

        Returns:
            List of request IDs that were processed
        """
        if not finished_reqs:
            return []

        if not self.config.need_send_cache:
            return list(finished_reqs.keys())

        if not self.connector:
            logger.warning("No connector available, skipping KV transfer but freeing resources")
            return list(finished_reqs.keys())

        logger.debug(f"Processing KV transfer for {len(finished_reqs)} requests")

        extracted_ids = []
        for req_id, data in finished_reqs.items():
            try:
                seq_len = data.get("seq_len", 0)
                block_ids = data.get("block_ids", [])
                if not block_ids:
                    logger.warning(f"Request {req_id} has no block IDs, skipping")
                    continue

                # Extract KV cache from GPU blocks -> CPU tensors
                kv_data = self._extract_kv_cache(req_id, block_ids, seq_len, kv_caches, block_size, cache_dtype)
                if kv_data:
                    # Resolve global request ID if available
                    transfer_req_id = request_id_resolver(req_id) if request_id_resolver else req_id

                    # Transfer to downstream stage via connector
                    self._transfer_kv_cache(kv_data, transfer_req_id)

            except Exception as e:
                logger.error(f"Failed KV transfer for {req_id}: {e}")
            finally:
                extracted_ids.append(req_id)

        return extracted_ids

    def _extract_kv_cache(
        self,
        req_id: str,
        block_ids: list[int],
        seq_len: int,
        kv_caches: list[torch.Tensor],
        block_size: int,
        cache_dtype: str,
    ) -> KVCacheTransferData | None:
        """Extract KV cache from GPU blocks for a single request.

        Args:
            req_id: Request identifier
            block_ids: List of block IDs to extract
            seq_len: Sequence length
            kv_caches: List of KV cache tensors per layer
            block_size: Size of each cache block
            cache_dtype: Data type of the cache

        Returns:
            KVCacheTransferData if extraction successful, None otherwise
        """
        num_layers = len(kv_caches)
        key_cache: list[torch.Tensor | None] = [None] * num_layers
        value_cache: list[torch.Tensor | None] = [None] * num_layers

        for layer_idx, kv_tensor in enumerate(kv_caches):
            # Validate block IDs - shape: [2, num_blocks, block_size, n_heads, head_dim]
            max_block = kv_tensor.shape[1] - 1
            valid_ids = [bid for bid in block_ids if 0 <= bid <= max_block]
            if not valid_ids:
                continue

            # Extract and reshape: [2, n_blocks, block_size, n_heads, head_dim]
            # -> [2, seq_len, n_heads, head_dim]
            selected = kv_tensor[:, valid_ids]  # [2, n_valid, block_size, n_heads, head_dim]
            n_kv, n_blks, blk_sz, n_heads, d_head = selected.shape
            flat = selected.reshape(n_kv, n_blks * blk_sz, n_heads, d_head)
            if seq_len < flat.shape[1]:
                flat = flat[:, :seq_len]

            # Move to CPU
            flat_cpu = flat.detach().cpu().contiguous()
            key_cache[layer_idx] = flat_cpu[0]
            value_cache[layer_idx] = flat_cpu[1]

        if not any(k is not None for k in key_cache):
            return None

        return KVCacheTransferData(
            request_id=req_id,
            layer_blocks={"key_cache": key_cache, "value_cache": value_cache},
            block_ids=block_ids,
            metadata={
                "block_size": block_size,
                "num_layers": num_layers,
                "dtype": str(cache_dtype),
                "seq_len": seq_len,
            },
        )

    def _transfer_kv_cache(self, kv_data: KVCacheTransferData, transfer_req_id: str) -> None:
        """Transfer KV cache data to downstream stage via OmniConnector.

        Args:
            kv_data: The extracted KV cache data
            transfer_req_id: The request ID to use for transfer
        """
        from_stage, to_stage = self.send_stages
        if not from_stage or not to_stage:
            raise ValueError("Transfer stages (omni_from_stage, omni_to_stage) not configured")

        # Prepare data and transfer with retry
        data_dict = kv_data.to_dict()
        data_dict["request_id"] = transfer_req_id

        success, size, _ = self._transfer_with_retry(from_stage, to_stage, f"kv_cache_{transfer_req_id}", data_dict)

        if success:
            logger.info(f"KV transfer OK: {transfer_req_id}, {size} bytes")
        else:
            logger.error(f"KV transfer FAILED: {transfer_req_id}")

    def _transfer_with_retry(
        self,
        from_stage: str,
        to_stage: str,
        request_id: str,
        data: dict[str, Any],
        max_retries: int = 3,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        """Transfer data with retry and exponential backoff.

        Args:
            from_stage: Source stage identifier
            to_stage: Target stage identifier
            request_id: Request identifier for the key
            data: Data to transfer
            max_retries: Maximum number of retry attempts

        Returns:
            Tuple of (success, size, metadata)
        """
        for attempt in range(max_retries):
            try:
                # Build the full key for connector
                full_request_id = f"omni_{from_stage}_to_{to_stage}_{request_id}"
                success, size, metadata = self.connector.put(
                    from_stage=from_stage, to_stage=to_stage, put_key=full_request_id, data=data
                )
                if success:
                    return success, size, metadata
                logger.warning(f"Transfer attempt {attempt + 1} failed for {request_id}")
            except Exception as e:
                logger.warning(f"Transfer attempt {attempt + 1} exception: {e}")

            if attempt < max_retries - 1:
                time.sleep(0.1 * (2**attempt))

        return False, 0, None

    @torch.inference_mode()
    def receive_kv_cache_for_request(
        self,
        request_id: str,
        target_device: torch.device | None = None,
    ) -> tuple[dict[str, Any] | None, int]:
        """Receive KV cache for a specific request.

        This implements the receiving logic from gpu_diffusion_model_runner.py.

        Args:
            request_id: The request ID to receive KV cache for
            target_device: Optional device to move tensors to

        Returns:
            Tuple of (data dict, size) if successful, (None, 0) otherwise
        """
        if not self.connector:
            logger.warning("No connector available for receiving KV cache")
            return None, 0

        from_stage, to_stage = self.recv_stages
        if not from_stage or not to_stage:
            logger.warning("Receive stages not configured")
            return None, 0

        # Check if we should receive KV cache based on config
        if not self.config.need_recv_cache:
            logger.info(f"Skip receiving KV cache for {request_id} (need_recv_cache=False)")
            return None, 0

        timeout = self.config.recv_timeout
        start_time = time.time()

        logger.info(f"Wait for KV cache for request {request_id} from stage {from_stage} to {to_stage}...")

        try:
            while True:
                # Build the full key for connector
                full_request_id = f"omni_{from_stage}_to_{to_stage}_kv_cache_{request_id}"
                result = self.connector.get(
                    from_stage=from_stage,
                    to_stage=to_stage,
                    get_key=full_request_id,
                )
                if result:
                    data, size = result
                    logger.info(f"Successfully received KV cache for {request_id}, {size} bytes")

                    # Move tensors to target device if specified
                    if target_device is not None and isinstance(data, dict) and "layer_blocks" in data:
                        layer_blocks = data["layer_blocks"]
                        for cache_list in [
                            layer_blocks.get("key_cache", []),
                            layer_blocks.get("value_cache", []),
                        ]:
                            for i, tensor in enumerate(cache_list):
                                if isinstance(tensor, torch.Tensor) and tensor.device != target_device:
                                    cache_list[i] = tensor.to(target_device).contiguous()

                    return data, size

                if time.time() - start_time > timeout:
                    logger.error(f"Timeout waiting for KV cache for request {request_id} after {timeout}s")
                    return None, 0

                time.sleep(0.5)

        except Exception as e:
            logger.error(f"Error receiving KV cache for {request_id}: {e}")
            import traceback

            traceback.print_exc()
            return None, 0

    def apply_kv_cache_to_request(self, req: Any, data: dict[str, Any]) -> None:
        """Apply received KV cache data to a request object.

        Args:
            req: The request object to apply KV cache to
            data: The received KV cache data dictionary
        """
        if isinstance(data, dict) and "layer_blocks" in data:
            layer_blocks = data["layer_blocks"]
            from types import SimpleNamespace

            kv_obj = SimpleNamespace(**layer_blocks)
            req.past_key_values = kv_obj

            # [Omni] Also attach to sampling_params for BagelPipeline compatibility
            # BagelPipeline checks req.sampling_params.past_key_values
            if hasattr(req, "sampling_params") and req.sampling_params is not None:
                req.sampling_params.past_key_values = kv_obj

        if "metadata" in data:
            req.kv_metadata = data["metadata"]

    # Legacy compatibility method
    def receive_kv_cache(self, req: Any, target_device: torch.device | None = None) -> bool:
        """Receive KV cache and populate request object (legacy interface).

        Args:
            req: Request object with request_id attribute
            target_device: Optional device to move tensors to

        Returns:
            True if successful, False otherwise
        """
        request_id = getattr(req, "request_id", None)
        if not request_id and hasattr(req, "request_ids") and req.request_ids:
            # Adaptation for new OmniDiffusionRequest which has list of prompts/ids
            request_id = req.request_ids[0]

        if not request_id:
            logger.warning("Request has no ID, cannot receive KV cache")
            return False

        data, size = self.receive_kv_cache_for_request(request_id, target_device)
        if data:
            self.apply_kv_cache_to_request(req, data)
            return True
        return False
