"""
OmniScheduler mixin for unified chunk management.

Provides common chunk management initialization for both
OmniARScheduler and OmniGenerationScheduler.
"""

from vllm.logger import init_logger

from vllm_omni.core.chunk_manager import create_chunk_manager
from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec

logger = init_logger(__name__)


class OmniSchedulerMixin:
    """Mixin providing unified chunk management for Omni schedulers.

    This mixin provides `_init_chunk_management()` method that creates
    the chunk manager using the new factory when async_chunk_stream is enabled.
    """

    def _init_chunk_management(self, stage_strategy_type: str = "ar") -> None:
        """Initialize chunk management components.

        Only initializes if async_chunk_stream is enabled in scheduler_config.

        Args:
            stage_strategy_type: "ar" or "generation"

        Must be called after super().__init__() in scheduler __init__.
        Requires self.vllm_config to be set.
        """
        scheduler_config = self.vllm_config.scheduler_config
        model_config = self.vllm_config.model_config

        self.stage_id = getattr(model_config, "stage_id", None)
        self.omni_connector = None
        self.chunk_manager = None

        # Only initialize if async_chunk_stream is enabled
        if not getattr(scheduler_config, "async_chunk_stream", False):
            return

        # Create connector
        connector_specs = ConnectorSpec(
            name=scheduler_config.stage_connector_name, extra=scheduler_config.stage_connector_extra
        )
        self.omni_connector = OmniConnectorFactory.create_connector(connector_specs)

        # Get processor path from config (if specified)
        chunk_processor = getattr(model_config, "chunk_processor", None)

        # Create unified chunk manager
        self.chunk_manager = create_chunk_manager(
            connector=self.omni_connector,
            stage_id=self.stage_id,
            stage_strategy_type=stage_strategy_type,
            chunk_processor=chunk_processor,
        )

        logger.info(f"Initialized ChunkManager for stage {self.stage_id} (type={stage_strategy_type})")

    def cleanup_request_chunk_state(self, request_id: str) -> None:
        """Cleanup chunk state for a finished request."""
        if self.chunk_manager is not None:
            self.chunk_manager.cleanup_request(request_id)
