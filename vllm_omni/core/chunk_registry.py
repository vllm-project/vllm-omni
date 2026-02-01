"""Chunk processor registry with auto-discovery."""

import importlib
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# Auto-discovery registry: (model_arch, model_stage) -> processor path
_PROCESSOR_REGISTRY: dict[tuple[str, str], str] = {
    # Qwen2.5-Omni
    (
        "Qwen2_5OmniForConditionalGeneration",
        "thinker",
    ): "vllm_omni.model_executor.chunk_processors.qwen25_chunk_processors.Qwen25ThinkerChunkProcessor",
    (
        "Qwen2_5OmniForConditionalGeneration",
        "talker",
    ): "vllm_omni.model_executor.chunk_processors.qwen25_chunk_processors.Qwen25TalkerChunkProcessor",
    (
        "Qwen2_5OmniForConditionalGeneration",
        "code2wav",
    ): "vllm_omni.model_executor.chunk_processors.qwen25_chunk_processors.Qwen25Code2WavChunkProcessor",
    # Qwen3-Omni-MoE
    (
        "Qwen3OmniMoeForConditionalGeneration",
        "thinker",
    ): "vllm_omni.model_executor.chunk_processors.qwen3_chunk_processors.Qwen3ThinkerChunkProcessor",
    (
        "Qwen3OmniMoeForConditionalGeneration",
        "talker",
    ): "vllm_omni.model_executor.chunk_processors.qwen3_chunk_processors.Qwen3TalkerChunkProcessor",
    (
        "Qwen3OmniMoeForConditionalGeneration",
        "code2wav",
    ): "vllm_omni.model_executor.chunk_processors.qwen3_chunk_processors.Qwen3Code2WavChunkProcessor",
}


class ChunkProcessorRegistry:
    """Factory for chunk processor discovery and instantiation."""

    _instances: dict[str, Any] = {}

    @classmethod
    def register(cls, model_arch: str, model_stage: str, processor_path: str) -> None:
        """Register a new processor."""
        _PROCESSOR_REGISTRY[(model_arch, model_stage)] = processor_path
        logger.info(f"Registered: {model_arch}/{model_stage} -> {processor_path}")

    @classmethod
    def get_processor(
        cls,
        model_arch: str | None = None,
        model_stage: str | None = None,
        chunk_processor: str | None = None,
        config: dict | None = None,
    ) -> Any:
        """Get processor instance. Priority: explicit > registry lookup."""
        processor_path = chunk_processor
        if not processor_path and model_arch and model_stage:
            processor_path = _PROCESSOR_REGISTRY.get((model_arch, model_stage))

        if not processor_path:
            raise ValueError(
                f"No processor for {model_arch}/{model_stage}. "
                f"Specify 'chunk_processor' in config or register via "
                f"ChunkProcessorRegistry.register()"
            )

        # Check cache
        cache_key = f"{processor_path}:{config}"
        if cache_key in cls._instances:
            return cls._instances[cache_key]

        # Dynamic import
        module_path, class_name = processor_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        processor_class = getattr(module, class_name)

        # Instantiate with config if provided
        instance = processor_class(**(config or {}))
        cls._instances[cache_key] = instance
        return instance

    @classmethod
    def list_registered(cls) -> list[tuple[str, str, str]]:
        """List all registered processors."""
        return [(a, s, p) for (a, s), p in _PROCESSOR_REGISTRY.items()]

    @classmethod
    def clear_cache(cls) -> None:
        """Clear instance cache (mainly for testing)."""
        cls._instances.clear()
