import torch
import torch.nn as nn
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.minicpmo import (
    MiniCPMO4_5,
    MiniCPMODummyInputsBuilder,
    MiniCPMOMultiModalProcessor,
    MiniCPMOProcessingInfo,
)
from vllm.model_executor.models.minicpmv import Resampler4_5
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype


class MiniCPMO4_5Resampler(Resampler4_5):
    """MiniCPM-o 4.5 resampler with device-aware positional caches."""

    def _cache_device(self, device: torch.types.Device | None) -> torch.types.Device:
        if device is not None:
            return device
        query = getattr(self, "query", None)
        if isinstance(query, torch.Tensor) and not query.is_meta:
            return query.device
        return current_platform.device_type

    def _set_2d_pos_cache(
        self,
        max_size: tuple[int, int],
        device: torch.types.Device | None = None,
    ) -> None:
        super()._set_2d_pos_cache(max_size, device=self._cache_device(device))

    def _set_temporal_pos_cache(
        self,
        max_temporal_size: int,
        device: torch.types.Device | None = None,
    ) -> None:
        super()._set_temporal_pos_cache(
            max_temporal_size,
            device=self._cache_device(device),
        )


@MULTIMODAL_REGISTRY.register_processor(
    MiniCPMOMultiModalProcessor,
    info=MiniCPMOProcessingInfo,
    dummy_inputs=MiniCPMODummyInputsBuilder,
)
class MiniCPMO4_5ThinkerForConditionalGeneration(MiniCPMO4_5):
    """Thinker-only MiniCPM-o 4.5 model."""

    def init_resampler(
        self,
        embed_dim: int,
        vision_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> nn.Module:
        with set_default_torch_dtype(torch.float16):
            resampler = MiniCPMO4_5Resampler(
                num_queries=self.config.query_num,
                embed_dim=embed_dim,
                num_heads=embed_dim // 128,
                kv_dim=vision_dim,
                quant_config=quant_config,
                prefix=prefix,
            )

        target_device = current_platform.device_type
        target_dtype = torch.get_default_dtype()
        if any(p.is_meta for p in resampler.parameters()):
            return resampler.to_empty(device=target_device).to(dtype=target_dtype)
        return resampler.to(device=target_device, dtype=target_dtype)
