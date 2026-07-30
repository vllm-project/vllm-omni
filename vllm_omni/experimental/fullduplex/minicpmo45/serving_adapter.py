from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm_omni.experimental.fullduplex.base.data_plane import (
    DataPlaneContext,
    EncodeAudio,
)
from vllm_omni.experimental.fullduplex.base.serving_adapter import (
    BaseServingRuntimeAdapter,
)
from vllm_omni.experimental.fullduplex.minicpmo45.adapter import (
    MiniCPMO45NativeDuplexServingAdapter,
)
from vllm_omni.experimental.fullduplex.minicpmo45.data_plane import (
    MiniCPMO45DataPlaneSession,
)
from vllm_omni.experimental.fullduplex.openai.protocol import DuplexCapabilities

MiniCPMO45DataPlaneContext = DataPlaneContext


class MiniCPMO45ServingRuntimeAdapter(BaseServingRuntimeAdapter):
    @property
    def adapter_id(self) -> str:
        return "minicpmo45"

    @property
    def private_runtime_config_keys(self) -> frozenset[str]:
        return MiniCPMO45NativeDuplexServingAdapter.PRIVATE_RUNTIME_CONFIG_KEYS

    def _create_data_plane(self, encode_audio: EncodeAudio) -> MiniCPMO45DataPlaneSession:
        return MiniCPMO45DataPlaneSession(encode_audio)

    @staticmethod
    def is_enabled(config: object) -> bool:
        return MiniCPMO45NativeDuplexServingAdapter.is_enabled(config)  # type: ignore[arg-type]

    @staticmethod
    def capabilities(*, max_sessions: int) -> DuplexCapabilities:
        return DuplexCapabilities.minicpmo45_native(max_sessions=max_sessions)

    @staticmethod
    def validate_client_extra_body(extra_body: object) -> None:
        MiniCPMO45NativeDuplexServingAdapter.validate_client_extra_body(extra_body)

    @staticmethod
    async def prepare_runtime_config(config: object, *, model_config: Any) -> dict[str, object]:
        return await MiniCPMO45NativeDuplexServingAdapter.prepare_runtime_config(
            config,  # type: ignore[arg-type]
            model_config=model_config,
        )

    @staticmethod
    def runtime_config_for_update(
        config: object,
        current: Mapping[str, object],
    ) -> dict[str, object]:
        return MiniCPMO45NativeDuplexServingAdapter.runtime_config_for_update(
            config,  # type: ignore[arg-type]
            dict(current),
        )
