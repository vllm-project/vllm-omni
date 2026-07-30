# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm_omni.experimental.fullduplex.base.data_plane import (
    EncodeAudio,
)
from vllm_omni.experimental.fullduplex.base.serving_adapter import (
    BaseServingRuntimeAdapter,
)
from vllm_omni.experimental.fullduplex.openai.protocol import DuplexCapabilities
from vllm_omni.experimental.fullduplex.qwen3omni.adapter import (
    Qwen3OmniNativeDuplexServingAdapter,
)
from vllm_omni.experimental.fullduplex.qwen3omni.data_plane import (
    Qwen3OmniDataPlaneSession,
)


class Qwen3OmniServingRuntimeAdapter(BaseServingRuntimeAdapter):
    @property
    def adapter_id(self) -> str:
        return "qwen3omni"

    @property
    def private_runtime_config_keys(self) -> frozenset[str]:
        return Qwen3OmniNativeDuplexServingAdapter.PRIVATE_RUNTIME_CONFIG_KEYS

    def _create_data_plane(self, encode_audio: EncodeAudio) -> Qwen3OmniDataPlaneSession:
        return Qwen3OmniDataPlaneSession(encode_audio)

    @staticmethod
    def is_enabled(config: object) -> bool:
        return Qwen3OmniNativeDuplexServingAdapter.is_enabled(config)  # type: ignore[arg-type]

    @staticmethod
    def capabilities(*, max_sessions: int) -> DuplexCapabilities:
        return DuplexCapabilities.qwen3omni_native(max_sessions=max_sessions)

    @staticmethod
    def validate_client_extra_body(extra_body: object) -> None:
        Qwen3OmniNativeDuplexServingAdapter.validate_client_extra_body(extra_body)

    @staticmethod
    async def prepare_runtime_config(config: object, *, model_config: Any) -> dict[str, object]:
        return await Qwen3OmniNativeDuplexServingAdapter.prepare_runtime_config(
            config,  # type: ignore[arg-type]
            model_config=model_config,
        )

    @staticmethod
    def runtime_config_for_update(
        config: object,
        current: Mapping[str, object],
    ) -> dict[str, object]:
        return Qwen3OmniNativeDuplexServingAdapter.runtime_config_for_update(
            config,  # type: ignore[arg-type]
            dict(current),
        )
