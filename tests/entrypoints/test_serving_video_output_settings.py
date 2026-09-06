# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest
from omegaconf import OmegaConf

from vllm_omni.diffusion import data as diffusion_data
from vllm_omni.diffusion import model_metadata
from vllm_omni.diffusion.data import VideoOutputTransportConfig
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.entrypoints.openai.video_api_utils import resolve_video_output_settings

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _DiffusionConfig:
    video_output_transport: VideoOutputTransportConfig


@dataclass
class _GetterClient:
    config: _DiffusionConfig

    def get_diffusion_od_config(self) -> _DiffusionConfig:
        return self.config


@dataclass
class _AttributeClient:
    od_config: _DiffusionConfig


@dataclass
class _FailingGetterClient:
    od_config: _DiffusionConfig

    def get_diffusion_od_config(self) -> _DiffusionConfig:
        raise RuntimeError("bridge unavailable")


@dataclass
class _InvalidGetterClient:
    od_config: _DiffusionConfig

    def get_diffusion_od_config(self) -> _DiffusionConfig:
        raise ValueError("invalid diffusion config")


@dataclass
class _ModelMetadata:
    supports_multimodal_inputs: bool = True
    max_multimodal_image_inputs: int | None = None
    supports_mixed_reference_inputs: bool = False


def _config(
    *,
    transport_mode: Literal["bytes", "base64", "url", "shared_memory"] = "bytes",
    shared_memory_ttl_seconds: int = 300,
    output_format: Literal["mp4", "webm"] = "mp4",
    video_codec: str | None = None,
    video_codec_options: dict[str, str] | None = None,
) -> _DiffusionConfig:
    return _DiffusionConfig(
        VideoOutputTransportConfig(
            transport_mode=transport_mode,
            shared_memory_ttl_seconds=shared_memory_ttl_seconds,
            output_format=output_format,
            video_codec=video_codec,
            video_codec_options=video_codec_options or {},
        )
    )


@pytest.mark.parametrize("client_kind", ["getter", "attribute"])
def test_default_settings_preserve_the_existing_http_encoder(client_kind: str) -> None:
    config = _config()
    client = _GetterClient(config) if client_kind == "getter" else _AttributeClient(config)

    settings = resolve_video_output_settings(client)

    assert settings.codec == "h264"
    assert settings.codec_options == {"preset": "ultrafast", "threads": "0"}
    assert settings.output_format == "mp4"
    assert settings.media_type == "video/mp4"
    assert settings.transport_mode == "bytes"


def test_out_of_process_engine_view_preserves_final_stage_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = AsyncOmniEngine.__new__(AsyncOmniEngine)
    engine.model = "unused"
    engine._diffusion_od_config_view = None
    engine.stage_configs = OmegaConf.create(
        [
            {
                "stage_type": "diffusion",
                "final_output": False,
                "engine_args": {"video_output_transport": {"transport_mode": "bytes"}},
            },
            {
                "stage_type": "diffusion",
                "final_output": True,
                "engine_args": {
                    "video_output_transport": {
                        "transport_mode": "url",
                        "output_format": "webm",
                        "video_codec_options": {"deadline": "realtime"},
                    }
                },
            },
        ]
    )
    monkeypatch.setattr(diffusion_data, "resolve_model_class_name", lambda model: "WanPipeline")
    monkeypatch.setattr(model_metadata, "get_diffusion_model_metadata", lambda model_class: _ModelMetadata())

    settings = resolve_video_output_settings(engine)

    assert settings.transport_mode == "url"
    assert settings.output_format == "webm"
    assert settings.codec == "libvpx-vp9"
    assert settings.codec_options == {"deadline": "realtime"}


def test_bridge_runtime_failure_is_not_silently_replaced_by_attribute_defaults() -> None:
    with pytest.raises(RuntimeError, match="bridge unavailable"):
        resolve_video_output_settings(_FailingGetterClient(_config(output_format="webm")))


def test_invalid_bridge_config_is_not_silently_replaced_by_defaults() -> None:
    with pytest.raises(ValueError, match="invalid diffusion config"):
        resolve_video_output_settings(_InvalidGetterClient(_config()))


def test_request_encoder_overrides_take_precedence() -> None:
    client = _GetterClient(
        _config(
            video_codec="h264",
            video_codec_options={"crf": "30"},
        )
    )

    settings = resolve_video_output_settings(
        client,
        {
            "video_codec": "libx264",
            "video_codec_options": {"preset": "medium"},
        },
    )

    assert settings.codec == "libx264"
    assert settings.codec_options == {"preset": "medium"}


def test_webm_uses_format_derived_defaults() -> None:
    settings = resolve_video_output_settings(_GetterClient(_config(output_format="webm")))

    assert settings.codec == "libvpx-vp9"
    assert settings.output_format == "webm"
    assert settings.media_type == "video/webm"


def test_streaming_forces_mp4_and_low_latency_options() -> None:
    client = _GetterClient(_config(output_format="webm"))

    settings = resolve_video_output_settings(client, low_latency=True, force_output_format="mp4")

    assert settings.output_format == "mp4"
    assert settings.codec == "h264"
    assert settings.codec_options == {
        "preset": "ultrafast",
        "threads": "0",
        "tune": "zerolatency",
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"video_codec": ""},
        {"video_codec_options": {"crf": 18}},
        {"output_format": ["mp4"]},
        {"output_format": ""},
    ],
)
def test_invalid_request_encoder_overrides_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        resolve_video_output_settings(_GetterClient(_config()), overrides)


def test_transport_and_ttl_are_resolved_from_deployment_config() -> None:
    settings = resolve_video_output_settings(
        _GetterClient(_config(transport_mode="shared_memory", shared_memory_ttl_seconds=17))
    )

    assert settings.transport_mode == "shared_memory"
    assert settings.shared_memory_ttl_seconds == 17
