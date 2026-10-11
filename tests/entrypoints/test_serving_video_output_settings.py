# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest
from omegaconf import OmegaConf

from vllm_omni.config.omni_config import VllmOmniConfig, VllmOmniDiffusionStageConfig
from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig
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


@pytest.mark.parametrize("typed_stage", [False, True], ids=["legacy", "typed"])
@pytest.mark.parametrize("transport_mode", ["url", "shared_memory"])
@pytest.mark.parametrize("inline", [False, True], ids=["subprocess", "inline"])
def test_out_of_process_engine_view_preserves_final_stage_transport(
    monkeypatch: pytest.MonkeyPatch,
    typed_stage: bool,
    transport_mode: Literal["url", "shared_memory"],
    inline: bool,
) -> None:
    engine = AsyncOmniEngine.__new__(AsyncOmniEngine)
    engine.model = "unused"
    engine._diffusion_od_config_view = None
    transport = {
        "transport_mode": transport_mode,
        "output_format": "webm",
        "video_codec": "libvpx-vp9",
        "video_codec_options": {"crf": "0"},
        "shared_memory_ttl_seconds": 17,
    }
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
                "engine_args": {"video_output_transport": transport},
            },
        ]
    )
    if typed_stage:
        pipeline = PipelineConfig(
            model_type="generic_diffusion",
            stages=tuple(
                StagePipelineConfig(
                    stage_id=index,
                    model_stage="diffusion",
                    execution_type=StageExecutionType.DIFFUSION,
                    final_output=index == 1,
                    final_output_type="video",
                )
                for index in range(2)
            ),
        )
        engine.stage_configs = VllmOmniConfig.from_pipeline_config(
            pipeline,
            cli_overrides={
                "stage_0_video_output_transport": {"transport_mode": "bytes"},
                "stage_1_video_output_transport": transport,
            },
        ).stage_configs
        assert isinstance(engine.stage_configs[1], VllmOmniDiffusionStageConfig)
    monkeypatch.setattr(diffusion_data, "resolve_model_class_name", lambda model: "WanPipeline")
    monkeypatch.setattr(model_metadata, "get_diffusion_model_metadata", lambda model_class: _ModelMetadata())

    client = engine
    if inline:
        from vllm_omni.diffusion.inline_stage_diffusion_client import InlineStageDiffusionClient
        from vllm_omni.entrypoints.async_omni import AsyncOmni

        engine.stage_clients = []
        for index in range(2):
            stage_client = InlineStageDiffusionClient.__new__(InlineStageDiffusionClient)
            stage_client.stage_id = index
            stage_client.final_output = index == 1
            stage_client.od_config = diffusion_data.OmniDiffusionConfig(
                model=None,
                video_output_transport=VideoOutputTransportConfig.from_value(transport if index == 1 else None),
            )
            engine.stage_clients.append(stage_client)
        client = AsyncOmni.__new__(AsyncOmni)
        client.engine = engine

    settings = resolve_video_output_settings(client)

    assert settings.transport_mode == transport_mode
    assert settings.output_format == "webm"
    assert settings.codec == "libvpx-vp9"
    assert settings.codec_options == {"crf": "0"}
    assert settings.shared_memory_ttl_seconds == 17

    overridden = resolve_video_output_settings(
        client,
        {"output_format": "mp4", "video_codec": "libx265", "video_codec_options": {"crf": "18"}},
    )
    assert overridden.transport_mode == transport_mode
    assert overridden.output_format == "mp4"
    assert overridden.codec == "libx265"
    assert overridden.codec_options == {"crf": "18"}


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


@pytest.mark.parametrize(
    "overrides",
    [
        None,
        {
            "output_format": "webm",
            "video_codec": "libvpx-vp9",
            "video_codec_options": {"deadline": "realtime"},
        },
    ],
)
def test_streaming_forces_mp4_and_ignores_artifact_codec_policy(
    overrides: dict[str, object] | None,
) -> None:
    client = _GetterClient(
        _config(
            output_format="webm",
            video_codec="libvpx-vp9",
            video_codec_options={"deadline": "realtime"},
        )
    )

    settings = resolve_video_output_settings(
        client,
        overrides,
        low_latency=True,
        force_output_format="mp4",
    )

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
@pytest.mark.parametrize("force_output_format", [None, "mp4"])
def test_invalid_request_encoder_overrides_are_rejected(
    overrides: dict[str, object],
    force_output_format: str | None,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        resolve_video_output_settings(
            _GetterClient(_config()),
            overrides,
            force_output_format=force_output_format,
        )


def test_transport_and_ttl_are_resolved_from_deployment_config() -> None:
    settings = resolve_video_output_settings(
        _GetterClient(_config(transport_mode="shared_memory", shared_memory_ttl_seconds=17))
    )

    assert settings.transport_mode == "shared_memory"
    assert settings.shared_memory_ttl_seconds == 17
