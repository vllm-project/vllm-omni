# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.qwen3_omni.first_frame_decoder import talker_first_audio_enabled

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("env", "extra", "async_chunk", "v2", "expected"),
    [
        (None, {}, True, True, False),
        ("1", {}, True, True, False),
        (None, {"talker_first_audio": True, "codec_chunk_ramp": [1, 2]}, True, True, True),
        ("0", {"talker_first_audio": True, "codec_chunk_ramp": [1, 2]}, True, True, False),
        (None, {"talker_first_audio": True, "codec_chunk_ramp": [4, 8]}, True, True, False),
        (None, {"talker_first_audio": True, "initial_codec_chunk_frames": 1}, True, True, True),
        (None, {"talker_first_audio": True, "initial_codec_chunk_frames": 4}, True, True, False),
        (None, {"talker_first_audio": True, "codec_chunk_ramp": [1]}, True, True, False),
        (None, {"talker_first_audio": True, "codec_chunk_ramp": [1, 2]}, False, True, False),
        (None, {"talker_first_audio": True, "codec_chunk_ramp": [1, 2]}, True, False, False),
    ],
)
def test_talker_first_audio_default_and_conditions(monkeypatch, env, extra, async_chunk, v2, expected):
    if env is None:
        monkeypatch.delenv("VLLM_OMNI_TALKER_FIRST_AUDIO", raising=False)
    else:
        monkeypatch.setenv("VLLM_OMNI_TALKER_FIRST_AUDIO", env)
    from vllm_omni.platforms import current_omni_platform

    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: True)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            async_chunk=async_chunk, use_v2_model_runner=v2, stage_connector_config={"extra": extra}
        ),
        device_config=SimpleNamespace(device="cuda"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1, distributed_executor_backend=None
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
    )
    assert talker_first_audio_enabled(config) is expected
    for section, field, value in [
        ("device_config", "device", "cpu"),
        ("parallel_config", "tensor_parallel_size", 2),
        ("parallel_config", "pipeline_parallel_size", 2),
        ("parallel_config", "distributed_executor_backend", "mp"),
        ("cache_config", "enable_prefix_caching", True),
    ]:
        with monkeypatch.context() as patch:
            patch.setattr(getattr(config, section), field, value)
            assert not talker_first_audio_enabled(config)
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: False)
    assert not talker_first_audio_enabled(config)


@pytest.mark.parametrize("option", [False, True])
def test_tts_requires_connector_opt_in(monkeypatch, option):
    from vllm_omni.model_executor.models.qwen3_tts import first_audio
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code_predictor_vllm import (
        Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM as Predictor,
    )

    monkeypatch.setattr(Predictor, "_stage_connector_extra_config", lambda _: {"talker_first_audio": option})
    monkeypatch.setattr(first_audio, "supports_talker_first_audio", lambda _: True)
    assert first_audio.talker_first_audio_enabled(None) is option
