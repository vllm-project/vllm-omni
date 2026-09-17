# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU contract tests for the Qwen3-Omni duplex plugin."""

from __future__ import annotations

import base64

import numpy as np
import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.engine.duplex.plugin import load_duplex_plugin, validate_duplex_plugin_sampling
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import (
    QWEN3_OMNI_TALKER_EOS_TOKEN_ID,
    Qwen3OmniDuplexPlugin,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PLUGIN_PATH = "vllm_omni.model_executor.models.qwen3_omni.duplex.plugin.Qwen3OmniDuplexPlugin"


def _encode_audio(audio: object, sample_rate: int, fmt: str, speed: float | None) -> str | None:
    del audio, sample_rate, fmt, speed
    return "ZmFrZQ=="


def _pcm_payload(samples: int = 800) -> dict[str, object]:
    audio = np.zeros(samples, dtype="<f4")
    return {
        "type": "audio",
        "format": "pcm_f32le",
        "sample_rate_hz": 16000,
        "audio": base64.b64encode(audio.tobytes()).decode("ascii"),
        "final": True,
        "turn_commit": True,
        "is_speech": True,
    }


def test_load_qwen3_omni_duplex_plugin_and_capabilities() -> None:
    plugin = load_duplex_plugin(PLUGIN_PATH, _encode_audio)
    assert isinstance(plugin, Qwen3OmniDuplexPlugin)
    assert plugin.plugin_id == "qwen3_omni"
    defaults = (
        SamplingParams(max_tokens=16),
        SamplingParams(max_tokens=4096),
        SamplingParams(max_tokens=16),
    )
    validate_duplex_plugin_sampling(plugin, sampling_defaults=defaults)
    configured = plugin.configure_sampling_params(runtime_config={}, defaults=defaults)
    assert len(configured) == 3
    assert QWEN3_OMNI_TALKER_EOS_TOKEN_ID in (configured[1].stop_token_ids or [])
    caps = plugin.capabilities(max_sessions=1)
    assert caps.supports_turn_commit_only is True
    assert caps.supports_core_resumable_request is False
    assert caps.supports_chat_completions is True
    assert caps.supports_barge_in is False


def test_plan_append_commit_builds_thinker_prompt() -> None:
    plugin = Qwen3OmniDuplexPlugin(_encode_audio)
    fence = DuplexFence("sess-1", epoch=2, turn_id=5)
    plan = plugin.plan_append(
        request_id="req",
        fence=fence,
        session_config={},
        runtime_config={"qwen3_system_prompt": "sys"},
        seq=1,
        turn_seq=1,
        payload=_pcm_payload(),
        final=True,
        sampling_params=SamplingParams(max_tokens=8),
    )
    info = plan.prompt["additional_information"]
    assert info["qwen3_duplex"] is True
    assert info["session_id"] == "sess-1"
    assert "audio" in plan.prompt["multi_modal_data"]
    assert "<|audio_pad|>" in plan.prompt["prompt"]
    assert "prompt_token_ids" not in plan.prompt


def test_plan_append_rejects_non_final_chunks() -> None:
    plugin = Qwen3OmniDuplexPlugin(_encode_audio)
    fence = DuplexFence("sess-1")
    streaming = dict(_pcm_payload())
    streaming.pop("turn_commit", None)
    streaming["final"] = False
    with pytest.raises(ValueError, match="committed final turn"):
        plugin.plan_append(
            request_id="req",
            fence=fence,
            session_config={},
            runtime_config={},
            seq=1,
            turn_seq=1,
            payload=streaming,
            final=False,
            sampling_params=SamplingParams(max_tokens=8),
        )


def test_plan_append_requires_audio() -> None:
    plugin = Qwen3OmniDuplexPlugin(_encode_audio)
    fence = DuplexFence("sess-1")
    missing = dict(_pcm_payload())
    missing["audio"] = ""
    with pytest.raises(ValueError, match="requires audio"):
        plugin.plan_append(
            request_id="req",
            fence=fence,
            session_config={},
            runtime_config={},
            seq=1,
            turn_seq=1,
            payload=missing,
            final=True,
            sampling_params=SamplingParams(max_tokens=8),
        )


def test_plan_append_rejects_bad_base64() -> None:
    plugin = Qwen3OmniDuplexPlugin(_encode_audio)
    fence = DuplexFence("sess-1")
    payload = dict(_pcm_payload())
    payload["audio"] = "%%%not-base64%%%"
    with pytest.raises(ValueError, match="valid base64"):
        plugin.plan_append(
            request_id="req",
            fence=fence,
            session_config={},
            runtime_config={},
            seq=1,
            turn_seq=1,
            payload=payload,
            final=True,
            sampling_params=SamplingParams(max_tokens=8),
        )


def test_observe_stage_output_targets_thinker_only() -> None:
    plugin = Qwen3OmniDuplexPlugin(_encode_audio)
    assert plugin.observe_stage_output(stage_id=0, output=object(), context=object()) is True
    assert plugin.observe_stage_output(stage_id=1, output=object(), context=object()) is False
    assert plugin.observe_stage_output(stage_id=2, output=object(), context=object()) is False


def test_decide_output_never_short_circuits() -> None:
    plugin = Qwen3OmniDuplexPlugin(_encode_audio)
    assert (
        plugin.decide_output(
            stage_id=0,
            final_stage_id=2,
            segment_finished=True,
            segment_token_ids=(1,),
            segment_output_metadata={},
            output=object(),
        )
        is None
    )


@pytest.mark.asyncio
async def test_prepare_runtime_config_seeds_chat_text() -> None:
    plugin = Qwen3OmniDuplexPlugin(_encode_audio)
    config = DuplexSessionConfig(
        model="Qwen/Qwen3-Omni",
        instructions="be brief",
        initial_user_text="hello from chat",
        extra_body={},
    )
    runtime = await plugin.prepare_runtime_config(config, model_config=None)
    assert runtime["qwen3_system_prompt"] == "be brief"
    assert runtime["initial_user_text"] == "hello from chat"


def test_validate_client_extra_body_requires_object() -> None:
    plugin = Qwen3OmniDuplexPlugin(_encode_audio)
    plugin.validate_client_extra_body(None)
    plugin.validate_client_extra_body({})
    with pytest.raises(ValueError, match="extra_body must be an object"):
        plugin.validate_client_extra_body(["not", "a", "dict"])
