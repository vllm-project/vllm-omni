# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Gander Unit8 speech on the shared full-duplex endpoint.

Set GANDER_MODEL to a directory composed with minicpmo_4_5.gander.
"""

import os
import wave
from pathlib import Path

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import (
    OmniServerParams,
    send_duplex_audio_request,
    send_duplex_soft_interrupt_request,
    send_duplex_tool_context_request,
)
from tests.helpers.stage_config import get_deploy_config_path

MODEL = os.environ.get("GANDER_MODEL", "")
pytestmark = [pytest.mark.omni, pytest.mark.skipif(not MODEL, reason="Set GANDER_MODEL to the composed release")]
SERVER_PARAMS = [
    OmniServerParams(
        model=MODEL,
        stage_config_path=get_deploy_config_path("gander.yaml"),
        use_stage_cli=False,
        server_args=["--trust-remote-code"],
    )
]


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_gander_streaming_speech(omni_server, tmp_path):
    # A recorded speech question followed by microphone silence gives the
    # native model time to finish without a text prompt or transcript hint.
    source = Path(__file__).resolve().parents[2] / "assets/minicpmo_4_5/response_required_16k.wav"
    input_wav = tmp_path / "question_and_silence.wav"
    with wave.open(str(source), "rb") as audio:
        params = audio.getparams()
        samples = audio.readframes(audio.getnframes())
    with wave.open(str(input_wav), "wb") as audio:
        audio.setparams(params)
        audio.writeframes(samples + bytes(params.framerate * params.sampwidth * params.nchannels * 20))
    send_duplex_audio_request(
        url=f"ws://{omni_server.host}:{omni_server.port}/v1/realtime?duplex=1",
        model=omni_server.model,
        input_wav=input_wav,
        ref_audio=Path(MODEL) / "assets/ref_audio.wav",
        output_dir=tmp_path / "gander",
    )


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_gander_soft_interrupt_streaming_contract(omni_server, tmp_path):
    # Gander must choose its native interrupt action, cancel the old reply,
    # and answer the follow-up. A complete short answer may use one audio delta.
    # The MiniCPM driver's default non-cancelling contract remains separate.
    source = Path(__file__).resolve().parents[2] / "assets/minicpmo_4_5/soft_interrupt_16k.wav"
    send_duplex_soft_interrupt_request(
        url=f"ws://{omni_server.host}:{omni_server.port}/v1/realtime?duplex=1",
        model=omni_server.model,
        input_wav=source,
        ref_audio=Path(MODEL) / "assets/ref_audio.wav",
        output_dir=tmp_path / "soft_interrupt",
        input_sha256="cadae6d0ddc510310f16f8775d6379f30a5195369ccb54ff10a8e3f9a1f4a2ea",
        require_model_interrupt=True,
    )


def _tool_context_request(omni_server, tmp_path, **overrides):
    from vllm_omni.clients.duplex import reference_audio_data_url
    from vllm_omni.clients.minicpmo_4_5 import create_duplex_session_config

    tools = [
        {
            "name": "task_start",
            "description": "用当前用户原话新建一个后台任务。name是简短语义名称，仅用于任务列表展示和引用；完整任务内容由Runtime绑定当前用户turn。",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
        }
    ]
    config = create_duplex_session_config(
        ref_audio=reference_audio_data_url(Path(MODEL) / "assets/ref_audio.wav"),
        temperature=0,
        extra_body={"realtime_tools": tools, "gander_task_slate": "当前没有后台任务。"},
    )
    result = send_duplex_tool_context_request(
        url=f"ws://{omni_server.host}:{omni_server.port}/v1/realtime?duplex=1",
        model=omni_server.model,
        session_config=config,
        input_wav=Path(__file__).resolve().parents[2] / "assets/minicpmo_4_5/tool_request_16k.wav",
        output_dir=tmp_path / "tools",
        expected_tool="task_start",
        tool_output={"status": "completed", "answer": "蓝鲸四七二"},
        context_before=[
            {"kind": "runtime_event", "output": {"status": "running", "progress": "本地查询已开始"}},
            {"kind": "task_slate", "version": 1, "slate": "取货暗号查询：进行中。"},
        ],
        context_after=[{"kind": "task_slate", "version": 2, "slate": "取货暗号查询：已完成。当前没有进行中的任务。"}],
        expected_text="蓝鲸",
        require_cancelled_response=True,
        **overrides,
    )
    return result


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_gander_tool_result_and_task_slate(omni_server, tmp_path):
    result = _tool_context_request(omni_server, tmp_path)
    assert result["call"]["name"] == "task_start"


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_gander_task_slate_reference(omni_server, tmp_path):
    # Separate semantic acceptance from successful KV delivery. This strict
    # check verifies the latest protected slate is used without another task.
    _tool_context_request(
        omni_server,
        tmp_path,
        followup_wav=Path(__file__).resolve().parents[2] / "assets/minicpmo_4_5/task_slate_query_16k.wav",
        expected_followup_text="没有",
    )


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_gander_history_replacement_and_rollover(omni_server, tmp_path):
    from tests.helpers.runtime import send_duplex_context_edit_request
    from vllm_omni.clients.duplex import reference_audio_data_url
    from vllm_omni.clients.minicpmo_4_5 import create_duplex_session_config

    config = create_duplex_session_config(
        ref_audio=reference_audio_data_url(Path(MODEL) / "assets/ref_audio.wav"),
        temperature=0,
        extra_body={"gander_history": {"max_units": 8, "retain_units": 4}},
    )
    send_duplex_context_edit_request(
        url=f"ws://{omni_server.host}:{omni_server.port}/v1/realtime?duplex=1",
        model=omni_server.model,
        session_config=config,
        output_dir=tmp_path / "history",
        input_wav=Path(__file__).resolve().parents[2] / "assets/minicpmo_4_5/response_required_16k.wav",
    )


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_gander_historical_event_insertion(omni_server, tmp_path):
    _tool_context_request(
        omni_server, tmp_path, history_event={"status": "accepted", "source": "local deterministic handler"}
    )


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_gander_default_history_window(omni_server, tmp_path):
    from tests.helpers.runtime import send_duplex_context_edit_request
    from vllm_omni.clients.duplex import reference_audio_data_url
    from vllm_omni.clients.minicpmo_4_5 import create_duplex_session_config

    config = create_duplex_session_config(
        ref_audio=reference_audio_data_url(Path(MODEL) / "assets/ref_audio.wav"), temperature=0
    )
    send_duplex_context_edit_request(
        url=f"ws://{omni_server.host}:{omni_server.port}/v1/realtime?duplex=1",
        model=omni_server.model,
        session_config=config,
        output_dir=tmp_path / "default_history",
        expected_max_units=128,
        rollover_input_seconds=140,
        input_wav=Path(__file__).resolve().parents[2] / "assets/minicpmo_4_5/response_required_16k.wav",
    )


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_gander_four_sessions_two_turns_and_admission(omni_server, tmp_path):
    from tests.helpers.runtime import send_duplex_concurrent_audio_request

    send_duplex_concurrent_audio_request(
        server=omni_server,
        input_wav=Path(__file__).resolve().parents[2] / "assets/minicpmo_4_5/response_required_16k.wav",
        ref_audio=Path(MODEL) / "assets/ref_audio.wav",
        output_dir=tmp_path / "concurrent",
        sessions=4,
    )
