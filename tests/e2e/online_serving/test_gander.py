# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Gander Unit8 speech on the shared full-duplex endpoint.

Set GANDER_MODEL to a directory composed with minicpmo_4_5.gander.
"""

import base64
import io
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
    # Natural speech and completion require real Thinker/Talker weights.
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
    options = {
        "context_before": [
            {"kind": "runtime_event", "output": {"status": "running", "progress": "本地查询已开始"}},
            {"kind": "task_slate", "version": 1, "slate": "取货暗号查询：进行中。"},
        ],
        "context_after": [
            {"kind": "task_slate", "version": 2, "slate": "取货暗号查询：已完成。当前没有进行中的任务。"}
        ],
        "require_cancelled_response": True,
        **overrides,
    }
    result = send_duplex_tool_context_request(
        url=f"ws://{omni_server.host}:{omni_server.port}/v1/realtime?duplex=1",
        model=omni_server.model,
        session_config=config,
        input_wav=Path(__file__).resolve().parents[2] / "assets/minicpmo_4_5/tool_request_16k.wav",
        output_dir=tmp_path / "tools",
        expected_tool="task_start",
        tool_output={"status": "completed", "answer": "蓝鲸四七二"},
        expected_text="蓝鲸",
        **options,
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
def test_gander_pending_tool_survives_native_speech_interrupt(omni_server, tmp_path):
    # The application deliberately withholds the result until the model has
    # interrupted a separate spoken answer and answered the arithmetic follow-up.
    # Only playback is interrupted; the original call must still accept its result.
    result = _tool_context_request(
        omni_server,
        tmp_path,
        context_before=(),
        context_after=(),
        require_cancelled_response=False,
        pending_interrupt_wav=Path(__file__).resolve().parents[2] / "assets/minicpmo_4_5/soft_interrupt_16k.wav",
        expected_interrupt_text_pattern=r"(?:一加一(?:等于|是)(?:二|2)|(?:答案|结果|結果)是(?:二|2))(?=[。.!！\s]|$)",
    )
    assert result["interrupted_response"]


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


@pytest.mark.core_model
@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_gander_websocket_protocol(omni_server):
    from tests.helpers.runtime import send_duplex_protocol_request

    send_duplex_protocol_request(server=omni_server, ref_audio=Path(MODEL) / "assets/ref_audio.wav")


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_gander_video_two_turns(omni_server, tmp_path):
    from tests.helpers.runtime import send_duplex_video_turns_request

    send_duplex_video_turns_request(
        server=omni_server,
        input_wav=Path(__file__).resolve().parents[2] / "assets/minicpmo_4_5/response_required_16k.wav",
        ref_audio=Path(MODEL) / "assets/ref_audio.wav",
        output_dir=tmp_path / "video",
    )


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_gander_resume_and_takeover(omni_server, tmp_path):
    from tests.helpers.runtime import send_duplex_concurrent_audio_request

    send_duplex_concurrent_audio_request(
        server=omni_server,
        input_wav=Path(__file__).resolve().parents[2] / "assets/minicpmo_4_5/response_required_16k.wav",
        ref_audio=Path(MODEL) / "assets/ref_audio.wav",
        output_dir=tmp_path / "resume",
        sessions=2,
        resume_and_takeover=True,
    )


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_gander_pending_tool_result_after_resume(omni_server, tmp_path):
    _tool_context_request(omni_server, tmp_path, resume_before_result=True)


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_gander_public_client_live_session(omni_server):
    from tests.helpers.runtime import send_duplex_client_session_request

    send_duplex_client_session_request(
        server=omni_server,
        input_wav=Path(__file__).resolve().parents[2] / "assets/minicpmo_4_5/response_required_16k.wav",
        ref_audio=Path(MODEL) / "assets/ref_audio.wav",
    )


def _jpeg_frame(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "Please say exactly: the quick brown fox jumps over the lazy dog.",
            r"the quick brown fox jumps over the lazy dog",
        ),
        ("请朗读：今天天气很好，我们一起去公园散步。", r"今天天气很好[，,\s]*我们一起去公园散步"),
    ],
)
def test_gander_seeded_text_to_audio(omni_server, text, expected):
    from tests.helpers.runtime import send_duplex_seeded_text_request

    send_duplex_seeded_text_request(
        server=omni_server,
        text=text,
        ref_audio=Path(MODEL) / "assets/ref_audio.wav",
        expected_text_pattern=expected,
    )


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_gander_seeded_text_without_reference_voice(omni_server):
    from tests.helpers.runtime import send_duplex_seeded_text_request

    send_duplex_seeded_text_request(
        server=omni_server,
        text="What is the capital of France? Answer in one short sentence.",
        ref_audio=None,
        modalities=("text",),
        expected_text_pattern=r"\bParis\b|巴黎",
    )


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
@pytest.mark.parametrize("color,wrong", [("red", "blue"), ("blue", "red")])
def test_gander_image_color_semantics(omni_server, tmp_path, color, wrong):
    from PIL import Image, ImageDraw

    from tests.helpers.runtime import send_duplex_multimodal_request

    image = Image.new("RGB", (448, 448), "white")
    ImageDraw.Draw(image).rectangle((64, 64, 384, 384), fill=color)
    send_duplex_multimodal_request(
        server=omni_server,
        input_wav=Path(__file__).resolve().parents[2] / "assets/gander/color_question_16k.wav",
        ref_audio=Path(MODEL) / "assets/ref_audio.wav",
        output_dir=tmp_path / color,
        video_frames=[_jpeg_frame(image)],
        repeat_last_frame=True,
        expected_text_pattern=rf"\b{color}\b",
        forbidden_text_pattern=rf"\b{wrong}\b",
    )


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
@pytest.mark.parametrize("number,spoken", [("472", "four seven two"), ("815", "eight one five")])
def test_gander_image_ocr_semantics(omni_server, tmp_path, number, spoken):
    from PIL import Image, ImageDraw, ImageFont

    from tests.helpers.runtime import send_duplex_multimodal_request

    image = Image.new("RGB", (448, 448), "white")
    ImageDraw.Draw(image).text((224, 224), number, fill="black", font=ImageFont.load_default(size=96), anchor="mm")
    spoken_pattern = spoken.replace(" ", r"[\s,-]+")
    send_duplex_multimodal_request(
        server=omni_server,
        input_wav=Path(__file__).resolve().parents[2] / "assets/gander/ocr_question_16k.wav",
        ref_audio=Path(MODEL) / "assets/ref_audio.wav",
        output_dir=tmp_path / number,
        video_frames=[_jpeg_frame(image)],
        repeat_last_frame=True,
        expected_text_pattern=rf"\b{number}\b|\b{spoken_pattern}\b",
    )


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
@pytest.mark.parametrize("direction", ["left", "right"])
def test_gander_video_motion_semantics(omni_server, tmp_path, direction):
    from PIL import Image, ImageDraw

    from tests.helpers.runtime import send_duplex_multimodal_request
    from vllm_omni.experimental.fullduplex.video_stacking import concat_frames_b64

    positions = list(range(70, 361, 10))
    if direction == "left":
        positions.reverse()
    frames = []
    for x in positions:
        image = Image.new("RGB", (448, 448), "white")
        ImageDraw.Draw(image).ellipse((x - 24, 200, x + 24, 248), fill="red")
        frames.append(_jpeg_frame(image))
    opposite = "right" if direction == "left" else "left"
    pattern = r"(?:^DIRECTION[.!。]?$|\bDIRECTIONwards?\b|(?:to|towards?|mov(?:e[sd]?|ing)|went|going)\s+(?:the\s+)?DIRECTION\b)"
    send_duplex_multimodal_request(
        server=omni_server,
        input_wav=Path(__file__).resolve().parents[2] / "assets/gander/motion_question_16k.wav",
        ref_audio=Path(MODEL) / "assets/ref_audio.wav",
        output_dir=tmp_path / direction,
        video_frames=frames[::3],
        stacked_frames=[concat_frames_b64(frames[i + 1 : i + 3]) for i in range(0, len(frames), 3)],
        # Ask after the entire clip. A full-duplex model may answer while the
        # question is still being spoken, before later frames have arrived.
        video_lead_in_seconds=len(frames[::3]),
        expected_text_pattern=pattern.replace("DIRECTION", direction),
        forbidden_text_pattern=pattern.replace("DIRECTION", opposite),
    )
