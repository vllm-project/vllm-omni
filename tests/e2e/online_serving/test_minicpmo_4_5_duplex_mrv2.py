# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-weight MRv2 duplex concurrency guard, additional to the V1 CI jobs."""

import asyncio
from pathlib import Path

import pytest

from tests.e2e.online_serving.helpers.minicpmo_4_5_duplex import (
    MODEL,
    multi_session_args,
    realtime_url,
    resolve_ref_audio,
    validated_input_wav,
)
from tests.e2e.online_serving.run_minicpmo_realtime_duplex_multi_session import run_multi_session
from tests.e2e.online_serving.test_minicpmo_4_5_duplex import _run_seeded_text_to_audio
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.omni, pytest.mark.advanced_model]

_SERVER = OmniServerParams(
    model=MODEL,
    stage_config_path=get_deploy_config_path("minicpmo_4_5_duplex_mrv2.yaml"),
    use_stage_cli=False,
    server_args=["--trust-remote-code"],
)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", [pytest.param(_SERVER, id="mrv2-real-weights")], indirect=True)
@pytest.mark.parametrize("sessions", [1, 2, 4])
def test_mrv2_duplex_overlapping_turns(omni_server, tmp_path: Path, sessions: int):
    args = multi_session_args(
        omni_server=omni_server,
        input_wav=validated_input_wav(),
        ref_audio=resolve_ref_audio(),
        output_dir=tmp_path / f"concurrency_{sessions}",
        response_required=True,
    )
    args.sessions = sessions
    args.turns = 2
    args.turn_duration_ms = [args.first_turn_ms] * args.turns
    args.synchronized_start = True
    args.disconnect_session_index = None
    args.takeover_session_index = None
    result = asyncio.run(run_multi_session(args))
    assert result["ok"] is True, result
    assert result["session_count"] == sessions
    assert result["identity_isolation_ok"] is True
    assert not result["failures"]
    for session in result["sessions"]:
        assert session["audio_delta_count"] > 0, session
        assert session["done_count"] == args.turns, session
        assert session["error_count"] == 0, session


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", [pytest.param(_SERVER, id="mrv2-real-weights")], indirect=True)
def test_mrv2_duplex_new_session_during_long_response(omni_server):
    # Unequal prompts and a delayed second session prevent identical clients
    # from advancing in lockstep and hiding shared condition-state bugs.
    ref_audio = resolve_ref_audio()

    async def speak(text, delay):
        await asyncio.sleep(delay)
        return await _run_seeded_text_to_audio(
            url=realtime_url(omni_server),
            model=omni_server.model,
            ref_audio=ref_audio,
            text=text,
            silence_seconds=30.0,
        )

    async def overlap():
        return await asyncio.gather(
            speak("What is the capital of China? Answer in about 40 words.", 0),
            speak("What is the capital of France? Answer in one short sentence.", 2),
        )

    first, second = asyncio.run(overlap())
    for result in (first, second):
        assert "response.done" in result["event_types"], result
        assert result["audio_bytes"] > 0, result
        assert str(result["transcript"]).strip(), result
    assert first["audio_bytes"] > 96_000, first
    assert str(first["transcript"]).strip() != str(second["transcript"]).strip()


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", [pytest.param(_SERVER, id="mrv2-real-weights")], indirect=True)
def test_mrv2_duplex_server_answers_image_chat_from_the_image(omni_server, openai_client):
    """The duplex Thinker also serves /v1/chat/completions: its image features must reach the prompt."""
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (224, 224), (255, 0, 0)).save(buffer, format="JPEG")
    image_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    request_config = {
        "model": omni_server.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "What color is this image? Answer with one word."},
                ],
            }
        ],
        "stream": True,
        "modalities": ["text"],
        "key_words": {"text": ["red"]},
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    openai_client.send_omni_request(request_config)
