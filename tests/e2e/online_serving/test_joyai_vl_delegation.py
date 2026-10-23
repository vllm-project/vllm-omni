# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real JoyAI delegation over the shared Realtime client.

The unit and session-harness tests prove the plugin side. This module drives the
whole chain on a served deployment: JoyAI chooses to delegate, the session
publishes a standard function call, the application answers it with a
``function_call_output``, and the next observation speaks that answer.
"""

import asyncio
import base64
import io
import json

import pytest
from PIL import Image

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni.clients.duplex import DuplexClient, EventCollector, SessionConfig
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.delegation import delegation_tool_definition

pytestmark = [pytest.mark.advanced_model, pytest.mark.omni]

#: The answer the application fabricates for the delegated question.
DELEGATED_ANSWER = "the code is ZB-4471"
DELEGATION_INSTRUCTION = (
    "You are watching a camera feed. Speak only when you have something to report. "
    "If a question needs information the image cannot show, delegate it to the background helper "
    "instead of guessing. When a delegated result arrives, always answer it aloud with that result."
)
DELEGATION_PROMPT = "Which internal product code applies to the object in view? Delegate if you cannot read it."
#: JoyAI answers the goal of the current chunk; asking for the result once the
#: helper has come back is what the application does after a delegation.
ANSWER_REQUEST = "Answer now with the result you received."


def frame_base64(color: str) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (224, 224), color).save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


def delegation_session_config() -> SessionConfig:
    """A JoyAI session whose only declared tool is the delegation function.

    ``tools`` is a top-level Realtime session field; the engine maps it onto the
    internal ``extra_body["realtime_tools"]`` the plugin validates.
    """
    return SessionConfig(
        auto_response=False,
        temperature=0.0,
        instructions=DELEGATION_INSTRUCTION,
        extra_body={"tts_language": "English"},
    )


async def declare_delegation_tool(client: DuplexClient) -> None:
    await client.send({"type": "session.update", "session": {"tools": [delegation_tool_definition()]}})


def function_calls(collector: EventCollector) -> list[dict[str, object]]:
    return [event for event in collector.events if event.get("type") == "response.function_call_arguments.done"]


def error_code(event: dict[str, object]) -> object:
    """The ``error.code`` of one error event; the code is nested, not top-level."""
    error = event.get("error")
    return error.get("code") if isinstance(error, dict) else None


async def wait_for_function_call(collector: EventCollector, timeout_s: float = 180) -> dict[str, object]:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        calls = function_calls(collector)
        if calls:
            return calls[0]
        await asyncio.sleep(0.05)
    raise AssertionError(f"JoyAI published no function call; recent events: {collector.events[-5:]}")


def main_response_done(collector: EventCollector, seen: set[str]) -> dict[str, object] | None:
    """The newest terminal belonging to the main observation response.

    A delegating observation produces two terminals: the synthesised
    function-call response and the spoken one. Only the latter owns the note's
    audio, so a playback acknowledgement must name it.
    """
    for event in reversed(collector.events):
        if event.get("type") != "response.done":
            continue
        response_id = str(event.get("response_id") or "")
        if response_id.startswith("resp-duplex-") and response_id not in seen:
            return event
    return None


async def observe(client: DuplexClient, collector: EventCollector, color: str, seen: set[str], timeout_s: float = 180):
    await client.append_audio(b"", video_frames=[frame_base64(color)])
    await client.commit(create_response=True)
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        terminal = main_response_done(collector, seen)
        if terminal is not None:
            seen.add(str(terminal["response_id"]))
            return terminal
        await asyncio.sleep(0.1)
    raise AssertionError("the observation produced no spoken response")


async def acknowledge_note(client: DuplexClient, collector: EventCollector, terminal) -> None:
    """Pay the playback debt of the note so the next observation is admitted.

    JoyAI is serial: the session only accepts the next camera commit once the
    audio it already sent has been acknowledged in full.
    """
    response_id = str(terminal["response_id"])
    pcm = collector.audio_bytes(response_id)
    played_ms = max(1, len(pcm) // 2 * 1000 // collector.output_sample_rate_hz)
    await client.ack_playback(played_ms, response_id=response_id)


async def run_delegation(url, model, output_dir):
    config = delegation_session_config()
    collector = EventCollector()
    seen: set[str] = set()
    async with DuplexClient(url, model=model, config=config, reconnect=None) as client:
        consumer = asyncio.create_task(collector.consume(client))
        await asyncio.sleep(0)
        try:
            await declare_delegation_tool(client)
            await client.send(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": DELEGATION_PROMPT}],
                    },
                }
            )

            # 1. The model delegates: a declared tool must never surface as an
            #    unsupported call, and the call must carry its own question.
            first = await observe(client, collector, "red", seen)
            call = await wait_for_function_call(collector)
            arguments = json.loads(str(call.get("arguments") or "{}"))
            assert arguments.get("question"), f"the call must carry a delegated question: {arguments}"
            assert "unsupported_tool_call" not in json.dumps(collector.errors()), collector.errors()

            # 2. The application answers the call while the note is still playing.
            before_result = len(collector.events)
            await client.send(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": DELEGATED_ANSWER,
                    },
                }
            )
            await asyncio.sleep(2)
            refused = [
                event
                for event in collector.events[before_result:]
                if event.get("type") == "error" and error_code(event) == "invalid_function_call_output"
            ]
            assert not refused, f"the first result for a pending call must be accepted: {refused}"

            # 3. A repeated result for the accepted call is refused by the ledger.
            #    It is checked here, while the call is still fresh, because the
            #    refusal is a property of admission rather than of the answer.
            await client.send(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "id": "duplicate-result",
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": "a second answer",
                    },
                }
            )
            deadline = asyncio.get_running_loop().time() + 60
            duplicates: list[dict[str, object]] = []
            while asyncio.get_running_loop().time() < deadline:
                duplicates = [
                    error for error in collector.errors() if error_code(error) == "invalid_function_call_output"
                ]
                if duplicates:
                    break
                await asyncio.sleep(0.2)
            assert duplicates, f"a duplicate function_call_output must be refused: {collector.errors()}"

            # 4. JoyAI is serial: the next observation is only admitted once the
            #    note's playback is acknowledged. The application also tells the
            #    model to answer, which is how a delegated goal gets spoken.
            await acknowledge_note(client, collector, first)
            await client.send(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": ANSWER_REQUEST}],
                    },
                }
            )

            # 5. A later observation speaks the delegated answer, with audio.
            needle = DELEGATED_ANSWER.split()[-1].lower()
            transcript = ""
            answer_terminal = None
            for color in ("blue", "red", "blue", "red"):
                terminal = await observe(client, collector, color, seen)
                transcript = collector.response_text(str(terminal["response_id"]))
                if needle in transcript.lower():
                    answer_terminal = terminal
                    break
                await acknowledge_note(client, collector, terminal)
            assert needle in transcript.lower(), f"the delegated answer never reached speech: {transcript!r}"

            # 6. The answer is actually spoken, not just transcribed: it carries
            #    PCM16 audio, and that playback is acknowledged so the session is
            #    free again. The floor is 50 ms at the session's own output rate,
            #    so the check keeps its meaning if that rate changes.
            assert answer_terminal is not None
            answer_pcm = collector.audio_bytes(str(answer_terminal["response_id"]))
            min_bytes = collector.output_sample_rate_hz * 2 // 20
            assert len(answer_pcm) >= min_bytes and len(answer_pcm) % 2 == 0, (
                f"the delegated answer must produce nonempty PCM16 speech, got {len(answer_pcm)} bytes "
                f"at {collector.output_sample_rate_hz} Hz"
            )
            await acknowledge_note(client, collector, answer_terminal)

            # 7. Serial admission depends on that acknowledgement: one further
            #    camera observation must complete, not hang behind a stuck request.
            errors_before = len(collector.errors())
            final = await observe(client, collector, "red", seen)
            assert final["response"]["status"] == "completed", final["response"]
            assert len(collector.errors()) == errors_before, collector.errors()[errors_before:]

            await client.close()
            await consumer
        finally:
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
            # Written on every path, so a failure in steps 6-7 still leaves the
            # events that explain it instead of stopping at step 5.
            (output_dir / "delegation-events.json").write_text(json.dumps(collector.events, indent=2), encoding="utf-8")


@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=1)
@pytest.mark.parametrize(
    "omni_server",
    [
        OmniServerParams(
            model="jdopensource/JoyAI-VL-Interaction-Preview",
            stage_config_path=modify_stage_config(
                get_deploy_config_path("joyai_vl_interaction.yaml"),
                updates={"session_mode": "duplex"},
            ),
        )
    ],
    indirect=True,
)
def test_delegated_question_round_trips_into_speech(omni_server, tmp_path):
    asyncio.run(
        run_delegation(
            f"ws://{omni_server.host}:{omni_server.port}/v1/realtime?duplex=1",
            omni_server.model,
            tmp_path,
        )
    )
