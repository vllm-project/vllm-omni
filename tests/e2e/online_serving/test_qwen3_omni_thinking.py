# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
E2E Online tests for Qwen3-Omni model thinking variant
"""

import asyncio
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import pytest
from openai import AsyncOpenAI
from openai.resources.realtime.realtime import AsyncRealtimeConnection
from openai.types.realtime import (
    ConversationItemCreateEvent,
    RealtimeAudioConfig,
    RealtimeAudioConfigInput,
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemFunctionCallOutput,
    RealtimeConversationItemUserMessage,
    RealtimeError,
    RealtimeErrorEvent,
    RealtimeFunctionTool,
    RealtimeSessionCreateRequest,
    ResponseCreateEvent,
    ResponseDoneEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    SessionUpdateEvent,
)
from openai.types.realtime.realtime_conversation_item_user_message import Content

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio, generate_synthetic_image, generate_synthetic_video
from tests.helpers.realtime import (
    WEATHER_TOOL_ANSWER,
    WEATHER_TOOL_OUTPUT,
    WEATHER_TOOL_PROMPT,
    WEATHER_TOOL_SPEC,
)
from tests.helpers.runtime import OmniServerParams, dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
STAGE_CONFIG_PATH = get_deploy_config_path("qwen3_omni_moe.yaml")
# The checkpoint's template concatenates a content-parts list in its tools
# branch; this copy renders both content shapes.
THINKING_CHAT_TEMPLATE = (
    Path(__file__).resolve().parents[3]
    / "vllm_omni"
    / "transformers_utils"
    / "chat_templates"
    / "qwen3_omni_thinking.jinja"
)

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
LARGE_IMAGE_WIDTH = 1920
LARGE_IMAGE_HEIGHT = 1080
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
NUM_VIDEO_FRAMES = 300
LONG_VIDEO_WIDTH = 224
LONG_VIDEO_HEIGHT = 224
LONG_VIDEO_NUM_FRAMES = 3600
LONG_AUDIO_DURATION = 120

test_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=STAGE_CONFIG_PATH,
            server_args=[
                "--chat-template",
                str(THINKING_CHAT_TEMPLATE),
                "--reasoning-parser",
                "qwen3",
                "--tool-call-parser",
                "hermes",
                "--enable-auto-tool-choice",
            ],
        )
    )
]


def get_system_prompt():
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, "
                    "Alibaba Group, capable of perceiving text, image, audio or video inputs, "
                    "as well as generating text."
                ),
            }
        ],
    }


def get_text_only_prompt():
    return "What is the capital of China?"


def get_prompt():
    return "Analyse the audio, image or video provided"


@pytest.mark.omni
@pytest.mark.slow
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_analyse_text_input(omni_server, openai_client):
    """
    Input Modal: text
    """
    messages = dummy_messages_from_mix_data(system_prompt=get_system_prompt(), content_text=get_text_only_prompt())

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["text"],
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.omni
@pytest.mark.slow
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_analyse_prompt_without_attached_media(omni_server, openai_client):
    """
    Input Modal: text asking to describe media provided but it is absent
    """
    messages = dummy_messages_from_mix_data(system_prompt=get_system_prompt(), content_text=get_prompt())

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["text"],
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.omni
@pytest.mark.slow
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_analyse_audio_input(omni_server, openai_client):
    """
    Input Modal: audio
    """
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(5, 1)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(), content_text=get_prompt(), audio_data_url=audio_data_url
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["text"],
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.omni
@pytest.mark.slow
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_analyse_long_audio_input(omni_server, openai_client):
    """
    Input Modal: long audio
    """
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(LONG_AUDIO_DURATION, 1)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(), content_text=get_prompt(), audio_data_url=audio_data_url
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["text"],
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.omni
@pytest.mark.slow
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_analyse_image_input(omni_server, openai_client):
    """
    Input Modal: image
    """
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(IMAGE_WIDTH, IMAGE_HEIGHT)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(), content_text=get_prompt(), image_data_url=image_data_url
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["text"],
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.omni
@pytest.mark.slow
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_analyse_large_image_input(omni_server, openai_client):
    """
    Input Modal: large image
    """
    image_data_url = (
        f"data:image/jpeg;base64,{generate_synthetic_image(LARGE_IMAGE_WIDTH, LARGE_IMAGE_HEIGHT)['base64']}"
    )
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(), content_text=get_prompt(), image_data_url=image_data_url
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["text"],
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.omni
@pytest.mark.slow
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_analyse_video_input(omni_server, openai_client):
    """
    Input Modal: video
    """
    video_data_url = (
        f"data:video/mp4;base64,{generate_synthetic_video(VIDEO_WIDTH, VIDEO_HEIGHT, NUM_VIDEO_FRAMES)['base64']}"
    )
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(), content_text=get_prompt(), video_data_url=video_data_url
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["text"],
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.omni
@pytest.mark.slow
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_analyse_long_video_input(omni_server, openai_client):
    """
    Input Modal: long video
    """
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(LONG_VIDEO_WIDTH, LONG_VIDEO_HEIGHT, LONG_VIDEO_NUM_FRAMES)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(), content_text=get_prompt(), video_data_url=video_data_url
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["text"],
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.omni
@pytest.mark.slow
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_analyse_video_with_audio_input(omni_server, openai_client):
    """
    Input Modal: video with audio
    """
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(VIDEO_WIDTH, VIDEO_HEIGHT, NUM_VIDEO_FRAMES, embed_audio=True)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(), content_text=get_prompt(), video_data_url=video_data_url
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["text"],
    }

    openai_client.send_omni_request(request_config)


REASONING_PROMPT = "What is 17 * 23? Think it through, then reply with only the number."
REASONING_ANSWER = "391"
# One unambiguous tool and an explicit prompt, so tool_choice can stay 'auto'.
RECALL_PROMPT = "Which city did I ask about? Reply with only the city name."
RECALL_ANSWER = "Paris"
# Qwen3 wraps reasoning in  thinking...</think> and calls in <tool_call>{...}</tool_call>.
# If the parsers are missing or broken, that raw markup reaches the text deltas.
THINK_END_MARKER = "</think>"
TOOL_CALL_MARKER = "<tool_call>"
TOOL_SPEC = RealtimeFunctionTool(**WEATHER_TOOL_SPEC)


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: str


@dataclass
class Turn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    output_tokens: int | None = None
    status: str | None = None
    errors: list[str] = field(default_factory=list)


@dataclass
class ParserCheckResult:
    reasoning: Turn
    tool: Turn
    tool_result: Turn
    recall: Turn
    failures: list[str]

    def report(self) -> str:
        lines = []
        for name, turn in (
            ("reasoning parser", self.reasoning),
            ("tool-call parser", self.tool),
            ("tool result", self.tool_result),
            ("conversation recall", self.recall),
        ):
            verdict = "PASS" if not _turn_failures(self, name) else "FAIL"
            lines.append(f"=== {name}: {verdict} ===")
            lines.append(f"  status={turn.status} output_tokens={turn.output_tokens}")
            lines.append(f"  text={turn.text!r}")
            if turn.tool_calls:
                lines.append(f"  tool_calls={json.dumps([asdict(call) for call in turn.tool_calls])}")
        for failure in self.failures:
            lines.append(f"  ! {failure}")
        return "\n".join(lines)


def _turn_failures(result: ParserCheckResult, name: str) -> list[str]:
    return [failure for failure in result.failures if failure.startswith(f"{name}:")]


def _error_text(error: RealtimeError) -> str:
    return error.model_dump_json(exclude_none=True)


async def collect_turn(conn: AsyncRealtimeConnection, timeout: float) -> Turn:
    """Read events until response.done and summarize what the client saw."""
    turn = Turn()
    calls: dict[int, ToolCall] = {}
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            turn.errors.append(f"timed out after {timeout:.0f}s waiting for response.done")
            break
        event = await asyncio.wait_for(conn.recv(), timeout=remaining)
        if isinstance(event, ResponseTextDeltaEvent):
            turn.text += event.delta
        elif isinstance(event, ResponseOutputItemDoneEvent):
            item = event.item
            if isinstance(item, RealtimeConversationItemFunctionCall):
                calls[event.output_index] = ToolCall(
                    call_id=item.call_id or "",
                    name=item.name,
                    arguments=item.arguments,
                )
        elif isinstance(event, RealtimeErrorEvent):
            turn.errors.append(_error_text(event.error))
        elif isinstance(event, ResponseDoneEvent):
            turn.status = event.response.status
            turn.output_tokens = event.response.usage.output_tokens if event.response.usage else None
            break
    turn.tool_calls = [calls[index] for index in sorted(calls)]
    return turn


async def start_turn(
    conn: AsyncRealtimeConnection,
    *,
    prompt: str,
    instructions: str = "You are a helpful assistant. Think before answering.",
    tools: list[RealtimeFunctionTool] | None = None,
    tool_choice: Literal["auto", "required", "none"] = "auto",
) -> None:
    """Configure a text-only turn, append the user message, and create the response."""
    # Leaving turn_detection unset disables server VAD, so response.create drives the turn.
    session = RealtimeSessionCreateRequest(
        type="realtime",
        output_modalities=["text"],
        instructions=instructions,
        audio=RealtimeAudioConfig(input=RealtimeAudioConfigInput()),
    )
    if tools:
        session = session.model_copy(update={"tools": tools, "tool_choice": tool_choice})
    await conn.send(SessionUpdateEvent(type="session.update", session=session))
    await conn.send(
        ConversationItemCreateEvent(
            type="conversation.item.create",
            item=RealtimeConversationItemUserMessage(
                type="message",
                role="user",
                content=[Content(type="input_text", text=prompt)],
            ),
        )
    )
    await conn.send(ResponseCreateEvent(type="response.create"))


async def send_tool_output(conn: AsyncRealtimeConnection, call: ToolCall, output: str) -> None:
    """Append a tool result to the conversation and ask for the next response."""
    await conn.send(
        ConversationItemCreateEvent(
            type="conversation.item.create",
            item=RealtimeConversationItemFunctionCallOutput(
                type="function_call_output",
                call_id=call.call_id,
                output=output,
            ),
        )
    )
    await conn.send(ResponseCreateEvent(type="response.create"))


def _reasoning_failures(turn: Turn) -> list[str]:
    failures = [f"reasoning parser: {error}" for error in turn.errors]
    if not turn.text.strip():
        failures.append("reasoning parser: no text reached the client")
    if THINK_END_MARKER in turn.text:
        failures.append("reasoning parser: think markup leaked into output text")
    if REASONING_ANSWER not in turn.text:
        failures.append(f"reasoning parser: expected the answer {REASONING_ANSWER}, got {turn.text!r}")
    if not turn.output_tokens:
        failures.append("reasoning parser: response.done reported no output_tokens")
    return failures


def _tool_call_failures(turn: Turn) -> list[str]:
    failures = [f"tool-call parser: {error}" for error in turn.errors]
    if TOOL_CALL_MARKER in turn.text:
        failures.append("tool-call parser: tool markup leaked into output text")
    if not turn.tool_calls:
        failures.append("tool-call parser: the model did not choose the get_weather tool")
        return failures
    if len(turn.tool_calls) != 1:
        failures.append(f"tool-call parser: expected exactly one function_call item, got {len(turn.tool_calls)}")
        return failures
    call = turn.tool_calls[0]
    if call.name != TOOL_SPEC.name:
        failures.append(f"tool-call parser: expected {TOOL_SPEC.name}, got {call.name!r}")
    try:
        arguments = json.loads(call.arguments or "{}")
    except json.JSONDecodeError as exc:
        failures.append(f"tool-call parser: arguments are not valid JSON: {exc}: {call.arguments!r}")
    else:
        if "paris" not in json.dumps(arguments).lower():
            failures.append(f"tool-call parser: expected Paris in the arguments, got {arguments!r}")
    if not turn.output_tokens:
        failures.append("tool-call parser: response.done reported no output_tokens")
    return failures


def _tool_result_failures(tool: Turn, turn: Turn) -> list[str]:
    if not tool.tool_calls:
        return ["tool result: skipped because the model did not call the tool"]
    failures = [f"tool result: {error}" for error in turn.errors]
    if not turn.text.strip():
        failures.append("tool result: no text reached the client")
    if THINK_END_MARKER in turn.text or TOOL_CALL_MARKER in turn.text:
        failures.append("tool result: markup leaked into output text")
    if turn.text.strip() and WEATHER_TOOL_ANSWER not in turn.text:
        failures.append(f"tool result: expected the tool output {WEATHER_TOOL_ANSWER} in the answer, got {turn.text!r}")
    return failures


def _recall_failures(turn: Turn) -> list[str]:
    failures = [f"conversation recall: {error}" for error in turn.errors]
    if not turn.text.strip():
        failures.append("conversation recall: no text reached the client")
    if THINK_END_MARKER in turn.text or TOOL_CALL_MARKER in turn.text:
        failures.append("conversation recall: markup leaked into output text")
    if turn.text.strip() and RECALL_ANSWER.lower() not in turn.text.lower():
        failures.append(f"conversation recall: expected {RECALL_ANSWER}, got {turn.text!r}")
    return failures


async def run_parser_checks(
    *,
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    timeout: float = 600.0,
) -> ParserCheckResult:
    """Run a four-turn text-only conversation against a live server."""
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    try:
        # Force the turn-based handler even when a duplex handler is registered.
        async with client.realtime.connect(model=model, extra_query={"duplex": "0"}) as conn:
            await start_turn(conn, prompt=REASONING_PROMPT)
            reasoning = await collect_turn(conn, timeout)

            await start_turn(conn, prompt=WEATHER_TOOL_PROMPT, tools=[TOOL_SPEC], tool_choice="auto")
            tool = await collect_turn(conn, timeout)

            tool_result = Turn()
            if tool.tool_calls:
                await send_tool_output(conn, tool.tool_calls[0], WEATHER_TOOL_OUTPUT)
                tool_result = await collect_turn(conn, timeout)

            await start_turn(conn, prompt=RECALL_PROMPT)
            recall = await collect_turn(conn, timeout)
    finally:
        await client.close()

    return ParserCheckResult(
        reasoning=reasoning,
        tool=tool,
        tool_result=tool_result,
        recall=recall,
        failures=(
            _reasoning_failures(reasoning)
            + _tool_call_failures(tool)
            + _tool_result_failures(tool, tool_result)
            + _recall_failures(recall)
        ),
    )


@pytest.mark.advanced_model
@pytest.mark.omni
@pytest.mark.slow
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@hardware_test(res={"cuda": "H100"}, num_cards=2)
async def test_realtime_text_output_strips_reasoning_and_parses_tool_calls(omni_server) -> None:
    """Text-only /v1/realtime turns must hide think markup and surface tool calls."""
    result = await run_parser_checks(
        base_url=f"http://{omni_server.host}:{omni_server.port}/v1",
        model=omni_server.model,
    )
    assert not result.failures, result.report()
